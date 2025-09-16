import torch
from torch import nn
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.analytic import ExpectedImprovement, UpperConfidenceBound
from botorch.acquisition import LogExpectedImprovement
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.models.transforms import Normalize, Standardize
from bo4md.utils import save_initial_samples, create_outfolder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


def sample_simplex_dirichlet(n, d, device=DEVICE, dtype=DTYPE):
    g = torch.distributions.Gamma(concentration=torch.ones(d, dtype=dtype, device=device), rate=torch.ones(d, dtype=dtype, device=device))
    X = g.sample((n,))  # [n, d]
    return X / X.sum(dim=-1, keepdim=True)


def fit_model(X, Y):
    gp = SingleTaskGP(
        train_X=X,
        train_Y=Y,
        input_transform=Normalize(d=X.shape[-1]),
        outcome_transform=Standardize(m=Y.shape[-1]),
        ).to(device=DEVICE, dtype=DTYPE)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp

class SoftmaxAcq(nn.Module):
    def __init__(self, acq_under_x: nn.Module, temperature: float = 1.0):
        super().__init__()
        self.acq_under_x = acq_under_x
        self.T = float(temperature)
    def forward(self, Z):                 # Z: [q, d]
        X = torch.softmax(Z / self.T, dim=-1)  # on simplex
        return self.acq_under_x(X)


def gen_candidate_softmax(
    model,
    train_X,
    *,
    acq_type: str = "logei",     # "logei" | "ei" | "ucb" | "random"
    q: int = 1,
    d: int = None,
    temperature: float = 1.0,
    num_restarts: int = 32,
    raw_samples: int = 512,
    beta_ucb: float = 1.0,      # for UCB (minimization)
) -> torch.Tensor:
    if d is None:
        d = train_X.shape[-1]
    """
    Returns X_next with shape [q, d], each row on the simplex (>=0, sums to 1).
    """
    device, dtype = train_X.device, train_X.dtype

    # Build the acquisition on true-X space
    acq_type = acq_type.lower()
    if acq_type not in ("logei", "ei", "ucb", "random"):
        raise ValueError(f"Unknown acq_type='{acq_type}'. Choose from 'logei', 'ei', 'ucb', 'random'")
    
    if acq_type in ("logei", "ei", "ucb"):
        if q != 1:
            raise ValueError("Analytic EI/UCB supports q=1 only")
        if acq_type in ("logei", "ei"):
            with torch.no_grad():
                best_f = model.posterior(train_X).mean.min().item()
            if acq_type == "logei":
                acq_under_x = LogExpectedImprovement(model=model, best_f=best_f, maximize=False)
            else:
                acq_under_x = ExpectedImprovement(model=model, best_f=best_f, maximize=False)
        else:
            acq_under_x = UpperConfidenceBound(model=model, beta=beta_ucb, maximize=False)


        # Wrap with softmax reparam (optimize in logits Z)
        wrapped = SoftmaxAcq(acq_under_x=acq_under_x, temperature=temperature)

        # Unconstrained logits box
        logit_bounds = torch.tensor([[-4.0] * d, [4.0] * d], device=device, dtype=dtype)

        # Optimize acquisition in Z-space
        cands, _ = optimize_acqf(
            acq_function=wrapped,
            bounds=logit_bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            return_best_only=True,
        )
        # Map logits -> simplex X
        X_next = torch.softmax(cands / temperature, dim=-1)  # [q, d]

    else:  # "random"
        X_next = sample_simplex_dirichlet(1, d, dtype=dtype, device=device)  # [1, d]

    return X_next


def run_bo(
    black_box,                 # callable: X [n, d] -> y [n]
    fit_model,                 # callable: (train_X, train_Y) -> trained GP model
    gen_candidate_softmax,     # callable: returns X_next [1, d] on simplex
    *,
    d=3,
    n_init=10,
    n_iter=20,
    acq_type="ucb",
    temperature=1.0,
    patience=5,               # early stop if no improvement for 'patience' iterations
    device=DEVICE,
    dtype=DTYPE,
    seed=None,
    outfolder=None,
):
    if seed is not None:
        torch.manual_seed(seed)

    create_outfolder(outfolder)
        
    # Initial samples
    print("\n======== Initial Samples Collection Start ========")
    train_X = sample_simplex_dirichlet(n_init, d=d, device=device, dtype=dtype)  # [n_init, d]
    train_Y = black_box(train_X).unsqueeze(-1)  # [n_init, 1]
    save_initial_samples(train_X, train_Y, filename=outfolder+"/init_samples.txt")

    X_traj, y_traj = [], []
    best_y = float("inf")
    best_x = None
    best_y_hist = []
    no_improve = 0

    # Main BO loop
    print("\n======== BO Start ========")
    iters_run = 0
    for n in range(n_iter):
        model = fit_model(train_X, train_Y)

        X_next = gen_candidate_softmax(
            model,
            train_X,
            acq_type=acq_type,
            q=1,
            d=d,
            temperature=temperature,
        )  # [1, d]

        Y_next = black_box(X_next).unsqueeze(-1)  # [1, 1]

        train_X = torch.cat([train_X, X_next], dim=0)
        train_Y = torch.cat([train_Y, Y_next], dim=0)

        X_traj.append(X_next.detach().squeeze(0))
        y_val = float(Y_next.item())
        y_traj.append(y_val)

        if y_val < best_y:
            best_y = y_val
            best_x = X_next.detach().squeeze(0)
            no_improve = 0
        else:
            no_improve += 1

        best_y_hist.append(best_y)
        iters_run = n + 1

        x_str = ", ".join([f"{xi:.4f}" for xi in X_next.squeeze(0).tolist()])
        print(
            f"[Iter {n+1:02d}] y = {y_val:.6f} "
            f"(best so far = {best_y:.6f} at x = [{x_str}])"
        )

        if no_improve >= patience:
            print(f"[Early stop] No improvement in last {patience} iterations.")
            break

    results = {
        "train_X": train_X,           # [n_init + iters_run, d] (plus early stop)
        "train_Y": train_Y,           # [n_init + iters_run, 1]
        "X_traj": X_traj,             # list of tensors [d] (BO steps only)
        "y_traj": y_traj,             # list of floats (BO steps only)
        "best_y_hist": best_y_hist,   # length = iters_run (best-so-far after each iter)
        "best_x": best_x,             # tensor [d]
        "best_y": best_y,             # float
        "iters_run": iters_run,       # int
    }

    return results
