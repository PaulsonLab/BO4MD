import argparse
import warnings
import torch


from bo4md.bayes_opt import run_bo, fit_model, gen_candidate_softmax
from bo4md.utils import plot_traj, save_results
from bo4md.simulator import smoke_test, md

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Run Bayesian Optimization")
    parser.add_argument("--smoke_test", type=str2bool, default=True,
                        help="Smoke test (True/False). NOTE: smoke_test supports d=3 only.")
    parser.add_argument("--acq", type=str, default="logei",
                        choices=["ucb", "ei", "logei", "random"],
                        help="Acquisition function")
    parser.add_argument("--d", type=int, default=3,
                        help="Input dimension (number of simplex components)")
    parser.add_argument("--n-init", type=int, default=10, help="Number of initial samples")
    parser.add_argument("--n-iter", type=int, default=20, help="Max number of BO iterations")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--plot", type=str2bool, default=True,
                        help="Plot best-so-far target vs number of BO iterations (True/False)")
    parser.add_argument("--report", type=str2bool, default=True,
                        help="Report BO trajectory (True/False)")
    parser.add_argument("--outfolder", type=str, default="./out", help="Output folder")
    args = parser.parse_args()

    print(f"[INFO] Device: {DEVICE.type.upper()} | dtype: {DTYPE} | acq: {args.acq} | d={args.d} | "
          f"n_init={args.n_init} | n_iter={args.n_iter} | patience={args.patience} | seed={args.seed}")

    # Choose simulator
    if args.smoke_test:
        if args.d != 3:
            print("[WARN] smoke_test supports only d=3. Overriding --d to 3.")
            args.d = 3
        black_box = smoke_test
    else:
        black_box = md

    # Run BO
    results = run_bo(
        black_box=black_box,
        fit_model=fit_model,
        gen_candidate_softmax=gen_candidate_softmax,
        d=args.d,
        n_init=args.n_init,
        n_iter=args.n_iter,
        acq_type=args.acq,
        patience=args.patience,
        seed=args.seed,
        device=DEVICE,
        dtype=DTYPE,
        outfolder=args.outfolder,
    )

    # Final report
    print("\n======== Final Report ========")
    print(f"Best y: {results['best_y']:.6f}")
    print(f"Best x: {results['best_x'].tolist()}")
    print(f"Iterations run: {results['iters_run']}")

    # Optional outputs
    if args.plot:
        plot_traj(
            results,
            title=f"BO Trajectory ({args.acq.upper()}, Single Run)",
            save_path=args.outfolder + f"/bo_traj_{args.acq}_n_init_{args.n_init}_n_iter_{args.n_iter}_seed_{args.seed}.png",
        )

    if args.report:
        save_results(
            results,
            filename= args.outfolder + f"/bo_report__n_init_{args.n_init}_n_iter_{args.n_iter}_seed_{args.seed}.txt",
        )


if __name__ == "__main__":
    main()
    