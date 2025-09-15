import torch


def smoke_test(x12, noise_level=0.0): # Synthetic objective to MINIMIZE, with a unique minimum at (0.2, 0.5, 0.3)
    def to_x3(x12):
        # x12: [..., 2]
        x3 = 1.0 - x12[..., 0] - x12[..., 1]
        return torch.clamp(x3, min=0.0)
    x1 = x12[..., 0]
    x2 = x12[..., 1]
    x3 = to_x3(x12)
    # Quadratic bowl on the simplex
    y = (x1 - 0.2) ** 2 + (x2 - 0.5) ** 2 + (x3 - 0.3) ** 2
    # Add observation noise
    noise = noise_level * torch.randn_like(y)
    return y + noise


def md(x):
    """
    Molecular Dynamics (MD) simulation wrapper for Bayesian Optimization.

    This function should be modified to call MD code.

    Parameters
    ----------
    x : torch.Tensor of shape [n, d]
        - Each row is one input point (design parameter) in d-dimensional space.
        - Entries are non-negative and sum to 1 (i.e., points on the simplex).
        - For example, if d=3, each row is [x1, x2, x3].

    Returns
    -------
    y : torch.Tensor of shape [n]
        - Objective values (to MINIMIZE) evaluated at each input x.
        - Each value is a scalar float (the "loss" or "cost" from the MD simulation).
        - Example: radius of gyration, free energy, or another quantity of interest.

    Notes
    -----
    - You can write this function in plain Python/NumPy and convert to torch at the end.
    - Example usage inside run_bo:
        Y = md(X)  # where X is a torch.Tensor of shape [n, d]
    """

    # ======== Template ========
    # Convert X to numpy if your MD code expects numpy arrays
    # x_np = x.detach().cpu().numpy()   # shape [n, d]

    # --- Your code here ---
    # Run MD simulation(s) for each input row in x_np
    # Compute the output objective values as a numpy array, shape [n]
    # For now, we'll just return dummy values.
    y_np = [0.0 for _ in range(x.shape[0])]

    # Convert back to torch tensor
    y = torch.tensor(y_np, dtype=x.dtype, device=x.device)
    return y
