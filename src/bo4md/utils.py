import matplotlib.pyplot as plt


def plot_traj(results, title="BO Trajectory", dpi=300, save_path=None):
    """
    Plot best-so-far y vs iteration for a single BO run.

    Args:
        results: dict returned by run_bo
        title: optional plot title
        save_path: if not None, save figure to this path instead of showing
    """
    best_y_hist = results["best_y_hist"]
    iters = range(1, len(best_y_hist) + 1)

    plt.figure(figsize=(12, 8), dpi=dpi)
    plt.plot(iters, best_y_hist, marker="o", color="C0")
    plt.xlabel("Iteration", fontsize=16)
    plt.ylabel("Best-so-far Target", fontsize=16)
    plt.title(title, fontsize=16)
    plt.grid(True)
    plt.xticks(iters, fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def save_results(results, filename="out.txt"):
    """
    Save BO run results (final best, initial design, BO trajectory,
    and best-so-far history) to a text file.

    Args:
        results: dict returned by run_bo_simplex3_once
        filename: path to save the .txt file
    """
    N_total = results["train_X"].shape[0]
    N_iters = results["iters_run"]
    N_init = N_total - N_iters  # number of initial samples

    with open(filename, "w") as f:
        f.write("Bayesian Optimization Run Results\n")
        f.write("=" * 40 + "\n\n")

        # Final best
        best_x = results["best_x"].tolist() if results["best_x"] is not None else None
        best_y = results["best_y"]
        f.write(f"Final best y: {best_y:.6f}\n")
        if best_x is not None:
            f.write(f"At x = {best_x}\n")
        f.write(f"Iterations run: {results['iters_run']} (plus {N_init} initial samples)\n\n")

        # Initial design
        f.write("Initial design samples (X_init, Y_init):\n")
        for i in range(N_init):
            x_list = [f"{xi:.6f}" for xi in results["train_X"][i].tolist()]
            y_val = float(results["train_Y"][i].item())
            f.write(f"Init {i+1:02d}: X = [{', '.join(x_list)}], Y = {y_val:.6f}\n")
        f.write("\n")

        # BO candidate trajectory
        f.write("BO candidate trajectory (X_next, Y_next):\n")
        for i, (x, y) in enumerate(zip(results["X_traj"], results["y_traj"]), 1):
            x_list = [f"{xi:.6f}" for xi in x.tolist()]
            f.write(f"Iter {i:02d}: X = [{', '.join(x_list)}], Y = {y:.6f}\n")
        f.write("\n")

        # Best-so-far trajectory (last section)
        f.write("Best-so-far y history:\n")
        for i, y in enumerate(results["best_y_hist"], 1):
            f.write(f"Iter {i:02d}: {y:.6f}\n")

    print(f"Results written to {filename}")


def save_initial_samples(X_init, Y_init, filename="init_samples.txt"):
    """
    Save initial design samples (X_init, Y_init) to a text file.

    Args:
        X_init: Tensor [N_init, d]
        Y_init: Tensor [N_init, 1]
        filename: output file path
    """
    N_init, d = X_init.shape
    with open(filename, "w") as f:
        f.write("Initial Design Samples\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Number of initial samples: {N_init}\n\n")

        for i in range(N_init):
            x_list = [f"{xi:.6f}" for xi in X_init[i].tolist()]
            y_val = float(Y_init[i].item())
            f.write(f"Init {i+1:02d}: X = [{', '.join(x_list)}], Y = {y_val:.6f}\n")

    print(f"[Init] Saved {N_init} initial samples to {filename}")
