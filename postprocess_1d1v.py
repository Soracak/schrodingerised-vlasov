# postprocess_1d1v.py
import numpy as np
import matplotlib.pyplot as plt

def plot_f(F, x_grid, v_grid, *, name=None, show=True):
    fig = plt.figure()
    # imshow's extent denotes pixel boundaries, not grid-point centres.
    dx = float(x_grid[1] - x_grid[0])
    dv = float(v_grid[1] - v_grid[0])
    plt.imshow(
        F.T,
        origin="lower",
        aspect="auto",
        extent=[x_grid[0] - dx/2, x_grid[-1] + dx/2,
                v_grid[0] - dv/2, v_grid[-1] + dv/2],
    )
    plt.xlabel("x"); plt.ylabel("v")
    plt.title(name)
    plt.colorbar()
    plt.tight_layout()
    if show:
        plt.show()
    return fig

def initial_slices(F, x_grid, v_grid, show=True):
    rho_x = np.sum(F, axis=1)
    g_v  = np.sum(F, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(x_grid, rho_x, "ko-")
    ax1.set_xlabel("x"); ax1.set_ylabel(r"$\sum_{k} f(x, v_k)$")

    ax2.plot(v_grid, g_v, "ko-")
    ax2.set_xlabel("v"); ax2.set_ylabel(r"$\sum_{j} f(x_j, v)$")

    fig.tight_layout()
    if show:
        plt.show()
    return fig