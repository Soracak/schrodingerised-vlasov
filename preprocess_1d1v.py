# preprocess_1d1v.py
import numpy as np
import math
import matplotlib.pyplot as plt

def build_p_profile(
    p_grid: np.ndarray,
    *,
    profile: str = "original",
    eps: float = 1e-6,
    a: float | None = None,
    normalize: bool = False,
):
    """
    Build the auxiliary p-profile fp on the warped Fourier grid.

    Supported profiles
    ------------------
    original
        fp(p) = exp(-|p|)

    hermite
        Piecewise Hermite-smoothed profile:
            exp(p),                p < -1
            h_interp(p),         -1 <= p < 0
            exp(-p),               p >= 0

    erf
        Jin et al. optimal-style profile:
            fp(p) = exp(-p) * (1 + erf(a p)) / 2

        If a is None, use:
            a = 2 * sqrt(log(1/eps))

    Parameters
    ----------
    p_grid
        Warped Fourier grid.
    profile
        Choice of profile: "original", "hermite", or "erf".
    eps
        Error tolerance used only for default erf parameter choice.
    a
        erf smoothing parameter. If None and profile="erf", use Jin et al. choice.
    normalize
        If True, normalize fp before returning.

    Returns
    -------
    fp : np.ndarray
        Real nonnegative profile values on p_grid.
    meta : dict
        Metadata including chosen profile and a if relevant.
    """
    profile = profile.lower()

    if profile == "original":
        fp = np.exp(-np.abs(p_grid))
        meta = {"profile": "original"}

    elif profile == "hermite":
        e1 = np.exp(-1.0)

        def h_interp(p):
            return (-3.0 + 3.0 * e1) * p**3 + (-5.0 + 4.0 * e1) * p**2 - p + 1.0

        fp = np.where(
            p_grid < -1.0,
            np.exp(p_grid),
            np.where(p_grid < 0.0, h_interp(p_grid), np.exp(-p_grid)),
        )
        fp = np.maximum(fp, 0.0)
        meta = {"profile": "hermite"}

    elif profile == "erf":
        if a is None:
            a = 2.0 * np.sqrt(np.log(1.0 / eps))

        erf_vals = np.array([math.erf(a * p) for p in p_grid], dtype=float)
        fp = np.exp(-p_grid) * 0.5 * (1.0 + erf_vals)
        fp = np.maximum(fp, 0.0)

        p_star = np.sqrt(np.log(1.0 / eps)) / a
        meta = {"profile": "erf", "a": a, "eps": eps, "p_star": p_star}

    else:
        raise ValueError(
            f"Unknown profile='{profile}'. Choose from "
            f"'original', 'hermite', or 'erf'."
        )

    if normalize:
        nrm = np.linalg.norm(fp)
        if nrm <= 0:
            raise ValueError("Constructed fp has zero norm.")
        fp = fp / nrm

    return fp, meta

def plot_p_profile(fp, p_grid, fp_meta, show=True):
    fig, ax = plt.subplots(figsize=(4.5, 4))

    ax.plot(p_grid, fp, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$f_p$")

    title = f"p-profile: {fp_meta['profile']}"
    if fp_meta["profile"] == "erf":
        title += rf"  ($a={fp_meta['a']:.3f}$, $p^*={fp_meta['p_star']:.3f}$)"

    ax.set_title(title)
    fig.tight_layout()

    if show:
        plt.show()

    return fig