"""Generate four separate Appendix C figure panels from two completed 1D1V runs.

Spyder: set BASE and, if needed, CELL_RUN/NODAL_RUN below; run (F5).
CLI:   python scripts/make_grid_sensitivity_panels.py

Requires two 'data/run_solution_centered.npz' files; never reruns Qiskit.
The saved fQ/fC/f0 are separately unit-mass normalised plotting arrays.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# %% USER SETTINGS: Change only as needed in Spyder.
BASE = Path.cwd() / "results"   # or Path.cwd() if cwd already contains the run folders
CELL_RUN = None   # Example: BASE / "grid-cell_nx3_nv3_na5_Nt1_dt0.05_v00_R4"
NODAL_RUN = None  # Example: BASE / "grid-nodal_nx3_nv3_na5_Nt1_dt0.05_v00_R4"
OUTDIR = BASE / "grid_sensitivity_panels"
PNG_DPI = 300


def _choose_folder(explicit, prefix):
    if explicit is not None:
        candidate = Path(explicit)
        if not (candidate / "data" / "run_solution_centered.npz").is_file():
            raise FileNotFoundError(candidate / "data" / "run_solution_centered.npz")
        return candidate
    candidates = sorted(p for p in BASE.glob(prefix + "*")
                        if (p / "data" / "run_solution_centered.npz").is_file())
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one {prefix}* run under {BASE}; found {len(candidates)}. "
                           f"Set CELL_RUN/NODAL_RUN explicitly if multiple runs exist.")
    return candidates[0]


def _load(folder):
    with np.load(folder / "data" / "run_solution_centered.npz", allow_pickle=False) as d:
        expected = {"x_grid", "f0", "fQ", "fC", "rel_l2", "Nt", "dt"}
        missing = expected.difference(d.files)
        if missing:
            raise ValueError(f"Missing {missing} from {folder}")
        data = {key: d[key].copy() for key in expected}
    x = data["x_grid"]
    for key in ("f0", "fQ", "fC"):
        if data[key].ndim != 2 or data[key].shape[0] != x.size:
            raise ValueError(f"Incompatible shape: {key}")
    return data


def _save(fig, name):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        opts = {"dpi": PNG_DPI} if ext == "png" else {}
        fig.savefig(OUTDIR / f"{name}.{ext}", bbox_inches="tight", **opts)
    plt.close(fig)


def _make_marginal_panel(data, mode, stem):
    x = data["x_grid"]
    rho0, rhoQ, rhoC = (np.sum(data[k], axis=1) for k in ("f0", "fQ", "fC"))
    fig, ax = plt.subplots(figsize=(5.3, 4.0))
    ax.plot(x, rho0, "k--", lw=1.4, label=r"Initial $t=0$")
    ax.plot(x, rhoC, "-", lw=1.8, label="Classical")
    ax.plot(x, rhoQ, "o", ms=4.5, label="Quantum")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\rho(x)=\sum_k f(x,v_k)$")
    ax.set_title(f"{mode}; " + rf"$T={float(data['Nt'])*float(data['dt']):g}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save(fig, stem)


def _make_difference_panel(data, mode, stem, ylim):
    x = data["x_grid"]
    difference = np.sum(data["fQ"]-data["fC"], axis=1)
    fig, ax = plt.subplots(figsize=(5.3, 4.0))
    ax.axhline(0, color="k", linestyle="--", lw=0.8)
    ax.plot(x, difference, "o-", lw=1.4, ms=4.5)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\sum_k[f_Q(x,v_k)-f_C(x,v_k)]$")
    ax.set_ylim(-ylim, ylim)
    ax.set_title(f"{mode}; quantum $-$ classical")
    fig.tight_layout()
    _save(fig, stem)


def main():
    cell = _load(_choose_folder(CELL_RUN, "grid-cell_"))
    nodal = _load(_choose_folder(NODAL_RUN, "grid-nodal_"))
    max_difference = max(np.max(np.abs(np.sum(d["fQ"] - d["fC"], axis=1)))
                         for d in (cell, nodal))
    ylim = float(max(max_difference * 1.15, 1e-12))
    _make_marginal_panel(cell, "Cell-centred initialization", "panel_a_cell_marginal")
    _make_marginal_panel(nodal, "Nodal initialization", "panel_b_nodal_marginal")
    _make_difference_panel(cell, "Cell-centred initialization", "panel_c_cell_difference", ylim)
    _make_difference_panel(nodal, "Nodal initialization", "panel_d_nodal_difference", ylim)
    print(f"Saved 4 PDF and 4 PNG panels in: {OUTDIR.resolve()}")
    print(f"Stored unit-L2 quantum-classical errors: cell={float(cell['rel_l2']):.9g}, "
          f"nodal={float(nodal['rel_l2']):.9g}")


if __name__ == "__main__":
    main()
