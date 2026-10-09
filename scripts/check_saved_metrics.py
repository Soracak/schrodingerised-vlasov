"""Check stored unit-L2 discrepancy and both overlaps from 1D1V result NPZ files.

Usage: python scripts/check_saved_metrics.py results/grid-cell_*/data/run_solution_centered.npz
Also usable with any completed 1D1V run_solution_centered.npz file.
"""
from pathlib import Path
import sys
import numpy as np


def compare(path):
    with np.load(path, allow_pickle=False) as d:
        u = np.asarray(d["u_quant"], dtype=float)
        c = np.asarray(d["f_class"], dtype=float)
        stored = float(d["rel_l2"])
    if u.shape != c.shape or not np.linalg.norm(u) or not np.linalg.norm(c):
        raise ValueError(f"Invalid arrays in {path}")
    ul2 = u / np.linalg.norm(u)
    cl2 = c / np.linalg.norm(c)
    calc = float(np.linalg.norm(ul2-cl2))
    overlap = float(np.vdot(ul2.ravel(), cl2.ravel()).real)
    print(f"{path}: stored={stored:.12g}, recalculated={calc:.12g}, "
          f"overlap={overlap:.9g}, match={np.isclose(stored,calc,rtol=1e-9,atol=1e-12)}")
    if not np.isclose(stored, calc, rtol=1e-9, atol=1e-12):
        raise AssertionError(f"Mismatch in {path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python scripts/check_saved_metrics.py <NPZ> [<NPZ> ...]")
    for a in sys.argv[1:]:
        compare(Path(a))
