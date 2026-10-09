"""Numerical conventions used in the 1D1V paper; no Qiskit required."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_utils import compare_quantum_classical
from classical_schemes_1d1v import a_x_from_centered_fixed_point


def test_comparison_uses_independent_l2_normalizations():
    c = np.array([[1.0, 2.0], [3.0, 4.0]])
    q = np.array([[1.0, 2.5], [3.0, 4.0]]) * 2.0
    err, *_ = compare_quantum_classical(q, c)
    target = np.linalg.norm(q / np.linalg.norm(q) - c / np.linalg.norm(c))
    assert np.isclose(err, target, rtol=0, atol=1e-14)
    assert not np.isclose(err, np.linalg.norm(q-c)/np.linalg.norm(c))


def test_grid_sensitivity_cubic_coefficient_remains_nonzero():
    nx, lx, s = 3, 6.0, 6
    nxpoints = 1 << nx
    dx = lx / nxpoints
    S = 1 << s
    alpha_fp = round(S * dx)
    beta_fp = round(-S * dx**3)
    assert (alpha_fp, beta_fp) == (48, -27)
    field = a_x_from_centered_fixed_point(nx, alpha_fp, beta_fp, S)
    c = np.arange(nxpoints)-nxpoints/2
    np.testing.assert_allclose(field, (alpha_fp*c+beta_fp*c**3)/S)


def test_grid_choice_only_changes_initial_coordinates():
    nx, lx = 3, 6.0
    j = np.arange(1 << nx)
    dx = lx / (1 << nx)
    cell = -lx/2+(j+0.5)*dx
    nodal = -lx/2+j*dx
    np.testing.assert_allclose(cell-nodal, np.full(j.shape, dx/2))
