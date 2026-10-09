"""CPU-only invariant checks; does not need Qiskit."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classical_schemes_1d1v import (
    a_x_from_centered_fixed_point,
    strang_integrate_classical_hu_fft_scaled,
)
from preprocess_1d1v import build_p_profile


def test_t1_sampling_offset_and_effective_field():
    nx, lx, s = 5, 6.0, 6
    nx_points = 1 << nx
    dx = lx / nx_points
    j = np.arange(nx_points)
    x_initial = -lx/2 + (j+0.5)*dx
    x_field = (j-nx_points/2)*dx
    assert np.allclose(x_initial-x_field, dx/2)
    alpha_fp = round((dx)*2**s)
    beta_fp = round((-(dx**3))*2**s)
    assert (alpha_fp, beta_fp) == (12, 0)
    field = a_x_from_centered_fixed_point(nx,alpha_fp,beta_fp,2**s)
    assert np.allclose(field,x_field)


def test_classical_reference_mass_conservation():
    nx, nv = 4, 4
    lx, lv, r = 6.0, 6.0, 6.0
    x = -lx/2 + (np.arange(1<<nx)+0.5)*lx/(1<<nx)
    v = -lv + (np.arange(1<<nv)+0.5)*2*lv/(1<<nv)
    f0 = np.exp(-0.5*(x[:,None]**2 + v[None,:]**2))
    f0 /= f0.sum()
    ax = a_x_from_centered_fixed_point(nx,24,0,64)
    f = strang_integrate_classical_hu_fft_scaled(f0,v,ax,0.01,3,lx,lv,r)
    assert np.all(np.isfinite(f))
    assert np.isclose(np.sum(f),1.0,rtol=0,atol=1e-12)


def test_auxiliary_profiles_are_finite_and_nonnegative():
    p = -6*np.pi + np.arange(128)*(12*np.pi/128)
    for name in ('original','hermite','erf'):
        fp,_ = build_p_profile(p,profile=name)
        assert fp.shape == p.shape
        assert np.all(np.isfinite(fp))
        assert np.all(fp>=0)
        assert np.linalg.norm(fp)>0
