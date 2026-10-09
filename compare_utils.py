# compare_utils.py
import numpy as np
import matplotlib.pyplot as plt

def compare_quantum_classical(u_quant, f_class):
    uq = u_quant / np.linalg.norm(u_quant)
    fc = f_class / np.linalg.norm(f_class)
    rel_l2 = np.linalg.norm(uq - fc) / np.linalg.norm(fc)

    mass_q = float(np.sum(np.maximum(u_quant, 0)))
    mass_c = float(np.sum(f_class))
    return rel_l2, mass_q, mass_c, uq, fc

def compare_slices(uq, fc, f0, x_grid, v_grid, Nt, dt, ix=None, iv=None):
    Nx, Nv = uq.shape
    if iv is None: iv = Nv // 2
    if ix is None: ix = Nx // 2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    ax1.plot(x_grid, f0[:, iv], "k--", label="initial (t=0)")
    ax1.plot(x_grid, fc[:, iv], "k-", label="classical")
    ax1.plot(x_grid, uq[:, iv], "r-", label="quantum")
    ax1.set_xlabel("x"); ax1.set_ylabel(f"f(x,v[{iv}])")
    ax1.set_title(f"v-slice (Nt={Nt}, dt={dt:g})")
    ax1.legend(frameon=False)

    ax2.plot(v_grid, f0[ix, :], "k--", label="initial (t=0)")
    ax2.plot(v_grid, fc[ix, :], "k-", label="classical")
    ax2.plot(v_grid, uq[ix, :], "r-", label="quantum")
    ax2.set_xlabel("v"); ax2.set_ylabel(f"f(x[{ix}],v)")
    ax2.set_title(f"x-slice (Nt={Nt}, dt={dt:g})")
    ax2.legend(frameon=False)

    fig.tight_layout()
    return fig

