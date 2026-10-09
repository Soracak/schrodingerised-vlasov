import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Select completed run
# ============================================================

if len(sys.argv) != 2:
    raise SystemExit(
        "Usage: python plot_completed_run.py results/RUN_ID "
        "(requires run_solution_centered.npz and run_params_centered.npz)"
    )
run_dir = Path(sys.argv[1])

data_file = run_dir / "data" / "run_solution_centered.npz"

if not data_file.exists():
    raise FileNotFoundError(f"Cannot find:\n{data_file}")

D = np.load(data_file, allow_pickle=False)

fQ = D["fQ"]
fC = D["fC"]
f0 = D["f0"]

x = D["x_grid"]
v = D["v_grid"]

Nt = int(D["Nt"])
dt = float(D["dt"])
T = Nt * dt

rel_l2 = float(D["rel_l2"])

print(f"Run       : {run_dir}")
print(f"Nt        : {Nt}")
print(f"dt        : {dt}")
print(f"T         : {T}")
print(f"Relative L2 error : {rel_l2:.6e}")

fig_dir = run_dir / "figures_post"
fig_dir.mkdir(parents=True, exist_ok=True)



# ============================================================
# 0. Auxiliary Schrödingerisation profile f(p)
# ============================================================

# Load auxiliary p-profile data
params_file = run_dir / "data" / "run_params_centered.npz"

if not params_file.exists():
    raise FileNotFoundError(f"Cannot find:\n{params_file}")

P = np.load(params_file, allow_pickle=False)

p_grid = P["p_grid"]
fp = P["fp"]

# Normalize by the maximum only for plotting, matching the convention used
# in the main simulation script when optional initial plots are enabled.
fp_plot = fp / max(np.max(np.abs(fp)), 1e-15)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    p_grid,
    fp_plot,
    "ko-",
    markersize=3,
    linewidth=1.2,
)

ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0)

ax.tick_params(labelsize=14)
ax.set_xlabel(r"$p$", fontsize=16)
#ax.set_ylabel(r"$f(p)/\max |f(p)|$", fontsize=16)
ax.set_ylabel(r"$\widetilde{f}(p)$", fontsize=16)
#ax.set_title(r"Auxiliary profile $f(p)$", fontsize=12)

fig.savefig(
    fig_dir / "p_profile.png",
    dpi=300,
)
fig.savefig(
    fig_dir / "p_profile.pdf",
)

plt.close(fig)


# ============================================================
# 0. Initial solution at t = 0
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.pcolormesh(
    x,
    v,
    f0.T,
    shading="auto",
)

cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=14)

ax.tick_params(labelsize=14)
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$v_x$", fontsize=16)
ax.set_title(r"$f_0$", fontsize=16)

ax.set_box_aspect(1)
#fig.tight_layout()


fig.savefig(
    fig_dir / "initial_solution.png",
    dpi=300,
)
fig.savefig(
    fig_dir / "initial_solution.pdf",
)

plt.close(fig)

# ============================================================
# 1. Quantum solution
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.pcolormesh(
    x,
    v,
    fQ.T,
    shading="auto",
)

cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=14)

ax.tick_params(labelsize=14)
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$v_x$", fontsize=16)
ax.set_title(
    rf"$f_Q$, "
    rf"$\Delta t={dt:g}$, $T={T:g}$", fontsize=16)

ax.set_box_aspect(1) 
#fig.tight_layout()


fig.savefig(
    fig_dir / "quantum_solution.png",
    dpi=300,
)
fig.savefig(
    fig_dir / "quantum_solution.pdf",
)

plt.close(fig)




# ============================================================
# 2. Classical solution
# ============================================================

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.pcolormesh(
    x,
    v,
    fC.T,
    shading="auto",
)

cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=14)

ax.tick_params(labelsize=14)
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$v_x$", fontsize=16)
ax.set_title(
    rf"$f_C$, "
    rf"$\Delta t={dt:g}$, $T={T:g}$", fontsize=16)

ax.set_box_aspect(1) 
#fig.tight_layout()


fig.savefig(
    fig_dir / "classical_solution.png",
    dpi=300,
)
fig.savefig(
    fig_dir / "classical_solution.pdf",
)

plt.close(fig)


# ============================================================
# 3. Quantum - classical difference
# ============================================================

diff = fQ - fC

fig, ax = plt.subplots(figsize=(6, 5))

lim = np.max(np.abs(diff))

im = ax.pcolormesh(
    x,
    v,
    diff.T,
    shading="auto",
    vmin=-lim,
    vmax=lim,
)

cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=14)
cbar.ax.yaxis.get_offset_text().set_size(14)

ax.tick_params(labelsize=14)
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$v_x$", fontsize=16)
ax.set_title(
    rf"$f_Q-f_C$, "
    rf"$\Delta t={dt:g}$, $T={T:g}$", fontsize=16)

ax.set_box_aspect(1) 
#fig.tight_layout()


fig.savefig(
    fig_dir / "quantum_classical_difference.png",
    dpi=300,
)
fig.savefig(
    fig_dir / "quantum_classical_difference.pdf",
)

plt.close(fig)


# ============================================================
# 4. Spatial marginal
# ============================================================

rhoQ = np.sum(fQ, axis=1)
rhoC = np.sum(fC, axis=1)
rho0 = np.sum(f0, axis=1)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(x, rho0, "k--", label=r"$t=0$")
ax.plot(x, rhoC, "k-", label="Classical")
ax.plot(x, rhoQ, "ko", markersize=5, label="Quantum")

ax.tick_params(labelsize=14)
ax.set_xlabel(r"$x$", fontsize=16)
ax.set_ylabel(r"$\sum_{v_x} f(x,v_x)$", fontsize=16)
ax.set_title(rf"Spatial marginal, $T={T:g}$", fontsize=16)
ax.legend(frameon=False, fontsize=14) 

fig.tight_layout()

fig.savefig(
    fig_dir / "spatial_marginal.png",
    dpi=300,
)
fig.savefig(
    fig_dir / "spatial_marginal.pdf",
)

plt.close(fig)


# ============================================================
# 5. Velocity marginal
# ============================================================

gQ = np.sum(fQ, axis=0)
gC = np.sum(fC, axis=0)
g0 = np.sum(f0, axis=0)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(v, g0, "k--", label=r"$t=0$")
ax.plot(v, gC, "k-", label="Classical")
ax.plot(v, gQ, "ko", markersize=5, label="Quantum")

ax.tick_params(labelsize=14)
ax.set_xlabel(r"$v_x$", fontsize=16)
ax.set_ylabel(r"$\sum_{x} f(x,v_x)$", fontsize=16)
ax.set_title(rf"Velocity marginal, $T={T:g}$", fontsize=16)
ax.legend(frameon=False, fontsize=14)  

fig.tight_layout()

fig.savefig(
    fig_dir / "velocity_marginal.png",
    dpi=300,
)
fig.savefig(
    fig_dir / "velocity_marginal.pdf",
)

plt.close(fig)

print(f"\nFigures saved to:\n{fig_dir}")