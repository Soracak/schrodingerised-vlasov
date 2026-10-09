"""
schro_1d1v_dwell_centered_mps.py
================================

Purpose
-------
End-to-end *proof-of-concept* for 1D1V prescribed-field (linear) Vlasov evolution
using Schrödingerization-style circuit blocks, plus a classical reference.

We simulate the 1D1V transport equation with fixed external acceleration a(x):

    ∂_t f(x,v,t) + v ∂_x f(x,v,t) + a(x) ∂_v f(x,v,t) = 0,

using Strang splitting (Ux half-step, Uv full-step, Ux half-step) repeated Nt times.

External field
--------------
We use a double-well external potential:
    Φ(x) = (κ/4) (x^2 - a0^2)^2,
so the prescribed acceleration is:
    a(x) = -dΦ/dx = -κ (x^3 - a0^2 x).

Centered-grid fixed-point E(x)
-------------------------------------------------

The coherent force uses a centred-integer grid, shifted by half a cell
relative to the cell-centred initial-condition grid:

    c = j - 2^(nx-1)                         (centered integer coordinate)
    E_fp(c) = alpha_fp * c + beta_fp * c^3   (two's complement integer)

The floating acceleration used by the classical reference is:
    a_x[j] = E_fp(j) / S,   where S = 2^s is the fixed-point scale.

Practical implications:
- The X register still stores the index j; the E oracle internally forms centered c.
- The E accumulator width must be:
      nE = compute_nE_centered(nx, alpha_fp, beta_fp)
- The E oracle uses additional workspace (cube/adder scratch + carry regs).

Circuit structure (high level)
------------------------------
State prep:
- |X,V> amplitude-encoded from f0(x,v)
- |P> amplitude-encoded from fp(p) (warped Fourier grid)

Main unitary:
- Transform P to the auxiliary Fourier basis
- Strang-splitting sequence using reusable Ux / Uv blocks
- Transform P back to the p-grid for readout

Prerequisite files
------------------------------
schro_1d1v_blocks.py
build_E_centered.py
classical_schemes_1d1v.py
ft_resources.py
postprocess_1d1v.py
preprocess_1d1v.py
compare_utils.py
readout_utils_mps.py
io_utils.py



Diagnostics
-----------
- Compare quantum output against a scheme-matched classical FFT reference
- Report FT-style resources using rotation-synthesis proxy (counts of u/p/rz/cp + cx)
  rather than Toffoli counts (the circuit is compiled into {u,cx} form).
"""

import numpy as np
import argparse

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFTGate, Initialize
from IPython.display import display

import time
import os
import platform
import psutil
from importlib.metadata import version, PackageNotFoundError

# Safe default for headless cluster jobs; an explicit MPLBACKEND still wins.
os.environ.setdefault("MPLBACKEND", "Agg")

# Builders
from build_E_centered import compute_E_poly_centered, compute_nE_centered
from ft_resources import ft_resource_row
from schro_1d1v_blocks import (
    cvV1, inverse_cvV1, vV2,
    cEV1, inverse_cEV1, eV2,
    theta_from_v_grid_linear,
    theta_bits_from_twos_complement
)    

# Helpers
from preprocess_1d1v import build_p_profile
from postprocess_1d1v import plot_f, initial_slices
from classical_schemes_1d1v import (
    strang_integrate_classical_hu_fft_scaled,
    a_x_from_centered_fixed_point,
)

from io_utils import export_centered_run

# ============================================================
# Output helpers
# ============================================================
from pathlib import Path
import json
import matplotlib.pyplot as plt

def make_output_dirs(run_id: str = "Q1", base_dir: str = "results"):
    """
    Create a run-specific output folder.

    Example:
        results/Q1/
            figures/
            data/
            resources/
            logs/
    """
    out_root = Path(base_dir) / run_id

    out = {
        "root": out_root,
        "figures": out_root / "figures",
        "data": out_root / "data",
        "resources": out_root / "resources",
        "logs": out_root / "logs",
    }

    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)

    return out


def save_fig(fig, out_dir, stem: str, *, dpi: int = 300):
    """
    Save a matplotlib figure as PNG and PDF.
    """
    if fig is None:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def _json_default(obj):
    """
    Convert NumPy objects to JSON-safe Python objects.
    """
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        
def _package_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def collect_environment():
    try:
        affinity = psutil.Process().cpu_affinity()
    except (AttributeError, psutil.Error):
        affinity = None

    return {
        "python_version": platform.python_version(),
        "operating_system": platform.platform(),
        "processor": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cpus": psutil.cpu_count(logical=True),
        "process_cpu_affinity": affinity,
        "qiskit_version": _package_version("qiskit"),
        "qiskit_aer_version": _package_version("qiskit-aer"),
        "numpy_version": _package_version("numpy"),
        "psutil_version": _package_version("psutil"),
        "thread_environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        },
    }

# ============================================g
# 1d1v landau double well fixed E
# ============================================

if __name__ == "__main__":

    # -----------------------------------------------------------------
    # Command-line configuration
    # -----------------------------------------------------------------
    # Production-safe defaults: MPS readout enabled; diagnostic MPS logging,
    # FT resource decomposition, circuit drawing, and initial plotting disabled.
    # Launch T1/T2/T5/T10 via CLI arguments so this file can remain frozen while
    # multiple PBS jobs are queued.
    parser = argparse.ArgumentParser(
        description="1D1V centered-field Schrödingerization MPS benchmark"
    )
    parser.add_argument("--nx", type=int, default=5)
    parser.add_argument("--nv", type=int, default=5)
    parser.add_argument("--na", type=int, default=7)
    parser.add_argument("--nt", "--Nt", dest="Nt", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--v0", type=float, default=4.0,
                        help="Initial mean velocity of the Maxwellian")
    parser.add_argument("--lx", type=float, default=6.0)
    parser.add_argument("--lv", type=float, default=6.0)
    parser.add_argument("--R", "--r", dest="R", type=float, default=6.0)
    parser.add_argument("--run-id", default=None)

    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--swap-direction",
        choices=("mps_swap_left", "mps_swap_right"),
        default="mps_swap_left",
    )
    parser.add_argument("--lapack", action="store_true")
    parser.add_argument("--mps-log-data", action="store_true")
    parser.add_argument("--parallel-threshold", type=int, default=14)
    parser.add_argument("--max-bond", type=int, default=512)
    parser.add_argument("--truncation", type=float, default=1e-12)
    parser.add_argument("--readout-reps", type=int, default=4)

    parser.add_argument("--no-readout", action="store_true")
    parser.add_argument("--statevector", action="store_true")
    parser.add_argument("--resource-report", action="store_true")
    parser.add_argument("--draw-circuit", action="store_true")
    parser.add_argument("--initial-plots", action="store_true")
    parser.add_argument("--no-output-plots", action="store_true")
    parser.add_argument("--export", action="store_true")

    args = parser.parse_args()

    # -----------------------------
    # Grids and parameters
    # -----------------------------
    nx = args.nx
    nv = args.nv
    na = args.na
    Nt = args.Nt
    dt = args.dt
    v0 = args.v0

    Lx = args.lx              # full x-domain length: [-Lx/2, Lx/2)
    Lv = args.lv              # velocity half-width: [-Lv, Lv]
    R = args.R                # finite auxiliary Fourier limit

    if min(nx, nv, na) <= 0:
        raise ValueError("nx, nv and na must all be positive.")
    if Nt <= 0:
        raise ValueError("Nt must be positive.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if Lx <= 0.0 or Lv <= 0.0:
        raise ValueError("Lx and Lv must be positive.")
    if R <= 0.0:
        raise ValueError("R must be positive.")

    Nx = 1 << nx
    Nv = 1 << nv
    dx = Lx / Nx
    dv = (2.0 * Lv) / Nv

    def x_centered(j: int) -> float:
        return (j + 0.5) * dx - 0.5 * Lx

    def v_centered(k: int) -> float:
        return -Lv + (k + 0.5) * dv

    # -----------------------
    # Runtime / readout flags
    # -----------------------
    RUN_ID = args.run_id or f"nx{nx}_nv{nv}_na{na}_Nt{Nt}_dt{dt:g}"
    OUT = make_output_dirs(RUN_ID, base_dir="results")

    DO_READOUT = not args.no_readout
    DO_STATEVECTOR = bool(args.statevector)
    DO_EXPORT = bool(args.export)
    DO_DRAW_CIRCUIT = bool(args.draw_circuit)
    DO_RESOURCE_REPORT = bool(args.resource_report)
    DO_INITIAL_PLOTS = bool(args.initial_plots)
    DO_OUTPUT_PLOTS = not args.no_output_plots
    READOUT_REPS = args.readout_reps

    if READOUT_REPS < 0:
        raise ValueError("--readout-reps must be non-negative.")

    # Resource estimation recursively decomposes a separate large circuit.
    # Keep it out of multi-day MPS production jobs.
    if DO_RESOURCE_REPORT and DO_READOUT:
        raise ValueError(
            "Do not combine --resource-report with MPS readout in a production "
            "job. Run resource counting separately with --resource-report "
            "--no-readout."
        )

    # The full-statevector branch is for tiny diagnostics only.
    if DO_READOUT and DO_STATEVECTOR and (nx + nv + na) > 15:
        raise ValueError(
            "--statevector is intended only for tiny validation cases. "
            "Use the default flagged MPS readout for production."
        )

    # ------------------------------------------------------------
    # MPS production options
    # ------------------------------------------------------------
    MPS_SWAP_DIRECTION = args.swap_direction
    MPS_THREADS = args.threads
    MPS_USE_LAPACK = bool(args.lapack)
    MPS_LOG_DATA = bool(args.mps_log_data)
    MPS_PARALLEL_THRESHOLD = args.parallel_threshold
    MPS_MAX_BOND_DIMENSION = args.max_bond
    MPS_TRUNCATION_THRESHOLD = args.truncation

    if MPS_THREADS <= 0:
        raise ValueError("--threads must be positive.")
    if MPS_PARALLEL_THRESHOLD < 0:
        raise ValueError("--parallel-threshold must be non-negative.")
    if MPS_MAX_BOND_DIMENSION <= 0:
        raise ValueError("--max-bond must be positive.")
    if not (0.0 <= MPS_TRUNCATION_THRESHOLD < 1.0):
        raise ValueError("--truncation must lie in [0,1).")

    MPS_SIMULATOR_OPTIONS = {
        "matrix_product_state_max_bond_dimension": MPS_MAX_BOND_DIMENSION,
        "matrix_product_state_truncation_threshold": MPS_TRUNCATION_THRESHOLD,
        "mps_log_data": MPS_LOG_DATA,
        "mps_swap_direction": MPS_SWAP_DIRECTION,
        "max_parallel_threads": MPS_THREADS,
        "max_parallel_experiments": 1,
        "max_parallel_shots": 1,
        "mps_omp_threads": MPS_THREADS,
        "mps_parallel_threshold": MPS_PARALLEL_THRESHOLD,
        "mps_lapack": MPS_USE_LAPACK,
    }

    ENVIRONMENT_INFO = collect_environment()

    swap_tag = MPS_SWAP_DIRECTION.replace("mps_swap_", "")
    
    CPU_LOG_PATH = (
        OUT["logs"]
        / (
            f"cpu_nx{nx}_nv{nv}_na{na}_Nt{Nt}"
            f"_thr{MPS_THREADS}"
            f"_threshold{MPS_PARALLEL_THRESHOLD}"
            f"_swap-{swap_tag}"
            f"_lapack-{int(MPS_USE_LAPACK)}.json"
        )
    )
    
    ENVIRONMENT_LOG_PATH = (
        OUT["logs"]
        / (
            f"environment_nx{nx}_nv{nv}_na{na}_Nt{Nt}"
            f"_thr{MPS_THREADS}"
            f"_swap-{swap_tag}"
            f"_lapack-{int(MPS_USE_LAPACK)}.json"
        )
    )
    
    save_json(ENVIRONMENT_LOG_PATH, ENVIRONMENT_INFO)

    launch_config = {
        "RUN_ID": RUN_ID,
        "nx": nx,
        "nv": nv,
        "na": na,
        "Nt": Nt,
        "dt": dt,
        "v0": v0,
        "Lx": Lx,
        "Lv": Lv,
        "R": R,
        "DO_READOUT": DO_READOUT,
        "DO_STATEVECTOR": DO_STATEVECTOR,
        "DO_EXPORT": DO_EXPORT,
        "DO_DRAW_CIRCUIT": DO_DRAW_CIRCUIT,
        "DO_RESOURCE_REPORT": DO_RESOURCE_REPORT,
        "DO_INITIAL_PLOTS": DO_INITIAL_PLOTS,
        "DO_OUTPUT_PLOTS": DO_OUTPUT_PLOTS,
        "READOUT_REPS": READOUT_REPS,
        "MPS_SWAP_DIRECTION": MPS_SWAP_DIRECTION,
        "MPS_THREADS": MPS_THREADS,
        "MPS_USE_LAPACK": MPS_USE_LAPACK,
        "MPS_LOG_DATA": MPS_LOG_DATA,
        "MPS_PARALLEL_THRESHOLD": MPS_PARALLEL_THRESHOLD,
        "MPS_MAX_BOND_DIMENSION": MPS_MAX_BOND_DIMENSION,
        "MPS_TRUNCATION_THRESHOLD": MPS_TRUNCATION_THRESHOLD,
    }
    save_json(OUT["data"] / "launch_config.json", launch_config)
    
    print("\n[environment]")
    print(f"Python              : {ENVIRONMENT_INFO['python_version']}")
    print(f"OS                  : {ENVIRONMENT_INFO['operating_system']}")
    print(f"physical cores      : {ENVIRONMENT_INFO['physical_cores']}")
    print(f"logical CPUs        : {ENVIRONMENT_INFO['logical_cpus']}")
    print(f"Qiskit              : {ENVIRONMENT_INFO['qiskit_version']}")
    print(f"Qiskit Aer          : {ENVIRONMENT_INFO['qiskit_aer_version']}")
    print(f"environment log     : {ENVIRONMENT_LOG_PATH}")
    print(f"CPU monitor log     : {CPU_LOG_PATH}")

    # Auxiliary-Fourier convention.
    # We fix the convention used in the manuscript, consistent with
    # ∂_p -> -iη: apply IQFT before the Schrödingerised evolution and
    # QFT after it.  With this convention the A1/Fourier-mode sign is positive.
    
    # ----------------------------------
    # Choose auxiliary p-profile here
    # ----------------------------------
    P_PROFILE = "original"   # options: "original", "hermite", "erf"
    P_EPS     = 1e-6
    P_A       = None         # set a float to override, else uses Jin et al. default for erf

    # Readout rule for the erf profile.  The original profile still uses p=0.
    P_READOUT_RULE   = "bounded_pi"  # options for erf: "bounded_pi", "first_after_pstar"
    P_READOUT_TARGET = np.pi
    P_READOUT_GUARD  = 3             # guard, in p-grid points, from transition/boundary


    
    print("\n[output]")
    print(f"run folder: {OUT['root']}")
    print("P_FOURIER_CONVENTION = paper  (fixed: IQFT -> evolution -> QFT)")
    print(f"P_READOUT_RULE       = {P_READOUT_RULE}")
    print(f"P_READOUT_TARGET     = {P_READOUT_TARGET}")
    print(f"P_READOUT_GUARD      = {P_READOUT_GUARD}")
    print(f"MPS_SWAP_DIRECTION   = {MPS_SWAP_DIRECTION}")
    print(f"MPS_THREADS          = {MPS_THREADS}")
    print(f"MPS_USE_LAPACK       = {MPS_USE_LAPACK}")
    print(f"MPS_LOG_DATA         = {MPS_LOG_DATA}")
    print(f"DO_READOUT           = {DO_READOUT}")
    print(f"DO_RESOURCE_REPORT   = {DO_RESOURCE_REPORT}")
    print(f"DO_INITIAL_PLOTS     = {DO_INITIAL_PLOTS}")

    # -----------------------------------------------------------------------------
    # Double-well external potential and acceleration field a(x) = -dΦ/dx
    # -----------------------------------------------------------------------------
    a0      = 1.0           # well locations at x = ±E0
    kappa   = 1.0           # stiffness parameter
    s       = 6             # fixed point precision
    
    def Phi(x: float) -> float:
        # Φ(x) = (κ/4) (x^2 - a0^2)^2
        return 0.25 * kappa * (x * x - a0 * a0) ** 2
    
    def accel(x: float) -> float:
        # a(x) = -dΦ/dx = -κ (x^3 - a0^2 x)
        return -kappa * (x ** 3 - (a0 * a0) * x)
    
    def fixed_point_coeffs(dx, a0, kappa, s, *, poly="double_well"):
        """
        Compute fixed-point integer coefficients for acceleration as a function of
        centered integer c, where x = dx * c.
    
        Default "double_well" corresponds to:
            a(x) = -kappa*(x^3 - a0^2*x) = (kappa*a0^2)*x + (-kappa)*x^3
        So:
            a(x) = alpha*c + beta*c^3
            alpha = kappa*a0^2*dx
            beta  = -kappa*dx^3
    
        Returns:
            alpha_fp, beta_fp, S, alpha, beta
        where alpha_fp ~ alpha*S and beta_fp ~ beta*S, S=2^s.
        """
        S = 1 << s
    
        if poly == "double_well":
            alpha = kappa * (a0 * a0) * dx
            beta  = -kappa * (dx ** 3)
        else:
            raise ValueError(f"Unknown poly='{poly}'. Only 'double_well' supported.")
    
        alpha_fp = int(round(alpha * S))
        beta_fp  = int(round(beta  * S))
        
        return alpha_fp, beta_fp, alpha, beta, S

    # Compute fixed point parameters
    alpha_fp, beta_fp, alpha, beta, S = fixed_point_coeffs(
                                            dx=dx, a0=a0, kappa=kappa, s=s
                                        )

    # -----------------------------------------------------------------------------
    # Timestep dt estimate (CFL)
    # -----------------------------------------------------------------------------

    # Conservative analytic bound: |a(x)| ≤ κ(Lx/2)^3 + κ a0^2 (Lx/2)
    Lx_half     = 0.5 * Lx
    a_max_bound = kappa * (Lx_half ** 3 + (a0 * a0) * Lx_half)
    
    # Characteristic speeds: x-advection speed max |v| ≈ Lv; v-advection max |a| ≤ a_max_bound.
    # dt <= nu * min(dx/|v|max, dv/|a|max)
    nu          = 0.5  # safety factor; try 0.2 (very safe) to 0.8 (aggressive)
    v_max       = Lv
    dt_cfl      = nu * min(dx / max(v_max, 1e-12), dv / max(a_max_bound, 1e-12))
       
    # -----------------------------------------------------------------------------
    # Initial condition f0(x,v): Boltzmann-in-x (external potential) × Maxwellian-in-v
    # -----------------------------------------------------------------------------
    beta_0  = 2.0   # controls concentration near wells
    sigma   = 1.0   # velocity thermal width
    
    def init_f(x: float, v: float) -> float:
        # Shifted Maxwellian with bulk drift v0; v_grid itself is unchanged.
        return np.exp(-beta_0 * Phi(x)) * np.exp(-((v - v0) ** 2) / (2.0 * sigma * sigma))

    # -----------------------------------------------------------------------------
    # Precompute grids / normalization (useful for debugging and small-state prep)
    # -----------------------------------------------------------------------------
    x_grid = np.array([x_centered(j) for j in range(Nx)], dtype=float)
    v_grid = np.array([v_centered(k) for k in range(Nv)], dtype=float)
    
    # f0 on grid
    f_init = np.zeros((Nx, Nv), dtype=float)
    
    for j in range(Nx):
        for k in range(Nv):
            f_init[j, k] = init_f(x_grid[j], v_grid[k])
    
    # Normalize to sum to 1 (treat as discrete probability mass for amplitude encoding)
    f_sum = float(np.sum(f_init))
    if f_sum <= 0:
        raise ValueError("Initial distribution has non-positive total mass.")
    f0 = f_init / f_sum
    
    # Initial figures are opt-in for production runs.
    fig0 = None
    fig0_slices = None
    if DO_INITIAL_PLOTS:
        fig0 = plot_f(f0, x_grid, v_grid, name=r"$f_0$")
        fig0_slices = initial_slices(f0, x_grid, v_grid)
        save_fig(fig0, OUT["figures"], "fig0_initial_f0")
        save_fig(fig0_slices, OUT["figures"], "fig0_initial_slices")
        plt.close(fig0)
        plt.close(fig0_slices)

    # Quick diagnostic prints
    print("\n=== 1D1V double-well prescribed-field Vlasov setup ===")
    print(f"\nRUN_ID = {RUN_ID}")
    print(f"n_x = {nx}, n_v = {nv} => Nx = {Nx}, Nv = {Nv}")
    print(f"x domain: centered in [{-Lx_half}, {Lx_half}), dx={dx}]")
    print(f"v domain: centered in [{-Lv}, {Lv}], dv={dv}]")
    print(f"double-well params: a0={a0}, kappa={kappa}")
    print(f"initial Maxwellian: v0={v0}, sigma={sigma}, beta_0={beta_0}")
    print(f"conservative bound |a(x)| <= {a_max_bound:.6g}")
    print(f"N_t = {Nt}, dt = {dt:.6g} (dt_cfl={dt_cfl:.6g}, nu={nu})\n")
    print(f"\u03B1_fp = {alpha_fp}, \u03B2_fp = {beta_fp}, "
          f"\u03B1 = {alpha}, \u03B2 = {beta}, S = {S}")
    
    # -----------------------------------------------------------------
    # Schrodingerisation
    # -----------------------------------------------------------------
    
    def choose_R(n_a, eps=1e-3, grid_frac=0.25):
        """
        Heuristic choice of R for given phase register size n_a.
    
        eps: desired overall error scale (order-of-magnitude)
        grid_frac: require R/Np <= grid_frac  (controls p-grid resolution)
                  smaller => stricter grid accuracy, larger => allow coarser p-grid
        """
        Np = 1 << n_a
    
        # tail target: make exp(-pi R) ~ eps
        R_tail = (1.0 / np.pi) * np.log(1.0 / eps)
    
        # grid limit: keep R/Np <= grid_frac
        R_grid = grid_frac * Np
    
        # choose feasible R (cap by grid)
        R = min(R_tail, R_grid)
    
        return R, {"Np": Np, "R_tail": R_tail, "R_grid_cap": R_grid}
    
    # Choose R
    R_suggest, info = choose_R(na, eps=1e-3, grid_frac=0.25)
    print(f"R(manual): {R:.3f} | R_suggest={R_suggest:.3f}")

    dp      = 2 * R * np.pi / 2**na                     # warped phase step
    p_grid  = (-R * np.pi) + dp * np.arange(2**na)      # warped phase grid
    
    
    
    fp, fp_meta = build_p_profile(
        p_grid,
        profile=P_PROFILE,
        eps=P_EPS,
        a=P_A,
        normalize=False,
    )
    
    print("\nP-profile selection:")
    print(f"  profile = {fp_meta['profile']}")
    if fp_meta["profile"] == "erf":
        print(f"  a       = {fp_meta['a']:.6f}")
        print(f"  eps     = {fp_meta['eps']:.1e}")
        print(f"  p_star  = {fp_meta['p_star']:.6f}")
        print(f"  width~1/a = {1.0 / fp_meta['a']:.6f}")
        print(f"  dp/(1/a)  = {dp * fp_meta['a']:.6f}")
        
    # Plot the P profile only when explicitly requested.
    fig_p = None
    if DO_INITIAL_PLOTS:
        fp_plot = fp / max(np.max(np.abs(fp)), 1e-15)

        fig_p, ax_p = plt.subplots(figsize=(4.5, 4))
        ax_p.plot(p_grid, fp_plot, "k-o", markersize=3, linewidth=1.4)
        ax_p.axvline(0.0, linestyle="--", linewidth=1.0, color="k")

        if fp_meta["profile"] == "erf":
            ax_p.axvline(fp_meta["p_star"], linestyle=":", linewidth=1.0)

        ax_p.set_xlabel(r"$p$")
        ax_p.set_ylabel(r"normalized $f_p(p)$")

        title = rf"$P$-profile: {fp_meta['profile']}"
        if fp_meta["profile"] == "erf":
            title += rf", $a={fp_meta['a']:.3f}$, $p^\ast={fp_meta['p_star']:.3f}$"

        ax_p.set_title(title)
        fig_p.tight_layout()
        save_fig(fig_p, OUT["figures"], "fig_p_profile")
        plt.close(fig_p)

    norm_base = np.linalg.norm(f0) * np.linalg.norm(fp)     # Initial norm factor
    norm1     = np.linalg.norm(fp[2**(na-1):])              # Positive-branch norm diagnostic only
    
    # ------------------------------------------------------------
    # Choose p-readout slice
    # ------------------------------------------------------------
    p_center_idx = 2**(na - 1)
    Lambda_plus_T = 0.0
    Lambda_minus_T = 0.0
    p_min_recovery = np.nan
    p_max_recovery = np.nan
    p_target_recovery = np.nan
    
    if fp_meta["profile"] == "erf":
        # The erf profile is one-sided: p=0 lies in the switching layer.
        # Use an interior positive recovery point, with a finite-grid guard.
        if P_READOUT_RULE == "bounded_pi":
            a_x_tmp = a_x_from_centered_fixed_point(nx, alpha_fp, beta_fp, S)
            vmax_grid = float(np.max(np.abs(v_grid)))
            amax_grid = float(np.max(np.abs(a_x_tmp)))

            # Conservative right-boundary drift for the present split blocks.
            # The dissipative Hu/Sato A1 part has no positive drift in this setup,
            # so Lambda_plus_T is set to zero.
            Lambda_plus_T = 0.0
            Lambda_minus_T = (Nt * dt) * (
                2.0 * vmax_grid / (R * dx)
                +
                2.0 * amax_grid / (R * dv)
            )

            p_min_recovery = fp_meta["p_star"] + Lambda_plus_T + P_READOUT_GUARD * dp
            p_max_recovery = p_grid[-1] - Lambda_minus_T - P_READOUT_GUARD * dp
            if p_min_recovery >= p_max_recovery:
                raise ValueError(
                    f"No safe erf recovery window: p_min={p_min_recovery}, "
                    f"p_max={p_max_recovery}. Increase R/na or reduce T."
                )

            p_target_recovery = float(np.clip(P_READOUT_TARGET, p_min_recovery, p_max_recovery))
            p_readout_idx = int(np.argmin(np.abs(p_grid - p_target_recovery)))
        elif P_READOUT_RULE == "first_after_pstar":
            p_readout_idx = int(np.searchsorted(p_grid, fp_meta["p_star"]))
            p_readout_idx = min(max(p_readout_idx, p_center_idx), len(p_grid) - 1)
            p_target_recovery = float(p_grid[p_readout_idx])
        else:
            raise ValueError(f"Unknown P_READOUT_RULE={P_READOUT_RULE!r}")
    else:
        # For original exp(-|p|), keep the validated center readout.
        p_readout_idx = p_center_idx
    
    p_readout = float(p_grid[p_readout_idx])
    fp_readout = float(fp[p_readout_idx])
    
    if abs(fp_readout) < 1e-15:
        raise ValueError("Readout p-profile value is too small for normalization.")
    
    if fp_meta["profile"] == "erf":
        # On the positive branch f_p^erf(p) ≈ e^{-p}; recover by multiplying by e^p.
        readout_scale = norm_base * np.exp(p_readout)
        readout_scale_mode = "exp(p)"
    else:
        readout_scale = norm_base / fp_readout
        readout_scale_mode = "1/fp"
    
    print("\n[p-readout]")
    print(f"p_center_idx  = {p_center_idx}")
    print(f"p_readout_idx = {p_readout_idx}")
    print(f"p_readout     = {p_readout:.12g}")
    print(f"fp_readout    = {fp_readout:.12e}")
    print(f"readout_scale = {readout_scale:.12e}")
    print(f"scale_mode    = {readout_scale_mode}")
    print(f"dp            = {dp:.12g}")

    if fp_meta["profile"] == "erf":
        print("\n[erf recovery bounds]")
        print(f"P_READOUT_RULE  = {P_READOUT_RULE}")
        print(f"Lambda_plus_T   = {Lambda_plus_T:.12g}")
        print(f"Lambda_minus_T  = {Lambda_minus_T:.12g}")
        print(f"p_min           = {p_min_recovery:.12g}")
        print(f"p_max           = {p_max_recovery:.12g}")
        print(f"p_target        = {p_target_recovery:.12g}")
    
    # -----------------------------------------------------------------
    # Quantum circuit setup
    # -----------------------------------------------------------------
    
    # Start timer
    t_build0 = time.perf_counter()
    
    # Setup main registers
    rx      = QuantumRegister(nx, name='X')
    rv      = QuantumRegister(nv, name='V')
    ra      = QuantumRegister(na, name='P')
    
    # Setup sub registers and padding workspaces
    #nE      = 3*nx + 1
    nE      = compute_nE_centered(nx, alpha_fp, beta_fp)
    nW_core = 3*nx + 1
    n_wpad  = max(0, nE - nW_core)
    
    xcopy   = QuantumRegister(nx, "xcopy")
    tmp     = QuantumRegister(nx, "tmp")
    sq      = QuantumRegister(2*nx, "sq")
    cube    = QuantumRegister(4*nx, "cube")
    E       = QuantumRegister(nE, "E")
    w_msb   = QuantumRegister(1, "w_msb")
    cin     = QuantumRegister(1, "cin")
    cout    = QuantumRegister(1, "cout") 
    
    # extra centered E ancillas
    c        = QuantumRegister(nx,   "c")
    cabs     = QuantumRegister(nx,   "cabs")
    carry_nx = QuantumRegister(nx,   "carry_nx")
    carry_3nx= QuantumRegister(3*nx, "carry_3nx")
    
    # -----------------------------------------------------------------
    # Optional W padding to reach nE
    # -----------------------------------------------------------------
    
    wpad = QuantumRegister(n_wpad, "wpad") if n_wpad > 0 else None
    
    # Full circuit (add new regs)
    # if wpad is None:
    #     qc = QuantumCircuit(
    #         rx, rv, ra, c, cabs, xcopy, tmp, sq, cube, 
    #         E, w_msb, cin, cout, carry_nx, carry_3nx
    #     )
    # else:
    #     qc = QuantumCircuit(
    #         rx, rv, ra, c, cabs, xcopy, tmp, sq, cube, 
    #         E, w_msb, wpad, cin, cout, carry_nx, carry_3nx
    #     )
    
    # rE must match the subcircuit’s "everything except x"
    # rE = (
    #     c[:] + cabs[:] + xcopy[:] + tmp[:] + sq[:] + cube[:] +
    #     E[:] + w_msb[:] + ([] if wpad is None else wpad[:]) +
    #     cin[:] + cout[:] + carry_nx[:] + carry_3nx[:]
    # )
    
    
    if wpad is None:
        qc = QuantumCircuit(
            rx, rv, ra, c, cabs, xcopy, tmp, sq, cube,
            E, w_msb, cin, cout, carry_nx, carry_3nx
        )
    else:
        qc = QuantumCircuit(
            rx, rv, ra, c, cabs, xcopy, tmp, sq, cube,
            E, w_msb, cin, cout, carry_nx, carry_3nx, wpad
        )
    
    # rE must match compute_E_poly_centered qubit order:
    # c, cabs, xcopy, tmp, sq, cube, E, w_msb, cin, cout, carry_nx, carry_3nx, wpad
    rE = (
        c[:] + cabs[:] + xcopy[:] + tmp[:] + sq[:] + cube[:] +
        E[:] + w_msb[:] + cin[:] + cout[:] + carry_nx[:] + carry_3nx[:] +
        ([] if wpad is None else wpad[:])
    )

    # -----------------------------------------------------------------
    # Prepare |P>|V>|X> amplitudes
    # -----------------------------------------------------------------

    # Prepare |V>|X> amplitudes    
    amps_xv = np.zeros(Nx * Nv, dtype=complex)
    for jx in range(Nx):
        for jv in range(Nv):
            idx = jx + Nx * jv   # |x_jx> |v_jv>
            amps_xv[idx] = f0[jx, jv]
    amps_xv /= np.linalg.norm(amps_xv)
    qc.append(Initialize(amps_xv), rx[:] + rv[:])
    
    # Prepare |P> amplitudes 
    amps_p = fp.astype(complex)
    amps_p /= np.linalg.norm(amps_p)
    qc.append(Initialize(amps_p), ra[:])
    
    # -----------------------------------------------------------------
    # Prepare reusable circuit blocks (Ux / Uv)
    # -----------------------------------------------------------------
    # These blocks implement the Hu/Sato-style discretized transport operators:
    # - Ux blocks depend on the v-grid (free streaming: v ∂_x)
    # - Uv blocks depend on the E(x) register (kick: a(x) ∂_v)
    #
    # Notes:
    # - R controls the Fourier truncation parameter in the Hu/Sato symbol.
    # - S is the fixed-point scale used by the centered E oracle (a = E_fp / S).
       
    def make_sato_hu_blocks(
        *,
        nx: int, nv: int, na: int, nE: int,
        v_grid,
        dt: float,
        dx: float,
        dv: float,
        R: float,
        S: int,
    ):
        """
        Build reusable circuit blocks for Strang splitting:
          - Ux blocks (half/full): cvH1_dt2, icvH1_dt2, vH2_dt2, cvH1, icvH1, vH2
          - Uv blocks (full only): cEH1, icEH1, eH2
    
        Requires these to be imported from schro_1d1v_blocks:
          theta_from_v_grid_linear, theta_bits_from_twos_complement
          cvV1, inverse_cvV1, vV2, cEV1, inverse_cEV1, eV2
        """
    
        # -------------------------
        # (1) X-step (free-streaming): depends on v-grid
        # -------------------------
        gamma_x = 1.0 / (2.0 * dx)
        theta_x = 2.0 * gamma_x * dt          # = dt/dx
        theta1_x = theta_x / R
        theta2_x = theta_x
    
        # half-step angles
        th_off1_h, th_bits1_h = theta_from_v_grid_linear(v_grid, theta1_x / 2.0, nv)
        th_off2_h, th_bits2_h = theta_from_v_grid_linear(v_grid, theta2_x / 2.0, nv)
    
        cvH1_dt2  = cvV1(th_off1_h, th_bits1_h, th_off1_h, th_bits1_h, nx, nv)
        icvH1_dt2 = inverse_cvV1(-th_off1_h, -th_bits1_h, -th_off1_h, -th_bits1_h, nx, nv)
        vH2_dt2   = vV2(th_off2_h, th_bits2_h, nx, nv)
    
        # full-step angles
        th_off1, th_bits1 = theta_from_v_grid_linear(v_grid, theta1_x, nv)
        th_off2, th_bits2 = theta_from_v_grid_linear(v_grid, theta2_x, nv)
    
        cvH1  = cvV1(th_off1, th_bits1, th_off1, th_bits1, nx, nv)
        icvH1 = inverse_cvV1(-th_off1, -th_bits1, -th_off1, -th_bits1, nx, nv)
        vH2   = vV2(th_off2, th_bits2, nx, nv)
    
        # -------------------------
        # (2) V-step (kick): depends on E(x) in fixed-point
        # -------------------------
        gamma_v = 1.0 / (2.0 * dv)
        theta_v = 2.0 * gamma_v * dt          # = dt/dv
        theta1_v = theta_v / (R * S)
        theta2_v = theta_v / S
    
        ph_off1, ph_bits1 = theta_bits_from_twos_complement(theta1_v, nE)
        ph_off2, ph_bits2 = theta_bits_from_twos_complement(theta2_v, nE)
    
        cEH1  = cEV1(ph_off1, ph_bits1, ph_off1, ph_bits1, nv, nE)
        icEH1 = inverse_cEV1(-ph_off1, -ph_bits1, -ph_off1, -ph_bits1, nv, nE)
        eH2   = eV2(ph_off2, ph_bits2, nv, nE)
    
        return cvH1_dt2, icvH1_dt2, vH2_dt2, cvH1, icvH1, vH2, cEH1, icEH1, eH2

    cvH1_dt2, icvH1_dt2, vH2_dt2, cvH1, icvH1, vH2, cEH1, icEH1, eH2 = make_sato_hu_blocks(
        nx=nx, nv=nv, na=na, nE=nE,
        v_grid=v_grid,
        dt=dt, dx=dx, dv=dv,
        R=R, S=S,
    )
    
    # Centered-grid fixed-point oracle: computes E_fp(j) into E (two's complement),
    # where c = j - 2^(nx-1) and E_fp = alpha_fp*c + beta_fp*c^3.
    E1 = compute_E_poly_centered(nx, alpha_fp, beta_fp, S)
    
    # -----------------------------------------------------------------
    # Transform P to Fourier basis -> Strang splitting -> return to p-grid
    # ----------------------------------------------------------------- 
    
    # Paper/manuscript convention: enter auxiliary Fourier basis.
    # Qiskit's IQFT implements the sign convention used here.
    qc.append(QFTGate(na).inverse(), ra[:])
    
    
    # -----------------------------------------------------------------
    # Build the full Schrödingerization circuit (Strang splitting)
    # -----------------------------------------------------------------
    # We evolve f(x,v) under:
    #   ∂t f + v ∂x f + a(x) ∂v f = 0
    # using Strang splitting over Nt steps:
    #   Ux(dt/2) -> [Uv(dt) -> Ux(dt)]^(Nt-1) -> Uv(dt) -> Ux(dt/2)
    #
    # Implementation notes:
    # - Ux uses blocks derived from the v-grid (free streaming).
    # - Uv uses the centered-grid fixed-point oracle E_fp(j) (kick), computed/uncomputed
    #   once per Uv step so the workspace returns to |0...0>.
    # - X holds the index j; the oracle internally forms c = j - 2^(nx-1) and writes
    #   E_fp = alpha_fp*c + beta_fp*c^3 in two's-complement into the E register.

    def _append_Ux_step(qc, *, half: bool):
        """Append one Ux step: either half-step (dt/2 blocks) or full-step (dt blocks)."""
        if half:
            cvH1_blk, icvH1_blk, vH2_blk = cvH1_dt2, icvH1_dt2, vH2_dt2
        else:
            cvH1_blk, icvH1_blk, vH2_blk = cvH1, icvH1, vH2
    
        # --- H1 ladder (controlled by P bits), then H2 ---
        for j in range(na - 1):
            qc.append(cvH1_blk.repeat(2**j), [ra[j]] + rx[:] + rv[:])
    
        qc.append(icvH1_blk.repeat(2**(na - 1)), [ra[na - 1]] + rx[:] + rv[:])
        qc.append(vH2_blk, rx[:] + rv[:])
    
    
    def _append_Uv_step(qc):
        """Append one Uv(dt) step, including compute/uncompute of E(x)."""
    
        # Compute E(x) into the E register (and workspace in rE)
        qc.append(E1, rx[:] + rE[:])
    
        # --- H1 ladder (controlled by P bits), then H2 ---
        for j in range(na - 1):
            qc.append(cEH1.repeat(2**j), [ra[j]] + rv[:] + E[:])
    
        qc.append(icEH1.repeat(2**(na - 1)), [ra[na - 1]] + rv[:] + E[:])
        qc.append(eH2, rv[:] + E[:])
    
        # Uncompute E(x) to return workspace to |0...0>
        qc.append(E1.inverse(), rx[:] + rE[:])
        
   
    # -----------------------
    # Assemble Strang splitting
    # -----------------------
    # Start with Ux(dt/2)
    _append_Ux_step(qc, half=True)
    
    # Main loop: [Uv(dt) -> Ux(dt)]*(N_t-1)
    for _ in range(max(0, Nt - 1)):
        _append_Uv_step(qc)
        _append_Ux_step(qc, half=False)
    
    # Final: Uv(dt) -> Ux(dt/2)
    _append_Uv_step(qc)
    _append_Ux_step(qc, half=True)                                  

    # Return from Fourier basis to p-grid
    # Return from auxiliary Fourier basis to the p-grid.
    qc.append(QFTGate(na), ra[:])
    
    # -----------------------------------------------------------------
    # Post-processing
    # ----------------------------------------------------------------- 
    
    # Stop timer
    t_build1 = time.perf_counter()
    print(f"\n[quantum wall] build time: {t_build1 - t_build0:.2f} s")
    
    # Plot circuit
    fig1 = None
    if DO_DRAW_CIRCUIT:
        fig1 = qc.draw(output="mpl", fold=250)
        display(fig1)
        save_fig(fig1, OUT["figures"], "fig1_circuit")
        plt.close(fig1)
    
    # FT resource counting is intentionally a separate job.  The old script
    # decomposed the same circuit twice by calling both report_ft_resources()
    # and ft_resource_row().  One row calculation is sufficient here.
    resource_row = None
    if DO_RESOURCE_REPORT:
        resource_row = ft_resource_row(
            qc,
            decompose_reps=5,
            rot_eps=1e-10,
        )
        save_json(
            OUT["resources"] / "resource_row.json",
            resource_row,
        )
        print("\n[FT resources]")
        for key, value in resource_row.items():
            print(f"  {key:18s}: {value}")

    if DO_READOUT:
        from readout_utils_mps import (
            remove_idle_wires,
            obtain_u_quant_endian_safe,
            align_and_extract_p_slice,
        )
        from compare_utils import compare_quantum_classical, compare_slices
        
        if DO_STATEVECTOR:
            qc_small = remove_idle_wires(qc)
            if qc_small.num_qubits > 28:
                raise MemoryError(
                    "Full statevector readout is disabled above 28 active "
                    f"qubits (this circuit has {qc_small.num_qubits})."
                )
            psi_ref = Statevector(qc_small).data
            u_quant, info = align_and_extract_p_slice(
                psi_ref, nx, nv, na, readout_scale, p_readout_idx=p_readout_idx
            )
        else:
            u_quant, info = obtain_u_quant_endian_safe(
                qc, rx, rv, ra,
                nx, nv, na,
                readout_scale,
                reps=READOUT_REPS,
                p_readout_idx=p_readout_idx,
                simulator_options=MPS_SIMULATOR_OPTIONS,
                mps_log_path=(
                    OUT["logs"] / "mps_log.txt"
                    if MPS_LOG_DATA else None
                ),
                cpu_log_path=CPU_LOG_PATH,
                # Production jobs do not reuse qc after readout.  With
                # DO_EXPORT=False the helper may clear qc's hierarchical
                # instructions after qc_run has been created, reducing memory
                # held during the multi-hour Aer simulation.
                copy_circuit=DO_EXPORT,
            )
            
        # Classical solution fc
        a_x = a_x_from_centered_fixed_point(nx, alpha_fp, beta_fp, S)
        f_class = strang_integrate_classical_hu_fft_scaled(f0, v_grid, a_x, dt, Nt, Lx, Lv, R)
        
        # Compare uQ and fc 
        rel_l2, mass_q, mass_c, uq, fc = compare_quantum_classical(u_quant, f_class)
        uq = u_quant / np.linalg.norm(u_quant)
        fc = f_class / np.linalg.norm(f_class)
        print("Relative L2:", rel_l2)
        print("L1 mass quantum:", mass_q)
        print("L1 mass classical:", mass_c)
        
        # Normalized outputs are always saved numerically.  Final figures are
        # optional and are created only after the expensive MPS run completes.
        fC = f_class / np.sum(f_class)
        fQ = u_quant / np.sum(u_quant)

        fig2 = None
        fig3 = None
        if DO_OUTPUT_PLOTS:
            fig2 = plot_f(fQ, x_grid, v_grid, name=rf"$f_Q (t = {Nt*dt})$")
            fig3 = compare_slices(fQ, fC, f0, x_grid, v_grid, Nt, dt)
            save_fig(fig2, OUT["figures"], "fig2_quantum_fQ")
            save_fig(fig3, OUT["figures"], "fig3_quantum_classical_slices")
            plt.close(fig2)
            plt.close(fig3)

        np.savez(
            OUT["data"] / "run_solution_centered.npz",
            fQ=fQ,
            fC=fC,
            f0=f0,
            u_quant=u_quant,
            f_class=f_class,
            x_grid=x_grid,
            v_grid=v_grid,
            dx=dx,
            dv=dv,
            nx=nx,
            nv=nv,
            na=na,
            Nx=Nx,
            Nv=Nv,
            Nt=Nt,
            dt=dt,
            v0=v0,
            T=Nt * dt,
            R=R,
            rel_l2=rel_l2,
            mass_q=mass_q,
            mass_c=mass_c,
            P_PROFILE=P_PROFILE,
            P_FOURIER_CONVENTION="paper",
            P_READOUT_RULE=P_READOUT_RULE,
            p_readout_idx=p_readout_idx,
            p_readout=p_readout,
            fp_readout=fp_readout,
            readout_scale=readout_scale,
            readout_scale_mode=readout_scale_mode,
        )
        
        readout_metrics = {
            "RUN_ID": RUN_ID,
            "nx": nx,
            "nv": nv,
            "na": na,
            "Nx": Nx,
            "Nv": Nv,
            "Nt": Nt,
            "dt": dt,
            "v0": v0,
            "T": Nt * dt,
            "R": R,
            "DO_STATEVECTOR": DO_STATEVECTOR,
            "READOUT_REPS": READOUT_REPS,
            "MPS_SWAP_DIRECTION": MPS_SWAP_DIRECTION,
            "MPS_THREADS": MPS_THREADS,
            "MPS_USE_LAPACK": MPS_USE_LAPACK,
            "MPS_LOG_DATA": MPS_LOG_DATA,
            "MPS_PARALLEL_THRESHOLD": MPS_PARALLEL_THRESHOLD,
            "MPS_MAX_BOND_DIMENSION": MPS_MAX_BOND_DIMENSION,
            "MPS_TRUNCATION_THRESHOLD": MPS_TRUNCATION_THRESHOLD,
            "P_PROFILE": P_PROFILE,
            "P_FOURIER_CONVENTION": "paper",
            "P_READOUT_RULE": P_READOUT_RULE,
            "P_READOUT_TARGET": P_READOUT_TARGET,
            "P_READOUT_GUARD": P_READOUT_GUARD,
            "Lambda_plus_T": Lambda_plus_T,
            "Lambda_minus_T": Lambda_minus_T,
            "p_min_recovery": p_min_recovery,
            "p_max_recovery": p_max_recovery,
            "p_target_recovery": p_target_recovery,
            "fp_a": fp_meta.get("a", np.nan),
            "fp_p_star": fp_meta.get("p_star", np.nan),
            "p_center_idx": p_center_idx,
            "p_readout_idx": p_readout_idx,
            "p_readout": p_readout,
            "fp_readout": fp_readout,
            "readout_scale": readout_scale,
            "readout_scale_mode": readout_scale_mode,
            "dp": dp,
            "transition_width": (1.0 / fp_meta["a"]) if fp_meta["profile"] == "erf" else np.nan,
            "rel_l2": rel_l2,
            "mass_q": mass_q,
            "mass_c": mass_c,
        }
        
        if info is not None:
            readout_metrics["selected_slice_phase"] = info.get("slice_phase", np.nan)
            readout_metrics["selected_imag_ratio"] = info.get("imag_ratio", np.nan)

        save_json(
            OUT["data"] / "readout_metrics.json",
            readout_metrics,
        )
        
        if info is not None:
            save_json(
                OUT["data"] / "readout_info.json",
                info,
            )
    
    
    if DO_EXPORT:
        figs = {"fig0": locals().get("fig0"), "fig1": locals().get("fig1"),
                "fig2": locals().get("fig2"), "fig3": locals().get("fig3")}
    
        export_centered_run(
            nx=nx, nv=nv, na=na,
            Nx=Nx, Nv=Nv,
            Nt=Nt, dt=dt, dt_cfl=dt_cfl, nu=nu,
            Lx=Lx, Lv=Lv, dx=dx, dv=dv,
            a0=a0, kappa=kappa, a_max_bound=a_max_bound,
            alpha_fp=alpha_fp, beta_fp=beta_fp, alpha=alpha, beta=beta, S=S,
            R=R, R_suggest=R_suggest,
            nE=nE,
            qc=qc,
            ft_resource_row_fn=ft_resource_row,
            info=locals().get("info"),
            rel_l2=locals().get("rel_l2"),
            u_quant=locals().get("u_quant"),
            f_class=locals().get("f_class"),
            t_build0=locals().get("t_build0"),
            t_build1=locals().get("t_build1"),
            figures=figs,
            results_root=OUT["logs"],
            clear_log=True,     # match your current behavior
            run_id="",          # auto timestamp (or set "" to disable suffix)
        )
        
    # ------------------------------------------------------------
    # Save lightweight run parameters for plotting scripts
    # This does not require DO_READOUT.
    # ------------------------------------------------------------
    run_params = {
        "RUN_ID": RUN_ID,
        "nx": nx,
        "nv": nv,
        "na": na,
        "Nx": Nx,
        "Nv": Nv,
        "Lx": Lx,
        "Lv": Lv,
        "dx": dx,
        "dv": dv,
        "dt": dt,
        "Nt": Nt,
        "T": Nt * dt,
        "R": R,
        "R_suggest": R_suggest,
        "a0": a0,
        "kappa": kappa,
        "s": s,
        "S": S,
        "alpha": alpha,
        "beta": beta,
        "alpha_fp": alpha_fp,
        "beta_fp": beta_fp,
        "beta_0": beta_0,
        "sigma": sigma,
        "v0": v0,
        "nE": nE,
        "dt_cfl": dt_cfl,
        "nu": nu,
        "P_PROFILE": P_PROFILE,
        "P_EPS": P_EPS,
        "P_FOURIER_CONVENTION": "paper",
        "P_READOUT_RULE": P_READOUT_RULE,
        "P_READOUT_TARGET": P_READOUT_TARGET,
        "P_READOUT_GUARD": P_READOUT_GUARD,
        "DO_READOUT": DO_READOUT,
        "DO_RESOURCE_REPORT": DO_RESOURCE_REPORT,
        "DO_INITIAL_PLOTS": DO_INITIAL_PLOTS,
        "DO_OUTPUT_PLOTS": DO_OUTPUT_PLOTS,
        "MPS_THREADS": MPS_THREADS,
        "MPS_USE_LAPACK": MPS_USE_LAPACK,
        "MPS_LOG_DATA": MPS_LOG_DATA,
        "MPS_SWAP_DIRECTION": MPS_SWAP_DIRECTION,
        "MPS_MAX_BOND_DIMENSION": MPS_MAX_BOND_DIMENSION,
        "MPS_TRUNCATION_THRESHOLD": MPS_TRUNCATION_THRESHOLD,
        "Lambda_plus_T": Lambda_plus_T,
        "Lambda_minus_T": Lambda_minus_T,
        "p_min_recovery": p_min_recovery,
        "p_max_recovery": p_max_recovery,
        "p_target_recovery": p_target_recovery,
        "p_center_idx": p_center_idx,
        "p_readout_idx": p_readout_idx,
        "p_readout": p_readout,
        "fp_readout": fp_readout,
        "readout_scale": readout_scale,
        "readout_scale_mode": readout_scale_mode,
        "fp_a": fp_meta.get("a", np.nan),
        "fp_p_star": fp_meta.get("p_star", np.nan),
    }

    np.savez(
        OUT["data"] / "run_params_centered.npz",
        **run_params,
        f0=f0,
        x_grid=x_grid,
        v_grid=v_grid,
        p_grid=p_grid,
        fp=fp,
    )

    save_json(
        OUT["data"] / "run_params_centered.json",
        run_params,
    )

        
