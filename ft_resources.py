# ft_resources.py
# ============================================================
# Fault-tolerant-oriented resource reporting for circuits that
# compile to {CNOT + arbitrary single-qubit rotations}.
#
# In this setting, explicit Toffoli/CCX counts are often zero
# because the circuit is already expressed using u/p/rz (+cx).
# The dominant FT cost is then MAGIC-STATE consumption needed
# to synthesize these arbitrary rotations into Clifford+T.
#
# Recommended reported resources:
#   1) Peak logical qubits:       nq
#   2) CNOT count (and depth):    N_CX, depth_CX (optional)
#   3) Rotation-like count:       N_rot = #(u + p + rz + cp)
#   4) Estimated T-count at eps:  Tcount_est(eps) ≈ N_rot * T_per_rot(eps) + T_explicit
#   5) Optional proxy volume:     nq * Tcount_est  (VERY conservative)
#
# Notes:
# - eps is the target approximation error PER rotation (planning proxy).
# - T_per_rot is a rule-of-thumb ~ O(log2(1/eps)).
# - This script prints both a "decomposed view" and an optional
#   "transpiled view" in a chosen basis for consistency across runs.
# ============================================================

from __future__ import annotations
from collections import OrderedDict
import math
from qiskit import transpile


# ---------- helpers ----------

def _sorted_ops(ops_dict) -> OrderedDict:
    return OrderedDict(sorted(ops_dict.items(), key=lambda kv: (-kv[1], kv[0])))


def count_ops(qc) -> OrderedDict:
    """Stable sorted count_ops()."""
    return _sorted_ops(qc.count_ops())


def estimate_T_per_rotation(eps: float, *, k: float = 3.0) -> int:
    """
    Rough Clifford+T synthesis cost proxy for one arbitrary 1-qubit rotation
    to precision eps.

    Rule-of-thumb:
        T_per_rot ≈ ceil(k * log2(1/eps))

    Default k=3 is conservative-ish for planning.
    You MUST state eps (and k) in any write-up.
    """
    if not (0 < eps < 1):
        raise ValueError("eps must be in (0,1)")
    return max(1, int(math.ceil(k * math.log2(1.0 / eps))))


def extract_rotation_cost_counts(ops: dict) -> dict:
    """
    Extract counts relevant for a rotation-synthesis FT cost model.
    """
    cx  = int(ops.get("cx", 0))

    # rotation-like ops (typically require synthesis => magic states)
    n_u  = int(ops.get("u", 0))
    n_p  = int(ops.get("p", 0))
    n_rz = int(ops.get("rz", 0))
    n_cp = int(ops.get("cp", 0))
    rot_like = n_u + n_p + n_rz + n_cp

    # If you already have explicit Clifford+T decomposition somewhere
    n_t = int(ops.get("t", 0)) + int(ops.get("tdg", 0))

    return {
        "cx": cx,
        "u": n_u,
        "p": n_p,
        "rz": n_rz,
        "cp": n_cp,
        "rot_like": rot_like,
        "t_explicit": n_t,
    }


def cx_depth_upper_bound(qc) -> int:
    """
    Upper bound on CNOT depth via greedy layering.
    Useful only as a rough proxy (ignores commutation optimizations).
    """
    last = {q: 0 for q in qc.qubits}
    d = 0
    for inst, qargs, _ in qc.data:
        if inst.name == "cx":
            layer = 1 + max(last[q] for q in qargs)
            for q in qargs:
                last[q] = layer
            d = max(d, layer)
    return d


# ---------- main reporter ----------

def report_ft_resources(
    qc,
    *,
    label: str = "qc",
    decompose_reps: int = 5,
    do_transpile_after_decompose: bool = False,
    transpile_basis: tuple[str, ...] = ("u", "cx"),
    opt_level: int = 0,
    rot_eps: float = 1e-10,
    rot_k: float = 3.0,
    include_cx_depth_ub: bool = False,
    print_top_ops: int = 12,
):
    """
    Print FT-oriented resource indicators appropriate for u/cx-style circuits.

    Parameters:
      decompose_reps:
        Decompose composite instructions before counting. Use the same reps you
        need for Aer acceptance if you want consistent numbers.
      rot_eps:
        Target approximation precision per rotation (planning proxy).
      rot_k:
        Multiplier in T_per_rot ≈ ceil(k log2(1/eps)). Default k=3.
      include_cx_depth_ub:
        Also print a greedy upper bound on CNOT depth (extra pass over qc.data).
    """
    print("\n" + "=" * 80)
    print(f"[FT resources] {label}")

    nq = qc.num_qubits
    print(f"  logical_qubits (peak) : {nq}")
    print(f"  depth (high-level)    : {qc.depth()}")
    print(f"  size  (high-level)    : {qc.size()}")

    # Decomposed view (more faithful than high-level)
    qd = qc.decompose(reps=decompose_reps)
    ops_d = count_ops(qd)
    cnt_d = extract_rotation_cost_counts(ops_d)

    T_per = estimate_T_per_rotation(rot_eps, k=rot_k)
    Tcount_est = T_per * cnt_d["rot_like"] + cnt_d["t_explicit"]

    print(f"\n  After decompose(reps={decompose_reps}):")
    print(f"    depth               : {qd.depth()}")
    print(f"    size                : {qd.size()}")
    print(f"    cx count            : {cnt_d['cx']}")
    if include_cx_depth_ub:
        print(f"    cx depth (UB)        : {cx_depth_upper_bound(qd)}")
    print(f"    rotation-like count : {cnt_d['rot_like']} (u={cnt_d['u']}, p={cnt_d['p']}, rz={cnt_d['rz']}, cp={cnt_d['cp']})")
    print(f"    explicit T/TDG      : {cnt_d['t_explicit']}")

    print("\n  Magic-state (T) cost proxy from rotation synthesis:")
    print(f"    eps per rotation    : {rot_eps:g}")
    print(f"    T_per_rot estimate  : {T_per}  (ceil({rot_k}*log2(1/eps)))")
    print(f"    estimated Tcount    : {Tcount_est}  (≈ {T_per}*{cnt_d['rot_like']} + {cnt_d['t_explicit']})")

    # VERY conservative "volume" proxy: assume no T parallelism
    print("\n  Space–time proxy (VERY conservative upper bound):")
    print(f"    nq * Tcount_est     : {nq} * {Tcount_est} = {nq*Tcount_est}")

    # Optional: transpile after decompose for a consistent basis view
    if do_transpile_after_decompose:
        qt = transpile(qd, basis_gates=list(transpile_basis), optimization_level=opt_level)
        ops_t = count_ops(qt)
        cnt_t = extract_rotation_cost_counts(ops_t)

        Tcount_est_t = T_per * cnt_t["rot_like"] + cnt_t["t_explicit"]

        print(f"\n  After transpile(basis={transpile_basis}, opt_level={opt_level}):")
        print(f"    depth               : {qt.depth()}")
        print(f"    size                : {qt.size()}")
        print(f"    cx count            : {cnt_t['cx']}")
        if include_cx_depth_ub:
            print(f"    cx depth (UB)        : {cx_depth_upper_bound(qt)}")
        print(f"    rotation-like count : {cnt_t['rot_like']} (u={cnt_t['u']}, p={cnt_t['p']}, rz={cnt_t['rz']}, cp={cnt_t['cp']})")
        print(f"    explicit T/TDG      : {cnt_t['t_explicit']}")
        print(f"    estimated Tcount    : {Tcount_est_t}")

        print(f"    top ops             : {list(ops_t.items())[:print_top_ops]}")
        
    # Headline one-liner for logs/sweeps
    print("\n  Summary:")
    print(f"    nq={nq}, cx={cnt_d['cx']}, rot_like={cnt_d['rot_like']}, Tcount_est(eps={rot_eps:g})≈{Tcount_est}")
    print("=" * 80)


# ---------- optional: machine-readable output for sweeps ----------

def ft_resource_row(
    qc,
    *,
    decompose_reps: int = 5,
    rot_eps: float = 1e-10,
    rot_k: float = 3.0,
):
    """
    Returns a dict of key FT-relevant metrics for building tables/CSV.
    """
    nq = qc.num_qubits
    qd = qc.decompose(reps=decompose_reps)
    ops = qd.count_ops()
    cnt = extract_rotation_cost_counts(ops)
    T_per = estimate_T_per_rotation(rot_eps, k=rot_k)
    Tcount_est = T_per * cnt["rot_like"] + cnt["t_explicit"]

    return {
        "nq": nq,
        "depth_hi": qc.depth(),
        "size_hi": qc.size(),
        "depth_dec": qd.depth(),
        "size_dec": qd.size(),
        "cx": cnt["cx"],
        "rot_like": cnt["rot_like"],
        "u": cnt["u"],
        "p": cnt["p"],
        "rz": cnt["rz"],
        "cp": cnt["cp"],
        "t_explicit": cnt["t_explicit"],
        "rot_eps": rot_eps,
        "rot_k": rot_k,
        "T_per_rot_est": T_per,
        "Tcount_est": Tcount_est,
    }
