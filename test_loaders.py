"""Corrected structured loaders for the auxiliary p-register.

The constructions implement Appendix A of the Vlasov manuscript:

* exact positive-real binary-tree loading;
* exact loading on the nonnegative half of the p-grid;
* geometric half-support loading with r = exp(-Delta p);
* zero-controlled local replacement of the first P = 2**b amplitudes.

Qiskit endianness
-----------------
Qiskit statevector entry ``state[j]`` corresponds to the computational basis
integer ``j`` with qubit 0 as the least-significant bit.  This module follows
that convention directly:

* q[0] is the least-significant within-half index bit;
* q[n_a - 2] is the most-significant within-half index bit;
* q[n_a - 1] selects the nonnegative half of the p-grid.

No statevector reversal or bit-reversal postprocessing is used.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Iterable

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import Initialize
from qiskit.quantum_info import Statevector


_DEFAULT_BASIS = ("u", "cx")


def _power_of_two_qubits(length: int) -> int:
    length = int(length)
    if length < 2 or (length & (length - 1)):
        raise ValueError("Amplitude-vector length must be a power of two >= 2.")
    return length.bit_length() - 1


def _normalized_vector(
    values: Iterable[complex],
    *,
    name: str,
    require_real_nonnegative: bool = False,
    atol: float = 1e-14,
) -> np.ndarray:
    vec = np.asarray(values, dtype=complex).reshape(-1).copy()
    if vec.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} contains non-finite entries.")

    if require_real_nonnegative:
        if np.max(np.abs(vec.imag), initial=0.0) > atol:
            raise ValueError(f"{name} must be real for this loader.")
        real = vec.real
        if np.min(real, initial=0.0) < -atol:
            raise ValueError(f"{name} must be nonnegative.")
        real[np.abs(real) < atol * 0.0] = 0.0  # intentionally keep tiny positives
        vec = real.astype(complex)

    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"{name} has zero or non-finite norm.")
    return vec / norm


# ---------------------------------------------------------------------------
# Grid and target profile
# ---------------------------------------------------------------------------


def make_p_grid(na: int, R: float) -> tuple[np.ndarray, float]:
    if not isinstance(na, (int, np.integer)) or int(na) < 1:
        raise ValueError("na must be a positive integer.")
    na = int(na)
    R = float(R)
    if not np.isfinite(R) or R <= 0.0:
        raise ValueError("R must be finite and positive.")

    N = 1 << na
    dp = 2.0 * math.pi * R / N
    p_grid = -math.pi * R + dp * np.arange(N, dtype=float)
    return p_grid, dp


def erf_profile_on_p(
    p: np.ndarray,
    eps: float = 1e-6,
    a: float | None = None,
) -> tuple[np.ndarray, float]:
    eps = float(eps)
    if not np.isfinite(eps) or not (0.0 < eps < 1.0):
        raise ValueError("eps must lie strictly between 0 and 1.")

    if a is None:
        a_used = 2.0 * math.sqrt(math.log(1.0 / eps))
    else:
        a_used = float(a)
        if not np.isfinite(a_used) or a_used <= 0.0:
            raise ValueError("a must be finite and positive.")

    p = np.asarray(p, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("p contains non-finite entries.")

    # Keep the manuscript's sampled formula exactly.  Using math.erf also
    # reproduces the rounded Appendix-A benchmark values used in the paper.
    switch = np.fromiter(
        (0.5 * (1.0 + math.erf(a_used * float(x))) for x in p),
        dtype=float,
        count=p.size,
    )
    profile = np.exp(-p) * switch
    profile = np.maximum(profile, 0.0)
    return profile, a_used


def normalized_target_erf(
    na: int,
    R: float,
    eps: float = 1e-6,
    a: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    p_grid, _ = make_p_grid(na=na, R=R)
    profile, a_used = erf_profile_on_p(p_grid, eps=eps, a=a)
    target = _normalized_vector(profile, name="error-function profile")
    return p_grid, target, a_used


# ---------------------------------------------------------------------------
# Exact positive-real tree loader, written directly in Qiskit little endian
# ---------------------------------------------------------------------------


def _append_prefix_controlled_ry(
    qc: QuantumCircuit,
    *,
    theta: float,
    controls: list[int],
    control_values: tuple[int, ...],
    target: int,
    angle_atol: float,
) -> None:
    if len(controls) != len(control_values):
        raise ValueError("controls/control_values length mismatch.")
    if abs(theta) <= angle_atol:
        return

    open_controls = [q for q, bit in zip(controls, control_values) if bit == 0]
    for q in open_controls:
        qc.x(q)

    if not controls:
        qc.ry(theta, target)
    elif len(controls) == 1:
        qc.cry(theta, controls[0], target)
    else:
        qc.mcry(
            theta,
            controls,
            target,
            q_ancillae=None,
            mode="noancilla",
        )

    for q in reversed(open_controls):
        qc.x(q)


def build_positive_real_amplitude_loader(
    amplitudes: Iterable[complex],
    *,
    label: str = "TREE",
    probability_atol: float = 0.0,
    angle_atol: float = 0.0,
) -> QuantumCircuit:
    """Prepare nonnegative real amplitudes with a binary rotation tree.

    At depth d, the circuit writes the bit q[n-1-d].  Hence the first split
    acts on the most-significant qubit and the final split acts on q[0], which
    is exactly Qiskit's little-endian integer convention.
    """
    state = _normalized_vector(
        amplitudes,
        name="amplitudes",
        require_real_nonnegative=True,
    ).real
    n = _power_of_two_qubits(state.size)
    probs = state * state
    qc = QuantumCircuit(n, name=label)

    def recurse(
        start: int,
        size: int,
        prefix: tuple[int, ...],
    ) -> None:
        if size <= 1:
            return

        depth = len(prefix)
        target = n - 1 - depth
        half = size // 2

        left = float(np.sum(probs[start : start + half]))
        right = float(np.sum(probs[start + half : start + size]))
        total = left + right

        if total <= probability_atol:
            theta = 0.0
        else:
            # atan2 is stable when one side is very small.
            theta = 2.0 * math.atan2(math.sqrt(max(right, 0.0)),
                                     math.sqrt(max(left, 0.0)))

        # Prefix bits are ordered MSB -> current bit.  Map them directly to
        # q[n-1], q[n-2], ... in the same order.
        controls = list(range(n - 1, target, -1))
        _append_prefix_controlled_ry(
            qc,
            theta=theta,
            controls=controls,
            control_values=prefix,
            target=target,
            angle_atol=angle_atol,
        )

        recurse(start, half, prefix + (0,))
        recurse(start + half, half, prefix + (1,))

    recurse(0, state.size, ())
    return qc


# ---------------------------------------------------------------------------
# Exact and structured loaders
# ---------------------------------------------------------------------------


def build_initialize_loader(target: Iterable[complex]) -> QuantumCircuit:
    state = _normalized_vector(target, name="target")
    n = _power_of_two_qubits(state.size)
    qc = QuantumCircuit(n, name="initialize_loader")
    qc.append(Initialize(state), qc.qubits)
    return qc


def build_tree_loader(target: Iterable[complex]) -> QuantumCircuit:
    state = _normalized_vector(
        target,
        name="target",
        require_real_nonnegative=True,
    )
    return build_positive_real_amplitude_loader(state, label="exact_tree")


def build_half_support_loader(target: Iterable[complex]) -> QuantumCircuit:
    state = _normalized_vector(
        target,
        name="target",
        require_real_nonnegative=True,
    )
    na = _power_of_two_qubits(state.size)
    if na < 2:
        raise ValueError("The half-support loader requires na >= 2.")

    m = na - 1
    half = 1 << m
    positive = state[half:].copy()
    positive /= np.linalg.norm(positive)

    qc = QuantumCircuit(na, name="exact_half_support")
    # q[m] is the full-register MSB and therefore shifts lower-register
    # integer k to full-register index half + k.
    qc.x(m)
    tail_loader = build_positive_real_amplitude_loader(
        positive,
        label="positive_tree",
    )
    qc.compose(tail_loader, qubits=list(range(m)), inplace=True)
    return qc


def infer_geometric_ratio_from_target(
    target: Iterable[complex],
    *,
    fit_start: int = 2,
    ratio_atol: float = 1e-15,
) -> tuple[float, dict[str, Any]]:
    """Infer the asymptotic ratio r = exp(-Delta p) from the positive tail.

    The switching factor affects the first few positive-grid samples.  The
    final half of consecutive positive-tail ratios is already in the saturated
    geometric region for the Appendix-A grids.  Taking its median avoids the
    erroneous least-squares fit across the switching layer.
    """
    state = _normalized_vector(
        target,
        name="target",
        require_real_nonnegative=True,
    ).real
    na = _power_of_two_qubits(state.size)
    positive = state[1 << (na - 1) :]

    numerator = positive[1:]
    denominator = positive[:-1]
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (numerator > ratio_atol)
        & (denominator > ratio_atol)
    )
    ratios = numerator[valid] / denominator[valid]
    if ratios.size == 0:
        raise ValueError("Could not infer a geometric ratio from the target.")

    fit_start = max(0, int(fit_start))
    start = max(fit_start, ratios.size // 2)
    asymptotic = ratios[start:]
    if asymptotic.size == 0:
        asymptotic = ratios[-1:]

    ratio = float(np.median(asymptotic))
    if not np.isfinite(ratio) or not (0.0 < ratio <= 1.0 + 1e-12):
        raise ValueError(f"Invalid inferred geometric ratio: {ratio!r}.")
    ratio = min(ratio, 1.0)

    spread = float(np.max(np.abs(asymptotic - ratio)))
    return ratio, {
        "r_fit": ratio,                 # backwards-compatible key
        "ratio": ratio,
        "ratio_source": "median of asymptotic positive-tail ratios",
        "ratio_samples": int(asymptotic.size),
        "ratio_max_abs_spread": spread,
        "fit_start": fit_start,
    }


def build_geometric_tail_state(m: int, r: float) -> QuantumCircuit:
    m = int(m)
    r = float(r)
    if m < 1:
        raise ValueError("m must be >= 1.")
    if not np.isfinite(r) or not (0.0 < r <= 1.0):
        raise ValueError("r must satisfy 0 < r <= 1.")

    qc = QuantumCircuit(m, name="geometric_tail")
    # q[ell] has integer weight 2**ell because q[0] is the LSB.
    for ell in range(m):
        alpha = r ** (1 << ell)
        qc.ry(2.0 * math.atan(alpha), ell)
    return qc


def _normalized_geometric_vector(m: int, r: float) -> np.ndarray:
    k = np.arange(1 << m, dtype=float)
    values = r**k
    return values / np.linalg.norm(values)


def build_half_support_geometric_loader(
    target: Iterable[complex],
    fit_start: int = 2,
    *,
    r: float | None = None,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    state = _normalized_vector(
        target,
        name="target",
        require_real_nonnegative=True,
    )
    na = _power_of_two_qubits(state.size)
    if na < 2:
        raise ValueError("The geometric half-support loader requires na >= 2.")

    m = na - 1
    if r is None:
        ratio, ratio_meta = infer_geometric_ratio_from_target(
            state,
            fit_start=fit_start,
        )
    else:
        ratio = float(r)
        if not np.isfinite(ratio) or not (0.0 < ratio <= 1.0):
            raise ValueError("r must satisfy 0 < r <= 1.")
        ratio_meta = {
            "r_fit": ratio,
            "ratio": ratio,
            "ratio_source": "explicit argument",
            "ratio_samples": 0,
            "ratio_max_abs_spread": 0.0,
            "fit_start": int(fit_start),
        }

    qc = QuantumCircuit(na, name="geometric_half_support")
    qc.x(m)
    qc.compose(
        build_geometric_tail_state(m, ratio),
        qubits=list(range(m)),
        inplace=True,
    )

    meta = {
        "na": na,
        "m": m,
        **ratio_meta,
        "positive_half_amplitudes": _normalized_geometric_vector(m, ratio),
    }
    return qc, meta


def _append_zero_controlled_gate(
    qc: QuantumCircuit,
    *,
    gate,
    controls: list[int],
    targets: list[int],
) -> None:
    """Append a gate controlled on every listed control qubit being zero.

    Explicit X conjugation is used instead of ``ctrl_state=0`` so that the
    implementation is independent of control-state bit-string conventions.
    """
    if not controls:
        qc.append(gate, targets)
        return

    for q in controls:
        qc.x(q)
    qc.append(gate.control(len(controls)), controls + targets)
    for q in reversed(controls):
        qc.x(q)


def build_patched_geometric_half_support_loader(
    target: Iterable[complex],
    patch_qubits: int = 1,
    *,
    fit_start: int = 2,
    r: float | None = None,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    state = _normalized_vector(
        target,
        name="target",
        require_real_nonnegative=True,
    )
    na = _power_of_two_qubits(state.size)
    if na < 3:
        raise ValueError("The patched loader requires na >= 3.")

    m = na - 1
    b = int(patch_qubits)
    if not (1 <= b < m):
        raise ValueError(f"patch_qubits must satisfy 1 <= b < {m}.")
    P = 1 << b

    qc, geom_meta = build_half_support_geometric_loader(
        state,
        fit_start=fit_start,
        r=r,
    )
    qc.name = f"patched_geometric_P{P}"
    ratio = float(geom_meta["ratio"])

    half = 1 << m
    target_patch = state[half : half + P].copy()
    target_patch /= np.linalg.norm(target_patch)

    # G_b acts on q[0],...,q[b-1], which are exactly the b LSBs encoding
    # k mod P.  T_b is built with the same little-endian convention.
    G_b = build_geometric_tail_state(b, ratio)
    T_b = build_positive_real_amplitude_loader(
        target_patch,
        label=f"target_patch_P{P}",
    )

    local = QuantumCircuit(b, name=f"T{P}_G{P}dg")
    local.compose(G_b.inverse(), qubits=list(range(b)), inplace=True)
    local.compose(T_b, qubits=list(range(b)), inplace=True)

    controls = list(range(b, m))
    targets = list(range(b))
    _append_zero_controlled_gate(
        qc,
        gate=local.to_gate(label=f"T{P}G{P}†"),
        controls=controls,
        targets=targets,
    )

    geometric = _normalized_geometric_vector(m, ratio)
    B_P = float(np.linalg.norm(geometric[:P]))
    patched = geometric.copy()
    patched[:P] = B_P * target_patch.real

    meta = {
        **geom_meta,
        "patch_qubits": b,
        "patch_size": P,
        "patch_controls": len(controls),
        "geometric_patch_norm": B_P,
        "target_patch": target_patch,
        "patched_positive_half_amplitudes": patched,
    }
    return qc, meta


# ---------------------------------------------------------------------------
# Analysis and validation helpers
# ---------------------------------------------------------------------------


def statevector_from_circuit(qc: QuantumCircuit) -> np.ndarray:
    if not isinstance(qc, QuantumCircuit):
        raise TypeError("qc must be a qiskit.QuantumCircuit.")
    return np.asarray(Statevector.from_instruction(qc).data, dtype=complex)


def fidelity_with_target(
    target: Iterable[complex],
    state: Iterable[complex] | Statevector,
) -> float:
    target_vec = _normalized_vector(target, name="target")
    state_vec = _normalized_vector(
        state.data if isinstance(state, Statevector) else state,
        name="state",
    )
    if target_vec.shape != state_vec.shape:
        raise ValueError(
            f"Target/state shape mismatch: {target_vec.shape} vs {state_vec.shape}."
        )
    value = float(abs(np.vdot(target_vec, state_vec)) ** 2)
    return min(1.0, max(0.0, value))


def circuit_metrics(
    qc: QuantumCircuit,
    *,
    decompose_reps: int = 10,
    basis_gates: tuple[str, ...] = _DEFAULT_BASIS,
    optimization_level: int = 0,
) -> dict[str, Any]:
    """Return elementary-gate metrics used for the accuracy-cost figure.

    The explicit tree and controlled-patch circuits are first recursively
    decomposed and then reduced to Qiskit's one-/two-qubit ``u,cx`` basis.
    Keeping ``optimization_level=0`` avoids silently changing the architecture.
    """
    if not isinstance(qc, QuantumCircuit):
        raise TypeError("qc must be a qiskit.QuantumCircuit.")

    dec = qc.decompose(reps=int(decompose_reps))
    compiled = transpile(
        dec,
        basis_gates=list(basis_gates),
        optimization_level=int(optimization_level),
    )
    ops = OrderedDict(
        sorted(
            ((str(k), int(v)) for k, v in compiled.count_ops().items()),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return {
        "num_qubits": int(compiled.num_qubits),
        "depth": int(compiled.depth()),
        "size": int(compiled.size()),
        "count_ops": ops,
        "basis_gates": tuple(basis_gates),
        "optimization_level": int(optimization_level),
        "decompose_reps": int(decompose_reps),
    }


def expected_half_support_state(target: Iterable[complex]) -> np.ndarray:
    state = _normalized_vector(target, name="target")
    half = state.size // 2
    out = np.zeros_like(state)
    out[half:] = state[half:] / np.linalg.norm(state[half:])
    return out


def expected_geometric_half_support_state(
    target: Iterable[complex],
    *,
    r: float | None = None,
) -> np.ndarray:
    state = _normalized_vector(target, name="target")
    na = _power_of_two_qubits(state.size)
    m = na - 1
    if r is None:
        r, _ = infer_geometric_ratio_from_target(state)
    out = np.zeros_like(state)
    out[1 << m :] = _normalized_geometric_vector(m, float(r))
    return out


def expected_patched_state(
    target: Iterable[complex],
    *,
    patch_qubits: int,
    r: float | None = None,
) -> np.ndarray:
    state = _normalized_vector(target, name="target")
    na = _power_of_two_qubits(state.size)
    m = na - 1
    b = int(patch_qubits)
    P = 1 << b
    if not (1 <= b < m):
        raise ValueError(f"patch_qubits must satisfy 1 <= b < {m}.")
    if r is None:
        r, _ = infer_geometric_ratio_from_target(state)

    geometric = _normalized_geometric_vector(m, float(r))
    B_P = np.linalg.norm(geometric[:P])
    patch = state[(1 << m) : (1 << m) + P].copy()
    patch /= np.linalg.norm(patch)
    geometric[:P] = B_P * patch.real

    out = np.zeros_like(state)
    out[1 << m :] = geometric
    return out


def run_self_checks(*, verbose: bool = True) -> dict[str, Any]:
    """Run endianness, statevector, normalization, and benchmark checks."""
    report: dict[str, Any] = {}

    # Distinct amplitudes detect accidental bit reversal immediately.
    sentinel = np.arange(1.0, 9.0)
    sentinel /= np.linalg.norm(sentinel)
    qc_sentinel = build_positive_real_amplitude_loader(sentinel)
    sv_sentinel = statevector_from_circuit(qc_sentinel)
    sentinel_error = float(np.max(np.abs(sv_sentinel - sentinel)))
    report["tree_endianness_max_abs_error"] = sentinel_error
    if sentinel_error > 1e-10:
        raise AssertionError(
            "Tree-loader endianness check failed: "
            f"max error={sentinel_error:.3e}."
        )

    benchmark_expected = {
        5: {
            "half": 1.0000,
            "geom": 0.8985823292227041,
            "P2": 0.9849558743402199,
            "P4": 0.9993923662139300,
        },
        7: {
            "half": 0.9997555907871980,
            "geom": 0.9282456023428295,
            "P2": 0.9760525018559348,
            "P4": 0.9930698691959332,
        },
    }

    benchmark_report: dict[int, dict[str, float]] = {}
    for na, expected in benchmark_expected.items():
        _, target, _ = normalized_target_erf(na=na, R=4.0, eps=1e-6)

        qc_half = build_half_support_loader(target)
        qc_geom, meta = build_half_support_geometric_loader(target)
        qc_p2, _ = build_patched_geometric_half_support_loader(
            target, patch_qubits=1
        )
        qc_p4, _ = build_patched_geometric_half_support_loader(
            target, patch_qubits=2
        )

        states = {
            "half": statevector_from_circuit(qc_half),
            "geom": statevector_from_circuit(qc_geom),
            "P2": statevector_from_circuit(qc_p2),
            "P4": statevector_from_circuit(qc_p4),
        }
        expected_states = {
            "half": expected_half_support_state(target),
            "geom": expected_geometric_half_support_state(
                target, r=float(meta["ratio"])
            ),
            "P2": expected_patched_state(
                target, patch_qubits=1, r=float(meta["ratio"])
            ),
            "P4": expected_patched_state(
                target, patch_qubits=2, r=float(meta["ratio"])
            ),
        }

        row: dict[str, float] = {}
        for name, sv in states.items():
            state_error = float(np.max(np.abs(sv - expected_states[name])))
            if state_error > 1e-10:
                raise AssertionError(
                    f"{name} statevector check failed for na={na}: "
                    f"max error={state_error:.3e}."
                )
            fid = fidelity_with_target(target, sv)
            row[name] = fid
            if abs(fid - expected[name]) > 5e-12:
                raise AssertionError(
                    f"{name} benchmark mismatch for na={na}: "
                    f"got {fid:.15f}, expected {expected[name]:.15f}."
                )
        benchmark_report[na] = row

    report["benchmarks"] = benchmark_report
    report["status"] = "PASS"

    if verbose:
        print("test_loaders self-check: PASS")
        print(f"  tree endian max error: {sentinel_error:.3e}")
        for na, row in benchmark_report.items():
            print(
                f"  na={na}: half={row['half']:.12f}, "
                f"geom={row['geom']:.12f}, "
                f"P2={row['P2']:.12f}, P4={row['P4']:.12f}"
            )
    return report


__all__ = [
    "make_p_grid",
    "erf_profile_on_p",
    "normalized_target_erf",
    "build_positive_real_amplitude_loader",
    "build_initialize_loader",
    "build_tree_loader",
    "build_half_support_loader",
    "infer_geometric_ratio_from_target",
    "build_geometric_tail_state",
    "build_half_support_geometric_loader",
    "build_patched_geometric_half_support_loader",
    "statevector_from_circuit",
    "fidelity_with_target",
    "circuit_metrics",
    "expected_half_support_state",
    "expected_geometric_half_support_state",
    "expected_patched_state",
    "run_self_checks",
]


if __name__ == "__main__":
    run_self_checks(verbose=True)
