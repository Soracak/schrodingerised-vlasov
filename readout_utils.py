"""Readout helpers for the 1D1V Schrödingerisation circuit.

The default readout marks the requested P-register basis state with one clean
flag qubit and saves only the reduced density matrix on X, V, and that flag.
All arithmetic/work registers are traced out by Aer.  Extracting the flag=1
block gives the selected P slice without constructing the full X,V,P density
matrix and without assuming that every workspace qubit is exactly |0>.
"""

from collections import OrderedDict
import time

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator


def remove_idle_wires(circ: QuantumCircuit) -> QuantumCircuit:
    used_qubits = []
    for instruction in circ.data:
        for qubit in instruction.qubits:
            if qubit not in used_qubits:
                used_qubits.append(qubit)

    qubit_map = {q: i for i, q in enumerate(used_qubits)}
    new_circ = QuantumCircuit(len(used_qubits), circ.num_clbits)

    for instruction in circ.data:
        new_qubits = [new_circ.qubits[qubit_map[q]] for q in instruction.qubits]
        new_circ.append(instruction.operation, new_qubits, instruction.clbits)

    return new_circ


def ops_unitary_string(qc_like) -> str:
    ops = OrderedDict(
        sorted(qc_like.count_ops().items(), key=lambda kv: (-kv[1], kv[0]))
    )
    for key in list(ops.keys()):
        if key in ("barrier", "reset") or key.startswith("save_"):
            ops.pop(key, None)
    return ", ".join(f"{key}={value}" for key, value in ops.items())


def _align_complex_xv_slice(u_complex, *, p_readout_idx: int):
    """Remove one global phase and return the real X,V field."""
    u_complex = np.asarray(u_complex, dtype=complex)

    total = np.sum(u_complex)
    if abs(total) > 1e-15:
        slice_phase = float(np.angle(total))
    else:
        pivot = int(np.argmax(np.abs(u_complex)))
        slice_phase = float(np.angle(u_complex.ravel()[pivot]))

    u_aligned = u_complex * np.exp(-1j * slice_phase)
    imag_ratio = float(
        np.linalg.norm(np.imag(u_aligned))
        / max(np.linalg.norm(u_aligned), 1e-15)
    )
    u_quant = np.real(u_aligned)

    meta = {
        "p_readout_idx": int(p_readout_idx),
        "slice_phase": slice_phase,
        "imag_ratio": imag_ratio,
        "slice_complex_norm": float(np.linalg.norm(u_complex)),
        "slice_real_norm": float(np.linalg.norm(u_quant)),
        "slice_mass_real": float(np.sum(u_quant)),
    }
    return u_quant, meta


def align_p_slice_from_psi_full(psi_full, p_readout_idx: int, norm: float):
    """Extract and phase-align one P slice from an array shaped (P,V,X)."""
    p_readout_idx = int(p_readout_idx)
    u_complex = np.asarray(psi_full, dtype=complex)[p_readout_idx].T * norm
    return _align_complex_xv_slice(u_complex, p_readout_idx=p_readout_idx)


def align_and_extract_p_slice(
    psi_ref,
    nx,
    nv,
    na,
    norm,
    *,
    p_readout_idx=None,
):
    """Compatibility helper for the optional full-statevector path."""
    if p_readout_idx is None:
        p_readout_idx = 1 << (na - 1)

    expected = 1 << (nx + nv + na)
    psi_ref = np.asarray(psi_ref, dtype=complex)
    if psi_ref.size != expected:
        raise ValueError(
            "The statevector path requires a circuit containing only X,V,P: "
            f"expected {expected} amplitudes, got {psi_ref.size}."
        )

    psi_full = psi_ref.reshape(1 << na, 1 << nv, 1 << nx)
    return align_p_slice_from_psi_full(psi_full, p_readout_idx, norm)


def scan_p_slices_from_psi_full(
    psi_full,
    *,
    fp,
    p_grid,
    norm_base: float,
    f_class,
    compare_fn,
    q_indices,
    fp_floor: float = 1e-14,
):
    """Scan candidate P slices from an already available full X,V,P state."""
    rows = []
    for q in q_indices:
        q = int(q)
        if q < 0 or q >= len(fp):
            continue

        fp_q = float(fp[q])
        if abs(fp_q) < fp_floor:
            continue

        scale_q = float(norm_base) / fp_q
        u_q, meta = align_p_slice_from_psi_full(psi_full, q, scale_q)
        rel_l2, mass_q, mass_c, _, _ = compare_fn(u_q, f_class)

        rows.append(
            {
                "q": q,
                "p": float(p_grid[q]),
                "fp": fp_q,
                "scale": scale_q,
                "rel_l2": float(rel_l2),
                "mass_q": float(mass_q),
                "mass_c": float(mass_c),
                **meta,
            }
        )
    return rows


def _register_positions(qc, register):
    return [qc.find_bit(qubit).index for qubit in register]


def _append_p_match_flag(qc, p_qubits, p_readout_idx: int):
    """Append a flag that is one exactly when P equals p_readout_idx."""
    p_readout_idx = int(p_readout_idx)
    if p_readout_idx < 0 or p_readout_idx >= (1 << len(p_qubits)):
        raise ValueError(
            f"p_readout_idx={p_readout_idx} does not fit in {len(p_qubits)} qubits."
        )

    existing_names = {reg.name for reg in qc.qregs}
    flag_name = "PSEL"
    suffix = 0
    while flag_name in existing_names:
        suffix += 1
        flag_name = f"PSEL_{suffix}"

    flag = QuantumRegister(1, flag_name)
    qc.add_register(flag)

    # Convert the desired bit pattern into the all-ones control state.
    flipped = []
    for bit, qubit in enumerate(p_qubits):
        if ((p_readout_idx >> bit) & 1) == 0:
            qc.x(qubit)
            flipped.append(qubit)

    qc.mcx(list(p_qubits), flag[0])

    for qubit in reversed(flipped):
        qc.x(qubit)

    return flag[0]


def obtain_u_quant_endian_safe(
    qc,
    rx,
    rv,
    ra,
    nx,
    nv,
    na,
    norm,
    *,
    p_readout_idx=None,
    purity_warn=0.999,
    reps=5,
    do_transpile=False,
    verbose_timing=True,
    return_psi_full=False,
):
    """Recover one P slice using a one-qubit flag and reduced X,V density matrix.

    The simulator evolves the full circuit, but the saved dense object contains
    only X, V, and one selection flag.  Every other qubit, including P and all
    arithmetic workspaces, is traced out by Aer.

    If the conditioned X,V state is pure, its dominant eigenvector is the
    desired amplitude slice.  The dominant eigenvalue retains the selected-P
    probability, so the output has the same normalization convention as the
    earlier full-XVP method.
    """
    if p_readout_idx is None:
        p_readout_idx = 1 << (na - 1)
    p_readout_idx = int(p_readout_idx)

    if return_psi_full:
        # The flagged method deliberately does not construct a full P stack.
        return obtain_u_quant_density_matrix(
            qc,
            rx,
            rv,
            ra,
            nx,
            nv,
            na,
            norm,
            p_readout_idx=p_readout_idx,
            purity_warn=purity_warn,
            reps=reps,
            do_transpile=do_transpile,
            verbose_timing=verbose_timing,
            return_psi_full=True,
        )

    if len(rx) != nx or len(rv) != nv or len(ra) != na:
        raise ValueError(
            "Register lengths do not match nx, nv, na: "
            f"len(X)={len(rx)}, len(V)={len(rv)}, len(P)={len(ra)}, "
            f"nx={nx}, nv={nv}, na={na}."
        )

    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer.library import SaveDensityMatrix

    sim = AerSimulator(method="matrix_product_state")
    t_total0 = time.perf_counter()

    # Record logical qubit positions before copying/adding the flag.
    x_positions = _register_positions(qc, rx)
    v_positions = _register_positions(qc, rv)
    p_positions = _register_positions(qc, ra)

    # 1) Add a P-selection flag, then decompose the complete circuit hierarchy.
    t0 = time.perf_counter()
    qc_marked = qc.copy()
    p_marked = [qc_marked.qubits[pos] for pos in p_positions]
    flag_qubit = _append_p_match_flag(qc_marked, p_marked, p_readout_idx)
    flag_position = qc_marked.find_bit(flag_qubit).index
    qc_run = qc_marked.decompose(reps=reps)
    t_decompose = time.perf_counter() - t0

    # 2) Save only rho_(X,V,flag).  P and every arithmetic register are traced out.
    t0 = time.perf_counter()
    x_run = [qc_run.qubits[pos] for pos in x_positions]
    v_run = [qc_run.qubits[pos] for pos in v_positions]
    flag_run = qc_run.qubits[flag_position]
    xvf = x_run + v_run + [flag_run]
    qc_run.append(SaveDensityMatrix(len(xvf), label="rho_xvf"), xvf)
    t_saveappend = time.perf_counter() - t0

    # 3) Optional transpilation.
    t_transpile = 0.0
    if do_transpile:
        t0 = time.perf_counter()
        qc_run = transpile(qc_run, sim, optimization_level=0)
        t_transpile = time.perf_counter() - t0

    unitary_ops_str = ops_unitary_string(qc_run)

    # 4) Run the exact MPS simulation.
    t0 = time.perf_counter()
    result = sim.run(qc_run, shots=1).result()
    t_run = time.perf_counter() - t0

    if not result.success:
        raise RuntimeError(f"Aer MPS simulation failed: {result.status}")

    # 5) Extract the flag=1 block and its dominant pure component.
    t0 = time.perf_counter()
    rho_xvf = DensityMatrix(result.data(0)["rho_xvf"]).data

    Nx = 1 << nx
    Nv = 1 << nv
    dim_xv = Nx * Nv
    expected_shape = (2 * dim_xv, 2 * dim_xv)
    if rho_xvf.shape != expected_shape:
        raise ValueError(
            f"Expected rho_xvf shape {expected_shape}, got {rho_xvf.shape}."
        )

    # With qargs ordered as X,V,flag, the flag is the most significant saved bit.
    rho_selected = np.asarray(
        rho_xvf[dim_xv : 2 * dim_xv, dim_xv : 2 * dim_xv],
        dtype=complex,
    )
    rho_selected = 0.5 * (rho_selected + rho_selected.conj().T)

    p_probability = float(np.real(np.trace(rho_selected)))
    if p_probability <= 1e-15:
        raise ValueError(
            f"Selected P outcome has near-zero probability: {p_probability:.3e}."
        )

    rho_conditional = rho_selected / p_probability
    conditional_purity = float(
        np.real(np.trace(rho_conditional @ rho_conditional))
    )

    evals, evecs = np.linalg.eigh(rho_selected)
    dominant_index = int(np.argmax(evals))
    dominant_eigenvalue = max(float(np.real(evals[dominant_index])), 0.0)
    dominant_fraction = dominant_eigenvalue / p_probability

    psi_selected = evecs[:, dominant_index] * np.sqrt(dominant_eigenvalue)
    u_complex = psi_selected.reshape(Nv, Nx).T * norm
    u_quant, slice_meta = _align_complex_xv_slice(
        u_complex,
        p_readout_idx=p_readout_idx,
    )

    t_post = time.perf_counter() - t0
    t_total = time.perf_counter() - t_total0

    info = {
        "purity": conditional_purity,
        "readout_mode": "flagged_xv_density_matrix",
        "t_decompose": t_decompose,
        "t_saveappend": t_saveappend,
        "t_transpile": t_transpile,
        "t_run": t_run,
        "t_post": t_post,
        "t_total": t_total,
        "unitary_ops": unitary_ops_str,
        "num_total_qubits": int(qc_run.num_qubits),
        "saved_density_matrix_qubits": int(nx + nv + 1),
        "saved_density_matrix_dimension": int(2 * dim_xv),
        "p_probability": p_probability,
        "conditional_purity": conditional_purity,
        "dominant_eigenvalue": dominant_eigenvalue,
        "dominant_fraction": dominant_fraction,
        **slice_meta,
    }

    if conditional_purity < purity_warn:
        info["warning"] = (
            "The X,V state conditioned on the selected P outcome is mixed: "
            f"purity={conditional_purity:.8f}. Check arithmetic uncomputation."
        )

    print("[readout] decomposed ops:")
    print(qc_run.count_ops())

    if verbose_timing:
        print(
            f"\n[quantum wall] decompose={t_decompose:.2f}s | "
            f"save_append={t_saveappend:.2f}s | "
            f"transpile={t_transpile:.2f}s | run={t_run:.2f}s | "
            f"post={t_post:.2f}s\n"
        )
        print(
            f"[readout] mode=flagged_xv_density_matrix, "
            f"p_readout_idx={p_readout_idx}"
        )
        print(
            f"[readout] Pr(P=q)={p_probability:.12e}, "
            f"conditional purity={conditional_purity:.12e}, "
            f"dominant fraction={dominant_fraction:.12e}"
        )
        print(
            f"[readout] slice_phase={slice_meta['slice_phase']:.6e}, "
            f"imag_ratio={slice_meta['imag_ratio']:.6e}"
        )

    return u_quant, info


def obtain_u_quant_density_matrix(
    qc,
    rx,
    rv,
    ra,
    nx,
    nv,
    na,
    norm,
    *,
    p_readout_idx=None,
    purity_warn=0.999,
    reps=5,
    do_transpile=False,
    verbose_timing=True,
    return_psi_full=False,
):
    """Legacy dense X,V,P density-matrix readout for small validation runs."""
    from qiskit.quantum_info import DensityMatrix
    from qiskit_aer.library import SaveDensityMatrix

    if p_readout_idx is None:
        p_readout_idx = 1 << (na - 1)
    p_readout_idx = int(p_readout_idx)

    sim = AerSimulator(method="matrix_product_state")
    t_total0 = time.perf_counter()

    t0 = time.perf_counter()
    qc_run = qc.copy().decompose(reps=reps)
    t_decompose = time.perf_counter() - t0

    t0 = time.perf_counter()
    xvp = list(rx) + list(rv) + list(ra)
    qc_run.append(SaveDensityMatrix(len(xvp), label="rho_xvp"), xvp)
    t_saveappend = time.perf_counter() - t0

    t_transpile = 0.0
    if do_transpile:
        t0 = time.perf_counter()
        qc_run = transpile(qc_run, sim, optimization_level=0)
        t_transpile = time.perf_counter() - t0

    unitary_ops_str = ops_unitary_string(qc_run)

    t0 = time.perf_counter()
    result = sim.run(qc_run, shots=1).result()
    t_run = time.perf_counter() - t0

    t0 = time.perf_counter()
    rho = DensityMatrix(result.data(0)["rho_xvp"]).data
    purity = float(np.real(np.sum(np.abs(rho) ** 2)))

    evals, evecs = np.linalg.eigh(0.5 * (rho + rho.conj().T))
    dominant_index = int(np.argmax(evals))
    dominant_eigenvalue = max(float(np.real(evals[dominant_index])), 0.0)
    psi = evecs[:, dominant_index] * np.sqrt(dominant_eigenvalue)

    Nx = 1 << nx
    Nv = 1 << nv
    Np = 1 << na
    psi_full = psi.reshape(Np, Nv, Nx)
    u_quant, slice_meta = align_p_slice_from_psi_full(
        psi_full,
        p_readout_idx=p_readout_idx,
        norm=norm,
    )

    t_post = time.perf_counter() - t0
    t_total = time.perf_counter() - t_total0

    info = {
        "purity": purity,
        "readout_mode": "density_matrix_xvp",
        "t_decompose": t_decompose,
        "t_saveappend": t_saveappend,
        "t_transpile": t_transpile,
        "t_run": t_run,
        "t_post": t_post,
        "t_total": t_total,
        "unitary_ops": unitary_ops_str,
        "dominant_eigenvalue": dominant_eigenvalue,
        **slice_meta,
    }

    if purity < purity_warn:
        info["warning"] = "X,V,P appears mixed; workspace may not be fully uncomputed."

    if verbose_timing:
        print(
            f"\n[quantum wall] decompose={t_decompose:.2f}s | "
            f"save_append={t_saveappend:.2f}s | "
            f"transpile={t_transpile:.2f}s | run={t_run:.2f}s | "
            f"post={t_post:.2f}s\n"
        )

    if return_psi_full:
        return u_quant, info, psi_full
    return u_quant, info
