"""Readout helpers for the 1D1V Schrödingerisation circuit.

The default readout marks the requested P-register basis state with one clean
flag qubit and saves only the reduced density matrix on X, V, and that flag.
All arithmetic/work registers are traced out by Aer.  Extracting the flag=1
block gives the selected P slice without constructing the full X,V,P density
matrix and without assuming that every workspace qubit is exactly |0>.
"""

from collections import OrderedDict
from pathlib import Path
import json
import time
import gc

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit_aer import AerSimulator

import os
import threading
import psutil


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


def _release_circuit_instructions(circ: QuantumCircuit) -> None:
    """Drop hierarchical instructions while retaining bits/registers.

    Used only after a separate decomposed ``qc_run`` has already been created
    and the caller explicitly selected destructive production readout with
    ``copy_circuit=False``.
    """
    try:
        circ.data.clear()
    except AttributeError:
        del circ.data[:]


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
    simulator_options=None,
    mps_log_path=None,
    cpu_log_path=None,
    copy_circuit=True,
):
    """Recover one P slice using a one-qubit flag and reduced X,V density matrix.

    The simulator evolves the full circuit, but the saved dense object contains
    only X, V, and one selection flag.  Every other qubit, including P and all
    arithmetic workspaces, is traced out by Aer.

    If the conditioned X,V state is pure, its dominant eigenvector is the
    desired amplitude slice.  The dominant eigenvalue retains the selected-P
    probability, so the output has the same normalization convention as the
    earlier full-XVP method.

    Set ``copy_circuit=False`` for memory-constrained production jobs when the
    caller will not reuse ``qc`` after readout.  This avoids holding a second
    hierarchical copy before decomposition; the input circuit is then marked
    in place with the P-selection flag.
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

    from qiskit_aer.library import SaveDensityMatrix

    # The default values preserve the exact high-accuracy behavior used in the
    # validated runs.  Pass simulator_options from the separate MPS example
    # main script to benchmark swap direction, CPU threads, LAPACK, or logging.
    options = {
        "matrix_product_state_max_bond_dimension": None,
        "matrix_product_state_truncation_threshold": 1e-16,
        "mps_log_data": False,
        "mps_swap_direction": "mps_swap_left",
        "max_parallel_threads": 1,
        "max_parallel_experiments": 1,
        "max_parallel_shots": 1,
        "mps_omp_threads": 1,
        "mps_parallel_threshold": 14,
        "mps_lapack": False,
    }
    if simulator_options:
        options.update(dict(simulator_options))

    configured_threads = int(options.get("mps_omp_threads", 1))

    # Keep the method fixed here: this helper relies on MPS reduced-state saves.
    options.pop("method", None)
    sim = AerSimulator(method="matrix_product_state", **options)
    t_total0 = time.perf_counter()

    # Record logical qubit positions before copying/adding the flag.
    x_positions = _register_positions(qc, rx)
    v_positions = _register_positions(qc, rv)
    p_positions = _register_positions(qc, ra)

    # 1) Add a P-selection flag, then decompose the complete circuit hierarchy.
    t0 = time.perf_counter()
    qc_marked = qc.copy() if copy_circuit else qc
    p_marked = [qc_marked.qubits[pos] for pos in p_positions]
    flag_qubit = _append_p_match_flag(qc_marked, p_marked, p_readout_idx)
    flag_position = qc_marked.find_bit(flag_qubit).index
    qc_run = qc_marked.decompose(reps=reps)

    # In destructive production mode, merely deleting the local qc_marked
    # reference is insufficient because the caller still owns qc.  Once
    # qc_run exists, clear the caller's hierarchical instruction list so those
    # references are not retained throughout the multi-hour Aer run.
    if not copy_circuit:
        _release_circuit_instructions(qc)

    del p_marked, flag_qubit, qc_marked
    gc.collect()

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
    #result = sim.run(qc_run, shots=1).result()
    result, cpu_metrics = run_with_cpu_monitor(
        sim,
        qc_run,
        shots=1,
        configured_threads=configured_threads,
        sample_interval=5.0,
        record_interval=60.0,
        flush_interval=600.0,
        output_file=cpu_log_path,
    )
    t_run = time.perf_counter() - t0

    if not result.success:
        raise RuntimeError(f"Aer MPS simulation failed: {result.status}")

    # Optional MPS bond-dimension log.  Do not even materialize this
    # metadata unless logging was explicitly enabled: the diagnostic string can
    # become extremely large for production circuits.
    mps_logging_enabled = bool(options.get("mps_log_data", False))
    mps_log_file = None
    mps_log_chars = 0
    mps_log_present = False

    if mps_logging_enabled:
        result_metadata = result.results[0].metadata or {}
        mps_log = result_metadata.get(
            "MPS_log_data", result_metadata.get("mps_log_data")
        )
        mps_log_present = mps_log is not None
        if mps_log is not None:
            mps_log_text = str(mps_log)
            mps_log_chars = len(mps_log_text)
            if mps_log_path is not None:
                mps_log_file = Path(mps_log_path)
                mps_log_file.parent.mkdir(parents=True, exist_ok=True)
                mps_log_file.write_text(mps_log_text, encoding="utf-8")
            del mps_log_text
    else:
        mps_log = None

    # 5) Extract the flag=1 block and its dominant pure component.
    t0 = time.perf_counter()

    # Avoid constructing a second qiskit.quantum_info.DensityMatrix wrapper.
    # Copy only the selected flag=1 block, then release the much larger Aer
    # Result object before the eigensolve.
    rho_xvf = np.asarray(result.data(0)["rho_xvf"], dtype=np.complex128)

    Nx = 1 << nx
    Nv = 1 << nv
    dim_xv = Nx * Nv
    expected_shape = (2 * dim_xv, 2 * dim_xv)
    if rho_xvf.shape != expected_shape:
        raise ValueError(
            f"Expected rho_xvf shape {expected_shape}, got {rho_xvf.shape}."
        )

    # With qargs ordered as X,V,flag, the flag is the most significant saved bit.
    rho_selected = np.array(
        rho_xvf[dim_xv : 2 * dim_xv, dim_xv : 2 * dim_xv],
        dtype=np.complex128,
        copy=True,
    )

    del rho_xvf, result
    gc.collect()

    rho_selected = 0.5 * (rho_selected + rho_selected.conj().T)

    p_probability = float(np.real(np.trace(rho_selected)))
    if p_probability <= 1e-15:
        raise ValueError(
            f"Selected P outcome has near-zero probability: {p_probability:.3e}."
        )

    # For Hermitian rho, Tr[(rho/p)^2] = ||rho||_F^2 / p^2.  This avoids
    # allocating a second dense conditional matrix and a dense matrix product.
    conditional_purity = float(
        np.sum(np.abs(rho_selected) ** 2) / (p_probability * p_probability)
    )

    evals, evecs = np.linalg.eigh(rho_selected)
    dominant_index = int(np.argmax(evals))
    dominant_eigenvalue = max(float(np.real(evals[dominant_index])), 0.0)
    dominant_fraction = dominant_eigenvalue / p_probability

    dominant_vector = np.array(evecs[:, dominant_index], copy=True)
    del evals, evecs, rho_selected
    gc.collect()

    psi_selected = dominant_vector * np.sqrt(dominant_eigenvalue)
    del dominant_vector

    u_complex = psi_selected.reshape(Nv, Nx).T * norm
    del psi_selected

    u_quant, slice_meta = _align_complex_xv_slice(
        u_complex,
        p_readout_idx=p_readout_idx,
    )
    del u_complex
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
        "cpu_monitor": cpu_metrics,
        "unitary_ops": unitary_ops_str,
        "num_total_qubits": int(qc_run.num_qubits),
        "saved_density_matrix_qubits": int(nx + nv + 1),
        "saved_density_matrix_dimension": int(2 * dim_xv),
        "p_probability": p_probability,
        "conditional_purity": conditional_purity,
        "dominant_eigenvalue": dominant_eigenvalue,
        "dominant_fraction": dominant_fraction,
        "simulator_options": options,
        "copy_circuit": bool(copy_circuit),
        "mps_log_present": bool(mps_log_present),
        "mps_log_chars": int(mps_log_chars),
        "mps_log_path": str(mps_log_file) if mps_log_file is not None else None,
        **slice_meta,
    }

    if conditional_purity < purity_warn:
        info["warning"] = (
            "The X,V state conditioned on the selected P outcome is mixed: "
            f"purity={conditional_purity:.8f}. Check arithmetic uncomputation."
        )

    if verbose_timing:
        print("[readout] decomposed ops:")
        print(unitary_ops_str if unitary_ops_str else "(none)")
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
        print(f"[mps] options={json.dumps(options, sort_keys=True)}")
        if mps_log_present:
            where = str(mps_log_file) if mps_log_file is not None else "metadata only"
            print(f"[mps] log captured ({mps_log_chars} characters): {where}")

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

    dense_saved_qubits = nx + nv + na
    if dense_saved_qubits > 14:
        raise MemoryError(
            "Legacy X,V,P dense-density readout is disabled above 14 saved "
            f"qubits (requested {dense_saved_qubits}). Use "
            "obtain_u_quant_endian_safe() instead."
        )

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

def run_with_cpu_monitor(
    sim,
    circuit,
    *,
    shots=1,
    configured_threads=1,
    sample_interval=5.0,
    record_interval=60.0,
    flush_interval=600.0,
    output_file=None,
):
    """Run Aer while recording bounded CPU/RSS diagnostics.

    The old monitor appended one Python dict every 0.5 s for the entire run.
    That is unnecessary for multi-day jobs.  This version samples lightweight
    peak statistics every ``sample_interval`` seconds but stores only one trace
    point every ``record_interval`` seconds.  When ``output_file`` is supplied,
    it also writes a partial JSON file periodically so a scheduler timeout still
    leaves recent diagnostics on disk.

    ``equivalent_busy_cpus`` is based on process CPU-time accumulation.  A
    value of 1.0 means one logical CPU was fully occupied during the interval.
    """
    process = psutil.Process(os.getpid())
    stop_event = threading.Event()
    samples = []

    sample_interval = max(float(sample_interval), 0.5)
    record_interval = max(float(record_interval), sample_interval)
    flush_interval = max(float(flush_interval), record_interval)

    def cpu_seconds_now():
        times = process.cpu_times()
        return float(times.user + times.system)

    try:
        affinity = list(process.cpu_affinity())
    except (AttributeError, psutil.Error):
        affinity = None

    logical_cpus = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    wall_start = time.perf_counter()
    cpu_start = cpu_seconds_now()

    peak_busy_cpus = 0.0
    peak_rss_gib = 0.0
    max_process_threads = 0

    output_path = Path(output_file) if output_file is not None else None
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def snapshot_summary(*, final=False):
        now = time.perf_counter()
        try:
            cpu_now = cpu_seconds_now()
        except psutil.Error:
            cpu_now = cpu_start

        wall_seconds = now - wall_start
        cpu_seconds_used = cpu_now - cpu_start
        average_busy_cpus = (
            cpu_seconds_used / wall_seconds if wall_seconds > 0.0 else 0.0
        )

        return {
            "configured_mps_threads": int(configured_threads),
            "physical_cores": physical_cores,
            "logical_cpus": logical_cpus,
            "cpu_affinity": affinity,
            "usable_logical_cpus": (
                len(affinity) if affinity is not None else logical_cpus
            ),
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds_used,
            "average_equivalent_busy_cpus": average_busy_cpus,
            "peak_sampled_equivalent_busy_cpus": peak_busy_cpus,
            "maximum_process_thread_count": max_process_threads,
            "sample_interval_seconds": sample_interval,
            "record_interval_seconds": record_interval,
            "peak_rss_gib": peak_rss_gib,
            "final": bool(final),
        }

    def write_snapshot(*, final=False):
        if output_path is None:
            return
        payload = {
            "summary": snapshot_summary(final=final),
            "samples": list(samples),
        }
        # Atomic replace avoids leaving a half-written JSON if the monitor
        # happens to flush while the file is being inspected.
        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, output_path)

    def monitor():
        nonlocal peak_busy_cpus, peak_rss_gib, max_process_threads

        previous_wall = time.perf_counter()
        previous_cpu = cpu_seconds_now()
        last_record = 0.0
        last_flush = 0.0

        while not stop_event.wait(sample_interval):
            current_wall = time.perf_counter()
            current_cpu = cpu_seconds_now()

            elapsed_wall = current_wall - previous_wall
            elapsed_cpu = current_cpu - previous_cpu
            equivalent_cpus = (
                elapsed_cpu / elapsed_wall if elapsed_wall > 0.0 else 0.0
            )

            try:
                rss_gib = process.memory_info().rss / (1024**3)
                thread_count = process.num_threads()
            except psutil.Error:
                rss_gib = float("nan")
                thread_count = -1

            if np.isfinite(equivalent_cpus):
                peak_busy_cpus = max(peak_busy_cpus, equivalent_cpus)
            if np.isfinite(rss_gib):
                peak_rss_gib = max(peak_rss_gib, rss_gib)
            if thread_count >= 0:
                max_process_threads = max(max_process_threads, thread_count)

            elapsed_from_start = current_wall - wall_start

            if elapsed_from_start - last_record >= record_interval:
                samples.append(
                    {
                        "elapsed_s": elapsed_from_start,
                        "equivalent_busy_cpus": equivalent_cpus,
                        "process_threads": thread_count,
                        "rss_gib": rss_gib,
                    }
                )
                last_record = elapsed_from_start

            if (
                output_path is not None
                and elapsed_from_start - last_flush >= flush_interval
            ):
                try:
                    write_snapshot(final=False)
                except OSError:
                    pass
                last_flush = elapsed_from_start

            previous_wall = current_wall
            previous_cpu = current_cpu

    # Seed the peak memory/thread statistics before Aer starts.
    try:
        peak_rss_gib = process.memory_info().rss / (1024**3)
        max_process_threads = process.num_threads()
    except psutil.Error:
        peak_rss_gib = 0.0
        max_process_threads = 0

    monitor_thread = threading.Thread(
        target=monitor,
        name="aer_cpu_monitor",
        daemon=True,
    )
    monitor_thread.start()

    run_error = None
    try:
        result = sim.run(circuit, shots=shots).result()
    except BaseException as exc:
        run_error = exc
        raise
    finally:
        stop_event.set()
        monitor_thread.join(timeout=sample_interval + 1.0)
        if run_error is not None and output_path is not None:
            try:
                payload = {
                    "summary": snapshot_summary(final=False),
                    "samples": list(samples),
                    "error": {
                        "type": type(run_error).__name__,
                        "message": str(run_error),
                    },
                }
                tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
                tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                os.replace(tmp_path, output_path)
            except OSError:
                pass

    wall_end = time.perf_counter()
    cpu_end = cpu_seconds_now()
    wall_seconds = wall_end - wall_start
    cpu_seconds_used = cpu_end - cpu_start

    average_busy_cpus = (
        cpu_seconds_used / wall_seconds if wall_seconds > 0.0 else 0.0
    )

    try:
        final_rss_gib = process.memory_info().rss / (1024**3)
        final_threads = process.num_threads()
        if np.isfinite(final_rss_gib):
            peak_rss_gib = max(peak_rss_gib, final_rss_gib)
        max_process_threads = max(max_process_threads, final_threads)
    except psutil.Error:
        pass

    if peak_busy_cpus <= 0.0:
        peak_busy_cpus = average_busy_cpus

    metrics = {
        "configured_mps_threads": int(configured_threads),
        "physical_cores": physical_cores,
        "logical_cpus": logical_cpus,
        "cpu_affinity": affinity,
        "usable_logical_cpus": (
            len(affinity) if affinity is not None else logical_cpus
        ),
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": cpu_seconds_used,
        "average_equivalent_busy_cpus": average_busy_cpus,
        "peak_sampled_equivalent_busy_cpus": peak_busy_cpus,
        "maximum_process_thread_count": max_process_threads,
        "sample_interval_seconds": sample_interval,
        "record_interval_seconds": record_interval,
        "peak_rss_gib": peak_rss_gib,
        "final": True,
    }

    print("\n[cpu monitor]")
    print(f"  physical cores        : {physical_cores}")
    print(f"  logical CPUs          : {logical_cpus}")
    print(
        "  process affinity      : "
        f"{len(affinity) if affinity is not None else 'unknown'} CPUs"
    )
    print(f"  configured MPS threads: {configured_threads}")
    print(f"  process CPU time      : {cpu_seconds_used:.2f} s")
    print(f"  wall time             : {wall_seconds:.2f} s")
    print(f"  average busy CPUs     : {average_busy_cpus:.2f}")
    print(f"  peak sampled CPUs     : {peak_busy_cpus:.2f}")
    print(f"  maximum OS threads    : {max_process_threads}")
    print(f"  peak process memory   : {peak_rss_gib:.3f} GiB")

    if output_path is not None:
        write_snapshot(final=True)
        print(f"  CPU log written to    : {output_path}")

    return result, metrics

