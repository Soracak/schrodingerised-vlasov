# schro_1d1v_blocks.py
"""
Quantum circuit block builders for 1D1V Schrödingerization (Hu/Sato style).

This module is intentionally "dumb":
- it contains ONLY circuit constructors (QuantumCircuit builders)
- and small helper functions used by those constructors.

The main experiment script should:
- compute physical parameters (dx, dv, dt, R, S, alpha_fp/beta_fp, etc.)
- call these builders to obtain gates/blocks
- assemble the Strang splitting circuit.

Conventions:
- Register ordering in the main pipeline is typically X, V, P (LSB..MSB) for idx decoding.
- These blocks should not assume global register objects; they receive qubit lists.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple
from qiskit import QuantumCircuit

# ---------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------

def theta_from_v_grid_linear(v: np.ndarray, theta_scale: float, nv: int) -> Tuple[float, np.ndarray]:
    """
    For uniform cell-centered v-grid v_k = v0 + k*dv, build:
        theta(v_k) = theta_off + sum_r theta_bits[r] * b_r(k)
                  = theta_scale * v_k
    where b_r(k) are the bits of integer k (LSB first).
    """
    Nv = len(v)
    assert Nv == 2**nv, "Nv must be 2**nv for binary encoding."

    dv = v[1] - v[0]
    if not np.allclose(v, v[0] + dv * np.arange(Nv)):
        print("Warning: v-grid not exactly uniform; using v_k = v0 + dv*k.")

    v0 = float(v[0])
    theta_off  = theta_scale * v0
    theta_bits = np.array([theta_scale * dv * (2**r) for r in range(nv)], dtype=float)
    return theta_off, theta_bits


def theta_bits_from_twos_complement(theta_scale: float, nE: int) -> Tuple[float, np.ndarray]:
    """
    For an nE-bit two's-complement integer E, return weights so that:
        theta(E) = theta_scale * E_signed
               = theta_off + sum_r theta_bits[r] * bit_r(E)
    with MSB having negative weight.
    """
    theta_off = 0.0
    theta_bits = [theta_scale * (2**r) for r in range(nE - 1)]
    theta_bits.append(-theta_scale * (2**(nE - 1)))
    return theta_off, np.array(theta_bits, dtype=float)

# ---------------------------------------------------------------------
# PASTE YOUR EXISTING BLOCK BUILDERS FROM MAIN HERE (unchanged)
#
# IMPORTANT:
# - Do NOT call .to_gate() on blocks that include non-gate Instructions.
# - Prefer returning QuantumCircuit or Instruction from to_instruction().
# - If you currently return Gate, keep it, but then plotting will show the gate as a box.
# ---------------------------------------------------------------------

# ============================================
# Circuit blocks:
# Block 1: vWj, cvV1, inverse_cvV1, vV2,
# Block 2: eWj, cEV1, inverse_cEV1, eV2,
# ============================================

# ============================================
# Block 1
# ============================================

def vWj(n_x, n_v, j, theta_off, theta_bits, lam,
        xgate=False, use_abs_v=False, name=None):
    """
    Velocity-dependent Hu-style W_j acting on X (n_x qubits) and V (n_v qubits).

    Layout inside this gate:
        X-qubits: 0 .. n_x-1
        V-qubits: n_x .. n_x+n_v-1

    For a fixed velocity basis |v_k>, the base angle is
        theta_base(v_k) = theta_off + sum_r theta_bits[r] * b_r(k),

    If use_abs_v=True, we correct this so that effectively
        theta_eff(v_k) ∝ |v_k|
    using the MSB of the v-register as a sign bit.
    """
    if name is None:
        name = f"vW{j}"
    qc = QuantumCircuit(n_x + n_v, name=name)

    x_ctrls = list(range(j-1))
    x_top   = j - 1
    v_qubits = list(range(n_x, n_x + n_v))
    v_msb    = v_qubits[-1]  # treat MSB as "sign" indicator

    # --- Hu scaffold on X ---
    if j > 1:
        qc.cx(x_top, x_ctrls)
    if lam != 0:
        qc.p(lam, x_top)
    qc.h(x_top)
    if xgate and j > 1:
        qc.x(x_ctrls)

    # --- base linear-in-v part: theta_off + sum_r theta_bits[r] b_r(k) ---
    if abs(theta_off) > 1e-12:
        if j > 1:
            qc.mcrz(theta_off, x_ctrls, x_top)
        else:
            qc.rz(theta_off, x_top)

    for r, th_r in enumerate(theta_bits):
        if abs(th_r) < 1e-12:
            continue
        vq = v_qubits[r]
        if j > 1:
            controls = x_ctrls + [vq]
            qc.mcrz(th_r, controls, x_top)
        else:
            qc.crz(th_r, vq, x_top)

    # --- optional |v|-correction: flip sign for v<0 (msb == 0) ---
    if use_abs_v:
        # For v<0 (msb==0), we want theta_abs = -theta_base,
        # so add a correction of -2*theta_base when msb==0.
        if abs(theta_off) > 1e-12:
            qc.x(v_msb)  # now v_msb==1 encodes "v<0"
            if j > 1:
                controls = x_ctrls + [v_msb]
                qc.mcrz(-2 * theta_off, controls, x_top)
            else:
                qc.crz(-2 * theta_off, v_msb, x_top)
            qc.x(v_msb)

        for r, th_r in enumerate(theta_bits):
            if abs(th_r) < 1e-12:
                continue
            vq = v_qubits[r]

            # Skip MSB; its correction coefficient is effectively zero
            if vq == v_msb:
                continue

            qc.x(v_msb)
            if j > 1:
                controls = x_ctrls + [vq, v_msb]
                qc.mcrz(-2 * th_r, controls, x_top)
            else:
                qc.mcrz(-2 * th_r, [vq, v_msb], x_top)
            qc.x(v_msb)

    # --- uncompute scaffold ---
    if xgate and j > 1:
        qc.x(x_ctrls)
    qc.h(x_top)
    if lam != 0:
        qc.p(-lam, x_top)
    if j > 1:
        qc.cx(x_top, x_ctrls)

    return qc

def cvV1(theta_off, theta_bits, phi_off, phi_bits, n_x, n_v):
    """
    p-controlled, velocity-dependent V1 block on X⊗V, using |v| in H1:

    - W_j and U_1^(1) implement angles ∝ |v|
      via vWj(..., use_abs_v=True).
    - The P-qubit phase (diagonal part) is also made proportional to |v|
      using the MSB of the V-register as a sign bit.

    Layout inside this gate:
        P-qubit : 0
        X-qubits: 1 .. n_x
        V-qubits: 1+n_x .. 1+n_x+n_v-1
    """
    n_tot = 1 + n_x + n_v   # P + X + V
    qc = QuantumCircuit(n_tot, name='$cvV_1$')

    P = 0
    v_start  = 1 + n_x
    v_qubits = list(range(v_start, v_start + n_v))
    v_msb    = v_qubits[-1]   # sign bit

    # --- base velocity-dependent phase on P (linear in v) ---
    if abs(phi_off) > 1e-12:
        qc.p(phi_off, P)
    for r, ph_r in enumerate(phi_bits):
        if abs(ph_r) < 1e-12:
            continue
        vq = v_qubits[r]
        qc.crz(ph_r, vq, P)

    # --- correct phase to depend on |v| instead of v ---
    # For v>0: keep phi_base; for v<0: flip sign => add -2*phi_base when msb==0.
    if abs(phi_off) > 1e-12:
        qc.x(v_msb)
        qc.crz(-2 * phi_off, v_msb, P)
        qc.x(v_msb)
    for r, ph_r in enumerate(phi_bits):
        if abs(ph_r) < 1e-12:
            continue
        vq = v_qubits[r]

        # Skip MSB in |v|-correction
        if vq == v_msb:
            continue

        qc.x(v_msb)
        qc.mcrz(-2 * ph_r, [vq, v_msb], P)
        qc.x(v_msb)

    # --- controlled W-ladder (j = 1..n_x), |v|-dependent ---
    for j in range(1, n_x + 1):
        base_wj = vWj(
            n_x=n_x,
            n_v=n_v,
            j=j,
            theta_off=theta_off,
            theta_bits=theta_bits,
            lam=0.0,
            xgate=False,
            use_abs_v=True,
            name=f"vW{j}"
        ).to_gate()

        cWj = base_wj.control()  # single P control
        qc.append(cWj, [P] + list(range(1, n_tot)))

    # --- controlled U_1^(1) block (full X, xgate=True, |v|) ---
    base_u1 = vWj(
        n_x=n_x,
        n_v=n_v,
        j=n_x,
        theta_off=theta_off,
        theta_bits=theta_bits,
        lam=0.0,
        xgate=True,
        use_abs_v=True,
        name='$U_1^{(1)}$'
    ).to_gate()
    cU1 = base_u1.control()
    qc.append(cU1, [P] + list(range(1, n_tot)))

    return qc

def inverse_cvV1(theta_off, theta_bits, phi_off, phi_bits, n_x, n_v):
    """
    Inverse of cvV1: p-controlled, velocity-dependent V1† on X⊗V,
    with the same |v| semantics.

    To obtain the actual adjoint of cvV1(theta_off,theta_bits,
    phi_off,phi_bits, ...), call

        inverse_cvV1(-theta_off, -theta_bits, -phi_off, -phi_bits, ...)

    which negates both the W-angles and the P-phases.
    """
    n_tot = 1 + n_x + n_v  # P + X + V
    qc = QuantumCircuit(n_tot, name='$icvV_1$')

    P = 0
    v_start  = 1 + n_x
    v_qubits = list(range(v_start, v_start + n_v))
    v_msb    = v_qubits[-1]

    # --- controlled U_1^(1)† (implemented by negative angles, |v|) ---
    base_u1 = vWj(
        n_x=n_x,
        n_v=n_v,
        j=n_x,
        theta_off=theta_off,
        theta_bits=theta_bits,
        lam=0.0,
        xgate=True,
        use_abs_v=True,
        name='$U_1^{(1)}$'
    ).to_gate()
    cU1 = base_u1.control()
    qc.append(cU1, [P] + list(range(1, n_tot)))

    # --- controlled W-ladder†, reversed order, |v| ---
    for j in reversed(range(1, n_x + 1)):
        base_wj = vWj(
            n_x=n_x,
            n_v=n_v,
            j=j,
            theta_off=theta_off,
            theta_bits=theta_bits,
            lam=0.0,
            xgate=False,
            use_abs_v=True,
            name=f"vW{j}"
        ).to_gate()
        cWj = base_wj.control()
        qc.append(cWj, [P] + list(range(1, n_tot)))

    # --- P-phase† with |v| (caller passes phi_off,phi_bits already negated) ---
    if abs(phi_off) > 1e-12:
        qc.p(phi_off, P)
    for r, ph_r in enumerate(phi_bits):
        if abs(ph_r) < 1e-12:
            continue
        vq = v_qubits[r]
        qc.crz(ph_r, vq, P)

    if abs(phi_off) > 1e-12:
        qc.x(v_msb)
        qc.crz(-2 * phi_off, v_msb, P)
        qc.x(v_msb)
    for r, ph_r in enumerate(phi_bits):
        if abs(ph_r) < 1e-12:
            continue
        vq = v_qubits[r]

        # Skip MSB in |v|-correction
        if vq == v_msb:
            continue

        qc.x(v_msb)
        qc.mcrz(-2 * ph_r, [vq, v_msb], P)
        qc.x(v_msb)

    return qc

def vV2(theta_off, theta_bits, n_x, n_v):
    """
    Velocity-dependent V2 block on X⊗V (H2 part).

    This mirrors Hu's V2:
        - W-ladder with lam = -π/2,
        - U_2^(1) with lam = +π/2 and xgate=True.

    No P-phase here: the diagonal term is handled in H1.
    """
    n_tot = n_x + n_v  # X + V
    qc = QuantumCircuit(n_tot, name='$vV_2$')

    # W-ladder j=1..n_x with lam=-π/2
    for j in range(1, n_x + 1):
        base_wj = vWj(
            n_x=n_x,
            n_v=n_v,
            j=j,
            theta_off=theta_off,
            theta_bits=theta_bits,
            lam=-np.pi/2,
            xgate=False,
            use_abs_v=False,
            name=f"vW{j}"
        ).to_gate()
        qc.append(base_wj, list(range(n_tot)))

    # U_2^(1) with lam=+π/2, xgate=True
    base_u2 = vWj(
        n_x=n_x,
        n_v=n_v,
        j=n_x,
        theta_off=theta_off,
        theta_bits=theta_bits,
        lam=np.pi/2,
        xgate=True,
        use_abs_v=False,
        name='$U_2^{(1)}$'
    ).to_gate()
    qc.append(base_u2, list(range(n_tot)))

    return qc

# ============================================
# Block 2
# ============================================

def eWj(n_v, nE, j, theta_off, theta_bits, lam, vgate, use_abs_E, name=None):
    """
    Hu/Sato-style W_j gate acting on V⊗E, where the rotation angle is linear in E bits.

    Layout inside this gate (total n_v + nE qubits):
        V-qubits: 0 .. n_v-1
        E-qubits: n_v .. n_v+nE-1  (E_msb = last one)

    'vgate' plays the same role as 'xgate' in your vWj.
    If use_abs_E=True, we flip angles for negative E (two's complement => E_msb==1).
    """
    n_tot = n_v + nE
    qc = QuantumCircuit(n_tot, name=(name or f"eW{j}"))

    # V scaffold indices
    v_top   = j - 1
    v_ctrls = list(range(v_top))     # empty when j=1

    # E indices
    e_start  = n_v
    e_qubits = list(range(e_start, e_start + nE))
    e_msb    = e_qubits[-1]

    # --- compute scaffold on V: same pattern as vWj on X ---
    if j > 1:
        qc.cx(v_top, v_ctrls)
    if lam != 0:
        qc.p(lam, v_top)
    qc.h(v_top)
    if vgate and j > 1:
        qc.x(v_ctrls)

    # --- base linear-in-E part: theta_off + sum_r theta_bits[r] * b_r(E) ---
    if abs(theta_off) > 1e-12:
        if j > 1:
            qc.mcrz(theta_off, v_ctrls, v_top)
        else:
            qc.rz(theta_off, v_top)

    for r, th_r in enumerate(theta_bits):
        if abs(th_r) < 1e-12:
            continue
        eq = e_qubits[r]
        if j > 1:
            qc.mcrz(th_r, v_ctrls + [eq], v_top)
        else:
            qc.crz(th_r, eq, v_top)

    # --- optional |E|-correction for two's complement E (negative iff msb==1) ---
    # For E<0: want theta_abs = -theta_base => add correction -2*theta_base when e_msb==1.
    if use_abs_E:
        # offset correction
        if abs(theta_off) > 1e-12:
            if j > 1:
                qc.mcrz(-2 * theta_off, v_ctrls + [e_msb], v_top)
            else:
                qc.crz(-2 * theta_off, e_msb, v_top)

        # bit corrections
        for r, th_r in enumerate(theta_bits):
            if abs(th_r) < 1e-12:
                continue
            eq = e_qubits[r]

            if eq == e_msb:
                # condition (eq==1 and e_msb==1) reduces to just e_msb==1
                if j > 1:
                    qc.mcrz(-2 * th_r, v_ctrls + [e_msb], v_top)
                else:
                    qc.crz(-2 * th_r, e_msb, v_top)
            else:
                if j > 1:
                    qc.mcrz(-2 * th_r, v_ctrls + [eq, e_msb], v_top)
                else:
                    qc.mcrz(-2 * th_r, [eq, e_msb], v_top)

    # --- uncompute scaffold ---
    if vgate and j > 1:
        qc.x(v_ctrls)
    qc.h(v_top)
    if lam != 0:
        qc.p(-lam, v_top)
    if j > 1:
        qc.cx(v_top, v_ctrls)

    return qc

def cEV1(theta_off, theta_bits, phi_off, phi_bits, n_v, nE):
    """
    P-controlled, E-dependent V1 block on P⊗V⊗E, using |E| in H1.

    Layout inside this gate:
        P-qubit : 0
        V-qubits: 1 .. n_v
        E-qubits: 1+n_v .. 1+n_v+nE-1  (E_msb is last)
    """
    n_tot = 1 + n_v + nE
    qc = QuantumCircuit(n_tot, name="$cEV_1$")

    P = 0
    e_start = 1 + n_v
    e_qubits = list(range(e_start, e_start + nE))
    e_msb = e_qubits[-1]

    # --- base E-dependent phase on P (linear in E bits) ---
    if abs(phi_off) > 1e-12:
        qc.p(phi_off, P)
    for r, ph_r in enumerate(phi_bits):
        if abs(ph_r) < 1e-12:
            continue
        eq = e_qubits[r]
        qc.crz(ph_r, eq, P)

    # --- correct phase to depend on |E| instead of E (two's complement: msb==1 means negative) ---
    # For E<0: flip sign => add -2*phi_base when e_msb==1.
    if abs(phi_off) > 1e-12:
        qc.crz(-2 * phi_off, e_msb, P)

    for r, ph_r in enumerate(phi_bits):
        if abs(ph_r) < 1e-12:
            continue
        eq = e_qubits[r]
        if eq == e_msb:
            qc.crz(-2 * ph_r, e_msb, P)
        else:
            qc.mcrz(-2 * ph_r, [eq, e_msb], P)

    # --- controlled W-ladder (j=1..n_v), |E|-dependent ---
    for j in range(1, n_v + 1):
        base_wj = eWj(
            n_v=n_v, nE=nE, j=j,
            theta_off=theta_off, theta_bits=theta_bits,
            lam=0.0, vgate=False,
            use_abs_E=True,
            name=f"eW{j}"
        ).to_gate()
        qc.append(base_wj.control(), [P] + list(range(1, n_tot)))

    # --- controlled U_1^(1) (full V, vgate=True, |E|) ---
    base_u1 = eWj(
        n_v=n_v, nE=nE, j=n_v,
        theta_off=theta_off, theta_bits=theta_bits,
        lam=0.0, vgate=True,
        use_abs_E=True,
        name="$U_1^{(1)}$"
    ).to_gate()
    qc.append(base_u1.control(), [P] + list(range(1, n_tot)))

    return qc

def inverse_cEV1(theta_off, theta_bits, phi_off, phi_bits, n_v, nE):
    """
    Inverse of cEV1. As with your inverse_cvV1, pass NEGATED angles/phases to
    obtain the true adjoint:
        inverse_cEV1(-theta_off, -theta_bits, -phi_off, -phi_bits, ...)
    """
    n_tot = 1 + n_v + nE
    qc = QuantumCircuit(n_tot, name="$icEV_1$")

    P = 0
    e_start = 1 + n_v
    e_qubits = list(range(e_start, e_start + nE))
    e_msb = e_qubits[-1]

    # --- controlled U_1^(1)† (implemented by negative angles, |E|) ---
    base_u1 = eWj(
        n_v=n_v, nE=nE, j=n_v,
        theta_off=theta_off, theta_bits=theta_bits,
        lam=0.0, vgate=True,
        use_abs_E=True,
        name="$U_1^{(1)}$"
    ).to_gate()
    qc.append(base_u1.control(), [P] + list(range(1, n_tot)))

    # --- controlled W-ladder† in reverse order, |E| ---
    for j in reversed(range(1, n_v + 1)):
        base_wj = eWj(
            n_v=n_v, nE=nE, j=j,
            theta_off=theta_off, theta_bits=theta_bits,
            lam=0.0, vgate=False,
            use_abs_E=True,
            name=f"eW{j}"
        ).to_gate()
        qc.append(base_wj.control(), [P] + list(range(1, n_tot)))

    # --- P-phase† with |E| (caller passes phi_off,phi_bits already negated) ---
    if abs(phi_off) > 1e-12:
        qc.p(phi_off, P)
    for r, ph_r in enumerate(phi_bits):
        if abs(ph_r) < 1e-12:
            continue
        eq = e_qubits[r]
        qc.crz(ph_r, eq, P)

    if abs(phi_off) > 1e-12:
        qc.crz(-2 * phi_off, e_msb, P)

    for r, ph_r in enumerate(phi_bits):
        if abs(ph_r) < 1e-12:
            continue
        eq = e_qubits[r]
        if eq == e_msb:
            qc.crz(-2 * ph_r, e_msb, P)
        else:
            qc.mcrz(-2 * ph_r, [eq, e_msb], P)

    return qc

def eV2(theta_off, theta_bits, n_v, nE):
    """
    E-dependent V2 block on V⊗E (H2 part), analogous to your vV2.

    - W-ladder with lam = -π/2,
    - U_2^(1) with lam = +π/2 and vgate=True,
    - uses signed E (no |E| and no P-phase).
    """
    n_tot = n_v + nE
    qc = QuantumCircuit(n_tot, name="$eV_2$")

    # W-ladder j=1..n_v with lam=-π/2
    for j in range(1, n_v + 1):
        base_wj = eWj(
            n_v=n_v, nE=nE, j=j,
            theta_off=theta_off, theta_bits=theta_bits,
            lam=-np.pi/2, vgate=False,
            use_abs_E=False,
            name=f"eW{j}"
        ).to_gate()
        qc.append(base_wj, list(range(n_tot)))

    # U_2^(1) with lam=+π/2, vgate=True
    base_u2 = eWj(
        n_v=n_v, nE=nE, j=n_v,
        theta_off=theta_off, theta_bits=theta_bits,
        lam=np.pi/2, vgate=True,
        use_abs_E=False,
        name="$U_2^{(1)}$"
    ).to_gate()
    qc.append(base_u2, list(range(n_tot)))

    return qc
