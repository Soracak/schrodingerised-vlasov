"""
build_E_centered.py  —  Centered-grid fixed-point E(x) builder via Schrödingerization workflow

This file replaces the original "index polynomial" demo (e.g. E(j) = -j^3 + j with j=0..2^nx-1)
with the physically correct centered-grid, fixed-point polynomial used by the Schrödingerization
pipeline.

Key conceptual change
---------------------
Input register x stores an *index* j in unsigned binary:
    j ∈ {0, 1, ..., 2^nx - 1}

We convert j to a *centered coordinate* c in nx-bit two's complement:
    c = j - 2^(nx-1)   ∈ {-2^(nx-1), ..., 2^(nx-1)-1}

Two's-complement centering is implemented by:
    c <- x (copy) followed by flipping the MSB of c:
    c[msb] ^= 1
This maps the lower half of indices to negative coordinates and the upper half to nonnegative.

Physical polynomial (fixed-point)
---------------------------------
We build a signed fixed-point accumulator E_fp (two's complement, width nE):
    E_fp = alpha_fp * c + beta_fp * c^3
where alpha_fp and beta_fp are integer fixed-point coefficients (scaling handled upstream).

Important implementation detail:
- c is signed (two's complement).
- The cube pipeline uses unsigned multiplication, so we compute |c| first and then reapply sign:
    cabs = |c|
    cube = |c|^3
    if sign(c)=1: cube_low <- -cube_low  (two's complement)
  where cube_low denotes the low 3*nx bits that carry c^3 over the intended range.

Bit ordering convention
-----------------------
All registers use little-endian indexing:
    reg[0] is LSB, reg[-1] is MSB/sign bit for two's complement sources.
Measurement helper returns int(bitstring, 2) with NO bitstring reversal.

Main entry point
----------------
compute_E_poly_centered(nx, alpha_fp, beta_fp, S, ...)
builds the full centered E-polynomial circuit and returns either:
- QuantumCircuit only, or
- (QuantumCircuit, regs dict) if return_regs=True for testing/diagnostics.
"""


from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import MultiplierGate, CDKMRippleCarryAdder
from qiskit_aer import AerSimulator
from qiskit.circuit import Gate
from IPython.display import display

import random

#from build_E import build_cube_circuit_regs, uncompute_sq_regs, CDKMRippleCarryAdder

# ==========================
# Circuits

# ==========================
# Cache helpers
# ==========================
from functools import lru_cache

@lru_cache(maxsize=None)
def _cached_add_const_into_E_instr(nE: int, const: int, shift: int):
    """
    Memoized version of make_controlled_add_const_into_E_circuit(...).to_instruction().

    Cache key = (nE, const, shift). This is safe because the circuit structure depends
    only on these inputs (no external state).
    """
    circ = make_controlled_add_const_into_E_circuit(nE=nE, const=const, shift=shift)
    return circ.to_instruction()

@lru_cache(maxsize=None)
def _cached_cdkm_full_adder_instr(nE: int):
    """
    Reusable decomposed CDKM full ripple-carry adder instruction.

    Qubit order expected by the instruction:
        [cin] + W(nE) + E(nE) + [cout]
    """
    adder = CDKMRippleCarryAdder(num_state_qubits=nE, kind="full")
    return adder.decompose().to_instruction()

# @lru_cache(maxsize=None)
# def _cached_multiplier_instr(n: int):
#     """
#     Reusable decomposed MultiplierGate instruction.

#     Qubit order expected:
#         a(n) + b(n) + out(2n)
#     """
#     return MultiplierGate(num_state_qubits=n).decompose().to_instruction()


@lru_cache(maxsize=None)
def _cached_multiplier_instr(n: int):
    """
    Cached decomposed MultiplierGate instruction.

    The returned instruction expects qubits in the same order as MultiplierGate:
      [a(n)] + [b(n)] + [out(2n)] (+ any ancillas if the implementation uses them)

    We build a wrapper QuantumCircuit, append the gate, then decompose the circuit.
    """
    g = MultiplierGate(num_state_qubits=n)

    # Wrap the gate into a circuit so we can call .decompose()
    qc = QuantumCircuit(g.num_qubits, name=f"mul_{n}")
    qc.append(g, qc.qubits)

    # Decompose the circuit (do more reps if you want it flatter)
    qc_dec = qc.decompose()  # or qc.decompose(reps=5)

    return qc_dec.to_instruction()

# ==========================
# Circuits
# ==========================

def build_cube_circuit_regs(qc: QuantumCircuit, *, x, xcopy, tmp, sq, cube, nx: int):
    """
    Computes x^2 into sq (2nx bits) and x^3 into cube (4nx bits) using multiplier gates.
    Leaves:
      x unchanged,
      xcopy = x,
      tmp = 0,
      sq = x^2,
      cube = x^3
    """
    # xcopy = x, tmp = x
    for i in range(nx):
        qc.cx(x[i], xcopy[i])
    
    for i in range(nx):
        qc.cx(xcopy[i], tmp[i])

    # sq = xcopy * tmp  (nx-bit multiply -> 2nx product)
    # mult_n = MultiplierGate(num_state_qubits=nx)
    # qc.append(mult_n, [*xcopy, *tmp, *sq])
    
    mult_n_instr = _cached_multiplier_instr(nx)
    qc.append(mult_n_instr, [*xcopy, *tmp, *sq])

    # tmp <- 0 (uncopy x)
    for i in reversed(range(nx)):
        qc.cx(xcopy[i], tmp[i])

    # Multiply (2nx-bit) sq by (2nx-bit) xpad -> cube (4nx bits)
    # Here we use xcopy (nx bits) as the low nx bits of the 2nx multiplicand and tmp as padding workspace
    # tmp is currently 0, so treating [xcopy,tmp] as a 2nx-bit number equals x.
    # mult_2n = MultiplierGate(num_state_qubits=2 * nx)
    # qc.append(mult_2n, [*xcopy, *tmp, *sq, *cube])
    
    mult_2n_instr = _cached_multiplier_instr(2 * nx)
    qc.append(mult_2n_instr, [*xcopy, *tmp, *sq, *cube])

    return qc

def uncompute_sq_regs(qc: QuantumCircuit, *, x, tmp, sq, nx: int):
    """
    Uncomputes sq from x and tmp (tmp must be 0 on entry).
    After: sq=0 and tmp=0, x unchanged.
    """
    # tmp = x
    for i in range(nx):
        qc.cx(x[i], tmp[i])

    # mult_n = MultiplierGate(num_state_qubits=nx)
    # qc.append(mult_n.inverse(), [*x, *tmp, *sq])  # sq -> 0
    
    mult_n_instr = _cached_multiplier_instr(nx)
    qc.append(mult_n_instr.inverse(), [*x, *tmp, *sq])  # sq -> 0

    # tmp -> 0
    for i in reversed(range(nx)):
        qc.cx(x[i], tmp[i])

    return qc

def compute_nE_centered(nx: int, alpha_fp: int, beta_fp: int) -> int:
    """
    Conservative 2's-complement width planning for centered coordinate c in [-2^(nx-1), 2^(nx-1)-1].
    Matches the design note. Example: nx=3, alpha_fp=48, beta_fp=-27 => nE=12.
    """
    cmax = 1 << (nx - 1)
    max_abs = abs(int(alpha_fp)) * cmax + abs(int(beta_fp)) * (cmax ** 3)
    n_twos = 1 if max_abs == 0 else int(max_abs).bit_length() + 1  # +1 sign bit
    return max(3 * nx + 1, n_twos)


def _ctrl_twos_complement_inplace_mcx(qc: QuantumCircuit, reg, ctrl, carry=None):
    """
    If ctrl=1: reg <- (~reg + 1)  (two's complement negation).
    If ctrl=0: no-op.

    Correct reversible increment: apply carry-propagation from MSB -> 1
    using the *pre-toggle* lower bits, then toggle bit 0 last.
    """
    n = len(reg)

    # 1) conditional bitwise NOT
    for i in range(n):
        qc.cx(ctrl, reg[i])

    # 2) conditional +1 (increment) — DO MSB->LSB (except bit0), then bit0 last
    for i in reversed(range(1, n)):
        controls = [ctrl] + [reg[k] for k in range(i)]  # ctrl & reg[0..i-1] (pre-toggle!)
        qc.mcx(controls, reg[i])

    qc.cx(ctrl, reg[0])  # toggle LSB last

def _ctrl_twos_complement_inplace(qc, reg, ctrl, carry):
    """
    If ctrl=1: reg <- (~reg + 1)
    If ctrl=0: no-op
    carry: >= len(reg) clean ancillas (all 0 in/out)

    This version keeps the +1 effect while uncomputing carry ancillas.
    """
    n = len(reg)
    if len(carry) < n:
        raise ValueError("carry too short")

    # 1) conditional NOT
    for i in range(n):
        qc.cx(ctrl, reg[i])
    qc.barrier()

    # 2) conditional +1 via carry chain, but keep the sum
    # carry[0] = ctrl
    qc.cx(ctrl, carry[0])

    # (a) compute carry bits forward: carry[i+1] = carry[i] & reg[i]
    for i in range(n - 1):
        qc.ccx(carry[i], reg[i], carry[i + 1])

    # (b) apply sum: reg[i] ^= carry[i]
    for i in range(n):
        qc.cx(carry[i], reg[i])
    qc.barrier()

    # (c) uncompute carries backward WITHOUT destroying the sum
    for i in reversed(range(n - 1)):
        # Temporarily restore original reg[i]
        qc.cx(carry[i], reg[i])               # reg[i] back to original
        qc.ccx(carry[i], reg[i], carry[i + 1])# uncompute carry[i+1]
        qc.cx(carry[i], reg[i])               # restore sum bit

    # clear carry[0]
    qc.cx(ctrl, carry[0])


def make_ctrl_twos_complement_block(n: int, name: str = None) -> Gate:
    """
    Returns a Gate that performs:
      if ctrl=1: reg <- (~reg + 1)
      if ctrl=0: no-op
    using a carry-chain incrementer.

    Qubits order in the block:
      [ctrl] + reg[0..n-1] + carry[0..n-1]
    """
    ctrl  = QuantumRegister(1, "ctrl")
    reg   = QuantumRegister(n, "r")
    carry = QuantumRegister(n, "k")

    qc = QuantumCircuit(ctrl, reg, carry, name=name or f"NEG2C_{n}")

    # conditional NOT
    for i in range(n):
        qc.cx(ctrl[0], reg[i])

    # carry[0] = ctrl
    qc.cx(ctrl[0], carry[0])

    # forward carry compute: carry[i+1] = carry[i] & reg[i]
    for i in range(n - 1):
        qc.ccx(carry[i], reg[i], carry[i + 1])

    # apply sum: reg[i] ^= carry[i]
    for i in range(n):
        qc.cx(carry[i], reg[i])

    # uncompute carries without destroying sum
    for i in reversed(range(n - 1)):
        qc.cx(carry[i], reg[i])                # temporarily restore original reg[i]
        qc.ccx(carry[i], reg[i], carry[i + 1]) # uncompute carry[i+1]
        qc.cx(carry[i], reg[i])                # restore sum bit

    # clear carry[0]
    qc.cx(ctrl[0], carry[0])

    return qc.to_gate()

def make_ctrl_twos_complement_circuit(n: int, name: str = None) -> QuantumCircuit:
    """
    Build a reversible controlled two's-complement negation circuit.
    
    Action:
        if ctrl = 1: reg <- (~reg + 1) mod 2^n
        if ctrl = 0: reg unchanged
    
    Ancilla:
        carry must be initialized to |0...0> and is restored to |0...0>.
    Qubit order:
        [ctrl] + reg[0:n] + carry[0:n]
    """
    ctrl  = QuantumRegister(1, "ctrl")
    reg   = QuantumRegister(n, "r")
    carry = QuantumRegister(n, "k")

    qc = QuantumCircuit(ctrl, reg, carry, name=name or f"NEG2C_{n}")

    # conditional NOT
    for i in range(n):
        qc.cx(ctrl[0], reg[i])
    qc.barrier()

    # carry[0] = ctrl
    qc.cx(ctrl[0], carry[0])

    # forward carry compute
    for i in range(n - 1):
        qc.ccx(carry[i], reg[i], carry[i + 1])

    # apply sum: reg ^= carry
    for i in range(n):
        qc.cx(carry[i], reg[i])
    qc.barrier()
    
    # uncompute carries without destroying sum
    for i in reversed(range(n - 1)):
        qc.cx(carry[i], reg[i])
        qc.ccx(carry[i], reg[i], carry[i + 1])
        qc.cx(carry[i], reg[i])

    qc.cx(ctrl[0], carry[0])
    return qc

# qc_neg = make_ctrl_twos_complement_circuit(4, name="NEG_IF_SIGN_4")
# fig = qc_neg.draw(output="mpl", fold=140, idle_wires=False, scale=0.7)
# display(fig)

def _controlled_add_const_into_E(qc: QuantumCircuit, *, const: int, shift: int, ctrl, E, W, cin, cout, nE: int):
    """
    If ctrl=1: E += (const << shift)   (mod 2^nE, two's complement)
    If ctrl=0: no-op

    Uses W (nE qubits) as workspace and CDKM full adder [cin, W, E, cout].
    Works for negative const by loading W = ~(addend) and setting cin=1 iff ctrl=1.
    """
    if const == 0:
        return

    # sanity
    if len(W) != nE or len(E) != nE:
        raise ValueError("W/E length mismatch")
    if len(cin) != 1 or len(cout) != 1:
        raise ValueError("cin/cout must be 1-qubit regs")

    neg = (const < 0)
    mag = -const if neg else const

    # --- Load W (conditionally) ---
    if neg:
        # W := all-ones iff ctrl=1
        for q in W:
            qc.cx(ctrl, q)
        # XOR in the mag bits (so positions with mag-bit=1 become 0, i.e. NOT(addend))
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl, W[pos])
            b >>= 1
            bitpos += 1
        # cin := 1 iff ctrl=1
        qc.cx(ctrl, cin[0])
    else:
        # W := (mag<<shift) iff ctrl=1
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl, W[pos])
            b >>= 1
            bitpos += 1
        # cin stays 0

    # --- Add: E += W + cin ---
    adder = CDKMRippleCarryAdder(num_state_qubits=nE, kind="full")
    qc.append(adder, [cin[0], *W, *E, cout[0]])

    # --- Uncompute W and cin ---
    if neg:
        qc.cx(ctrl, cin[0])
        # undo XOR mag bits
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl, W[pos])
            b >>= 1
            bitpos += 1
        # undo all-ones
        for q in W:
            qc.cx(ctrl, q)
    else:
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl, W[pos])
            b >>= 1
            bitpos += 1
        # cin already 0

def _controlled_add_const_into_E_block(nE: int, const: int, shift: int, name: str = None):
    """
    Returns a Gate implementing:
        if ctrl=1: E <- E + (const << shift)  (mod 2^nE, two's complement)
        if ctrl=0: no-op

    Qubit order for the returned gate:
        [ctrl] + E[0..nE-1] + W[0..nE-1] + [cin] + [cout]

    Requirements:
      - W is clean ancilla (all 0 in/out)
      - cin, cout are clean ancillas (all 0 in/out)
    """
    # Handle trivial case as a tiny identity gate (still plottable)
    if const == 0:
        ctrl = QuantumRegister(1, "ctrl")
        E    = QuantumRegister(nE, "E")
        W    = QuantumRegister(nE, "W")
        cin  = QuantumRegister(1, "cin")
        cout = QuantumRegister(1, "cout")
        qc = QuantumCircuit(ctrl, E, W, cin, cout, name=name or f"ADD0_{nE}")
        return qc.to_gate()

    ctrl = QuantumRegister(1, "ctrl")
    E    = QuantumRegister(nE, "E")
    W    = QuantumRegister(nE, "W")
    cin  = QuantumRegister(1, "cin")
    cout = QuantumRegister(1, "cout")

    default_name = f"ADD({const}<<{shift})"
    qc = QuantumCircuit(ctrl, E, W, cin, cout, name=name or default_name)

    neg = (const < 0)
    mag = -const if neg else const

    # ---- Load W conditionally so W = (const<<shift) mod 2^nE (two's comp trick for neg) ----
    if neg:
        # W := all-ones iff ctrl=1  (so we can make W = ~(mag<<shift))
        for q in W:
            qc.cx(ctrl[0], q)

        # Toggle the 1-bits of mag<<shift off (gives bitwise NOT of the shifted magnitude)
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl[0], W[pos])
            b >>= 1
            bitpos += 1

        # cin := 1 iff ctrl=1  (so add (~A + 1) = -A mod 2^nE)
        qc.cx(ctrl[0], cin[0])

    else:
        # W := (mag<<shift) iff ctrl=1
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl[0], W[pos])
            b >>= 1
            bitpos += 1
        # cin stays 0

    # ---- Add: E += W + cin (mod 2^nE) ----
    adder = CDKMRippleCarryAdder(num_state_qubits=nE, kind="full")
    qc.append(adder, [cin[0], *W, *E, cout[0]])

    # ---- Uncompute W and cin ----
    if neg:
        qc.cx(ctrl[0], cin[0])

        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl[0], W[pos])
            b >>= 1
            bitpos += 1

        for q in W:
            qc.cx(ctrl[0], q)

    else:
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl[0], W[pos])
            b >>= 1
            bitpos += 1

    return qc.to_instruction() 



def make_controlled_add_const_into_E_circuit(nE: int, const: int, shift: int, name: str = None) -> QuantumCircuit:
    ctrl = QuantumRegister(1, "ctrl")
    E    = QuantumRegister(nE, "E")
    W    = QuantumRegister(nE, "W")
    cin  = QuantumRegister(1, "cin")
    cout = QuantumRegister(1, "cout")

    qc = QuantumCircuit(ctrl, E, W, cin, cout, name=name or f"ADD({const}<<{shift})")

    if const == 0:
        return qc

    neg = const < 0
    mag = -const if neg else const

    # Load W (controlled)
    if neg:
        for q in W:
            qc.cx(ctrl[0], q)
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl[0], W[pos])
            b >>= 1
            bitpos += 1
        qc.cx(ctrl[0], cin[0])
    else:
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl[0], W[pos])
            b >>= 1
            bitpos += 1

    # ---- Inline CDKM gates ----
    # adder = CDKMRippleCarryAdder(num_state_qubits=nE, kind="full")
    # adder_circ = adder.decompose()  # adder-only
    # qc.compose(adder_circ, qubits=[cin[0], *W, *E, cout[0]], inplace=True)
    # qc.barrier()
    
    # ---- Inline CDKM gates (cached) ----
    adder_instr = _cached_cdkm_full_adder_instr(nE)
    qc.append(adder_instr, [cin[0], *W, *E, cout[0]])
    qc.barrier()
    
    # Uncompute W and cin
    if neg:
        qc.cx(ctrl[0], cin[0])
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl[0], W[pos])
            b >>= 1
            bitpos += 1
        for q in W:
            qc.cx(ctrl[0], q)
    else:
        b = mag
        bitpos = 0
        while b:
            if b & 1:
                pos = shift + bitpos
                if 0 <= pos < nE:
                    qc.cx(ctrl[0], W[pos])
            b >>= 1
            bitpos += 1

    return qc



def _add_const_times_twos_src(qc: QuantumCircuit, *, const: int, src, n_src: int, E, W, cin, cout, nE: int):
    """
    Add const * src_twos into E, where src is interpreted as two's complement of width n_src:

      src_twos = sum_{i=0..n_src-2} src[i]*2^i  - src[msb]*2^(n_src-1)

    So:
      E += const * sum_{i<msb} src[i]*2^i  + (-const)*src[msb]*2^(n_src-1)

    Implemented as controlled adds of classical constants.
    """
    if const == 0:
        return
    if n_src != len(src):
        raise ValueError("n_src mismatch")
   
    msb = n_src - 1

    # for i in range(msb):

    #     add_block = make_controlled_add_const_into_E_circuit(
    #         nE=nE, 
    #         const=const, 
    #         shift=i
    #     )
    #     qc.append(add_block, [src[i]] + list(E) + list(W) + [cin[0]] + [cout[0]])

    for i in range(msb):
        add_instr = _cached_add_const_into_E_instr(nE, const, i)
        qc.append(add_instr, [src[i]] + list(E) + list(W) + [cin[0]] + [cout[0]])


    # add_block = make_controlled_add_const_into_E_circuit(
    #     nE=nE, 
    #     const=-const, 
    #     shift=msb
    # )    
    # qc.append(add_block, [src[msb]] + list(E) + list(W) + [cin[0]] + [cout[0]])
    
    add_instr = _cached_add_const_into_E_instr(nE, -const, msb)
    qc.append(add_instr, [src[msb]] + list(E) + list(W) + [cin[0]] + [cout[0]])
    
    qc.barrier()


def compute_E_poly_centered(nx: int, alpha_fp: int, beta_fp: int, S: int, 
                            *, xval=None, return_regs=False, debug_cache=True):
    """
    Centered version:
      c = j - 2^(nx-1)   (encoded by flipping MSB after copying x)
      E_fp = alpha_fp*c + beta_fp*c^3   (two's complement, width nE)

    Flow matches the design note:
      1) c = x; c[msb] ^= 1
      2) cabs = |c|  via controlled two's complement on cabs if sign=1
      3) cube = |c|^3  (unsigned multiplier path)
      4) if sign=1, negate cube_low (3nx bits) to make it signed c^3
      5) E += alpha_fp*c + beta_fp*c^3 using signed-source accumulation
    """
    nE = compute_nE_centered(nx, alpha_fp, beta_fp)

    # Registers
    x     = QuantumRegister(nx,   "x")        # input index j
    c     = QuantumRegister(nx,   "c")        # centered two's complement
    cabs  = QuantumRegister(nx,   "cabs")     # unsigned |c|

    # cube workspace (reuses your existing cube builder)
    xcopy = QuantumRegister(nx,   "xcopy")
    tmp   = QuantumRegister(nx,   "tmp")
    sq    = QuantumRegister(2*nx, "sq")
    cube  = QuantumRegister(4*nx, "cube")

    # output + adder workspace
    E     = QuantumRegister(nE,   "E")
    w_msb = QuantumRegister(1,    "w_msb")
    cin   = QuantumRegister(1,    "cin")
    cout  = QuantumRegister(1,    "cout")

    # dedicated carries for controlled two's complement
    carry_nx  = QuantumRegister(nx,    "carry_nx")
    carry_3nx = QuantumRegister(3*nx,  "carry_3nx")
    
    # Quantum circuit
    qc = QuantumCircuit(
        x, c, cabs,
        xcopy, tmp, sq, cube,
        E, w_msb, cin, cout,
        carry_nx, carry_3nx,
        name="E_poly_centered"
    )
    
    # Prepare twos blocks
    neg_nx  = make_ctrl_twos_complement_circuit(nx,   name="NEG_IF_SIGN_nx")
    neg_3nx = make_ctrl_twos_complement_circuit(3*nx, name="NEG_IF_SIGN_3nx")
    
    # Optional basis-state init for testing
    if xval is not None:
        if xval < 0 or xval >= (1 << nx):
            raise ValueError(f"xval out of range for nx={nx}")
        for i in range(nx):
            if (xval >> i) & 1:
                qc.x(x[i])
        qc.barrier()

    # ---- 1) c = x; then flip MSB to center ----
    for i in range(nx):
        qc.cx(x[i], c[i])
    qc.x(c[nx - 1])  # c[msb] ^= 1  => centered coordinate

    sign = c[nx - 1]  # sign control (1 means negative in two's complement)

    # ---- 2) cabs = c; if sign==1 then cabs = -cabs ----
    for i in range(nx):
        qc.cx(c[i], cabs[i])
    #_ctrl_twos_complement_inplace(qc, cabs, sign, carry_nx)
    qc.append(neg_nx, [sign] + list(cabs) + list(carry_nx))

    # ---- 3) cube = |c|^3 using your cube builder (unsigned) ----
    build_cube_circuit_regs(qc, x=cabs, xcopy=xcopy, tmp=tmp, sq=sq, cube=cube, nx=nx)

    # clear sq/tmp so they can become W
    uncompute_sq_regs(qc, x=xcopy, tmp=tmp, sq=sq, nx=nx)
    qc.barrier()

    # ---- 4) if sign==1 negate cube_low to form signed c^3 ----
    cube_low = cube[: 3 * nx]  # low 3nx bits hold |c|^3 for |c| <= 2^(nx-1)
    qc.append(neg_3nx, [sign] + list(cube_low) + list(carry_3nx))
    qc.barrier()
    
    # ---- 5) E += alpha_fp*c + beta_fp*c^3 (signed sources) ----
    W_core = [*tmp, *sq, w_msb[0]]  # length 3nx+1
    n_pad = nE - len(W_core)
    if n_pad > 0:
        wpad = QuantumRegister(n_pad, "wpad")
        qc.add_register(wpad)
        W = W_core + list(wpad)
    else:
        wpad = None
        W = W_core

    # signed-source accumulation
    _add_const_times_twos_src(qc, const=int(alpha_fp), src=c,        n_src=nx,    E=E, W=W, cin=cin, cout=cout, nE=nE)
    _add_const_times_twos_src(qc, const=int(beta_fp),  src=cube_low, n_src=3*nx, E=E, W=W, cin=cin, cout=cout, nE=nE)

    if debug_cache:
        print("[cache] add_const_into_E:", _cached_add_const_into_E_instr.cache_info())
        print("[cache] cdkm_full_adder:", _cached_cdkm_full_adder_instr.cache_info())
        print("[cache] multiplier:", _cached_multiplier_instr.cache_info())
        
    if return_regs:
        regs = {
            "x": x, "c": c, "cabs": cabs,
            "xcopy": xcopy, "tmp": tmp, "sq": sq, "cube": cube,
            "E": E, "W": W, "w_msb": w_msb, "wpad": wpad,
            "cin": cin, "cout": cout,
            "carry_nx": carry_nx, "carry_3nx": carry_3nx,
            "nE": nE, "S": S
        }
        return qc, regs
    return qc


# ---- Measurement helper  ----
def measure_qubits_value(qc: QuantumCircuit, qubits, name="q", shots=1, seed=1):
    qc_m = qc.copy()
    cr = ClassicalRegister(len(qubits), f"m_{name}")  # avoid name clashes with qregs
    qc_m.add_register(cr)
    qc_m.measure(qubits, cr)

    sim = AerSimulator(method="matrix_product_state", seed_simulator=seed)
    tqc = qc_m.decompose(reps=12)
    tqc = transpile(tqc, basis_gates=["u", "cx"], optimization_level=0)  # no backend target limits
    res = sim.run(tqc, shots=shots).result()
    counts = res.get_counts()
    bitstr = max(counts, key=counts.get)
    return int(bitstr, 2), bitstr, counts  # NO reverse


# ---- Quick classical reference for testing ----
def twos_to_int(u: int, n: int) -> int:
    """Interpret u (0..2^n-1) as signed two's complement."""
    if (u >> (n - 1)) & 1:
        return u - (1 << n)
    return u


def int_to_twos(z: int, n: int) -> int:
    """Encode signed int z into n-bit two's complement (mod 2^n)."""
    return z & ((1 << n) - 1)

# ==========================
# Test harness
# ==========================
if __name__ == "__main__":
    def classical_centered_ref(nx: int, alpha_fp: int, beta_fp: int, j: int, nE: int):
        """
        Classical reference for centered scheme:
          c = j - 2^(nx-1)
          E = alpha_fp*c + beta_fp*c^3   (wrapped into nE-bit two's complement)
        Also returns expected encodings for:
          c in nx-bit two's complement
          c^3 in 3nx-bit two's complement (low 3nx bits)
        """
        c = int(j) - (1 << (nx - 1))
        c_enc = int_to_twos(c, nx)
    
        c3 = c**3
        c3_enc = int_to_twos(c3, 3 * nx)
    
        E = int(alpha_fp) * c + int(beta_fp) * c3
        E_enc = int_to_twos(E, nE)
    
        return c, c_enc, c3, c3_enc, E, E_enc
    
    
    def test_centered_all_j(nx: int, alpha_fp: int, beta_fp: int, S: int, *, verbose=False, stop_on_first=True):
        """
        Exhaustive test over all j for given nx.
        If verbose=True: prints a line for EVERY j (PASS/FAIL).
        If stop_on_first=False: continues after failures and prints them all.
        Always returns: (ok: bool, failures: list)
        """
        nE = compute_nE_centered(nx, alpha_fp, beta_fp)
        print("nE:", nE)
        failures = []
    
        for j in range(1 << nx):
            qc, regs = compute_E_poly_centered(nx, alpha_fp, beta_fp, S, xval=j, return_regs=True)
    
            # Measure c
            c_u, c_bits, _ = measure_qubits_value(qc, regs["c"], name=f"c_j{j}", shots=1)
            # Measure cube_low
            cube_low_qubits = regs["cube"][: 3 * nx]
            c3_u, c3_bits, _ = measure_qubits_value(qc, cube_low_qubits, name=f"c3_j{j}", shots=1)
            # Measure E
            E_u, E_bits, _ = measure_qubits_value(qc, regs["E"], name=f"E_j{j}", shots=1)
    
            # Classical expected
            c, c_enc, c3, c3_enc, E, E_enc = classical_centered_ref(nx, alpha_fp, beta_fp, j, nE)
    
            ok = (c_u == c_enc) and (c3_u == c3_enc) and (E_u == E_enc)
    
            if ok:
                if verbose:
                    print(
                        f"PASS j={j:>2} | "
                        f"c={twos_to_int(c_u,nx):>4} ({c_bits}) | "
                        f"c^3={twos_to_int(c3_u,3*nx):>6} ({c3_bits}) | "
                        f"E={twos_to_int(E_u,nE):>8} ({E_bits})"
                    )
            else:
                failures.append((j, c_u, c_enc, c3_u, c3_enc, E_u, E_enc))
    
                print(f"\nFAIL j={j}")
                print(f"  c    got {c_u} ({c_bits}) expected {c_enc} ({format(c_enc, f'0{nx}b')})  int(c)={c}")
                print(f"  c^3  got {c3_u} ({c3_bits}) expected {c3_enc} ({format(c3_enc, f'0{3*nx}b')})  int(c^3)={c3}")
                print(f"  E    got {E_u} ({E_bits}) expected {E_enc} ({format(E_enc, f'0{nE}b')})  int(E)={E}")
    
                if stop_on_first:
                    return False, failures
    
        if failures:
            print(f"\nDONE with failures: nx={nx}, alpha_fp={alpha_fp}, beta_fp={beta_fp}, nE={nE}, fails={len(failures)}")
            return False, failures
    
        print(f"\nPASS: nx={nx}, alpha_fp={alpha_fp}, beta_fp={beta_fp}, nE={nE} (all j)")
        return True, []
    
    def random_stress(nx, S, trials=50, alpha_range=(-100,100), beta_range=(-50,50), seed=1):
        random.seed(seed)
        for t in range(trials):
            alpha_fp = random.randint(*alpha_range)
            beta_fp  = random.randint(*beta_range)
    
            ok, fails = test_centered_all_j(
                nx=nx, alpha_fp=alpha_fp, beta_fp=beta_fp, S=S,
                verbose=False,
                stop_on_first=False
            )
            if not ok:
                print(f"\nFAILED at trial {t}: alpha={alpha_fp}, beta={beta_fp}, fails={len(fails)}")
                # show first failure tuple for quick diagnosis
                j, c_u, c_enc, c3_u, c3_enc, E_u, E_enc = fails[0]
                print("first failing j =", j)
                return False
            print(f"trial {t:>2}: alpha={alpha_fp:>4}, beta={beta_fp:>4} -> PASS")
        print("\nAll random trials passed.")
        return True
    
    random_stress(nx=2, S=64, trials=1, alpha_range=(-200,200), beta_range=(-100,100), seed=7)
    

    # --- Production oracle test ---
    nx, alpha_fp, beta_fp, S = 3, 48, -27, 64
    
    ok, fails = test_centered_all_j(
        nx=nx,
        alpha_fp=alpha_fp,
        beta_fp=beta_fp,
        S=S,
        verbose=True,
        stop_on_first=False,
    )
    
    print("num fails:", len(fails))
    
        
