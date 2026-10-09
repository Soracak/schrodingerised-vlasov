# classical_schemes_1d1v.py
import numpy as np

def advect_x_1step_periodic(f, v_grid, dt, Lx):
    """
    ∂_t f + v ∂_x f = 0  (free-stream in x)
    semi-Lagrangian backtrace in x with periodic BC.
    f: (Nx, Nv)
    """
    Nx, Nv = f.shape
    dx = Lx / Nx
    j_grid = np.arange(Nx, dtype=float)

    f_new = np.empty_like(f)
    for kv in range(Nv):
        s = (v_grid[kv] * dt) / dx           # shift in x-index units
        j_src = (j_grid - s) % Nx            # backtrace (periodic)
        j0 = np.floor(j_src).astype(int)
        a = j_src - j0
        j1 = (j0 + 1) % Nx
        f_new[:, kv] = (1.0 - a) * f[j0, kv] + a * f[j1, kv]
    return f_new


def advect_v_1step_periodic(f, a_x, dt, Lv):
    """
    ∂_t f + a(x) ∂_v f = 0  (kick in v)
    semi-Lagrangian backtrace in v with periodic BC in v.
    f: (Nx, Nv), a_x: (Nx,)
    """
    Nx, Nv = f.shape
    dv = (2.0 * Lv) / Nv
    k_grid = np.arange(Nv, dtype=float)

    f_new = np.empty_like(f)
    for jx in range(Nx):
        s = (a_x[jx] * dt) / dv              # shift in v-index units
        k_src = (k_grid - s) % Nv            # backtrace (periodic)
        k0 = np.floor(k_src).astype(int)
        a = k_src - k0
        k1 = (k0 + 1) % Nv
        f_new[jx, :] = (1.0 - a) * f[jx, k0] + a * f[jx, k1]
    return f_new

def strang_step_classical(f, v_grid, a_x, dt, Lx, Lv):
    f = advect_x_1step_periodic(f, v_grid, 0.5 * dt, Lx)
    f = advect_v_1step_periodic(f, a_x, dt, Lv)
    f = advect_x_1step_periodic(f, v_grid, 0.5 * dt, Lx)
    return f

def evolve_strang_classical(f0, v_grid, a_x, dt, Lx, Lv, Nt):
    f = f0.copy()
    for _ in range(Nt):
        f = strang_step_classical(f, v_grid, a_x, dt, Lx, Lv)
    return f

def advect_x_1step_hu_symbol_fft_scaled(f, v_grid, dt, Lx, R):
    """
    Scheme-matched to your quantum Ux: apply exp(dt * Lx_op) along x for each v.
      lambda_x(kx) = -(|v|/(R*dx))*(1-cos kx) - i*(v/dx)*sin kx
    Periodic in x. FFT is taken along axis=0 (x-axis).
    f: (Nx, Nv), v_grid: (Nv,)
    """
    Nx, Nv = f.shape
    dx = Lx / Nx

    m = np.arange(Nx, dtype=float)
    k = 2.0 * np.pi * m / Nx
    sin_k = np.sin(k)
    one_minus_cos = 1.0 - np.cos(k)

    f_new = np.empty_like(f, dtype=complex)

    for kv in range(Nv):
        v = float(v_grid[kv])
        lam = -(abs(v) / (R * dx)) * one_minus_cos - 1j * (v / dx) * sin_k
        F = np.fft.fft(f[:, kv])
        f_new[:, kv] = np.fft.ifft(F * np.exp(dt * lam))

    return f_new.real if not np.iscomplexobj(f) else f_new

def advect_v_1step_hu_symbol_fft_scaled(f, a_x, dt, Lv, R):
    """
    Scheme-matched to your quantum Uv: apply exp(dt * Lv_op) along v for each x.
      lambda_v(kv) = -(|a|/(R*dv))*(1-cos kv) - i*(a/dv)*sin kv
    Periodic in v. FFT is taken along axis=1 (v-axis).
    f: (Nx, Nv), a_x: (Nx,)
    """
    Nx, Nv = f.shape
    dv = (2.0 * Lv) / Nv

    m = np.arange(Nv, dtype=float)
    k = 2.0 * np.pi * m / Nv
    sin_k = np.sin(k)
    one_minus_cos = 1.0 - np.cos(k)

    f_new = np.empty_like(f, dtype=complex)

    for jx in range(Nx):
        a = float(a_x[jx])
        lam = -(abs(a) / (R * dv)) * one_minus_cos - 1j * (a / dv) * sin_k
        F = np.fft.fft(f[jx, :])
        f_new[jx, :] = np.fft.ifft(F * np.exp(dt * lam))

    return f_new.real if not np.iscomplexobj(f) else f_new

def strang_integrate_classical_hu_fft_scaled(f0, v_grid, a_x, dt, Nt, Lx, Lv, R):
    """
    Nt Strang steps, arranged so half-steps combine:
      f <- Ux(dt/2)
      repeat Nt times: f <- Uv(dt); if not last: f <- Ux(dt)
      f <- Ux(dt/2)

    This matches the composed operator:
      [Ux(dt/2) Uv(dt) Ux(dt)]^{Nt-1} Ux(dt/2) Uv(dt) Ux(dt/2)
    i.e. same as doing Strang steps but with merged half-steps.
    """
    f = f0.copy()

    # initial half step in x
    f = advect_x_1step_hu_symbol_fft_scaled(f, v_grid, 0.5 * dt, Lx, R)

    for n in range(Nt):
        # full v kick
        f = advect_v_1step_hu_symbol_fft_scaled(f, a_x, dt, Lv, R)

        # between steps, half+half merges to full x step
        if n != Nt - 1:
            f = advect_x_1step_hu_symbol_fft_scaled(f, v_grid, dt, Lx, R)

    # final half step in x
    f = advect_x_1step_hu_symbol_fft_scaled(f, v_grid, 0.5 * dt, Lx, R)

    return f


def a_x_from_quantum_E_index_poly(nx):
    """
    LEGACY comparison helper, not used by the centred-field MPS driver.
    This function evaluates an uncentred-index polynomial:
        E(j) = -j^3 + j   (integer index j = 0..Nx-1)
    Returned as a float array a_x[j].
    """
    Nx = 1 << nx
    j = np.arange(Nx, dtype=float)
    return -(j**3) + j

def a_x_from_centered_fixed_point(nx: int, alpha_fp: int, beta_fp: int, S: int):
    """
    Classical acceleration a_x[j] that matches the QUANTUM centered fixed-point E block.

    Quantum semantics:
        c = j - 2^(nx-1)  (nx-bit two's complement)
        E_fp = alpha_fp * c + beta_fp * c^3   (integer, two's complement)
    Classical float value used by FFT-based Hu symbol update:
        a_x = E_fp / S

    Returns:
        a_x: float array of length Nx
    """
    Nx = 1 << nx
    j = np.arange(Nx, dtype=int)
    c = j - (1 << (nx - 1))          # centered integer coordinate (matches circuit)
    E_fp = alpha_fp * c + beta_fp * (c**3)
    return E_fp.astype(float) / float(S)