from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np

def export_centered_run(
    *,
    nx: int, nv: int, na: int,
    Nx: int, Nv: int,
    Nt: int, dt: float, dt_cfl: float, nu: float,
    Lx: float, Lv: float, dx: float, dv: float,
    a0: float, kappa: float, a_max_bound: float,
    alpha_fp: int, beta_fp: int, alpha: float, beta: float, S: int,
    R: float, R_suggest: float,
    nE: int,
    qc,
    ft_resource_row_fn,
    # optional/conditional stuff
    info: dict | None = None,
    rel_l2: float | None = None,
    u_quant: np.ndarray | None = None,
    f_class: np.ndarray | None = None,
    t_build0: float | None = None,
    t_build1: float | None = None,
    figures: dict[str, object] | None = None,   # e.g. {"fig0": fig0, "fig1": fig1, ...}
    results_root: str | Path | None = None,
    run_id: str | None = None,
    clear_log: bool = True,
    decompose_reps: int = 5,
    rot_eps: float = 1e-10,
) -> Path:
    """
    Export log-style run summary + figures for the centered 1D1V dwell script.

    figures: mapping like {"fig0": fig0, "fig1": fig1, "fig2": fig2, "fig3": fig3}
             objects must implement .savefig(path, ...)

    Returns
    -------
    results_dir : Path
        Folder where outputs were written.
    """

    def _fmt(x, *, prec=6) -> str:
        if x is None:
            return ""
        if isinstance(x, (float, np.floating)):
            return f"{float(x):.{prec}g}"
        return str(x)

    def _safe_get(d: dict | None, key: str, default=""):
        if not isinstance(d, dict):
            return default
        try:
            return d.get(key, default)
        except Exception:
            return default

    # ---------- paths ----------
    if results_root is None:
        results_root = Path.cwd() / "results_1d1v"
    else:
        results_root = Path(results_root)

    results_dir = results_root / f"nx_{nx}_nv_{nv}" / f"Nt_{Nt}_dt_{dt}"
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "run_log_centered.txt"

    # ---------- run id ----------
    # Your current code sets RUN_ID='' which disables uniqueness; keep it optional but sane.
    if run_id is None:
        run_id = datetime.now().strftime("%m%d_%H%M%S")

    def with_run_id(filename: str) -> str:
        p = Path(filename)
        return f"{p.stem}_{run_id}{p.suffix}" if run_id else filename

    # ---------- clear log if requested ----------
    if clear_log:
        log_path.write_text("", encoding="utf-8")

    # ---------- FT resources ----------
    try:
        ft = ft_resource_row_fn(qc, decompose_reps=decompose_reps, rot_eps=rot_eps)
    except Exception as e:
        ft = {"error": f"ft_resource_row failed: {e}"}

    # ---------- timing ----------
    t_build = None
    if t_build0 is not None and t_build1 is not None:
        t_build = t_build1 - t_build0

    # ---------- assemble log block ----------
    block: list[str] = []
    block.append("=" * 90)
    block.append("=== 1D1V double-well prescribed-field Vlasov setup ===")
    block.append(f"n_x = {nx}, n_v = {nv} => Nx = {Nx}, Nv = {Nv}")
    block.append(f"x domain: centered in [{-0.5*Lx}, {0.5*Lx}), dx={_fmt(dx)}")
    block.append(f"v domain: centered in [{-Lv}, {Lv}], dv={_fmt(dv)}")
    block.append(f"double-well params: a0={_fmt(a0)}, kappa={_fmt(kappa)}")
    block.append(f"conservative bound |a(x)| <= {_fmt(a_max_bound)}")
    block.append(f"N_t = {Nt}, dt = {_fmt(dt)} (dt_cfl={_fmt(dt_cfl)}, nu={_fmt(nu)})")
    block.append("")
    block.append(f"α_fp = {alpha_fp}, β_fp = {beta_fp}, α = {_fmt(alpha)}, β = {_fmt(beta)}, S = {S}")
    block.append(f"R(manual): {_fmt(R, prec=3)} | R_suggest={_fmt(R_suggest, prec=3)}")
    block.append(f"nE = {nE}")
    block.append("")

    # qc stats
    block.append("[qc stats]")
    try:
        block.append(f"  depth={qc.depth()}  size={qc.size()}  nq={qc.num_qubits}")
    except Exception as e:
        block.append(f"  (qc stats unavailable: {e})")

    # unitary op census if present
    if isinstance(info, dict) and "unitary_ops" in info:
        block.append("[resource ops] (unitary only)")
        block.append(f"  {info['unitary_ops']}")
        block.append("")
    block.append("")

    # FT resources
    block.append("[FT resources] (rotation-synthesis proxy)")
    if "error" in ft:
        block.append(f"  {ft['error']}")
    else:
        block.append(f"  nq={ft.get('nq','')}  cx={ft.get('cx','')}  rot_like={ft.get('rot_like','')}")
        block.append(f"  T_per_rot_est={ft.get('T_per_rot_est','')}  Tcount_est={ft.get('Tcount_est','')}")
        block.append(f"  depth_dec={ft.get('depth_dec','')}  size_dec={ft.get('size_dec','')}")
    block.append("")

    # timings
    block.append("[timing]")
    block.append(f"  build_time_s: {_fmt(t_build)}")
    if isinstance(info, dict):
        block.append(
            "  readout wall: "
            f"decompose={_fmt(_safe_get(info,'t_decompose'))}s | "
            f"save_append={_fmt(_safe_get(info,'t_saveappend'))}s | "
            f"transpile={_fmt(_safe_get(info,'t_transpile'))}s | "
            f"run={_fmt(_safe_get(info,'t_run'))}s | "
            f"post={_fmt(_safe_get(info,'t_post'))}s | "
            f"total={_fmt(_safe_get(info,'t_total'))}s"
        )
        block.append(f"  purity: {_fmt(_safe_get(info,'purity'))}")
        if "warning" in info:
            block.append(f"  warning: {info['warning']}")
    else:
        block.append("  readout: (skipped or not available)")
    block.append("")

    # accuracy
    block.append("[accuracy]")
    if rel_l2 is not None:
        block.append(f"  Relative L2 (after L2-normalization): {_fmt(rel_l2)}")
    else:
        block.append("  Relative L2: (not computed)")

    # masses (optional)
    if u_quant is not None:
        try:
            mass_q = float(np.sum(np.maximum(u_quant, 0)))
            block.append(f"  L1 mass quantum (raw): {_fmt(mass_q)}")
        except Exception:
            pass
    if f_class is not None:
        try:
            mass_c = float(np.sum(f_class))
            block.append(f"  L1 mass classical (raw): {_fmt(mass_c)}")
        except Exception:
            pass

    block.append("=" * 90 + "\n")

    # ---------- write log ----------
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block))

    # ---------- save figures ----------
    saved = []
    if figures:
        for key, fig in figures.items():
            if fig is None:
                continue
            fname = with_run_id(f"{key}.png") if not key.endswith(".png") else with_run_id(key)
            outpath = results_dir / fname
            fig.savefig(outpath, bbox_inches="tight", pad_inches=0.1)
            saved.append(fname)

    print(f"[log] appended run block to: {log_path.resolve()}")
    if saved:
        print(f"[plot] saved {len(saved)} PNG(s) to: {results_dir} -> {', '.join(saved)}")
    else:
        print("[plot] no figures saved (none provided)")

    return results_dir