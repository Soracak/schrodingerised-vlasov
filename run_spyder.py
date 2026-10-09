import os
import sys
import runpy

# Always run from this project directory
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

# ============================================================
# Parameters for this run
# ============================================================

    # parser.add_argument("--nx", type=int, default=5)
    # parser.add_argument("--nv", type=int, default=5)
    # parser.add_argument("--na", type=int, default=7)
    # parser.add_argument("--nt", "--Nt", dest="Nt", type=int, default=1)
    # parser.add_argument("--dt", type=float, default=0.01)
    # parser.add_argument("--v0", type=float, default=4.0)
    # parser.add_argument("--lx", type=float, default=6.0)
    # parser.add_argument("--lv", type=float, default=6.0)
    # parser.add_argument("--R", "--r", dest="R", type=float, default=6.0)
    # parser.add_argument("--run-id", default=None)

    # parser.add_argument("--threads", type=int, default=2)
    # parser.add_argument(
    #     "--swap-direction",
    #     choices=("mps_swap_left", "mps_swap_right"),
    #     default="mps_swap_left",
    # )
    # parser.add_argument("--lapack", action="store_true")
    # parser.add_argument("--mps-log-data", action="store_true")
    # parser.add_argument("--parallel-threshold", type=int, default=14)
    # parser.add_argument("--max-bond", type=int, default=512)
    # parser.add_argument("--truncation", type=float, default=1e-12)
    # parser.add_argument("--readout-reps", type=int, default=4)

    # parser.add_argument("--no-readout", action="store_true")
    # parser.add_argument("--statevector", action="store_true")
    # parser.add_argument("--resource-report", action="store_true")
    # parser.add_argument("--draw-circuit", action="store_true")
    # parser.add_argument("--initial-plots", action="store_true")
    # parser.add_argument("--no-output-plots", action="store_true")
    # parser.add_argument("--export", action="store_true")

sys.argv = [
    "schro_1d1v_dwell_centered_mps.py",
    "--nx", "2",
    "--nv", "2",
    "--na", "5",
    "--nt", "1",
    "--dt", "0.01",
    "--v0", "2",
    "--run-id", "test_225_1_001",
    "--threads", "2",
    "--max-bond", "512",
    "--truncation", "1e-12",
]

# ============================================================
# Run main program
# ============================================================

runpy.run_path(
    os.path.join(HERE, "schro_1d1v_dwell_centered_mps.py"),
    run_name="__main__",
)