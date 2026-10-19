# Schrödingerised circuits for 1D1V Vlasov transport

This repository contains the **1D1V portion only** of the research code accompanying *Quantum Circuits for Variable-Coefficient Vlasov Dynamics via Schrödingerisation*.

It implements a prescribed-field 1D1V Vlasov benchmark with coherent velocity-controlled streaming, cubic-field reversible fixed-point arithmetic, Strang splitting, auxiliary Schrödingerisation register and recovery, and a **scheme-matched classical FFT reference**. This is research/benchmark code, not a production plasma solver, or a demonstration of quantum advantage.

## Directory guide

- `schro_1d1v_dwell_centered_mps.py`: original NSCC-style production driver (command-line flags, default drift `v0=4`).
- `schro_1d1v_grid_sensitivity_spyder.py`: stand-alone Spyder desktop variant; one manual `X_GRID_MODE` toggle.
- `run_spyder.py`: editable launcher for a **small demonstration** of the production driver (`v0=2`, not the paper's zero-drift benchmark).
- `build_E_centered.py`, `schro_1d1v_blocks.py`, `classical_schemes_1d1v.py`: core field arithmetic, circuit blocks and discrete classical solver.
- `preprocess_1d1v.py`, `postprocess_1d1v.py`, `compare_utils.py`: state preparation, plotting and metric routines.
- `readout_utils_mps.py`, `readout_utils.py`: production flagged MPS readout and an older readout utility kept for reference.
- `ft_resources.py`: decomposed logical gate-count proxy and approximate rotation-synthesis estimates.
- `io_utils.py`, `plot_completed_run.py`: run-file export and postprocessing.
- `scripts/`: Appendix C four-panel generator and saved-metric checker.
- `examples/`: numerical reference metrics for the two Appendix C desktop runs.
- `tests/`: lightweight CPU-only validation; `test_loaders.py` is an optional Qiskit-based auxiliary-loader check.
- `docs/`: appendix instructions and pre-publication checklist.

## Install

Use **Python 3.10+** in a clean environment, then install dependencies:

```sh
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt   # optional tests
python -m pytest -q tests
```

Requirements are deliberately unpinned. **For exact NSCC reproduction**, use the Qiskit and Qiskit Aer versions recorded in the original run's `environment_*.json` files. Differences in Qiskit decompositions, gates and MPS configuration may change runtime or even compatibility.

## Quick checks

A small run (still potentially slow on a desktop):

```sh
python schro_1d1v_dwell_centered_mps.py --nx 2 --nv 2 --na 5 --nt 1 --dt 0.01 --v0 0.0 --run-id example_225_zero --threads 2 --max-bond 512 --truncation 1e-12 --no-output-plots
```

Resource report **without** executing an MPS readout:

```sh
python schro_1d1v_dwell_centered_mps.py --nx 2 --nv 2 --na 5 --nt 1 --dt 0.01 --v0 0.0 --run-id resources_225_zero --resource-report --no-readout --no-output-plots
```

Original **1D1V paper-like T1 parameters** (a much more expensive simulation; not a desktop smoke test):

```sh
python schro_1d1v_dwell_centered_mps.py --nx 5 --nv 5 --na 7 --nt 1 --dt 0.05 --v0 0.0 --lx 6 --lv 6 --R 6 --threads 2 --max-bond 512 --truncation 1e-12 --readout-reps 4 --run-id T1_557_005 --no-output-plots
```

Results are written to `results/<RUN_ID>/`. Avoid reusing an existing run ID. To plot a completed run:

```sh
python plot_completed_run.py results/example_225_zero
```

## Appendix C in Spyder

Open `schro_1d1v_grid_sensitivity_spyder.py`, set `X_GRID_MODE = "cell"`, and press F5. Then change it to `"nodal"` and press F5 again. No command-line flags are needed. The default small experiment uses `nx=nv=3`, `na=5`, `Nt=1`, `dt=0.05`, `R=4`, `Lx=Lv=6` and `v0=0`. Outputs go to different `results/grid-cell_*` and `results/grid-nodal_*` directories. For generating the four separate appendix plots, see [docs/GRID_SENSITIVITY.md](docs/GRID_SENSITIVITY.md).

## Numerical scope and important caveats

1. **Initial grid / field grid:** The T1 production driver samples the initial state at `x_j = -Lx/2 + (j+1/2)*dx`, while its coherent field circuit uses the centred integer `c(j)=j-Nx/2`, corresponding to `x_tilde_j = x_j - dx/2`. The classical reference uses **that same nodal fixed-point field**. The Appendix C script varies the initial grid but leaves the field unchanged.
2. **Coarse-grid cubic coefficient:** The production driver has `s=6` fixed in source. For the original `nx=5`, `Lx=6`, `a0=kappa=1` benchmark, `alpha_fp=12` and `beta_fp=0`: the *implemented discrete force is linear*, despite the inclusion of cubic-field arithmetic in the compiled circuit. The `nx=3` Appendix C case instead has `alpha_fp=48`, `beta_fp=-27`, retaining a nonzero cubic contribution. Results validate the **implemented discrete circuit** and MPS cost, not continuum nonlinear dynamics at all resolutions.
3. **Resources:** `ft_resources.py` counts decomposed CX and rotation-like (`u`, `p`, `rz`, `cp`) operations. The estimated T-count is a rotation-synthesis **proxy**, not an exact fault-tolerant Clifford+T gate count.
4. **Computational scope:** full Qiskit Aer matrix-product-state verification may be very costly even when the physical phase-space grid is small. Classical solver runtimes are not the same as MPS emulation runtimes.


