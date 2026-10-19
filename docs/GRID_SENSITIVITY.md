# Appendix C: desktop grid-convention sensitivity

The standalone `schro_1d1v_grid_sensitivity_spyder.py` differs from the production 1D1V driver only in the specified small desktop defaults, grid-choice toggle, run naming, and saved mode metadata. It does not change the field operator, classical reference force, or Strang evolution.

Open the script in Spyder. Change the single line `X_GRID_MODE = "cell"` to `"nodal"` for the second run, using F5 for each. The default parameters are `nx=nv=3`, `na=5` (called `n_p` in the paper), `Nt=1`, `dt=0.05`, `R=4`, `Lx=Lv=6`, `v0=0`, `s=6` fixed in source. At this resolution `alpha_fp=48` and `beta_fp=-27`.

`"cell"`: initial density on x_j = -Lx/2+(j+1/2)*dx, force on shifted nodal grid.

`"nodal"`: initial density on x_j = -Lx/2+j*dx, force on that same nodal grid.

The script writes separate `results/grid-cell_*` and `results/grid-nodal_*` directories. It does **not** use a CLI flag to choose the grid. Other numeric CLI flags remain present as in the original driver, but running with Spyder's default no-command-line-arguments configuration is recommended.

To regenerate the four independent figure panels after both runs:

```sh
python scripts/make_grid_sensitivity_panels.py
```

Or run that file in Spyder after setting `BASE = Path.cwd() / "results"` (or `BASE = Path.cwd()` if working inside the results folder). The script writes four PDFs and four PNGs. It finds the two runs automatically when there is exactly one matching run per mode; otherwise set the full `CELL_RUN` and `NODAL_RUN` paths at the top.

To verify the metric from raw solution amplitudes without rerunning Aer:

```sh
python scripts/check_saved_metrics.py results/grid-cell_*/data/run_solution_centered.npz results/grid-nodal_*/data/run_solution_centered.npz
```

The discrepancy uses independently unit-L2-normalized `u_quant` and `f_class`, whereas marginal panels plot saved unit-mass-normalized `fQ`, `fC`, and `f0`. Results check circuit-reference agreement for a small discrete model, **not** convergence to a continuum solution. The half-cell shift changes the sampled initial distribution significantly at this coarse resolution.
