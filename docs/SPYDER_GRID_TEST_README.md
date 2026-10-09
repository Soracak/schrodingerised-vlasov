# Spyder: cell-centred vs nodal initialization

Open **schro_1d1v_grid_sensitivity_spyder.py** in Spyder and set the working
folder to the directory containing this script and its helper Python modules.

Near the start of the script find:

```python
X_GRID_MODE = "cell"  # change to "nodal" for the second run
```

1. Run the script with `X_GRID_MODE = "cell"` (original NSCC initialization).
2. Change the setting to `X_GRID_MODE = "nodal"` and run the same script again.

No command-line flags are required. The other **desktop defaults** remain
`nx=nv=3`, `na=5`, `Nt=1`, `dt=0.05`, `R=4`, `v0=0`, `Lx=Lv=6`, `s=6`.
The existing argparse defaults remain available for adjustments, but no grid
switch is passed through argparse. The run ID is generated automatically and
includes `grid-cell` or `grid-nodal`, preventing these two runs from overwriting
each other under normal use (unless a custom `--run-id` is supplied).

Results are stored in `results/grid-cell_.../` and
`results/grid-nodal_.../`, containing the saved quantum and classical
solutions, parameters, readout metrics, and, when plotting is enabled, figures.
The grid mode is written to the metadata and solution files.

**Important:** Only the initial-condition sampling grid changes between modes.
The fixed-point electric field is evaluated on the nodal centered-integer
coordinate in both modes, and the classical reference uses the same force.
The appendix comparison therefore probes grid-initialization sensitivity,
not absolute continuum-discretization accuracy.

If Spyder passes any unintended command-line arguments to the run, clear
`Run > Configuration per file > Command line options` and rerun.

An MPS run can still take substantial time on a desktop. The script was
syntax-checked and its coordinate selections were tested, but no Aer MPS
simulation was executed as part of preparing this package.
