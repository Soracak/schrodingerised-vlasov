# Checklist before making the GitHub repository public

- [ ] Agree on a software license with the authors and add a top-level `LICENSE` file. No license is assigned by this package.
- [ ] Verify that the manuscript's 1D1V code version and numerical settings match the version being released.
- [ ] If citing results, add the precise Qiskit/Qiskit-Aer versions and NSCC configuration captured in your original environment logs; requirements are intentionally not pinned.
- [ ] Run `python -m pytest -q tests` and the small Qiskit smoke test on a machine with dependencies installed.
- [ ] Verify both Spyder modes of the desktop grid-sensitivity script if you are regenerating Appendix C.
- [ ] Confirm no machine-specific credentials, output logs, large data, or protected third-party assets have been added.
- [ ] Choose the target GitHub repository and set visibility/publication intentionally.

The 3D3V implementation is intentionally excluded from this first release.
