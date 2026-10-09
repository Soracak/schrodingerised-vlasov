# Publish the 1D1V repository (after authors select a license)

The ZIP contains the repository **contents**, not an existing GitHub history or Git remote.

1. Extract `schrodingerised-vlasov-1d1v-github.zip` and open a terminal in the extracted `schrodingerised-vlasov-1d1v` folder.
2. Add a suitable `LICENSE` agreed with the authors; verify your figures and code can be redistributed.
3. Create an empty GitHub repository (e.g. `schrodingerised-vlasov`) without automatically adding README/LICENSE files.
4. From the extracted directory run (replace `<YOUR_REPOSITORY_URL>` with the GitHub HTTPS or SSH URL):

```bash
git init
git add .
git commit -m "Initial release: 1D1V Schrödingerised Vlasov circuits"
git branch -M main
git remote add origin <YOUR_REPOSITORY_URL>
git push -u origin main
```

5. Check that only the intended code and documentation were pushed and that `results/`, local logs and environment-specific files are not tracked.

Later, add a `3d3v/` directory with its **complete dependency tree** and independent tests/configurations. Avoid renaming the original 1D1V production driver or silently changing the numerical conventions after publishing; use a tagged release for manuscripts.
