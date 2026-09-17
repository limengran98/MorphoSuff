# Main Figures 1–6

This directory provides one current plotting package for each main figure.
Code, source tables, figure specifications and panel-level documentation are
versioned; generated PDF, SVG, PNG and TIFF artwork is intentionally omitted.

Run the individual `build.ps1` entry point within `figure1/` through `figure6/`,
or use:

```powershell
./build_all.ps1
```

Before a full rebuild, extract `MorphoSuff_FigureBinaryAssets.zip` from the
tagged GitHub release at the repository root. The archive supplies the compact
registered arrays and two workflow-panel source pairs that are not stored in Git.

The quantitative source tables bundled here cover atlas topology, the ten-method
benchmark, same-cell controls, response fidelity, target reliability,
environmental transfer, measurement decisions and the external applications.
The manuscript PDFs in `paper/` are the canonical submitted display documents.
