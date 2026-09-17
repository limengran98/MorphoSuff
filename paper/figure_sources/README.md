# Figure source data and plotting code

This directory contains the final scientific source layer for all six main and
four Supplementary figures. Each package keeps its plotting entry points,
machine-readable source tables, panel contracts, methods and provenance.
Generated artwork, design previews and historical versions are excluded.

| Figure | Source package | Build entry point |
| --- | --- | --- |
| Main 1 | `final/figure1/` | `final/figure1/build.ps1` |
| Main 2 | `final/figure2/` | `final/figure2/build.ps1` |
| Main 3 | `final/figure3/` | `final/figure3/build.ps1` |
| Main 4 | `final/figure4/` | `final/figure4/build.ps1` |
| Main 5 | `final/figure5/` | `final/figure5/build.ps1` |
| Main 6 | `final/figure6/` | `final/figure6/build.ps1` |
| Supplementary 1 | `supplementary_figure1/` | `supplementary_figure1/build.ps1` |
| Supplementary 2 | `supplementary_figure2/` | `supplementary_figure2/build.ps1` |
| Supplementary 3 | `supplementary_figure3/` | `supplementary_figure3/code/build_supplementary_figure3.py` |
| Supplementary 4 | `supplementary_figure4/` | `supplementary_figure4/code/build_supplementary_figure4.py` |

The source tables are frozen analysis outputs. Figure builders render and
validate them; they do not refit predictors, tune thresholds or choose examples.
`publication_style.py` defines the shared print typography and colour system.

## Binary inputs

To keep the Git checkout compact, registered microscopy arrays and the adopted
Figure 1a/2a workflow source pairs are supplied in the tagged GitHub release asset
`MorphoSuff_FigureBinaryAssets.zip`. Extract it at the repository root before a
full figure rebuild. The archive restores the expected relative paths and
includes SHA-256 checksums.

Large processed training data are available separately from the
[MorphoSuff dataset release](https://huggingface.co/datasets/Amanda1998/MorphoSuff).
Raw microscopy remains with the cited upstream resources.
