# Manuscript and reproducibility resources

[Manuscript](manuscript.pdf) ·
[Supplementary Information](supplementary.pdf) ·
[Figure sources](figure_sources/README.md)

This public directory contains the two submitted PDF documents and the compact
scientific materials needed to inspect or reproduce the reported analyses.
Manuscript LaTeX sources, submission forms, cover letters, build logs and internal editing
records are intentionally excluded.

## Contents

| Resource | Location |
| --- | --- |
| Main manuscript | [manuscript.pdf](manuscript.pdf) |
| Supplementary Information | [supplementary.pdf](supplementary.pdf) |
| Main Figures 1–6 | [figure_sources/final](figure_sources/final/README.md) |
| Supplementary Figures 1–4 | [figure_sources](figure_sources/README.md) |
| Supplementary Data 1–3 | [supplementary_data](supplementary_data) |

The figure packages contain the final plotting code, machine-readable source
tables, panel contracts, methods and provenance. Generated figures and
intermediate renderings are not versioned.

## Binary figure inputs

Registered microscopy arrays and the two adopted workflow-panel source pairs are
distributed with the tagged GitHub release as
`MorphoSuff_FigureBinaryAssets.zip`. Extract the archive at the repository root
before rebuilding panels that use those inputs:

```bash
unzip MorphoSuff_FigureBinaryAssets.zip
```

The archive restores repository-relative paths and includes a checksum
manifest. Large processed OPS training arrays remain in the
[companion dataset release](https://huggingface.co/datasets/Amanda1998/MorphoSuff);
full raw microscopy remains with the cited source resources.

## Numerical supplements

Each Supplementary Data directory contains its own README, frozen inputs,
analysis code and integrity manifest:

- [Supplementary Data 1](supplementary_data/SupplementaryData1/README.md):
  predictor-specific criteria and measurement decisions.
- [Supplementary Data 2](supplementary_data/SupplementaryData2/README.md):
  external evidence and threshold sensitivity.
- [Supplementary Data 3](supplementary_data/SupplementaryData3/README.md):
  held-out OASIS predictions, rankings and selection replay.

Source datasets and third-party components retain their original licences and
attribution. Study-owned code is released under the repository's MIT License.
