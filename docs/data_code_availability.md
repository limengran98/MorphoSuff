# Data and code availability

## Software and paper

This repository contains the reusable MorphoSuff package, dataset-specific
workflows, the [manuscript](../paper/manuscript.pdf),
[Supplementary Information](../paper/supplementary.pdf) and
[six main and six Supplementary figure PDFs](../paper/figures/).
Study-owned code is released under the
[MIT License](../LICENSE). See [software citation metadata](../CITATION.cff)
and [code ownership and licensing](../CODE_OWNERSHIP_AND_LICENSE.md).

## Processed study data

The [companion dataset card](../release/huggingface/README.md) provides the
public repository locator, inventory, loading example and provenance. The
[OPS training guide](../release/huggingface/OPS_TRAINING_DATA.md) documents the
Figure 2 download and training workflow.

The public Hugging Face repository contains
`MorphoSuff_processed_OPS_training_v1.zip`. It supplies the shared 172D phase
matrix, frozen field/gene assignments, 52 exact-linked sparse reporter target
blocks and metadata. Its embedded `ARCHIVE_MANIFEST.tsv` records file sizes,
SHA-256 hashes and provenance.

Raw microscopy, fitted model states, frozen predictions and figure source
tables are not included in this training-data archive.

## Supplementary data

Three data archives accompany the manuscript submission:

- **Supplementary Data 1:** OPS predictor-specific evidence, calibration–evaluation
  decisions and functional analyses.
- **Supplementary Data 2:** external response evidence and threshold sensitivity.
- **Supplementary Data 3:** held-out OASIS, Cell Health and LINCS predictions,
  reference measurements and selection records.

These archives include reconstruction and verification scripts for the reported
results. They are provided separately from this software repository.

## Upstream resource

The [public OPS source archive](https://doi.org/10.5281/zenodo.20495192)
provides the upstream resource. Public processed single-cell objects are
acquired from the unsigned S3 source recorded in the
[source manifest](../studies/ops/data_sources.yaml); see the
[acquisition and preparation guide](ops_data_and_preparation.md). Upstream
data terms remain applicable.
