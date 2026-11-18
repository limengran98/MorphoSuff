# Data and code availability

## Software and paper

This repository contains the reusable MorphoSuff package, dataset-specific
workflows, the [manuscript](../paper/manuscript.pdf),
[Supplementary Information](../paper/supplementary.pdf) and
[six main and nine Supplementary figure PDFs](../paper/figures/).
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

Selected fitted states and their required preprocessing are listed separately
in the [checkpoint inventory](../release/huggingface/MODEL_METADATA.md). The
[Hub file list](https://huggingface.co/datasets/Amanda1998/MorphoSuff/tree/main)
shows which files have been deposited. Raw microscopy, frozen predictions and
figure source tables are not included in the Hub release.

## Supplementary data

One Supplementary Data archive accompanies the manuscript submission. It
contains OPS predictor-specific evidence, calibration–evaluation decisions,
functional analyses and external-use evidence for OASIS, Cell Health and LINCS.
It is supplied separately from this software repository and the Hub deposit.

## Upstream resource

The [public OPS source archive](https://doi.org/10.5281/zenodo.20495192)
provides the upstream resource. Public processed single-cell objects are
acquired from the unsigned S3 source recorded in the
[source manifest](../studies/ops/data_sources.yaml); see the
[acquisition and preparation guide](ops_data_and_preparation.md). Upstream
data terms remain applicable.
