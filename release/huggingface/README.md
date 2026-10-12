---
pretty_name: MorphoSuff processed OPS training data
license: other
task_categories:
  - tabular-regression
tags:
  - cell-imaging
  - perturbation-biology
  - measurement-sufficiency
  - optical-pooled-screening
---

# MorphoSuff processed OPS training data

The public dataset at
[Amanda1998/MorphoSuff](https://huggingface.co/datasets/Amanda1998/MorphoSuff)
provides `MorphoSuff_processed_OPS_training_v1.zip`, a processed tabular
training resource for the OPS prediction benchmark.

## Contents

```text
processed_ops/
  phase172/          Shared phase features, identifiers and frozen field/gene folds
  exact_reporters/    52 exact-linked fluorescence target blocks
  metadata/          Reporter, assay, endpoint and linkage metadata
README.md
ARCHIVE_MANIFEST.tsv  File sizes, SHA-256 hashes and source attribution
```

The shared input matrix contains 8,410,291 rows and 172 features before the
downstream analysis-specific inclusion filters. The reporter blocks contain
9,996,286 exact same-cell reporter observations across 52 reporters, 73 physical
screens and 99 reporter–screen assays. The input matrix is stored once;
reporter HDF5 files reference its rows.

The archive supplies processed training data, not raw microscopy, fitted model
weights, frozen predictions or figure source tables. Supplementary Data 1–3
are separate manuscript-submission attachments.

## Download

Install `huggingface_hub`, then download the archive to a data directory
outside the code checkout:

```python
from huggingface_hub import hf_hub_download

archive = hf_hub_download(
    repo_id="Amanda1998/MorphoSuff",
    repo_type="dataset",
    filename="MorphoSuff_processed_OPS_training_v1.zip",
    local_dir="../morphosuff_data",
)
print(archive)
```

See the [training-data guide](OPS_TRAINING_DATA.md) for the extracted layout,
validation and canonical-export commands. Model runners are documented in
[the software repository](https://github.com/limengran98/MorphoSuff/tree/main/studies/ops/runners).

## Provenance and citation

The upstream OPS archive is
[10.5281/zenodo.20495192](https://doi.org/10.5281/zenodo.20495192).
Processed data retain its attribution and applicable data terms. MorphoSuff
code is MIT licensed. Please cite the MorphoSuff study, the original OPS
resource and any external predictor used in your analysis.
