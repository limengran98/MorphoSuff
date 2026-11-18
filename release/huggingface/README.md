---
pretty_name: MorphoSuff processed OPS data and reference checkpoints
license: other
task_categories:
  - tabular-regression
tags:
  - cell-imaging
  - perturbation-biology
  - measurement-sufficiency
  - optical-pooled-screening
---

# MorphoSuff · processed data and reference checkpoints

[Code and tutorials](https://github.com/limengran98/MorphoSuff) ·
[Current manuscript and figures](https://github.com/limengran98/MorphoSuff/tree/main/paper) ·
[Hugging Face files](https://huggingface.co/datasets/Amanda1998/MorphoSuff/tree/main)

MorphoSuff tests which biological analyses can use image-based predictions
instead of direct targeted measurements in perturbation screens. This
companion deposit covers the processed OPS inputs, measured targets and a
small set of study-fitted checkpoints; the software and paper are on GitHub.

## Training data

`MorphoSuff_processed_OPS_training_v1.zip` contains a shared 172-feature phase
matrix, frozen field and gene assignments, 52 exact-linked fluorescence
target blocks and their metadata. It spans 73 screens, 99 target–screen
combinations and 9,996,286 paired observations. The archive stores
8,410,291 phase-index rows before downstream inclusion; the current manuscript
counts 7,344,374 distinct phase cells in the linked analysis set.

The training ZIP is unchanged in the v3-20261118 update. See the
[training-data guide](OPS_TRAINING_DATA.md) for the schema, checksums and
commands to export training tables.

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Amanda1998/MorphoSuff",
    repo_type="dataset",
    filename="MorphoSuff_processed_OPS_training_v1.zip",
    local_dir="../morphosuff_data",
)
```

## Selected fitted states

The [checkpoint inventory](MODEL_METADATA.md) records two representative
study-fitted states and their configurations. They are reusable examples
for the documented settings, not the full ten-predictor ensemble.
Files in this inventory can be downloaded once listed in the Hub Files tab.

This deposit does not include raw microscopy, frozen predictions, result
tables, analysis reports, plotting files or manuscript supplements.

## Attribution

The OPS source data are available from
[Zenodo](https://doi.org/10.5281/zenodo.20495192) under CC BY 4.0.
The processed data retain this attribution. MorphoSuff software is MIT-licensed;
see the [upstream model notices](https://github.com/limengran98/MorphoSuff/blob/main/THIRD_PARTY_NOTICES.md)
and the checkpoint inventory for component-specific provenance and terms.
