---
pretty_name: MorphoSuff processed OPS training and evidence release
license: other
task_categories:
  - tabular-regression
  - image-to-image
  - feature-extraction
tags:
  - cell-imaging
  - perturbation-biology
  - measurement-sufficiency
  - optical-pooled-screening
---

# MorphoSuff processed OPS training and evidence release

This repository accompanies the MorphoSuff codebase at
<https://github.com/limengran98/MorphoSuff>. It provides compact, processed
artifacts required both to train the Figure 2 reference models and to inspect
the reported measurement-sufficiency evidence without redistributing the
original microscopy resource.

## Contents

```text
source_data/
  figure1/ ... figure6/     canonical numeric source tables and contracts
processed_ops/
  phase172/                  shared 8.41M × 172 input representation and folds
  exact_reporters/           52 sparse exact-linked reporter target caches
  metadata/                  reporter, assay and endpoint dictionaries
figures/
  Figure1.png ... Figure6.png
models/
  scpair_npm1_gene_fold0_1pct/
  cytoland_lamp1_gene_fold0/
MANIFEST.tsv                SHA-256, byte count, licence and provenance
```

The source tables cover the atlas topology, ten-method benchmark summaries,
reporter/screen/assay recoverability, same-cell falsification, response
fidelity, stability constraints, environmental transfer, conditional
ambiguity, low-label summaries, image-representation summaries and
measurement-sufficiency decisions.

The processed training layer contains 8,410,291 unique phase cells, 52
reporters, 73 screens, 99 reporter–screen assays and 9,996,286 exact-linked
same-cell observations. The input matrix is stored once and reporter HDF5
files reference its rows, avoiding 52 redundant copies. See
`processed_ops/README.md` for its schema, validation command and the
one-command Figure 2 training route.

## Explicit exclusions

This release does **not** contain raw microscopy, field images, single-cell
image crops, public H5AD objects, full per-cell image embeddings or the full
model checkpoint collection. The six PNG files are rendered manuscript
figures, not raw image data. The released 172D matrix and reporter HDF5 files
are processed tabular training assets, not raw images or source H5AD objects.

## Representative fitted states

Only two fitted states are selected:

1. an scPair NPM1 held-out-gene fold-0 checkpoint at 1% target labels; and
2. a Cytoland VSCyto2D partial-fine-tuning state for LAMP1 held-out-gene fold
   0.

They are examples for loader and inference validation, not a replacement for
the frozen five-fold campaigns. Each file is referenced directly from the
study result tree during upload, so no second local copy is created. The
Cytoland state requires the public base checkpoint identified in the model
metadata; upstream terms remain applicable.

## Provenance and licence

The upstream OPS archive is <https://doi.org/10.5281/zenodo.20495192> and is
released under CC BY 4.0. Processed tables retain attribution to that resource.
Code is MIT licensed in the GitHub repository. Component-specific checkpoint
licences and source identities are recorded in `MANIFEST.tsv` and in
`studies/ops/model_sources.yaml` in the code repository.

## Loading a table

```python
from huggingface_hub import hf_hub_download
import pandas as pd

path = hf_hub_download(
    repo_id="Amanda1998/MorphoSuff",
    repo_type="dataset",
    filename="source_data/figure3/panel_c/reporter_consensus_10method.csv",
)
table = pd.read_csv(path)
```

## Citation

Please cite the MorphoSuff study, the public OPS resource and any external
model used in a derivative analysis. Release-specific archival DOIs will be
added when the code and processed-data deposits are minted.
