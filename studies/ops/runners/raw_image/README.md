# OPS raw-phase representation runners

This directory contains the complete study-owned execution chain for the
six-reporter raw-phase experiment:

```text
public OPS image locations
  -> deterministic 384-pixel crop plan and resumable crops
  -> frozen DINOv2 or Cytoland embeddings
  -> identical reporter-specific MLP heads on five frozen gene folds

public Cytoland checkpoint + OPS crops
  -> partial fine-tuning of encoder stages 2-3 and endpoint head
  -> the same five frozen gene-fold evaluation
```

Data, images, embeddings, upstream repositories, public checkpoints and fitted
states live outside the source checkout. Every such path is supplied by the
caller through `assets.json`; no script contains a machine-specific path or
downloads a model silently from a training worker.

## Frozen panel and evaluation

The panel contains two high-, two intermediate- and two low-recoverability
reporters selected before inspecting image-model results:

| Stratum | Reporters |
|---|---|
| high | `early_endosome_eea1`, `stress_granule_g3bp1` |
| intermediate | `lysosome_lamp1`, `mitochondria_tomm20` |
| low | `nucleoli_npm1`, `ps6` |

Every representation uses the exact paired cells, observed technical-core
endpoints, five frozen held-out-gene folds, next-fold validation, train-only
preprocessing and screen-matched control-relative evaluation. Thus encoder
choice changes while the scientific cohort and prediction head remain fixed.

## Installation

Install the local image dependencies:

```bash
python -m pip install -e ".[image]"
```

Acquire the two upstream repositories and public weights at the recorded
revisions:

```bash
git clone https://github.com/facebookresearch/dinov2.git external/dinov2
git -C external/dinov2 checkout 7764ea0f912e53c92e82eb78a2a1631e92725fc8
curl -L --fail --output weights/dinov2_vits14_pretrain.pth \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth

git clone https://github.com/mehta-lab/VisCy.git external/VisCy
git -C external/VisCy checkout 4b62365c0df25929bffc7f01b3bb2d1c11d69cce
python -m pip install -e "external/VisCy[metrics,visual]"
curl -L --fail --output weights/VSCyto2D_epoch399-step23200.ckpt \
  https://public.czbiohub.org/comp.micro/viscy/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt
```

Expected SHA-256 digests:

| Asset | SHA-256 |
|---|---|
| DINOv2 ViT-S/14 | `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9` |
| Cytoland VSCyto2D | `1abdbc1c727aba33dad0d174dba57d128a12019b712db946ce06012ecd64c1fe` |

Cytoland provenance and architecture are also described by the [official
model card](https://virtualcellmodels.cziscience.com/model/cytoland). DINOv2
weights and loading entry points come from the [official DINOv2
repository](https://github.com/facebookresearch/dinov2).

## OPS inputs

Use `studies/ops/acquisition/public_s3.py` to inventory and fetch the public OPS
processed phase and reporter tables. The OPS adapter produces:

- a phase cache containing row-aligned `screen_code`, `well_code`, `tile_code`,
  `gene_code` and `gene_holdout_main.fold.npy` arrays;
- one exact-pair reporter HDF5 file named
  `all_cells_fluor_<reporter>.exact.h5`;
- the observed target-feature dictionary; and
- an image-location table derived from the public OPS OME-Zarr metadata.

The image-location table has one row per reporter observation and these
columns:

```text
phase_row_index, reporter_slugs, remote_zarr_https, level0_array_path,
x_pheno, y_pheno
```

`remote_zarr_https` is restricted by the downloader to the anonymous
`ops-explorer-public` bucket. Build and fetch the deduplicated phase crops:

```bash
RAW=studies/ops/runners/raw_image

python "$RAW/build_crop_plan.py" \
  --locations local/ops/raw_image_locations.parquet \
  --config "$RAW/configs/cytoland_frozen_pilot6.json" \
  --output-root local/ops/raw_phase_crops_384_plan

python "$RAW/download_crops.py" \
  --plan-root local/ops/raw_phase_crops_384_plan \
  --output-root local/ops/raw_phase_crops_384
```

The crop directory contains a manifest plus `shards/shard_XXXXXX.h5`; every
shard stores float32 `phase[N,384,384]`, the exact phase row ID and its valid
unpadded window. Both download and embedding extraction resume completed
shards.

Copy `assets.example.json` to an untracked `assets.json` and replace every
path with the corresponding local asset.

## Frozen encoder experiments

Run the fail-closed asset check, then extract each embedding once:

```bash
python "$RAW/preflight.py" --mode dinov2 \
  --config "$RAW/configs/dinov2_frozen_pilot6.json" --assets assets.json

python "$RAW/extract_embeddings.py" --encoder dinov2 \
  --config "$RAW/configs/dinov2_frozen_pilot6.json" \
  --data-root local/ops/raw_phase_crops_384 \
  --cells local/ops/raw_phase_crops_384_plan/cells.parquet \
  --repository external/dinov2 \
  --checkpoint weights/dinov2_vits14_pretrain.pth \
  --output-root local/ops/embeddings/dinov2_vits14_pilot6

python "$RAW/extract_embeddings.py" --encoder cytoland \
  --config "$RAW/configs/cytoland_frozen_pilot6.json" \
  --data-root local/ops/raw_phase_crops_384 \
  --cells local/ops/raw_phase_crops_384_plan/cells.parquet \
  --repository external/VisCy \
  --checkpoint weights/VSCyto2D_epoch399-step23200.ckpt \
  --output-root local/ops/embeddings/cytoland_vscyto2d_pilot6
```

Build and run the 30 reporter-by-fold heads:

```bash
python "$RAW/build_plan.py" --mode cytoland-head \
  --config "$RAW/configs/cytoland_frozen_pilot6.json" --assets assets.json \
  --result-root local/results/cytoland_frozen \
  --output-plan local/plans/cytoland_frozen.json

python "$RAW/run_plan.py" --plan local/plans/cytoland_frozen.json \
  --run-root local/queues/cytoland_frozen --gpu-ids 0 --workers 1
```

Use `--mode dinov2-head` for the DINOv2 comparison.

## Cytoland partial fine-tuning

The registered transfer protocol freezes the stem and stages 0-1, trains
stages 2-3 plus a `[256,128]` head, and restores the validation-best state.

```bash
python "$RAW/preflight.py" --mode cytoland-finetune \
  --config "$RAW/configs/cytoland_finetune_pilot6.json" --assets assets.json

python "$RAW/build_plan.py" --mode cytoland-finetune \
  --config "$RAW/configs/cytoland_finetune_pilot6.json" --assets assets.json \
  --result-root local/results/cytoland_finetune \
  --output-plan local/plans/cytoland_finetune.json

python "$RAW/run_plan.py" --plan local/plans/cytoland_finetune.json \
  --run-root local/queues/cytoland_finetune --gpu-ids 0,1 --workers 2 \
  --continue-on-error
```

## Validation and smoke test

All data-construction, training and queue entry points support `--dry-run`;
`preflight.py` performs read-only asset validation. The following commands construct a
synthetic exact-pair cache and complete frozen embedding, trains a real
PyTorch head on CPU, evaluates cell- and gene-level outputs, and checks the
written result contract:

```bash
python studies/ops/runners/raw_image/smoke.py
python studies/ops/runners/raw_image/contract_smoke.py
python -m compileall -q studies/ops/runners/raw_image
```

The first smoke test trains and evaluates the frozen head. The contract smoke
also checks `--help` for every entry point, exercises crop-plan/download,
preflight, extraction, fine-tuning and plan dry runs, and performs a synthetic
Cytoland-stage/head forward and backward pass. Both write only to temporary
directories and require no OPS data, network access, GPU or model checkpoint.
