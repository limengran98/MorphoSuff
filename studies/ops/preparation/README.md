# Public OPS preparation

This directory provides the complete processed-data path used by the OPS
training runners:

```text
unsigned public phase/reporter H5ADs
  -> exact assay-local same-cell caches
  -> row-aligned 172D morphology cache + frozen folds
  -> runnable reporter-level canonical bundles
```

Install the data dependencies:

```bash
python -m pip install -e '.[ops-data,parquet]'
```

For the already processed public training release, skip source-H5AD
preparation and run `reproducibility/ops/figure2_from_huggingface.py`. It
downloads the shared 172D matrix and 52 exact-linked target blocks, validates
their frozen census and continues at the canonical-export stage documented
below.

## One command

After acquiring the public H5ADs with `studies/ops/acquisition/public_s3.py`,
run:

```bash
python studies/ops/preparation/prepare_public_ops.py \
  --h5ad-dir /path/to/ops_h5ad \
  --output-dir /path/to/ops_prepared \
  --reporter-registry configs/ops/data/reporter_registry.csv
```

The command resumes completed exact caches and an interrupted `phase172.npy`.
Use `--dry-run` to print the complete stage plan. Use repeated `--reporter`
arguments for a reporter subset.

## Explicit stages

```bash
# Exact same-cell linkages for all downloaded reporters
python studies/ops/preparation/build_exact_caches.py \
  --phase /path/to/ops_h5ad/all_cells_phase.h5ad \
  --reporter-dir /path/to/ops_h5ad \
  --output-root /path/to/ops_prepared/exact/reporters

# Frozen 172D input representation and field/gene folds
python studies/ops/preparation/build_phase172.py \
  --phase /path/to/ops_h5ad/all_cells_phase.h5ad \
  --exact-cache-root /path/to/ops_prepared/exact/reporters \
  --output-dir /path/to/ops_prepared/phase172

# Canonical inputs, sparse targets and assignments
python studies/ops/preparation/export_canonical.py \
  --exact-cache-root /path/to/ops_prepared/exact/reporters \
  --phase-cache /path/to/ops_prepared/phase172 \
  --reporter-registry configs/ops/data/reporter_registry.csv \
  --output-dir /path/to/ops_prepared/canonical

# Full-atlas topology and cache audit
python studies/ops/preparation/audit_prepared_ops.py \
  --prepared-root /path/to/ops_prepared \
  --output /path/to/ops_prepared/preparation_audit.json
```

Each canonical reporter directory contains `inputs`, observed long-form
`targets`, `observations`, `pairing`, endpoint/assay metadata, five field
folds, five gene folds and one destination task for every repeated screen.
These paths can be passed directly to `measurement-sufficiency run-fold` or
`studies/ops/runners/run_external_fold.py`.

## Exact-linkage rule

The phase and target tables are joined within screen by public
`well_canonical`, `tile_pheno`, `segmentation_id`, `x_pheno` and `y_pheno`.
The builder keeps only keys that are unique on both sides, requires `op_match`
on both records, and checks gene and sgRNA agreement. Every retained linkage
therefore has `pairing_confidence=1.0` in the canonical export.

## Frozen 172D rule

`features.py` reconstructs the ordered 172D panel directly from public H5AD
`var` metadata. The rule selects the morphology, localization and intensity
descriptors frozen for the full-52 benchmark and asserts exactly 172 unique
features. This eliminates dependence on a workstation-generated feature list
while retaining the exact historical feature order.
