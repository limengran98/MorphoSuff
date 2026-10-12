# Processed OPS training data

`processed_ops/` is the compact, training-ready representation used by the
MorphoSuff Figure 2 benchmark. It contains no raw microscopy and no source
H5AD object.

Download `MorphoSuff_processed_OPS_training_v1.zip` from the
[public dataset](https://huggingface.co/datasets/Amanda1998/MorphoSuff), then
extract it to a data directory outside the code checkout. The paths below
are relative to the extracted archive root.

```text
processed_ops/
  phase172/
    phase172.npy                       8,410,291 × 172 float32 input matrix
    *_code.npy                         stable screen/well/tile/gene/guide codes
    *.fold.npy                         frozen five-fold field and gene roles
    categories.json                    code dictionaries
    phase_features_172d.txt            ordered input schema
  exact_reporters/
    all_cells_fluor_*.exact.h5         52 sparse reporter target blocks
  metadata/
    reporter_targets.csv               reporter and assay registry
    target_feature_dictionary.csv      endpoint definitions and technical core
    target_screen_map.csv              99 reporter–screen assays
    exact_pairing_integrity.csv         exact-link audit
```

The 172D matrix is shared once across reporters. Each reporter HDF5 stores
exact row links into that matrix, its observed fluorescence endpoint block,
screen, perturbation, guide and control metadata. This avoids duplicating the
same phase features 52 times while preserving exact same-cell linkage.

## Training archive census

- 8,410,291 unique phase cells;
- 172 phase features;
- 52 reporters;
- 73 physical screens;
- 99 reporter–screen assays;
- 9,996,286 exact same-cell reporter observations;
- endpoint blocks: 22 × 24D, 6 × 30D and 24 × 72D.

This is the processed training archive's inventory before downstream
analysis-specific inclusion and endpoint selection.

## Validate without copying

```bash
python studies/ops/preparation/validate_processed_release.py \
  --dataset-root /path/to/MorphoSuff
```

## Export canonical training tables

After extracting and validating the archive, create canonical reporter bundles
using the existing preparation command:

```bash
python studies/ops/preparation/export_canonical.py \
  --exact-cache-root /path/to/MorphoSuff/processed_ops/exact_reporters \
  --phase-cache /path/to/MorphoSuff/processed_ops/phase172 \
  --reporter-registry configs/ops/data/reporter_registry.csv \
  --output-dir /path/to/ops_canonical \
  --format parquet
```

Install `.[ops-data,parquet]` from the software repository before exporting.
Use `--reporter lysosome_lamp1` for a single-reporter export. Then follow the
[full-label runner guide](../../studies/ops/runners/full_label/README.md)
to materialize a fold and fit a chosen predictor. Field and gene role
assignments are supplied in the archive; strict whole-screen tasks are
constructed from the canonical assay topology.

These files derive from the public OPS resource
(<https://doi.org/10.5281/zenodo.20495192>); upstream attribution and data
terms apply. MorphoSuff preparation and benchmark code is MIT licensed.
