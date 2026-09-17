# Processed OPS training data

`processed_ops/` is the compact, training-ready representation used by the
MorphoSuff Figure 2 benchmark. It contains no raw microscopy and no source
H5AD object.

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

## Frozen census

- 8,410,291 unique phase cells;
- 172 phase features;
- 52 reporters;
- 73 physical screens;
- 99 reporter–screen assays;
- 9,996,286 exact same-cell reporter observations;
- endpoint blocks: 22 × 24D, 6 × 30D and 24 × 72D.

## Validate without copying

```bash
python studies/ops/preparation/validate_processed_release.py \
  --dataset-root /path/to/MorphoSuff
```

## Run a Figure 2 fold

The code repository downloads only `processed_ops/**`, validates the frozen
census, creates canonical reporter bundles in a caller-selected workspace,
freezes the requested split and invokes the same ten-method benchmark facade:

```bash
python reproducibility/ops/figure2_from_huggingface.py \
  --workspace work/figure2 \
  --split gene --fold 0 --method ridge --device cpu
```

Use `--reporters lysosome_lamp1 --smoke` for a compact end-to-end check. Use
`--split field`, `--split gene`, or `--split strict_whole_screen`; repeat five
folds and select any frozen method documented in
`studies/ops/runners/full_label/README.md` for the complete benchmark.
The minimal route defaults to CSV. Install the `parquet` extra and pass
`--format parquet` for the complete all-52 campaign to reduce I/O and disk use.

These files derive from the public OPS resource
(<https://doi.org/10.5281/zenodo.20495192>); upstream attribution and data
terms apply. MorphoSuff preparation and benchmark code is MIT licensed.
