# OPS data source, verification, and preparation

The released commands cover the full path from the unsigned public OPS
processed H5ADs to exact-pair caches, the frozen 172D representation, split
assignments and canonical model inputs. Public source locators are frozen in
`studies/ops/data_sources.yaml`.

## Acquisition record

For every asset, make an untracked JSON manifest from
`reproducibility/ops/manifests/acquisition_manifest.example.json`.  Record its
logical name, user-local path, upstream DOI/URL, byte count, the checksum
published by the release, and a locally computed SHA-256. Never commit signed
URLs, tokens, or the local manifest.

Run:

```bash
python reproducibility/ops/verify_acquisition.py --manifest my_acquisition.json
```

The Zenodo revision used here is record 20495192 revision 6, published
1 June 2026. Its `altair.zip` file is 57,025,221 bytes and the release record
publishes `md5:baf7974ac926445c726760fed57525ab` under CC-BY-4.0. The verifier
checks that MD5 and also requires a local SHA-256 computed after download; the
latter is local provenance, not a checksum attributed to Zenodo. Analysis code
is pinned to `czbiohub-sf/ops-paper-analysis` commit
`c51e707916fe42c29814cf2db3ea0b70fc4cc8ec`.

## Preparation

Acquire the processed phase and reporter H5ADs:

```bash
python studies/ops/acquisition/public_s3.py inventory \
  --output local/ops_public_h5ad_inventory.csv
python studies/ops/acquisition/public_s3.py download \
  --inventory local/ops_public_h5ad_inventory.csv \
  --output-dir local/ops_h5ad \
  --kind phase --kind reporter
```

Build all training assets:

```bash
python studies/ops/preparation/prepare_public_ops.py \
  --h5ad-dir local/ops_h5ad \
  --output-dir local/ops_prepared \
  --reporter-registry configs/ops/data/reporter_registry.csv
```

The three explicit stages are documented in
`studies/ops/preparation/README.md`. Outputs include:

```text
ops_prepared/exact/reporters/*.exact.h5
ops_prepared/phase172/phase172.npy
ops_prepared/phase172/{field_holdout_sanity,gene_holdout_main}.fold.npy
ops_prepared/canonical/atlas/*.csv
ops_prepared/canonical/reporters/<reporter>/{inputs,targets,split_*}.parquet
```

The exact builder uses assay-local unique coordinate linkage, dual `op_match`
and gene/guide concordance. The 172D builder regenerates the frozen feature
order from H5AD metadata and validates 1,024 source rows exactly. The canonical
export writes observed targets only, preserving heterogeneous endpoint blocks.

Audit the full 52-reporter topology after preparation:

```bash
python studies/ops/preparation/audit_prepared_ops.py \
  --prepared-root local/ops_prepared \
  --output local/ops_prepared/preparation_audit.json
```

The expected public census is 52 reporters, 99 reporter-screen assays and 73
screens; repeated reporters yield 81 whole-screen destination tasks.

## Adapter preparation from other normalized releases

Use a separate, untracked input manifest to point to four prepared tables:
`observations`, `reporters`, `assays`, and `pairing`.  The portable result
contains only logical names, byte counts, digests, schema IDs, preparation time
and the OPS inventory (10 systems / 52 reporters / 99 assays / 73 screens).
It drops local paths by design.

```bash
python studies/ops/workflows/prepare_ops.py \
  --input-manifest my_preparation_inputs.json \
  --output my_ops_derived_manifest.json
```

The adapter must then check that the 99 assay rows are valid reporter-screen
edges, reporter blocks retain their native endpoint masks, and each same-cell
analysis is supported by an exact assay-local one-to-one pairing. A physical
phase cell may participate once in several reporter assays. See
`studies/ops/schemas/tables.md` for required table columns.

## Data handling

- Phase features, targeted measurements, raw microscopy, crops, embeddings,
  and encoder weights stay in external data storage.
- Regenerate caches from verified source assets with the released commands.
- Use screen-matched controls to construct perturbation responses.
- Record split-manifest and software-version digests with every external run.
