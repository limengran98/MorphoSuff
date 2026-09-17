# OPS all-reporter same-cell falsification

This directory is the public campaign facade for the formal same-cell
information intervention. It materializes and executes the complete primary
design:

```text
52 reporters x 5 frozen held-out-gene folds x 6 information arms
  = 1,560 reporter-fold-arm jobs
```

The estimator is fixed to the reporter-specific MLP so that the only changing
quantity is the information intervention. The facade delegates every training,
validation, held-out prediction and metric operation to the study
runner in `studies/ops/runners/low_label/legacy_exact/scripts`; it does not
duplicate model or intervention logic.

## Six public arms

| Public arm | Question |
|---|---|
| `exact_same_cell` | Performance with the true phase-target cell linkage |
| `gene_screen_derangement` | Signal remaining after breaking cell identity within gene and screen |
| `covariate_matched_derangement` | Signal remaining after breaking identity while matching area, field density and eccentricity |
| `size_shape_only` | Information available from the 16 size/shape features alone |
| `phase_without_size_shape` | Information retained in the other 156 phase features |
| `cross_fitted_within_gene_screen_residual` | Exact-cell association after out-of-fold removal of gene-by-screen condition means |

The residual arm is a conditional association analysis. Its group-level KO
metric is intentionally not applicable because the condition mean has been
removed.

## Inputs

Prepare the public OPS files using `studies/ops/preparation/prepare_public_ops.py`.
The campaign consumes:

- the row-addressable 172-dimensional phase cache;
- the 52 exact reporter HDF5 caches;
- the 52-row reporter target table; and
- the observed technical-core target-feature dictionary.

Every path is supplied explicitly at plan creation.

## Build the frozen primary plan

```bash
SC=studies/ops/runners/same_cell

python "$SC/build_plan.py" \
  --phase-cache local/ops/phase172 \
  --exact-cache-root local/ops/exact/reporters \
  --target-table local/ops/reporter_targets.csv \
  --target-feature-dictionary local/ops/target_feature_dictionary.csv \
  --result-root local/results/same_cell_mlp \
  --output-plan local/plans/same_cell_mlp_all52.json
```

The builder verifies all 52 exact caches and enforces the 1,560-job identity
contract. `--reporters`, `--folds` and `--arms` permit explicit subsets for
development without changing the full primary plan.

Inspect a plan without launching training:

```bash
python "$SC/run_plan.py" \
  --plan local/plans/same_cell_mlp_all52.json \
  --run-root local/queues/same_cell \
  --gpu-ids 0,1 --workers 2 \
  --require-full-primary-contract --dry-run
```

## Execute and resume

```bash
python "$SC/run_plan.py" \
  --plan local/plans/same_cell_mlp_all52.json \
  --run-root local/queues/same_cell \
  --gpu-ids 0,1 --workers 2 \
  --require-full-primary-contract --continue-on-error
```

Each job writes its canonical result to a unique
`reporter/fold_<n>/<arm>/same_cell_result.json` directory. Re-running the same
plan verifies and skips completed identities. Queue state and per-job logs are
written below `--run-root`.

## Contract smoke

```bash
python studies/ops/runners/same_cell/smoke.py
```

The smoke test checks both CLI entry points, exercises the reusable information
interventions on synthetic data, constructs the complete 1,560-job plan, and
validates the resumable queue dry run. It does not require OPS data, a GPU or a
network connection.
