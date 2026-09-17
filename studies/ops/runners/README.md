# OPS biological-model runners

The common ten-method full-label entry point, fold materializer and strict
whole-screen builder are documented in [`full_label/README.md`](full_label/README.md).
The commands below describe the biological-model implementation layer that the
unified facade invokes for scButterfly, MIDAS and scPair.

This directory contains the complete study-owned execution path for the three
biological model families retained in the OPS benchmark:

- scButterfly: independent exact-pair reporter specialists with phase and
  phenotype pretraining followed by alternating paired translation;
- MIDAS: one shared phase-plus-sparse-reporter multimodal model; and
- scPair: independent reporter specialists built from the pinned public
  scPair source modules.

`run_external_fold.py` handles canonical data loading, frozen role assignment,
train-only preprocessing, model fitting, validation-only checkpoint selection,
test prediction and canonical observation-level export.  It accepts CSV or
Parquet tables and contains no machine-specific data or result path.

## Input tables

The input table contains one row per `observation_id`, with `reporter_id`,
`screen_id`, `perturbation_id` and the 172 phase feature columns.  The target
table is long-form:

```text
observation_id,reporter_id,endpoint_id,y_true,is_observed
```

The assignment table contains exactly one role per observation in a fold:

```text
observation_id,fold,role
```

Roles are `train`, `validation` and `test`.  The same manifests are used by
all model families.  Preparation of these tables from the public OPS release
is described in `docs/ops_data_and_preparation.md`.

## Environment and upstream source

Install this repository with the PyTorch extra.  scButterfly and MIDAS use the
study-owned OPS adapters included under `studies/ops/external_runners`.
scPair additionally loads the public model source from a caller-supplied clone:

```bash
git clone https://github.com/quon-titative-biology/scPair.git external/scPair
git -C external/scPair checkout --detach c585949ca8ea1314f5e68b260e3d9c5b2dabe61c
```

The runner verifies the pinned `scpair/model.py` SHA-256 before training.
The scButterfly and MIDAS upstream revisions and the exact OPS adapter hashes
are recorded in `studies/ops/model_sources.yaml`.

## Contract check and smoke run

Set these paths once:

```bash
INPUTS=local_data/ops_inputs.parquet
TARGETS=local_data/ops_targets.parquet
ASSIGNMENTS=local_data/gene_assignments.csv
```

A side-effect-free contract check is:

```bash
python studies/ops/runners/run_external_fold.py \
  --method midas \
  --inputs "$INPUTS" --targets "$TARGETS" --assignments "$ASSIGNMENTS" \
  --split-name gene --fold 0 --output-root runs/contracts/midas \
  --contract-check-only
```

The executable one-reporter smoke paths are:

```bash
python studies/ops/runners/run_external_fold.py \
  --method scbutterfly \
  --inputs "$INPUTS" --targets "$TARGETS" --assignments "$ASSIGNMENTS" \
  --split-name gene --fold 0 --output-root runs/scbutterfly_smoke \
  --device cuda --smoke

python studies/ops/runners/run_external_fold.py \
  --method midas \
  --inputs "$INPUTS" --targets "$TARGETS" --assignments "$ASSIGNMENTS" \
  --split-name gene --fold 0 --output-root runs/midas_smoke \
  --device cuda --smoke

python studies/ops/runners/run_external_fold.py \
  --method scpair --scpair-source external/scPair \
  --inputs "$INPUTS" --targets "$TARGETS" --assignments "$ASSIGNMENTS" \
  --split-name gene --fold 0 --output-root runs/scpair_smoke \
  --device cuda --smoke
```

`--smoke` uses one reporter, one epoch and one update per stage.  It verifies
the complete I/O, optimization, checkpoint and export chain; it is not a
benchmark result.

## Full fold

Remove `--smoke`, retain `--reporters all`, and supply the frozen method
configuration:

```bash
python studies/ops/runners/run_external_fold.py \
  --method scbutterfly \
  --inputs "$INPUTS" --targets "$TARGETS" --assignments "$ASSIGNMENTS" \
  --split-name gene --fold 0 --reporters all \
  --config configs/ops/methods/biological_runners.json \
  --output-root runs/scbutterfly/gene/fold_0 --device cuda
```

Use `--method midas` or `--method scpair --scpair-source external/scPair`
for the other two methods.  Field and strict whole-screen runs use their own
frozen assignment manifest and corresponding `--split-name`; the runner logic
is otherwise unchanged.

Each completed run writes:

- `run_contract.json`;
- `preprocessing.json`;
- validation-selected checkpoint(s);
- `validation_trajectory.csv`;
- `predictions.csv` in the common observation-level schema; and
- `training_result.json` with provenance and SHA-256 digests.

No raw data, prediction result or trained checkpoint is stored in the source
repository.
