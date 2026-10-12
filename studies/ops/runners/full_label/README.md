# Full-label OPS benchmark

This directory is the single execution surface for the frozen ten-method
full-label benchmark.  Every method consumes the same 172-dimensional phase
table, sparse observed-endpoint target table and immutable role assignments,
then writes the same validated long prediction schema.

The frozen roster is:

| Training route | Methods |
|---|---|
| reporter specialist | Ridge, GBDT, CatBoost, MLP, TabM, scButterfly |
| shared reporter model | ResMLP, MultiTab |
| biological completion | MIDAS, scPair |

`run.py` routes the seven internal methods to `run_internal_fold.py` and the
three biological methods to `../run_external_fold.py`.  The routing changes no
data or split semantics.  Neural checkpoints are selected on validation data;
outer-test targets are loaded only after selection.

## 1. Prepare the public OPS release

The complete public-data path is documented in
`studies/ops/preparation/README.md` and can be run through
`studies/ops/preparation/prepare_public_ops.py`.  Its final output is a
`canonical_release.json` plus one canonical bundle per reporter.

## 2. Freeze strict whole-screen assignments

Field and gene assignments are exported with each reporter bundle.  Build the
global strict-screen plan once from assay topology and cell counts:

```bash
python studies/ops/runners/full_label/build_strict_screen.py \
  --canonical-root local_data/ops_canonical \
  --output-dir local_data/ops_strict_screen \
  --n-folds 5 --seed 20260722
```

The builder never opens target values.  A held physical screen is removed from
every reporter, and the emitted audit verifies that every frozen
reporter-to-destination-screen direction occurs exactly once.

## 3. Materialize one common fold

```bash
python studies/ops/runners/full_label/materialize_fold.py \
  --canonical-root local_data/ops_canonical \
  --split gene --fold 0 \
  --output-dir local_data/folds/gene/fold_0
```

For strict whole-screen, add:

```bash
  --split strict_whole_screen \
  --strict-screen-plan local_data/ops_strict_screen/strict_screen_plan.json
```

The resulting `inputs`, `targets` and `assignments` paths in
`fold_manifest.json` are used unchanged by all ten methods.

## 4. Train any frozen method

```bash
python studies/ops/runners/full_label/run.py \
  --method mlp \
  --inputs local_data/folds/gene/fold_0/inputs.parquet \
  --targets local_data/folds/gene/fold_0/targets.parquet \
  --assignments local_data/folds/gene/fold_0/assignments.parquet \
  --split-name gene --fold 0 \
  --output-root runs/full_label/mlp/gene/fold_0 \
  --device cuda
```

Replace `mlp` with any of:

```text
ridge gbdt catboost mlp tabm scbutterfly resmlp multitab midas scpair
```

TabM uses the pinned public source file:

```bash
git clone https://github.com/yandex-research/tabm.git external/tabm
git -C external/tabm checkout --detach 28e47ae301c92ec37787dde1ce923a0793f405b4
--tabm-source external/tabm/tabm.py
```

scPair uses the pinned public checkout:

```bash
git clone https://github.com/quon-titative-biology/scPair.git external/scPair
git -C external/scPair checkout --detach c585949ca8ea1314f5e68b260e3d9c5b2dabe61c
--scpair-source external/scPair
```

Pinned commits and source digests are recorded in
`studies/ops/model_sources.yaml`.  The same command applies to `field`, `gene`
and `strict_whole_screen`; only the materialized assignment path and
`--split-name` change.

## Contract and smoke checks

`--print-command` prints the exact resolved child command without writing a
run directory.  `--contract-check-only` validates the fold contract without
fitting.  `--smoke --device cpu` runs the complete training, selection,
prediction and export path on a compact reporter panel.

The synthetic regression suite exercises all ten training identities, the
unified routing surface and strict-screen materialization:

```bash
pytest -q tests/test_full_label_public_facade.py tests/test_external_ops_runners.py
```

Every completed fold contains `run_contract.json`, train-only preprocessing,
validation trajectory, selected checkpoint, canonical `predictions.csv` and a
checksummed `training_result.json`.
