# OPS scientific protocols

The task maps phase morphology to a sparse, reporter-specific targeted
phenotype block.  Each observed reporter-screen assay retains only its native
endpoint set.  Evaluation aggregation (cell, perturbation response, physical
screen, reporter-screen assay, reporter) is distinct from the split that
created the held-out data.

## Registry and consensus

Use only the ten names in `configs/ops/methods/final_ten.yaml`: Ridge, GBDT,
CatBoost, MLP, TabM, scButterfly, ResMLP, MultiTab, MIDAS, and scPair.  Align
out-of-fold predictions by observation, reporter and endpoint, then take the
unweighted element-wise median where a consensus is specified.  Do not
performance-weight methods, average a legacy roster, or use methods as
independent replicates.

## Split contracts

- **Field:** hold out complete `field_id` groups.
- **Gene:** hold out all guides for a perturbation gene.
- **Strict screen:** hold out an entire destination screen from fitting, tuning,
  normalizer fitting and label-budget selection.  Transfer is directed and
  records source and destination.

The exact constraints are versioned in `configs/ops/splits/` and must be
materialized as immutable manifests before fitting.

## Full label and low label

The full-label benchmark uses all eligible labels and all three split types.
The low-label calibration is an exploratory, frozen target-panel experiment at
0.1%, 1%, and 20% labels with a separately frozen matched 100% anchor.  Its
five outer gene folds and sampling manifests must prevent donor exposure from
test partitions.  It does not redefine the full-atlas measurement map.
The reusable package exposes the exact-budget sampler. The corresponding
model-specific study runners, complete local dependency closure and frozen
campaign configurations are released under
`studies/ops/runners/low_label`; all data and output paths are caller supplied.

## Same-cell and image protocols

The same-cell falsification uses all 52 reporters and five held-out-gene folds:
exact pair, gene×screen derangement, covariate-matched derangement, size/shape
only, phase minus size/shape, and cross-fitted within-gene×screen residuals.
Residual condition means are fit out of fold.  The primary estimator is MLP;
the reporter is the statistical unit. The complete 1,560-job campaign is
materialized by `studies/ops/runners/same_cell/build_plan.py` and executed by
`studies/ops/runners/same_cell/run_plan.py`. Full-contract validation and
resumable execution preserve the reporter-fold-arm identity; the accompanying
CPU contract smoke exercises the intervention seams and the full plan without
accessing study data.

The image representation comparison is exploratory and limited to a frozen
six-reporter panel. Images and checkpoints remain external, with inventory
and checksum validation. It is not an image-learner upper bound. The complete
crop, frozen DINOv2/Cytoland embedding, reporter-head and Cytoland partial-
fine-tuning chain is released under `studies/ops/runners/raw_image`. All paths
are caller supplied, upstream repositories are revision pinned, public
checkpoints are SHA-256 checked, and a CPU-only synthetic smoke test exercises
the frozen-head path end to end. See that directory's README for commands.

## Measurement tiers

The reference rule evaluated in OPS/A549 uses recoverability, magnitude rank,
amplitude, top-5% hit recall (`top5pct_recall`, produced by
`analysis.response_fidelity.top_fraction_recall`; not the absolute top-k
quantity) and reliability to label a reporter as quantitative proxy,
ranking proxy, measurement required, not identifiable, or unresolved.  Domain
sensitivity is a flag, never an additional tier.  Thresholds are in
`configs/ops/protocols/measurement_tiers.yaml`.

The external OASIS and PERISCOPE applications retain these operating points
for comparison. For a new dataset, define acceptance criteria for its intended
use before final evaluation; see the [new-dataset guide](adapt_new_dataset.md).
