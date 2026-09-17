# Assessing a new paired-measurement dataset

MorphoSuff separates dataset preparation from measurement-sufficiency analysis.
A new cell system can use the same prediction, response-fidelity and decision
functions after its observations, targets and experimental design are mapped
to the shared tables.

There are two starting points:

- **Paired measurements:** prepare the tables below, fit a reference predictor,
  then evaluate the held-out predictions.
- **Existing predictions:** export the canonical long prediction table and
  start at response analysis. The predictor can be trained outside MorphoSuff.

For complete examples of preparation through tier assignment, see
[OASIS and PERISCOPE](../studies/external/README.md).

## 1. Define the assessment

Specify the input modality, targeted readout, intended use and held-out unit.
Examples include estimating response magnitude for unseen perturbations,
ranking candidate hits, or assessing performance in an independent screen.
Pairing and replication determine the analyses available:

| Experimental information | Analysis it supports |
| --- | --- |
| Exact same-cell input–target linkage | Same-cell pairing controls |
| Matched control observations in each screen | Control-relative perturbation responses |
| Replicate wells or guides | Target reliability at the declared replicate unit |
| Repeated measurements across screens | Environment-transfer and cross-screen stability analyses |
| Independently available size, shape or density covariates | Covariate-matched pairing controls |
| Raw input images | Image-representation comparisons |

Keep the physical pairing unit explicit: a paired well supports well-level
analysis, whereas a same-cell analysis requires cell-level linkage. Choose the
replicate unit and endpoint scaling before interpreting reliability.

## 2. Prepare the input tables

The command-line fitting workflow reads CSV files with the following contents:

| File | Contents |
| --- | --- |
| `observations.csv` | One row per assay-qualified observation; identifiers for the observation, assay, perturbation, screen, well and field; `is_control` |
| `inputs.csv` | One row per `observation_id`, numeric input features, and the associated `reporter_id`, `screen_id`, `perturbation_id` and `is_control` metadata |
| `targets.csv` | Long measured targets: `observation_id`, `endpoint_id`, `y_true`, optionally `is_observed` |

Include the grouping identifiers needed for the chosen split in both
observation and input metadata. Keep target endpoints in their original sparse
assay structure: unmeasured endpoints remain absent or explicitly unobserved.
Select only features available from the intended input measurement.

For reusable dataset loading and metadata validation, describe these four
canonical tables in a `LocalManifestAdapter` manifest:

| Table | Required columns |
| --- | --- |
| `observations` | `observation_id`, `input_cell_id`, `target_cell_id`, `assay_id`, `perturbation_id`, `guide_id`, `screen_id`, `well_id`, `field_id`, `is_control` |
| `reporters` | `reporter_id`, `reporter_name`, `biological_system`, `target_modality`, `endpoint_schema_id` |
| `assays` | `assay_id`, `reporter_id`, `screen_id`, `endpoint_ids` |
| `pairing` | `observation_id`, `input_cell_id`, `target_cell_id`, `assay_id`, `pairing_key`, `pairing_confidence` |

`guide_id` may be null when the study has no guides. The reporter and assay
counts are set by the new dataset. `observation_id` identifies an assay-specific
row; physical input identifiers can recur across different assays. Exact
pairing, when declared, is one-to-one within an assay.

A manifest stored beside its metadata tables can look like this:

```json
{
  "dataset_id": "my_paired_screen",
  "capabilities": {
    "exact_pairing": true,
    "guides": true,
    "screen_matched_controls": true,
    "repeated_screens": true,
    "covariates": false,
    "raw_images": false
  },
  "input_schema": {
    "schema_id": "my_morphology_v1",
    "representation": "tabular",
    "feature_names": ["feature_001", "feature_002"]
  },
  "tables": {
    "observations": "observations.csv",
    "reporters": "reporters.csv",
    "assays": "assays.csv",
    "pairing": "pairing.csv"
  }
}
```

Set the capability flags to match the experiment. Paths are relative to the
manifest. Validate the metadata structure and relationships with:

```bash
measurement-sufficiency validate-dataset canonical_manifest.json
```

The adapter reports the declared capabilities; analysis functions use these
declarations to check whether the required design is available. Optional
adapter payloads use `input_cell_id` and `target_cell_id` for loading; join
them to observations to construct the observation-keyed CSVs used by the
fitting CLI.

For normalized source tables with different column names, the existing
[mapping template](../reproducibility/ops/manifests/upstream_to_canonical.example.json)
and extractor provide a preparation route:

```bash
python studies/ops/workflows/extract_ops_canonical.py \
  --upstream-manifest my_upstream_manifest.json --output-dir canonical
```

This extractor accepts CSV, TSV and Parquet and writes canonical CSVs, a local
manifest and provenance. A custom `MeasurementDatasetAdapter` can instead load
HDF5 or streamed arrays. The external workflows include dataset-specific
loading and joins.

## 3. Fit on a declared split, or import predictions

Install `.[sklearn]` for the built-in predictors. This example holds out
`perturbation_id` groups and retains controls for the later response analysis:

```bash
measurement-sufficiency split gene observations.csv gene_split.csv \
  --folds 5 --seed 0 --control-column is_control

measurement-sufficiency run-fold \
  --inputs inputs.csv --targets targets.csv \
  --assignments gene_split.csv --split-name gene --fold 0 \
  --model ridge --features feature_001,feature_002 \
  --output predictions_fold0.csv
```

Replace the feature names and run each fold using the same assignments.
`gene` is the CLI name for perturbation holdout, including compound holdout.
Verify that each evaluated reporter–endpoint–screen has matched controls in
its held-out fold. If the dataset has no controls, omit `--control-column`
from the split command and use analyses that do not require control subtraction.
`field` and `strict_whole_screen` provide alternative holdout units. Field
identifiers should be unique across screens.

Preprocessing is fitted on training data; validation data support model
selection; test data provide the reported assessment. Preserve these roles
when supplying predictions from another model.

The canonical prediction columns are:

```text
observation_id, reporter_id, endpoint_id, split_name, fold, model_id,
y_true, y_pred, screen_id, perturbation_id
```

Include `is_control` for response analysis, and retain predictions for control
observations alongside perturbed observations. Validate an imported file with:

```bash
measurement-sufficiency validate-predictions predictions_fold0.csv
```

## 4. Measure prediction and response fidelity

Basic prediction metrics are available from the CLI:

```bash
measurement-sufficiency metrics cell predictions_fold0.csv cell_metrics.csv
measurement-sufficiency metrics ko predictions_fold0.csv aggregate_metrics.csv
```

`cell` scores individual observations. `ko` scores perturbation-by-screen
means; it does not subtract control means. Both exclude declared control rows
from accuracy scores.

For control-relative response fidelity, use the shared analysis function:

```python
import pandas as pd
from measurement_sufficiency.analysis.response_fidelity import response_fidelity

predictions = pd.read_csv("predictions_fold0.csv")
fidelity = response_fidelity(predictions)
for name, table in fidelity.items():
    table.to_csv(f"fidelity_{name}.csv", index=False)
```

The function subtracts same-screen controls and summarizes response-magnitude
ranking, predicted-to-observed variance and top-5% hit recall. Put endpoints on
a scientifically appropriate common scale before combining their responses.
For an all-fold assessment, follow the dataset's declared pooling procedure;
the external workflows provide complete examples.

Estimate target reliability from the available replicate design, keeping its
measurement scale and aggregation compatible with the prediction assessment.
The [stability functions](../src/measurement_sufficiency/analysis/stability.py)
and [external analyses](../studies/external/README.md) cover replicate-half
estimation. Record the replicate unit, perturbation set and sampling depth with
the estimate. These determine what variation the estimate captures.

## 5. Assemble evidence and assign a measurement tier

The manuscript rule consumes one evidence record per target and assessment
context, with five numeric columns:

| Column | Meaning |
| --- | --- |
| `recoverability_r` | Pearson recovery at the declared evaluation unit |
| `magnitude_spearman` | Observed–predicted rank correlation of perturbation-response magnitudes |
| `variance_ratio` | Predicted / observed variance of response magnitudes |
| `top5pct_recall` | Recovery of the observed top 5% of perturbations |
| `reliability` | Replicate agreement for the targeted measurement |

Once these quantities have been derived, the existing API applies the rule:

```python
import pandas as pd
from measurement_sufficiency.analysis.tiers import tier_table

evidence = pd.read_csv("evidence.csv")
decisions = tier_table(
    evidence,
    group_columns=["reporter_id"],
    aggregate="none",
)
decisions.to_csv("measurement_tiers.csv", index=False)
```

This example expects one row per reporter. Add context columns to
`group_columns` when evaluating multiple cell systems or environments.
`aggregate="none"` retains those already-summarized evidence records.

The published rule first evaluates reliability, then quantitative-proxy,
ranking-proxy and measurement-required criteria. Missing or low reliability
yields `not_identifiable`; remaining incomplete or intermediate evidence is
`unresolved`. Exact thresholds and precedence are recorded in the
[measurement-tier protocol](../configs/ops/protocols/measurement_tiers.yaml).

The OASIS and PERISCOPE examples reuse this rule and its numerical thresholds
to examine its behavior across measurement designs. For a new application,
choose acceptance criteria for the intended scientific use and record them
before evaluating results. `OPSTierRules` allows caller-specified thresholds
within this decision structure. The separate top-level `assign_tier` API uses
`TierThresholds` and a different evidence set, including environment drop and
ambiguity; use `analysis.tiers` to reproduce the manuscript rule.

## 6. Report a reusable assessment

Keep the evidence table with its decisions, split assignments, target and
endpoint definitions, control construction, replicate units and model settings.
Report uncertainty at the biological unit appropriate to the experiment.
Pairing or within-screen input-permutation controls can establish how the
assessment responds when informative input–target associations are removed.

When evaluating a new cell system, the reusable components are the data
interface, analysis functions and decision procedure. Predictions, reliability
and the resulting target assignments are estimated in that system. This is the
cross-dataset workflow illustrated in Fig. 6f.
