# MorphoSuff documentation

MorphoSuff evaluates whether a measured cellular input supports the intended
use of a targeted readout. Start with the workflow that matches your data.

## Use MorphoSuff

| Task | Documentation |
| --- | --- |
| Install the package and run a first assessment | [Quick start](../README.md#quick-start) |
| Assess a new cell system or paired dataset | [New-dataset workflow](adapt_new_dataset.md) |
| Evaluate predictions from an existing model | [Prediction format and analysis](adapt_new_dataset.md#3-fit-on-a-declared-split-or-import-predictions) |
| Run preparation through measurement-tier assignment | [OASIS and PERISCOPE examples](../studies/external/README.md) |
| Choose a predictor or integrate another method | [Model adapters](model_adapters.md) · [External model runners](external_model_runners.md) |

## Reproduce the study

| Resource | Documentation |
| --- | --- |
| Processed data and source datasets | [Data access](data_code_availability.md) |
| OPS acquisition and canonical preparation | [Preparation guide](ops_data_and_preparation.md) |
| Splits, metrics and measurement criteria | [Study protocols](ops_protocols.md) |
| Full-label, low-label and image comparisons | [Study runners](../studies/ops/runners/README.md) |
| Six main and six Supplementary figures | [Figure PDFs](../paper/figures/) |
| External measurement evidence and threshold sensitivity | [Supplementary Information](../paper/supplementary.pdf) |
| Brightfield-supported metabolic-loss prioritization | [Worked example](../studies/external/README.md#oasis-prioritize-metabolic-activity-loss) |
| Manuscript and Supplementary Information | [Main text](../paper/manuscript.pdf) · [Supplementary Information](../paper/supplementary.pdf) |
| Code-to-analysis mapping | [Source map](../reproducibility/ops/source_map.tsv) |
| OPS-native implementation identities | [Model-code provenance](ops_model_code_provenance.md) |

Supplementary Data 1–3 accompany the manuscript submission as separate
archives. They contain frozen study predictions, decision ledgers and
result-verification scripts; these archives are not part of this Git checkout.

## Package interfaces

- [Dataset adapters](../src/measurement_sufficiency/adapters.py): load paired
  datasets and describe their experimental capabilities.
- [Splits](../src/measurement_sufficiency/splits.py) and
  [training](../src/measurement_sufficiency/training.py): construct held-out
  assignments, fit sparse-target predictors and export common prediction tables.
- [Response fidelity](../src/measurement_sufficiency/analysis/response_fidelity.py)
  and [stability](../src/measurement_sufficiency/analysis/stability.py): quantify
  perturbation-response preservation and target reliability.
- [Measurement tiers](../src/measurement_sufficiency/analysis/tiers.py): apply
  the manuscript's decision rule to an evidence table.

See [Contributing](../CONTRIBUTING.md) for extending a dataset or model adapter,
and [Third-party notices](../THIRD_PARTY_NOTICES.md) for source licences.
