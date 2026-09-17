# Scientific contract

This presentation summarizes the protocol frozen before final fitting and scoring
of the 217 reserved compounds. The original protocol hash and recorded chronology
are retained in `provenance.json`; the wording below is not a new preregistration.
The labels belong to an existing public resource, historically also used for the
earlier CellProfiler analysis, and were not inspected for effects in the new
brightfield-DINO pilot.

## Target, training and inference units

The target is the deposited plate-normalized RealTime-Glo metabolic-activity signal
(`Metadata_mtt_normalized`). Subtract the same-plate DMSO mean from the measured
signal and negate the signed difference to obtain metabolic-activity loss. Positive
activity responses are not counted as injury by taking an absolute value.

Use 868 development compounds (13,157 wells) and 217 held-out compounds (3,300
wells). Every assayed dose and production source of each held-out compound is
excluded from fitting. Model imputation and scaling are fitted to development
wells. Brightfield features and predictions have the declared same-plate DMSO
calibration. The 768 brightfield-channel DINO features do not include compound
identity, concentration, fluorescence or cell-count inputs.

Primary predictor: standardized Ridge with alpha = 100. Fixed sensitivity predictor:
histogram gradient boosting with 100 iterations, 15 leaf nodes, learning rate 0.1,
L2 regularization 10, no early stopping and seed 20260911. Fluorescence-derived
cell count and acquisition metadata are separate diagnostic inputs evaluated with
both predictor specifications. Metadata include row, column, dose position, plate
and production source, not compound identity or biochemical targets. Fluorescence
count is an abundance diagnostic, not part of the brightfield prediction route.

The independent unit for uncertainty is a compound, not a cell or well. Within a
compound and actual concentration, average wells within each source, then give
available sources equal weight, and finally average the eight assayed concentration
positions equally. Concentration ranges can differ between compounds; this summary
is not a common-range AUC or IC50. No values are silently omitted or imputed during
outcome scoring.

## Evaluation contexts and selection

The main context contains all 217 held-out compounds. A second context restricts
the same available-source average to the 94 compounds with eight identical actual
concentrations measured in both sources. Cross-source contexts on these same 94
compounds rank predictions from one source against measurement in the other.
Source 0 is source 26; source 1 consists of source 27 for 38 compounds and source
30 for 56. The 123 other compounds lack complete source-25 dose coverage. Thus
the complete-source comparison does not establish repeat agreement for all 217
compounds or for source 25. Actual 26/27 and 26/30 strata are also retained.

Rank signed loss from largest to smallest; break exact ties by compound identifier.
The primary budget is 5%, with k = ceil(N × 0.05): 11 for N = 217 and 5 for N = 94.
The 10% and 20% budgets are fixed sensitivity analyses, not alternative success
criteria. Recall is overlap/k; prediction and reference sets have the same k.

## Uncertainty and controls

Each context uses 2,000 compound-bootstrap samples (seed 20260911). Resampling
retains the full dose/source group and assigns duplicate draws unique instance
positions for ranking. Recompute top sets in each sample. The same bootstrap
draws are used for prediction and measured-repeat recall in each paired difference.
Intervals are percentile 95% intervals, conditional on fixed models and observed
production sources; they do not cover refitting, new-source or new-donor uncertainty.
No equivalence or non-inferiority margin is inferred from the results.

The fixed-size random selection expectation is k/N. Exact hypergeometric overlap
intervals in the score table describe random selections, not confidence intervals
on model performance. For brightfield predictors in the 217-compound context,
2,000 correspondence nulls exchange fixed predicted responses within plate × dose
position before compound aggregation. Here dose position is constant within plate,
so these are 61 within-plate strata; all test wells can move. This preserves the
specified acquisition structure and disrupts compound correspondence without
refitting. These nulls do not provide cross-source-specific randomization results.

## Relationship to response fidelity

The original OASIS analysis asks how faithfully response magnitudes and their
ordering are retained under the common measurement-sufficiency metrics. This
additional analysis asks whether a fixed follow-up budget can recover compounds
with the strongest signed decrease in metabolic activity. The brightfield-DINO
input route, included wells and signed dose-mean target differ from the original
CellProfiler magnitude analysis. Both analyses inform measurement use at different
levels; a useful task-specific ranking need not pass every full-response gate.
