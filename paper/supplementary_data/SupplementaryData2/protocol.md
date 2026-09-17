# Supplementary Data 2: external-threshold sensitivity protocol

This is a descriptive sensitivity analysis of the existing OASIS and PERISCOPE applications. Their metric values and a historical OASIS rank-only sweep are already known. The analysis reuses the OPS perturbation ranges and hierarchy; it is not prospective threshold calibration, validation on previously unseen outcomes, or selection of new operating points. Inputs, scenarios, this protocol and executable are hashed before the first evaluation in this bundle.

## Fixed applications and evidence

The main set has four rows: OASIS metabolic activity and LDH release, and PERISCOPE anti-TOMM20 at the cell and guide-profile levels. Their five exact metric strings come from the admitted rows of `paper/figure_sources/final/figure6/f_cross_context_decisions/figure_source_gate_rows.csv`. Published baseline labels are read from the adjacent `figure_source_data.csv`.

The second set has eight rows: the same two OASIS biochemical readouts in production batches 25, 26, 27 and 30. Their metrics and baseline labels come from `paper/figure_sources/supplementary_figure1/source_data/external_context/batch_verdicts.csv`. No value is reverse-engineered from rounded manuscript tables. Main and batch rows retain their separate estimands and are not pooled as biological replicates or combined into a cross-dataset effect size.

These are the existing fold-summary applications, not the later 217-compound BF-DINO metabolic-loss prioritization analysis. The four fidelity metrics retain their reported five-fold medians. OASIS retains its original compound/plate aggregation and dose-averaged split-half reliability scalar. For PERISCOPE, cell and guide-profile denote fitting/input resolution; both rows evaluate held-out-gene response fidelity. Their reliability is the reported median across nine plates of agreement between disjoint guide partitions. Reusing a numerical rule does not make these reliability estimators, predictors or biological units equivalent. The OASIS batch hit-recall budgets are coarse (reported k = 7 in batches 25–27 and k = 3 in batch 30); no new hit selection is performed.

The primary canonical input is `inputs/context_metrics.csv`; source paths, row identifiers and SHA256 values are recorded. No prediction, target, reliability, transformation, model, method weighting, partition or metric is recalculated or changed.

## Exactly reused threshold grid

The implementation source is `paper/figure_sources/final/figure6/code/rebuild_coherent_statistics.py`, specifically its ordered `RULES`, `DIRECTIONS`, `assign` and scenario construction. Baseline thresholds are:

| Rule key | Baseline | Predicate |
|---|---:|---|
| recoverability_min | 0.70 | recovery ≥ threshold |
| magnitude_spearman_min | 0.70 | magnitude rank ≥ threshold |
| top5pct_quantitative_min | 0.60 | hit recall ≥ threshold |
| top5pct_ranking_min | 0.50 | hit recall ≥ threshold |
| reliability_min | 0.30 | reliability ≥ threshold |
| measurement_recoverability_max_exclusive | 0.60 | recovery < threshold: eligibility for M, not good performance |
| variance_ratio_min | 0.50 | variance ratio ≥ threshold |
| variance_ratio_max | 1.50 | variance ratio ≤ threshold |

The grid contains 32 single-threshold scenarios: eight thresholds × two directions × two schemes. Absolute perturbations are ±0.05 metric units; relative perturbations are ±10% of that threshold's baseline value. Four additional joint scenarios vary all eight thresholds together using the same two directions and schemes. Every changed threshold is rounded to ten decimal places, exactly as in the OPS implementation. The variance upper limit has reversed direction so that the two variance limits move toward 1.0 under the inherited `tighter` label and apart under `looser`. Every other rule uses the original direction, including the M-eligibility threshold. Direction names are inherited grid labels, not a guarantee of monotonic improvement of final hierarchical labels.

Baseline is retained as a separate 37th table row but excluded from all 32/4/36 sensitivity denominators. The scenario table is frozen before applying it to the 12 contexts.

## Hierarchy and criterion evidence

First assign N (`not_identifiable`) if reference reliability is missing or below its scenario threshold. If reference reliability passes but any required fidelity value is nonfinite, assign U (`unresolved`). Next evaluate Q (`quantitative_proxy`) using recovery, rank, both variance bounds and quantitative hit recall. Then evaluate R (`ranking_proxy`) using rank and ranking hit recall. Only after R is considered, assign M (`measurement_required`) if recovery is strictly below its M-eligibility threshold; otherwise assign U.

For every context/scenario, record all eight atomic criterion predicates, irrespective of which stage terminates classification. In particular, `measurement_recoverability_max_exclusive` being satisfied means M eligibility, not that a desirable fidelity criterion passed. Record changes in both thresholds and predicate statuses relative to baseline, plus final tier/proxy-status changes. Separately record the four quantitative-fidelity pass statuses: recovery, rank, amplitude (both variance limits combined) and quantitative top-5% recall. Count scenarios in which any component of this four-item vector changes, independently of reliability and the final tier. Aggregate quantitative and ranking fidelity predicates are also reported separately from reliability, so that an unchanged reliability-gated N label cannot hide threshold-sensitive fidelity evidence.

Numerical evidence is fixed in every scenario. Tier stability therefore does not prove that fidelity values are invariant to a different experiment or predictor; it also need not imply that every criterion predicate is stable when thresholds move.

## Outputs and summaries

- 444 assignments: 12 contexts × (baseline + 36 scenarios).
- 3,552 atomic criterion rows: 444 assignments × eight rules.
- Per-context summaries separately for single32, joint4 and all36: exact-tier and proxy-status stability, criterion-status changes and possible tiers (with baseline included in the possible-tier set).
- Scenario summaries retain main4 and OASIS-batch8 as separate descriptive groups.
- A candidate 12-row Supplementary Table 7 row fragment reports baseline, unchanged single32, unchanged joint4 and other possible tiers, grouped into main4 and batch8; full stability and fidelity-vector-change summaries remain in the CSVs.

The historical check changes only the rank minimum to 0.65, 0.60 or 0.55 for the two main OASIS readouts, with every other threshold at baseline. It is separately labeled `historical_rank_only_not_in_36`; these six re-evaluations are never added to the 36-scenario grid, stability denominators or candidate table counts. They reproduce the historical one-dimensional recipe on current frozen metric values, rather than claiming a newly calibrated threshold.

No model training, external-data download, bootstrapping, threshold optimization, manuscript edit or modification of earlier result tables is part of this analysis. Any implementation correction must be recorded without selecting a more favorable threshold range or dropping an unfavorable context.
