# Supplementary Figure 4a methods

One point is one reporter (`n=52`). Four predictors are coherent checkpoint-average ensemble recovery, response-rank fidelity, amplitude fidelity `exp(-abs(log(variance_ratio)))`, and within-screen reliability. Recovery is the arithmetic fivefold mean of endpoint-macro Pearson for the same gene×screen elementwise median ensemble used to replace responses.

`../code/rebuild_coherent_statistics.py` recomputes strict nested leave-one-reporter-out Ridge from compact frozen inputs. Each outer training set contains 51 reporters and each inner training set 50. The four utility components are Top-5% recall and top-ten GO BP, GO CC and protein-complex Jaccard overlaps. Training targets average average-rank percentile scores. A held-out component is mapped to the average training rank percentile if its value occurs in training; otherwise it is mapped to the training fraction strictly below it, bounded by 0 and 1. Neither target mapping nor StandardScaler uses a held-out reporter. Inner mean squared error selects alpha from 0.01, 0.1, 1, 10, 100; exact ties choose the first value. The outer training fit predicts the held-out target. No image-to-reporter predictor is retrained.

The agreement display includes identity and Bland–Altman residual views; correlation is not treated as agreement. Four direct labels are the largest absolute outer-LOO residuals. Associations are Spearman correlations with deterministic 95% reporter-percentile bootstrap intervals (10,000 resamples, seed 20260831). A temporary utility copy is rounded to 12 decimals before average ranking to preserve mathematical ties; saved targets, predictions, R2, residuals and plot coordinates are not rounded. No cells, genes or folds are independent bootstrap units.

Supplementary Figure 4a and Figure 6b use training-reference utility targets; Figure 6a,c use whole-cohort percentile ranks of the same components. Run `python -B build_panel_b.py --verify-statistics` for read-only checks of the headline, four intervals, Figure 6b summary and a known-answer tie test.

## Analysis specification

- Statistical and bootstrap unit: reporter (`n=52`). No cell, gene or fold pseudoreplication.
- Portable calculation: `../code/rebuild_coherent_statistics.py`; three local leaf CSVs suffice. The four-feature reporter-level Ridge is fitted, but the ten image-to-reporter predictors are never retrained.
- Outer training references use 51 reporters; inner references use 50. StandardScaler is fitted within each training partition. Alpha grid: 0.01, 0.1, 1, 10, 100; minimum inner MSE, first grid value on an exact tie.
- Recovery and response fidelity describe the same checkpoint-average median ensemble. The numerical summary is `figure_source_nested_loo_summary.csv`.
- Association intervals: deterministic 10,000-replicate reporter-percentile bootstrap, seed 20260831; utility rank ties stabilized at 12 decimals.
- Label rule: four largest absolute outer-LOO residuals; the source records every selected reporter.
- The loader verifies headlines and associations at absolute tolerance `1e-12`. `--verify-statistics` checks all intervals, the known-answer tie case and Figure 6b summary.
- Supplementary Figure 4a and Figure 6b use training-reference utility; Figure 6a,c use whole-cohort utility. Association and calibration do not establish causal determinants or external-dataset validation.
