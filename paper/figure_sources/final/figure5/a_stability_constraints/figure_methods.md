Reporter-level recoverability was defined as the unweighted median of the ten frozen held-out-gene Pearson scores. Spearman associations were computed between recoverability and within-screen reliability, guide-half consistency, cross-screen consistency or log10 exact-linked cell coverage. Intervals are 95% reporter-bootstrap confidence intervals from 3,000 draws, with the seed recorded in the plotted source table. The overlap matrix reports pairwise-complete Spearman correlations in its lower triangle and pairwise-complete reporter counts in its upper triangle. The prespecified joint model used within-screen reliability, guide-half consistency and log10 exact-linked coverage in the 35 complete reporters. Its null distribution was reconstructed deterministically by permuting reporter outcome labels 3,000 times with seed 20260848 and refitting the same ordinary least-squares model. The model is explanatory and in-sample; no coefficient is interpreted as an independent causal effect.

## Analysis specification

- Statistical unit: reporter.
- Recoverability source: unweighted median of ten frozen reporter-level held-out-gene Pearson scores.
- Coverage: 52 reporters for within-screen reliability and coverage, 35 for guide-half consistency, and 34 for cross-screen consistency.
- Missingness: no imputation; pairwise-complete observations are used in the overlap matrix and complete cases in the joint model.
- Uncertainty: 95% reporter-bootstrap confidence intervals, 3,000 draws per association.
- Joint model: 35 complete reporters; within-screen reliability, guide-half consistency and log10 exact links; in-sample R².
- Permutation: reporter outcome labels, 3,000 draws, seed 20260848. The displayed null was deterministically regenerated and exactly reproduces the frozen P value.
- Leakage: these explanatory measurements do not enter model training or consensus weighting.
- Can say: recoverability covaries with overlapping stability and coverage constraints.
- Cannot say: any one constraint independently causes reporter recoverability.
