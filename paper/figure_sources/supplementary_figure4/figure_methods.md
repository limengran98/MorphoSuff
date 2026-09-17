# Figure methods and claim boundary

All panels render frozen outcome tables. The figure renderer does not train
predictors, recompute uncertainty or choose thresholds. OPS source components
are in `../final/figure6/`; external-use evidence is in Supplementary Data 3.

## Reporter-level utility calibration

In panel a, one point and one bootstrap unit are one of 52 reporters. The four
features are recovery, response-rank fidelity, amplitude fidelity
`exp(-abs(log(variance_ratio)))` and within-screen reliability, all for the
coherent response-replacement ensemble. Utility averages top-5% recall and
top-ten biological-process, cellular-component and protein-complex Jaccard
percentiles. Training references and scaling exclude each held-out reporter;
inner leave-one-reporter-out selection chooses the utility Ridge penalty.
The four direct labels identify the largest absolute outer-holdout residuals.

The forest intervals are the deposited 95% reporter-percentile intervals from
10,000 resamples (seed 20260831). Rank statistics preserve mathematical utility
ties by rounding a temporary copy to 12 decimals; displayed coordinates remain
unrounded. The residual view shows prediction minus observation, mean bias and
mean ±1.96 standard deviations. This panel and Figure 6b use training-reference
utility; Figure 6a,c use whole-cohort percentiles. The complete calculation
contract is in `../final/figure6/b_nested_loo_audit/figure_methods.md`.

## Destination-screen evidence

Panel b includes all 81 directed destination-screen evaluations for the 34
reporters observed in multiple screen contexts. Each point is the unweighted
median of ten benchmark method scores, rather than a score of the
response-replacement ensemble. Diamonds summarize reporter medians across
destinations and segments show their observed ranges. Rows follow the mean of
the reporter's recovery and magnitude-rank medians, continued down two blocks
of 17. Blue remains at or above 0.70 in every destination, grey remains below
and gold crosses. The full five-criterion tier is not recalculated by
destination. Reporter-associated variation is descriptive; destination points
are nested in reporters and are not independent reporter replicates.

## External repeat reference and selection budgets

Panels c,d use the fixed brightfield Ridge predictor with the deposited
768-coordinate DINO representation and paired normalized metabolic activity.
The penalty (alpha 100) was fixed before final fitting and outcome evaluation.
Development contains 868 compounds; all doses and sources of the 217 held-out
compounds are excluded from fitting. The score is signed loss relative to
same-plate DMSO, averaged over eight assayed concentrations after the deposited
well/source aggregation. Training and input provenance are documented in
Supplementary Data 3; the all-compound scatter and six-predictor comparison
appear in Figure 6d,e.

Panel c uses the 94 compounds with complete matched concentrations in both
source roles. Source 0 is 26; source 1 is 27 for 38 compounds and 30 for 56.
The displayed intervals are the deposited paired recall differences from
2,000 compound-bootstrap samples, keeping entire dose/source groups linked.
Duplicate bootstrap draws receive distinct instance positions. These intervals
condition on the fitted model and observed sources, not new donors or batches.
Matching 4/5 point counts provide a measured-repeat reference, not evidence of
equivalence.

Panel d uses three budgets fixed before final scoring. N = 217 gives
k = 11, 22 and 44; N = 94 gives k = 5, 10 and 19. All four curves use the same
Ridge predictor. Overlapping points retain their exact values, without jitter.
The full results include source-pair strata and all six predictors. Signed-loss
prioritization and the broad response-fidelity assessment in Figure 6f have
distinct estimands; the latter's values and assignments remain unchanged.
The endpoint is metabolic activity, not confirmed cell death. Images came from
the existing fixed/stained experiment, so this assesses brightfield-supported
prioritization rather than an unstained live-cell acquisition pipeline.

## Rendering

The composite is 183 × 130 mm. Shared drawing functions are in
`../final/figure6/code/figure6_layout.py`; 9 pt bold letters are added only by
the compositor. Clean panels share the geometry of their composite positions
and contain neither letters nor explanatory titles. Ordinary text is at least
6 pt in DejaVu Sans. PDF and SVG retain live text; PNG exports are 600 dpi.
Both text and complete marker extents are checked at final size.
