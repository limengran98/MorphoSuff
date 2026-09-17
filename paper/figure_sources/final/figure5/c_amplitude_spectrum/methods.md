The frozen reporter-level table contains out-of-fold KO-response profiles obtained by element-wise unweighted median aggregation across ten aligned prediction models. For each reporter, magnitude-ranking fidelity is the Spearman correlation between observed and predicted KO-response magnitudes across 1,000 genes. Amplitude fidelity is the ratio of predicted to observed KO-response variance. Rows were sorted deterministically by amplitude fidelity, with reporter slug as the tie breaker; no values were normalized, clipped, imputed or re-estimated for plotting. The prespecified amplitude-compression boundary was a variance ratio below 0.5, and retained ranking was defined as magnitude-rank Spearman correlation above 0.5.

The p53 and NPM1 annotations are positioned in the existing lower margin of
the rank track to avoid the connected reporter curve and vertical data stems.
Leaders retain registration to the same reporter points; the data, axis limits,
panel geometry and all other predeclared callouts remain unchanged. Rendering
checks require both labels to avoid data markers, curve/reference lines and
other text and to remain clear of the adjacent variance track.

## Analysis specification

- Statistical unit: reporter.
- Rows in frozen input: 52 unique reporters.
- Aggregation: element-wise unweighted median of ten aligned out-of-fold model predictions, followed by reporter-level metrics.
- Ordering: ascending predicted/observed KO-response variance ratio; deterministic reporter-slug tie break.
- Exclusions: none.
- Missing plotted values: none.
- Frozen thresholds: variance ratio <0.5; magnitude-rank Spearman rho >0.5.
- Frozen counts: 11/52 below the variance threshold; 10/11 of those retain rank rho >0.5.
- Labels: pRb, p53, TOMM20, NPM1 and LysoTracker only, selected before this redraw.
- Biology colours are annotation only and do not alter ordering.
- Claim boundary: ranking fidelity does not establish quantitative amplitude recovery; the panel does not estimate an irreducible information ceiling.
