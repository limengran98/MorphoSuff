For each predictor, scores were averaged arithmetically across the predefined folds or transfer directions. The displayed consensus was then computed as the unweighted median across the ten frozen predictors; Q25 and Q75 quantify predictor spread. Units were ranked within each split with stable sorting. Difficulty tiers were fixed before plotting at r < 0.60, 0.60–0.75 and r ≥ 0.75. The microscopy manifest was frozen independently of the scores.

## Analysis specification

- Statistical units: reporters, physical screens and directional reporter–screen assays.
- Counts: Field 52/73/99; Gene 52/73/99; Whole-screen 34/66/81.
- Missing values: only units eligible for the frozen split are plotted; no score-based exclusions.
- Aggregation: fold/direction mean within model, then unweighted median across ten models.
- Q25–Q75: variation across models, not a sampling or biological confidence interval.
- Microscopy: one fixed case per statistical-unit row; the three cases jointly span all three splits and do not enter the numerical analysis.
- Leakage: the callout selection manifest records `selection_depends_on_score=False`.
