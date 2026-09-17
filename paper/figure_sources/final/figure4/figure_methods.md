# Figure 4 methods

All panels are plotting-only views of frozen source tables. No model is fit,
selected or tuned by a Figure 4 builder.

Panel **a** uses one reporter as the inferential unit (*n* = 52). Cell- and
KO-level Pearson correlations are arithmetic means across five frozen
held-out-gene folds from a fixed reporter-specific MLP. Exact-minus-control
effects are paired by reporter. Displayed intervals are reporter-resampled 95%
bootstrap intervals for the mean paired effect; folds and cells are not treated
as replicates.

Panel **b** pairs exact-cell recovery with cross-fitted within-gene×screen
residual recovery within each of 52 reporters. Reporter order is determined by
the residual score solely for display. The marginal difference summary uses a
reporter bootstrap; the 52 reporter-level differences remain visible.

Panel **c** uses three examples frozen before this redraw. Heatmaps contain 12
frozen gene×screen response profiles per reporter. Each predicted element is
the unweighted median of ten aligned out-of-fold model predictions. Magnitude
plots retain all 1,000 held-out genes per reporter. Each heatmap pair uses a
reporter-specific symmetric 97.5th-percentile absolute scale, which is printed
on its color bar and is not used for cross-reporter magnitude comparison. The
pRb microscopy crop and segmentation identifier are fixed in the local
selection manifest.

The magnitude axes retain their full linear ranges and all 1,000 paired values.
The pRb inset additionally enlarges 0–5 on both axes: 996 gene pairs are within
that square, while four are outside and remain visible in the main plot. All
reported correlations and variance ratios still use the complete frozen gene
set. Increased point size and opacity affect visibility only; neither the main
view nor the inset uses a transformed magnitude scale.

Panel **d** retains two frozen physical reporter-to-destination-screen examples.
Summary distributions contain all 81 directions for each of ten methods. Each
half-raincloud shows every direction, the 5th–95th percentile, interquartile
range and median. No averaging across methods is used.

Dense heatmaps, point clouds and microscopy are rasterized inside otherwise
vector PDF/SVG outputs. Text and axes use the shared DejaVu Sans typography.
