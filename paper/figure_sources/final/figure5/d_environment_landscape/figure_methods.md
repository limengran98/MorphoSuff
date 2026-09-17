# Figure 5d methods

Directed reporter-to-destination-screen evaluations were ordered by the difference between held-out-gene and strict whole-screen gene-level Pearson correlation. Phase and targeted-phenotype NTC shifts were displayed on an aligned logarithmic colour scale. Spearman correlations were calculated across 81 directions, with 95% confidence intervals obtained by resampling complete reporter clusters. For two-sided permutation tests, each shift and transfer-loss measure was first summarized by its median within each of the 34 reporters; reporter-level shift summaries were then permuted 3,000 times, with plus-one correction. The two FeRhoNox directions were linked to registered screen-specific NTC medoid images selected without reference to transfer performance.

## Analysis specification

- Statistical unit: directed reporter-to-destination-screen evaluation.
- Displayed units: 81 directions nested in 34 reporters and spanning 66 destination screens.
- Consensus: unweighted median of the frozen ten predictors.
- Ordering: ascending `gene_minus_strict_screen_pearson`, with stable reporter and screen tie-breaks.
- Association inference: direction-level Spearman correlations and reporter-cluster bootstrap confidence intervals; two-sided P values permute reporter-median shift summaries 3,000 times, with plus-one correction, as in the frozen source.
- Sign convention: archived correlations with screen penalty were multiplied by −1 to display positive transfer loss.
- Microscopy: all two directions of the prespecified FeRhoNox repeated-screen case; each image is the outcome-agnostic phase/geometry NTC medoid for that screen.
- Missingness: no plotted direction is missing the primary transfer-loss or NTC-shift variables.
- Leakage: microscopy selection did not use model performance or transfer loss.
