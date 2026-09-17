# Figure 5e methods

For each gene, cells were matched to the nearest morphology neighbour either in the engineered 172-dimensional phase representation or in a 32-dimensional principal-component representation of raw phase thumbnails under the frozen cross-guide diagnostic. Target-state discrepancy for the nearest match was divided by the median discrepancy obtained from random matches. Ratios below one indicate that morphologically similar cells are more similar in the targeted state than random cells; ratios at or above one indicate unresolved conditional ambiguity. All 5,990 available gene-level ratios are represented. Half-density envelopes were estimated in log2-ratio space; points show individual genes, thick intervals show the interquartile range, thin intervals show the 10th--90th percentiles, and open circles show medians.

## Analysis specification

- Statistical unit: gene within a frozen reporter--screen case and neighbour representation.
- Coverage: 5,990 rows across three cases and two representations; per-group n is 992--1,000.
- Exclusions: rows absent from the upstream cross-guide constrained diagnostic are not imputed (8 and 2 missing phase172 gene rows in OPS0047 and OPS0067, respectively).
- Display statistic: endpoint discrepancy for the nearest morphology match divided by the median discrepancy for a random match; 1 is the no-advantage reference.
- Pairing: the panel shows marginal gene-level distributions and does not connect genes across representations.
- Transform: KDEs are evaluated in log2-ratio space and displayed on a base-2 logarithmic axis. Medians and quantiles are computed on the original ratio scale.
- Leakage check: source rows are restricted to the frozen cross-guide diagnostic; no model fitting or label-informed re-selection occurs in this plotting step.
- Images: none; microscopy evidence is reserved for panels b and d.
