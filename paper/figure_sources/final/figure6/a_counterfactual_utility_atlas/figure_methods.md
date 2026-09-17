# Figure 6a methods

One row represents one of the 52 frozen OPS reporters. Reporter order and the five counterfactual response-replacement utility measures are read directly from the final Figure 6 display table. Rank and top-hit recall summarize response ordering; Gene Ontology biological-process, cellular-component and protein-complex Jaccard indices summarize retained biological programmes. No value is re-estimated, normalized or imputed by the plotting script. Half-violin densities use a common Gaussian bandwidth of 0.055 on the shared 0–1 scale and are descriptive summaries of the 52 reporter values. The right-hand index is the frozen scientific utility index. Five reporter labels are fixed in the plotting code to match the manuscript examples.

## Analysis specification

- Statistical unit: reporter (`n=52`).
- Source: final Figure 6 canonical display tables for panels a and c.
- Reporter order: frozen `display_order` from the counterfactual response-replacement table.
- Missing plotted utility values: prohibited; the builder fails if any are found.
- Merge: one-to-one on `reporter_slug`; the builder requires 52 unique reporters.
- Inference: none. Density, median and IQR marks are descriptive reporter-level summaries.
- Exclusions and imputation: none.
- Claim boundary: the panel shows utility retained after counterfactual response replacement; it does not by itself certify assay substitutability.
