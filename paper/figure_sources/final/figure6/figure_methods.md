# Figure 6 methods

The figure asks which scientific uses remain supported when a targeted
measurement is predicted. Panels a–c summarize OPS reporter-level evidence;
d,e assess a specified external prioritization use; f presents the original
response criteria across measurement contexts. Rendering uses deposited
results without fitting predictors, searching thresholds or resampling.

## OPS response replacement and decisions

Panels a–c retain all 52 reporters. The replacement estimator is the elementwise
median of ten frozen checkpoint-average predictions at gene×screen level,
followed by equal-screen gene averaging. Ensemble recoverability is the
arithmetic mean of five fold-specific endpoint-macro Pearson correlations,
not the median of scalar benchmark method scores.

Panel a displays downstream retention with reporter-level distributions,
medians and interquartile ranges. Utility averages the cohort-percentile ranks
of top-5% hit recall and top-ten biological-process, cellular-component and
protein-complex Jaccard overlap, excluding magnitude rank. Panel b uses the
same four utility components mapped against the other 51 reporters' training
references. Its vertical line is the frozen recovery criterion of 0.70; the
horizontal cohort median is descriptive. The six directly labelled reporters
and domain-sensitive diamonds follow the deposited display specification.
Additional axis padding preserves complete markers without changing values.

Panel c displays retained utility, margins to five response criteria and exact
tier retention across 36 frozen threshold scenarios. The baseline counts are
30 quantitative, 7 ranking, 3 measurement-required, 9 not-identifiable and
3 unresolved reporters. The numerical contracts remain with the atlas,
prediction–utility and decision-ledger components. The nested-LOO utility
calibration and complete destination-screen analyses appear in Supplementary
Figure 4a,b, using the reusable `b_nested_loo_audit/` and
`e_reporter_context_stability/` sources.

## External metabolic-loss prioritization

Panels d,e read Supplementary Data 3. All 217 held-out compounds are included;
all their doses and sources were excluded from fitting on 868 development
compounds. One scatter point is one compound, paired by compound identifier.
The score is signed metabolic-activity loss relative to same-plate DMSO,
averaged over eight assayed concentrations after the deposited well/source
aggregation. Negative loss means increased activity. No point is removed,
jittered or substituted by a density estimate.

Panel d uses the fixed brightfield Ridge predictor (`BF_Ridge100`) in
`A_all_heldout` at a 5% selection budget. Both selected sets contain eleven
compounds, with ten shared and twelve in their union. Blue circles mark shared
hits, gold triangles predicted-only hits, rose open squares measured-only hits
and muted circles neither set. Equal axes span −0.15 to 1.00; identity is a
scale reference. Marginal histograms contain the same 217 compounds using 23
common equal-width bins and one shared count scale. Dashed separators lie at
the midpoint between losses ranked eleven and twelve; their coordinates are
0.22308898685277317 (measured) and 0.19146541468621303 (predicted). Both boundaries
are untied and exactly reproduce the deposited memberships. They represent
the existing selection budget, not measurement-sufficiency thresholds. Pale
shading marks the intersection. The compact “10/11 retained” annotation is
6.8 pt. It is grouped with the sample count and selection key in the space
beside the scatter, without covering plotted points or marginal histograms.

Panel e retains six predictors in fixed input/model order: brightfield Ridge
and HGB, fluorescence-derived cell-count Ridge and HGB, and acquisition-metadata
Ridge and HGB. The 6×11 matrix uses the same eleven measured hits, ordered by
descending measured loss with compound identifier as the deterministic tie
key. Filled cells are recovered hits; empty cells are misses, not missing
measurements or zero activity. Row sums are 10, 10, 8, 7, 0 and 1. The adjacent
recalls and 95% percentile intervals are copied from the deposited 2,000
compound-bootstrap results. Shared compounds do not make models independent
replicates. The dashed reference is the random fixed-size expectation 11/217,
not the correspondence-permutation mean; overlapping intervals do not establish
predictor superiority or equivalence.

Input labels appear once above each model pair, in 6.3 pt semibold text. The
6.2 pt Ridge/HGB labels are right-aligned beside the matrix. The matrix and
recall axis occupy 34.4 and 34.6 mm widths, respectively, and their six rows
share the same vertical positions. The compact key sits below the data axes.
These text and spacing choices do not alter the model order or data encoding.

Exact external-use rows, histogram counts, boundaries and compound memberships
are in `source_data/external_use/`. These signed-loss selections assess a
different use from the broad response-fidelity criteria in f, whose values
and assignments are preserved. The existing fixed/stained images support a
brightfield-based prioritization assessment, not an unstained live-cell
acquisition claim. Repeat-reference and budget evidence is in Supplementary
Figure 4c,d.

## Cross-context criteria and rendering

Panel f treats each external row as a distinct measurement context rather
than a biological replicate. Reliability uses cell halves for OPS,
dose-averaged well halves for OASIS and guide halves for PERISCOPE. Red rings
identify the first failed criterion; external rows are not pooled.

`code/build_figure6.py` composes the 183 × 210 mm figure using the native
components and shared `code/figure6_layout.py` functions. Four nominal rows
have relative heights 48:57:65:40; the first contains a,b and the third d,e.
Canonical clean panels in `panels/` use identical physical geometry without
letters or explanatory headings. Only the compositor adds 9 pt bold letters.
DejaVu Sans ordinary text is at least 6 pt; the manuscript places the figure
at 168 mm width. PDF/SVG retain editable text and PNG/TIFF exports are 600 dpi.
`published/figure_source_manifest.csv` records input hashes and
`published/figure_qa_report.json` records scientific and layout checks. Final
inspection covers text, complete marker extents and the manuscript-size PDF.
