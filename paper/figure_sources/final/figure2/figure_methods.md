# Figure 2 methods

Figure 2 numerical tables are rebuilt from the compact frozen partition metrics in
`source_data/frozen_partition_scores.csv.gz`; rendering then reads those local
per-method tables. `code/rebuild_benchmark_sources.py` implements the stated
arithmetic mean over the predefined partitions.
No model is trained or selected and no cell-level prediction is recalculated.
Reporter, physical-screen and reporter × screen assay remain distinct reporting units.
Cross-model consensus is the unweighted median across ten aligned methods. Splits,
eligible directions, missing-value rules and method identities remain frozen.

Reporter-level method comparisons use the prespecified reporter-specific MLP as
reference. Within each metric and held-out test, reporters are paired by identity:
52 in field/gene tests and 34 repeated reporters in whole-screen transfer. Each
reporter score is the arithmetic mean over its five predefined partitions or eligible
destination-screen evaluations. The effect is median(method score − MLP score) over
these matched reporters, not the difference between their marginal medians.
Two-sided paired Wilcoxon signed-rank tests use `zero_method="wilcox"` and
`method="auto"`; Holm correction spans all 54 reporter-level comparisons (nine
comparators × three tests × two metrics). A difference meets the prespecified
statistical and practical criteria only when `p_holm_reporter54 < 0.05` and the
absolute median paired effect is at least 0.02. Failure to meet these criteria is
not an equivalence result.

The recomputed 54-row inference table is deposited at
`source_data/statistics/reporter_level_primary_inference.csv`. Run
`python -B code/fig2_data.py --verify-inference` to reproduce its primary statistics
from the frozen reporter tables without writing outputs. This optional check uses
SciPy and was verified with SciPy 1.11.4, NumPy 1.26.4 and pandas 2.1.4.
The table reports Holm adjustment over the prespecified 54 comparisons.

Panel a is exported natively from the accepted editable PowerPoint at
`a_workflow/source/Figure2a.pptx`.
The adjacent `export_manifest.json` binds the current PPTX and versioned
`a_workflow/source/Figure2a.pdf` hashes. A matching native PPT/PDF text note,
"Example cell crops", distinguishes the illustrative field-row images from
the actual whole-field held-out unit.
Rebuilds verify and import the bound PDF. Biological illustrations are schematic;
the registered phase/reporter image pairs retain their source provenance. Labels,
connectors and graphical encodings remain native PDF objects. Panel b shows
reporter-level distributions; c ranks physical screens by consensus and compares matched
interquartile spreads; d shows reporter × screen assays. Every quantitative panel
groups KO-response on the left and cell-level results on the right. The frozen
table IDs b/c, d/e and f/g map to current panels b, c and d, respectively;
no scientific table identifier is relabelled. Named reporter guides in d
are annotations only and do not alter the values or their order.

Screen and assay summaries are descriptive because units share reporters, genes and
acquisition structure. The screen-level distributions combine target composition and
experimental context; they do not estimate a pure environmental effect or a variance
decomposition. Field/gene cell-level screen and assay views reweight the frozen
reporter-level scores by observed membership because common per-cell predictions are
unavailable for every shared method. Their 99 assay marks represent 52 distinct
reporter-level values. Whole-screen cell scores are evaluated directly in destination
cells. Cross-method bands describe method spread, not sampling uncertainty.

KO assay scores are independent of reporter-level partition aggregation. For field
and gene tests the builder concatenates the five saved out-of-fold
gene×screen profile arrays within each physical assay and computed endpoint-macro
Pearson once; strict assay scores are direct destination evaluations. Physical-screen
scores are the per-method median over assay members. Cell field/gene assay scores
use reporter means, and screen cell scores use the median over assay members.
Strict-screen cell assay values are direct destination evaluations. Rank curves
are sorted by the resulting
consensus; paired assay rows retain the existing independent KO-based row order.

The composite places the accepted panel a in a full-width row above the existing
three-row quantitative body, with no non-uniform scaling. Its actual dimensions are
read from the PDF by the print-scale audit. The audit also reads the current
manuscript placement and resolves nested PDF text transformations.
Quantitative tick, key and annotation text is at least 6.5 pt at authored size;
axis labels are 6.8 pt and explanatory panel titles are absent. The panel-c test keys occupy a
dedicated strip above the curves, and the adjacent IQR keys have separate rows.
Both panel-b x-axes extend to −0.025 so the existing −0.005407 KO-response
reporter score is visible; tick labels remain 0, 0.5 and 1. No value is clipped
to zero, removed or changed for presentation.
Both screen-IQR axes extend to 0.28, covering the whole-screen KO IQR
of 0.255567. Tick labels are 0 and 0.2;
the renderer verifies that no IQR bar exceeds the visible range.
The renderer rejects intersecting visible text-artist boxes or text outside
each standalone crop. This mechanical screen is followed by visual inspection
of keys, curves, bars, points and the final manuscript-size figure; it does not
reinterpret deliberate data overplotting as a rendering defect.
Standalone panels and the composite have PDF/SVG and 600-dpi PNG companions.
Panel a remains editable in PowerPoint and as live PDF text; its derived SVG may
use vector glyph outlines. Panels b-d retain native editable SVG text. Standalone
panels contain no letters; only the composite adds a–d in 9-pt bold type.
