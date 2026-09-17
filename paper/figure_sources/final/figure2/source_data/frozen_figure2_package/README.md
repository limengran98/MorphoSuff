# Figure 2 — full-label benchmark, frozen source views B–G

Current manuscript mapping: frozen views B/C form panel b, D/E form panel c,
and F/G form panel d. Every grouped panel shows KO-response on the left and
cell phenotype on the right. The original numerical filenames and table IDs
below are retained for provenance; they are not current panel letters.

This numerical source package contains KO-response and paired-cell Pearson views for reporter, physical screen and reporter×screen assay aggregation. Figure 2a is a schematic: its editable source is `a_workflow/source/Figure2a.pptx` at the Figure 2 package root. The PowerPoint and native PDF export are SHA-256 bound by the adjacent `export_manifest.json`. Microscopy provenance and schematic-asset records accompany that source.

Reporter KO/cell scores use arithmetic means over predefined folds or destination
directions. Field/gene cell assay/screen reweightings use those reporter means.
KO assay/screen tables have separate aggregation definitions, and strict cell
assay values are evaluated directly. The portable input is `../frozen_partition_scores.csv.gz` and
the reproducible numerical builder is `../../code/rebuild_benchmark_sources.py`.

Each of the ten models has an individual saturated colour. The four scientific model families are visible as pale background bands and in the top key; they do not replace method identity. Source data, code and vector/raster outputs are stored next to every panel.

The sealed cell-layer provenance is retained in `source_contract.json`: strict whole-screen values are direct destination-cell evaluation, while field/gene screen and assay views reweight frozen reporter-cell scores across physical membership because a common raw per-cell archive does not exist for every joint method.
