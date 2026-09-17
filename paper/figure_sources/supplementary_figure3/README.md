# Supplementary Figure 3 — Predictor-conditional measurement-sufficiency tiers

The only canonical published figure is
`../../figures/SupplementaryFigure3.pdf` (183 × 162 mm), corresponding to
`paper/figures/SupplementaryFigure3.pdf` from the repository root. This directory
is its source package, not an additional figure-publication location. It archives
drawing and verification code, the caption and methods, the plotting specification,
plotted keys and source-dependency hashes.

Running the builder locally creates clean standalone panels in
`a_predictor_tiers/` and `b_lamp1_gates/`, and an intermediate PDF/SVG/600-dpi PNG
composition in `composite/`. These generated files, together with `qa/` and
`build/` outputs, are rebuildable local products and are not archived source
requirements. Only the composition contains 9-pt panel letters; all other text is
at least 6 pt. Rendering uses the shared `../publication_style.py` without
changing data or thresholds.

The sole numerical authority is
`../../supplementary_data/SupplementaryData1/evidence.csv`.
`source_data/plotted_point_keys.csv` lists every displayed row/metric key, and
`source_data/input_manifest_sha256.csv` hashes the required inputs. Those locators
are relative to this figure-package root. No duplicate editable evidence table is
maintained here. Reporter and method order are deposited in Supplementary Data 1.

From the repository root:

```bash
python -B paper/figure_sources/supplementary_figure3/code/build_supplementary_figure3.py
python -B paper/figure_sources/supplementary_figure3/code/verify_figure.py
```

The builder publishes its rendered PDF to the single canonical path above.
It needs NumPy, pandas and Matplotlib. Vector-file checks additionally use
PyMuPDF. See `figure_plot_spec.yaml`, `figure_legend.md`, `figure_methods.md`
for the archived scientific and rendering contracts. Verification creates
`qa/figure_qa_report.json` locally; visual review remains necessary after a rebuild.

`manifest_sha256.csv` covers only archived source files and never requires the
ignored rendering or QA products. `source_data/input_manifest_sha256.csv` records
the canonical numerical and shared-style dependencies. A separate generated
`build/artifact_manifest_sha256.csv` inventories the local render/QA products and
the canonical published PDF; it is not part of the source archive.

The complete package includes two state-specific matrices, all continuous
evidence, method counts, gate crossings, reporter stability and construction
sensitivity in Supplementary Data 1. This supplementary figure displays only
the checkpoint-average primary state. Methods are sensitivity axes, not
independent biological replicates. LAMP1 is an illustrative result-selected case,
not a prespecified test. These response-tier assignments do not establish
functional-analysis equivalence or transport beyond this evaluation design.
