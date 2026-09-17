# Supplementary Figure 4

Utility prediction, screen context and repeat measurements refine use-specific
assessment.

| Panel | Evidence | Frozen source |
| --- | --- | --- |
| a | Nested leave-one-reporter-out utility calibration for 52 OPS reporters | `../final/figure6/b_nested_loo_audit/` |
| b | Recovery and rank across 81 destination-screen evaluations in 34 reporters | `../final/figure6/e_reporter_context_stability/` |
| c | Cross-source recall relative to measured-repeat prioritization in 94 OASIS compounds | `../../supplementary_data/SupplementaryData3/` |
| d | Recall at 5%, 10% and 20% selection budgets in four OASIS contexts | `../../supplementary_data/SupplementaryData3/` |

The 217-compound metabolic-loss scatter and six-predictor hit comparison are
shown in Figure 6d,e. This supplement provides the OPS utility and context
analyses together with external repeat-reference and budget evidence.

From the repository root:

```bash
python -B paper/figure_sources/supplementary_figure4/code/build_supplementary_figure4.py
```

The script imports shared rendering functions from
`../final/figure6/code/figure6_layout.py`, reads frozen outcome tables and produces the editable composite
`SupplementaryFigure4.pdf` / `.svg`, a 600 dpi `.png` preview and four clean
standalone panels in `panels/`. The adopted PDF is copied to
`paper/figures/SupplementaryFigure4.pdf`. Only the compositor adds a–d.

`figure_source_data.csv` contains the plotted OPS and external-use evidence,
tagged by panel. `source_manifest.csv` identifies the frozen inputs.
`correspondence_reference.csv` retains the primary fixed-model
correspondence-null summary for the accompanying text; it is distinct from
the fixed-size random expectation shown in Figure 6e. `figure_plot_spec.yaml`,
`figure_legend.md` and `figure_methods.md` record units, transformations, pairing,
uncertainty, highlights and claim boundaries. `qa/` contains collision screening;
the final visual and scientific audit is `figure_qa_report.json`.

The composite is 183 × 130 mm and follows Figure 6's typography and colour
conventions. The signed metabolic-loss selection task complements the
response-fidelity analysis in Figure 6f; the two tasks have distinct estimands.
