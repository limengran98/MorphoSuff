# Figure 6 source package

Measurement decisions and use-specific external validation.

| Panel | Evidence | Frozen source component |
| --- | --- | --- |
| a | Downstream retention after response replacement for 52 OPS reporters | `a_counterfactual_utility_atlas/` |
| b | Ensemble recoverability versus retained scientific utility | `d_prediction_utility/` |
| c | Reporter decisions and sensitivity to 36 threshold scenarios | `c_decision_ledger/` |
| d | Measured and predicted metabolic loss for 217 held-out OASIS compounds | `../../../supplementary_data/SupplementaryData3/` |
| e | Recovery of the eleven measured hits by six fixed predictors | `../../../supplementary_data/SupplementaryData3/` |
| f | Measurement criteria across OPS, OASIS and PERISCOPE | `f_cross_context_decisions/` |

The nested leave-one-reporter-out utility calibration in `b_nested_loo_audit/`
and destination-screen evidence in `e_reporter_context_stability/` supply
Supplementary Figure 4a,b. These reusable component directories retain their
source and calculation interfaces; displayed panel letters are assigned by the
current compositors. Supplementary Figure 4c,d contain the OASIS repeat-reference
and selection-budget analyses. Label-efficiency and representation analyses are
in `../../supplementary_figure2/`.

## Build

From this directory:

```bash
python -B code/build_figure6.py
```

On Windows, `./build.ps1 -Python <python.exe>` provides the package build and
verification entry point; `../build_all.ps1` invokes it with the other main
figures. The composite is exported to `published/` as editable PDF/SVG and
600-dpi raster artwork. The adopted PDF is `paper/figures/Figure6.pdf`.
The canonical clean panels are `panels/Figure6a` through `panels/Figure6f`.

`code/figure6_layout.py` supplies the shared drawing functions for this figure
and Supplementary Figure 4. Builds read public frozen inputs and do not depend
on a design-preview directory. The three compact CSVs in `source_data/`
reconstruct the OPS utility Ridge, reporter-bootstrap intervals, 36 threshold
scenarios and decision ledger. The ten image-to-reporter models are not
retrained. `code/freeze_coherent_inputs.py` is a provenance import utility, not a
routine build dependency.
To rederive the reporter-level utility and decision summaries before rendering,
run `python -B code/rebuild_coherent_statistics.py`; ordinary artwork builds
use the already deposited summaries.

## Scientific contract

Panels a–c and the OPS row of f describe one replacement estimator: the
elementwise median of ten frozen checkpoint-average predictions at the
gene×screen level, before equal-screen gene averaging. Four specialist methods
have fixed predictions and six neural comparators use checkpoint-average
evaluation states. `ensemble_recoverability_r` is the arithmetic mean of five
fold-specific endpoint-macro Pearson correlations for this same ensemble;
benchmark medians of individual method scores are a distinct estimand.

Panels a,c use whole-cohort utility percentiles. Panel b and Supplementary
Figure 4a use leave-one-reporter-out training-reference utility. The horizontal
median in b is descriptive, not a decision gate. Run
`python -B b_nested_loo_audit/build_panel_b.py --verify-statistics` for the
reporter-level utility checks, including tie-preserving rank statistics.

Panels d,e evaluate a specified use: selecting strong, signed metabolic-loss
responses in held-out compounds. The 217-compound scatter, top-eleven
memberships and six predictors are fixed. The histograms summarize the same
compounds; the selection boundaries mark the existing budget, not sufficiency
criteria. The 6×11 matrix and recall intervals describe the same measured hits.
Intervals are the deposited 95% compound-bootstrap intervals from 2,000
resamples. Full predictions and scoring replay are in Supplementary Data 3.
The exact plotted external-use rows, histogram bins, selection boundaries and
compound-hit memberships are deposited in `source_data/external_use/`.
This use differs from the broad response-fidelity criteria in f; those values
and assignments remain unchanged.

## Presentation

The 183 × 210 mm canvas uses the shared `../../publication_style.py`:
DejaVu Sans, 9 pt bold composite-only letters and a 6 pt ordinary-text floor.
Standalone panels have no letters or explanatory titles. Panels d/e share a
65 mm nominal row; their captions carry the context and encoding explanations.
Panel d groups the sample count, retained-hit summary and selection key beside
the scatter. Panel e uses a compact label column: each input name appears once
above its Ridge/HGB pair, with right-aligned model labels next to the hit matrix.
The matrix and recall axis are 34.4 and 34.6 mm wide, respectively, with shared
row alignment. These physical dimensions are specified in
`figure_plot_spec.yaml` and used for both the composite and clean panels.
Q/R/M/N/U in c mean quantitative, ranking, measurement-required,
not-identifiable and unresolved. The transition inset uses vector cells and
aligned outlines on one grid. Inspect the complete figure at manuscript size
after rendering, as described in `paper/BUILD.md`.
