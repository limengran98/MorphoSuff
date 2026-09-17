# Supplementary Data 1 — Predictor-conditional measurement-sufficiency evidence

This is the canonical complete numerical supplement for Supplementary Figure 3
and the predictor-sensitivity analysis in Figure 6. It is a compact evidence
release, not a second prediction-training dataset. No models were retrained.

## Authoritative tables and reference

`evidence.csv` is the one authoritative continuous-evidence table: 52 reporters ×
11 predictors (ten individual methods and one coherent median ensemble) × two
evaluation states = 1,144 rows. Keys are `run_state`, `reporter_slug`, `method_id`.
All other primary tables are derived from these rows. The primary state is
`checkpoint_average`; `single` is a checkpoint-state sensitivity. The four
specialists Ridge, GBDT, CatBoost and MLP use the same frozen files in both states;
the two labels do not imply independent fits for those methods.

**Every reference comparison in this package uses the manuscript's
`checkpoint_average` / `coherent_median_ensemble` reference, including comparisons
for the `single` state.** `reference_state` and `reference_method_id` are recorded
explicitly in derived comparison tables. A state-specific reference is never
substituted silently. The coherent ensemble's own single-state row remains
available in `evidence.csv` for a separate state comparison.

`definition_sensitivity.csv` is a separate, non-primary table for two retained
construction checks. `state_matched_hybrid` uses the median of individual-method
recovery scores with profile-median fidelity; `hybrid_median_before_screen_mean`
reconstructs the historical mixed-state ledger. Neither is a deployed unified
predictor or the current reference. Their definitions and summary comparison are
in `analysis_spec.json` and `derived/ensemble_definition_comparison.csv`.
Abandoned preflight variants are not included.

## Contents

| File | Meaning |
|---|---|
| `evidence.csv` | Continuous metrics, gate flags and tiers for all primary predictors and both states |
| `reporter_order.csv`, `method_roster.csv` | Fixed Figure 6 reporter order and fixed ten-method roster |
| `derived/*_tier_matrix.csv` | Human-readable matrices derived from the canonical evidence |
| `derived/changes_vs_reference.csv` | Continuous differences and every changed gate flag versus the fixed manuscript reference |
| `derived/transition_counts.csv` | Exact Q/R/M/N/U transitions, with the reliability denominator identified |
| `derived/method_tier_counts.csv` | Tier counts and exact-tier/proxy agreement for 43 and 52 reporters separately |
| `derived/reporter_stability.csv` | Individual-method tier diversity and proxy-boundary stability |
| `derived/checkpoint_state_comparison.csv` | Matched same-method state differences |
| `derived/summary.json` | Compact descriptive counts checked against the complete tables |
| `definition_sensitivity.csv` | Separate score-median and historical-reference construction checks |
| `scale_sensitivity.csv` | 104 rows: 52 coherent-ensemble reporters × common-control/fold-standardized scales; checkpoint_average is fixed |
| `provenance/per_method_fold_recoverability.csv` | Five original fold-level recovery values per reporter and individual method, for both states |
| `provenance/endpoint_scales.csv` | Endpoint identities, common control scales and each training-fold target scale |
| `provenance/alignment_audit.csv` | Complete reporter/method/fold key and target-array alignment checks |
| `provenance/frozen_input_manifest.csv.gz` | Deduplicated relative archive locators, state membership, sizes and available input SHA-256 values |
| `provenance/control_scale_audit.json` | Unique-control and exhaustive OOF checks with selected-control digests |
| `analysis_spec.json`, `data_dictionary.md` | Frozen estimands, rules, comparison conventions and column definitions |
| `code/` | Deterministic table rebuilding, independent QA and optional raw-profile replay |

The figure package stores plotted row/metric keys rather than an independently
editable second copy of these evidence values.

## Rebuild without the original prediction archive

From the repository root:

```bash
python -B paper/supplementary_data/SupplementaryData1/code/build_tables.py
python -B paper/supplementary_data/SupplementaryData1/code/verify_data.py
python -B paper/figure_sources/supplementary_figure3/code/build_supplementary_figure3.py
```

The first two commands need NumPy and pandas. Figure rendering additionally needs
Matplotlib; vector QA uses PyMuPDF. `manifest_sha256.csv` freezes the deposited
files. Generated derived tables can be recreated exactly from the canonical
evidence; figure PDF timestamps may vary between runs.

## Optional replay from original frozen prediction profiles

The compact package does **not** contain the complete gene-screen prediction
arrays, saved raw-target arrays or all original checkpoints. No claim is made
that a public model-output repository already contains every required input.
An investigator who separately has the original `results/` archive can run:

```bash
python -B paper/supplementary_data/SupplementaryData1/code/replay_frozen_predictions.py \
  --mm-root PATH_TO_ARCHIVE --state checkpoint_average --output NEW_OUTPUT_DIRECTORY
```

The same command with `--state single` reproduces the sensitivity. An optional
`--reporters lysosome_lamp1` selects a provenance smoke test, not the primary
analysis. The script resolves all inputs beneath the explicitly supplied archive
root and never trains or edits a predictor. It verifies endpoint order, complete
gene-screen keys, matched target transforms and unique OOF-control coverage;
it converts saved fold-scaled responses to the common control scale before
combining folds. Full scale reconstruction requires the saved raw control targets,
not microscopy images. A complete imported analysis can be frozen into a new
copy of this package with `build_tables.py --analysis-root ANALYSIS_DIRECTORY`.

Input locators in provenance are relative to the original archive root. Blank
`sha256` means that a large raw-target array was not fully rehashed during the
source audit, not that its content was verified by a missing digest. The selected
control values, IDs and screen IDs have separate content digests. Locator hashes
refer to the frozen analysis inputs; later manuscript-source updates need not have
the same hashes.

`code/check_ensemble_scale.py --mm-root PATH_TO_ARCHIVE` separately reproduces
`scale_sensitivity.csv` from frozen profiles and deposited endpoint-scale factors.
Its common-control rows must reproduce the adopted ensemble evidence before the
output is saved. Changing only to the old fold-standardized response coordinates
gives Q/R/M/N/U counts of 30/6/3/9/4, versus 30/7/3/9/3 on the common scale; only
ER NCLN changes from R to U. Recovery and reliability remain fixed in this check.

## Interpretation boundary

Reporter is the biological reporting unit. Methods and checkpoint states are
sensitivity axes, not independent biological replicates or probability samples.
Of 43 reliability-passing reporters, the primary-state ten-method comparison has
20 always supporting a proxy, 6 never supporting a proxy and 17 crossing that
boundary. The single-state sensitivity gives 22, 6 and 15, respectively. The other
9 reporters are not identifiable because observed-target reliability is below
0.30; they are excluded from the stability denominator. Their invariance is
structural, not evidence of model agreement. M and U are distinct outcomes:
unresolved is not a positive requirement for direct measurement.

Changed gate flags describe criterion crossings; they are not causal attributions
to a single metric, and flags below a higher-precedence reliability gate can be
irrelevant to the final tier. Q and R concern the stated response-level tasks;
they do not establish unbiased absolute calibration or functional-term equivalence.
Thresholds, reporter order and model roster were held fixed in this sensitivity
analysis. LAMP1 was chosen after examining these results to illustrate Q/R/M
differences and was not a prespecified confirmatory case.
