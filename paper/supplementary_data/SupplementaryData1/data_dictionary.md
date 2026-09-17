# Supplementary Data 1 column dictionary

`reporter_slug` is the stable target key; `short_name` is its source label.
`biological_category` is the frozen source-study annotation. `display_order`
retains the Figure 6 order, without clustering or reordering by method sensitivity.
`method_id` identifies one individual predictor or a separately defined ensemble.

| Evidence column | Definition |
|---|---|
| `run_state` | Primary `checkpoint_average` or sensitivity `single`; four specialists share their frozen output between state labels |
| `recoverability_r` | Arithmetic mean of five fold-specific, equal-endpoint macro-Pearson correlations across eligible gene-screen responses |
| `magnitude_spearman` | Spearman correlation between observed and predicted Euclidean response magnitudes across 1,000 held-out genes |
| `variance_ratio` | Population variance of predicted gene magnitudes divided by that of observed magnitudes |
| `top5pct_recall` | Number of observed top-50 genes also in predicted top-50, divided by 50 |
| `reliability` | Model-independent median within-screen observed-target split-half magnitude correlation |
| `n_genes` | Number of evaluated genes; 1,000 in every evidence row |
| `observed_boundary_ties`, `predicted_boundary_ties` | Number of magnitudes equal to the fiftieth-largest magnitude; stable numeric-gene-code ordering breaks any ties |
| `tier` | Frozen hierarchical decision: quantitative_proxy, ranking_proxy, measurement_required, not_identifiable or unresolved |
| `proxy_supported` | True for Q or R only |
| `historical_reference_tier` | Pre-revision mixed-state ledger tier; present only in the separate definition-sensitivity table |
| `failed_quantitative_gates`, `failed_ranking_gates` | Semicolon-delimited criterion flags that fail; these do not override hierarchical precedence |

Gate booleans are conditions, not universally higher-is-better quality labels:

| Flag suffix | Condition |
|---|---|
| `reliability` | reliability ≥0.30 |
| `recovery` | recoverability ≥0.70 |
| `rank` | magnitude Spearman ≥0.70 |
| `variance_lower` | variance ratio ≥0.50 |
| `variance_upper` | variance ratio ≤1.50 |
| `quantitative_hit` | top-5% recall ≥0.60 |
| `ranking_hit` | top-5% recall ≥0.50 |
| `measurement_recovery` | recoverability <0.60; True means the measurement-required recovery condition holds, subject to earlier rules |

Precedence: not identifiable for missing/low reliability; unresolved for missing
required evidence; Q if every quantitative condition holds; otherwise R if ranking
and ranking-hit conditions hold; otherwise M if recovery <0.60; otherwise U.
Ranking does not itself require recovery ≥0.70. All current inputs are complete.

`changes_vs_reference` fields are `candidate − fixed manuscript reference` for
every `delta_*`. `reference_state` is always checkpoint_average and
`reference_method_id` is always coherent_median_ensemble. `transition` points
from that reference tier to the candidate tier; `changed_gate_flags` lists every
boolean that changed. `proxy_status_changed` compares the sets {Q,R} and {M,N,U}.
`reliability_pass` identifies the primary 43-reporter denominator. `_43` and `_52`
agreement columns give counts, not percentages. N is never counted as a reliable
target and is not merged with M or U for exact-tier transitions.

`stability_category` compares only the ten individual methods in the specified
state: `all_proxy` (10/10 Q or R), `all_no_proxy` (0/10),
`proxy_boundary_crossing` (1–9/10), or `fixed_reliability_N` (reliability failure).
`exact_tier_stable` requires all ten exact tier names to agree. These are
descriptive finite-roster counts, not uncertainty estimates.

The provenance tables preserve fold IDs, gene-screen counts, endpoint counts,
target-array maximum discrepancies and reordering flags. Endpoint scales retain
zero-based endpoint order. `fold_to_common_factor = fold_target_scale /
common_control_scale`; it multiplies both observed and predicted fold-scaled
response coordinates. No new normalization is fitted on knockout outcomes.
