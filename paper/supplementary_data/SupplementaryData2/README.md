# Supplementary Data 2 — sensitivity of existing external applications

This compact bundle applies the manuscript's existing 36 OPS threshold scenarios to four main OASIS/PERISCOPE summaries and eight OASIS batch summaries. It preserves full-precision published source values and the exact tier hierarchy. It does not retrain predictors, recalibrate thresholds, or incorporate the newer OASIS BF-DINO metabolic-prioritization study.

The analysis is descriptive and retrospective: external metrics and the historical rank-only sweep were already known. See `protocol.md` for estimands, source selection, threshold definitions and interpretation limits.

## Files

- `inputs/context_metrics.csv`: the only canonical metric input, 12 contexts with source keys and published baseline labels.
- `inputs/threshold_scenarios.csv`: baseline plus the exact 36 inherited scenarios.
- `inputs/source_manifest.csv`: repository-relative original sources, bytes and hashes.
- `analysis_freeze.json`: pre-evaluation input/protocol/executable/scenario hashes and recorded freeze time.
- `derived/assignments.csv`: all 444 assignments and aggregate evidence predicates.
- `derived/criterion_evidence.csv`: all 3,552 atomic criterion conditions and changes.
- `derived/context_summary.csv`: single32, joint4 and all36 stability and possible tiers for every context.
- `derived/scenario_summary.csv`: main4 and batch8 counts kept separate.
- `derived/historical_rank_only.csv`: six historical OASIS rank-only rechecks, excluded from the 36-scenario counts.
- `table_rows.tex`: the numerical row fragment included directly by Supplementary Table 7.
- `independent_audit.json`: independent reconstruction checks, context summaries and replay details.
- `manifest_sha256.csv`: hashes of the source bundle and generated results, excluding the manifest itself.

`code/build_sensitivity.py` requires Python 3.10+ and the standard library only. It verifies the input freeze before evaluation and refuses to replace a file with different content. From the bundle directory:

```sh
python -B code/build_sensitivity.py evaluate
python -B code/build_sensitivity.py manifest
```

The one-time provenance import was performed with `freeze --repo-root <repository-root>` before evaluation. Readers can regenerate all derived tables from the included canonical inputs without the original raw predictions or any local absolute path. Re-running with the same inputs is idempotent; output changes require an explicit, documented new analysis rather than silently overwriting the frozen bundle.

Q/R/M/N/U mean quantitative proxy, ranking proxy, measurement required, not identifiable and unresolved. These are the inherited response-based rules, not a universal clinical or biological decision ordering. An M-eligibility predicate being true is not “good performance.” An unchanged N tier can coexist with changed fidelity-criterion predicates. Counts across threshold scenarios are sensitivity summaries, not confidence probabilities or independent experimental replications.

The source manifest includes the numerical evidence and machine-readable verification record. The manuscript and release registry are managed separately.
