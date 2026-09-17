# Supplementary Data 3: brightfield-supported metabolic-loss prioritization

Frozen predictions, compound rankings and retained-hit results for a defined OASIS
use: prioritizing compounds with the largest decrease in a paired metabolic-activity
readout in primary hepatocytes. Figure 6d,e and Supplementary Figure 4c,d are
drawn from these tables.

## Original data and attribution

Source profiles and metadata: Jessica Ewald, **Axiom OASIS**,
[Zenodo record 17067683](https://doi.org/10.5281/zenodo.17067683), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The study is described in
[Ewald et al., Cell Systems (2026)](https://doi.org/10.1016/j.cels.2026.101566).
Brightfield columns follow the source authors'
[pinned DINO formatter](https://github.com/broadinstitute/2024_09_09_Axiom_OASIS/blob/ca4872de0f1f0a4f203bee41271a6512b314a076/0_prepare_data/2B_format_dino.py).

The deposited source-derived measurements, well metadata and their analysis tables
retain the source attribution and CC BY 4.0 terms. Changes comprise extraction of
the brightfield channel, fixed development/test partitioning, plate-relative
response calculation, fitted predictions and compound-level prioritization scores;
the exact source and result hashes are recorded in `provenance.json`. Raw images
and full feature matrices are not redistributed here. Study-authored scoring and
plotting code is covered by the repository code licence; it does not relicense
the source-derived data.

## Contents

- `data/heldout_predictions.parquet`: 19,800 rows (3,300 held-out wells × six fixed
  predictors), with the declared plate-relative measured and predicted responses.
- `data/context_compound_scores.csv`: 4,122 compound–model–context rows; signed
  losses for 217 held-out compounds, the 94-compound complete-source subset, both
  ordered source directions and actual source-pair strata.
- `data/utility_scores.csv`: all 144 model–context–budget results, including
  compound-bootstrap intervals and measured-repeat comparisons where available.
- `data/selection_ledger.csv`: all 12,366 compound–model–context–budget entries,
  including ranks, selected sets, retained hits and missed hits.
- `data/correspondence_null_summary.csv` and
  `data/correspondence_null_distribution.csv.gz`: six summaries and 12,000
  fixed-model correspondence-null results, restricted to all 217 compounds.
- `data/compound_partition.csv`, `data/training_membership.csv` and
  `data/test_membership.csv`: compound partition and fitted/evaluated well membership.
- `protocol.md`: readable scientific contract and relationship to the original
  response-fidelity analysis.
- `provenance.json`: original source hashes, freeze chronology, model specifications
  and prior independent numerical-audit record. Private filesystem prefixes have
  been removed; numerical files and `code/frozen_scoring.py` retain original bytes.
- `code/replay_scores.py`: portable verification from the deposited predictions;
  no fitting or raw-image download. `code/frozen_scoring.py` contains the original
  scoring functions; use the replay entry point rather than its historical main.
- `manifest_sha256.csv`: hashes for the public bundle, excluding the manifest itself.

## Reproduce the scoring

With Python 3.10+, NumPy, pandas, SciPy and PyArrow installed, from the repository root:

```bash
python -B paper/supplementary_data/SupplementaryData3/code/replay_scores.py
```

The command checks source hashes and membership, reconstructs compound scores,
replays all frozen metrics, bootstrap intervals, selected sets and correspondence
nulls, and compares them with the deposited tables. It does not modify those tables
or refit models. CSV inputs use round-trip float parsing to preserve deterministic
ties in near-identical diagnostic predictions.

To rebuild the figures without recalculating statistical results:

```bash
python -B paper/figure_sources/final/figure6/code/build_figure6.py
python -B paper/figure_sources/supplementary_figure4/code/build_supplementary_figure4.py
```

## Interpretation

At the fixed top-5% budget, the primary brightfield Ridge predictor retained 10 of
11 measured hits among 217 held-out compounds. In the 94-compound complete-source
subset, cross-source prioritization retained 4 of 5 in each direction; measured
repeats also retained 4 of 5. The paired uncertainty intervals remain broad, so
equal point estimates are not evidence of measurement equivalence. All secondary
models, budgets and source strata remain visible in the full tables.

The target is signed metabolic-activity loss averaged across each compound's eight
assayed concentrations, not absolute response magnitude, a common-range dose AUC,
cell death or general toxicity. The fixed brightfield input route, well inclusion
and task differ from the original OASIS response-fidelity analysis in Figure 6f;
the new hit count does not replace that analysis or reassign its measurement tier.

Publicly released brightfield-channel DINO features were used without the old
CellProfiler fluorescence-derived segmentation features. The source images came
from an existing fixed/stained experiment; these results establish the retained
prioritization use of that brightfield representation, not a newly validated
unstained live-cell acquisition workflow. Paired target measurements were required
for development and evaluation; the OPS predictor weights were not transferred.
