#!/usr/bin/env python3
"""Run the measurement-sufficiency chain on guide-level cpg0021-periscope tables.

Two things make this different from the OASIS analysis, and both are consequences of
scale rather than of taste.

**Reliability comes from guide halves, which is the better estimate.** PERISCOPE
carries a median of four independent sgRNAs per gene, and 98.8% of genes have at
least two. Splitting a gene's guides into two disjoint halves and correlating the half
means over genes is exactly what ``analysis.stability.guide_half_consistency``
consumes, and it measures what the tier rule gates on: how reproducible the targeted
measurement is when the perturbation is held fixed and only the reagent changes. The
OASIS preparation had to approximate this by splitting wells, because that dataset
declares ``guides: false``.

**Per-observation predictions are never materialised in long form.** A reporter with
823 endpoints over nine plates of guide profiles would be 549 million prediction rows;
the same information reduced to gene responses is 16.8 million, and the gene by
endpoint matrix behind it is 134 MB. Predictions are therefore aggregated to
gene-level responses inside the fold loop, screen by screen, and only that reduction
is handed to the shared analysis primitives.

**What this level cannot say.** Fitting on profiles already averaged within a
(gene, sgRNA) removes noise from the input and the target together, so the
recoverability it reports is a different and easier quantity than the published one,
which predicts per cell and aggregates the predictions. It must be labelled as its
own number, never quoted against the paper's. The manifest declares
``aggregation_level: guide`` and ``exact_pairing: false`` so that the same-cell
analyses refuse to run here.

Example:

    python studies/external/periscope/analyse_guide.py \\
      --prepared ../external/periscope_guide \\
      --output-dir ../external/periscope_guide_analysis --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _common import reject_repo_paths  # noqa: E402

from _reliability import (  # noqa: E402
    reliability_by_block,
    split_half_blocks,
    summarise_reliability,
)

from _ridge import (  # noqa: E402
    fit_folds,
    impute_from_controls,
    permute_rows_within,
    recoverability_by_fold,
)

from measurement_sufficiency.adapters import LocalManifestAdapter  # noqa: E402
from measurement_sufficiency.analysis.response_fidelity import (  # noqa: E402
    magnitude_spearman,
    response_magnitudes,
    top_fraction_recall,
    variance_ratio,
)
from measurement_sufficiency.analysis.tiers import OPS_TIER_RULES, tier_table  # noqa: E402

REPORTER_ID = "tomm20_mito"
NON_TARGETING = "nontargeting"
#: Read from the frozen rule rather than restated, so a report about the floor cannot
#: drift from the floor the verdict was decided by.
RELIABILITY_FLOOR = OPS_TIER_RULES.reliability_min


def load(prepared: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    manifest = json.loads((prepared / "canonical_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("aggregation_level") != "guide":
        raise SystemExit(
            f"this script analyses guide-level tables; the manifest declares "
            f"{manifest.get('aggregation_level')!r}. Use analyse.py for single-cell."
        )
    features = list(manifest["input_schema"]["feature_names"])
    endpoints = list(manifest["target_endpoint_columns"])
    observations = pd.read_parquet(prepared / manifest["tables"]["observations"])
    inputs = pd.read_parquet(prepared / manifest["tables"]["inputs"])
    targets = pd.read_parquet(prepared / manifest["tables"]["targets"])
    frame = observations.merge(inputs, on="input_cell_id", how="inner").merge(
        targets, on="observation_id", how="inner"
    )
    return frame, features, endpoints


def as_matrices(
    frame: pd.DataFrame, features: list[str], endpoints: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Split the merged table into identifiers and two float32 blocks.

    The fitting itself is shared with the single-cell level, so the only thing this
    level has to do is present its rows in the same shape. A guide-level row is a
    (gene, sgRNA, plate) profile rather than a cell; nothing downstream depends on
    which, and that is the point.
    """
    identifiers = frame[["screen_id", "perturbation_id", "is_control"]].reset_index(drop=True)
    x = frame[features].to_numpy(dtype=np.float32)
    y = frame[endpoints].to_numpy(dtype=np.float32)
    return identifiers, x, y


def reliability_from_guides(
    frame: pd.DataFrame,
    endpoints: list[str],
    *,
    seed: int,
    within_screen: bool = True,
    reporter_id: str = REPORTER_ID,
) -> pd.DataFrame:
    """Split a gene's sgRNAs in two and correlate the two halves' responses.

    The estimator lives in ``_reliability`` because the single-cell level, the
    stain-dropout arm and the gene-set audit all need the same quantity, and a
    quantity computed in four places is four quantities. See that module for why the
    pooling is standardised and for the measurement showing that the native pooling it
    replaced moves when only the number of genes changes.

    Args:
        within_screen: when true, guides are split inside each screen and the halves
            are made control-relative to that screen. When false the split pools
            screens, which is reported alongside so the gap between the two is visible
            rather than assumed.
    """
    blocks = split_half_blocks(
        frame, endpoints, seed=seed, group_column="screen_id" if within_screen else None
    )
    table = reliability_by_block(blocks)
    table.insert(0, "reporter_id", reporter_id)
    table = table.rename(columns={"block": "screen_id"})
    for _, row in table.iterrows():
        print(f"  reliability {row['screen_id']}: {int(row['n_genes']):,} genes, "
              f"{int(row['n_endpoints'])} endpoints, r = {row['reliability']:.4f} "
              f"(native pooling would say {row['reliability_pooled_native_all_endpoints']:.4f})",
              flush=True)
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--alpha",
        default="auto",
        help="ridge penalty on standardised features. 'auto' selects it per fold on "
             "an inner validation fold that the held-out fold never enters. A number "
             "fixes it, which reproduces the first run: scikit-learn warned "
             "Ill-conditioned matrix (rcond=2.2e-09) on every fold and the fixed 1.0 "
             "was a ratio of 1.8e-06 against the row count, which regularises nothing",
    )
    parser.add_argument(
        "--standardize-endpoints", choices=("none", "observed_sd"), default="observed_sd",
        help="the 823 endpoints are in native units and one of them holds most of the "
             "squared spread, so the default puts them on a common footing before the norm",
    )
    parser.add_argument(
        "--permute-inputs", choices=("none", "within_screen"), default="none",
        help="negative control. 'within_screen' shuffles the input block inside each "
             "screen, leaving the targets, the perturbation labels and the control "
             "flags where they are, so the only thing destroyed is the association the "
             "model is asked to learn. The reliability path never touches the input "
             "block, so its value must come back unchanged; recoverability must "
             "collapse. A run with this flag is not a result about this dataset and "
             "its output directory is marked as a control in provenance.json",
    )
    parser.add_argument(
        "--permutation-seed", type=int, default=None,
        help="seed for --permute-inputs; defaults to --seed",
    )
    parser.add_argument(
        "--alpha-ratio", type=float, default=None,
        help="fix the penalty as a ratio to the training row count, the unit the "
             "nested selection searches in. Required for --permute-inputs: with the "
             "association destroyed the best penalty is the largest on the grid, which "
             "the selection refuses to return, and a control that re-selects its own "
             "penalty differs from its reference run in two things rather than one",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = reject_repo_paths(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = LocalManifestAdapter(args.prepared / "canonical_manifest.json")
    frame, features, endpoints = load(args.prepared)
    print(json.dumps({
        "rows": len(frame), "screens": int(frame.screen_id.nunique()),
        "genes": int(frame.perturbation_id.nunique()), "guides": int(frame.guide_id.nunique()),
        "controls": int(frame.is_control.sum()),
        "features": len(features), "endpoints": len(endpoints),
        "capabilities": dataset.capabilities.__dict__,
    }, indent=2))

    identifiers, x, y = as_matrices(frame, features, endpoints)
    fill, n_columns_filled, n_cells_filled = impute_from_controls(
        x, identifiers["is_control"].to_numpy()
    )
    print(f"  imputed {n_cells_filled:,} missing inputs across {n_columns_filled} "
          f"columns from the control median", flush=True)
    # After imputation, so that the fill values are the real control medians rather
    # than the medians of a shuffled block.
    permutation = None
    if args.permute_inputs == "within_screen":
        seed = args.seed if args.permutation_seed is None else args.permutation_seed
        permutation = permute_rows_within(
            x, identifiers["screen_id"].to_numpy(), seed=seed
        ) | {"mode": args.permute_inputs, "seed": seed}
        print(
            "\n  NEGATIVE CONTROL: the input block was shuffled inside each screen. "
            f"{permutation['n_rows_moved']:,} of {permutation['n_rows']:,} rows moved "
            f"({permutation['fraction_moved']:.4f}) across {permutation['n_groups']} "
            "screens.\n  Nothing written to this directory describes cpg0021-periscope. "
            "Recoverability should collapse;\n  reliability reads only the target block "
            "and must come back unchanged.\n",
            flush=True,
        )
    (output / "provenance.json").write_text(json.dumps({
        "prepared": str(args.prepared),
        "reporter_id": REPORTER_ID,
        "aggregation_level": "guide",
        "seed": args.seed,
        "folds": args.folds,
        "alpha": args.alpha,
        "standardize_endpoints": args.standardize_endpoints,
        "alpha_ratio": args.alpha_ratio,
        "run_kind": "inputs_permuted_within_screen" if permutation else "observed",
        "permutation": permutation,
    }, indent=2) + "\n", encoding="utf-8")
    alpha = None if str(args.alpha).lower() == "auto" else float(args.alpha)
    if args.alpha_ratio is not None and alpha is not None:
        raise SystemExit("pass --alpha or --alpha-ratio, not both")
    responses, penalties = fit_folds(
        identifiers, x, y, endpoints, n_folds=args.folds, seed=args.seed,
        alpha=alpha, alpha_ratio=args.alpha_ratio, reporter_id=REPORTER_ID,
        unit="guide profiles",
    )
    penalties.to_csv(output / "penalty_selection.csv", index=False)
    responses.to_parquet(output / "responses.parquet", index=False)

    magnitudes = response_magnitudes(responses, standardize_endpoints=args.standardize_endpoints)
    magnitudes.to_csv(output / "magnitudes.csv", index=False)
    print(f"\nmax_endpoint_scale_share = "
          f"{float(magnitudes['max_endpoint_scale_share'].median()):.4f} "
          f"(1/n_endpoints would be {1 / len(endpoints):.5f} if they contributed evenly)")
    print(f"max_standardized_share  = "
          f"{float(magnitudes['max_standardized_share'].median()):.4f} "
          f"per fold: "
          + ", ".join(
              f"{int(f)}:{v:.4f}"
              for f, v in magnitudes.groupby("fold")["max_standardized_share"].median().items()
          )
          + "\n  the share of the standardised norm held by one endpoint after the "
            "divisor is applied. Near 1 means the norm is measuring a divisor.")
    print(f"endpoints entering the norm, per fold: "
          + ", ".join(
              f"{int(f)}:{int(v)}"
              for f, v in magnitudes.groupby("fold")["n_endpoints"].max().items()
          )
          + f" of {len(endpoints)}")

    spearman = magnitude_spearman(magnitudes)
    ratio = variance_ratio(magnitudes)
    recall = top_fraction_recall(magnitudes)
    for name, table in (("magnitude_spearman", spearman), ("variance_ratio", ratio),
                        ("strong_hit_recall", recall)):
        table.to_csv(output / f"{name}.csv", index=False)

    # Within-screen is the quantity the rule means; the pooled version is computed
    # too so the gap between them is reported rather than assumed.
    within = reliability_from_guides(frame, endpoints, seed=args.seed, within_screen=True)
    pooled = reliability_from_guides(frame, endpoints, seed=args.seed, within_screen=False)
    within.to_csv(output / "reliability_within_screen.csv", index=False)
    pooled.to_csv(output / "reliability_pooled_screens.csv", index=False)
    summary = summarise_reliability(within, floor=RELIABILITY_FLOOR)
    reliability = pd.DataFrame([{
        "reporter_id": REPORTER_ID,
        **summary,
        "reliability_pooled_across_screens": float(pooled["reliability"].median()),
    }])
    reliability.to_csv(output / "reliability.csv", index=False)
    print(f"\nwithin-screen reliability (median over {int(summary['n_blocks'])} screens): "
          f"{summary['reliability']:.4f}")
    print(f"  the same block pooled in native units would say "
          f"{summary['reliability_pooled_native_all_endpoints']:.4f}; see _reliability")
    print(f"  full guide complement by Spearman-Brown: "
          f"{summary['reliability_full_complement']:.4f}")
    print(f"  sgRNAs per gene that would be needed to reach the {RELIABILITY_FLOOR:.2f} "
          f"floor: {summary['guides_per_gene_for_floor']:.1f}, against a median of "
          f"{summary['median_guides_per_gene']:.0f} in the deposit")
    print(f"pooled across screens:                       "
          f"{reliability['reliability_pooled_across_screens'].iloc[0]:.4f}")

    recoverability = recoverability_by_fold(responses)
    recoverability.insert(0, "reporter_id", REPORTER_ID)
    recoverability.to_csv(output / "recoverability_by_fold.csv", index=False)
    print("\nrecoverability by fold. The first column is the one the rule consumes; the "
          "\nnative pooling beside it is what a block that does not share a scale reports, "
          "\nand the share column says how much of it is one endpoint.")
    print(recoverability.to_string(index=False))
    evidence = recoverability.groupby("reporter_id", as_index=False)["recoverability_r"].median()
    for table, column in ((spearman, "magnitude_spearman"), (ratio, "variance_ratio"),
                          (recall, "top5pct_recall")):
        evidence = evidence.merge(
            table.groupby("reporter_id", as_index=False)[column].median(), on="reporter_id", how="left"
        )
    # Name the column, never take it by position. ``_summarize`` emits
    # [group columns..., n_pairs, guide_half_consistency]; taking the first non-group
    # column picked up n_pairs, and a gene count of 20,119 sailed through the
    # reliability gate as if it were a correlation and decided the tier.
    evidence = evidence.merge(
        reliability[["reporter_id", "reliability"]], on="reporter_id", how="left"
    )
    if not evidence["reliability"].between(-1.0, 1.0).all():
        raise SystemExit(
            "reliability must be a correlation in [-1, 1]; got "
            f"{evidence['reliability'].tolist()}. A count or a sample size has leaked in."
        )
    evidence.to_csv(output / "evidence.csv", index=False)
    print("\nevidence table the frozen tier rule consumes:")
    print(evidence.to_string(index=False))

    tiers = tier_table(evidence, group_columns=["reporter_id"])
    tiers.to_csv(output / "tiers.csv", index=False)
    print("\ntier assignment under the frozen OPS rule, applied without tuning:")
    print(tiers.to_string(index=False))
    print("\nNOTE: guide-level recoverability is not the published quantity. Fitting on "
          "profiles already averaged within a (gene, sgRNA) removes noise from both "
          "sides; the paper predicts per cell and aggregates the predictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
