#!/usr/bin/env python3
"""Run the measurement-sufficiency chain on single-cell cpg0021-periscope tables.

This is the level that can be quoted against the published recoverability. The guide
level fits on profiles already averaged within a (gene, sgRNA), which removes noise
from the input and the target together and therefore reports an easier quantity. Here
the model predicts one cell at a time and the predictions are aggregated afterwards,
which is the paper's own construction.

Three implementation decisions are forced by scale, and each is stated because each
could otherwise be mistaken for a modelling choice.

**The ridge is solved from sufficient statistics, not from a design matrix per fold.**
1.08 million cells by 2,508 features is 10.8 GB in float32; five folds of centring and
scaling would copy it five times, and scikit-learn upcasts to float64 to form the
normal equations, which doubles it again. Instead one pass accumulates, per fold block,
the row count, the column sums and the uncentred second moments ``X'X`` and ``X'Y``.
Train statistics for a fold are the totals minus that fold's block, so all folds come
out of a single pass. The centring, scaling and solve are then pure algebra on a
2,508 by 2,508 matrix. This is not an approximation: ``ridge_via_moments`` is tested
against ``sklearn.linear_model.Ridge`` and agrees to floating point.

**Missing inputs are filled with the median over non-targeting control cells.** 27 of
the 2,508 feature columns carry a missing value somewhere, at an overall rate of 3e-6.
Controls are in the training set of every fold by construction, so a statistic computed
from controls alone is train-only for every fold and cannot leak a held-out gene. The
rate is far too low for the choice to move any number; it is made explicitly so that
nothing is silently dropped or filled from data the fold is supposed to be blind to.

**The control baseline is predicted in-sample.** A gene response is measured against
the same screen's non-targeting mean, so the predicted response needs a predicted
baseline, and controls never leave the training set. Both the observed and the
predicted baseline come from the same cells, and this matches the guide-level script.

**The ridge penalty is selected inside the training set, not fixed.** The 2,508
CellProfiler features are rank deficient: the training correlation matrix carries 155
eigenvalues below 1e-8, with the smallest at machine zero. A penalty of 1.0 against a
Gram whose diagonal is the row count is a ratio of 1.1e-6, far below those directions,
so the solve inverts numerical noise. The first run of this script did exactly that,
and one fold in five came back with predicted response magnitudes 52 times the
observed spread while the other four shrank normally. Alpha is therefore chosen by
nested cross-validation: for each outer fold, one of the remaining folds is held out
again as an inner validation set, alpha is picked on it, and only then is the model
refitted on the whole outer training set. The outer fold enters neither inner half, so
nothing about the held-out genes reaches the choice. The grid is expressed as a ratio
to the row count so it does not depend on how many cells happen to be in the dataset,
and a choice landing on either end of the grid is an error rather than a result.

Example:

    python studies/external/periscope/analyse_cells.py \\
      --prepared ../external/periscope_cells \\
      --output-dir ../external/periscope_cells_analysis --folds 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _common import reject_repo_paths  # noqa: E402

from analyse_guide import (  # noqa: E402
    NON_TARGETING,
    RELIABILITY_FLOOR,
    REPORTER_ID,
    reliability_from_guides,
)

from _reliability import summarise_reliability  # noqa: E402

from _ridge import fit_folds, impute_from_controls, recoverability_by_fold  # noqa: E402

from measurement_sufficiency.adapters import LocalManifestAdapter  # noqa: E402
from measurement_sufficiency.analysis.response_fidelity import (  # noqa: E402
    magnitude_spearman,
    response_magnitudes,
    top_fraction_recall,
    variance_ratio,
)
from measurement_sufficiency.analysis.tiers import tier_table  # noqa: E402

_IDENTIFIER_COLUMNS = {"input_cell_id", "observation_id", "target_cell_id", "plate"}


def load_cells(
    prepared: Path, *, verbose: bool = True
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], list[str]]:
    """Return observations plus the aligned float32 input and target matrices.

    Rows are ordered by the observations table, plate by plate. The parquet
    partitions are read once and copied into preallocated arrays, so the peak cost is
    the arrays themselves plus one plate of float64 that pandas hands back.
    """
    manifest = json.loads((prepared / "canonical_manifest.json").read_text(encoding="utf-8"))
    level = manifest.get("aggregation_level")
    if level != "single_cell":
        raise SystemExit(
            f"this script analyses single-cell tables; the manifest declares {level!r}. "
            f"Use analyse_guide.py for guide-level tables."
        )
    features = list(manifest["input_schema"]["feature_names"])
    endpoints = list(manifest["target_endpoint_columns"])

    observations = pd.read_parquet(prepared / manifest["tables"]["observations"])
    observations = observations.sort_values(["screen_id", "observation_id"], kind="stable")
    observations = observations.reset_index(drop=True)

    inputs_root = prepared / manifest["tables"]["inputs"]
    targets_root = prepared / manifest["tables"]["targets"]
    x = np.empty((len(observations), len(features)), dtype=np.float32)
    y = np.empty((len(observations), len(endpoints)), dtype=np.float32)
    filled = np.zeros(len(observations), dtype=bool)

    for screen, block in observations.groupby("screen_id", sort=True):
        input_files = sorted(inputs_root.glob(f"plate={screen}.*.parquet"))
        target_files = sorted(targets_root.glob(f"plate={screen}.*.parquet"))
        if not input_files or not target_files:
            raise SystemExit(f"screen {screen} has no input or target partition on disk")
        frames_in = [pd.read_parquet(path) for path in input_files]
        frames_tg = [pd.read_parquet(path) for path in target_files]
        plate_inputs = pd.concat(frames_in, ignore_index=True) if len(frames_in) > 1 else frames_in[0]
        plate_targets = pd.concat(frames_tg, ignore_index=True) if len(frames_tg) > 1 else frames_tg[0]
        del frames_in, frames_tg

        input_rows = plate_inputs.set_index("input_cell_id").reindex(block["input_cell_id"])
        target_rows = plate_targets.set_index("observation_id").reindex(block["observation_id"])
        if input_rows[features[0]].isna().all() or target_rows[endpoints[0]].isna().all():
            raise SystemExit(
                f"screen {screen}: the observation identifiers do not match the partition"
            )
        position = block.index.to_numpy()
        x[position] = input_rows[features].to_numpy(dtype=np.float32)
        y[position] = target_rows[endpoints].to_numpy(dtype=np.float32)
        filled[position] = True
        del plate_inputs, plate_targets, input_rows, target_rows
        if verbose:
            print(f"  loaded {screen}: {len(block):,} cells", flush=True)

    if not filled.all():
        raise SystemExit(f"{int((~filled).sum()):,} observations never received a feature row")
    return observations, x, y, features, endpoints


def guide_profiles(
    observations: pd.DataFrame, y: np.ndarray, endpoints: list[str]
) -> pd.DataFrame:
    """Collapse cells to one row per (screen, gene, guide) for the reliability estimate.

    The reliability the tier rule gates on is a property of the targeted measurement,
    so it is estimated on the same cells the model saw rather than imported from the
    guide-level table. That table is computed on every cell of the screen rather than
    this subsample, and the two are printed side by side by the caller.
    """
    keys = observations[["screen_id", "perturbation_id", "guide_id", "is_control"]]
    frame = pd.concat([keys.reset_index(drop=True), pd.DataFrame(y, columns=endpoints)], axis=1)
    collapsed = frame.groupby(
        ["screen_id", "perturbation_id", "guide_id", "is_control"], as_index=False, sort=True
    )[endpoints].mean()
    return collapsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--alpha",
        default="auto",
        help="ridge penalty on standardised features. 'auto' selects it per fold on an "
             "inner validation fold that the held-out fold never enters. A number "
             "fixes it, which reproduces the unregularised first run: the design "
             "carries 155 correlation eigenvalues at numerical zero, so a fixed 1.0 "
             "against a Gram whose diagonal is the row count regularises nothing",
    )
    parser.add_argument(
        "--standardize-endpoints", choices=("none", "observed_sd"), default="observed_sd",
        help="the 823 endpoints are in native units and one of them holds most of the "
             "squared spread, so the default puts them on a common footing before the norm",
    )
    parser.add_argument(
        "--write-responses", action="store_true",
        help="also write the 150M-row response table; the summaries are written either way",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = reject_repo_paths(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = LocalManifestAdapter(args.prepared / "canonical_manifest.json")
    observations, x, y, features, endpoints = load_cells(args.prepared)
    is_control = observations["is_control"].to_numpy()
    fill, n_columns_filled, n_cells_filled = impute_from_controls(x, is_control)
    print(json.dumps({
        "cells": len(observations),
        "screens": int(observations.screen_id.nunique()),
        "genes": int(observations.perturbation_id.nunique()),
        "guides": int(observations.guide_id.nunique()),
        "control_cells": int(is_control.sum()),
        "features": len(features),
        "endpoints": len(endpoints),
        "input_columns_imputed": n_columns_filled,
        "input_cells_imputed": n_cells_filled,
        "capabilities": dataset.capabilities.__dict__,
    }, indent=2), flush=True)

    alpha = None if str(args.alpha).lower() == "auto" else float(args.alpha)
    responses, penalties = fit_folds(
        observations, x, y, endpoints, n_folds=args.folds, seed=args.seed,
        alpha=alpha, reporter_id=REPORTER_ID, unit="cells",
    )
    penalties.to_csv(output / "penalty_selection.csv", index=False)
    if args.write_responses:
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

    profiles = guide_profiles(observations, y, endpoints)
    within = reliability_from_guides(profiles, endpoints, seed=args.seed, within_screen=True)
    pooled = reliability_from_guides(profiles, endpoints, seed=args.seed, within_screen=False)
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
          f"{summary['median_guides_per_gene']:.0f} in this subsample")
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
            table.groupby("reporter_id", as_index=False)[column].median(),
            on="reporter_id", how="left",
        )
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
    print(
        "\nNOTE: this is the level comparable to the published recoverability, but the "
        "cheap input here is four fluorescent dyes rather than a transmitted-light "
        "channel, so the number tests the perturbation and environment half of the "
        "claim and must never be quoted as label-free evidence. Segmentation is also "
        "driven by images that include the target channel."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
