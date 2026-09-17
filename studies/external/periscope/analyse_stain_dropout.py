#!/usr/bin/env python3
"""Predict each Cell Painting dye from the other three, on cpg0021-periscope guides.

This asks a second question of the same deposit. The anti-TOMM20 arm asks whether four
dyes can stand in for an antibody. This asks whether three dyes can stand in for the
fourth, which is the decision a laboratory actually faces when it wants a channel back
for something else, and it exercises the frozen criteria on four more reporters at no
new download.

**This is not an admitted validation case, and it must never be presented as one.**
The admission audit judges a candidate on six capabilities, and C2 asks for a targeted
protein, reporter, antibody or biochemical readout. A generic organelle dye is not
that; ``cpg0000-jump-pilot`` was rejected on exactly this ground, for exactly this
reason, and admitting these four rows through the same gate that rejected it would
make the gate meaningless. What these rows test is the criteria's behaviour on a
different kind of substitution question, not the transfer of the measurement
sufficiency claim to a new dataset. The manuscript reports them in their own band.

**The mask confound is load-bearing here in a way it is not for the antibody arm.**
Segmentation is driven by phenotypic images that include every channel, so a target
dye participates in defining the mask its own features are measured in.
``_common.partition_features`` removes the two routes by which that becomes a direct
leak, excluding every column attributable to more than one channel and every column
attributable to none, which is where ``AreaShape`` and the threshold features live. It
cannot remove the residual: an intensity measured inside a shared mask is still
measured inside a shared mask. The antibody arm carries the same caveat and fails
anyway, so nothing there rests on it. A dye row that reaches a proxy tier would rest
on it entirely, which is why the caveat is stated in Methods rather than in a footnote,
and why the strict control, re-segmenting from the three input channels alone, is named
as not done.

Feature attribution is positive and is checked rather than trusted: after the split,
no input column may be attributable to the held-out dye. The check runs before any
fitting and raises.

Example:

    python studies/external/periscope/analyse_stain_dropout.py \\
      --prepared ../external/periscope_guide \\
      --output-dir ../external/periscope_stain_dropout --folds 5
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

from _common import CellProfilerChannels, partition_features, reject_repo_paths  # noqa: E402

from _reliability import summarise_reliability  # noqa: E402

from _ridge import fit_folds, impute_from_controls, recoverability_by_fold  # noqa: E402

# The shared penalty grid is used unchanged. The first two runs of this arm failed with
# "the selected penalty ratio is at the edge of the grid", and the obvious reading was
# that predicting one dye from three others is so easy that shrinkage never helps, so
# the grid was extended four decades downward. It failed again at the new edge, and the
# reading was wrong: every validation score on the first held-out dye was NaN, and
# ``min`` over NaN scores returns the first element, which is the smallest ratio, which
# the boundary check then reported. The cause was a missing value in the target block,
# handled below. Measured afterwards on the WGA split, the validation error is
# well behaved and its minimum sits at alpha/n = 1e-3, comfortably inside the shared
# grid. ``select_alpha`` now refuses a non-finite score outright, so this cannot present
# as a boundary problem again.

from analyse_guide import RELIABILITY_FLOOR, reliability_from_guides  # noqa: E402

from measurement_sufficiency.analysis.response_fidelity import (  # noqa: E402
    magnitude_spearman,
    response_magnitudes,
    top_fraction_recall,
    variance_ratio,
)
from measurement_sufficiency.analysis.tiers import tier_table  # noqa: E402

#: The same vocabulary the preparation used. Mito is listed so that a Mito column
#: appearing in the input block would be recognised and refused rather than silently
#: attributed to nothing; the preparation already routes those to the target table.
CHANNELS = CellProfilerChannels(("DAPI", "ConA", "Mito", "Phalloidin", "WGA"))
DYES = ("DAPI", "ConA", "Phalloidin", "WGA")

#: What each dye stains, for the reporter table. These are the standard Cell Painting
#: assignments as the PERISCOPE deposit names its channels, not an interpretation.
STAINED = {
    "DAPI": "nucleus",
    "ConA": "endoplasmic reticulum",
    "Phalloidin": "F-actin",
    "WGA": "Golgi and plasma membrane",
}


def load_inputs(prepared: Path) -> tuple[pd.DataFrame, list[str]]:
    """Observations joined to the dye feature block. The Mito target block is not read.

    The 823 anti-TOMM20 columns belong to the other arm and are several gigabytes;
    nothing here uses them.
    """
    manifest = json.loads((prepared / "canonical_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("aggregation_level") != "guide":
        raise SystemExit(
            f"this arm runs on guide-level tables; the manifest declares "
            f"{manifest.get('aggregation_level')!r}"
        )
    features = list(manifest["input_schema"]["feature_names"])
    observations = pd.read_parquet(prepared / manifest["tables"]["observations"])
    inputs = pd.read_parquet(prepared / manifest["tables"]["inputs"])
    return observations.merge(inputs, on="input_cell_id", how="inner"), features


def split_for(target: str, features: list[str]) -> dict[str, list[str]]:
    """Partition the dye features into input and target for one held-out dye.

    Raises if any surviving input column is attributable to the held-out dye. That
    cannot happen given how :func:`partition_features` decides, which is the point:
    the guarantee is asserted where it is relied on rather than inferred from a
    docstring two modules away.
    """
    inputs = tuple(dye for dye in DYES if dye != target)
    partition = partition_features(
        features, channels=CHANNELS, input_channels=inputs, target_channels=(target,)
    )
    leaked = [column for column in partition["input"] if target in CHANNELS.of(column)]
    if leaked:
        raise SystemExit(
            f"{len(leaked)} input columns are attributable to the held-out {target} "
            f"channel, for example {leaked[:3]}. The split leaks the target."
        )
    if not partition["target"]:
        raise SystemExit(f"no column is attributable to {target} alone")
    return partition


def analyse_one(
    frame: pd.DataFrame, partition: dict[str, list[str]], target: str, args: argparse.Namespace,
    output: Path,
) -> pd.DataFrame:
    """Run the full chain for one held-out dye and return its evidence row."""
    reporter_id = f"stain_dropout_{target.lower()}"
    output.mkdir(parents=True, exist_ok=True)
    features, endpoints = partition["input"], partition["target"]
    print(f"\n=== {target}: {len(features):,} input columns from "
          f"{', '.join(dye for dye in DYES if dye != target)}, "
          f"{len(endpoints):,} target columns ===", flush=True)
    print("    excluded: "
          + ", ".join(f"{key.removeprefix('excluded_')} {len(value):,}"
                      for key, value in partition.items() if key.startswith("excluded")),
          flush=True)

    # A missing value in the target block is not imputable: imputing the quantity being
    # predicted would fabricate the answer. It is also not survivable, because sumsq_y
    # and the cross moments carry it into every endpoint's validation error at once.
    # So an endpoint with any missing value is dropped and counted. This is why the
    # first run of this arm failed: 30 columns of the dye block carry a missing value
    # somewhere, they land in the DAPI target block, and every penalty then scored NaN.
    incomplete = [
        column for column in endpoints if frame[column].isna().any()
    ]
    if incomplete:
        endpoints = [column for column in endpoints if column not in set(incomplete)]
        print(f"    dropped {len(incomplete)} target columns carrying a missing value, "
              f"for example {incomplete[:2]}; {len(endpoints):,} endpoints remain",
              flush=True)
    if not endpoints:
        raise SystemExit(f"every {target} target column carries a missing value")

    identifiers = frame[["screen_id", "perturbation_id", "is_control"]].reset_index(drop=True)
    x = frame[features].to_numpy(dtype=np.float32)
    y = frame[endpoints].to_numpy(dtype=np.float32)
    _, n_columns_filled, n_cells_filled = impute_from_controls(
        x, identifiers["is_control"].to_numpy()
    )
    print(f"    imputed {n_cells_filled:,} missing inputs across {n_columns_filled} columns",
          flush=True)

    alpha = None if str(args.alpha).lower() == "auto" else float(args.alpha)
    responses, penalties = fit_folds(
        identifiers, x, y, endpoints, n_folds=args.folds, seed=args.seed,
        alpha=alpha, reporter_id=reporter_id, unit="guide profiles",
    )
    penalties.to_csv(output / "penalty_selection.csv", index=False)
    del x, y

    magnitudes = response_magnitudes(responses, standardize_endpoints=args.standardize_endpoints)
    magnitudes.to_csv(output / "magnitudes.csv", index=False)
    spearman = magnitude_spearman(magnitudes)
    ratio = variance_ratio(magnitudes)
    recall = top_fraction_recall(magnitudes)
    for name, table in (("magnitude_spearman", spearman), ("variance_ratio", ratio),
                        ("strong_hit_recall", recall)):
        table.to_csv(output / f"{name}.csv", index=False)

    within = reliability_from_guides(
        frame, endpoints, seed=args.seed, within_screen=True, reporter_id=reporter_id
    )
    within.to_csv(output / "reliability_within_screen.csv", index=False)
    summary = summarise_reliability(within, floor=RELIABILITY_FLOOR)
    pd.DataFrame([{"reporter_id": reporter_id, **summary}]).to_csv(
        output / "reliability.csv", index=False
    )

    recoverability = recoverability_by_fold(responses)
    recoverability.insert(0, "reporter_id", reporter_id)
    recoverability.to_csv(output / "recoverability_by_fold.csv", index=False)

    evidence = recoverability.groupby("reporter_id", as_index=False)["recoverability_r"].median()
    for table, column in ((spearman, "magnitude_spearman"), (ratio, "variance_ratio"),
                          (recall, "top5pct_recall")):
        evidence = evidence.merge(
            table.groupby("reporter_id", as_index=False)[column].median(),
            on="reporter_id", how="left",
        )
    evidence["reliability"] = summary["reliability"]
    if not evidence["reliability"].between(-1.0, 1.0).all():
        raise SystemExit(
            f"reliability must be a correlation in [-1, 1]; got "
            f"{evidence['reliability'].tolist()}. A count or a sample size has leaked in."
        )
    evidence["stain"] = target
    evidence["stains"] = STAINED[target]
    evidence["n_input_features"] = len(features)
    evidence["n_endpoints"] = len(endpoints)
    evidence["reliability_full_complement"] = summary["reliability_full_complement"]
    evidence["guides_per_gene_for_floor"] = summary["guides_per_gene_for_floor"]
    evidence.to_csv(output / "evidence.csv", index=False)
    tier_table(evidence, group_columns=["reporter_id"]).to_csv(output / "tiers.csv", index=False)
    print(evidence.to_string(index=False), flush=True)
    del responses, magnitudes
    return evidence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-dir", required=True, help="destination outside the repository")
    parser.add_argument("--target-dye", action="append", choices=DYES, default=None,
                        help="repeatable; default is all four")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", default="auto")
    parser.add_argument("--standardize-endpoints", choices=("none", "observed_sd"),
                        default="observed_sd")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = reject_repo_paths(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = tuple(args.target_dye) if args.target_dye else DYES

    frame, features = load_inputs(args.prepared)
    stray = [column for column in features if "Mito" in CHANNELS.of(column)]
    if stray:
        raise SystemExit(
            f"{len(stray)} input columns are attributable to the anti-TOMM20 channel, "
            f"for example {stray[:3]}. The preparation routes those to the target table; "
            "this arm must not see them."
        )
    print(json.dumps({
        "rows": len(frame), "screens": int(frame.screen_id.nunique()),
        "genes": int(frame.perturbation_id.nunique()), "guides": int(frame.guide_id.nunique()),
        "controls": int(frame.is_control.sum()), "dye_features": len(features),
        "targets": list(targets),
    }, indent=2), flush=True)

    # Every split is validated before any of them is fitted, so a leak in the fourth
    # split is not discovered after three fits have already been written to disk.
    partitions = {target: split_for(target, features) for target in targets}
    rows = [
        analyse_one(frame, partitions[target], target, args, output / target.lower())
        for target in targets
    ]

    evidence = pd.concat(rows, ignore_index=True)
    evidence.to_csv(output / "evidence.csv", index=False)
    tiers = tier_table(evidence, group_columns=["reporter_id"])
    tiers.to_csv(output / "tiers.csv", index=False)
    print("\nfrozen OPS rule applied without tuning, four held-out dyes:")
    print(tiers.to_string(index=False))
    print("\nNOTE: a generic organelle dye is not a targeted readout under C2, so these "
          "rows are not admitted validation datasets. They are a second substitution "
          "question on an admitted deposit, and the shared-mask caveat applies to every "
          "one of them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
