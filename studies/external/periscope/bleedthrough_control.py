#!/usr/bin/env python3
"""Decide whether four dyes predicting anti-TOMM20 is biology or spectral crosstalk.

PERISCOPE offers something no other candidate does: its expensive readout is
anti-TOMM20, and this study measures ``mitochondria, TOMM20 (cp)`` as one of its own
52 reporters, so it is a same-target comparison across platforms. That is worth having
only if the prediction is measuring cells rather than the microscope.

The concern is concrete. Anti-TOMM20 is Alexa Fluor 594 with emission at 615/24 nm;
ConA sits at 680/42 nm, phalloidin at 530/30 nm, and DAPI near 450 nm. The published
destaining addresses phenotype-versus-in-situ-sequencing overlap and reports no
phenotype-to-phenotype unmixing, so some AF594 signal may leak into the dye channels.
If it does, an input feature is partly a copy of the target and a high score would be
optics, not sufficiency.

**A within-site permutation test alone cannot settle this**, which is worth stating
because it is the obvious control to reach for. Crosstalk is a per-cell effect, exactly
like biology: a cell with more TOMM20 leaks more into its own neighbouring channel.
Permutation destroys both together. It answers "is the signal cell-specific rather than
site-level", which is necessary but does not separate the two cell-specific
explanations.

What does separate them is a physical constraint on where a leak can go. A fluorophore
emitting at 615 nm can contribute to a detection band above it but not to one at
450 nm, so **the DAPI channel is a physical negative control for AF594 crosstalk**:

    optics    the signal concentrates in ConA (680 nm, adjacent to the target) and
              DAPI (450 nm) carries little, because a leak into DAPI is not available
    biology   DAPI carries real signal too, because nuclear morphology reflects cell
              state, and the four channels are comparably informative

Three tests are run. The first is the decisive one and the third is weaker than it
looks:

1. **Per-channel ablation.** Fit on one input channel at a time. A substantial
   DAPI-only score cannot be crosstalk from a 615 nm emitter and is therefore
   evidence for biology; a large ConA-over-DAPI gap is the crosstalk signature.
2. **Within-site permutation.** Permute the target among cells inside each imaging
   site, refit, and compare against a site-mean-only predictor, which is the floor a
   model exploiting site structure alone could reach. This separates cell-level signal
   from site-level artefact; it does not separate optics from biology, because
   crosstalk is cell-specific too.
3. **Feature-family ablation inside the adjacent channel.** Reported for completeness
   and **not** treated as discriminating. The intuition that a leak shows up only in
   intensity is wrong: a linear leak adds a scaled copy of the whole target image, so
   texture and granularity are contaminated as well. A large gap here would be
   suggestive, but its absence proves nothing.

Whole imaging sites are held out, which is the field-split analogue for this dataset.
Nothing here decides the panel by itself: it decides whether spending 125 GB on the
full arm is warranted.

Example:

    python studies/external/periscope/bleedthrough_control.py \\
      --prepared ../external/periscope_ctrl \\
      --output-dir ../external/periscope_control --max-cells 60000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from _common import CellProfilerChannels, reject_repo_paths  # noqa: E402

CHANNELS = CellProfilerChannels(("DAPI", "ConA", "Mito", "Phalloidin", "WGA"))
INPUT_CHANNELS = ("DAPI", "ConA", "Phalloidin", "WGA")

#: Emission maxima, used only to order the channels by spectral distance from the
#: AF594 target. Provenance differs by channel and is recorded rather than implied,
#: because the test rests on this ordering:
#:
#:   Mito 615        verified: anti-TOMM20 secondary is Alexa Fluor 594, 615/24 nm
#:   ConA 680        verified: 680/42 nm
#:   Phalloidin 530  verified: 530/30 nm
#:   DAPI 450        standard for DAPI; not read from this study's methods
#:   WGA 560         NOT VERIFIED for this deposit; a conventional Alexa 555 value
#:
#: The discriminating comparison is ConA against DAPI, whose values are the verified
#: adjacent channel and the well-established distant one. WGA and phalloidin are
#: reported but must not be leaned on for the verdict while WGA's value is a guess.
EMISSION_NM = {"DAPI": 450, "Phalloidin": 530, "Mito": 615, "ConA": 680, "WGA": 560}
UNVERIFIED_EMISSION = ("WGA",)
TARGET_EMISSION_NM = EMISSION_NM["Mito"]

#: CellProfiler feature families. Intensity carries a linear optical leak; the others
#: describe spatial structure at scales a leak does not reproduce.
INTENSITY_FAMILIES = ("Intensity", "ImageQuality")
STRUCTURE_FAMILIES = ("Texture", "Granularity", "RadialDistribution")


def load(prepared: Path, max_cells: int | None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    manifest = json.loads((prepared / "canonical_manifest.json").read_text(encoding="utf-8"))
    features = list(manifest["input_schema"]["feature_names"])

    def read(name: str) -> pd.DataFrame:
        path = prepared / manifest["tables"][name]
        return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)

    observations = read("observations")
    if max_cells is not None and len(observations) > max_cells:
        # Truncate on whole sites so the permutation test keeps complete sites.
        sites = observations["field_id"].drop_duplicates()
        keep, taken = [], 0
        counts = observations.groupby("field_id").size()
        for site in sites:
            if taken >= max_cells:
                break
            keep.append(site)
            taken += int(counts[site])
        observations = observations.loc[observations["field_id"].isin(keep)]
    # The inputs table is one row per cell by 2,508 features; reading it whole to keep
    # a fifth of it would cost several gigabytes for no reason. Filter while reading.
    wanted = set(observations["input_cell_id"])
    inputs = _read_filtered(prepared / manifest["tables"]["inputs"], "input_cell_id", wanted)
    targets = _read_filtered(
        prepared / manifest["tables"]["targets"], "observation_id", set(observations["observation_id"])
    )
    frame = observations[["observation_id", "input_cell_id", "field_id"]].merge(
        inputs, on="input_cell_id", how="inner"
    )
    return observations, frame.merge(
        targets.pivot(index="observation_id", columns="endpoint_id", values="y_true"),
        on="observation_id",
        how="inner",
    ), features


def _read_filtered(path: Path, key: str, keep: set[str]) -> pd.DataFrame:
    """Read a table keeping only rows whose ``key`` is in ``keep``, batch by batch."""
    if path.suffix != ".parquet":
        frame = pd.read_csv(path)
        return frame.loc[frame[key].isin(keep)]
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    # Filter in Arrow, not in pandas. A long target table for a reporter with 823
    # endpoints holds one row per cell per endpoint, so a 150,000-cell subset is
    # already 123 million rows. Converting it to pandas batch by batch spends all its
    # time materialising object-dtype strings; Arrow keeps them dictionary-encoded and
    # runs ``is_in`` over the whole column at once.
    table = pq.read_table(path)
    selected = table.filter(pc.is_in(table[key], value_set=pa.array(sorted(keep))))
    if selected.num_rows == 0:
        raise ValueError(f"no rows of {path.name} matched the requested {key} values")
    return selected.to_pandas()


def channel_columns(features: list[str], channel: str) -> list[str]:
    return [column for column in features if CHANNELS.of(column) == frozenset({channel})]


def family_columns(columns: list[str], families: tuple[str, ...]) -> list[str]:
    return [column for column in columns if any(family in column for family in families)]


def fit_and_score(
    frame: pd.DataFrame, feature_columns: list[str], target_columns: list[str], *, seed: int
) -> dict[str, float]:
    """Hold out whole sites, fit Ridge, and return the median per-endpoint Pearson."""
    from sklearn.linear_model import Ridge

    sites = np.sort(frame["field_id"].unique())
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(sites)[: max(1, len(sites) // 5)])
    train = frame.loc[~frame["field_id"].isin(held)]
    test = frame.loc[frame["field_id"].isin(held)]
    if train.empty or test.empty:
        raise ValueError("site holdout produced an empty partition")

    x_train = train[feature_columns].to_numpy(dtype=np.float32)
    x_test = test[feature_columns].to_numpy(dtype=np.float32)
    y_train = train[target_columns].to_numpy(dtype=np.float32)
    y_test = test[target_columns].to_numpy(dtype=np.float32)

    # Train-only imputation and standardisation, as everywhere else in this study.
    median = np.nanmedian(y_train, axis=0)
    x_median = np.nanmedian(x_train, axis=0)
    x_train = np.where(np.isnan(x_train), x_median, x_train)
    x_test = np.where(np.isnan(x_test), x_median, x_test)
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale == 0] = 1.0
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    y_train = np.where(np.isnan(y_train), median, y_train)

    model = Ridge(alpha=1.0, random_state=seed).fit(x_train, y_train)
    predicted = model.predict(x_test)

    scores = []
    for index in range(y_test.shape[1]):
        observed = y_test[:, index]
        mask = ~np.isnan(observed)
        if mask.sum() < 3:
            continue
        a, b = observed[mask], predicted[mask, index]
        if a.std() == 0 or b.std() == 0:
            continue
        scores.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "n_features": len(feature_columns),
        "n_endpoints_scored": len(scores),
        "median_pearson": float(np.median(scores)) if scores else float("nan"),
        "n_train_cells": int(len(train)),
        "n_test_cells": int(len(test)),
        "n_held_sites": len(held),
    }


def permute_within_site(frame: pd.DataFrame, target_columns: list[str], *, seed: int) -> pd.DataFrame:
    """Shuffle the target block among cells inside each imaging site.

    Everything shared at the site level survives: illumination, focus, well, batch and
    local density. Only the cell-to-cell correspondence is destroyed.
    """
    rng = np.random.default_rng(seed)
    permuted = frame.copy()
    # An Arrow-backed frame hands back a read-only view, so copy before shuffling.
    values = np.array(frame[target_columns].to_numpy(), copy=True)
    for _, index in frame.groupby("field_id", sort=False).indices.items():
        if len(index) > 1:
            values[index] = values[index[rng.permutation(len(index))]]
    permuted[target_columns] = values
    return permuted


def site_mean_floor(frame: pd.DataFrame, target_columns: list[str], *, seed: int) -> dict[str, float]:
    """Score a predictor that knows only which site a cell came from.

    This is the level a model exploiting site structure rather than cell content could
    reach, and it is the reference the permuted fit must be compared against rather
    than against zero.
    """
    sites = np.sort(frame["field_id"].unique())
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(sites)[: max(1, len(sites) // 5)])
    train = frame.loc[~frame["field_id"].isin(held)]
    test = frame.loc[frame["field_id"].isin(held)]
    means = train.groupby("field_id")[target_columns].mean()
    grand = train[target_columns].mean()
    predicted = test["field_id"].map(lambda site: site).to_frame("field_id").join(
        means, on="field_id"
    )[target_columns].fillna(grand)
    scores = []
    for column in target_columns:
        a, b = test[column].to_numpy(), predicted[column].to_numpy()
        mask = ~np.isnan(a) & ~np.isnan(b)
        if mask.sum() < 3 or a[mask].std() == 0 or b[mask].std() == 0:
            continue
        scores.append(float(np.corrcoef(a[mask], b[mask])[0, 1]))
    return {"n_endpoints_scored": len(scores),
            "median_pearson": float(np.median(scores)) if scores else float("nan")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cells", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = reject_repo_paths(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    observations, frame, features = load(args.prepared, args.max_cells)
    manifest = json.loads((args.prepared / "canonical_manifest.json").read_text(encoding="utf-8"))
    target_columns = [
        column for column in frame.columns
        if column not in {"observation_id", "input_cell_id", "field_id"}
        and column not in set(manifest["input_schema"]["feature_names"])
    ]
    print(json.dumps({
        "cells": len(frame), "sites": int(frame.field_id.nunique()),
        "input_features": len(features), "target_endpoints": len(target_columns),
    }, indent=2))

    rows = []

    # Test 1: per-channel ablation, ordered by spectral distance from the target.
    for channel in sorted(INPUT_CHANNELS, key=lambda name: abs(EMISSION_NM[name] - TARGET_EMISSION_NM)):
        columns = channel_columns(features, channel)
        if not columns:
            raise SystemExit(f"no features attributable to {channel} alone")
        result = fit_and_score(frame, columns, target_columns, seed=args.seed)
        rows.append({
            "test": "per_channel", "arm": channel,
            "emission_nm": EMISSION_NM[channel],
            "spectral_distance_nm": abs(EMISSION_NM[channel] - TARGET_EMISSION_NM),
            "emission_verified": channel not in UNVERIFIED_EMISSION,
            **result,
        })
        print(f"  channel {channel:11s} ({EMISSION_NM[channel]} nm, "
              f"{abs(EMISSION_NM[channel] - TARGET_EMISSION_NM):3d} nm from target): "
              f"median r = {result['median_pearson']:.3f}", flush=True)

    all_input = [column for column in features if CHANNELS.of(column) and CHANNELS.of(column) <= set(INPUT_CHANNELS)]
    baseline = fit_and_score(frame, all_input, target_columns, seed=args.seed)
    rows.append({"test": "baseline", "arm": "all_four_channels", **baseline})
    print(f"  baseline, all four channels: median r = {baseline['median_pearson']:.3f}")

    # Test 2: within-site permutation against the site-mean floor.
    permuted = fit_and_score(
        permute_within_site(frame, target_columns, seed=args.seed), all_input, target_columns, seed=args.seed
    )
    floor = site_mean_floor(frame, target_columns, seed=args.seed)
    rows.append({"test": "permutation", "arm": "within_site_permuted", **permuted})
    rows.append({"test": "permutation", "arm": "site_mean_only", **floor})
    print(f"  within-site permuted:        median r = {permuted['median_pearson']:.3f}")
    print(f"  site-mean-only floor:        median r = {floor['median_pearson']:.3f}")

    # Test 3: feature families inside the spectrally adjacent channel.
    # Restricted to channels whose emission is verified, so the family ablation is not
    # anchored on a guessed wavelength.
    adjacent = min(
        [name for name in INPUT_CHANNELS if name not in UNVERIFIED_EMISSION],
        key=lambda name: abs(EMISSION_NM[name] - TARGET_EMISSION_NM),
    )
    adjacent_columns = channel_columns(features, adjacent)
    for label, families in (("intensity", INTENSITY_FAMILIES), ("structure", STRUCTURE_FAMILIES)):
        columns = family_columns(adjacent_columns, families)
        if not columns:
            continue
        result = fit_and_score(frame, columns, target_columns, seed=args.seed)
        rows.append({"test": "feature_family", "arm": f"{adjacent}_{label}", **result})
        print(f"  {adjacent} {label:9s} only:  median r = {result['median_pearson']:.3f}")

    table = pd.DataFrame(rows)
    table.to_csv(output / "bleedthrough_control.csv", index=False)
    print("\n" + table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
