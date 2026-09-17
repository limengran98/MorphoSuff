"""Rebuild the frozen 172-dimensional OPS morphology representation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

try:  # direct script and package import are both supported
    from .h5ad import categorical_values, decode, file_sha256, require_h5ad_columns
except ImportError:  # pragma: no cover - exercised by CLI smoke
    from h5ad import categorical_values, decode, file_sha256, require_h5ad_columns


ORGANELLE_ORDER = (
    "phase2d_tubular",
    "phase2d_vesicular_dark",
    "phase2d_vesicular",
)
LOCALIZATION_METRICS = (
    "distance_from_cell_edge",
    "distance_from_nucleus",
    "distance_from_nucleus_centroid",
    "normalized_radial_position",
)
LOCALIZATION_AGGREGATIONS = ("sum", "mean", "median", "std", "min", "max", "count")
INTENSITY_ORGANELLE_ORDER = (
    "phase2d_vesicular_dark",
    "phase2d_tubular",
    "phase2d_vesicular",
)
INTENSITY_METRICS = ("intensity_min", "intensity_max", "intensity_mean", "intensity_range")
INTENSITY_AGGREGATIONS = ("mean", "median", "std", "sum", "min", "max")
SHAPE_METRICS = (
    "area",
    "perimeter",
    "axis_major_length",
    "axis_minor_length",
    "aspect_ratio",
    "solidity",
    "extent",
    "orientation",
    "circularity",
    "hu_moment_0",
    "hu_moment_1",
    "hu_moment_2",
    "hu_moment_3",
    "hu_moment_4",
    "hu_moment_5",
    "hu_moment_6",
)


def derive_feature_names(phase_path: Path) -> list[str]:
    """Return the exact ordered 172D feature panel from public H5AD metadata."""

    with h5py.File(phase_path, "r") as phase:
        require_h5ad_columns(
            phase,
            var=("_index", "source", "organelle", "category", "metric", "aggregation"),
        )
        frame = {
            "name": decode(phase["var/_index"][:]),
            "source": categorical_values(phase, "var", "source"),
            "organelle": categorical_values(phase, "var", "organelle"),
            "category": categorical_values(phase, "var", "category"),
            "metric": categorical_values(phase, "var", "metric"),
            "aggregation": categorical_values(phase, "var", "aggregation"),
        }
    lookup: dict[tuple[str, str, str, str], str] = {}
    for row_index, values in enumerate(zip(
        frame["name"], frame["organelle"], frame["category"], frame["metric"], frame["aggregation"]
    )):
        name, organelle, category, metric, aggregation = map(str, values)
        if frame["source"][row_index] == "organelle_profiler":
            lookup[(organelle, category, metric, aggregation)] = name

    selected: list[str] = []
    keys: list[tuple[str, str, str, str]] = []
    for organelle in ORGANELLE_ORDER:
        for metric in LOCALIZATION_METRICS:
            for aggregation in LOCALIZATION_AGGREGATIONS:
                keys.append((organelle, "localization", metric, aggregation))
    for organelle in INTENSITY_ORGANELLE_ORDER:
        for metric in INTENSITY_METRICS:
            for aggregation in INTENSITY_AGGREGATIONS:
                keys.append((organelle, "intensity", metric, aggregation))
    for metric in SHAPE_METRICS:
        keys.append(("cell", "cell_morphology", metric, ""))
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise ValueError(f"public phase H5AD lacks {len(missing)} locked 172D descriptors: {missing[:5]}")
    selected = [lookup[key] for key in keys]
    if len(selected) != 172 or len(set(selected)) != 172:
        raise AssertionError("the OPS 172D rule must yield 172 unique descriptors")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json", type=Path, help="optional machine-readable provenance record")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    features = derive_feature_names(args.phase)
    if args.dry_run:
        print(json.dumps({"n_features": len(features), "first": features[:5], "last": features[-5:]}, indent=2))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(features) + "\n", encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "schema_id": "ops-phase172-v1",
                    "source_phase": str(args.phase.resolve()),
                    "source_sha256": file_sha256(args.phase),
                    "n_features": 172,
                    "feature_names": features,
                    "selection": "frozen ordered morphology/localization/intensity descriptor rule",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"wrote 172 ordered phase features to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
