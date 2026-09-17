#!/usr/bin/env python3
"""Validate the compact processed OPS training release.

The release stores the shared 172D phase matrix once and one sparse target
cache per reporter.  It deliberately contains no raw microscopy or source
H5AD objects.  Validation is read-only and does not materialize a second copy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


EXPECTED = {
    "n_phase_cells": 8_410_291,
    "n_phase_features": 172,
    "n_reporters": 52,
    "n_screens": 73,
    "n_reporter_screen_assays": 99,
    "n_exact_same_cell_observations": 9_996_286,
    "endpoint_block_distribution": {"24": 22, "30": 6, "72": 24},
}


def _decode(values: Any) -> set[str]:
    return {value.decode() if isinstance(value, bytes) else str(value) for value in values}


def validate(root: Path, *, strict_census: bool = True) -> dict[str, Any]:
    root = root.expanduser().resolve()
    phase_root = root / "processed_ops" / "phase172"
    exact_root = root / "processed_ops" / "exact_reporters"
    metadata_root = root / "processed_ops" / "metadata"
    required_phase = {
        "phase172.npy",
        "phase_features_172d.txt",
        "categories.json",
        "screen_code.npy",
        "well_code.npy",
        "tile_code.npy",
        "gene_code.npy",
        "sgrna_code.npy",
        "is_target.npy",
        "field_holdout_sanity.fold.npy",
        "gene_holdout_main.fold.npy",
    }
    required_metadata = {
        "reporter_targets.csv",
        "target_feature_dictionary.csv",
        "target_screen_map.csv",
        "exact_pairing_integrity.csv",
    }
    missing = [str(phase_root / name) for name in required_phase if not (phase_root / name).is_file()]
    missing += [str(metadata_root / name) for name in required_metadata if not (metadata_root / name).is_file()]
    if missing:
        raise FileNotFoundError("Processed OPS release is incomplete:\n" + "\n".join(sorted(missing)))
    features = [line for line in (phase_root / "phase_features_172d.txt").read_text().splitlines() if line.strip()]
    if len(features) != 172 or len(set(features)) != 172:
        raise RuntimeError("phase_features_172d.txt must contain 172 unique names")
    phase = np.load(phase_root / "phase172.npy", mmap_mode="r")
    if phase.ndim != 2 or phase.shape[1] != 172 or phase.dtype != np.dtype("float32"):
        raise RuntimeError(f"Invalid phase matrix: shape={phase.shape}, dtype={phase.dtype}")
    for name in (
        "screen_code.npy",
        "well_code.npy",
        "tile_code.npy",
        "gene_code.npy",
        "sgrna_code.npy",
        "is_target.npy",
        "field_holdout_sanity.fold.npy",
        "gene_holdout_main.fold.npy",
    ):
        array = np.load(phase_root / name, mmap_mode="r")
        if len(array) != len(phase):
            raise RuntimeError(f"{name} length {len(array)} != phase rows {len(phase)}")
    reporter_table = pd.read_csv(metadata_root / "reporter_targets.csv")
    if reporter_table.reporter_slug.astype(str).duplicated().any():
        raise RuntimeError("reporter_targets.csv contains duplicate reporter_slug values")
    exact_paths = sorted(exact_root.glob("all_cells_fluor_*.exact.h5"))
    required_h5 = {
        "phase_row_index",
        "fluorescence_row_index",
        "fluorescence",
        "features/target_feature_names",
        "metadata/screen_categories",
        "metadata/gene_codes",
        "metadata/sgRNA_codes",
        "metadata/is_control",
    }
    total_rows = 0
    assays = 0
    screens: set[str] = set()
    endpoint_distribution: dict[str, int] = {}
    for path in exact_paths:
        with h5py.File(path, "r") as handle:
            absent = sorted(key for key in required_h5 if key not in handle)
            if absent:
                raise RuntimeError(f"{path.name} lacks {absent}")
            rows = int(handle["phase_row_index"].shape[0])
            if int(handle["fluorescence_row_index"].shape[0]) != rows:
                raise RuntimeError(f"Pairing row mismatch in {path.name}")
            if int(handle["fluorescence"].shape[0]) != rows:
                raise RuntimeError(f"Target row mismatch in {path.name}")
            phase_rows = np.asarray(handle["phase_row_index"][:], dtype=np.int64)
            if len(phase_rows) and (phase_rows.min() < 0 or phase_rows.max() >= len(phase)):
                raise RuntimeError(f"Out-of-range phase_row_index in {path.name}")
            local_screens = _decode(handle["metadata/screen_categories"][:])
            endpoints = str(int(handle["fluorescence"].shape[1]))
            endpoint_distribution[endpoints] = endpoint_distribution.get(endpoints, 0) + 1
            total_rows += rows
            assays += len(local_screens)
            screens.update(local_screens)
    observed = {
        "n_phase_cells": int(phase.shape[0]),
        "n_phase_features": int(phase.shape[1]),
        "n_reporters": len(exact_paths),
        "n_screens": len(screens),
        "n_reporter_screen_assays": assays,
        "n_exact_same_cell_observations": total_rows,
        "endpoint_block_distribution": dict(sorted(endpoint_distribution.items(), key=lambda item: int(item[0]))),
    }
    if len(reporter_table) != len(exact_paths):
        raise RuntimeError(f"Reporter registry/cache mismatch: {len(reporter_table)} != {len(exact_paths)}")
    if strict_census and observed != EXPECTED:
        raise RuntimeError(f"Frozen processed OPS census changed:\nobserved={observed}\nexpected={EXPECTED}")
    return {
        "status": "PASS",
        "dataset_root": str(root),
        "phase_root": str(phase_root),
        "exact_root": str(exact_root),
        "metadata_root": str(metadata_root),
        **observed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--allow-nonfrozen-census", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset_root, strict_census=not args.allow_nonfrozen_census), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
