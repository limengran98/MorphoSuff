#!/usr/bin/env python3
"""Audit prepared OPS caches and canonical bundle cardinalities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

try:
    from .h5ad import decode
except ImportError:  # pragma: no cover
    from h5ad import decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--expected-reporters", type=int, default=52)
    parser.add_argument("--expected-assays", type=int, default=99)
    parser.add_argument("--expected-screens", type=int, default=73)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.prepared_root.expanduser().resolve()
    exact_paths = sorted((root / "exact" / "reporters").glob("all_cells_fluor_*.exact.h5"))
    if len(exact_paths) != args.expected_reporters:
        raise ValueError(f"reporter cache count {len(exact_paths)} != {args.expected_reporters}")
    screens: set[str] = set()
    assays = 0
    exact_pairs = 0
    for path in exact_paths:
        with h5py.File(path, "r") as cache:
            if cache.attrs.get("schema_version", "") != "ops-full-reporter-exact-v2":
                raise ValueError(f"cache schema mismatch: {path}")
            if len(np.unique(cache["phase_row_index"][:])) != len(cache["phase_row_index"]):
                raise ValueError(f"duplicate phase row within reporter: {path}")
            reporter_screens = set(decode(cache["metadata/screen_categories"][:]))
            screens.update(reporter_screens)
            assays += len(reporter_screens)
            exact_pairs += len(cache["phase_row_index"])
    if assays != args.expected_assays or len(screens) != args.expected_screens:
        raise ValueError(
            f"atlas topology reporters/assays/screens={len(exact_paths)}/{assays}/{len(screens)}; "
            f"expected {args.expected_reporters}/{args.expected_assays}/{args.expected_screens}"
        )
    phase_manifest = json.loads((root / "phase172" / "manifest.json").read_text(encoding="utf-8"))
    matrix = np.load(root / "phase172" / "phase172.npy", mmap_mode="r")
    if tuple(matrix.shape) != tuple(phase_manifest["shape"]) or matrix.shape[1] != 172:
        raise ValueError("phase172 matrix and manifest disagree")
    release = json.loads((root / "canonical" / "canonical_release.json").read_text(encoding="utf-8"))
    if release["n_reporters"] != args.expected_reporters or release["n_assays"] != args.expected_assays:
        raise ValueError("canonical release and exact-cache topology disagree")
    strict = pd.read_csv(root / "canonical" / "atlas" / "strict_screen_tasks.csv")
    report = {
        "status": "PASS",
        "reporters": len(exact_paths),
        "assays": assays,
        "screens": len(screens),
        "strict_screen_directions": int(len(strict)),
        "exact_assay_observations": int(exact_pairs),
        "phase_rows": int(matrix.shape[0]),
        "phase_features": int(matrix.shape[1]),
        "canonical_reporter_bundles": int(release["n_reporters"]),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
