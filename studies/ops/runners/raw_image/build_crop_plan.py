#!/usr/bin/env python3
"""Build deterministic 384-pixel raw-phase crop shards from OPS image locations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import atomic_json, read_table, sha256_file


REQUIRED = {
    "phase_row_index", "reporter_slugs", "remote_zarr_https",
    "level0_array_path", "x_pheno", "y_pheno",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=4096)
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.shard_size <= 0:
        raise ValueError("shard-size must be positive")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    reporters = set(map(str, config["reporters"]))
    table = read_table(args.locations)
    missing = REQUIRED - set(table.columns)
    if missing:
        raise RuntimeError(f"Image-location table lacks columns: {sorted(missing)}")
    selected = table.reporter_slugs.astype(str).map(
        lambda value: bool(set(filter(None, re.split(r"[;|]", value))) & reporters)
    )
    table = table.loc[selected, sorted(REQUIRED)].copy()
    if table.empty:
        raise RuntimeError("No rows match the frozen reporter panel")
    location_columns = ["remote_zarr_https", "level0_array_path", "x_pheno", "y_pheno"]
    if table.groupby("phase_row_index")[location_columns].nunique(dropna=False).max().max() > 1:
        raise RuntimeError("One phase row maps to conflicting image locations")
    table = table.sort_values("phase_row_index", kind="stable").drop_duplicates("phase_row_index")
    table["source_chunk_y"] = np.floor(table.y_pheno.astype(float) / 512).astype(int)
    table["source_chunk_x"] = np.floor(table.x_pheno.astype(float) / 512).astype(int)
    table = table.sort_values(
        ["remote_zarr_https", "level0_array_path", "source_chunk_y", "source_chunk_x", "phase_row_index"],
        kind="stable",
    ).reset_index(drop=True)
    table["shard_id"] = np.arange(len(table), dtype=np.int64) // args.shard_size
    table["shard_offset"] = np.arange(len(table), dtype=np.int64) % args.shard_size
    summary = {
        "schema_version": "measurement-sufficiency-ops-crop-plan-v1",
        "n_cells": len(table), "n_shards": int(table.shard_id.nunique()),
        "shard_size": args.shard_size, "crop_size": 384,
        "reporters": sorted(reporters), "locations_sha256": sha256_file(args.locations),
    }
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", **summary}, indent=2)); return
    args.output_root.mkdir(parents=True, exist_ok=True)
    cells = args.output_root / f"cells.{args.format}"
    if args.format == "parquet":
        table.to_parquet(cells, index=False)
    else:
        table.to_csv(cells, index=False)
    summary["cells_file"] = cells.name
    summary["cells_sha256"] = sha256_file(cells)
    summary["plan_sha256"] = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode("utf-8")
    ).hexdigest()
    atomic_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
