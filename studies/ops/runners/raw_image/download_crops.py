#!/usr/bin/env python3
"""Download resumable 384-pixel OPS phase crops from public OME-Zarr assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import atomic_json, read_table, sha256_file


def public_s3(url: str) -> str:
    prefix = "https://ops-explorer-public.s3.amazonaws.com/"
    if not str(url).startswith(prefix):
        raise ValueError(f"Only the declared public OPS bucket is accepted: {url}")
    return "s3://ops-explorer-public/" + str(url)[len(prefix):]


class RemoteArrays:
    def __init__(self) -> None:
        try:
            import fsspec
            import zarr
        except ImportError as error:
            raise RuntimeError("Crop download requires the repository image extra") from error
        self.fsspec, self.zarr = fsspec, zarr
        self.fs = fsspec.filesystem("s3", anon=True)
        self.groups, self.arrays = {}, {}

    def array(self, url: str, path: str):
        key = (url, path)
        if key not in self.arrays:
            if url not in self.groups:
                self.groups[url] = self.zarr.open_group(self.fs.get_mapper(public_s3(url)), mode="r")
            self.arrays[key] = self.groups[url][path]
        return self.arrays[key]


def crop(array, x: float, y: float, size: int = 384) -> tuple[np.ndarray, np.ndarray]:
    if array.ndim < 2:
        raise RuntimeError(f"Image array has invalid shape {array.shape}")
    height, width = map(int, array.shape[-2:])
    cy, cx, half = int(round(float(y))), int(round(float(x))), size // 2
    y0, y1, x0, x1 = cy - half, cy - half + size, cx - half, cx - half + size
    sy0, sy1, sx0, sx1 = max(0, y0), min(height, y1), max(0, x0), min(width, x1)
    if sy0 >= sy1 or sx0 >= sx1:
        raise RuntimeError(f"Crop center ({x},{y}) lies outside image {array.shape}")
    leading = (0,) * (array.ndim - 2)
    source = np.asarray(array[(*leading, slice(sy0, sy1), slice(sx0, sx1))], dtype=np.float32)
    output = np.zeros((size, size), dtype=np.float32)
    dy0, dx0 = sy0 - y0, sx0 - x0
    output[dy0:dy0 + source.shape[0], dx0:dx0 + source.shape[1]] = source
    return output, np.asarray([dy0, dy0 + source.shape[0], dx0, dx0 + source.shape[1]], dtype=np.int16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.plan_root / "summary.json").read_text(encoding="utf-8"))
    cells_path = args.plan_root / summary.get("cells_file", "cells.parquet")
    if sha256_file(cells_path) != summary["cells_sha256"]:
        raise RuntimeError("Crop plan checksum mismatch")
    cells = read_table(cells_path)
    shard_ids = sorted(map(int, cells.shard_id.unique()))
    if args.max_shards:
        shard_ids = shard_ids[:args.max_shards]
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "n_cells": len(cells), "n_shards_selected": len(shard_ids)}, indent=2)); return
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "shards").mkdir(exist_ok=True)
    remote = RemoteArrays()
    completed = []
    for shard_id in shard_ids:
        final = args.output_root / "shards" / f"shard_{shard_id:06d}.h5"
        sidecar = final.with_suffix(".json")
        block = cells.loc[cells.shard_id.eq(shard_id)].sort_values("shard_offset", kind="stable")
        if final.is_file() and sidecar.is_file():
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            if record.get("sha256") == sha256_file(final):
                completed.append(shard_id); continue
        temporary = final.with_suffix(".h5.partial")
        if temporary.exists():
            temporary.unlink()
        with h5py.File(temporary, "w") as handle:
            phase = handle.create_dataset("phase", shape=(len(block), 384, 384), dtype="f4", chunks=(1, 384, 384), compression="lzf")
            rows = handle.create_dataset("phase_row_index", data=block.phase_row_index.to_numpy(np.int64))
            windows = handle.create_dataset("valid_window", shape=(len(block), 4), dtype="i2")
            for destination, row in enumerate(block.itertuples(index=False)):
                image, window = crop(remote.array(row.remote_zarr_https, row.level0_array_path), row.x_pheno, row.y_pheno)
                phase[destination] = image; windows[destination] = window
            handle.attrs["schema_version"] = "measurement-sufficiency-ops-crops-v1"
            handle.attrs["plan_sha256"] = summary["plan_sha256"]
            rows.attrs["sha256"] = hashlib.sha256(block.phase_row_index.to_numpy("<i8").tobytes()).hexdigest()
        os.replace(temporary, final)
        atomic_json(sidecar, {"shard_id": shard_id, "n_cells": len(block), "bytes": final.stat().st_size, "sha256": sha256_file(final)})
        completed.append(shard_id)
        print(json.dumps({"completed_shards": len(completed), "selected_shards": len(shard_ids), "shard_id": shard_id}), flush=True)
    complete = len(completed) == int(summary["n_shards"])
    atomic_json(args.output_root / "manifest.json", {
        "schema_version": "measurement-sufficiency-ops-crops-v1",
        "status": "COMPLETE" if complete else "PARTIAL",
        "plan_sha256": summary["plan_sha256"], "n_cells": len(cells),
        "n_shards": int(summary["n_shards"]), "completed_shards": completed,
    })


if __name__ == "__main__":
    main()
