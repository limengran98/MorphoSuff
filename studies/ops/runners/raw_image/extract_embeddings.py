#!/usr/bin/env python3
"""Extract resumable frozen DINOv2 or Cytoland features from OPS phase crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from common import atomic_json, read_table, sha256_file
from encoders import (
    cytoland_embedding,
    dinov2_embedding,
    load_cytoland,
    load_dinov2,
    normalize_cytoland,
    normalize_dinov2,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("dinov2", "cytoland"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True, help="Stage-B cells CSV or Parquet table")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def reporter_selected(value: str, selected: set[str]) -> bool:
    return bool(set(filter(None, re.split(r"[;|]", str(value)))) & selected)


def validate_inputs(args: argparse.Namespace, config: dict) -> pd.DataFrame:
    required = [args.config, args.cells, args.checkpoint, args.data_root / "manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing raw-image inputs: {missing}")
    if args.encoder == "dinov2" and not (args.repository / "hubconf.py").is_file():
        raise FileNotFoundError(f"Not a DINOv2 checkout: {args.repository}")
    if args.encoder == "cytoland" and not args.repository.is_dir():
        raise FileNotFoundError(f"Not a VisCy checkout: {args.repository}")
    if args.batch_size <= 0 or args.max_shards < 0:
        raise ValueError("batch-size must be positive and max-shards non-negative")
    columns = ["shard_id", "shard_offset", "phase_row_index", "reporter_slugs"]
    table = read_table(args.cells, columns=columns)
    selected_reporters = set(map(str, config["reporters"]))
    table = table.loc[
        table.reporter_slugs.map(lambda value: reporter_selected(value, selected_reporters))
    ].sort_values(["shard_id", "shard_offset"], kind="stable").reset_index(drop=True)
    if table.empty or table.phase_row_index.duplicated().any():
        raise RuntimeError("Selected Stage-B cells are empty or contain duplicate phase rows")
    return table


def main() -> None:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    selected = validate_inputs(args, config)
    selected["embedding_row"] = np.arange(len(selected), dtype=np.int64)
    summary = {
        "encoder": args.encoder,
        "reporters": sorted(map(str, config["reporters"])),
        "n_cells": len(selected),
        "n_shards": int(selected.shard_id.nunique()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "repository": str(args.repository.resolve()),
        "data_root": str(args.data_root.resolve()),
        "cells": str(args.cells.resolve()),
    }
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", **summary}, indent=2))
        return

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    encoder_config = config["encoder"]
    if args.encoder == "dinov2":
        model = load_dinov2(
            args.repository, args.checkpoint, str(encoder_config["entrypoint"]), device
        )
        transform = config["image_transform"]

        def encode(images: np.ndarray, windows: np.ndarray) -> "torch.Tensor":
            batch = normalize_dinov2(
                images,
                windows,
                low=float(transform["clip_low"]),
                high=float(transform["clip_high"]),
                output_size=int(transform["resize"]),
                device=device,
            )
            return dinov2_embedding(model, batch)

        dummy_shape = (1, 3, int(transform["resize"]), int(transform["resize"]))
    else:
        model = load_cytoland(
            args.repository, args.checkpoint, dict(encoder_config["model_config"]), device
        )

        def encode(images: np.ndarray, windows: np.ndarray) -> "torch.Tensor":
            return cytoland_embedding(
                model, normalize_cytoland(images, windows, device=device)
            )

        dummy_shape = (1, 1, 1, 384, 384)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    ):
        if args.encoder == "dinov2":
            dimension = int(dinov2_embedding(model, torch.zeros(dummy_shape, device=device)).shape[1])
        else:
            dimension = int(cytoland_embedding(model, torch.zeros(dummy_shape, device=device)).shape[1])
    expected = int(encoder_config["expected_embedding_dimension"])
    if dimension != expected:
        raise RuntimeError(f"Embedding dimension {dimension} != declared {expected}")

    source_manifest = json.loads((args.data_root / "manifest.json").read_text(encoding="utf-8"))
    fingerprint_payload = {
        **summary,
        "config_sha256": sha256_file(args.config),
        "source_plan_sha256": source_manifest.get("plan_sha256"),
        "embedding_dimension": dimension,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    args.output_root.mkdir(parents=True, exist_ok=True)
    binding_path = args.output_root / "extraction_binding.json"
    if binding_path.exists():
        previous = json.loads(binding_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise RuntimeError("Output root is bound to another extraction")
    else:
        atomic_json(binding_path, {"fingerprint": fingerprint, **fingerprint_payload})

    paths = {
        "embedding": args.output_root / "embedding.float16.npy",
        "phase_row": args.output_root / "phase_row_index.npy",
        "sort_order": args.output_root / "phase_row_sort_order.npy",
    }
    if paths["embedding"].exists():
        embedding = np.lib.format.open_memmap(paths["embedding"], mode="r+")
        if embedding.shape != (len(selected), dimension):
            raise RuntimeError(f"Existing embedding shape mismatch: {embedding.shape}")
    else:
        embedding = np.lib.format.open_memmap(
            paths["embedding"], mode="w+", dtype=np.float16, shape=(len(selected), dimension)
        )
        phase_rows = selected.phase_row_index.to_numpy(np.int64)
        np.save(paths["phase_row"], phase_rows, allow_pickle=False)
        np.save(paths["sort_order"], np.argsort(phase_rows), allow_pickle=False)
        embedding.flush()
    progress_path = args.output_root / "progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else {"fingerprint": fingerprint, "completed_shards": [], "started_at": time.time()}
    )
    if progress.get("fingerprint") != fingerprint:
        raise RuntimeError("Progress fingerprint differs from the current extraction")
    completed = set(map(int, progress.get("completed_shards", [])))
    shard_ids = selected.shard_id.drop_duplicates().astype(int).tolist()
    if args.max_shards:
        shard_ids = shard_ids[: args.max_shards]
    for shard_id in shard_ids:
        if shard_id in completed:
            continue
        block = selected.loc[selected.shard_id.eq(shard_id)]
        offsets = block.shard_offset.to_numpy(np.int64)
        destinations = block.embedding_row.to_numpy(np.int64)
        shard_path = args.data_root / "shards" / f"shard_{shard_id:06d}.h5"
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        with h5py.File(shard_path, "r") as handle:
            for start in range(0, len(offsets), args.batch_size):
                stop = min(start + args.batch_size, len(offsets))
                images = np.asarray(handle["phase"][offsets[start:stop]], dtype=np.float32)
                windows = np.asarray(handle["valid_window"][offsets[start:stop]], dtype=np.int16)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
                ):
                    values = encode(images, windows).float().cpu().numpy()
                embedding[destinations[start:stop]] = values.astype(np.float16)
        embedding.flush()
        completed.add(shard_id)
        progress.update(
            {
                "completed_shards": sorted(completed),
                "completed_cells": int(selected[selected.shard_id.isin(completed)].shape[0]),
                "total_cells": len(selected),
                "total_shards": int(selected.shard_id.nunique()),
                "updated_at": time.time(),
            }
        )
        atomic_json(progress_path, progress)
        print(json.dumps({"shard": shard_id, **progress}), flush=True)
    complete = len(completed) == int(selected.shard_id.nunique())
    atomic_json(
        args.output_root / "manifest.json",
        {
            "status": "COMPLETE" if complete else "PARTIAL_SMOKE",
            "fingerprint": fingerprint,
            "encoder": args.encoder,
            "encoder_config": encoder_config,
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "embedding_dimension": dimension,
            "n_cells": len(selected),
            "n_shards": int(selected.shard_id.nunique()),
            "reporters": summary["reporters"],
            "backbone_frozen": True,
        },
    )


if __name__ == "__main__":
    main()
