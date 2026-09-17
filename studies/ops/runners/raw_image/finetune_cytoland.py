#!/usr/bin/env python3
"""Partially fine-tune public Cytoland VSCyto2D for one OPS reporter/fold.

The protocol freezes the stem and encoder stages 0--1 and updates stages 2--3
plus an endpoint head.  Source checkout, public checkpoint and every OPS asset
are explicit command-line inputs; no worker downloads data or weights.
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd

from common import (
    PhaseCache,
    aggregate_gene_profiles,
    assert_gene_split_integrity,
    atomic_json,
    atomic_npy,
    bootstrap_gene_metrics,
    load_reporter_data,
    matrix_metrics,
    partition_indices,
    profile_metrics,
    read_table,
    sha256_file,
    stable_seed,
    technical_core,
)
from encoders import load_cytoland, normalize_cytoland


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reporter", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-b-root", type=Path, required=True)
    parser.add_argument("--stage-b-plan", type=Path, required=True)
    parser.add_argument("--viscy-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--phase-cache", type=Path, required=True)
    parser.add_argument("--exact-cache-root", type=Path, required=True)
    parser.add_argument("--target-feature-dictionary", type=Path, required=True)
    parser.add_argument("--split", default="gene_holdout_main")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-cells", type=int, default=0)
    parser.add_argument("--max-validation-cells", type=int, default=0)
    parser.add_argument("--max-test-cells", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int, default=300)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def choose_subset(indices: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if maximum <= 0 or len(indices) <= maximum:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=maximum, replace=False)).astype(np.int64)


def standardize_targets(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(y_train, axis=0, dtype=np.float64).astype(np.float32)
    scale = np.std(y_train, axis=0, dtype=np.float64).astype(np.float32)
    keep = np.flatnonzero(np.isfinite(scale) & (scale > 1e-8))
    if not len(keep):
        raise RuntimeError("All reporter endpoints are constant in training")
    return keep, mean[keep], scale[keep]


@dataclass(frozen=True)
class ImageLocations:
    shard_id: np.ndarray
    shard_offset: np.ndarray


def stage_b_locations(phase_rows: np.ndarray, plan_path: Path) -> ImageLocations:
    table = read_table(
        plan_path, columns=["phase_row_index", "shard_id", "shard_offset"]
    ).sort_values("phase_row_index", kind="stable")
    if table.phase_row_index.duplicated().any():
        raise RuntimeError("Stage-B plan contains duplicate phase rows")
    stored = table.phase_row_index.to_numpy(np.int64)
    query = np.asarray(phase_rows, dtype=np.int64)
    where = np.searchsorted(stored, query)
    safe = np.minimum(where, max(len(stored) - 1, 0))
    valid = (where < len(stored)) & (stored[safe] == query)
    if not np.all(valid):
        raise RuntimeError(
            f"Stage-B crops lack {int((~valid).sum())}/{len(query)} exact cells"
        )
    return ImageLocations(
        table.shard_id.to_numpy(np.int32)[where],
        table.shard_offset.to_numpy(np.int32)[where],
    )


class ShardReader:
    def __init__(self, root: Path, maximum_open: int = 12) -> None:
        self.root = root
        self.maximum_open = maximum_open
        self.handles: collections.OrderedDict[int, h5py.File] = collections.OrderedDict()

    def handle(self, shard_id: int) -> h5py.File:
        if shard_id in self.handles:
            self.handles.move_to_end(shard_id)
            return self.handles[shard_id]
        path = self.root / "shards" / f"shard_{shard_id:06d}.h5"
        if not path.is_file():
            raise FileNotFoundError(path)
        handle = h5py.File(path, "r")
        self.handles[shard_id] = handle
        if len(self.handles) > self.maximum_open:
            _, oldest = self.handles.popitem(last=False)
            oldest.close()
        return handle

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()
        self.handles.clear()


def shard_batches(
    indices: np.ndarray,
    locations: ImageLocations,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> list[np.ndarray]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    rng = np.random.default_rng(seed)
    grouped: dict[int, list[int]] = {}
    for index in np.asarray(indices, dtype=np.int64):
        grouped.setdefault(int(locations.shard_id[index]), []).append(int(index))
    groups = list(grouped.values())
    if shuffle:
        rng.shuffle(groups)
    batches: list[np.ndarray] = []
    for members in groups:
        values = np.asarray(members, dtype=np.int64)
        if shuffle:
            rng.shuffle(values)
        batches.extend(values[start : start + batch_size] for start in range(0, len(values), batch_size))
    return batches


def phase_batch(indices: np.ndarray, locations: ImageLocations, reader: ShardReader, device):
    shard_ids = np.unique(locations.shard_id[indices])
    if len(shard_ids) != 1:
        raise RuntimeError("A batch crossed Stage-B shard boundaries")
    offsets = np.asarray(locations.shard_offset[indices], dtype=np.int64)
    order = np.argsort(offsets, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    handle = reader.handle(int(shard_ids[0]))
    images = np.asarray(handle["phase"][offsets[order]], dtype=np.float32)[inverse]
    windows = np.asarray(handle["valid_window"][offsets[order]], dtype=np.int16)[inverse]
    return normalize_cytoland(images, windows, device=device)


def endpoint_model_from_encoder(config: dict, n_outputs: int, encoder, device):
    """Attach the registered trainable stages and head to an encoder.

    Keeping this seam separate from checkpoint loading permits a true
    synthetic forward/backward contract test without impersonating public
    Cytoland weights.
    """
    import torch
    from torch import nn

    training = config["training"]
    trainable = list(map(int, training["trainable_stage_indices"]))
    if trainable != [2, 3]:
        raise RuntimeError(f"Registered protocol requires stages [2,3], got {trainable}")
    encoder.stem.requires_grad_(False)
    for index, stage in enumerate(encoder.stages):
        stage.requires_grad_(index in trainable)

    class EndpointModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = encoder
            widths = [
                int(config["encoder"]["expected_embedding_dimension"]),
                *map(int, training["head_hidden"]),
                n_outputs,
            ]
            layers: list[nn.Module] = []
            for left, right in zip(widths[:-2], widths[1:-1]):
                layers.extend((nn.Linear(left, right), nn.GELU(), nn.Dropout(0.10)))
            layers.append(nn.Linear(widths[-2], widths[-1]))
            self.head = nn.Sequential(*layers)

        def forward(self, image):
            feature_maps, mask = self.encoder(image, mask_ratio=0.0)
            if mask is not None or len(feature_maps) != 4:
                raise RuntimeError("Cytoland did not return four unmasked feature maps")
            pooled = [values.float().mean(dim=(-2, -1)) for values in feature_maps]
            return self.head(torch.cat(pooled, dim=1))

    model = EndpointModel().to(device)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    active = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, {"frozen_parameters": frozen, "active_parameters": active, "total_parameters": frozen + active}


def build_model(config: dict, n_outputs: int, repository: Path, checkpoint: Path, device):
    encoder = load_cytoland(
        repository, checkpoint, dict(config["encoder"]["model_config"]), device
    )
    return endpoint_model_from_encoder(config, n_outputs, encoder, device)


def epoch_pass(
    model,
    batches: Iterable[np.ndarray],
    y_scaled: np.ndarray,
    locations: ImageLocations,
    reader: ShardReader,
    device,
    optimizer=None,
):
    import torch

    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    predictions: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for indices in batches:
        image = phase_batch(indices, locations, reader, device)
        target = torch.as_tensor(y_scaled[indices], dtype=torch.float32, device=device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            prediction = model(image)
            loss = torch.mean((prediction.float() - target) ** 2)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
        total += float(loss.detach().cpu()) * len(indices)
        count += len(indices)
        if not training:
            positions.append(np.asarray(indices, dtype=np.int64))
            predictions.append(prediction.float().detach().cpu().numpy())
    if not count:
        raise RuntimeError("Empty image loader")
    if training:
        return total / count, None, None
    return total / count, np.concatenate(positions), np.concatenate(predictions)


def trainable_state(model) -> dict:
    return {
        "stage_2": {k: v.detach().cpu().clone() for k, v in model.encoder.stages[2].state_dict().items()},
        "stage_3": {k: v.detach().cpu().clone() for k, v in model.encoder.stages[3].state_dict().items()},
        "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
    }


def main() -> None:
    args = arguments()
    required = (
        args.config,
        args.stage_b_root / "manifest.json",
        args.stage_b_plan,
        args.checkpoint,
        args.phase_cache / "manifest.json",
        args.exact_cache_root / f"all_cells_fluor_{args.reporter}.exact.h5",
        args.target_feature_dictionary,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not args.viscy_repo.is_dir():
        raise FileNotFoundError(
            f"Missing fine-tuning inputs: {missing}; viscy_repo={args.viscy_repo}"
        )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.reporter not in config["reporters"]:
        raise ValueError(f"Reporter {args.reporter!r} is outside the frozen panel")
    phase = PhaseCache.open(args.phase_cache, args.split)
    n_folds = int(phase.manifest.get("n_folds", 5))
    if args.fold < 0 or args.fold >= n_folds:
        raise ValueError(f"fold must be in [0,{n_folds - 1}]")
    names = technical_core(args.target_feature_dictionary, args.reporter)
    data = load_reporter_data(required[5], args.reporter, names)
    metadata = {name: np.asarray(values[data.phase_rows]) for name, values in phase.metadata.items()}
    folds = np.asarray(phase.folds[data.phase_rows], dtype=np.int16)
    train, validation, test, validation_fold = partition_indices(folds, args.fold, n_folds)
    assert_gene_split_integrity(train, validation, test, data.is_control, metadata["gene_code"], data.phase_rows)
    locations = stage_b_locations(data.phase_rows, args.stage_b_plan)
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS", "reporter": args.reporter, "fold": args.fold,
            "validation_fold": validation_fold, "n_train": len(train),
            "n_validation": len(validation), "n_test": len(test),
            "n_endpoints": data.y.shape[1], "n_stage_b_shards": int(len(np.unique(locations.shard_id))),
            "checkpoint_sha256": sha256_file(args.checkpoint),
        }, indent=2))
        return

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    training = config["training"]
    epochs = int(args.epochs or training["epochs"])
    patience = int(args.patience or training["patience"])
    batch_size = int(args.batch_size or training["batch_size"])
    if min(epochs, patience, batch_size) <= 0:
        raise ValueError("epochs, patience and batch-size must be positive")
    seed = stable_seed("cytoland_partial_finetune", args.reporter, args.fold)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats()
    train_full = train
    train = choose_subset(train, args.max_train_cells, seed + 11)
    validation = choose_subset(validation, args.max_validation_cells, seed + 13)
    test = choose_subset(test, args.max_test_cells, seed + 17)
    keep, y_mean, y_scale = standardize_targets(data.y[train])
    y_scaled = (np.asarray(data.y[:, keep], dtype=np.float32) - y_mean) / y_scale
    model, counts = build_model(config, len(keep), args.viscy_repo, args.checkpoint, device)
    backbone = [p for index in (2, 3) for p in model.encoder.stages[index].parameters() if p.requires_grad]
    head = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": float(training["backbone_learning_rate"]), "weight_decay": float(training["backbone_weight_decay"])},
        {"params": head, "lr": float(training["head_learning_rate"]), "weight_decay": float(training["head_weight_decay"])},
    ])
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    reader = ShardReader(args.stage_b_root)
    best_loss, best_epoch, stale, best_state = math.inf, 0, 0, None
    history: list[dict] = []
    started = time.time()
    try:
        for epoch in range(epochs):
            warmup = int(training["warmup_epochs"])
            multiplier = (epoch + 1) / warmup if epoch < warmup else 0.10 + 0.90 * 0.5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup)))
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * multiplier
            train_mse, _, _ = epoch_pass(model, shard_batches(train, locations, batch_size, seed + epoch * 23, True), y_scaled, locations, reader, device, optimizer)
            validation_mse, _, _ = epoch_pass(model, shard_batches(validation, locations, batch_size, seed + 19, False), y_scaled, locations, reader, device)
            if validation_mse < best_loss - 1e-5:
                best_loss, best_epoch, stale, best_state = validation_mse, epoch + 1, 0, trainable_state(model)
            else:
                stale += 1
            row = {"epoch": epoch + 1, "train_mse": train_mse, "validation_mse": validation_mse, "best_validation_mse": best_loss}
            history.append(row); print(json.dumps(row), flush=True)
            if stale >= patience:
                break
        if best_state is None:
            raise RuntimeError("No valid fine-tuning checkpoint was produced")
        model.encoder.stages[2].load_state_dict(best_state["stage_2"])
        model.encoder.stages[3].load_state_dict(best_state["stage_3"])
        model.head.load_state_dict(best_state["head"])
        test_mse, positions, predictions = epoch_pass(model, shard_batches(test, locations, batch_size, seed + 29, False), y_scaled, locations, reader, device)
        assert positions is not None and predictions is not None
        ordered = np.empty((len(test), predictions.shape[1]), dtype=np.float32)
        lookup = {int(value): index for index, value in enumerate(test.tolist())}
        for position, prediction in zip(positions, predictions):
            ordered[lookup[int(position)]] = prediction
        target = ~data.is_control[test]
        raw_truth = data.y[test][:, keep]
        raw_prediction = ordered * y_scale + y_mean
        feature_names = data.target_feature_names[keep]
        cell_summary, cell_table = matrix_metrics(raw_truth[target], raw_prediction[target], np.broadcast_to(y_mean, raw_truth[target].shape), feature_names)
        profiles = aggregate_gene_profiles(y_scaled[test], ordered, metadata["screen_code"][test], metadata["gene_code"][test], data.is_control[test])
        gene_summary, gene_table = profile_metrics(profiles, feature_names)
        if args.bootstrap_draws:
            gene_summary.update(bootstrap_gene_metrics(profiles, args.bootstrap_draws, seed + 71))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = args.output_dir / "best_trainable_state.pt"
        torch.save({"public_checkpoint_sha256": sha256_file(args.checkpoint), "reporter": args.reporter, "fold": args.fold, "y_keep": keep, "y_mean": y_mean, "y_scale": y_scale, "state": best_state}, checkpoint_path)
        atomic_npy(args.output_dir / "gene_truth_control_relative_scaled.npy", profiles.truth)
        atomic_npy(args.output_dir / "gene_prediction_control_relative_scaled.npy", profiles.prediction)
        cell_table.to_csv(args.output_dir / "cell_endpoint_metrics.csv", index=False)
        gene_table.to_csv(args.output_dir / "gene_endpoint_metrics.csv", index=False)
        result = {
            "status": "PASS", "completed": True,
            "representation": "raw_phase_cytoland_partial_finetune",
            "reporter_slug": args.reporter, "fold": args.fold,
            "validation_fold": validation_fold, "split": args.split,
            "n_target_features": len(feature_names), "n_train": len(train),
            "n_train_full": len(train_full), "n_validation": len(validation), "n_test": len(test),
            "parameter_counts": counts, "cell_metrics": cell_summary,
            "gene_metrics": gene_summary, "test_standardized_mse": test_mse,
            "best_epoch": best_epoch, "training_protocol": {**training, "history": history},
            "public_checkpoint_sha256": sha256_file(args.checkpoint),
            "peak_gpu_memory_gib": float(torch.cuda.max_memory_allocated() / 1024**3) if device.type == "cuda" else 0.0,
            "runtime_seconds": time.time() - started,
        }
        atomic_json(args.output_dir / "training_result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        reader.close(); del model; gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
