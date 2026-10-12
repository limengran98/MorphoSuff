#!/usr/bin/env python3
"""Formal same-cell falsification campaign for the frozen OPS 172D -> Y task.

This runner is intentionally narrower than the full benchmark.  It asks one
question under the exact frozen gene-holdout contract: does phase-to-reporter
prediction require the *identity* of the paired cell, beyond KO/screen effects
and the obvious size/shape variables?

Each job owns one reporter, outer gene fold, model, and analysis arm.  The six
arms are:

* ``exact_pair``: the unchanged exact same-cell mapping;
* ``gene_screen_deranged``: labels are permuted within gene x screen;
* ``covariate_matched_deranged``: the same permutation is constrained by
  area, field cell-density, and eccentricity bins;
* ``crossfit_gene_screen_residual``: predict a leave-block-out residual after
  removing the gene x screen mean (a conditional association, not deployment
  performance);
* ``size_shape_only``: use only the 16 cell size/shape features;
* ``remove_size_shape``: remove those 16 features from the 172D input.

The stricter gene x screen x field shuffle requested during design is audited
but is deliberately not used as the primary null: for LAMP1 79% of such
strata have a single cell and cannot be permuted.  The matched null preserves
gene/screen and constrains field density, while remaining a real derangement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import rankdata

from ops_reporter_specialist_lib import (
    PhaseCache,
    aggregate_gene_profiles,
    assert_split_integrity,
    atomic_json,
    atomic_npy,
    bootstrap_gene_profile_metrics,
    fit_gbdt,
    fit_catboost_multirmse,
    fit_preprocessing,
    fit_ridge,
    hash_arrays,
    load_reporter_data,
    matrix_metrics,
    partition_indices,
    profile_metrics,
)


SCHEMA = "ops-same-cell-falsification-v1"
SEED = 20260804
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PANEL12 = (
    "ps6",
    "nucleoli_npm1",
    "er_golgi_cop-ii_sec23a",
    "lysosome_lamp1",
    "5xupre",
    "mitochondria_tomm20",
    "actin_filament_fastact_spy555_live_cell_dye",
    "early_endosome_eea1",
    "plasma_membrane_wga",
    "peroxisome_peroxi_spy650_live_cell_dye",
    "nuclei_hoechst",
    "stress_granule_g3bp1",
)
CATBOOST_ANCHORS = (
    "5xupre",
    "lysosome_lamp1",
    "actin_filament_fastact_spy555_live_cell_dye",
)
ARMS = (
    "exact_pair",
    "gene_screen_deranged",
    "covariate_matched_deranged",
    "crossfit_gene_screen_residual",
    "size_shape_only",
    "remove_size_shape",
)
DEFAULT_PHASE_CACHE = PROJECT_ROOT / "data/processed/ops_phase172_indexed"
DEFAULT_EXACT_ROOT = PROJECT_ROOT / "data/processed/ops_full_reporter_exact/reporters"
DEFAULT_TARGET_TABLE = PROJECT_ROOT / "results/ops_phase0_asset_audit/reporter_targets.csv"
DEFAULT_TARGET_DICTIONARY = PROJECT_ROOT / "results/ops_phase0_asset_audit/target_feature_dictionary.csv"


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(*tokens: object) -> int:
    value = "|".join(str(token) for token in tokens).encode("utf-8")
    return (SEED + int.from_bytes(hashlib.sha256(value).digest()[:8], "little")) % (2**31 - 1)


def _parse_choice(value: str, allowed: Iterable[str], label: str) -> list[str]:
    allowed_tuple = tuple(allowed)
    if value.strip().lower() == "all":
        return list(allowed_tuple)
    result = [token.strip() for token in value.split(",") if token.strip()]
    invalid = sorted(set(result) - set(allowed_tuple))
    if invalid or not result:
        raise ValueError(f"Invalid {label}: {invalid}; allowed={allowed_tuple}")
    return list(dict.fromkeys(result))


def _parse_folds(value: str, n_folds: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(n_folds))
    result = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not result or min(result) < 0 or max(result) >= n_folds:
        raise ValueError(f"Folds must be in [0, {n_folds - 1}]")
    return sorted(set(result))


def _technical_core(path: Path, target_table: pd.DataFrame) -> dict[str, list[str]]:
    table = pd.read_csv(path)
    if table.duplicated(["reporter_slug", "target_feature_name"]).any():
        raise RuntimeError("Duplicate reporter/target rows in frozen feature dictionary")
    selected = table["selected_in_technical_core"].astype(str).str.casefold().isin(
        {"true", "1", "yes"}
    )
    result = {
        str(slug): frame["target_feature_name"].astype(str).tolist()
        for slug, frame in table.loc[selected].groupby("reporter_slug", sort=False)
    }
    expected = set(target_table["reporter_slug"].astype(str))
    if set(result) != expected:
        raise RuntimeError("Frozen technical-core dictionary does not cover exactly 52 reporters")
    return result


def _group_inverse(*arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.column_stack([np.asarray(values) for values in arrays])
    _, inverse, counts = np.unique(matrix, axis=0, return_inverse=True, return_counts=True)
    return inverse.astype(np.int64), counts.astype(np.int64)


def _cyclic_derangement(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        return indices.copy()
    # Rotating the canonical row order, rather than a pre-shuffled order,
    # guarantees a genuine derangement: a random permutation followed by a
    # rotation can accidentally return an element to its original position.
    shift = int(rng.integers(1, len(indices)))
    return np.roll(indices, shift)


def _within_group_permutation(
    gene_codes: np.ndarray,
    screen_codes: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return donor row for a gene x screen derangement, without fixed points."""

    rng = np.random.default_rng(seed)
    inverse, counts = _group_inverse(gene_codes, screen_codes)
    donor = np.arange(len(inverse), dtype=np.int64)
    for group in range(len(counts)):
        rows = np.flatnonzero(inverse == group)
        if len(rows) > 1:
            donor[rows] = _cyclic_derangement(rows, rng)
    moved = donor != np.arange(len(donor))
    return donor, {
        "stratum": "gene_x_screen",
        "n_strata": int(len(counts)),
        "median_stratum_size": float(np.median(counts)),
        "singleton_cell_fraction": float(np.mean(counts[inverse] == 1)),
        "moved_cell_fraction": float(np.mean(moved)),
        "fixed_cells": int((~moved).sum()),
    }


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    if len(values) <= 1 or np.allclose(values, values[0]):
        return np.zeros(len(values), dtype=np.int64)
    ranks = rankdata(values, method="average") - 1.0
    return np.minimum((ranks * n_bins / len(values)).astype(np.int64), n_bins - 1)


def _covariate_matched_permutation(
    gene_codes: np.ndarray,
    screen_codes: np.ndarray,
    covariates: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Near-neighbour, bijective label derangement inside gene x screen.

    Each non-singleton stratum is greedily paired using short edges in the
    standardized ``area, field-density, eccentricity`` space, and pairs are
    swapped.  An odd final cell is included in a short three-cycle.  This is
    a true label derangement and a substantially stronger control than bins
    with an unconstrained group-level fallback.
    """

    rng = np.random.default_rng(seed)
    inverse, counts = _group_inverse(gene_codes, screen_codes)
    donor = np.arange(len(inverse), dtype=np.int64)
    match_distance = np.full(len(inverse), np.nan, dtype=np.float64)
    pair_count = 0
    triple_count = 0
    for group in range(len(counts)):
        rows = np.flatnonzero(inverse == group)
        if len(rows) < 2:
            continue
        cov = np.asarray(covariates[rows], dtype=np.float64)
        neighbours = min(len(rows), 12)
        distance, neighbour = cKDTree(cov).query(cov, k=neighbours)
        if neighbours == 2:
            distance = distance[:, None]
            neighbour = neighbour[:, None]
        edges: list[tuple[float, int, int]] = []
        for local in range(len(rows)):
            for rank in range(1, neighbours):
                other = int(neighbour[local, rank])
                if other != local:
                    edges.append((float(distance[local, rank]), local, other))
        edges.sort(key=lambda value: value[0])
        available = np.ones(len(rows), dtype=bool)
        pairs: list[tuple[int, int]] = []
        for dist, left, right in edges:
            if available[left] and available[right]:
                donor[rows[left]] = rows[right]
                donor[rows[right]] = rows[left]
                available[left] = False
                available[right] = False
                match_distance[[rows[left], rows[right]]] = dist
                pairs.append((left, right))
                pair_count += 1
        remaining = np.flatnonzero(available)
        while len(remaining) >= 2:
            left, right = int(remaining[0]), int(remaining[1])
            dist = float(np.linalg.norm(cov[left] - cov[right]))
            donor[rows[left]] = rows[right]
            donor[rows[right]] = rows[left]
            match_distance[[rows[left], rows[right]]] = dist
            available[left] = False
            available[right] = False
            pairs.append((left, right))
            pair_count += 1
            remaining = np.flatnonzero(available)
        if len(remaining) == 1:
            lone = int(remaining[0])
            if not pairs:
                raise RuntimeError("non-singleton matched stratum did not produce a pair")
            left, right = min(
                pairs,
                key=lambda pair: float(np.min(np.linalg.norm(cov[lone] - cov[list(pair)], axis=1))),
            )
            # Split a prior swap into an orientation-randomized three-cycle.
            if bool(rng.integers(0, 2)):
                left, right = right, left
            donor[rows[left]] = rows[right]
            donor[rows[right]] = rows[lone]
            donor[rows[lone]] = rows[left]
            cycle_distance = np.maximum(
                np.linalg.norm(cov[left] - cov[right]),
                np.maximum(np.linalg.norm(cov[right] - cov[lone]), np.linalg.norm(cov[lone] - cov[left])),
            )
            match_distance[[rows[left], rows[right], rows[lone]]] = cycle_distance
            triple_count += 1
    moved = donor != np.arange(len(donor))
    delta = np.abs(covariates - covariates[donor])
    matched = moved
    return donor, {
        "stratum": "gene_x_screen; nearest-neighbour area_x_field_density_x_eccentricity matching",
        "n_strata": int(len(counts)),
        "moved_cell_fraction": float(np.mean(moved)),
        "fixed_cells": int((~moved).sum()),
        "matching": {
            "nearest_neighbour_pairs": int(pair_count),
            "three_cycles_for_odd_strata": int(triple_count),
            "unshufflable_singleton": int(np.sum(counts[inverse] == 1)),
            "median_standardized_pair_distance": float(np.nanmedian(match_distance[moved])),
            "p95_standardized_pair_distance": float(np.nanquantile(match_distance[moved], 0.95)),
        },
        "matched_median_abs_z_delta": np.median(delta[matched], axis=0).astype(float).tolist()
        if np.any(matched)
        else [math.nan, math.nan, math.nan],
        "matched_p95_abs_z_delta": np.quantile(delta[matched], 0.95, axis=0).astype(float).tolist()
        if np.any(matched)
        else [math.nan, math.nan, math.nan],
    }


def _crossfit_group_residual(
    values: np.ndarray,
    gene_codes: np.ndarray,
    screen_codes: np.ndarray,
    seed: int,
    n_splits: int = 5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove a gene x screen mean without allowing a cell to centre itself."""

    rng = np.random.default_rng(seed)
    inverse, counts = _group_inverse(gene_codes, screen_codes)
    residual = np.empty_like(values, dtype=np.float32)
    fallback_cells = 0
    for group in range(len(counts)):
        rows = np.flatnonzero(inverse == group)
        if len(rows) < 2:
            residual[rows] = 0.0
            fallback_cells += len(rows)
            continue
        order = rows[rng.permutation(len(rows))]
        blocks = [block for block in np.array_split(order, min(n_splits, len(order))) if len(block)]
        for block in blocks:
            keep = np.setdiff1d(rows, block, assume_unique=True)
            centre = np.mean(values[keep], axis=0, dtype=np.float64).astype(np.float32)
            residual[block] = values[block] - centre
    return residual, {
        "centre": "cross_fitted_gene_x_screen_mean",
        "n_strata": int(len(counts)),
        "median_stratum_size": float(np.median(counts)),
        "singleton_fallback_cells": int(fallback_cells),
        "n_crossfit_splits": int(n_splits),
        "uses_heldout_peer_labels": True,
        "deployment_metric": False,
    }


def _field_density_and_eccentricity(
    x: np.ndarray,
    screen_codes: np.ndarray,
    well_codes: np.ndarray,
    tile_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    area = np.asarray(x[:, 156], dtype=np.float64)
    major = np.maximum(np.asarray(x[:, 158], dtype=np.float64), 1e-8)
    minor = np.clip(np.asarray(x[:, 159], dtype=np.float64), 0.0, major)
    eccentricity = np.sqrt(np.maximum(0.0, 1.0 - (minor / major) ** 2))
    field_inverse, field_counts = _group_inverse(screen_codes, well_codes, tile_codes)
    density = field_counts[field_inverse].astype(np.float64)
    cov = np.column_stack((np.log1p(np.maximum(area, 0.0)), np.log1p(density), eccentricity))
    median = np.nanmedian(cov, axis=0)
    scale = np.nanstd(cov, axis=0)
    scale[~np.isfinite(scale) | (scale <= 1e-8)] = 1.0
    cov = np.where(np.isfinite(cov), cov, median)
    cov = (cov - median) / scale
    strict_inverse, strict_counts = _group_inverse(
        screen_codes, well_codes, tile_codes
    )
    del strict_inverse
    return area.astype(np.float32), density.astype(np.float32), eccentricity.astype(np.float32), {
        "density_definition": "exact-reporter cells per screen_x_well_x_tile field",
        "n_fields": int(len(field_counts)),
        "median_field_cells": float(np.median(field_counts)),
        "field_p05_p95_cells": np.quantile(field_counts, [0.05, 0.95]).astype(float).tolist(),
        "field_singleton_fraction": float(np.mean(field_counts == 1)),
        "strict_field_group_count": int(len(strict_counts)),
    }, cov.astype(np.float32)


def _strict_joint_audit(
    gene_codes: np.ndarray,
    screen_codes: np.ndarray,
    well_codes: np.ndarray,
    tile_codes: np.ndarray,
) -> dict[str, Any]:
    inverse, counts = _group_inverse(gene_codes, screen_codes, well_codes, tile_codes)
    return {
        "strict_gene_screen_field_group_count": int(len(counts)),
        "strict_gene_screen_field_median_size": float(np.median(counts)),
        "strict_gene_screen_field_singleton_cell_fraction": float(np.mean(counts[inverse] == 1)),
        "strict_gene_screen_field_permutable_cell_fraction": float(np.mean(counts[inverse] > 1)),
    }


def _fit_mlp_gpu_staged(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    *,
    hidden: list[int],
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    x_train_t = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    x_val_t = torch.as_tensor(x_validation, dtype=torch.float32, device=device)
    y_val_t = torch.as_tensor(y_validation, dtype=torch.float32, device=device)
    x_test_t = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    layers: list[nn.Module] = []
    current = x_train.shape[1]
    for width in hidden:
        layers.extend((nn.Linear(current, width), nn.GELU(), nn.LayerNorm(width)))
        current = width
    layers.append(nn.Linear(current, y_train.shape[1]))
    model = nn.Sequential(*layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    no_improvement = 0
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        order = torch.randperm(len(x_train_t), device=device)
        running = 0.0
        n_seen = 0
        for start in range(0, len(order), batch_size):
            selected = order[start : start + batch_size]
            prediction = model(x_train_t[selected])
            loss = torch.mean((prediction - y_train_t[selected]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(selected)
            n_seen += len(selected)
        model.eval()
        with torch.no_grad():
            val_mse = float(torch.mean((model(x_val_t) - y_val_t) ** 2).detach().cpu())
        history.append({
            "epoch": float(epoch),
            "train_mse": running / max(n_seen, 1),
            "validation_mse": val_mse,
            "seconds": time.time() - epoch_started,
        })
        print(
            f"epoch {epoch}: train_mse={history[-1]['train_mse']:.6f}; "
            f"validation_mse={val_mse:.6f}; best={best_loss if math.isfinite(best_loss) else val_mse:.6f}",
            flush=True,
        )
        if val_mse < best_loss - 1e-6:
            best_loss = val_mse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improvement = 0
        else:
            no_improvement += 1
        if no_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("MLP did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    pieces: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_test_t), batch_size * 2):
            pieces.append(model(x_test_t[start : start + batch_size * 2]).detach().cpu().numpy())
    prediction = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    del x_train_t, y_train_t, x_val_t, y_val_t, x_test_t, model, optimizer
    torch.cuda.empty_cache()
    return prediction, {
        "backend": "pytorch_gpu_staged",
        "device": "cuda",
        "cpu_fallback_allowed": False,
        "hidden": hidden,
        "batch_size": int(batch_size),
        "epochs_requested": int(epochs),
        "epochs_completed": int(len(history)),
        "patience": int(patience),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "best_validation_mse": float(best_loss),
        "history": history,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "runtime_seconds": time.time() - started,
        "torch_version": torch.__version__,
    }


def _normal_or_residual_data(
    arm: str,
    y: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    gene_codes: np.ndarray,
    screen_codes: np.ndarray,
    covariates: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], bool]:
    if arm == "gene_screen_deranged":
        train_donor, train_audit = _within_group_permutation(gene_codes[train], screen_codes[train], seed + 1)
        val_donor, val_audit = _within_group_permutation(gene_codes[validation], screen_codes[validation], seed + 2)
        # ``*_donor`` is indexed locally to its partition.  Map it back to
        # the reporter-row coordinate system before fetching labels; using it
        # directly would silently leak rows from the start of the full table.
        return y[train[train_donor]], y[validation[val_donor]], y[test], {
            "train": train_audit,
            "validation": val_audit,
            "test_label_policy": "true_exact_pairs",
        }, False
    if arm == "covariate_matched_deranged":
        train_donor, train_audit = _covariate_matched_permutation(
            gene_codes[train], screen_codes[train], covariates[train], seed + 1
        )
        val_donor, val_audit = _covariate_matched_permutation(
            gene_codes[validation], screen_codes[validation], covariates[validation], seed + 2
        )
        return y[train[train_donor]], y[validation[val_donor]], y[test], {
            "train": train_audit,
            "validation": val_audit,
            "test_label_policy": "true_exact_pairs",
        }, False
    if arm == "crossfit_gene_screen_residual":
        train_y, train_audit = _crossfit_group_residual(y[train], gene_codes[train], screen_codes[train], seed + 1)
        val_y, val_audit = _crossfit_group_residual(y[validation], gene_codes[validation], screen_codes[validation], seed + 2)
        test_y, test_audit = _crossfit_group_residual(y[test], gene_codes[test], screen_codes[test], seed + 3)
        return train_y, val_y, test_y, {
            "train": train_audit,
            "validation": val_audit,
            "test": test_audit,
        }, True
    return y[train], y[validation], y[test], {"label_policy": "true_exact_pairs"}, False


def _feature_indices(arm: str) -> np.ndarray:
    if arm == "size_shape_only":
        return np.arange(156, 172, dtype=np.int64)
    if arm == "remove_size_shape":
        return np.arange(0, 156, dtype=np.int64)
    return np.arange(0, 172, dtype=np.int64)


def _model_prediction(
    model: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model == "ridge":
        return fit_ridge(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            args.ridge_alphas,
        )
    if model == "gbdt":
        return fit_gbdt(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            "xgboost",
            "cuda",
            args.workers,
            args.gbdt_max_depth,
            args.gbdt_estimators,
            args.gbdt_learning_rate,
            args.gbdt_patience,
            seed,
        )
    if model == "mlp":
        return _fit_mlp_gpu_staged(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            hidden=args.mlp_hidden,
            batch_size=args.mlp_batch_size,
            epochs=args.mlp_epochs,
            patience=args.mlp_patience,
            learning_rate=args.mlp_learning_rate,
            weight_decay=args.mlp_weight_decay,
            seed=seed,
        )
    if model in {"catboost", "catboost_multirmse"}:
        return fit_catboost_multirmse(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            "cuda",
            args.workers,
            args.catboost_iterations,
            args.catboost_depth,
            args.catboost_learning_rate,
            args.catboost_patience,
            args.catboost_l2_leaf_reg,
            args.catboost_border_count,
            "0",
            args.catboost_gpu_ram_part,
            seed,
        )
    raise ValueError(model)


def _write_failure(output: Path, args: argparse.Namespace, error: Exception, started: float) -> None:
    atomic_json(output / "failed.json", {
        "schema_version": SCHEMA,
        "completed": False,
        "model": args.model,
        "reporter_slug": args.reporter,
        "fold": args.fold,
        "arm": args.arm,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "runtime_seconds": time.time() - started,
    })


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    execution_device = "cpu" if args.model == "ridge" else "cuda"
    if execution_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU fallback is forbidden")
    if args.model == "catboost" and args.reporter not in CATBOOST_ANCHORS:
        raise RuntimeError(f"CatBoost is restricted to the three predeclared anchors: {CATBOOST_ANCHORS}")
    phase_cache = PhaseCache.open(args.phase_cache)
    if tuple(phase_cache.x.shape) != (8_410_291, 172):
        raise RuntimeError(f"Frozen phase cache has unexpected shape: {phase_cache.x.shape}")
    target_table = pd.read_csv(args.target_table)
    if args.reporter not in set(target_table["reporter_slug"].astype(str)):
        raise ValueError(f"Unknown reporter: {args.reporter}")
    technical_core = _technical_core(args.target_feature_dictionary, target_table)
    reporter_row = target_table.set_index("reporter_slug").loc[args.reporter]
    source = args.exact_cache_root / f"all_cells_fluor_{args.reporter}.exact.h5"
    data = load_reporter_data(source, args.reporter, technical_core[args.reporter])
    if len(np.unique(data.phase_rows)) != len(data.phase_rows):
        raise RuntimeError("Exact reporter cache has duplicate phase cells")
    if np.any(data.phase_rows < 0) or np.any(data.phase_rows >= len(phase_cache.x)):
        raise RuntimeError("Exact reporter cache has an out-of-range phase row")
    x_all = np.asarray(phase_cache.x[data.phase_rows], dtype=np.float32)
    metadata = {name: np.asarray(values[data.phase_rows]) for name, values in phase_cache.metadata.items()}
    folds = np.asarray(phase_cache.folds["gene_holdout_main"][data.phase_rows], dtype=np.uint8)
    train, validation, test, validation_fold = partition_indices(folds, args.fold, int(phase_cache.manifest["n_folds"]))
    assert_split_integrity(
        "gene_holdout_main", train, validation, test, data.is_control,
        metadata["gene_code"], data.phase_rows, metadata["screen_code"],
        metadata["well_code"], metadata["tile_code"],
    )
    _, _, _, density_audit, covariates = _field_density_and_eccentricity(
        x_all, metadata["screen_code"], metadata["well_code"], metadata["tile_code"]
    )
    strict_audit = _strict_joint_audit(
        metadata["gene_code"], metadata["screen_code"], metadata["well_code"], metadata["tile_code"]
    )
    # The information intervention is a property of the frozen observation
    # relation, not of the estimator.  Reuse the already-frozen primary-MLP
    # donor assignment for every robustness model; keep model stochasticity on
    # its own model-specific seed below.
    intervention_seed = _stable_seed(args.reporter, args.fold, args.arm, "mlp")
    model_seed = _stable_seed(args.reporter, args.fold, args.arm, args.model)
    y_train_arm, y_validation_arm, y_test_arm, arm_audit, residual_mode = _normal_or_residual_data(
        args.arm, data.y, train, validation, test, metadata["gene_code"],
        metadata["screen_code"], covariates, intervention_seed,
    )
    x_indices = _feature_indices(args.arm)
    x_train_source = x_all[train][:, x_indices]
    x_validation_source = x_all[validation][:, x_indices]
    x_test_source = x_all[test][:, x_indices]
    preprocessing = fit_preprocessing(x_train_source, y_train_arm, args.min_x_finite_fraction)
    x_train = preprocessing.transform_x(x_train_source)
    x_validation = preprocessing.transform_x(x_validation_source)
    x_test = preprocessing.transform_x(x_test_source)
    y_train = preprocessing.transform_y(y_train_arm)
    y_validation = preprocessing.transform_y(y_validation_arm)
    y_test = preprocessing.transform_y(y_test_arm)
    started = time.time()
    device_name = (
        "CPU deterministic float64 normal equations"
        if execution_device == "cpu"
        else torch.cuda.get_device_name(0)
    )
    print(json.dumps({
        "status": "CPU_EXACT_FP64" if execution_device == "cpu" else "GPU_STAGED_EXACT_FP32",
        "model": args.model,
        "reporter": args.reporter,
        "fold": args.fold,
        "arm": args.arm,
        "train": len(train), "validation": len(validation), "test": len(test),
        "x": x_train.shape[1], "y": y_train.shape[1],
        "device": device_name,
    }), flush=True)
    prediction_scaled, hyperparameters = _model_prediction(
        args.model, x_train, y_train, x_validation, y_validation, x_test, args, model_seed
    )
    if prediction_scaled.shape != y_test.shape or not np.all(np.isfinite(prediction_scaled)):
        raise RuntimeError("Model produced invalid prediction matrix")
    target_test = ~data.is_control[test]
    feature_names = data.target_feature_names[preprocessing.y_keep]
    prediction_raw = preprocessing.inverse_y(prediction_scaled)
    truth_raw = y_test_arm[:, preprocessing.y_keep]
    if residual_mode:
        baseline_raw = np.zeros_like(truth_raw[target_test])
        cell_summary, cell_features = matrix_metrics(
            truth_raw[target_test], prediction_raw[target_test], baseline_raw, feature_names
        )
        gene_summary: dict[str, Any] = {
            "not_applicable": True,
            "reason": "cross-fitted gene_x_screen residuals intentionally remove group-level KO response",
        }
        gene_features = pd.DataFrame()
    else:
        baseline_raw = np.broadcast_to(preprocessing.y_mean, truth_raw[target_test].shape)
        cell_summary, cell_features = matrix_metrics(
            truth_raw[target_test], prediction_raw[target_test], baseline_raw, feature_names
        )
        profiles = aggregate_gene_profiles(
            y_test, prediction_scaled, metadata["screen_code"][test], metadata["gene_code"][test], data.is_control[test]
        )
        gene_summary, gene_features = profile_metrics(profiles, feature_names)
        gene_summary.update(bootstrap_gene_profile_metrics(profiles, args.bootstrap_draws, model_seed + 71))
        atomic_npy(args.output_dir / "gene_truth_control_relative_scaled.npy", profiles.truth)
        atomic_npy(args.output_dir / "gene_prediction_control_relative_scaled.npy", profiles.prediction)
        del profiles
    cell_summary.update({
        "n_targeting_test_cells": int(target_test.sum()),
        "n_control_test_cells": int((~target_test).sum()),
        "conditional_residual_analysis": bool(residual_mode),
    })
    cell_features.to_csv(args.output_dir / "cell_endpoint_metrics.csv", index=False)
    if len(gene_features):
        gene_features.to_csv(args.output_dir / "gene_endpoint_metrics.csv", index=False)
    phase_feature_names = np.asarray(phase_cache.manifest["feature_names"], dtype=str)
    cohort_fingerprint = hash_arrays(
        data.phase_rows[train], data.phase_rows[validation], data.phase_rows[test]
    )
    result = {
        "schema_version": SCHEMA,
        "completed": True,
        "status": "complete",
        "mode": "formal_same_cell_falsification",
        "dataset_version": "ops_phase172_exact_frozen_v1",
        "split": "gene_holdout_main",
        "fold": int(args.fold),
        "validation_fold": int(validation_fold),
        "reporter_slug": args.reporter,
        "reporter_short_name": str(reporter_row.short_name),
        "model": args.model,
        "arm": args.arm,
        "execution_device": execution_device,
        "cpu_fallback_allowed": args.model == "ridge",
        "cohort_policy": "all_finite_exact_pairs; identical per reporter_fold across arms",
        "cohort_fingerprint_sha256": cohort_fingerprint,
        "partition_counts": {
            "train": int(len(train)), "validation": int(len(validation)), "test": int(len(test)),
            "train_targeting": int((~data.is_control[train]).sum()),
            "validation_targeting": int((~data.is_control[validation]).sum()),
            "test_targeting": int(target_test.sum()),
        },
        "input_feature_policy": {
            "n_input_before_train_filter": int(len(x_indices)),
            "n_input_after_train_filter": int(len(preprocessing.x_keep)),
            "selected_phase_feature_names": phase_feature_names[x_indices][preprocessing.x_keep].tolist(),
        },
        "target_feature_policy": "frozen_technical_core",
        "n_target_features": int(len(feature_names)),
        "target_feature_names": feature_names.astype(str).tolist(),
        "cell_metrics": cell_summary,
        "gene_metrics": gene_summary,
        "arm_audit": arm_audit,
        "intervention_seed_contract": {
            "seed": int(intervention_seed),
            "reference_estimator": "mlp",
            "shared_across_robustness_models": True,
        },
        "field_density_audit": density_audit,
        "strict_joint_shuffle_feasibility_audit": strict_audit,
        "hyperparameters": hyperparameters,
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__)),
            "phase_manifest_sha256": _sha256(args.phase_cache / "manifest.json"),
            "target_dictionary_sha256": _sha256(args.target_feature_dictionary),
        },
        "runtime_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "same_cell_result.json", result)
    print(json.dumps({
        "status": "COMPLETE",
        "model": args.model,
        "reporter": args.reporter,
        "fold": args.fold,
        "arm": args.arm,
        "cell_pearson": cell_summary.get("macro_feature_pearson"),
        "gene_pearson": gene_summary.get("macro_feature_pearson"),
        "runtime_minutes": result["runtime_seconds"] / 60.0,
    }), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    # The original formal campaign was intentionally frozen to PANEL12.  Do
    # not keep that panel restriction in the runner itself: the frozen target
    # table and exact H5 cache are the authoritative 52-reporter registry.
    # The plan builder still exposes PANEL12 as the default and an explicit
    # ``all52`` option for the expanded campaign.
    parser.add_argument("--reporter", required=True)
    parser.add_argument("--fold", required=True, type=int, choices=range(5))
    parser.add_argument(
        "--model",
        required=True,
        choices=("ridge", "gbdt", "mlp", "catboost", "catboost_multirmse"),
    )
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase-cache", type=Path, default=DEFAULT_PHASE_CACHE)
    parser.add_argument("--exact-cache-root", type=Path, default=DEFAULT_EXACT_ROOT)
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGET_TABLE)
    parser.add_argument("--target-feature-dictionary", type=Path, default=DEFAULT_TARGET_DICTIONARY)
    parser.add_argument("--min-x-finite-fraction", type=float, default=0.80)
    parser.add_argument("--bootstrap-draws", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--ridge-alphas",
        type=lambda value: [float(x) for x in value.split(",")],
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--gbdt-max-depth", type=int, default=6)
    parser.add_argument("--gbdt-estimators", type=int, default=300)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbdt-patience", type=int, default=20)
    parser.add_argument("--mlp-hidden", type=lambda value: [int(x) for x in value.split(",")], default=[256, 128])
    parser.add_argument("--mlp-batch-size", type=int, default=16384)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-patience", type=int, default=10)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05)
    parser.add_argument("--catboost-patience", type=int, default=20)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--catboost-border-count", type=int, default=128)
    parser.add_argument("--catboost-gpu-ram-part", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete = args.output_dir / "same_cell_result.json"
    if complete.exists():
        payload = json.loads(complete.read_text(encoding="utf-8"))
        expected = {"completed": True, "model": args.model, "reporter_slug": args.reporter, "fold": args.fold, "arm": args.arm}
        if all(payload.get(key) == value for key, value in expected.items()):
            print(json.dumps({"status": "RESUME_SKIP_COMPLETE", **expected}), flush=True)
            return
        raise RuntimeError("Output directory contains a result from a different job identity")
    started = time.time()
    try:
        run(args)
    except Exception as error:
        _write_failure(args.output_dir, args, error, started)
        print(f"FAILED: {type(error).__name__}: {error}", flush=True)
        raise


if __name__ == "__main__":
    main()
