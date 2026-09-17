"""Shared utilities for the full OPS phase-to-reporter specialist benchmark."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
from scipy.stats import rankdata


RUNNER_SCHEMA_VERSION = "ops-reporter-specialists-v2-technical-core"
# Keep existing model positions stable because the runner derives deterministic
# per-model seed offsets from this tuple.  New models must be appended.
VALID_MODELS = ("ridge", "knn", "gbdt", "mlp", "catboost_multirmse")
VALID_SPLITS = ("field_holdout_sanity", "gene_holdout_main")


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=str,
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def atomic_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    os.replace(temporary, path)


def hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def parse_csv_choice(value: str, allowed: Iterable[str], label: str) -> list[str]:
    allowed_tuple = tuple(allowed)
    if value.strip().lower() == "all":
        return list(allowed_tuple)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(allowed_tuple))
    if invalid:
        raise ValueError(f"Unknown {label}: {invalid}; allowed={allowed_tuple}")
    if not selected:
        raise ValueError(f"At least one {label} is required")
    return selected


@dataclass
class PhaseCache:
    root: Path
    manifest: dict[str, Any]
    x: np.ndarray
    metadata: dict[str, np.ndarray]
    categories: dict[str, list[str]]
    folds: dict[str, np.ndarray]

    @classmethod
    def open(cls, root: Path) -> "PhaseCache":
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing phase172 cache manifest: {manifest_path}. "
                "Run build_ops_phase172_cache.py first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        x = np.load(root / "phase172.npy", mmap_mode="r")
        if tuple(x.shape) != tuple(manifest["shape"]) or x.dtype != np.float32:
            raise RuntimeError("phase172.npy does not match its manifest")
        metadata = {
            name: np.load(root / f"{name}.npy", mmap_mode="r")
            for name in (
                "screen_code",
                "well_code",
                "tile_code",
                "gene_code",
                "sgrna_code",
                "is_target",
            )
        }
        folds = {
            split: np.load(root / f"{split}.fold.npy", mmap_mode="r")
            for split in VALID_SPLITS
        }
        categories = json.loads((root / "categories.json").read_text(encoding="utf-8"))
        n_rows = x.shape[0]
        for name, values in {**metadata, **folds}.items():
            if len(values) != n_rows:
                raise RuntimeError(f"Cache array {name} has {len(values)} rows, expected {n_rows}")
        return cls(root, manifest, x, metadata, categories, folds)


@dataclass
class ReporterData:
    slug: str
    phase_rows: np.ndarray
    y: np.ndarray
    is_control: np.ndarray
    target_feature_names: np.ndarray
    source_cache: Path
    source_size_bytes: int
    source_target_features: int
    dropped_nonfinite_target_rows: int


def load_reporter_data(
    cache_path: Path,
    slug: str,
    selected_target_feature_names: Iterable[str] | None = None,
) -> ReporterData:
    with h5py.File(cache_path, "r") as source:
        phase_rows = np.asarray(source["phase_row_index"][:], dtype=np.int64)
        y = np.asarray(source["fluorescence"][:], dtype=np.float32)
        is_control = np.asarray(source["metadata/is_control"][:], dtype=bool)
        names = decode(source["features/target_feature_names"][:])
    source_target_features = int(y.shape[1])
    if selected_target_feature_names is not None:
        selected = tuple(str(name) for name in selected_target_feature_names)
        if not selected or len(set(selected)) != len(selected):
            raise RuntimeError(f"Invalid locked target-feature selection for {slug}")
        available = set(names.astype(str))
        missing = sorted(set(selected) - available)
        if missing:
            raise RuntimeError(
                f"Locked technical-core features are absent from {cache_path}: {missing[:10]}"
            )
        selected_set = set(selected)
        keep = np.asarray([name in selected_set for name in names.astype(str)], dtype=bool)
        if int(keep.sum()) != len(selected):
            raise RuntimeError(f"Technical-core feature selection is ambiguous for {slug}")
        y = y[:, keep]
        names = names[keep]
    finite = np.all(np.isfinite(y), axis=1)
    dropped = int((~finite).sum())
    phase_rows = phase_rows[finite]
    y = y[finite]
    is_control = is_control[finite]
    if len(np.unique(phase_rows)) != len(phase_rows):
        raise RuntimeError(f"Reporter cache contains duplicate phase rows: {cache_path}")
    if y.shape[1] != len(names):
        raise RuntimeError(f"Target feature-name mismatch in {cache_path}")
    return ReporterData(
        slug=slug,
        phase_rows=phase_rows,
        y=y,
        is_control=is_control,
        target_feature_names=names,
        source_cache=cache_path,
        source_size_bytes=cache_path.stat().st_size,
        source_target_features=source_target_features,
        dropped_nonfinite_target_rows=dropped,
    )


@dataclass
class PreprocessingState:
    x_keep: np.ndarray
    x_median: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_keep: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    min_x_finite_fraction: float

    def transform_x(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values[:, self.x_keep], dtype=np.float32).copy()
        bad = ~np.isfinite(result)
        if np.any(bad):
            rows, columns = np.nonzero(bad)
            result[rows, columns] = self.x_median[columns]
        result -= self.x_mean
        result /= self.x_scale
        return result

    def transform_y(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values[:, self.y_keep], dtype=np.float32).copy()
        result -= self.y_mean
        result /= self.y_scale
        return result

    def inverse_y(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.y_scale + self.y_mean

    def to_json(self) -> dict[str, Any]:
        return {
            "min_x_finite_fraction": self.min_x_finite_fraction,
            "n_x_input": int(len(self.x_keep)),
            "x_kept_indices": self.x_keep.astype(int).tolist(),
            "x_median": self.x_median.astype(float).tolist(),
            "x_mean": self.x_mean.astype(float).tolist(),
            "x_scale": self.x_scale.astype(float).tolist(),
            "n_y_output": int(len(self.y_keep)),
            "y_kept_indices": self.y_keep.astype(int).tolist(),
            "y_mean": self.y_mean.astype(float).tolist(),
            "y_scale": self.y_scale.astype(float).tolist(),
        }


def fit_preprocessing(
    x_train: np.ndarray,
    y_train: np.ndarray,
    min_x_finite_fraction: float,
    variance_epsilon: float = 1e-8,
) -> PreprocessingState:
    finite_fraction = np.mean(np.isfinite(x_train), axis=0)
    candidate = np.flatnonzero(finite_fraction >= min_x_finite_fraction)
    if len(candidate) == 0:
        raise RuntimeError("No phase features pass the finite-fraction threshold")
    candidate_values = np.asarray(x_train[:, candidate], dtype=np.float32)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(np.where(np.isfinite(candidate_values), candidate_values, np.nan), axis=0)
    valid_median = np.isfinite(medians)
    candidate = candidate[valid_median]
    medians = np.asarray(medians[valid_median], dtype=np.float32)
    candidate_values = candidate_values[:, valid_median]
    bad = ~np.isfinite(candidate_values)
    if np.any(bad):
        rows, columns = np.nonzero(bad)
        candidate_values[rows, columns] = medians[columns]
    means = np.mean(candidate_values, axis=0, dtype=np.float64).astype(np.float32)
    scales = np.std(candidate_values, axis=0, dtype=np.float64).astype(np.float32)
    variable_x = np.isfinite(scales) & (scales > variance_epsilon)
    x_keep = candidate[variable_x]
    if len(x_keep) == 0:
        raise RuntimeError("All phase features are constant after train-only imputation")

    y_means = np.mean(y_train, axis=0, dtype=np.float64).astype(np.float32)
    y_scales = np.std(y_train, axis=0, dtype=np.float64).astype(np.float32)
    variable_y = np.isfinite(y_scales) & (y_scales > variance_epsilon)
    y_keep = np.flatnonzero(variable_y)
    if len(y_keep) == 0:
        raise RuntimeError("All target features are constant in the training partition")
    return PreprocessingState(
        x_keep=x_keep,
        x_median=medians[variable_x],
        x_mean=means[variable_x],
        x_scale=scales[variable_x],
        y_keep=y_keep,
        y_mean=y_means[variable_y],
        y_scale=y_scales[variable_y],
        min_x_finite_fraction=min_x_finite_fraction,
    )


def partition_indices(
    fold_values: np.ndarray,
    outer_fold: int,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    validation_fold = (outer_fold + 1) % n_folds
    test = np.flatnonzero(fold_values == outer_fold)
    validation = np.flatnonzero(fold_values == validation_fold)
    train = np.flatnonzero((fold_values != outer_fold) & (fold_values != validation_fold))
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError(
            f"Empty partition for outer={outer_fold}, validation={validation_fold}"
        )
    return train, validation, test, validation_fold


def assert_split_integrity(
    split_name: str,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    is_control: np.ndarray,
    gene_codes: np.ndarray,
    phase_rows: np.ndarray,
    screen_codes: np.ndarray,
    well_codes: np.ndarray,
    tile_codes: np.ndarray,
) -> None:
    if len(np.unique(phase_rows)) != len(phase_rows):
        raise RuntimeError("A phase cell occurs more than once within this reporter")
    if split_name == "field_holdout_sanity":
        groups: list[set[tuple[int, int, int]]] = []
        for indices in (train, validation, test):
            triples = np.column_stack(
                (screen_codes[indices], well_codes[indices], tile_codes[indices])
            )
            unique_triples = np.unique(triples, axis=0)
            groups.append(
                {
                    (int(screen), int(well), int(tile))
                    for screen, well, tile in unique_triples
                }
            )
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise RuntimeError("Field groups cross train/validation/test boundaries")
    if split_name == "gene_holdout_main":
        target = ~is_control
        sets = []
        for indices in (train, validation, test):
            sets.append(set(gene_codes[indices][target[indices]].tolist()))
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise RuntimeError("Target genes cross train/validation/test boundaries")


def validation_mse(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(prediction)) ** 2))


def fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    alphas: list[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    # Forming the normal equations in float32 and casting the result afterwards
    # is not numerically equivalent to a float64 solve.  The harmonized phase
    # features are strongly collinear, and the float32 Gram matrix can produce
    # catastrophic coefficients that pass validation but explode on another
    # field or gene fold.  Promote the operands before both matrix products.
    x_train_64 = np.asarray(x_train, dtype=np.float64)
    y_train_64 = np.asarray(y_train, dtype=np.float64)
    gram = x_train_64.T @ x_train_64
    gram = 0.5 * (gram + gram.T)
    cross = x_train_64.T @ y_train_64
    identity = np.eye(gram.shape[0], dtype=np.float64)
    eigenvalues = np.linalg.eigvalsh(gram)
    positive = eigenvalues[eigenvalues > 0]
    gram_condition = (
        float(eigenvalues[-1] / positive[0]) if len(positive) else math.inf
    )
    # Some otherwise finite CellProfiler ratios contain rare sentinel-like
    # values many orders of magnitude beyond the training support.  An
    # unbounded linear model can turn a single such held-out value into a
    # catastrophic prediction even when validation performance is normal.
    # Clamp evaluation inputs to the per-feature training range.  This is
    # train-only, preserves every training observation, and prevents Ridge
    # from extrapolating outside support it has actually seen.
    x_lower = np.min(x_train_64, axis=0)
    x_upper = np.max(x_train_64, axis=0)

    def clip_to_training_support(values: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        values_64 = np.asarray(values, dtype=np.float64)
        outside = (values_64 < x_lower) | (values_64 > x_upper)
        clipped = np.clip(values_64, x_lower, x_upper)
        return clipped, {
            "values_clipped": int(np.count_nonzero(outside)),
            "rows_clipped": int(np.count_nonzero(np.any(outside, axis=1))),
            "max_abs_before_clip": float(np.max(np.abs(values_64))),
            "max_abs_after_clip": float(np.max(np.abs(clipped))),
        }

    best: tuple[float, float, np.ndarray] | None = None
    validation_rows = []
    x_validation_64, validation_clip = clip_to_training_support(x_validation)
    for alpha in alphas:
        coefficients = np.linalg.solve(gram + float(alpha) * identity, cross)
        prediction = np.asarray(x_validation_64 @ coefficients, dtype=np.float32)
        score = validation_mse(y_validation, prediction)
        validation_rows.append({"alpha": float(alpha), "validation_mse": score})
        if best is None or score < best[0]:
            best = (score, float(alpha), coefficients)
    assert best is not None
    x_test_64, test_clip = clip_to_training_support(x_test)
    test_prediction = np.asarray(x_test_64 @ best[2], dtype=np.float32)
    return test_prediction, {
        "backend": "closed_form_numpy_float64_train_support_clip",
        "device": "cpu",
        "normal_equation_dtype": "float64",
        "evaluation_input_support": "per_feature_training_min_max",
        "validation_input_clip": validation_clip,
        "test_input_clip": test_clip,
        "gram_condition_number_unregularized": gram_condition,
        "gram_min_eigenvalue": float(eigenvalues[0]),
        "gram_max_eigenvalue": float(eigenvalues[-1]),
        "alpha": best[1],
        "validation_mse": best[0],
        "validation_grid": validation_rows,
        "fit_intercept": False,
        "note": "X and Y are centered using training-only statistics",
    }


def _distance_weighted_mean(
    distances_squared: np.ndarray,
    neighbor_indices: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> np.ndarray:
    distances = np.sqrt(np.maximum(distances_squared[:, :k], 0.0)).astype(
        np.float32, copy=False
    )
    indices = neighbor_indices[:, :k]
    values = y_train[indices]
    zero = distances <= 1e-12
    weights = np.empty_like(distances, dtype=np.float32)
    rows_with_zero = np.any(zero, axis=1)
    weights[rows_with_zero] = zero[rows_with_zero].astype(np.float32)
    weights[~rows_with_zero] = 1.0 / np.maximum(distances[~rows_with_zero], 1e-12)
    denominator = np.sum(weights, axis=1, keepdims=True)
    return np.sum(values * weights[:, :, None], axis=1) / denominator


def resolve_knn_backend(requested: str, device: str) -> tuple[str, Any | None]:
    if requested not in {"auto", "faiss", "torch", "sklearn"}:
        raise ValueError(f"Unknown kNN backend: {requested}")
    if requested in {"auto", "faiss"}:
        try:
            import faiss  # type: ignore

            return "faiss", faiss
        except ImportError:
            if requested == "faiss":
                raise RuntimeError("FAISS was requested but is not installed")
    if requested in {"auto", "torch"}:
        try:
            import torch

            return "torch", torch
        except ImportError:
            if requested == "torch":
                raise RuntimeError("torch kNN was requested but PyTorch is not installed")
    if device == "cuda":
        raise RuntimeError("CUDA kNN requires a FAISS build with GPU support")
    return "sklearn", None


def knn_predict_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_query: np.ndarray,
    k_values: list[int],
    backend: str,
    device: str,
    query_batch_size: int,
    train_block_size: int,
    workers: int,
    allow_slow_knn: bool,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    max_k = min(max(k_values), len(x_train))
    k_values = sorted({min(k, max_k) for k in k_values})
    resolved, backend_module = resolve_knn_backend(backend, device)
    predictions = {
        k: np.empty((len(x_query), y_train.shape[1]), dtype=np.float32) for k in k_values
    }
    x_train_c = np.ascontiguousarray(x_train, dtype=np.float32)
    if resolved == "faiss":
        faiss = backend_module
        index_cpu = faiss.IndexFlatL2(x_train_c.shape[1])
        index: Any = index_cpu
        gpu_used = False
        resources = None
        if (
            device == "cuda"
            and hasattr(faiss, "StandardGpuResources")
            and hasattr(faiss, "get_num_gpus")
            and faiss.get_num_gpus() >= 1
        ):
            resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(resources, 0, index_cpu)
            gpu_used = True
        index.add(x_train_c)
        for start in range(0, len(x_query), query_batch_size):
            stop = min(start + query_batch_size, len(x_query))
            distances, indices = index.search(
                np.ascontiguousarray(x_query[start:stop], dtype=np.float32), max_k
            )
            for k in k_values:
                predictions[k][start:stop] = _distance_weighted_mean(
                    distances, indices, y_train, k
                )
        details = {
            "backend": "faiss_index_flat_l2",
            "gpu": gpu_used,
            "device_fallback": "cpu" if device == "cuda" and not gpu_used else None,
        }
        del index
        del index_cpu
        del resources
    elif resolved == "torch":
        torch = backend_module
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("torch kNN requested CUDA but CUDA is unavailable")
        torch.set_num_threads(max(1, workers))
        train_tensor = torch.from_numpy(x_train_c).to(torch_device)
        train_norm = torch.sum(train_tensor * train_tensor, dim=1)
        train_block_size = max(train_block_size, max_k)
        with torch.no_grad():
            for start in range(0, len(x_query), query_batch_size):
                stop = min(start + query_batch_size, len(x_query))
                query = torch.from_numpy(
                    np.ascontiguousarray(x_query[start:stop], dtype=np.float32)
                ).to(torch_device)
                query_norm = torch.sum(query * query, dim=1, keepdim=True)
                best_distance = torch.full(
                    (len(query), max_k), float("inf"), device=torch_device
                )
                best_index = torch.full(
                    (len(query), max_k), -1, dtype=torch.long, device=torch_device
                )
                for train_start in range(0, len(train_tensor), train_block_size):
                    train_stop = min(train_start + train_block_size, len(train_tensor))
                    block = train_tensor[train_start:train_stop]
                    distance = query_norm + train_norm[train_start:train_stop][None, :]
                    distance = distance - 2.0 * (query @ block.T)
                    distance.clamp_(min=0.0)
                    local_k = min(max_k, train_stop - train_start)
                    local_distance, local_index = torch.topk(
                        distance, k=local_k, dim=1, largest=False, sorted=True
                    )
                    local_index += train_start
                    candidate_distance = torch.cat((best_distance, local_distance), dim=1)
                    candidate_index = torch.cat((best_index, local_index), dim=1)
                    best_distance, selected = torch.topk(
                        candidate_distance, k=max_k, dim=1, largest=False, sorted=True
                    )
                    best_index = torch.gather(candidate_index, 1, selected)
                    del distance, local_distance, local_index, candidate_distance, candidate_index
                distances = best_distance.cpu().numpy().astype(np.float32)
                indices = best_index.cpu().numpy().astype(np.int64)
                for k in k_values:
                    predictions[k][start:stop] = _distance_weighted_mean(
                        distances, indices, y_train, k
                    )
                del query, best_distance, best_index
        details = {
            "backend": "torch_blocked_exact_l2",
            "gpu": torch_device.type == "cuda",
            "train_block_size": train_block_size,
            "torch_version": torch.__version__,
        }
        del train_tensor, train_norm
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        pair_count = int(len(x_train)) * int(len(x_query))
        if pair_count > 2_000_000_000 and not allow_slow_knn:
            raise RuntimeError(
                f"sklearn exact kNN would evaluate {pair_count:,} train-query pairs. "
                "Install FAISS or pass --allow-slow-knn explicitly."
            )
        from sklearn.neighbors import NearestNeighbors

        index = NearestNeighbors(
            n_neighbors=max_k,
            algorithm="brute",
            metric="euclidean",
            n_jobs=workers,
        ).fit(x_train_c)
        for start in range(0, len(x_query), query_batch_size):
            stop = min(start + query_batch_size, len(x_query))
            distances, indices = index.kneighbors(
                np.asarray(x_query[start:stop], dtype=np.float32), return_distance=True
            )
            distances_squared = np.asarray(distances, dtype=np.float32) ** 2
            for k in k_values:
                predictions[k][start:stop] = _distance_weighted_mean(
                    distances_squared, indices, y_train, k
                )
        details = {"backend": "sklearn_brute_l2", "gpu": False}
    details.update(
        {
            "k_candidates": k_values,
            "max_k_queried_once": max_k,
            "query_batch_size": query_batch_size,
            "train_block_size": train_block_size if resolved == "torch" else None,
        }
    )
    return predictions, details


def fit_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    k_values: list[int],
    backend: str,
    device: str,
    query_batch_size: int,
    train_block_size: int,
    workers: int,
    allow_slow_knn: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    validation_predictions, details = knn_predict_candidates(
        x_train,
        y_train,
        x_validation,
        k_values,
        backend,
        device,
        query_batch_size,
        train_block_size,
        workers,
        allow_slow_knn,
    )
    scores = {
        k: validation_mse(y_validation, prediction)
        for k, prediction in validation_predictions.items()
    }
    selected_k = min(scores, key=scores.get)
    del validation_predictions
    test_predictions, test_details = knn_predict_candidates(
        x_train,
        y_train,
        x_test,
        [selected_k],
        backend,
        device,
        query_batch_size,
        train_block_size,
        workers,
        allow_slow_knn,
    )
    details.update(test_details)
    details.update(
        {
            "selected_k": int(selected_k),
            "validation_mse": float(scores[selected_k]),
            "validation_scores": {str(k): float(value) for k, value in scores.items()},
            "train_only_neighbor_index": True,
        }
    )
    return test_predictions[selected_k], details


def choose_torch_device(requested: str) -> str:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    try:
        import torch
    except ImportError:
        if requested == "cuda":
            raise RuntimeError("CUDA was requested but PyTorch is not installed")
        return "cpu"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    device: str,
    hidden: list[int],
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    workers: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("MLP requires PyTorch") from error

    resolved_device = choose_torch_device(device)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(max(1, workers))

    layers: list[nn.Module] = []
    current = x_train.shape[1]
    for width in hidden:
        layers.extend([nn.Linear(current, width), nn.GELU(), nn.LayerNorm(width)])
        current = width
    layers.append(nn.Linear(current, y_train.shape[1]))
    model = nn.Sequential(*layers).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=resolved_device == "cuda",
        generator=generator,
    )

    def predict(values: np.ndarray) -> np.ndarray:
        model.eval()
        output = np.empty((len(values), y_train.shape[1]), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(values), batch_size * 2):
                stop = min(start + batch_size * 2, len(values))
                batch = torch.from_numpy(np.asarray(values[start:stop], dtype=np.float32)).to(
                    resolved_device, non_blocking=True
                )
                output[start:stop] = model(batch).detach().cpu().numpy().astype(np.float32)
        return output

    best_loss = math.inf
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    history = []
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        count = 0
        epoch_start = time.time()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(resolved_device, non_blocking=True)
            batch_y = batch_y.to(resolved_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * len(batch_x)
            count += len(batch_x)
        validation_prediction = predict(x_validation)
        score = validation_mse(y_validation, validation_prediction)
        history.append(
            {
                "epoch": epoch,
                "train_mse": running / max(count, 1),
                "validation_mse": score,
                "seconds": time.time() - epoch_start,
            }
        )
        if score < best_loss - 1e-6:
            best_loss = score
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    prediction = predict(x_test)
    return prediction, {
        "backend": "pytorch",
        "device": resolved_device,
        "hidden": hidden,
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_validation_mse": best_loss,
        "runtime_seconds": time.time() - started,
        "history": history,
        "torch_version": torch.__version__,
    }


def fit_gbdt(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    backend: str,
    device: str,
    workers: int,
    max_depth: int,
    estimators: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if backend not in {"auto", "xgboost", "sklearn"}:
        raise ValueError("GBDT backend must be auto, xgboost, or sklearn")
    resolved = backend
    xgb = None
    if backend in {"auto", "xgboost"}:
        try:
            import xgboost as xgb_module

            xgb = xgb_module
            resolved = "xgboost"
        except ImportError:
            if backend == "xgboost":
                raise RuntimeError("XGBoost was requested but is not installed")
            resolved = "sklearn"
    if resolved == "sklearn" and device == "cuda":
        raise RuntimeError("sklearn HistGradientBoostingRegressor is CPU-only")

    xgboost_build_info: dict[str, Any] | None = None
    validation_dmatrix = None
    test_dmatrix = None
    if resolved == "xgboost":
        assert xgb is not None
        xgboost_build_info = dict(xgb.build_info())
        if device == "cuda" and not bool(xgboost_build_info.get("USE_CUDA", False)):
            raise RuntimeError(
                "CUDA GBDT was requested, but the imported XGBoost wheel was built "
                "without CUDA support. Install the standard CUDA-enabled XGBoost "
                "wheel in the persistent OPS Conda environment."
            )
        # Reuse device-independent DMatrix objects for every output regressor.  In
        # particular, this avoids XGBRegressor.predict receiving a CPU NumPy array
        # while the fitted booster lives on the GPU, which otherwise triggers a
        # slower prediction fallback and a misleading device-mismatch warning.
        validation_dmatrix = xgb.DMatrix(x_validation, nthread=workers)
        test_dmatrix = xgb.DMatrix(x_test, nthread=workers)

    prediction = np.empty((len(x_test), y_train.shape[1]), dtype=np.float32)
    validation_losses = []
    best_iterations = []
    actual_booster_devices: list[str] = []
    started = time.time()
    for output_index in range(y_train.shape[1]):
        if resolved == "xgboost":
            assert xgb is not None
            kwargs = {
                "n_estimators": estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "device": "cuda" if device == "cuda" else "cpu",
                "n_jobs": workers,
                "random_state": seed + output_index,
                "early_stopping_rounds": patience,
            }
            try:
                model = xgb.XGBRegressor(**kwargs)
                model.fit(
                    x_train,
                    y_train[:, output_index],
                    eval_set=[(x_validation, y_validation[:, output_index])],
                    verbose=False,
                )
            except (TypeError, ValueError) as error:
                # XGBoost <2.0 used gpu_hist and fit-time early stopping.
                legacy = dict(kwargs)
                legacy.pop("device", None)
                legacy.pop("early_stopping_rounds", None)
                if device == "cuda":
                    legacy["tree_method"] = "gpu_hist"
                model = xgb.XGBRegressor(**legacy)
                try:
                    model.fit(
                        x_train,
                        y_train[:, output_index],
                        eval_set=[(x_validation, y_validation[:, output_index])],
                        early_stopping_rounds=patience,
                        verbose=False,
                    )
                except Exception:
                    raise RuntimeError(
                        "XGBoost failed with both modern and legacy device APIs"
                    ) from error
            booster = model.get_booster()
            booster_config = json.loads(booster.save_config())
            actual_device = str(
                booster_config["learner"]["generic_param"].get("device", "unknown")
            )
            actual_booster_devices.append(actual_device)
            if device == "cuda" and not actual_device.startswith("cuda"):
                raise RuntimeError(
                    "XGBoost silently failed to use CUDA: "
                    f"requested cuda, fitted booster reports {actual_device!r}"
                )
            best_iteration = int(getattr(model, "best_iteration", estimators - 1))
            best_iterations.append(best_iteration)
            assert validation_dmatrix is not None and test_dmatrix is not None
            try:
                validation_prediction = booster.predict(
                    validation_dmatrix, iteration_range=(0, best_iteration + 1)
                )
                output_prediction = booster.predict(
                    test_dmatrix, iteration_range=(0, best_iteration + 1)
                )
            except TypeError:
                # Compatibility for old XGBoost releases retained by the legacy
                # training branch above.
                validation_prediction = booster.predict(
                    validation_dmatrix, ntree_limit=best_iteration + 1
                )
                output_prediction = booster.predict(
                    test_dmatrix, ntree_limit=best_iteration + 1
                )
            prediction[:, output_index] = output_prediction.astype(np.float32)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor

            model = HistGradientBoostingRegressor(
                max_iter=estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                early_stopping=True,
                n_iter_no_change=patience,
                random_state=seed + output_index,
            )
            model.fit(x_train, y_train[:, output_index])
            validation_prediction = model.predict(x_validation)
            prediction[:, output_index] = model.predict(x_test).astype(np.float32)
            best_iterations.append(int(model.n_iter_))
        validation_losses.append(
            float(np.mean((y_validation[:, output_index] - validation_prediction) ** 2))
        )
    return prediction, {
        "backend": resolved,
        "device": device if resolved == "xgboost" else "cpu",
        "actual_booster_devices": actual_booster_devices,
        "xgboost_version": xgb.__version__ if resolved == "xgboost" else None,
        "xgboost_build_info": xgboost_build_info,
        "one_regressor_per_target_feature": True,
        "n_output_regressors": int(y_train.shape[1]),
        "max_depth": max_depth,
        "estimators": estimators,
        "learning_rate": learning_rate,
        "patience": patience,
        "eval_metric": "rmse" if resolved == "xgboost" else None,
        "macro_validation_mse": float(np.mean(validation_losses)),
        "best_iterations": best_iterations,
        "runtime_seconds": time.time() - started,
    }


def fit_catboost_multirmse(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    device: str,
    workers: int,
    iterations: int,
    depth: int,
    learning_rate: float,
    patience: int,
    l2_leaf_reg: float,
    border_count: int,
    devices: str,
    gpu_ram_part: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one joint CatBoost MultiRMSE model for a reporter target vector."""
    try:
        import catboost
        from catboost import CatBoostRegressor, Pool
    except ImportError as error:
        raise RuntimeError("CatBoost MultiRMSE was requested but CatBoost is not installed") from error

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported CatBoost device: {device}")
    if y_train.ndim != 2 or y_validation.ndim != 2:
        raise RuntimeError("CatBoost MultiRMSE requires two-dimensional target matrices")
    if y_train.shape[1] != y_validation.shape[1] or y_train.shape[1] < 1:
        raise RuntimeError("CatBoost train/validation target dimensions are incompatible")
    for label, values in (
        ("x_train", x_train),
        ("y_train", y_train),
        ("x_validation", x_validation),
        ("y_validation", y_validation),
        ("x_test", x_test),
    ):
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"CatBoost MultiRMSE received non-finite {label}")

    requested_task_type = "GPU" if device == "cuda" else "CPU"
    gpu_device_count: int | None = None
    if device == "cuda":
        from catboost.utils import get_gpu_device_count

        gpu_device_count = int(get_gpu_device_count())
        if gpu_device_count < 1:
            raise RuntimeError(
                "CUDA CatBoost was requested, but CatBoost reports no available GPU. "
                "Run in the host GPU session with .ops_catboost_gpu_deps first in PYTHONPATH."
            )

    model_parameters: dict[str, Any] = {
        "loss_function": "MultiRMSE",
        "eval_metric": "MultiRMSE",
        "task_type": requested_task_type,
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "l2_leaf_reg": l2_leaf_reg,
        "border_count": border_count,
        "boosting_type": "Plain",
        "random_seed": seed,
        "od_type": "Iter",
        "od_wait": patience,
        "use_best_model": True,
        "allow_writing_files": False,
        "thread_count": workers,
        "verbose": False,
    }
    if device == "cuda":
        model_parameters.update(
            {
                "devices": devices,
                "gpu_ram_part": gpu_ram_part,
            }
        )

    started = time.time()
    train_pool = Pool(x_train, label=y_train)
    validation_pool = Pool(x_validation, label=y_validation)
    test_pool = Pool(x_test)
    model = CatBoostRegressor(**model_parameters)
    model.fit(train_pool, eval_set=validation_pool, verbose=False)

    effective_parameters = model.get_all_params()
    fitted_task_type = str(effective_parameters.get("task_type", "unknown")).upper()
    if fitted_task_type != requested_task_type:
        raise RuntimeError(
            "CatBoost fitted with an unexpected processing unit: "
            f"requested {requested_task_type}, fitted {fitted_task_type}"
        )
    if str(effective_parameters.get("loss_function")) != "MultiRMSE":
        raise RuntimeError("CatBoost fitted with an unexpected loss function")

    def checked_prediction(pool: Any, n_rows: int, label: str) -> np.ndarray:
        values = np.asarray(model.predict(pool), dtype=np.float32)
        if values.ndim == 1 and y_train.shape[1] == 1:
            values = values[:, None]
        expected_shape = (n_rows, y_train.shape[1])
        if values.shape != expected_shape:
            raise RuntimeError(
                f"CatBoost {label} prediction shape is {values.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"CatBoost produced non-finite {label} predictions")
        return values

    validation_prediction = checked_prediction(
        validation_pool, len(x_validation), "validation"
    )
    prediction = checked_prediction(test_pool, len(x_test), "test")
    validation_squared_error = (
        np.asarray(y_validation, dtype=np.float32) - validation_prediction
    ) ** 2
    macro_validation_mse = float(np.mean(validation_squared_error))
    validation_multirmse = float(
        np.sqrt(np.sum(validation_squared_error, dtype=np.float64) / len(y_validation))
    )
    best_score = model.get_best_score()
    best_validation_scores = [
        float(metrics["MultiRMSE"])
        for dataset, metrics in best_score.items()
        if dataset != "learn" and "MultiRMSE" in metrics
    ]
    if len(best_validation_scores) != 1 or not np.isfinite(best_validation_scores[0]):
        raise RuntimeError(f"CatBoost did not expose one finite validation MultiRMSE: {best_score}")
    best_iteration = int(model.get_best_iteration())
    tree_count = int(model.tree_count_)
    if tree_count < 1 or best_iteration < 0 or best_iteration >= iterations:
        raise RuntimeError(
            f"Invalid CatBoost best/tree iteration state: best={best_iteration}, trees={tree_count}"
        )

    return prediction, {
        "backend": "catboost",
        "model_variant": "joint_multirmse",
        "device": device,
        "requested_task_type": requested_task_type,
        "fitted_task_type": fitted_task_type,
        "devices": devices if device == "cuda" else None,
        "gpu_device_count": gpu_device_count,
        "loss_function": "MultiRMSE",
        "eval_metric": "MultiRMSE",
        "joint_multioutput": True,
        "one_regressor_per_target_feature": False,
        "n_output_dimensions": int(y_train.shape[1]),
        "iterations": iterations,
        "depth": depth,
        "learning_rate": learning_rate,
        "patience": patience,
        "l2_leaf_reg": l2_leaf_reg,
        "border_count": border_count,
        "gpu_ram_part": gpu_ram_part if device == "cuda" else None,
        "boosting_type": "Plain",
        "random_seed": seed,
        "tree_count": tree_count,
        "best_iteration": best_iteration,
        "best_validation_multirmse": best_validation_scores[0],
        "prediction_validation_multirmse": validation_multirmse,
        "macro_validation_mse": macro_validation_mse,
        "effective_model_params": effective_parameters,
        "catboost_version": catboost.__version__,
        "catboost_path": str(Path(catboost.__file__).resolve()),
        "gpu_training_nondeterministic": device == "cuda",
        "runtime_seconds": time.time() - started,
    }


def _pearson_columns(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth_centered = truth - np.mean(truth, axis=0, keepdims=True)
    prediction_centered = prediction - np.mean(prediction, axis=0, keepdims=True)
    numerator = np.sum(truth_centered * prediction_centered, axis=0)
    denominator = np.sqrt(
        np.sum(truth_centered**2, axis=0) * np.sum(prediction_centered**2, axis=0)
    )
    result = np.full(truth.shape[1], np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _spearman_columns(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    result = np.full(truth.shape[1], np.nan, dtype=float)
    for column in range(truth.shape[1]):
        result[column] = _pearson_columns(
            rankdata(truth[:, column]).reshape(-1, 1),
            rankdata(prediction[:, column]).reshape(-1, 1),
        )[0]
    return result


def matrix_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    feature_names: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    truth = np.asarray(truth, dtype=np.float32)
    prediction = np.asarray(prediction, dtype=np.float32)
    baseline = np.asarray(baseline, dtype=np.float32)
    squared_error = np.mean((truth - prediction) ** 2, axis=0)
    baseline_error = np.mean((truth - baseline) ** 2, axis=0)
    gain = np.full_like(squared_error, np.nan, dtype=float)
    valid = baseline_error > 0
    gain[valid] = 1.0 - squared_error[valid] / baseline_error[valid]
    variance = np.sum((truth - np.mean(truth, axis=0, keepdims=True)) ** 2, axis=0)
    r2 = np.full_like(squared_error, np.nan, dtype=float)
    valid_r2 = variance > 0
    r2[valid_r2] = 1.0 - np.sum(
        (truth[:, valid_r2] - prediction[:, valid_r2]) ** 2, axis=0
    ) / variance[valid_r2]
    pearson = _pearson_columns(truth, prediction)
    spearman = _spearman_columns(truth, prediction)
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "mse": squared_error,
            "baseline_mse": baseline_error,
            "gain_vs_train_mean": gain,
            "r2": r2,
            "pearson": pearson,
            "spearman": spearman,
        }
    )
    summary = {
        "mse": float(np.mean(squared_error)),
        "baseline_mse": float(np.mean(baseline_error)),
        "gain_vs_train_mean": float(1.0 - np.mean(squared_error) / np.mean(baseline_error)),
        "macro_feature_gain": float(np.nanmean(gain)),
        "macro_feature_r2": float(np.nanmean(r2)),
        "macro_feature_pearson": float(np.nanmean(pearson)),
        "macro_feature_spearman": float(np.nanmean(spearman)),
    }
    return summary, table


@dataclass
class GeneProfiles:
    truth: np.ndarray
    prediction: np.ndarray
    screen_code: np.ndarray
    gene_code: np.ndarray
    n_cells: np.ndarray
    n_control_cells: np.ndarray


def _group_means(
    values: np.ndarray,
    primary: np.ndarray,
    secondary: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if secondary is None:
        keys = np.asarray(primary, dtype=np.int64)
    else:
        keys = (np.asarray(primary, dtype=np.int64) << 32) ^ (
            np.asarray(secondary, dtype=np.int64) & 0xFFFFFFFF
        )
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    unique, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    sums = np.add.reduceat(np.asarray(values[order], dtype=np.float64), starts, axis=0)
    means = np.asarray(sums / counts[:, None], dtype=np.float32)
    return unique, means, counts.astype(np.int64)


def aggregate_gene_profiles(
    truth: np.ndarray,
    prediction: np.ndarray,
    screen_codes: np.ndarray,
    gene_codes: np.ndarray,
    is_control: np.ndarray,
) -> GeneProfiles:
    control_rows = np.flatnonzero(is_control)
    target_rows = np.flatnonzero(~is_control)
    if len(control_rows) == 0 or len(target_rows) == 0:
        raise RuntimeError("Gene-level matched-control aggregation needs test controls and KOs")
    control_screens, control_truth, control_counts = _group_means(
        truth[control_rows], screen_codes[control_rows]
    )
    pred_control_screens, control_prediction, pred_control_counts = _group_means(
        prediction[control_rows], screen_codes[control_rows]
    )
    if not np.array_equal(control_screens, pred_control_screens) or not np.array_equal(
        control_counts, pred_control_counts
    ):
        raise RuntimeError("True and predicted control groups are misaligned")
    group_keys, group_truth, counts = _group_means(
        truth[target_rows], screen_codes[target_rows], gene_codes[target_rows]
    )
    pred_keys, group_prediction, pred_counts = _group_means(
        prediction[target_rows], screen_codes[target_rows], gene_codes[target_rows]
    )
    if not np.array_equal(group_keys, pred_keys) or not np.array_equal(counts, pred_counts):
        raise RuntimeError("True and predicted KO groups are misaligned")
    group_screen = (group_keys >> 32).astype(np.int64)
    group_gene = (group_keys & 0xFFFFFFFF).astype(np.int64)
    positions = np.searchsorted(control_screens, group_screen)
    valid = positions < len(control_screens)
    valid[valid] &= control_screens[positions[valid]] == group_screen[valid]
    if not np.all(valid):
        missing = np.unique(group_screen[~valid]).astype(int).tolist()
        raise RuntimeError(
            "Matched-control aggregation would silently drop target groups because "
            f"test controls are absent for screen codes {missing}"
        )
    positions = positions[valid]
    return GeneProfiles(
        truth=group_truth[valid] - control_truth[positions],
        prediction=group_prediction[valid] - control_prediction[positions],
        screen_code=group_screen[valid],
        gene_code=group_gene[valid],
        n_cells=counts[valid],
        n_control_cells=control_counts[positions],
    )


def profile_metrics(
    profiles: GeneProfiles,
    feature_names: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    zero = np.zeros_like(profiles.truth)
    summary, table = matrix_metrics(profiles.truth, profiles.prediction, zero, feature_names)
    truth_norm = np.linalg.norm(profiles.truth, axis=1)
    prediction_norm = np.linalg.norm(profiles.prediction, axis=1)
    denominator = truth_norm * prediction_norm
    cosine = np.full(len(truth_norm), np.nan)
    valid = denominator > 0
    cosine[valid] = np.sum(
        profiles.truth[valid] * profiles.prediction[valid], axis=1
    ) / denominator[valid]
    summary["mean_profile_cosine"] = float(np.nanmean(cosine))
    summary["response_magnitude_spearman"] = float(
        _spearman_columns(truth_norm.reshape(-1, 1), prediction_norm.reshape(-1, 1))[0]
    )
    summary["n_gene_screen_groups"] = int(len(profiles.truth))
    summary["n_genes"] = int(len(np.unique(profiles.gene_code)))
    summary["n_screens"] = int(len(np.unique(profiles.screen_code)))
    return summary, table


def bootstrap_gene_profile_metrics(
    profiles: GeneProfiles,
    draws: int,
    seed: int,
) -> dict[str, float]:
    genes = np.unique(profiles.gene_code)
    by_gene = {gene: np.flatnonzero(profiles.gene_code == gene) for gene in genes}
    rng = np.random.default_rng(seed)
    gain_draws = np.empty(draws, dtype=float)
    magnitude_draws = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(genes, size=len(genes), replace=True)
        indices = np.concatenate([by_gene[gene] for gene in sampled])
        truth = profiles.truth[indices]
        prediction = profiles.prediction[indices]
        numerator = float(np.mean((truth - prediction) ** 2))
        denominator = float(np.mean(truth**2))
        gain_draws[draw] = 1.0 - numerator / denominator if denominator > 0 else np.nan
        truth_norm = np.linalg.norm(truth, axis=1)
        prediction_norm = np.linalg.norm(prediction, axis=1)
        magnitude_draws[draw] = _spearman_columns(
            truth_norm.reshape(-1, 1), prediction_norm.reshape(-1, 1)
        )[0]
    return {
        "gene_bootstrap_draws": int(draws),
        "gain_ci95_low": float(np.nanquantile(gain_draws, 0.025)),
        "gain_ci95_high": float(np.nanquantile(gain_draws, 0.975)),
        "magnitude_spearman_ci95_low": float(np.nanquantile(magnitude_draws, 0.025)),
        "magnitude_spearman_ci95_high": float(np.nanquantile(magnitude_draws, 0.975)),
    }


def save_gene_profiles(path: Path, profiles: GeneProfiles) -> None:
    path.mkdir(parents=True, exist_ok=True)
    atomic_npy(path / "truth_control_relative.npy", profiles.truth)
    atomic_npy(path / "prediction_control_relative.npy", profiles.prediction)
    atomic_npy(path / "screen_code.npy", profiles.screen_code)
    atomic_npy(path / "gene_code.npy", profiles.gene_code)
    atomic_npy(path / "n_cells.npy", profiles.n_cells)
    atomic_npy(path / "n_control_cells.npy", profiles.n_control_cells)
