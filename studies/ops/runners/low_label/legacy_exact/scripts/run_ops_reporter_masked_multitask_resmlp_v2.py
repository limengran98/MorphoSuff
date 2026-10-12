#!/usr/bin/env python3
"""Run the preregistered OPS masked-multitask ResMLP V2 factorial arms.

The frozen supervised unit is ``(phase_row_index, reporter_slug)``.  This
runner always assigns the *phase row* to an outer fold before expanding its
available reporter labels, so labels from the same cell cannot cross
train/validation/test boundaries.  Each reporter keeps its frozen technical
core in the exact H5 column order and receives train-only target
standardization.  Phase preprocessing is shared and is fit once on the unique
training phase rows in each split/fold.

Arms A/B train complete-core rows and arms C/D add train-fold partial rows
with an explicit endpoint mask.  A/C use the frozen V1 capacity; B/D only
increase 60D/72D head capacity while keeping the total parameter count matched.
All arms retain the frozen complete-core validation/test cohorts, preprocessing
reference, optimizer schedule, and evaluation metrics.  Formal runs are CUDA
only and fail closed rather than moving the model or data path to CPU.
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import fcntl
import gc
import hashlib
import importlib
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import h5py

import ops_reporter_cell_eval_adapter as cell_eval_adapter
from ops_reporter_specialist_lib import (
    VALID_SPLITS,
    PhaseCache,
    aggregate_gene_profiles,
    assert_split_integrity,
    atomic_json,
    atomic_npy,
    bootstrap_gene_profile_metrics,
    choose_torch_device,
    decode,
    hash_arrays,
    matrix_metrics,
    parse_csv_choice,
    partition_indices,
    profile_metrics,
)


SCHEMA_VERSION = "ops-reporter-masked-multitask-resmlp-v2-factorial"
MODEL_NAME = "masked_multitask_resmlp_v2"
SEED_TOKEN = "masked_multitask_resmlp"
VALID_ARMS = ("A", "B", "C", "D")
DEFAULT_PHASE_CACHE = Path("data/processed/ops_phase172_indexed")
DEFAULT_EXACT_ROOT = Path("data/processed/ops_full_reporter_exact/reporters")
DEFAULT_TARGET_TABLE = Path("results/ops_phase0_asset_audit/reporter_targets.csv")
DEFAULT_TARGET_FEATURE_DICTIONARY = Path(
    "results/ops_phase0_asset_audit/target_feature_dictionary.csv"
)
DEFAULT_SPECIALIST_REFERENCE = Path("results/ops_reporter_specialists_v1")
DEFAULT_V1_REFERENCE = Path("results/ops_reporter_masked_multitask_resmlp_v1")
DEFAULT_OUTPUT = Path("results/ops_reporter_masked_multitask_resmlp_v2/arm_unset")
METADATA_LOCK_TIMEOUT_SECONDS = 120.0
METADATA_LOCK_POLL_SECONDS = 0.05


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(base: int, *tokens: object) -> int:
    digest = hashlib.sha256(str(base).encode("utf-8"))
    for token in tokens:
        digest.update(b"\0")
        digest.update(str(token).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], "little") % (2**31 - 1)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


@contextlib.contextmanager
def output_metadata_lock(
    output_root: Path,
    *,
    timeout_seconds: float = METADATA_LOCK_TIMEOUT_SECONDS,
    poll_seconds: float = METADATA_LOCK_POLL_SECONDS,
) -> Iterable[None]:
    """Serialize shared metadata publication across parallel job workers.

    Some shared filesystems return ``EAGAIN`` from a nominally blocking
    ``flock`` instead of sleeping until the peer releases it.  Use an explicit
    non-blocking acquisition loop so that behavior is identical on local and
    shared storage, while retaining a finite timeout and the existing atomic
    metadata writes inside the critical section.
    """

    if timeout_seconds <= 0:
        raise ValueError("metadata-lock timeout must be positive")
    if poll_seconds <= 0:
        raise ValueError("metadata-lock poll interval must be positive")

    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".metadata.lock"
    with lock_path.open("a+b") as handle:
        deadline = time.monotonic() + timeout_seconds
        attempts = 0
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if error.errno not in {
                    errno.EACCES,
                    errno.EAGAIN,
                    errno.EINTR,
                    errno.EWOULDBLOCK,
                }:
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "Timed out waiting for shared metadata lock "
                        f"after {timeout_seconds:.1f}s: {lock_path}"
                    ) from error
                # A short capped backoff avoids a launch-time thundering herd
                # without adding material latency to the usual two-job smoke.
                backoff = poll_seconds * (2 ** min(attempts, 4))
                time.sleep(min(backoff, 0.5, remaining))
                attempts += 1
        try:
            yield
        finally:
            body_is_raising = sys.exc_info()[0] is not None
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                # Closing the handle below releases the lock.  Preserve a
                # scientifically useful body exception instead of masking it
                # with a secondary shared-filesystem unlock error.
                if not body_is_raising:
                    raise


def safe_remove_job(path: Path, output_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = output_root.resolve()
    if resolved_root not in resolved_path.parents:
        raise RuntimeError(f"Refusing to remove path outside output root: {path}")
    if path.exists():
        shutil.rmtree(path)


def parse_folds(value: str, n_folds: int) -> list[int]:
    if value.strip().lower() == "all":
        return list(range(n_folds))
    folds = [int(item.strip()) for item in value.split(",") if item.strip()]
    invalid = [fold for fold in folds if fold < 0 or fold >= n_folds]
    if not folds or invalid:
        raise ValueError(f"Folds must be in [0, {n_folds - 1}], found {invalid}")
    return sorted(set(folds))


def select_reporters(table: pd.DataFrame, requested: str) -> pd.DataFrame:
    if requested.strip().lower() == "all":
        return table.copy().reset_index(drop=True)
    tokens = [token.strip().casefold() for token in requested.split(",") if token.strip()]
    if not tokens:
        raise ValueError("At least one reporter is required")
    selected: list[pd.Series] = []
    for token in tokens:
        match = table[
            table["reporter_slug"].astype(str).str.casefold().eq(token)
            | table["short_name"].astype(str).str.casefold().eq(token)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Reporter {token!r} matched {len(match)} rows; use an exact slug"
            )
        selected.append(match.iloc[0])
    return (
        pd.DataFrame(selected)
        .drop_duplicates("reporter_slug")
        .reset_index(drop=True)
    )


def load_technical_core_features(
    path: Path, frozen_table: pd.DataFrame
) -> dict[str, list[str]]:
    dictionary = pd.read_csv(path)
    required = {"reporter_slug", "target_feature_name", "selected_in_technical_core"}
    missing = sorted(required - set(dictionary.columns))
    if missing:
        raise RuntimeError(f"Target feature dictionary lacks columns: {missing}")
    if dictionary.duplicated(["reporter_slug", "target_feature_name"]).any():
        raise RuntimeError("Target feature dictionary has duplicate reporter-feature rows")
    selected_column = dictionary["selected_in_technical_core"]
    if pd.api.types.is_bool_dtype(selected_column):
        selected = selected_column.fillna(False)
    else:
        selected = selected_column.astype(str).str.strip().str.casefold().isin(
            {"true", "1", "yes"}
        )
    grouped = {
        str(slug): frame["target_feature_name"].astype(str).tolist()
        for slug, frame in dictionary.loc[selected].groupby(
            "reporter_slug", sort=False
        )
    }
    expected_slugs = set(frozen_table["reporter_slug"].astype(str))
    if set(grouped) != expected_slugs:
        raise RuntimeError(
            "Technical-core reporter set mismatch: "
            f"missing={sorted(expected_slugs - set(grouped))}, "
            f"extra={sorted(set(grouped) - expected_slugs)}"
        )
    for row in frozen_table.itertuples(index=False):
        slug = str(row.reporter_slug)
        if len(grouped[slug]) != int(row.n_technical_core_features):
            raise RuntimeError(
                f"Technical-core count mismatch for {slug}: "
                f"expected {int(row.n_technical_core_features)}, "
                f"found {len(grouped[slug])}"
            )
    return grouped


@dataclass
class MaskedReporterData:
    """Frozen technical-core labels before complete-case filtering.

    Arrays keep exact-H5 row order.  Rows with no observed endpoint are counted
    for audit but excluded from the in-memory supervised view; every retained
    row therefore has at least one real label and never needs imputation.
    """

    slug: str
    source_h5_rows: np.ndarray
    phase_rows: np.ndarray
    y: np.ndarray
    observed_mask: np.ndarray
    is_complete: np.ndarray
    is_control: np.ndarray
    target_feature_names: np.ndarray
    source_cache: Path
    source_size_bytes: int
    source_target_features: int
    n_source_rows: int
    n_complete_rows: int
    n_partial_rows: int
    n_all_missing_rows: int
    all_missing_source_h5_rows: np.ndarray

    @property
    def dropped_nonfinite_target_rows(self) -> int:
        """V1-compatible name: partial plus all-missing technical-core rows."""

        return self.n_partial_rows + self.n_all_missing_rows


def load_masked_reporter_data(
    cache_path: Path,
    slug: str,
    selected_target_feature_names: Iterable[str],
) -> MaskedReporterData:
    """Load exact labels with a true endpoint mask and frozen H5 column order."""

    selected = tuple(str(name) for name in selected_target_feature_names)
    if not selected or len(set(selected)) != len(selected):
        raise RuntimeError(f"Invalid locked target-feature selection for {slug}")
    with h5py.File(cache_path, "r") as source:
        phase_rows_all = np.asarray(source["phase_row_index"][:], dtype=np.int64)
        y_all = np.asarray(source["fluorescence"][:], dtype=np.float32)
        is_control_all = np.asarray(source["metadata/is_control"][:], dtype=bool)
        names_all = decode(source["features/target_feature_names"][:])
    available = set(names_all.astype(str))
    missing = sorted(set(selected) - available)
    if missing:
        raise RuntimeError(
            f"Locked technical-core features are absent from {cache_path}: {missing[:10]}"
        )
    selected_set = set(selected)
    keep_feature = np.asarray(
        [name in selected_set for name in names_all.astype(str)], dtype=bool
    )
    if int(keep_feature.sum()) != len(selected):
        raise RuntimeError(f"Technical-core feature selection is ambiguous for {slug}")
    y_core = np.asarray(y_all[:, keep_feature], dtype=np.float32)
    names_core = names_all[keep_feature]
    observed_all = np.isfinite(y_core)
    any_observed = observed_all.any(axis=1)
    complete_all = observed_all.all(axis=1)
    if len(np.unique(phase_rows_all)) != len(phase_rows_all):
        raise RuntimeError(f"Reporter cache contains duplicate phase rows: {cache_path}")
    if y_core.shape[1] != len(names_core):
        raise RuntimeError(f"Target feature-name mismatch in {cache_path}")
    retained_y = y_core[any_observed]
    retained_mask = observed_all[any_observed]
    # Missing values remain nonfinite in y.  The sole legal conversion to a
    # zero storage sentinel happens in HeadYPreprocessing.transform_masked().
    if np.any(retained_mask & ~np.isfinite(retained_y)):
        raise RuntimeError(f"Observed-mask construction failed for {slug}")
    return MaskedReporterData(
        slug=slug,
        source_h5_rows=np.flatnonzero(any_observed).astype(np.int64),
        phase_rows=phase_rows_all[any_observed],
        y=retained_y,
        observed_mask=retained_mask,
        is_complete=complete_all[any_observed],
        is_control=is_control_all[any_observed],
        target_feature_names=names_core,
        source_cache=cache_path,
        source_size_bytes=cache_path.stat().st_size,
        source_target_features=int(y_all.shape[1]),
        n_source_rows=int(len(y_core)),
        n_complete_rows=int(complete_all.sum()),
        n_partial_rows=int((any_observed & ~complete_all).sum()),
        n_all_missing_rows=int((~any_observed).sum()),
        all_missing_source_h5_rows=np.flatnonzero(~any_observed).astype(np.int64),
    )


@dataclass
class GlobalXPreprocessing:
    kept_indices: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    finite_fraction: np.ndarray
    min_finite_fraction: float
    n_fit_unique_rows: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values[:, self.kept_indices], dtype=np.float32).copy()
        bad = ~np.isfinite(result)
        if np.any(bad):
            rows, columns = np.nonzero(bad)
            result[rows, columns] = self.median[columns]
        result -= self.mean
        result /= self.scale
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "policy": "train_unique_phase_rows_median_impute_zscore",
            "min_finite_fraction": self.min_finite_fraction,
            "n_fit_unique_rows": self.n_fit_unique_rows,
            "n_input_features": int(len(self.finite_fraction)),
            "n_output_features": int(len(self.kept_indices)),
            "kept_indices": self.kept_indices.astype(int).tolist(),
            "finite_fraction": self.finite_fraction.astype(float).tolist(),
            "median": self.median.astype(float).tolist(),
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "GlobalXPreprocessing":
        return cls(
            kept_indices=np.asarray(payload["kept_indices"], dtype=np.int64),
            median=np.asarray(payload["median"], dtype=np.float32),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            finite_fraction=np.asarray(payload["finite_fraction"], dtype=np.float64),
            min_finite_fraction=float(payload["min_finite_fraction"]),
            n_fit_unique_rows=int(payload["n_fit_unique_rows"]),
        )


def fit_global_x_preprocessing(
    phase_matrix: np.ndarray,
    unique_train_rows: np.ndarray,
    min_finite_fraction: float,
    variance_epsilon: float = 1e-8,
) -> GlobalXPreprocessing:
    if len(unique_train_rows) == 0:
        raise RuntimeError("Cannot fit phase preprocessing without training rows")
    if len(np.unique(unique_train_rows)) != len(unique_train_rows):
        raise RuntimeError("Global X preprocessing received duplicate phase rows")
    print(
        f"    materialize {len(unique_train_rows):,} unique training phase rows "
        "for exact train-only preprocessing",
        flush=True,
    )
    values = np.asarray(phase_matrix[unique_train_rows], dtype=np.float32)
    finite = np.isfinite(values)
    finite_fraction = np.mean(finite, axis=0, dtype=np.float64)
    candidate = np.flatnonzero(finite_fraction >= min_finite_fraction)
    if len(candidate) == 0:
        raise RuntimeError("No phase features pass the finite-fraction threshold")
    candidate_values = np.asarray(values[:, candidate], dtype=np.float32).copy()
    candidate_values[~np.isfinite(candidate_values)] = np.nan
    with np.errstate(all="ignore"):
        medians = np.nanmedian(candidate_values, axis=0)
    finite_median = np.isfinite(medians)
    candidate = candidate[finite_median]
    medians = np.asarray(medians[finite_median], dtype=np.float32)
    candidate_values = candidate_values[:, finite_median]
    bad = ~np.isfinite(candidate_values)
    if np.any(bad):
        rows, columns = np.nonzero(bad)
        candidate_values[rows, columns] = medians[columns]
    means = np.mean(candidate_values, axis=0, dtype=np.float64).astype(np.float32)
    scales = np.std(candidate_values, axis=0, dtype=np.float64).astype(np.float32)
    variable = np.isfinite(scales) & (scales > variance_epsilon)
    if not np.any(variable):
        raise RuntimeError("All phase features are constant after train-only imputation")
    state = GlobalXPreprocessing(
        kept_indices=candidate[variable],
        median=medians[variable],
        mean=means[variable],
        scale=scales[variable],
        finite_fraction=finite_fraction,
        min_finite_fraction=min_finite_fraction,
        n_fit_unique_rows=len(unique_train_rows),
    )
    del values, finite, candidate_values
    gc.collect()
    return state


@dataclass
class HeadYPreprocessing:
    kept_indices: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    n_fit_rows: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        result = np.asarray(values[:, self.kept_indices], dtype=np.float32).copy()
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Complete target transform received missing endpoints")
        result -= self.mean
        result /= self.scale
        return result

    def transform_masked(
        self, values: np.ndarray, observed_mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standardize observed endpoints; zero-fill only behind a false mask."""

        selected = np.asarray(values[:, self.kept_indices], dtype=np.float32)
        mask = np.asarray(
            observed_mask[:, self.kept_indices], dtype=bool
        ) & np.isfinite(selected)
        if np.any(~mask.all(axis=1) & ~mask.any(axis=1)):
            raise RuntimeError("All-missing target row reached masked transform")
        result = np.zeros(selected.shape, dtype=np.float32)
        centered = selected - self.mean
        scaled = centered / self.scale
        result[mask] = scaled[mask]
        if not np.all(np.isfinite(result)):
            raise RuntimeError("Masked target transform produced nonfinite storage")
        return result, mask

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32) * self.scale + self.mean

    def to_json(self) -> dict[str, Any]:
        return {
            "policy": "reporter_train_rows_zscore",
            "n_fit_rows": self.n_fit_rows,
            "kept_indices": self.kept_indices.astype(int).tolist(),
            "mean": self.mean.astype(float).tolist(),
            "scale": self.scale.astype(float).tolist(),
            "n_output_dimensions": int(len(self.kept_indices)),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "HeadYPreprocessing":
        return cls(
            kept_indices=np.asarray(payload["kept_indices"], dtype=np.int64),
            mean=np.asarray(payload["mean"], dtype=np.float32),
            scale=np.asarray(payload["scale"], dtype=np.float32),
            n_fit_rows=int(payload["n_fit_rows"]),
        )


def fit_head_y_preprocessing(
    y: np.ndarray, train_indices: np.ndarray, variance_epsilon: float = 1e-8
) -> HeadYPreprocessing:
    values = np.asarray(y[train_indices], dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("Nonfinite target survived strict technical-core filtering")
    means = np.mean(values, axis=0, dtype=np.float64).astype(np.float32)
    scales = np.std(values, axis=0, dtype=np.float64).astype(np.float32)
    keep = np.flatnonzero(np.isfinite(scales) & (scales > variance_epsilon))
    if len(keep) == 0:
        raise RuntimeError("All reporter target dimensions are constant in training")
    return HeadYPreprocessing(
        kept_indices=keep,
        mean=means[keep],
        scale=scales[keep],
        n_fit_rows=len(train_indices),
    )


@dataclass
class HeadPartition:
    slug: str
    reporter_row: pd.Series
    data: Any
    complete_train_indices: np.ndarray
    partial_train_indices: np.ndarray
    active_train_indices: np.ndarray
    complete_validation_indices: np.ndarray
    excluded_partial_validation_indices: np.ndarray
    complete_test_indices: np.ndarray
    excluded_partial_test_indices: np.ndarray
    validation_fold: int
    original_partition_counts: dict[str, int]
    y_preprocessing: HeadYPreprocessing | None = None

    @property
    def train_indices(self) -> np.ndarray:
        return self.active_train_indices

    @property
    def validation_indices(self) -> np.ndarray:
        return self.complete_validation_indices

    @property
    def test_indices(self) -> np.ndarray:
        return self.complete_test_indices

    @property
    def feature_names(self) -> np.ndarray:
        if self.y_preprocessing is None:
            return self.data.target_feature_names
        return self.data.target_feature_names[self.y_preprocessing.kept_indices]


@dataclass
class GPUTrainingStaging:
    """Exact FP32 device cache for split-local preprocessing outputs.

    Staging is an execution optimization only: values are produced by the same
    frozen NumPy preprocessing functions used by the host-fed CUDA path.  It
    does not alter sampling, batches, losses, seeds, or model hyperparameters.
    In particular, disabling staging never moves the model off CUDA.
    """

    phase: Any | None
    targets: dict[str, Any]
    target_masks: dict[str, Any]
    metadata: dict[str, Any]


def build_gpu_training_staging(
    args: argparse.Namespace,
    phase_cache: PhaseCache,
    x_state: GlobalXPreprocessing,
    heads: Sequence[HeadPartition],
    resolved_device: str,
) -> GPUTrainingStaging:
    """Stage standardized matrices without ever changing the model device."""

    disabled = {
        "requested": args.gpu_staging,
        "enabled": False,
        "fallback_reason": None,
        "dtype": "float32",
        "model_device": resolved_device,
        "model_cpu_fallback_allowed": False,
        "data_path": (
            "host_preprocess_cuda_training"
            if resolved_device == "cuda"
            else "host_preprocess_cpu_training"
        ),
    }
    if resolved_device != "cuda" or args.gpu_staging == "off":
        disabled["fallback_reason"] = (
            "non_cuda_device" if resolved_device != "cuda" else "disabled_by_cli"
        )
        return GPUTrainingStaging(None, {}, {}, disabled)

    import torch

    phase_shape = (len(phase_cache.x), len(x_state.kept_indices))
    phase_bytes = int(np.prod(phase_shape, dtype=np.int64)) * 4
    target_value_bytes = 0
    target_mask_bytes = 0
    for head in heads:
        if head.y_preprocessing is None:
            raise RuntimeError(f"Missing target preprocessing for {head.slug}")
        target_value_bytes += (
            len(head.data.y) * len(head.y_preprocessing.kept_indices) * 4
        )
        target_mask_bytes += len(head.data.y) * len(
            head.y_preprocessing.kept_indices
        )
    estimated_bytes = phase_bytes + target_value_bytes + target_mask_bytes
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    reserve_bytes = int(args.gpu_staging_reserve_gib * (1024**3))
    metadata = {
        **disabled,
        "estimated_bytes": estimated_bytes,
        "phase_bytes": phase_bytes,
        "target_bytes": target_value_bytes,
        "target_value_bytes": target_value_bytes,
        "target_mask_bytes": target_mask_bytes,
        "free_bytes_before": int(free_bytes),
        "total_bytes": int(total_bytes),
        "reserve_bytes": reserve_bytes,
        "phase_shape": list(phase_shape),
        "n_target_tensors": len(heads),
        "n_mask_tensors": len(heads),
        "chunk_rows": args.gpu_staging_chunk_rows,
    }
    if estimated_bytes + reserve_bytes > free_bytes:
        reason = (
            "insufficient_free_vram: estimated+reserve="
            f"{estimated_bytes + reserve_bytes}, free={free_bytes}"
        )
        if args.gpu_staging == "required":
            raise RuntimeError(reason)
        metadata["fallback_reason"] = reason
        print(
            "    GPU-resident dataset cache unavailable; using host "
            f"preprocessing with CUDA model training: {reason}",
            flush=True,
        )
        return GPUTrainingStaging(None, {}, {}, metadata)

    phase_tensor = None
    target_tensors: dict[str, Any] = {}
    target_mask_tensors: dict[str, Any] = {}
    started = time.time()
    try:
        phase_tensor = torch.empty(
            phase_shape, dtype=torch.float32, device=resolved_device
        )
        for start in range(0, phase_shape[0], args.gpu_staging_chunk_rows):
            stop = min(start + args.gpu_staging_chunk_rows, phase_shape[0])
            transformed = x_state.transform(
                np.asarray(phase_cache.x[start:stop], dtype=np.float32)
            )
            phase_tensor[start:stop].copy_(torch.from_numpy(transformed))
        for head in heads:
            assert head.y_preprocessing is not None
            transformed, transformed_mask = head.y_preprocessing.transform_masked(
                head.data.y, head.data.observed_mask
            )
            target_tensors[head.slug] = torch.from_numpy(transformed).to(
                resolved_device
            )
            target_mask_tensors[head.slug] = torch.from_numpy(
                transformed_mask
            ).to(resolved_device)
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        metadata.update(
            {
                "enabled": True,
                "fallback_reason": None,
                "data_path": "gpu_resident_preprocessing_and_cuda_training",
                "allocated_bytes_observed": int(free_bytes - free_after),
                "free_bytes_after": int(free_after),
                "seconds": time.time() - started,
            }
        )
        print(
            "    GPU staged exact FP32 phase+target matrices: "
            f"{estimated_bytes / (1024**3):.2f} GiB in "
            f"{metadata['seconds']:.1f}s",
            flush=True,
        )
        metadata["observed_target_elements"] = int(
            sum(mask.sum().item() for mask in target_mask_tensors.values())
        )
        metadata["missing_target_elements"] = int(
            sum((~mask).sum().item() for mask in target_mask_tensors.values())
        )
        return GPUTrainingStaging(
            phase_tensor, target_tensors, target_mask_tensors, metadata
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
        is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(error).casefold()
        )
        if not is_oom or args.gpu_staging == "required":
            raise
        del phase_tensor
        target_tensors.clear()
        target_mask_tensors.clear()
        gc.collect()
        torch.cuda.empty_cache()
        metadata.update(
            {
                "enabled": False,
                "fallback_reason": f"cuda_oom:{type(error).__name__}",
                "seconds": time.time() - started,
            }
        )
        print(
            "    GPU-resident dataset cache hit OOM; retaining CUDA model "
            "training with the original host-preprocessed per-batch input path",
            flush=True,
        )
        return GPUTrainingStaging(None, {}, {}, metadata)


def deterministic_cap(
    indices: np.ndarray, cap: int, seed: int, *tokens: object
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if cap <= 0 or len(indices) <= cap:
        return indices
    rng = np.random.default_rng(stable_seed(seed, *tokens))
    positions = np.sort(rng.choice(len(indices), size=cap, replace=False))
    return indices[positions]


def deterministic_test_cap_with_controls(
    indices: np.ndarray,
    cap: int,
    is_control: np.ndarray,
    screen_codes: np.ndarray,
    seed: int,
    *tokens: object,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if cap <= 0 or len(indices) <= cap:
        return indices
    rng = np.random.default_rng(stable_seed(seed, *tokens))
    local_control = is_control[indices]
    local_screen = screen_codes[indices]
    target_screens = set(np.unique(local_screen[~local_control]).astype(int).tolist())
    control_screens = set(np.unique(local_screen[local_control]).astype(int).tolist())
    eligible_screens = np.asarray(sorted(target_screens & control_screens), dtype=np.int64)
    max_screens = min(len(eligible_screens), cap // 2)
    if max_screens < 1:
        raise RuntimeError("A capped test set needs at least one target/control screen pair")
    if max_screens < len(eligible_screens):
        eligible_screens = np.sort(
            rng.choice(eligible_screens, size=max_screens, replace=False)
        )
    mandatory: list[int] = []
    allowed = np.zeros(len(indices), dtype=bool)
    for screen in eligible_screens:
        screen_mask = local_screen == screen
        controls = np.flatnonzero(screen_mask & local_control)
        targets = np.flatnonzero(screen_mask & ~local_control)
        mandatory.extend(
            [int(rng.choice(controls)), int(rng.choice(targets))]
        )
        allowed |= screen_mask
    mandatory_array = np.unique(np.asarray(mandatory, dtype=np.int64))
    remaining_positions = np.flatnonzero(allowed)
    remaining_positions = np.setdiff1d(
        remaining_positions, mandatory_array, assume_unique=False
    )
    n_fill = min(cap - len(mandatory_array), len(remaining_positions))
    if n_fill > 0:
        fill = rng.choice(remaining_positions, size=n_fill, replace=False)
        selected_positions = np.concatenate((mandatory_array, fill))
    else:
        selected_positions = mandatory_array
    return indices[np.sort(selected_positions)]


def build_partition(
    args: argparse.Namespace,
    phase_cache: PhaseCache,
    reporter_row: pd.Series,
    data: Any,
    split_name: str,
    outer_fold: int,
) -> HeadPartition:
    fold_values = np.asarray(
        phase_cache.folds[split_name][data.phase_rows], dtype=np.uint8
    )
    n_folds = int(phase_cache.manifest["n_folds"])
    validation_fold = (outer_fold + 1) % n_folds
    is_test = fold_values == outer_fold
    is_validation = fold_values == validation_fold
    is_train = ~(is_test | is_validation)
    complete = np.asarray(data.is_complete, dtype=bool)
    partial = ~complete
    complete_train = np.flatnonzero(is_train & complete).astype(np.int64)
    partial_train = np.flatnonzero(is_train & partial).astype(np.int64)
    complete_validation = np.flatnonzero(is_validation & complete).astype(np.int64)
    partial_validation = np.flatnonzero(is_validation & partial).astype(np.int64)
    complete_test = np.flatnonzero(is_test & complete).astype(np.int64)
    partial_test = np.flatnonzero(is_test & partial).astype(np.int64)
    metadata = {
        name: np.asarray(values[data.phase_rows])
        for name, values in phase_cache.metadata.items()
        if name != "is_target"
    }
    # Preserve the exact V1 integrity audit on the complete-core reference
    # cohorts.  Partial rows are audited separately and never cross folds.
    assert_split_integrity(
        split_name,
        complete_train,
        complete_validation,
        complete_test,
        data.is_control,
        metadata["gene_code"],
        data.phase_rows,
        metadata["screen_code"],
        metadata["well_code"],
        metadata["tile_code"],
    )
    slug = str(reporter_row.reporter_slug)
    original_partition_counts = {
        "train": int(len(complete_train)),
        "validation": int(len(complete_validation)),
        "test": int(len(complete_test)),
        "partial_train_eligible": int(len(partial_train)),
        "partial_validation_excluded": int(len(partial_validation)),
        "partial_test_excluded": int(len(partial_test)),
        "train_targeting": int((~data.is_control[complete_train]).sum()),
        "validation_targeting": int((~data.is_control[complete_validation]).sum()),
        "test_targeting": int((~data.is_control[complete_test]).sum()),
        "train_controls": int(data.is_control[complete_train].sum()),
        "validation_controls": int(data.is_control[complete_validation].sum()),
        "test_controls": int(data.is_control[complete_test].sum()),
    }
    complete_train = deterministic_cap(
        complete_train,
        args.max_train_observations_per_head,
        args.seed,
        split_name,
        outer_fold,
        slug,
        "train",
    )
    # A bounded partial sample is added only for smoke caps; formal runs use
    # all eligible partial rows.  This keeps real-data smoke fast while still
    # exercising false mask elements.
    partial_cap = (
        max(1, args.max_train_observations_per_head // 4)
        if args.max_train_observations_per_head > 0
        else 0
    )
    partial_train = deterministic_cap(
        partial_train,
        partial_cap,
        args.seed,
        split_name,
        outer_fold,
        slug,
        "partial_train",
    )
    complete_validation = deterministic_cap(
        complete_validation,
        args.max_validation_observations_per_head,
        args.seed,
        split_name,
        outer_fold,
        slug,
        "validation",
    )
    complete_test = deterministic_test_cap_with_controls(
        complete_test,
        args.max_test_observations_per_head,
        data.is_control,
        metadata["screen_code"],
        args.seed,
        split_name,
        outer_fold,
        slug,
        "test",
    )
    if min(len(complete_train), len(complete_validation), len(complete_test)) == 0:
        raise RuntimeError(f"Caps produced an empty partition for {slug}")
    use_partial = args.arm in {"C", "D"}
    active_train = (
        np.sort(np.concatenate((complete_train, partial_train))).astype(np.int64)
        if use_partial
        else complete_train.copy()
    )
    if use_partial and int(data.n_partial_rows) > 0 and len(partial_train) == 0:
        raise RuntimeError(f"Arm {args.arm} failed to include partial train rows for {slug}")
    if not use_partial and not np.all(data.observed_mask[active_train]):
        raise RuntimeError(f"Complete-only arm {args.arm} contains a masked endpoint")
    if np.any(~data.observed_mask[active_train].any(axis=1)):
        raise RuntimeError(f"All-missing row reached active pool for {slug}")
    return HeadPartition(
        slug=slug,
        reporter_row=reporter_row,
        data=data,
        complete_train_indices=complete_train,
        partial_train_indices=partial_train,
        active_train_indices=active_train,
        complete_validation_indices=complete_validation,
        excluded_partial_validation_indices=partial_validation,
        complete_test_indices=complete_test,
        excluded_partial_test_indices=partial_test,
        validation_fold=validation_fold,
        original_partition_counts=original_partition_counts,
    )


def preprocessing_payload(
    x_state: GlobalXPreprocessing, heads: Sequence[HeadPartition]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "normalization_reference": "complete_case_train_only",
        "partial_rows_used_for_x_fit": False,
        "partial_elements_used_for_y_fit": False,
        "x": x_state.to_json(),
        "heads": {
            head.slug: head.y_preprocessing.to_json()
            for head in heads
            if head.y_preprocessing is not None
        },
    }


def load_preprocessing_payload(
    path: Path, heads: Sequence[HeadPartition], *, allow_v1_schema: bool = False
) -> GlobalXPreprocessing:
    payload = json.loads(path.read_text(encoding="utf-8"))
    allowed_schemas = {SCHEMA_VERSION}
    if allow_v1_schema:
        allowed_schemas.add("ops-reporter-masked-multitask-resmlp-v1")
    if payload.get("schema_version") not in allowed_schemas:
        raise RuntimeError(f"Incompatible preprocessing schema: {path}")
    by_slug = payload.get("heads", {})
    if set(by_slug) != {head.slug for head in heads}:
        raise RuntimeError(f"Preprocessing head set mismatch: {path}")
    for head in heads:
        head.y_preprocessing = HeadYPreprocessing.from_json(by_slug[head.slug])
    return GlobalXPreprocessing.from_json(payload["x"])


def global_partition_integrity(
    heads: Sequence[HeadPartition],
) -> tuple[np.ndarray, dict[str, int]]:
    unique_by_partition: dict[str, np.ndarray] = {}
    for name, attribute in (
        ("normalization_train", "complete_train_indices"),
        ("active_train", "active_train_indices"),
        ("validation", "validation_indices"),
        ("test", "test_indices"),
    ):
        rows = np.concatenate(
            [
                head.data.phase_rows[getattr(head, attribute)]
                for head in heads
            ]
        ).astype(np.int64, copy=False)
        unique_by_partition[name] = np.unique(rows)
    for left, right in (
        ("active_train", "validation"),
        ("active_train", "test"),
        ("validation", "test"),
    ):
        overlap = np.intersect1d(
            unique_by_partition[left], unique_by_partition[right], assume_unique=True
        )
        if len(overlap):
            raise RuntimeError(
                f"Global phase rows cross {left}/{right} boundaries; "
                f"first rows={overlap[:10].astype(int).tolist()}"
            )
    counts = {
        f"n_unique_{name}_phase_rows": int(len(rows))
        for name, rows in unique_by_partition.items()
    }
    if not np.all(
        np.isin(
            unique_by_partition["normalization_train"],
            unique_by_partition["active_train"],
            assume_unique=True,
        )
    ):
        raise RuntimeError("Complete normalization rows are not an active-train subset")
    return unique_by_partition["normalization_train"], counts


def assert_specialist_comparability(
    args: argparse.Namespace,
    head: HeadPartition,
    split_name: str,
    outer_fold: int,
) -> dict[str, Any]:
    phase_rows = np.asarray(head.data.phase_rows[head.test_indices], dtype=np.int64)
    feature_names = head.feature_names.astype(str)
    reference = (
        args.specialist_reference_root
        / split_name
        / f"fold_{outer_fold}"
        / head.slug
        / "shared"
    )
    row_path = reference / "test_phase_row_index.npy"
    feature_path = reference / "target_feature_names.txt"
    if args.max_test_observations_per_head > 0:
        return {
            "strict_specialist_comparable": False,
            "reason": "test cap requested",
            "reference": str(reference.resolve()),
        }
    if not row_path.exists() or not feature_path.exists():
        if args.require_specialist_reference:
            raise FileNotFoundError(
                f"Missing specialist comparison bundle for {head.slug}: {reference}"
            )
        return {
            "strict_specialist_comparable": False,
            "reason": "reference absent",
            "reference": str(reference.resolve()),
        }
    reference_rows = np.load(row_path, mmap_mode="r")
    reference_features = np.asarray(
        [line for line in feature_path.read_text(encoding="utf-8").splitlines() if line],
        dtype=str,
    )
    if not np.array_equal(phase_rows, reference_rows):
        raise RuntimeError(
            f"Test phase rows differ from specialist bundle for "
            f"{split_name}/fold_{outer_fold}/{head.slug}"
        )
    if not np.array_equal(feature_names, reference_features):
        raise RuntimeError(
            f"Target feature order differs from specialist bundle for "
            f"{split_name}/fold_{outer_fold}/{head.slug}"
        )
    return {
        "strict_specialist_comparable": True,
        "reference": str(reference.resolve()),
        "test_phase_row_index_sha256": hash_arrays(phase_rows),
        "target_feature_names_sha256": hashlib.sha256(
            "\n".join(feature_names).encode("utf-8")
        ).hexdigest(),
    }


def load_budget_lock(
    args: argparse.Namespace, split_name: str, outer_fold: int
) -> dict[str, Any]:
    """Resolve the formal optimizer-update window from the frozen V1 marker."""

    if args.budget_policy == "smoke":
        total_smoke_steps = int(args.steps_per_epoch * args.epochs)
        return {
            "policy": "explicit_smoke",
            "source": None,
            "source_sha256": None,
            "steps_per_epoch": int(args.steps_per_epoch),
            "epochs_to_execute": int(args.epochs),
            "optimizer_updates": total_smoke_steps,
            "scheduler_total_steps": total_smoke_steps,
            "warmup_steps": min(
                total_smoke_steps,
                int(args.warmup_epochs * args.steps_per_epoch),
            ),
        }
    source = (
        args.v1_reference_root
        / split_name
        / f"fold_{outer_fold}"
        / "train"
        / "training_complete.json"
    )
    if not source.exists():
        raise FileNotFoundError(f"Missing frozen V1 budget marker: {source}")
    marker = json.loads(source.read_text(encoding="utf-8"))
    steps = int(marker["steps_per_epoch"])
    epochs = int(marker["epochs_completed"])
    updates = int(marker["global_steps_completed"])
    scheduler = marker["scheduler"]
    if updates != steps * epochs:
        raise RuntimeError(f"Frozen V1 update budget is internally inconsistent: {source}")
    return {
        "policy": "frozen_v1_actual_updates",
        "source": str(source.resolve()),
        "source_sha256": file_sha256(source),
        "steps_per_epoch": steps,
        "epochs_to_execute": epochs,
        "optimizer_updates": updates,
        "scheduler_total_steps": int(scheduler["total_optimizer_steps"]),
        "warmup_steps": int(scheduler["warmup_steps"]),
        "v1_best_epoch": int(marker["best_epoch"]),
    }


def model_config(
    args: argparse.Namespace,
    input_dimensions: int,
    head_dimensions: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    high_capacity = args.arm in {"B", "D"}
    expansion_width = 568 if high_capacity else 640
    config: dict[str, Any] = {
        "input_dim": int(input_dimensions),
        "shared_width": args.hidden_width,
        "expansion_width": expansion_width,
        "residual_blocks": args.residual_blocks,
        "dropout": args.dropout,
        "head_width": 56,
        "capacity_policy": (
            "matched_high_dim_heads_128_expansion568"
            if high_capacity
            else "v1_uniform_heads56_expansion640"
        ),
    }
    if head_dimensions is not None:
        widths = {
            slug: (128 if high_capacity and int(dimensions) >= 60 else 56)
            for slug, dimensions in head_dimensions.items()
        }
        config["head_width_by_task"] = widths
        config["head_width"] = widths
    return config


def construct_model(
    library: Any,
    head_dimensions: Mapping[str, int],
    config: Mapping[str, Any],
) -> Any:
    cls = getattr(library, "MaskedMultiTaskResMLP", None)
    if cls is None:
        raise RuntimeError(
            "ops_reporter_masked_multitask_lib lacks MaskedMultiTaskResMLP"
        )
    aliases = {
        "head_dimensions": dict(head_dimensions),
        "input_dim": config["input_dim"],
        "shared_width": config["shared_width"],
        "expansion_width": config["expansion_width"],
        "residual_blocks": config["residual_blocks"],
        "dropout": config["dropout"],
        "head_width": config.get("head_width_by_task", config["head_width"]),
    }
    import inspect

    signature = inspect.signature(cls)
    kwargs: dict[str, Any] = {}
    for name in signature.parameters:
        if name in aliases:
            kwargs[name] = aliases[name]
    if "head_dimensions" not in kwargs:
        raise RuntimeError(
            "MaskedMultiTaskResMLP constructor must accept head_dimensions"
        )
    return cls(**kwargs)


def grouped_forward(
    model: Any, batch_x: Any, task_names: Sequence[str], group_sizes: Sequence[int]
) -> dict[str, Any]:
    if hasattr(model, "forward_grouped"):
        output = model.forward_grouped(batch_x, list(task_names), list(group_sizes))
    else:
        output = model(batch_x, list(task_names), list(group_sizes))
    if isinstance(output, Mapping):
        return dict(output)
    if isinstance(output, (list, tuple)) and len(output) == len(task_names):
        return dict(zip(task_names, output))
    raise RuntimeError("Grouped model forward must return a task-keyed mapping")


def single_head_forward(model: Any, batch_x: Any, slug: str) -> Any:
    output = grouped_forward(model, batch_x, [slug], [len(batch_x)])
    if slug not in output:
        raise RuntimeError(f"Model did not return requested head {slug}")
    return output[slug]


def amp_context(library: Any, device: str, enabled: bool) -> Any:
    import torch

    if hasattr(library, "autocast_context"):
        policy = library.resolve_amp_policy(
            device, "auto" if enabled else "float32"
        )
        return library.autocast_context(policy)
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    from contextlib import nullcontext

    return nullcontext()


def create_grad_scaler(library: Any, device: str, enabled: bool) -> Any:
    import torch

    if hasattr(library, "create_grad_scaler"):
        policy = library.resolve_amp_policy(
            device, "auto" if enabled else "float32"
        )
        return library.create_grad_scaler(policy)
    return torch.cuda.amp.GradScaler(enabled=device == "cuda" and enabled)


def configure_deterministic_torch(args: argparse.Namespace, device: str) -> None:
    import torch

    if device == "cuda":
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "Deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG "
                "to be :4096:8 or :16:8"
            )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")


def assert_model_device(model: Any, expected_device: str) -> None:
    """Fail closed if a model is not wholly resident on the requested device."""

    import torch

    expected_type = torch.device(expected_device).type
    actual_types = {
        tensor.device.type
        for tensor in (*tuple(model.parameters()), *tuple(model.buffers()))
    }
    if not actual_types:
        raise RuntimeError("Cannot verify device placement for an empty model")
    if actual_types != {expected_type}:
        raise RuntimeError(
            f"Model device invariant failed: expected {expected_type}, "
            f"found {sorted(actual_types)}"
        )


def predict_head(
    library: Any,
    model: Any,
    phase_matrix: np.ndarray,
    phase_rows: np.ndarray,
    x_state: GlobalXPreprocessing,
    slug: str,
    output_dimensions: int,
    batch_size: int,
    device: str,
    amp_enabled: bool,
    staged_phase: Any | None = None,
) -> np.ndarray:
    import torch

    result = np.empty((len(phase_rows), output_dimensions), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(phase_rows), batch_size):
            stop = min(start + batch_size, len(phase_rows))
            if staged_phase is None:
                values = x_state.transform(
                    np.asarray(phase_matrix[phase_rows[start:stop]], dtype=np.float32)
                )
                batch_x = torch.from_numpy(values).to(device, non_blocking=True)
            else:
                row_index = torch.as_tensor(
                    phase_rows[start:stop], dtype=torch.long, device=device
                )
                batch_x = staged_phase.index_select(0, row_index)
            with amp_context(library, device, amp_enabled):
                prediction = single_head_forward(model, batch_x, slug)
            result[start:stop] = (
                prediction.detach().float().cpu().numpy().astype(np.float32)
            )
    return result


def train_one_job(
    args: argparse.Namespace,
    library: Any,
    phase_cache: PhaseCache,
    heads: Sequence[HeadPartition],
    x_state: GlobalXPreprocessing,
    train_dir: Path,
    split_name: str,
    outer_fold: int,
    run_fingerprint: str,
    head_schema_sha256: str,
    job_manifest_sha256: str,
    preprocessing_sha256: str,
    gpu_preflight_sha256: str | None,
    resolved_device: str,
    gpu_staging: GPUTrainingStaging,
    budget_lock: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch

    train_dir.mkdir(parents=True, exist_ok=True)
    best_path = train_dir / "best.pt"
    last_path = train_dir / "last.pt"
    complete_path = train_dir / "training_complete.json"
    history_path = train_dir / "history.csv"
    job_manifest_payload = json.loads(
        (train_dir.parent / "job_manifest.json").read_text(encoding="utf-8")
    )
    if complete_path.exists() and not args.overwrite:
        marker = json.loads(complete_path.read_text(encoding="utf-8"))
        if marker.get("job_manifest_sha256") != job_manifest_sha256:
            raise RuntimeError(f"Completed training marker is incompatible: {complete_path}")
        if not best_path.exists() or file_sha256(best_path) != marker.get("checkpoint_sha256"):
            raise RuntimeError(f"Completed checkpoint hash mismatch: {best_path}")
        model, _ = library.load_model_checkpoint(
            best_path, device=resolved_device, strict=True
        )
        assert_model_device(model, resolved_device)
        return model, marker

    job_seed = stable_seed(args.seed, split_name, outer_fold, SEED_TOKEN)
    torch.manual_seed(job_seed)
    np.random.seed(job_seed)
    random.seed(job_seed)
    if resolved_device == "cuda":
        torch.cuda.manual_seed_all(job_seed)
    else:
        torch.set_num_threads(max(1, args.workers))

    head_dimensions = {
        head.slug: int(head.data.y.shape[1]) for head in heads
    }
    config = model_config(args, len(x_state.kept_indices), head_dimensions)
    model = construct_model(library, head_dimensions, config).to(resolved_device)
    assert_model_device(model, resolved_device)
    parameter_budget = library.parameter_budget_report(model)
    if len(head_dimensions) == 52 and sum(head_dimensions.values()) == 1604:
        parameter_budget = library.assert_parameter_budget(model)
        expected_parameters = 4_310_636 if args.arm in {"B", "D"} else 4_310_852
        if parameter_budget["actual_joint_parameters"] != expected_parameters:
            raise RuntimeError(
                f"Arm {args.arm} parameter count mismatch: "
                f"{parameter_budget['actual_joint_parameters']} != {expected_parameters}"
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scaler = create_grad_scaler(library, resolved_device, args.amp)
    task_groups_per_round = math.ceil(
        len(heads) / min(args.batch_heads, len(heads))
    )
    locked_steps_per_epoch = int(budget_lock["steps_per_epoch"])
    if locked_steps_per_epoch <= 0:
        raise RuntimeError("Budget lock must specify positive steps_per_epoch")
    requested_rounds = math.ceil(
        locked_steps_per_epoch / task_groups_per_round
    )
    epoch_samples_per_task = requested_rounds * args.observations_per_head
    sampler = library.DeterministicCyclicTaskSampler(
        {head.slug: len(head.train_indices) for head in heads},
        tasks_per_batch=min(args.batch_heads, len(heads)),
        samples_per_task=args.observations_per_head,
        epoch_samples_per_task=epoch_samples_per_task,
        seed=job_seed,
    )
    steps_per_epoch = len(sampler)
    if steps_per_epoch != locked_steps_per_epoch:
        raise RuntimeError(
            f"Sampler produces {steps_per_epoch} steps, budget locks "
            f"{locked_steps_per_epoch}"
        )
    epochs_to_execute = int(budget_lock["epochs_to_execute"])
    total_optimizer_steps = int(budget_lock["scheduler_total_steps"])
    warmup_steps = int(budget_lock["warmup_steps"])
    minimum_lr_ratio = args.minimum_learning_rate / args.learning_rate

    def learning_rate_multiplier(step_index: int) -> float:
        if warmup_steps > 0 and step_index < warmup_steps:
            return max(1.0 / warmup_steps, (step_index + 1) / warmup_steps)
        denominator = max(1, total_optimizer_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step_index - warmup_steps) / denominator))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_multiplier
    )
    history: list[dict[str, Any]] = []
    start_epoch = 1
    best_epoch = 0
    best_validation = math.inf
    epochs_without_improvement = 0
    previous_runtime = 0.0
    global_step = 0
    resumed_epoch_state: dict[str, Any] | None = None
    if args.resume and last_path.exists():
        checkpoint = library.load_training_checkpoint(
            last_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=scaler,
            sampler=sampler,
            map_location=resolved_device,
            strict=True,
            restore_rng=True,
        )
        extra = checkpoint.get("extra", {})
        required = {
            "run_fingerprint_sha256": run_fingerprint,
            "job_manifest_sha256": job_manifest_sha256,
            "preprocessing_sha256": preprocessing_sha256,
            "head_schema_sha256": head_schema_sha256,
            "gpu_preflight_sha256": gpu_preflight_sha256,
            "arm": args.arm,
            "budget_lock_sha256": job_manifest_payload.get(
                "budget_lock_sha256"
            ),
        }
        for key, expected in required.items():
            if extra.get(key) != expected:
                raise RuntimeError(
                    f"Resume checkpoint {key} mismatch in {last_path}; use a new root"
                )
        training_state = checkpoint.get("training_state", {})
        history = list(training_state.get("history", []))
        completed_epoch = int(training_state.get("completed_epoch", 0))
        resumed_epoch_state = training_state.get("in_progress_epoch")
        start_epoch = (
            int(resumed_epoch_state["epoch"])
            if resumed_epoch_state is not None
            else completed_epoch + 1
        )
        best_epoch = int(training_state.get("best_epoch", 0))
        best_validation = float(
            training_state.get("best_validation_macro_mse", math.inf)
        )
        epochs_without_improvement = int(
            training_state.get("epochs_without_improvement", 0)
        )
        previous_runtime = float(training_state.get("runtime_seconds", 0.0))
        global_step = int(training_state.get("global_step", 0))
        print(f"    resume training at epoch {start_epoch}", flush=True)

    by_slug = {head.slug: head for head in heads}

    started = time.time()
    # Patience is recorded for diagnostics but never changes the preregistered
    # actual-update window.  Best checkpoint selection still uses the same
    # complete-core validation macro MSE as V1.
    epoch_range: Iterable[int] = range(start_epoch, epochs_to_execute + 1)
    for epoch in epoch_range:
        epoch_started = time.time()
        model.train()
        if resumed_epoch_state is not None and int(resumed_epoch_state["epoch"]) == epoch:
            running_head_loss = {
                slug: float(value)
                for slug, value in resumed_epoch_state["running_head_loss"].items()
            }
            running_head_count = {
                slug: int(value)
                for slug, value in resumed_epoch_state["running_head_count"].items()
            }
            epoch_prior_seconds = float(
                resumed_epoch_state.get("elapsed_seconds", 0.0)
            )
        else:
            running_head_loss = {head.slug: 0.0 for head in heads}
            running_head_count = {head.slug: 0 for head in heads}
            epoch_prior_seconds = 0.0
        if sampler.epoch != epoch - 1:
            raise RuntimeError(
                f"Sampler epoch {sampler.epoch} is incompatible with training epoch {epoch}"
            )
        for task_batch in sampler:
            step = task_batch.batch_index
            selected_slugs = list(task_batch.task_names)
            sampled_phase_rows: list[np.ndarray] = []
            target_parts: dict[str, np.ndarray] = {}
            target_masks_host: dict[str, np.ndarray] = {}
            local_indices_by_slug: dict[str, np.ndarray] = {}
            group_sizes: list[int] = []
            for slug, chosen_positions in zip(
                selected_slugs, task_batch.task_indices
            ):
                head = by_slug[slug]
                local_indices = head.train_indices[chosen_positions]
                rows = head.data.phase_rows[local_indices]
                sampled_phase_rows.append(np.asarray(rows, dtype=np.int64))
                assert head.y_preprocessing is not None
                local_indices_by_slug[slug] = np.asarray(
                    local_indices, dtype=np.int64
                )
                if not gpu_staging.metadata.get("enabled"):
                    transformed, transformed_mask = (
                        head.y_preprocessing.transform_masked(
                            head.data.y[local_indices],
                            head.data.observed_mask[local_indices],
                        )
                    )
                    target_parts[slug] = transformed
                    target_masks_host[slug] = transformed_mask
                group_sizes.append(len(local_indices))
            concatenated_rows = np.concatenate(sampled_phase_rows)
            unique_rows, inverse = np.unique(
                concatenated_rows, return_inverse=True
            )
            if gpu_staging.phase is None:
                unique_x_np = x_state.transform(
                    np.asarray(phase_cache.x[unique_rows], dtype=np.float32)
                )
                unique_x = torch.from_numpy(unique_x_np).to(
                    resolved_device, non_blocking=True
                )
            else:
                unique_row_index = torch.as_tensor(
                    unique_rows, dtype=torch.long, device=resolved_device
                )
                unique_x = gpu_staging.phase.index_select(0, unique_row_index)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(library, resolved_device, args.amp):
                encoded = model.encode(unique_x)
                predictions: dict[str, Any] = {}
                targets: dict[str, Any] = {}
                masks: dict[str, Any] = {}
                offset = 0
                for slug in selected_slugs:
                    head = by_slug[slug]
                    assert head.y_preprocessing is not None
                    group_size = group_sizes[selected_slugs.index(slug)]
                    route = torch.as_tensor(
                        inverse[offset : offset + group_size],
                        dtype=torch.long,
                        device=resolved_device,
                    )
                    offset += group_size
                    keep = torch.as_tensor(
                        head.y_preprocessing.kept_indices,
                        dtype=torch.long,
                        device=resolved_device,
                    )
                    prediction_full = model.predict_encoded(
                        encoded.index_select(0, route), slug
                    )
                    predictions[slug] = prediction_full.index_select(1, keep)
                    if gpu_staging.metadata.get("enabled"):
                        local_index = torch.as_tensor(
                            local_indices_by_slug[slug],
                            dtype=torch.long,
                            device=resolved_device,
                        )
                        target = gpu_staging.targets[slug].index_select(
                            0, local_index
                        )
                        target_mask = gpu_staging.target_masks[slug].index_select(
                            0, local_index
                        )
                    else:
                        target = torch.from_numpy(target_parts[slug]).to(
                            resolved_device, non_blocking=True
                        )
                        target_mask = torch.from_numpy(
                            target_masks_host[slug]
                        ).to(resolved_device, non_blocking=True)
                    targets[slug] = target
                    masks[slug] = target_mask
                    if not bool(target_mask.any(dim=1).all().item()):
                        raise RuntimeError(f"All-missing sampled row for {slug}")
                    if args.arm in {"A", "B"} and not bool(
                        target_mask.all().item()
                    ):
                        raise RuntimeError(
                            f"Complete-only arm {args.arm} sampled a missing endpoint"
                        )
                loss, head_losses = library.masked_balanced_mse(
                    predictions,
                    targets,
                    masks=masks,
                    return_per_task=True,
                )
                for slug, loss_h in head_losses.items():
                    running_head_loss[slug] += float(loss_h.detach().float().cpu())
                    running_head_count[slug] += 1
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(
                    f"Nonfinite training loss at epoch={epoch}, step={step}"
                )
            scaler.scale(loss).backward()
            if args.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip_norm
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            if global_step % args.checkpoint_every_steps == 0:
                in_progress = {
                    "epoch": epoch,
                    "sampler_batch_cursor": sampler.next_batch_index,
                    "running_head_loss": running_head_loss,
                    "running_head_count": running_head_count,
                    "elapsed_seconds": (
                        epoch_prior_seconds + time.time() - epoch_started
                    ),
                }
                periodic_state = {
                    "completed_epoch": epoch - 1,
                    "in_progress_epoch": in_progress,
                    "best_epoch": best_epoch,
                    "best_validation_macro_mse": best_validation,
                    "epochs_without_improvement": epochs_without_improvement,
                    "history": history,
                    "global_step": global_step,
                    "runtime_seconds": previous_runtime + time.time() - started,
                }
                periodic_extra = {
                    "runner_schema_version": SCHEMA_VERSION,
                    "model": MODEL_NAME,
                    "arm": args.arm,
                    "run_fingerprint_sha256": run_fingerprint,
                    "head_schema_sha256": head_schema_sha256,
                    "job_manifest_sha256": job_manifest_sha256,
                    "preprocessing_sha256": preprocessing_sha256,
                    "gpu_preflight_sha256": gpu_preflight_sha256,
                    "parameter_budget": parameter_budget,
                    "budget_lock_sha256": job_manifest_payload.get(
                        "budget_lock_sha256"
                    ),
                }
                library.save_training_checkpoint(
                    last_path,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    grad_scaler=scaler,
                    sampler=sampler,
                    training_state=periodic_state,
                    extra=periodic_extra,
                )

        validation_by_head: dict[str, float] = {}
        for head in heads:
            assert head.y_preprocessing is not None
            rows = head.data.phase_rows[head.validation_indices]
            prediction_full = predict_head(
                library,
                model,
                phase_cache.x,
                rows,
                x_state,
                head.slug,
                head.data.y.shape[1],
                args.prediction_batch_size,
                resolved_device,
                args.amp,
                gpu_staging.phase,
            )
            prediction = prediction_full[:, head.y_preprocessing.kept_indices]
            truth = head.y_preprocessing.transform(
                head.data.y[head.validation_indices]
            )
            mse = float(np.mean((truth - prediction) ** 2))
            if not np.isfinite(mse):
                raise RuntimeError(f"Nonfinite validation MSE for {head.slug}")
            validation_by_head[head.slug] = mse
        validation_macro = float(np.mean(list(validation_by_head.values())))
        train_by_head = {
            slug: running_head_loss[slug] / max(running_head_count[slug], 1)
            for slug in running_head_loss
        }
        epoch_row = {
            "epoch": epoch,
            "steps": steps_per_epoch,
            "train_macro_sampled_mse": float(np.mean(list(train_by_head.values()))),
            "validation_macro_mse": validation_macro,
            "seconds": epoch_prior_seconds + time.time() - epoch_started,
            "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
            "train_mse_by_head": train_by_head,
            "validation_mse_by_head": validation_by_head,
        }
        history.append(epoch_row)
        improved = validation_macro < best_validation - args.min_delta
        if improved:
            best_validation = validation_macro
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        training_state = {
            "completed_epoch": epoch,
            "in_progress_epoch": None,
            "best_epoch": best_epoch,
            "best_validation_macro_mse": best_validation,
            "epochs_without_improvement": epochs_without_improvement,
            "history": history,
            "global_step": global_step,
            "runtime_seconds": previous_runtime + time.time() - started,
        }
        checkpoint_extra = {
            "runner_schema_version": SCHEMA_VERSION,
            "model": MODEL_NAME,
            "arm": args.arm,
            "run_fingerprint_sha256": run_fingerprint,
            "head_schema_sha256": head_schema_sha256,
            "job_manifest_sha256": job_manifest_sha256,
            "preprocessing_sha256": preprocessing_sha256,
            "gpu_preflight_sha256": gpu_preflight_sha256,
            "parameter_budget": parameter_budget,
            "budget_lock_sha256": job_manifest_payload.get(
                "budget_lock_sha256"
            ),
        }
        # Commit a newly selected best checkpoint before advancing ``last``.
        # If the process dies between the two atomic writes, the older last
        # checkpoint deterministically replays this epoch instead of claiming
        # a best state that was never published.
        if improved:
            library.save_training_checkpoint(
                best_path,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                grad_scaler=scaler,
                sampler=sampler,
                training_state=training_state,
                extra=checkpoint_extra,
            )
        library.save_training_checkpoint(
            last_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_scaler=scaler,
            sampler=sampler,
            training_state=training_state,
            extra=checkpoint_extra,
        )
        history_frame = pd.DataFrame(
            [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"train_mse_by_head", "validation_mse_by_head"}
                }
                for row in history
            ]
        )
        atomic_csv(history_path, history_frame)
        print(
            f"    epoch {epoch}: train macro={epoch_row['train_macro_sampled_mse']:.6f}; "
            f"validation macro={validation_macro:.6f}; best={best_validation:.6f}",
            flush=True,
        )
        # Deliberately no patience break: all arms execute the same lock.

    if not best_path.exists() or best_epoch < 1:
        raise RuntimeError("Training did not produce a finite best checkpoint")
    expected_updates = int(budget_lock["optimizer_updates"])
    if global_step != expected_updates:
        raise RuntimeError(
            f"Actual optimizer updates {global_step} differ from lock {expected_updates}"
        )
    model, best_checkpoint = library.load_model_checkpoint(
        best_path, device=resolved_device, strict=True
    )
    assert_model_device(model, resolved_device)
    checkpoint_sha = file_sha256(best_path)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "outer_fold": outer_fold,
        "model": MODEL_NAME,
        "arm": args.arm,
        "device_requested": args.device,
        "device_resolved": resolved_device,
        "n_heads": len(heads),
        "reporter_slugs": [head.slug for head in heads],
        "checkpoint_path": str(best_path.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "head_schema_sha256": head_schema_sha256,
        "job_manifest_sha256": job_manifest_sha256,
        "preprocessing_sha256": preprocessing_sha256,
        "v1_preprocessing_sha256": job_manifest_payload.get(
            "v1_preprocessing_sha256"
        ),
        "preprocessing_reference_sha256": job_manifest_payload.get(
            "preprocessing_reference_sha256"
        ),
        "cohort_hashes": job_manifest_payload.get("cohort_hashes"),
        "complete_core_cohort_hashes": job_manifest_payload.get("cohort_hashes"),
        "budget_lock": dict(budget_lock),
        "budget_lock_sha256": job_manifest_payload.get("budget_lock_sha256"),
        "gpu_preflight_sha256": gpu_preflight_sha256,
        "run_fingerprint_sha256": run_fingerprint,
        "runtime_seconds": previous_runtime + time.time() - started,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_macro_mse": best_validation,
        "steps_per_epoch": steps_per_epoch,
        "global_steps_completed": global_step,
        "warmup_steps": warmup_steps,
        "minimum_learning_rate": args.minimum_learning_rate,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "amp_enabled": args.amp,
        "amp_policy": library.resolve_amp_policy(
            resolved_device, "auto" if args.amp else "float32"
        ).to_dict(),
        "parameter_budget": parameter_budget,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": args.weight_decay,
        },
        "scheduler": {
            "name": "linear_warmup_cosine_decay",
            "warmup_steps": warmup_steps,
            "total_optimizer_steps": total_optimizer_steps,
            "minimum_learning_rate": args.minimum_learning_rate,
        },
        "checkpoint_schema_version": best_checkpoint.get("schema_version"),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_device_name": (
            torch.cuda.get_device_name(0) if resolved_device == "cuda" else None
        ),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_matmul_enabled": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_enabled": bool(torch.backends.cudnn.allow_tf32),
        "gpu_staging": gpu_staging.metadata,
    }
    atomic_json(complete_path, marker)
    return model, marker


def load_completed_model(
    library: Any, marker: Mapping[str, Any], resolved_device: str
) -> Any:
    path = Path(str(marker["checkpoint_path"]))
    if not path.exists() or file_sha256(path) != marker["checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint absent or hash mismatch: {path}")
    model, _ = library.load_model_checkpoint(
        path, device=resolved_device, strict=True
    )
    assert_model_device(model, resolved_device)
    return model


def existing_evaluation_is_complete(
    args: argparse.Namespace,
    evaluation_dir: Path,
    marker: Mapping[str, Any],
    head: HeadPartition,
    training_marker: Mapping[str, Any],
) -> bool:
    expected_references = {
        "checkpoint_sha256": training_marker["checkpoint_sha256"],
        "head_schema_sha256": training_marker["head_schema_sha256"],
        "job_manifest_sha256": training_marker["job_manifest_sha256"],
        "preprocessing_sha256": training_marker["preprocessing_sha256"],
        "gpu_preflight_sha256": training_marker.get("gpu_preflight_sha256"),
        "run_fingerprint_sha256": training_marker["run_fingerprint_sha256"],
        "target_feature_names": head.feature_names.astype(str).tolist(),
        "evaluator_id": cell_eval_adapter.EVALUATOR_ID,
    }
    if any(marker.get(key) != value for key, value in expected_references.items()):
        return False
    n_rows = len(head.test_indices)
    n_outputs = len(head.feature_names)
    required = {
        "test_phase_row_index.npy": (n_rows,),
        "test_is_control.npy": (n_rows,),
        "truth_raw.npy": (n_rows, n_outputs),
        "gene_prediction_control_relative_scaled.npy": None,
        "gene_truth_control_relative_scaled.npy": None,
        "gene_screen_code.npy": None,
        "gene_gene_code.npy": None,
        "gene_n_cells.npy": None,
        "gene_n_control_cells.npy": None,
        "cell_eval_ceiling_half_a_control_relative_scaled.npy": None,
        "cell_eval_ceiling_half_b_control_relative_scaled.npy": None,
    }
    if args.save_predictions:
        required["test_prediction_raw.npy"] = (n_rows, n_outputs)
    try:
        for name, expected_shape in required.items():
            path = evaluation_dir / name
            if not path.exists():
                return False
            values = np.load(path, mmap_mode="r")
            if expected_shape is not None and tuple(values.shape) != expected_shape:
                return False
            if values.dtype.kind == "f" and not np.all(np.isfinite(values)):
                return False
        expected_rows = head.data.phase_rows[head.test_indices]
        if not np.array_equal(
            np.load(evaluation_dir / "test_phase_row_index.npy", mmap_mode="r"),
            expected_rows,
        ):
            return False
        feature_path = evaluation_dir / "target_feature_names.txt"
        if not feature_path.exists():
            return False
        feature_names = [
            line for line in feature_path.read_text(encoding="utf-8").splitlines() if line
        ]
        if feature_names != head.feature_names.astype(str).tolist():
            return False
        for name in ("cell_feature_metrics.csv", "gene_feature_metrics.csv"):
            path = evaluation_dir / name
            if not path.exists():
                return False
            frame = pd.read_csv(path)
            if len(frame) != n_outputs or frame["feature"].astype(str).tolist() != feature_names:
                return False
        n_groups = int(marker.get("gene_metrics", {}).get("n_gene_screen_groups", -1))
        if n_groups < 1:
            return False
        for name in ("gene_group_metrics.csv", "cell_eval_ceiling_group_metrics.csv"):
            path = evaluation_dir / name
            if not path.exists() or len(pd.read_csv(path)) != n_groups:
                return False
        adapter_path = evaluation_dir / "cell_eval_adapter_manifest.json"
        if not adapter_path.exists():
            return False
        cell_eval_adapter.validate_manifest(
            json.loads(adapter_path.read_text(encoding="utf-8"))
        )
        artifact_hashes = marker.get("artifact_sha256")
        if not isinstance(artifact_hashes, Mapping):
            return False
        for name, expected_hash in artifact_hashes.items():
            path = evaluation_dir / str(name)
            if not path.exists() or file_sha256(path) != expected_hash:
                return False
    except Exception:
        return False
    return True


def evaluate_head(
    args: argparse.Namespace,
    library: Any,
    model: Any,
    phase_cache: PhaseCache,
    x_state: GlobalXPreprocessing,
    head: HeadPartition,
    evaluation_dir: Path,
    split_name: str,
    outer_fold: int,
    training_marker: Mapping[str, Any],
    comparability: Mapping[str, Any],
    gpu_staging: GPUTrainingStaging,
) -> dict[str, Any]:
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    (evaluation_dir / "evaluation_failed.json").unlink(missing_ok=True)
    complete_path = evaluation_dir / "evaluation_complete.json"
    if complete_path.exists() and not args.overwrite:
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        if existing_evaluation_is_complete(
            args, evaluation_dir, existing, head, training_marker
        ):
            return existing
        complete_path.unlink()
    started = time.time()
    assert head.y_preprocessing is not None
    test_indices = head.test_indices
    phase_rows = np.asarray(head.data.phase_rows[test_indices], dtype=np.int64)
    prediction_full = predict_head(
        library,
        model,
        phase_cache.x,
        phase_rows,
        x_state,
        head.slug,
        head.data.y.shape[1],
        args.prediction_batch_size,
        str(training_marker["device_resolved"]),
        args.amp,
        gpu_staging.phase,
    )
    prediction_scaled = prediction_full[:, head.y_preprocessing.kept_indices]
    if not np.all(np.isfinite(prediction_scaled)):
        raise RuntimeError(f"Nonfinite test prediction for {head.slug}")
    prediction_raw = head.y_preprocessing.inverse(prediction_scaled)
    truth_raw = np.asarray(
        head.data.y[test_indices][:, head.y_preprocessing.kept_indices],
        dtype=np.float32,
    )
    truth_scaled = head.y_preprocessing.transform(head.data.y[test_indices])
    is_control = np.asarray(head.data.is_control[test_indices], dtype=bool)
    target_test = ~is_control
    if not np.any(target_test):
        raise RuntimeError(f"No targeting test cells for {head.slug}")
    baseline_raw = np.broadcast_to(
        head.y_preprocessing.mean, truth_raw[target_test].shape
    )
    cell_summary, cell_features = matrix_metrics(
        truth_raw[target_test],
        prediction_raw[target_test],
        baseline_raw,
        head.feature_names,
    )
    cell_summary["standardized_mse"] = float(
        np.mean((truth_scaled[target_test] - prediction_scaled[target_test]) ** 2)
    )
    standardized_baseline_mse = float(np.mean(truth_scaled[target_test] ** 2))
    cell_summary["standardized_baseline_mse"] = standardized_baseline_mse
    cell_summary["standardized_gain_vs_train_mean"] = float(
        1.0 - cell_summary["standardized_mse"] / standardized_baseline_mse
    )
    cell_summary["n_targeting_test_cells"] = int(target_test.sum())

    screen_codes = np.asarray(phase_cache.metadata["screen_code"][phase_rows])
    gene_codes = np.asarray(phase_cache.metadata["gene_code"][phase_rows])
    profiles = aggregate_gene_profiles(
        truth_scaled,
        prediction_scaled,
        screen_codes,
        gene_codes,
        is_control,
    )
    gene_summary, gene_features = profile_metrics(profiles, head.feature_names)
    gene_summary.update(
        bootstrap_gene_profile_metrics(
            profiles,
            args.bootstrap_draws,
            stable_seed(args.seed, split_name, outer_fold, head.slug, "bootstrap"),
        )
    )
    adapted_summary, gene_group_metrics = (
        cell_eval_adapter.phenotype_perturbation_metrics(profiles)
    )
    gene_summary.update(adapted_summary)
    ceiling_seed = stable_seed(
        args.seed, split_name, outer_fold, head.slug, "cell_eval_data_ceiling"
    )
    ceiling_profiles = cell_eval_adapter.bootstrap_data_ceiling_profiles(
        truth_scaled,
        screen_codes,
        gene_codes,
        is_control,
        ceiling_seed,
    )
    if not (
        np.array_equal(ceiling_profiles.screen_code, profiles.screen_code)
        and np.array_equal(ceiling_profiles.gene_code, profiles.gene_code)
    ):
        raise RuntimeError(f"Data-ceiling profile order drifted for {head.slug}")
    ceiling_summary, ceiling_group_metrics = (
        cell_eval_adapter.phenotype_perturbation_metrics(ceiling_profiles)
    )
    gene_summary.update(
        {
            f"cell_eval_ceiling_{key.removeprefix('cell_eval_')}": value
            for key, value in ceiling_summary.items()
        }
    )
    adapter_manifest = cell_eval_adapter.adapter_manifest()
    adapter_manifest["binding"] = {
        "reporter_slug": head.slug,
        "split": split_name,
        "outer_fold": outer_fold,
        "ceiling_seed": ceiling_seed,
        "n_gene_screen_groups": int(len(profiles.truth)),
        "n_endpoints": int(len(head.feature_names)),
        "outer_test_only": True,
        "used_for_training_checkpoint_or_state_selection": False,
    }
    cell_eval_adapter.validate_manifest(adapter_manifest)

    atomic_npy(evaluation_dir / "test_phase_row_index.npy", phase_rows)
    atomic_npy(evaluation_dir / "test_is_control.npy", is_control)
    atomic_npy(evaluation_dir / "truth_raw.npy", truth_raw)
    atomic_npy(
        evaluation_dir / "gene_prediction_control_relative_scaled.npy",
        profiles.prediction,
    )
    atomic_npy(
        evaluation_dir / "gene_truth_control_relative_scaled.npy", profiles.truth
    )
    atomic_npy(evaluation_dir / "gene_screen_code.npy", profiles.screen_code)
    atomic_npy(evaluation_dir / "gene_gene_code.npy", profiles.gene_code)
    atomic_npy(evaluation_dir / "gene_n_cells.npy", profiles.n_cells)
    atomic_npy(
        evaluation_dir / "gene_n_control_cells.npy", profiles.n_control_cells
    )
    atomic_npy(
        evaluation_dir / "cell_eval_ceiling_half_a_control_relative_scaled.npy",
        ceiling_profiles.truth,
    )
    atomic_npy(
        evaluation_dir / "cell_eval_ceiling_half_b_control_relative_scaled.npy",
        ceiling_profiles.prediction,
    )
    if args.save_predictions:
        atomic_npy(evaluation_dir / "test_prediction_raw.npy", prediction_raw)
    atomic_text(
        evaluation_dir / "target_feature_names.txt",
        "\n".join(head.feature_names.astype(str)) + "\n",
    )
    atomic_csv(evaluation_dir / "cell_feature_metrics.csv", cell_features)
    atomic_csv(evaluation_dir / "gene_feature_metrics.csv", gene_features)
    atomic_csv(evaluation_dir / "gene_group_metrics.csv", gene_group_metrics)
    atomic_csv(
        evaluation_dir / "cell_eval_ceiling_group_metrics.csv",
        ceiling_group_metrics,
    )
    atomic_json(evaluation_dir / "cell_eval_adapter_manifest.json", adapter_manifest)
    artifact_names = [
        "test_phase_row_index.npy",
        "test_is_control.npy",
        "truth_raw.npy",
        "gene_prediction_control_relative_scaled.npy",
        "gene_truth_control_relative_scaled.npy",
        "gene_screen_code.npy",
        "gene_gene_code.npy",
        "gene_n_cells.npy",
        "gene_n_control_cells.npy",
        "cell_eval_ceiling_half_a_control_relative_scaled.npy",
        "cell_eval_ceiling_half_b_control_relative_scaled.npy",
        "target_feature_names.txt",
        "cell_feature_metrics.csv",
        "gene_feature_metrics.csv",
        "gene_group_metrics.csv",
        "cell_eval_ceiling_group_metrics.csv",
        "cell_eval_adapter_manifest.json",
    ]
    if args.save_predictions:
        artifact_names.append("test_prediction_raw.npy")
    artifact_hashes = {
        name: file_sha256(evaluation_dir / name) for name in artifact_names
    }
    v1_evaluation_path = (
        args.v1_reference_root
        / split_name
        / f"fold_{outer_fold}"
        / "reporters"
        / head.slug
        / "evaluation_complete.json"
    )
    v1_test_fingerprint: str | None = None
    if args.budget_policy == "v1_locked":
        if not v1_evaluation_path.exists():
            raise FileNotFoundError(f"Missing V1 evaluation reference: {v1_evaluation_path}")
        v1_evaluation = json.loads(v1_evaluation_path.read_text(encoding="utf-8"))
        v1_test_fingerprint = str(v1_evaluation["test_split_fingerprint_sha256"])
        v1_dir = v1_evaluation_path.parent
        if not np.array_equal(phase_rows, np.load(v1_dir / "test_phase_row_index.npy")):
            raise RuntimeError(f"V2/V1 test phase-row mismatch for {head.slug}")
        if not np.array_equal(is_control, np.load(v1_dir / "test_is_control.npy")):
            raise RuntimeError(f"V2/V1 test control-mask mismatch for {head.slug}")
        if not np.array_equal(truth_raw, np.load(v1_dir / "truth_raw.npy")):
            raise RuntimeError(f"V2/V1 test truth mismatch for {head.slug}")
    evaluation_runtime = time.time() - started
    amortized_training_runtime = float(
        training_marker["runtime_seconds"] / training_marker["n_heads"]
    )
    marker = {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "outer_fold": outer_fold,
        "model": MODEL_NAME,
        "arm": args.arm,
        "reporter_slug": head.slug,
        "reporter_short_name": str(head.reporter_row.short_name),
        "biological_category": str(head.reporter_row.biological_category),
        "device_resolved": training_marker["device_resolved"],
        "checkpoint_path": training_marker["checkpoint_path"],
        "checkpoint_sha256": training_marker["checkpoint_sha256"],
        "head_schema_sha256": training_marker["head_schema_sha256"],
        "job_manifest_sha256": training_marker["job_manifest_sha256"],
        "preprocessing_sha256": training_marker["preprocessing_sha256"],
        "gpu_preflight_sha256": training_marker.get("gpu_preflight_sha256"),
        "run_fingerprint_sha256": training_marker["run_fingerprint_sha256"],
        "target_feature_names": head.feature_names.astype(str).tolist(),
        "n_output_dimensions": int(len(head.feature_names)),
        "n_test_rows": int(len(test_indices)),
        "n_test_targeting": int(target_test.sum()),
        "n_test_controls": int(is_control.sum()),
        "test_split_fingerprint_sha256": (
            v1_test_fingerprint
            if v1_test_fingerprint is not None
            else hash_arrays(phase_rows, head.data.source_h5_rows[test_indices])
        ),
        "test_semantic_sha256": hash_arrays(
            phase_rows,
            head.data.source_h5_rows[test_indices],
            truth_raw,
            is_control,
        ),
        "v1_evaluation_reference": (
            str(v1_evaluation_path.resolve()) if v1_evaluation_path.exists() else None
        ),
        "v1_evaluation_reference_sha256": (
            file_sha256(v1_evaluation_path) if v1_evaluation_path.exists() else None
        ),
        "runtime_seconds": evaluation_runtime,
        "shared_training_runtime_seconds": float(
            training_marker["runtime_seconds"]
        ),
        "amortized_training_runtime_seconds": amortized_training_runtime,
        "total_amortized_runtime_seconds": (
            evaluation_runtime + amortized_training_runtime
        ),
        "cell_metrics": cell_summary,
        "gene_metrics": gene_summary,
        "evaluator_id": cell_eval_adapter.EVALUATOR_ID,
        "cell_eval_adapter_manifest_sha256": artifact_hashes[
            "cell_eval_adapter_manifest.json"
        ],
        "comparability": dict(comparability),
        "artifact_sha256": artifact_hashes,
    }
    atomic_json(complete_path, marker)
    return marker


def rebuild_summaries(output_root: Path) -> None:
    training_rows = []
    for path in output_root.glob("*/fold_*/train/training_complete.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        training_rows.append(
            {
                key: value
                for key, value in payload.items()
                if key
                in {
                    "arm",
                    "split",
                    "outer_fold",
                    "model",
                    "device_resolved",
                    "n_heads",
                    "runtime_seconds",
                    "epochs_completed",
                    "best_epoch",
                    "best_validation_macro_mse",
                    "steps_per_epoch",
                    "checkpoint_sha256",
                }
            }
        )
    if training_rows:
        atomic_csv(
            output_root / "training_summary.csv",
            pd.DataFrame(training_rows).sort_values(["split", "outer_fold"]),
        )
    evaluation_rows = []
    for path in output_root.glob("*/fold_*/reporters/*/evaluation_complete.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "arm": payload["arm"],
            "reporter_slug": payload["reporter_slug"],
            "reporter_short_name": payload["reporter_short_name"],
            "split": payload["split"],
            "outer_fold": payload["outer_fold"],
            "model": payload["model"],
            "device": payload["device_resolved"],
            "n_output_dimensions": payload["n_output_dimensions"],
            "n_test_rows": payload["n_test_rows"],
            "runtime_seconds": payload["runtime_seconds"],
            "shared_training_runtime_seconds": payload[
                "shared_training_runtime_seconds"
            ],
            "amortized_training_runtime_seconds": payload[
                "amortized_training_runtime_seconds"
            ],
            "total_amortized_runtime_seconds": payload[
                "total_amortized_runtime_seconds"
            ],
            "checkpoint_sha256": payload["checkpoint_sha256"],
        }
        row.update(
            {f"cell_{key}": value for key, value in payload["cell_metrics"].items()}
        )
        row.update(
            {f"gene_{key}": value for key, value in payload["gene_metrics"].items()}
        )
        evaluation_rows.append(row)
    if evaluation_rows:
        atomic_csv(
            output_root / "benchmark_summary.csv",
            pd.DataFrame(evaluation_rows).sort_values(
                ["split", "outer_fold", "reporter_slug"]
            ),
        )


def run_synthetic_gpu_smoke(args: argparse.Namespace, library: Any) -> None:
    import torch

    resolved = choose_torch_device(args.device)
    if resolved != "cuda":
        raise RuntimeError("--synthetic-gpu-smoke requires a visible CUDA device")
    configure_deterministic_torch(args, resolved)
    args.output_root.mkdir(parents=True, exist_ok=True)
    result = library.run_synthetic_self_test(
        device="cuda",
        precision="auto" if args.amp else "float32",
        seed=args.seed,
        checkpoint_directory=args.output_root / "synthetic_checkpoint",
        capacity_variant=("high_dim" if args.arm in {"B", "D"} else "v1"),
    )
    marker: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "arm": args.arm,
        "device_resolved": result["device"],
        "cuda_device_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        **result,
        "status": "pass",
    }
    atomic_json(args.output_root / "synthetic_gpu_smoke.json", marker)
    print(json.dumps(marker, indent=2), flush=True)


def run_job(
    args: argparse.Namespace,
    library: Any,
    phase_cache: PhaseCache,
    reporter_table: pd.DataFrame,
    reporter_data: Mapping[str, Any],
    split_name: str,
    outer_fold: int,
    run_manifest: Mapping[str, Any],
    head_schema_sha256: str,
    resolved_device: str,
) -> None:
    job_dir = args.output_root / split_name / f"fold_{outer_fold}"
    if args.overwrite:
        safe_remove_job(job_dir, args.output_root)
    job_dir.mkdir(parents=True, exist_ok=True)
    failure_path = job_dir / "job_failed.json"
    failure_path.unlink(missing_ok=True)
    complete_path = job_dir / "job_complete.json"
    requested_slugs = reporter_table["reporter_slug"].astype(str).tolist()
    if complete_path.exists() and not args.overwrite and not args.train_only:
        existing = json.loads(complete_path.read_text(encoding="utf-8"))
        if existing.get("run_fingerprint_sha256") != run_manifest["run_fingerprint_sha256"]:
            raise RuntimeError(f"Existing job is incompatible: {complete_path}")
        # Do not trust a top-level marker alone.  Re-open the training marker,
        # checkpoint, and every reporter artifact below before re-marking the
        # job complete.  This also repairs interrupted/partially copied output.
        complete_path.unlink()
        print(f"  audit completed job before resume: {split_name}/fold_{outer_fold}", flush=True)

    heads = [
        build_partition(
            args,
            phase_cache,
            row,
            reporter_data[str(row.reporter_slug)],
            split_name,
            outer_fold,
        )
        for _, row in reporter_table.iterrows()
    ]
    unique_train_rows, global_partition_counts = global_partition_integrity(heads)
    train_dir = job_dir / "train"
    preprocessing_path = train_dir / "preprocessing.json"
    v1_preprocessing_path = (
        args.v1_reference_root
        / split_name
        / f"fold_{outer_fold}"
        / "train"
        / "preprocessing.json"
    )
    v1_preprocessing_sha: str | None = None
    if preprocessing_path.exists() and not args.overwrite:
        x_state = load_preprocessing_payload(preprocessing_path, heads)
    else:
        if args.eval_only:
            raise FileNotFoundError(
                f"--eval-only requires preprocessing: {preprocessing_path}"
            )
        if args.budget_policy == "v1_locked":
            if not v1_preprocessing_path.exists():
                raise FileNotFoundError(
                    f"Missing frozen V1 preprocessing reference: {v1_preprocessing_path}"
                )
            x_state = load_preprocessing_payload(
                v1_preprocessing_path, heads, allow_v1_schema=True
            )
            for head in heads:
                assert head.y_preprocessing is not None
                if head.y_preprocessing.n_fit_rows != len(
                    head.complete_train_indices
                ):
                    raise RuntimeError(
                        f"Frozen V1 Y-fit cohort count differs for {head.slug}"
                    )
            if x_state.n_fit_unique_rows != len(unique_train_rows):
                raise RuntimeError("Frozen V1 X-fit cohort count differs from V2 complete core")
        else:
            for head in heads:
                head.y_preprocessing = fit_head_y_preprocessing(
                    head.data.y, head.complete_train_indices
                )
            x_state = fit_global_x_preprocessing(
                phase_cache.x, unique_train_rows, args.min_x_finite_fraction
            )
        atomic_json(preprocessing_path, preprocessing_payload(x_state, heads))
    if v1_preprocessing_path.exists():
        v1_preprocessing_sha = file_sha256(v1_preprocessing_path)
    del unique_train_rows
    gc.collect()
    if len(heads) == 52:
        if len(x_state.kept_indices) != 172 or not np.array_equal(
            x_state.kept_indices, np.arange(172, dtype=np.int64)
        ):
            raise RuntimeError(
                "The frozen 52-head benchmark requires all 172 phase dimensions"
            )
        incomplete_heads = [
            head.slug
            for head in heads
            if head.y_preprocessing is None
            or not np.array_equal(
                head.y_preprocessing.kept_indices,
                np.arange(head.data.y.shape[1], dtype=np.int64),
            )
        ]
        if incomplete_heads:
            raise RuntimeError(
                "The frozen 52-head benchmark cannot silently drop train-constant "
                f"target dimensions: {incomplete_heads}"
            )
    preprocessing_sha = file_sha256(preprocessing_path)
    budget_lock = load_budget_lock(args, split_name, outer_fold)
    budget_lock_sha = json_sha256(budget_lock)

    comparability = {
        head.slug: assert_specialist_comparability(
            args, head, split_name, outer_fold
        )
        for head in heads
    }
    head_counts = {}
    original_head_counts = {}
    for head in heads:
        assert head.y_preprocessing is not None
        head_counts[head.slug] = {
            "source_exact_pairs": int(head.data.n_source_rows),
            "complete_core_pairs": int(head.data.n_complete_rows),
            "partial_usable_pairs": int(head.data.n_partial_rows),
            "all_missing_pairs": int(head.data.n_all_missing_rows),
            "train": int(len(head.train_indices)),
            "complete_train": int(len(head.complete_train_indices)),
            "partial_train_eligible": int(len(head.partial_train_indices)),
            "partial_train_included": int(
                len(head.partial_train_indices) if args.arm in {"C", "D"} else 0
            ),
            "validation": int(len(head.validation_indices)),
            "partial_validation_excluded": int(
                len(head.excluded_partial_validation_indices)
            ),
            "test": int(len(head.test_indices)),
            "partial_test_excluded": int(len(head.excluded_partial_test_indices)),
            "train_targeting": int((~head.data.is_control[head.train_indices]).sum()),
            "validation_targeting": int(
                (~head.data.is_control[head.validation_indices]).sum()
            ),
            "test_targeting": int((~head.data.is_control[head.test_indices]).sum()),
            "train_controls": int(head.data.is_control[head.train_indices].sum()),
            "validation_controls": int(
                head.data.is_control[head.validation_indices].sum()
            ),
            "test_controls": int(head.data.is_control[head.test_indices].sum()),
            "locked_output_dimensions": int(head.data.y.shape[1]),
            "active_output_dimensions": int(len(head.feature_names)),
            "n_phase_features_after_train_filter": int(len(x_state.kept_indices)),
            "n_target_features_after_train_filter": int(len(head.feature_names)),
            "active_train_source_h5_rows_sha256": hash_arrays(
                head.data.source_h5_rows[head.active_train_indices]
            ),
            "complete_train_source_h5_rows_sha256": hash_arrays(
                head.data.source_h5_rows[head.complete_train_indices]
            ),
            "validation_phase_rows_sha256": hash_arrays(
                head.data.phase_rows[head.validation_indices]
            ),
            "test_phase_rows_sha256": hash_arrays(
                head.data.phase_rows[head.test_indices]
            ),
            "active_train_mask_sha256": hash_arrays(
                head.data.observed_mask[head.active_train_indices]
            ),
            "active_train_observed_elements": int(
                head.data.observed_mask[head.active_train_indices].sum()
            ),
            "active_train_missing_elements": int(
                (~head.data.observed_mask[head.active_train_indices]).sum()
            ),
        }
        original_head_counts[head.slug] = {
            "source_exact_pairs": head_counts[head.slug]["source_exact_pairs"],
            "complete_core_pairs": head_counts[head.slug]["complete_core_pairs"],
            "partial_usable_pairs": head_counts[head.slug]["partial_usable_pairs"],
            "all_missing_pairs": head_counts[head.slug]["all_missing_pairs"],
            **head.original_partition_counts,
            "locked_output_dimensions": int(head.data.y.shape[1]),
            "active_output_dimensions": int(len(head.feature_names)),
            "n_phase_features_after_train_filter": int(len(x_state.kept_indices)),
            "n_target_features_after_train_filter": int(len(head.feature_names)),
        }
    job_manifest = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "arm": args.arm,
        "split": split_name,
        "outer_fold": outer_fold,
        "validation_fold": heads[0].validation_fold,
        "n_folds": int(phase_cache.manifest["n_folds"]),
        "n_heads": len(heads),
        "reporter_slugs": requested_slugs,
        "run_fingerprint_sha256": run_manifest["run_fingerprint_sha256"],
        "head_schema_sha256": head_schema_sha256,
        "preprocessing_sha256": preprocessing_sha,
        "v1_preprocessing_sha256": v1_preprocessing_sha,
        "preprocessing_reference_sha256": v1_preprocessing_sha,
        "budget_lock": budget_lock,
        "budget_lock_sha256": budget_lock_sha,
        "gpu_preflight_sha256": run_manifest.get("gpu_preflight_sha256"),
        "phase172_feature_manifest_sha256": phase_cache.manifest.get(
            "feature_manifest_sha256"
        ),
        "max_train_observations_per_head": (
            args.max_train_observations_per_head or None
        ),
        "max_validation_observations_per_head": (
            args.max_validation_observations_per_head or None
        ),
        "max_test_observations_per_head": (
            args.max_test_observations_per_head or None
        ),
        "per_head_partition_counts": head_counts,
        "per_head_original_partition_counts": original_head_counts,
        "strict_specialist_comparable": all(
            item["strict_specialist_comparable"] for item in comparability.values()
        ),
        "global_partition_counts": global_partition_counts,
        "cohort_hashes": {
            "complete_validation": json_sha256(
                {
                    h.slug: head_counts[h.slug]["validation_phase_rows_sha256"]
                    for h in heads
                }
            ),
            "complete_test": json_sha256(
                {
                    h.slug: head_counts[h.slug]["test_phase_rows_sha256"]
                    for h in heads
                }
            ),
            "complete_train": json_sha256(
                {
                    h.slug: head_counts[h.slug][
                        "complete_train_source_h5_rows_sha256"
                    ]
                    for h in heads
                }
            ),
        },
        "model_config": model_config(
            args,
            len(x_state.kept_indices),
            {head.slug: int(head.data.y.shape[1]) for head in heads},
        ),
    }
    job_manifest_path = job_dir / "job_manifest.json"
    atomic_json(job_manifest_path, job_manifest)
    job_manifest_sha = file_sha256(job_manifest_path)

    gpu_staging = build_gpu_training_staging(
        args, phase_cache, x_state, heads, resolved_device
    )

    training_complete_path = train_dir / "training_complete.json"
    if args.eval_only:
        if not training_complete_path.exists():
            raise FileNotFoundError(
                f"--eval-only requires training marker: {training_complete_path}"
            )
        training_marker = json.loads(
            training_complete_path.read_text(encoding="utf-8")
        )
        if training_marker.get("job_manifest_sha256") != job_manifest_sha:
            raise RuntimeError("Training marker and rebuilt job manifest differ")
        model = load_completed_model(library, training_marker, resolved_device)
    else:
        model, training_marker = train_one_job(
            args,
            library,
            phase_cache,
            heads,
            x_state,
            train_dir,
            split_name,
            outer_fold,
            str(run_manifest["run_fingerprint_sha256"]),
            head_schema_sha256,
            job_manifest_sha,
            preprocessing_sha,
            run_manifest.get("gpu_preflight_sha256"),
            resolved_device,
            gpu_staging,
            budget_lock,
        )
    rebuild_summaries(args.output_root)
    if args.train_only:
        print(f"  training-only complete: {split_name}/fold_{outer_fold}", flush=True)
        return

    evaluation_failures = []
    for number, head in enumerate(heads, start=1):
        evaluation_dir = job_dir / "reporters" / head.slug
        try:
            marker = evaluate_head(
                args,
                library,
                model,
                phase_cache,
                x_state,
                head,
                evaluation_dir,
                split_name,
                outer_fold,
                training_marker,
                comparability[head.slug],
                gpu_staging,
            )
            print(
                f"    evaluated {number}/{len(heads)} {head.slug}: "
                f"cell gain={marker['cell_metrics']['gain_vs_train_mean']:.4f}; "
                f"gene gain={marker['gene_metrics']['gain_vs_train_mean']:.4f}",
                flush=True,
            )
        except Exception as error:
            failure = {
                "schema_version": SCHEMA_VERSION,
                "split": split_name,
                "outer_fold": outer_fold,
                "reporter_slug": head.slug,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            atomic_json(evaluation_dir / "evaluation_failed.json", failure)
            evaluation_failures.append(failure)
            if not args.continue_on_error:
                raise
        rebuild_summaries(args.output_root)
    if evaluation_failures:
        raise RuntimeError(
            f"{len(evaluation_failures)} reporter evaluations failed in "
            f"{split_name}/fold_{outer_fold}"
        )
    evaluation_markers = list(
        (job_dir / "reporters").glob("*/evaluation_complete.json")
    )
    if len(evaluation_markers) != len(heads):
        raise RuntimeError(
            f"Expected {len(heads)} evaluation markers, found {len(evaluation_markers)}"
        )
    complete = {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "outer_fold": outer_fold,
        "model": MODEL_NAME,
        "arm": args.arm,
        "device_resolved": resolved_device,
        "n_heads": len(heads),
        "n_evaluations": len(evaluation_markers),
        "reporter_slugs": requested_slugs,
        "checkpoint_path": training_marker["checkpoint_path"],
        "checkpoint_sha256": training_marker["checkpoint_sha256"],
        "head_schema_sha256": head_schema_sha256,
        "job_manifest_sha256": job_manifest_sha,
        "training_complete_sha256": file_sha256(training_complete_path),
        "evaluation_complete_sha256": {
            path.parent.name: file_sha256(path) for path in evaluation_markers
        },
        "preprocessing_sha256": preprocessing_sha,
        "v1_preprocessing_sha256": v1_preprocessing_sha,
        "budget_lock_sha256": budget_lock_sha,
        "cohort_hashes": job_manifest["cohort_hashes"],
        "complete_core_cohort_hashes": job_manifest["cohort_hashes"],
        "gpu_preflight_sha256": run_manifest.get("gpu_preflight_sha256"),
        "run_fingerprint_sha256": run_manifest["run_fingerprint_sha256"],
        "strict_specialist_comparable": all(
            item["strict_specialist_comparable"] for item in comparability.values()
        ),
        "status": "complete",
    }
    atomic_json(complete_path, complete)
    rebuild_summaries(args.output_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-cache", type=Path, default=DEFAULT_PHASE_CACHE)
    parser.add_argument("--exact-cache-root", type=Path, default=DEFAULT_EXACT_ROOT)
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGET_TABLE)
    parser.add_argument(
        "--target-feature-dictionary",
        type=Path,
        default=DEFAULT_TARGET_FEATURE_DICTIONARY,
    )
    parser.add_argument(
        "--specialist-reference-root",
        type=Path,
        default=DEFAULT_SPECIALIST_REFERENCE,
    )
    parser.add_argument(
        "--v1-reference-root", type=Path, default=DEFAULT_V1_REFERENCE
    )
    parser.add_argument(
        "--arm",
        choices=VALID_ARMS,
        required=True,
        help="Preregistered 2x2 arm: A/B complete labels; C/D true endpoint mask",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reporters", default="all")
    parser.add_argument(
        "--splits", default="field_holdout_sanity,gene_holdout_main"
    )
    parser.add_argument("--folds", default="all")
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--min-x-finite-fraction", type=float, default=0.80)

    parser.add_argument("--hidden-width", type=int, default=512)
    parser.add_argument("--residual-blocks", type=int, default=4)
    parser.add_argument(
        "--budget-policy",
        choices=("v1_locked", "smoke"),
        default="v1_locked",
        help="Formal V1 actual-update lock or explicitly bounded smoke protocol",
    )
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-heads", type=int, default=4)
    parser.add_argument("--observations-per-head", type=int, default=1024)
    parser.add_argument("--prediction-batch-size", type=int, default=16384)
    parser.add_argument(
        "--gpu-staging",
        choices=("auto", "off", "required"),
        default="required",
        help=(
            "Stage exact FP32 standardized phase/target matrices on CUDA; auto "
            "falls back safely when the configured VRAM reserve would be crossed"
        ),
    )
    parser.add_argument("--gpu-staging-reserve-gib", type=float, default=12.0)
    parser.add_argument("--gpu-staging-chunk-rows", type=int, default=262144)
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=int(os.environ.get("OPS_MASKED_MULTITASK_PARALLEL_JOBS", "4")),
        help="Maximum concurrent independent split/fold jobs on CUDA",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--minimum-learning-rate", type=float, default=5e-6)
    parser.add_argument("--checkpoint-every-steps", type=int, default=250)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    tf32_group = parser.add_mutually_exclusive_group()
    tf32_group.add_argument("--tf32", dest="tf32", action="store_true")
    tf32_group.add_argument("--no-tf32", dest="tf32", action="store_false")
    parser.set_defaults(tf32=True)

    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument(
        "--max-train-observations-per-head", type=int, default=0
    )
    parser.add_argument(
        "--max-validation-observations-per-head", type=int, default=0
    )
    parser.add_argument("--max-test-observations-per-head", type=int, default=0)
    prediction_group = parser.add_mutually_exclusive_group()
    prediction_group.add_argument(
        "--save-predictions", dest="save_predictions", action="store_true"
    )
    prediction_group.add_argument(
        "--no-save-predictions", dest="save_predictions", action="store_false"
    )
    parser.set_defaults(save_predictions=True)
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument(
        "--require-specialist-reference",
        dest="require_specialist_reference",
        action="store_true",
    )
    reference_group.add_argument(
        "--allow-missing-specialist-reference",
        dest="require_specialist_reference",
        action="store_false",
    )
    parser.set_defaults(require_specialist_reference=True)

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", dest="resume", action="store_true")
    resume.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-reporters", action="store_true")
    parser.add_argument("--synthetic-gpu-smoke", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "workers": args.workers,
        "hidden_width": args.hidden_width,
        "residual_blocks": args.residual_blocks,
        "batch_heads": args.batch_heads,
        "observations_per_head": args.observations_per_head,
        "prediction_batch_size": args.prediction_batch_size,
        "gpu_staging_chunk_rows": args.gpu_staging_chunk_rows,
        "parallel_jobs": args.parallel_jobs,
        "epochs": args.epochs,
        "patience": args.patience,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "bootstrap_draws": args.bootstrap_draws,
    }
    invalid = {key: value for key, value in positive.items() if value < 1}
    if invalid:
        raise ValueError(f"Positive arguments required: {invalid}")
    if not 0 < args.min_x_finite_fraction <= 1:
        raise ValueError("--min-x-finite-fraction must be in (0, 1]")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    if (
        args.learning_rate <= 0
        or args.minimum_learning_rate <= 0
        or args.minimum_learning_rate > args.learning_rate
        or args.weight_decay < 0
    ):
        raise ValueError("Learning rate must be positive and weight decay nonnegative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be nonnegative")
    if args.gradient_clip_norm < 0 or args.min_delta < 0:
        raise ValueError("Gradient clipping and min delta must be nonnegative")
    if args.gpu_staging_reserve_gib < 0:
        raise ValueError("--gpu-staging-reserve-gib must be nonnegative")
    for name in (
        "max_train_observations_per_head",
        "max_validation_observations_per_head",
        "max_test_observations_per_head",
        "steps_per_epoch",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.max_test_observations_per_head > 0 and args.require_specialist_reference:
        raise ValueError(
            "A capped test set cannot pass strict specialist comparability; "
            "remove --max-test-observations-per-head or allow missing reference"
        )
    output_resolved = args.output_root.resolve()
    v1_resolved = args.v1_reference_root.resolve()
    if output_resolved == v1_resolved or v1_resolved in output_resolved.parents:
        raise ValueError("V2 output root must be completely separate from frozen V1")
    if args.gpu_staging != "required":
        raise ValueError("V2 formal/smoke runs require --gpu-staging required")
    if args.budget_policy == "v1_locked" and any(
        getattr(args, name) > 0
        for name in (
            "max_train_observations_per_head",
            "max_validation_observations_per_head",
            "max_test_observations_per_head",
            "steps_per_epoch",
        )
    ):
        raise ValueError("V1-locked production budget forbids observation/step caps")
    if args.budget_policy == "smoke" and args.steps_per_epoch < 1:
        raise ValueError("Smoke budget requires an explicit positive --steps-per-epoch")


def run_parallel_split_fold_jobs(args: argparse.Namespace) -> bool:
    """Fan out independent jobs without changing any frozen training setting.

    Returns True when this process acted only as an orchestrator.  Every child
    receives one split/fold and ``--parallel-jobs 1``; job seeds are already a
    stable function of split/fold, so serial and parallel execution are
    scientifically identical.
    """

    if (
        args.parallel_jobs <= 1
        or args.dry_run
        or args.list_reporters
        or args.synthetic_gpu_smoke
        or args.device == "cpu"
    ):
        return False
    phase_manifest = json.loads(
        (args.phase_cache / "manifest.json").read_text(encoding="utf-8")
    )
    splits = parse_csv_choice(args.splits, VALID_SPLITS, "split")
    folds = parse_folds(args.folds, int(phase_manifest["n_folds"]))
    jobs = [(split_name, fold) for split_name in splits for fold in folds]
    if len(jobs) <= 1:
        return False

    import torch

    if not torch.cuda.is_available():
        return False
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # One job stages roughly 6--8 GiB and needs room for its model, optimizer,
    # activations, CUDA context, and transient validation batches.  This gate
    # keeps at least 12 GiB/job plus the same global reserve used by staging.
    memory_parallelism = max(
        1, int(max(0.0, total_gib - args.gpu_staging_reserve_gib) // 12.0)
    )
    meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    available_kib = next(
        int(line.split()[1])
        for line in meminfo.splitlines()
        if line.startswith("MemAvailable:")
    )
    available_host_gib = available_kib / (1024**2)
    host_parallelism = max(1, int(max(0.0, available_host_gib - 64.0) // 16.0))
    parallelism = min(
        args.parallel_jobs,
        len(jobs),
        memory_parallelism,
        host_parallelism,
    )
    if parallelism <= 1:
        return False
    child_workers = max(1, args.workers // parallelism)
    print(
        f"[orchestrator] {len(jobs)} independent split/fold jobs; "
        f"parallelism={parallelism}; workers/job={child_workers}; "
        f"GPU={total_gib:.1f} GiB; host_available={available_host_gib:.1f} GiB",
        flush=True,
    )
    base_command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]]
    pending = list(jobs)
    active: list[tuple[str, int, subprocess.Popen[Any]]] = []
    failures: list[tuple[str, int, int]] = []
    previous_handlers = {
        code: signal.getsignal(code) for code in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupt_parallel_jobs(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    def terminate_parallel_children(
        children: Sequence[tuple[str, int, subprocess.Popen[Any]]],
    ) -> None:
        running = [item for item in children if item[2].poll() is None]
        for split_name, fold, process in running:
            print(
                f"[orchestrator] terminating {split_name}/fold_{fold} "
                f"pid={process.pid}",
                flush=True,
            )
            process.terminate()
        deadline = time.monotonic() + 15.0
        for _, _, process in running:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

    for code in previous_handlers:
        signal.signal(code, interrupt_parallel_jobs)
    try:
        while pending or active:
            while pending and len(active) < parallelism and not failures:
                split_name, fold = pending.pop(0)
                command = [
                    *base_command,
                    "--splits",
                    split_name,
                    "--folds",
                    str(fold),
                    "--parallel-jobs",
                    "1",
                    "--workers",
                    str(child_workers),
                ]
                environment = os.environ.copy()
                for variable in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                ):
                    environment[variable] = str(child_workers)
                process = subprocess.Popen(command, env=environment)
                active.append((split_name, fold, process))
                print(
                    f"[orchestrator] started {split_name}/fold_{fold} pid={process.pid}",
                    flush=True,
                )
            still_active: list[tuple[str, int, subprocess.Popen[Any]]] = []
            for split_name, fold, process in active:
                status = process.poll()
                if status is None:
                    still_active.append((split_name, fold, process))
                    continue
                if status != 0:
                    failures.append((split_name, fold, status))
                print(
                    f"[orchestrator] finished {split_name}/fold_{fold} status={status}",
                    flush=True,
                )
            active = still_active
            if failures:
                # Fail fast: a peer may otherwise continue a 100-epoch job for
                # hours after the first child has already invalidated the run.
                terminate_parallel_children(active)
                active = []
                pending.clear()
                break
            if active:
                time.sleep(0.25)
    finally:
        terminate_parallel_children(active)
        for code, handler in previous_handlers.items():
            signal.signal(code, handler)
    if failures:
        raise RuntimeError(f"Parallel split/fold jobs failed: {failures}")
    # Concurrent workers publish summaries atomically, but their snapshots can
    # legitimately finish out of order.  Rebuild once after the full barrier.
    rebuild_summaries(args.output_root)
    return True


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    try:
        library = importlib.import_module("ops_reporter_masked_multitask_v2_lib")
    except ImportError as error:
        raise RuntimeError(
            "Missing scripts/ops_reporter_masked_multitask_v2_lib.py or dependencies"
        ) from error
    if args.synthetic_gpu_smoke:
        run_synthetic_gpu_smoke(args, library)
        return
    if run_parallel_split_fold_jobs(args):
        return

    frozen_table = pd.read_csv(args.target_table)
    if len(frozen_table) != 52 or frozen_table["reporter_slug"].nunique() != 52:
        raise RuntimeError(
            f"Expected frozen 52-reporter table, found {len(frozen_table)} rows"
        )
    if args.list_reporters:
        print(
            frozen_table[
                [
                    "reporter_slug",
                    "short_name",
                    "biological_category",
                    "n_exact_assay_cell_links",
                    "n_technical_core_features",
                ]
            ].to_string(index=False)
        )
        return
    reporters = select_reporters(frozen_table, args.reporters)
    technical_core = load_technical_core_features(
        args.target_feature_dictionary, frozen_table
    )
    phase_manifest_path = args.phase_cache / "manifest.json"
    phase_manifest = json.loads(phase_manifest_path.read_text(encoding="utf-8"))
    splits = parse_csv_choice(args.splits, VALID_SPLITS, "split")
    folds = parse_folds(args.folds, int(phase_manifest["n_folds"]))
    gpu_preflight_path = args.output_root / "gpu_preflight.json"
    gpu_preflight_sha = (
        file_sha256(gpu_preflight_path) if gpu_preflight_path.exists() else None
    )
    runner_code_path = Path(__file__).resolve()
    multitask_library_path = Path(library.__file__).resolve()
    specialist_library_path = Path(
        sys.modules["ops_reporter_specialist_lib"].__file__
    ).resolve()
    source_provenance = {
        "phase_manifest": str(phase_manifest_path.resolve()),
        "phase_manifest_sha256": file_sha256(phase_manifest_path),
        "phase_feature_manifest_sha256": phase_manifest.get("feature_manifest_sha256"),
        "target_table": str(args.target_table.resolve()),
        "target_table_sha256": file_sha256(args.target_table),
        "target_feature_dictionary": str(args.target_feature_dictionary.resolve()),
        "target_feature_dictionary_sha256": file_sha256(
            args.target_feature_dictionary
        ),
        "gpu_preflight": str(gpu_preflight_path.resolve()),
        "gpu_preflight_sha256": gpu_preflight_sha,
        "runner_code": str(runner_code_path),
        "runner_code_sha256": file_sha256(runner_code_path),
        "multitask_library_code": str(multitask_library_path),
        "multitask_library_code_sha256": file_sha256(multitask_library_path),
        "specialist_library_code": str(specialist_library_path),
        "specialist_library_code_sha256": file_sha256(specialist_library_path),
    }
    stable_configuration = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "arm": args.arm,
        "factorial_identity": {
            "capacity_policy": (
                "matched_high_dim_heads_128_expansion568"
                if args.arm in {"B", "D"}
                else "v1_uniform_heads56_expansion640"
            ),
            "mask_policy": (
                "complete_plus_partial_true_endpoint_mask"
                if args.arm in {"C", "D"}
                else "complete_core_only"
            ),
        },
        "capacity_policy": (
            "matched_high_dim_heads_128_expansion568"
            if args.arm in {"B", "D"}
            else "v1_uniform_heads56_expansion640"
        ),
        "mask_policy": (
            "complete_plus_partial_true_endpoint_mask"
            if args.arm in {"C", "D"}
            else "complete_core_only"
        ),
        "expected_parameter_count": (
            4_310_636 if args.arm in {"B", "D"} else 4_310_852
        ),
        "gpu_staging_policy": "required",
        "reporter_slugs": reporters["reporter_slug"].astype(str).tolist(),
        "benchmark_splits": list(VALID_SPLITS),
        "benchmark_folds": list(range(int(phase_manifest["n_folds"]))),
        "source_provenance": source_provenance,
        "seed": args.seed,
        "min_x_finite_fraction": args.min_x_finite_fraction,
        "architecture": model_config(args, 172),
        "training": {
            "batch_heads": args.batch_heads,
            "observations_per_head": args.observations_per_head,
            "epochs": args.epochs,
            "steps_per_epoch": args.steps_per_epoch,
            "patience": args.patience,
            "min_delta": args.min_delta,
            "learning_rate": args.learning_rate,
            "optimizer": "AdamW",
            "optimizer_betas": [0.9, 0.999],
            "optimizer_eps": 1e-8,
            "scheduler": "linear_warmup_cosine_decay",
            "warmup_epochs": args.warmup_epochs,
            "minimum_learning_rate": args.minimum_learning_rate,
            "checkpoint_every_steps": args.checkpoint_every_steps,
            "weight_decay": args.weight_decay,
            "gradient_clip_norm": args.gradient_clip_norm,
            "amp": args.amp,
            "tf32": args.tf32,
            "budget_policy": args.budget_policy,
            "early_stopping_changes_update_budget": False,
        },
        "caps": {
            "train": args.max_train_observations_per_head,
            "validation": args.max_validation_observations_per_head,
            "test": args.max_test_observations_per_head,
        },
    }
    run_fingerprint = json_sha256(stable_configuration)
    plan = {
        **stable_configuration,
        "run_fingerprint_sha256": run_fingerprint,
        "gpu_preflight_sha256": gpu_preflight_sha,
        "requested_splits": splits,
        "requested_folds": folds,
        "n_training_jobs": len(splits) * len(folds),
        "n_evaluation_jobs": len(reporters) * len(splits) * len(folds),
        "device_requested": args.device,
        "train_only": args.train_only,
        "eval_only": args.eval_only,
        "resume": args.resume,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only; no reporter H5 or phase matrix was loaded.", flush=True)
        return

    resolved_device = choose_torch_device(args.device)
    if args.device == "cuda" and resolved_device != "cuda":
        raise RuntimeError("CUDA was explicitly requested but not resolved")
    if resolved_device == "cuda" and gpu_preflight_sha is None:
        raise FileNotFoundError(
            f"CUDA runs require the supervisor-owned GPU preflight: {gpu_preflight_path}"
        )
    configure_deterministic_torch(args, resolved_device)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.workers))
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_manifest_path = args.output_root / "run_manifest.json"
    run_manifest = {
        **stable_configuration,
        "run_fingerprint_sha256": run_fingerprint,
        "gpu_preflight_sha256": gpu_preflight_sha,
        "device_requested": args.device,
        "device_resolved": resolved_device,
        "n_expected_training_jobs": (
            len(VALID_SPLITS) * int(phase_manifest["n_folds"])
        ),
        "n_expected_evaluation_jobs": (
            len(reporters) * len(VALID_SPLITS) * int(phase_manifest["n_folds"])
        ),
    }
    with output_metadata_lock(args.output_root):
        if run_manifest_path.exists():
            existing = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if existing.get("run_fingerprint_sha256") != run_fingerprint:
                raise RuntimeError(
                    f"Output root has an incompatible run manifest: {run_manifest_path}"
                )
        else:
            atomic_json(run_manifest_path, run_manifest)
        atomic_json(args.output_root / "last_request_plan.json", plan)
        atomic_json(
            args.output_root
            / "launch_history"
            / f"launch_{time.time_ns()}_{os.getpid()}.json",
            plan,
        )

    phase_cache = PhaseCache.open(args.phase_cache)
    reporter_data: dict[str, Any] = {}
    head_rows = []
    print(
        f"Loading {len(reporters)} technical-core stores with endpoint masks",
        flush=True,
    )
    for number, (_, row) in enumerate(reporters.iterrows(), start=1):
        slug = str(row.reporter_slug)
        cache_path = args.exact_cache_root / f"all_cells_fluor_{slug}.exact.h5"
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        data = load_masked_reporter_data(cache_path, slug, technical_core[slug])
        if np.any(data.phase_rows < 0) or np.any(data.phase_rows >= len(phase_cache.x)):
            raise RuntimeError(f"Out-of-range phase_row_index in {cache_path}")
        reporter_data[slug] = data
        head_rows.append(
            {
                "reporter_slug": slug,
                "feature_names": data.target_feature_names.astype(str).tolist(),
                "output_dimensions": int(data.y.shape[1]),
                "n_source_exact_rows": data.n_source_rows,
                "n_complete_core_rows": data.n_complete_rows,
                "n_partial_usable_rows": data.n_partial_rows,
                "n_all_missing_rows": data.n_all_missing_rows,
                "n_any_observed_rows": int(len(data.phase_rows)),
                "observed_target_elements": int(data.observed_mask.sum()),
                "missing_target_elements": int((~data.observed_mask).sum()),
                "observed_mask_sha256": hash_arrays(data.observed_mask),
                "complete_source_h5_rows_sha256": hash_arrays(
                    data.source_h5_rows[data.is_complete]
                ),
                "partial_source_h5_rows_sha256": hash_arrays(
                    data.source_h5_rows[~data.is_complete]
                ),
                "source_exact_h5": str(cache_path.resolve()),
                "source_size_bytes": data.source_size_bytes,
                "source_cache_sha256": str(row.get("cache_sha256", "")),
            }
        )
        print(
            f"  [{number}/{len(reporters)}] {slug}: complete={data.n_complete_rows:,}, "
            f"partial={data.n_partial_rows:,}, all-missing={data.n_all_missing_rows:,}, "
            f"dimensions={data.y.shape[1]}",
            flush=True,
        )
    if len(reporters) == 52:
        source_total = sum(data.n_source_rows for data in reporter_data.values())
        complete_total = sum(data.n_complete_rows for data in reporter_data.values())
        partial_total = sum(data.n_partial_rows for data in reporter_data.values())
        all_missing_total = sum(
            data.n_all_missing_rows for data in reporter_data.values()
        )
        expected = (9_996_286, 8_924_045, 447_566, 624_675)
        observed = (source_total, complete_total, partial_total, all_missing_total)
        if observed != expected or source_total != (
            complete_total + partial_total + all_missing_total
        ):
            raise RuntimeError(
                "Frozen endpoint-mask audit mismatch: "
                f"observed={observed}, expected={expected}"
            )
    head_schema = {
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "n_heads": len(head_rows),
        "total_output_dimensions": int(
            sum(row["output_dimensions"] for row in head_rows)
        ),
        "heads": head_rows,
    }
    head_schema_path = args.output_root / "head_schema.json"
    with output_metadata_lock(args.output_root):
        if head_schema_path.exists():
            existing = json.loads(head_schema_path.read_text(encoding="utf-8"))
            if json_sha256(existing) != json_sha256(head_schema):
                raise RuntimeError(f"Incompatible head schema: {head_schema_path}")
        else:
            atomic_json(head_schema_path, head_schema)
        head_schema_sha = file_sha256(head_schema_path)
        run_manifest = {
            **run_manifest,
            "head_schema": str(head_schema_path.resolve()),
            "head_schema_sha256": head_schema_sha,
            "n_heads": len(head_rows),
            "total_output_dimensions": int(
                sum(row["output_dimensions"] for row in head_rows)
            ),
        }
        atomic_json(run_manifest_path, run_manifest)

    started = time.time()
    failures = []
    for split_name in splits:
        for outer_fold in folds:
            print(f"[{split_name} fold={outer_fold}]", flush=True)
            try:
                run_job(
                    args,
                    library,
                    phase_cache,
                    reporters,
                    reporter_data,
                    split_name,
                    outer_fold,
                    run_manifest,
                    head_schema_sha,
                    resolved_device,
                )
            except Exception as error:
                failure = {
                    "schema_version": SCHEMA_VERSION,
                    "split": split_name,
                    "outer_fold": outer_fold,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "runtime_seconds": time.time() - started,
                }
                job_dir = args.output_root / split_name / f"fold_{outer_fold}"
                atomic_json(job_dir / "job_failed.json", failure)
                failures.append(failure)
                print(
                    f"FAILED {split_name}/fold_{outer_fold}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                if not args.continue_on_error:
                    raise
            rebuild_summaries(args.output_root)
    if failures:
        raise RuntimeError(f"{len(failures)} requested split/fold jobs failed")
    elapsed = time.time() - started
    print(
        f"All requested shared-model jobs complete in {elapsed / 3600:.2f} h",
        flush=True,
    )


if __name__ == "__main__":
    main()
