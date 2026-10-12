#!/usr/bin/env python3
"""Frozen sampling utilities for OPS Target12 low-label experiments.

The sampling unit is one exact reporter observation.  Each target reporter is
subsampled independently inside the already-frozen train and inner-validation
partitions. Allocation uses the frozen five-fold exact-budget Hamilton rule,
stratified by control status and gene code.

The central manifest stores local H5-row indices together with phase/source
identities.  Method runners must read and verify this asset rather than making
their own random sample.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import run_ops_reporter_candidate_training as common_runner
from ops_reporter_specialist_lib import PhaseCache, atomic_json, hash_arrays


SCHEMA_VERSION = "ops-low-label-sampling-manifest-v1"
DEFAULT_SEED = 20260726
DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "ops_low_label_sampling_manifest_v1"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction_token(value: float) -> str:
    if not 0.0 < float(value) <= 1.0:
        raise ValueError("label fraction must be in (0, 1]")
    return format(float(value), ".12g")


def canonical_target12(target_table: Any) -> tuple[str, ...]:
    requested = set(common_runner.PANEL12)
    ordered = tuple(
        slug
        for slug in target_table["reporter_slug"].astype(str)
        if slug in requested
    )
    if len(ordered) != 12 or set(ordered) != requested:
        raise RuntimeError("Frozen target table does not contain canonical Target12")
    return ordered


def manifest_path(
    root: Path,
    *,
    split: str,
    fold: int,
    fraction: float,
) -> Path:
    return (
        Path(root)
        / split
        / f"fold_{int(fold)}"
        / f"fraction_{fraction_token(fraction)}"
        / "manifest.json"
    )


def exact_budget_subset(
    *,
    slug: str,
    indices: np.ndarray,
    phase_rows_all: np.ndarray,
    is_control_all: np.ndarray,
    phase_cache: PhaseCache,
    fraction: float,
    seed: int,
    partition_name: str,
) -> np.ndarray:
    """Return the deterministic exact-budget subset used by the v4 campaign."""

    indices = np.asarray(indices, dtype=np.int64)
    if float(fraction) >= 1.0 or len(indices) <= 1:
        return indices.copy()
    phase_rows = np.asarray(phase_rows_all[indices], dtype=np.int64)
    genes = np.asarray(
        phase_cache.metadata["gene_code"][phase_rows], dtype=np.int64
    )
    controls = np.asarray(is_control_all[indices], dtype=np.int8)
    keys = np.rec.fromarrays((controls, genes), names=("control", "gene"))
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    requested = max(1, int(round(float(fraction) * len(indices))))
    target_count = min(len(indices), requested)

    expected = target_count * counts.astype(np.float64) / int(counts.sum())
    quotas = np.minimum(counts, np.floor(expected).astype(np.int64))
    residual = target_count - int(quotas.sum())
    if residual:
        fractional = expected - quotas
        order = np.argsort(-fractional, kind="mergesort")
        for group in order:
            if residual == 0:
                break
            if quotas[group] < counts[group]:
                quotas[group] += 1
                residual -= 1
    if int(quotas.sum()) != target_count:
        raise RuntimeError("Hamilton low-label quota allocation failed")

    selected: list[np.ndarray] = []
    stable_seed = common_runner.frozen_engine.stable_seed(
        int(seed), str(slug), str(partition_name), f"{float(fraction):.8f}"
    )
    for group, quota in enumerate(quotas):
        positions = np.flatnonzero(inverse == group)
        rng = np.random.default_rng(stable_seed + group)
        chosen = positions[rng.permutation(len(positions))[: int(quota)]]
        selected.append(indices[chosen])
    result = np.sort(np.concatenate(selected).astype(np.int64, copy=False))
    if len(result) != target_count or len(np.unique(result)) != len(result):
        raise RuntimeError("Low-label subset is not exact and unique")
    return result


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_selection(
    path: Path,
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    phase_rows: np.ndarray,
    source_h5_rows: np.ndarray,
) -> dict[str, Any]:
    train_indices = np.asarray(train_indices, dtype=np.int64)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    arrays = {
        "train_local_indices": train_indices,
        "validation_local_indices": validation_indices,
        "train_phase_rows": np.asarray(phase_rows[train_indices], dtype=np.int64),
        "validation_phase_rows": np.asarray(
            phase_rows[validation_indices], dtype=np.int64
        ),
        "train_source_h5_rows": np.asarray(
            source_h5_rows[train_indices], dtype=np.int64
        ),
        "validation_source_h5_rows": np.asarray(
            source_h5_rows[validation_indices], dtype=np.int64
        ),
    }
    _atomic_npz(path, **arrays)
    return {
        "selection_file": path.name,
        "selection_file_sha256": file_sha256(path),
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "train_identity_sha256": hash_arrays(
            arrays["train_local_indices"],
            arrays["train_phase_rows"],
            arrays["train_source_h5_rows"],
        ),
        "validation_identity_sha256": hash_arrays(
            arrays["validation_local_indices"],
            arrays["validation_phase_rows"],
            arrays["validation_source_h5_rows"],
        ),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    payload = common_runner.read_json_without_duplicate_keys(path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Unexpected low-label sampling schema: {path}")
    required = {
        "schema_version",
        "split",
        "fold",
        "label_fraction",
        "seed",
        "target_reporters",
        "selection_policy",
        "selections",
    }
    if set(payload) != required:
        raise RuntimeError(
            f"Low-label sampling manifest fields changed: {sorted(set(payload))}"
        )
    if not isinstance(payload["selections"], dict):
        raise RuntimeError("Low-label sampling selections must be a mapping")
    return payload


def validate_manifest_binding(
    payload: Mapping[str, Any],
    *,
    split: str,
    fold: int,
    fraction: float,
    seed: int,
    target_reporters: Sequence[str] | None,
) -> None:
    expected = {
        "split": str(split),
        "fold": int(fold),
        "label_fraction": float(fraction),
        "seed": int(seed),
        **({"target_reporters": list(target_reporters)} if target_reporters is not None else {}),
    }
    for key, value in expected.items():
        observed = payload.get(key)
        if key == "label_fraction":
            if not np.isclose(float(observed), value, rtol=0.0, atol=1.0e-15):
                raise RuntimeError(
                    f"Sampling manifest {key} differs: {observed} != {value}"
                )
        elif observed != value:
            raise RuntimeError(
                f"Sampling manifest {key} differs: {observed!r} != {value!r}"
            )


def load_and_verify_selection(
    manifest_file: Path,
    payload: Mapping[str, Any],
    *,
    slug: str,
    phase_rows: np.ndarray,
    source_h5_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    record = payload["selections"].get(slug)
    if not isinstance(record, dict):
        raise RuntimeError(f"Sampling manifest lacks reporter {slug}")
    selection_file = manifest_file.parent / str(record["selection_file"])
    if (
        not selection_file.is_file()
        or file_sha256(selection_file) != record["selection_file_sha256"]
    ):
        raise RuntimeError(f"Sampling selection file changed for {slug}")
    with np.load(selection_file, allow_pickle=False) as source:
        required = {
            "train_local_indices",
            "validation_local_indices",
            "train_phase_rows",
            "validation_phase_rows",
            "train_source_h5_rows",
            "validation_source_h5_rows",
        }
        if set(source.files) != required:
            raise RuntimeError(f"Sampling selection arrays changed for {slug}")
        arrays = {name: np.asarray(source[name], dtype=np.int64) for name in required}
    train = arrays["train_local_indices"]
    validation = arrays["validation_local_indices"]
    if len(train) != int(record["train_count"]) or len(validation) != int(
        record["validation_count"]
    ):
        raise RuntimeError(f"Sampling selection counts changed for {slug}")
    if not np.array_equal(phase_rows[train], arrays["train_phase_rows"]):
        raise RuntimeError(f"Sampling train phase identity changed for {slug}")
    if not np.array_equal(
        source_h5_rows[train], arrays["train_source_h5_rows"]
    ):
        raise RuntimeError(f"Sampling train source identity changed for {slug}")
    if not np.array_equal(
        phase_rows[validation], arrays["validation_phase_rows"]
    ):
        raise RuntimeError(f"Sampling validation phase identity changed for {slug}")
    if not np.array_equal(
        source_h5_rows[validation], arrays["validation_source_h5_rows"]
    ):
        raise RuntimeError(f"Sampling validation source identity changed for {slug}")
    return train, validation


def apply_manifest_to_heads(
    *,
    heads: Sequence[Any],
    manifest_file: Path,
    split: str,
    fold: int,
    fraction: float,
    seed: int,
    target_reporters: Sequence[str],
) -> dict[str, Any]:
    payload = load_manifest(manifest_file)
    validate_manifest_binding(
        payload,
        split=split,
        fold=fold,
        fraction=fraction,
        seed=seed,
        target_reporters=target_reporters,
    )
    target_set = set(target_reporters)
    seen: set[str] = set()
    for head in heads:
        if head.slug not in target_set:
            continue
        train, validation = load_and_verify_selection(
            manifest_file,
            payload,
            slug=head.slug,
            phase_rows=head.data.phase_rows,
            source_h5_rows=head.data.source_h5_rows,
        )
        head.complete_train_indices = train.copy()
        head.active_train_indices = train.copy()
        head.partial_train_indices = np.empty(0, dtype=np.int64)
        head.complete_validation_indices = validation.copy()
        head.y_preprocessing = common_runner.frozen_engine.fit_head_y_preprocessing(
            head.data.y, train
        )
        seen.add(head.slug)
    missing = target_set - seen
    if missing:
        raise RuntimeError(
            f"Prepared model heads lack Target12 reporters: {sorted(missing)}"
        )
    return dict(payload)


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_json(path, dict(payload))
