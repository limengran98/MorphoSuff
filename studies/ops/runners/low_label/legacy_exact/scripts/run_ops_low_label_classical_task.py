#!/usr/bin/env python3
"""Run one target-only classical OPS low-label job.

This entry point is intentionally separate from the running low-label campaign.
It covers the four non-neural target-specific baselines:

* Ridge;
* exact k-nearest neighbours;
* endpoint-wise gradient-boosted trees; and
* CatBoost MultiRMSE.

All methods consume the same frozen, row-level low-label sampling artifact.
Phase preprocessing is the frozen full-training-cohort transformation, whereas
reporter Y preprocessing is re-fit using only the selected low-label training
rows.  Outer-test fluorescence labels are not opened until the fixed model
recipe and validation-selected state have produced test predictions.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import run_ops_low_label_transfer_fold0 as low_label
import run_ops_reporter_candidate_training as common
import run_ops_reporter_masked_multitask_resmlp_v2 as frozen_engine
from ops_reporter_specialist_lib import (
    PhaseCache,
    aggregate_gene_profiles,
    atomic_json,
    atomic_npy,
    bootstrap_gene_profile_metrics,
    fit_catboost_multirmse,
    fit_gbdt,
    fit_knn,
    fit_ridge,
    hash_arrays,
    matrix_metrics,
    profile_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ops-low-label-classical-task-v1"
RESULT_SCHEMA_VERSION = "ops-low-label-result-v1"
SAMPLING_SCHEMA_VERSION = "ops-low-label-sampling-manifest-v1"
METHOD_ALIASES = {
    "ridge": "ridge",
    "knn": "knn",
    "gbdt": "gbdt",
    "catboost": "catboost_multirmse",
    "catboost_multirmse": "catboost_multirmse",
}
PRIMARY_METRICS = (
    "cell_macro_feature_pearson",
    "cell_standardized_gain_vs_train_mean",
    "gene_macro_feature_pearson",
    "gene_gain_vs_train_mean",
    "gene_mean_profile_cosine",
    "gene_response_magnitude_spearman",
)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
KNN_K = (5, 15, 50, 100)


class ClassicalLowLabelError(RuntimeError):
    """A frozen classical low-label invariant was violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ClassicalLowLabelError(message)


def canonical_fraction_token(value: float) -> str:
    return format(float(value), ".12g")


def read_json(path: Path) -> dict[str, Any]:
    value = common.read_json_without_duplicate_keys(path)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def resolve_sampling_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    fold_root = (
        args.sampling_manifest_root
        / args.split
        / f"fold_{args.fold}"
    )
    preferred = (
        fold_root
        / f"fraction_{canonical_fraction_token(args.label_fraction)}"
        / "manifest.json"
    )
    candidates = [preferred] if preferred.is_file() else sorted(
        fold_root.glob("fraction_*/manifest.json")
    )
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        payload = read_json(path)
        try:
            observed_fraction = float(payload["label_fraction"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isclose(
            observed_fraction,
            args.label_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            matches.append((path.resolve(), payload))
    require(
        len(matches) == 1,
        "Expected exactly one matching sampling manifest at "
        f"{fold_root}; found {len(matches)}",
    )
    path, payload = matches[0]
    require(
        payload.get("schema_version") == SAMPLING_SCHEMA_VERSION,
        f"Unexpected sampling schema: {path}",
    )
    require(payload.get("split") == args.split, "Sampling split changed")
    require(int(payload.get("fold", -1)) == args.fold, "Sampling fold changed")
    require(int(payload.get("seed", -1)) == args.seed, "Sampling seed changed")
    require(
        math.isclose(
            float(payload["label_fraction"]),
            args.label_fraction,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "Sampling fraction changed",
    )
    target_reporters = tuple(str(value) for value in payload.get("target_reporters", ()))
    require(
        len(target_reporters) == len(common.PANEL12)
        and set(target_reporters) == set(common.PANEL12),
        "Sampling target reporter registry changed",
    )
    selections = payload.get("selections")
    require(isinstance(selections, Mapping), "Sampling manifest lacks selections")
    require(
        len(selections) == len(common.PANEL12)
        and set(selections) == set(common.PANEL12),
        "Sampling selection set differs from frozen panel12",
    )
    return path, payload


def load_selection_file(
    manifest_path: Path,
    entry: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    required = {
        "selection_file",
        "selection_file_sha256",
        "train_identity_sha256",
        "validation_identity_sha256",
        "train_count",
        "validation_count",
    }
    missing = sorted(required - set(entry))
    require(not missing, f"Sampling selection entry lacks fields: {missing}")
    relative = Path(str(entry["selection_file"]))
    require(not relative.is_absolute(), "Sampling selection file must be relative")
    path = (manifest_path.parent / relative).resolve()
    require(
        manifest_path.parent.resolve() in path.parents,
        "Sampling selection file escapes its manifest directory",
    )
    require(path.is_file(), f"Missing sampling selection file: {path}")
    require(
        common.file_sha256(path) == str(entry["selection_file_sha256"]),
        f"Sampling selection SHA256 mismatch: {path}",
    )
    expected_names = (
        "train_local_indices",
        "validation_local_indices",
        "train_phase_rows",
        "validation_phase_rows",
        "train_source_h5_rows",
        "validation_source_h5_rows",
    )
    with np.load(path, allow_pickle=False) as source:
        require(
            set(source.files) == set(expected_names),
            f"Sampling NPZ fields changed: {path}",
        )
        arrays = {
            name: np.asarray(source[name], dtype=np.int64) for name in expected_names
        }
    for name, values in arrays.items():
        require(values.ndim == 1, f"Sampling array is not one-dimensional: {name}")
        require(
            len(values) == len(np.unique(values)),
            f"Sampling array contains duplicate identities: {name}",
        )
    require(
        len(arrays["train_local_indices"]) == int(entry["train_count"]),
        "Sampling train count changed",
    )
    require(
        len(arrays["validation_local_indices"])
        == int(entry["validation_count"]),
        "Sampling validation count changed",
    )
    require(
        hash_arrays(
            arrays["train_local_indices"],
            arrays["train_phase_rows"],
            arrays["train_source_h5_rows"],
        )
        == str(entry["train_identity_sha256"]),
        "Sampling train identity hash changed",
    )
    require(
        hash_arrays(
            arrays["validation_local_indices"],
            arrays["validation_phase_rows"],
            arrays["validation_source_h5_rows"],
        )
        == str(entry["validation_identity_sha256"]),
        "Sampling validation identity hash changed",
    )
    return arrays


def validate_selection_against_head(
    args: argparse.Namespace,
    head: frozen_engine.HeadPartition,
    phase_cache: PhaseCache,
    arrays: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(arrays["train_local_indices"], dtype=np.int64)
    validation = np.asarray(arrays["validation_local_indices"], dtype=np.int64)
    require(
        len(train) > 0 and len(validation) > 0,
        f"Empty low-label selection for {head.slug}",
    )
    require(
        np.array_equal(train, np.sort(train))
        and np.array_equal(validation, np.sort(validation)),
        f"Sampling local indices are not sorted for {head.slug}",
    )
    require(
        np.all((0 <= train) & (train < len(head.data.y)))
        and np.all((0 <= validation) & (validation < len(head.data.y))),
        f"Sampling local index is out of bounds for {head.slug}",
    )
    require(
        np.all(np.isin(train, head.complete_train_indices))
        and np.all(np.isin(validation, head.complete_validation_indices)),
        f"Sampling selection crosses its frozen partition for {head.slug}",
    )
    expected_train = low_label.stratified_label_subset(
        head,
        head.complete_train_indices,
        phase_cache,
        args.label_fraction,
        args.seed,
        "train",
    )
    expected_validation = low_label.stratified_label_subset(
        head,
        head.complete_validation_indices,
        phase_cache,
        args.label_fraction,
        args.seed,
        "validation",
    )
    require(
        np.array_equal(train, expected_train),
        f"Central train selection differs from deterministic exact budget: {head.slug}",
    )
    require(
        np.array_equal(validation, expected_validation),
        "Central validation selection differs from deterministic exact budget: "
        f"{head.slug}",
    )
    identity_expectations = {
        "train_phase_rows": head.data.phase_rows[train],
        "validation_phase_rows": head.data.phase_rows[validation],
        "train_source_h5_rows": head.data.source_h5_rows[train],
        "validation_source_h5_rows": head.data.source_h5_rows[validation],
    }
    for name, expected in identity_expectations.items():
        require(
            np.array_equal(np.asarray(arrays[name], dtype=np.int64), expected),
            f"Sampling identity differs from sealed reporter data: {head.slug}/{name}",
        )
    return train, validation


def load_x_preprocessing(
    args: argparse.Namespace,
) -> tuple[frozen_engine.GlobalXPreprocessing, Path, str, Mapping[str, Any]]:
    path = (
        args.preprocessing_reference_root
        / args.split
        / f"fold_{args.fold}"
        / "train"
        / "preprocessing.json"
    )
    require(path.is_file(), f"Missing frozen preprocessing reference: {path}")
    payload = read_json(path)
    require(
        payload.get("schema_version") == "ops-reporter-masked-multitask-resmlp-v1",
        "Frozen preprocessing schema changed",
    )
    require(isinstance(payload.get("heads"), Mapping), "Missing reporter Y references")
    state = frozen_engine.GlobalXPreprocessing.from_json(payload["x"])
    require(
        np.array_equal(state.kept_indices, np.arange(172, dtype=np.int64)),
        "Frozen phase preprocessing no longer retains all 172 features",
    )
    return state, path.resolve(), common.file_sha256(path), payload["heads"]


def normalize_method(raw: str) -> str:
    try:
        return METHOD_ALIASES[raw.casefold()]
    except KeyError as error:
        raise ClassicalLowLabelError(f"Unsupported classical method: {raw}") from error


def fit_classical(
    method: str,
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    *,
    self_test: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "ridge":
        return fit_ridge(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            list(RIDGE_ALPHAS if not self_test else (0.1, 1.0)),
        )
    if method == "knn":
        return fit_knn(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            list(KNN_K if not self_test else (3, 5)),
            "sklearn" if self_test else args.knn_backend,
            "cpu" if self_test else args.device,
            64 if self_test else args.knn_query_batch_size,
            64 if self_test else args.knn_train_block_size,
            1 if self_test else args.workers,
            bool(self_test),
        )
    if method == "gbdt":
        return fit_gbdt(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            "sklearn" if self_test else args.gbdt_backend,
            "cpu" if self_test else args.device,
            1 if self_test else args.workers,
            2 if self_test else 6,
            5 if self_test else 300,
            0.1 if self_test else 0.05,
            2 if self_test else 20,
            seed,
        )
    if method == "catboost_multirmse":
        return fit_catboost_multirmse(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            "cpu" if self_test else args.device,
            1 if self_test else args.workers,
            5 if self_test else 300,
            2 if self_test else 6,
            0.1 if self_test else 0.05,
            2 if self_test else 20,
            3.0,
            32 if self_test else 128,
            "0",
            0.5 if self_test else 0.72,
            seed,
        )
    raise AssertionError(method)


def evaluate_prediction(
    args: argparse.Namespace,
    *,
    slug: str,
    reporter_row: pd.Series,
    phase_cache: PhaseCache,
    technical_features: Sequence[str],
    y_state: frozen_engine.HeadYPreprocessing,
    prediction_scaled: np.ndarray,
    destination: Path,
) -> dict[str, Any]:
    reference_rows, reference_controls, reference_root = (
        common.frozen_test_reference(args, slug)
    )
    data = common.load_test_only_data(
        common.exact_cache_path(args.exact_root, slug),
        slug,
        technical_features,
        reference_rows,
        reference_controls,
    )
    require(
        prediction_scaled.shape == (len(data.y), len(y_state.kept_indices)),
        f"Prediction shape differs from frozen outer test for {slug}",
    )
    require(np.all(np.isfinite(prediction_scaled)), f"Nonfinite prediction for {slug}")
    truth_scaled = y_state.transform(data.y)
    truth_raw = np.asarray(
        data.y[:, y_state.kept_indices], dtype=np.float32
    )
    prediction_raw = y_state.inverse(prediction_scaled)
    feature_names = data.target_feature_names[y_state.kept_indices]
    is_control = np.asarray(data.is_control, dtype=bool)
    target = ~is_control
    require(np.any(target), f"No targeting outer-test cells for {slug}")
    baseline_raw = np.broadcast_to(y_state.mean, truth_raw[target].shape)
    cell_summary, cell_features = matrix_metrics(
        truth_raw[target],
        prediction_raw[target],
        baseline_raw,
        feature_names,
    )
    cell_summary["standardized_mse"] = float(
        np.mean((truth_scaled[target] - prediction_scaled[target]) ** 2)
    )
    baseline_mse = float(np.mean(truth_scaled[target] ** 2))
    cell_summary["standardized_baseline_mse"] = baseline_mse
    cell_summary["standardized_gain_vs_train_mean"] = float(
        1.0 - cell_summary["standardized_mse"] / baseline_mse
    )
    cell_summary["n_targeting_test_cells"] = int(target.sum())

    screen_codes = np.asarray(
        phase_cache.metadata["screen_code"][data.phase_rows]
    )
    gene_codes = np.asarray(phase_cache.metadata["gene_code"][data.phase_rows])
    profiles = aggregate_gene_profiles(
        truth_scaled,
        prediction_scaled,
        screen_codes,
        gene_codes,
        is_control,
    )
    gene_summary, gene_features = profile_metrics(profiles, feature_names)
    gene_summary.update(
        bootstrap_gene_profile_metrics(
            profiles,
            args.bootstrap_draws,
            frozen_engine.stable_seed(
                args.seed, args.split, args.fold, slug, "bootstrap"
            ),
        )
    )
    destination.mkdir(parents=True, exist_ok=True)
    atomic_npy(destination / "test_phase_row_index.npy", data.phase_rows)
    atomic_npy(destination / "test_is_control.npy", is_control)
    atomic_npy(destination / "truth_raw.npy", truth_raw)
    atomic_npy(
        destination / "gene_prediction_control_relative_scaled.npy",
        profiles.prediction,
    )
    atomic_npy(
        destination / "gene_truth_control_relative_scaled.npy", profiles.truth
    )
    atomic_npy(destination / "gene_screen_code.npy", profiles.screen_code)
    atomic_npy(destination / "gene_gene_code.npy", profiles.gene_code)
    if args.save_predictions:
        atomic_npy(destination / "test_prediction_raw.npy", prediction_raw)
    frozen_engine.atomic_csv(destination / "cell_feature_metrics.csv", cell_features)
    frozen_engine.atomic_csv(destination / "gene_feature_metrics.csv", gene_features)
    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "reporter_slug": slug,
        "split": args.split,
        "fold": args.fold,
        "outer_test_reference": str(reference_root.resolve()),
        "outer_test_y_loaded_after_validation_selection": True,
        "cell_metrics": cell_summary,
        "gene_metrics": gene_summary,
    }
    atomic_json(destination / "evaluation_complete.json", marker)
    return marker


def prepare_reporter_head(
    args: argparse.Namespace,
    *,
    phase_cache: PhaseCache,
    reporter_row: pd.Series,
    technical_features: Sequence[str],
) -> frozen_engine.HeadPartition:
    slug = str(reporter_row.reporter_slug)
    data = common.load_sealed_training_data(
        common.exact_cache_path(args.exact_root, slug),
        slug,
        technical_features,
        phase_cache,
        args.split,
        args.fold,
    )
    return common.build_sealed_training_partition(
        args, phase_cache, reporter_row, data
    )


def reporter_result_row(
    evaluation: Mapping[str, Any],
    *,
    method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "evaluation_state": "single",
        "reporter_slug": evaluation["reporter_slug"],
        "method_id": method,
        "split": args.split,
        "fold": args.fold,
        **{
            f"cell_{key}": value
            for key, value in evaluation["cell_metrics"].items()
        },
        **{
            f"gene_{key}": value
            for key, value in evaluation["gene_metrics"].items()
        },
    }


def run_reporter(
    args: argparse.Namespace,
    *,
    method: str,
    phase_cache: PhaseCache,
    x_state: frozen_engine.GlobalXPreprocessing,
    x_reference_sha256: str,
    full_y_references: Mapping[str, Any],
    manifest_path: Path,
    manifest_payload: Mapping[str, Any],
    reporter_row: pd.Series,
    technical_features: Sequence[str],
) -> dict[str, Any]:
    slug = str(reporter_row.reporter_slug)
    root = args.output_root / "reporters" / slug
    evaluation_path = root / "evaluation" / "evaluation_complete.json"
    if args.resume and evaluation_path.is_file():
        existing = read_json(evaluation_path)
        require(existing.get("status") == "complete", "Invalid evaluation marker")
        return existing
    head = prepare_reporter_head(
        args,
        phase_cache=phase_cache,
        reporter_row=reporter_row,
        technical_features=technical_features,
    )
    require(slug in full_y_references, f"Frozen Y reference lacks {slug}")
    full_y_state = frozen_engine.HeadYPreprocessing.from_json(
        full_y_references[slug]
    )
    require(
        full_y_state.n_fit_rows == len(head.complete_train_indices),
        f"Frozen full-label training cohort changed for {slug}",
    )
    require(
        np.array_equal(
            full_y_state.kept_indices,
            np.arange(head.data.y.shape[1], dtype=np.int64),
        ),
        f"Frozen full-label reference drops endpoints for {slug}",
    )
    entry = manifest_payload["selections"][slug]
    require(isinstance(entry, Mapping), f"Invalid sampling entry for {slug}")
    arrays = load_selection_file(manifest_path, entry)
    train, validation = validate_selection_against_head(
        args, head, phase_cache, arrays
    )
    y_state = frozen_engine.fit_head_y_preprocessing(head.data.y, train)
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        root / "preprocessing.json",
        {
            "schema_version": SCHEMA_VERSION,
            "x_policy": "frozen_full_unlabelled_phase_training_cohort",
            "x_reference_sha256": x_reference_sha256,
            "y_policy": "selected_low_label_training_rows_only",
            "y": y_state.to_json(),
            "sampling_manifest_sha256": common.file_sha256(manifest_path),
            "selection_file_sha256": entry["selection_file_sha256"],
        },
    )
    prediction_path = root / "test_prediction_scaled.npy"
    training_path = root / "training_complete.json"
    if args.resume and prediction_path.is_file() and training_path.is_file():
        training = read_json(training_path)
        require(
            training.get("method_id") == method
            and training.get("sampling_manifest_sha256")
            == common.file_sha256(manifest_path)
            and training.get("selection_file_sha256")
            == entry["selection_file_sha256"],
            f"Training resume identity changed for {slug}",
        )
        prediction_scaled = np.asarray(
            np.load(prediction_path, mmap_mode="r"), dtype=np.float32
        )
    else:
        train_phase_rows = head.data.phase_rows[train]
        validation_phase_rows = head.data.phase_rows[validation]
        test_phase_rows, _test_controls, _reference_root = (
            common.frozen_test_reference(args, slug)
        )
        x_train = x_state.transform(phase_cache.x[train_phase_rows])
        x_validation = x_state.transform(phase_cache.x[validation_phase_rows])
        # Outer-test phase X is public model input.  Its fluorescence Y remains
        # unopened until fit_classical returns a validation-selected prediction.
        x_test = x_state.transform(phase_cache.x[test_phase_rows])
        y_train = y_state.transform(head.data.y[train])
        y_validation = y_state.transform(head.data.y[validation])
        reporter_seed = frozen_engine.stable_seed(
            args.seed, args.split, args.fold, slug, method
        )
        started = time.time()
        prediction_scaled, hyperparameters = fit_classical(
            method,
            args,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            reporter_seed,
        )
        require(
            prediction_scaled.shape
            == (len(test_phase_rows), len(y_state.kept_indices)),
            f"Classical prediction shape changed for {slug}",
        )
        require(
            np.all(np.isfinite(prediction_scaled)),
            f"Classical model produced nonfinite predictions for {slug}",
        )
        atomic_npy(prediction_path, prediction_scaled)
        training = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "method_id": method,
            "reporter_slug": slug,
            "split": args.split,
            "fold": args.fold,
            "label_fraction": args.label_fraction,
            "seed": args.seed,
            "n_train_labels": len(train),
            "n_validation_labels": len(validation),
            "sampling_manifest_sha256": common.file_sha256(manifest_path),
            "selection_file_sha256": entry["selection_file_sha256"],
            "x_reference_sha256": x_reference_sha256,
            "prediction_scaled_sha256": common.file_sha256(prediction_path),
            "validation_selected_hyperparameters": hyperparameters,
            "outer_test_phase_x_available_for_prediction": True,
            "outer_test_y_physically_loaded": False,
            "outer_test_used_for_selection": False,
            "runtime_seconds": time.time() - started,
        }
        atomic_json(training_path, training)
        del x_train, x_validation, x_test, y_train, y_validation
        gc.collect()
    require(
        common.file_sha256(prediction_path)
        == str(training["prediction_scaled_sha256"]),
        f"Prediction artifact changed for {slug}",
    )
    evaluation = evaluate_prediction(
        args,
        slug=slug,
        reporter_row=reporter_row,
        phase_cache=phase_cache,
        technical_features=technical_features,
        y_state=y_state,
        prediction_scaled=np.asarray(prediction_scaled, dtype=np.float32),
        destination=root / "evaluation",
    )
    del head, prediction_scaled
    gc.collect()
    return evaluation


def run_self_test() -> None:
    rng = np.random.default_rng(20260728)
    x_train = rng.normal(size=(64, 8)).astype(np.float32)
    x_validation = rng.normal(size=(16, 8)).astype(np.float32)
    x_test = rng.normal(size=(8, 8)).astype(np.float32)
    coefficients = rng.normal(size=(8, 3)).astype(np.float32)
    y_train = x_train @ coefficients + rng.normal(
        scale=0.1, size=(64, 3)
    ).astype(np.float32)
    y_validation = x_validation @ coefficients + rng.normal(
        scale=0.1, size=(16, 3)
    ).astype(np.float32)
    dummy_args = SimpleNamespace(
        knn_backend="sklearn",
        device="cpu",
        knn_query_batch_size=64,
        knn_train_block_size=64,
        workers=1,
        gbdt_backend="sklearn",
    )
    results: dict[str, Any] = {}
    for method in ("ridge", "knn", "gbdt", "catboost_multirmse"):
        prediction, details = fit_classical(
            method,
            dummy_args,
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            20260728,
            self_test=True,
        )
        require(prediction.shape == (8, 3), f"Self-test shape failed: {method}")
        require(np.all(np.isfinite(prediction)), f"Self-test nonfinite: {method}")
        results[method] = {
            "prediction_shape": list(prediction.shape),
            "backend": details.get("backend"),
            "device": details.get("device"),
        }
    with tempfile.TemporaryDirectory(prefix="ops-low-label-classical-") as temporary:
        root = Path(temporary)
        selection = root / "selection.npz"
        arrays = {
            "train_local_indices": np.asarray([0, 2], dtype=np.int64),
            "validation_local_indices": np.asarray([1], dtype=np.int64),
            "train_phase_rows": np.asarray([10, 12], dtype=np.int64),
            "validation_phase_rows": np.asarray([11], dtype=np.int64),
            "train_source_h5_rows": np.asarray([20, 22], dtype=np.int64),
            "validation_source_h5_rows": np.asarray([21], dtype=np.int64),
        }
        np.savez_compressed(selection, **arrays)
        entry = {
            "selection_file": selection.name,
            "selection_file_sha256": common.file_sha256(selection),
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
            "train_count": 2,
            "validation_count": 1,
        }
        loaded = load_selection_file(root / "manifest.json", entry)
        require(
            all(np.array_equal(loaded[name], values) for name, values in arrays.items()),
            "Sampling NPZ self-test changed identities",
        )
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "scope": "CPU synthetic fitting and sampling artifact parser",
                "methods": results,
            },
            indent=2,
        ),
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_ALIASES))
    parser.add_argument("--label-fraction", type=float)
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument(
        "--split",
        default="gene_holdout_main",
        choices=("gene_holdout_main",),
    )
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--sampling-manifest-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "ops_low_label_sampling_manifest_v1",
    )
    parser.add_argument("--mode", default="formal", choices=("formal",))
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument(
        "--gpu-staging", default="required", choices=("required",)
    )
    parser.add_argument("--gpu-staging-reserve-gib", type=float, default=16.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--knn-backend", default="torch", choices=("torch",))
    parser.add_argument("--knn-query-batch-size", type=int, default=256)
    parser.add_argument("--knn-train-block-size", type=int, default=32768)
    parser.add_argument("--gbdt-backend", default="xgboost", choices=("xgboost",))
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--phase-cache",
        type=Path,
        default=PROJECT_ROOT / common.DEFAULT_PHASE_CACHE,
    )
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=PROJECT_ROOT / common.DEFAULT_EXACT_ROOT,
    )
    parser.add_argument(
        "--target-table",
        type=Path,
        default=PROJECT_ROOT / common.DEFAULT_TARGET_TABLE,
    )
    parser.add_argument(
        "--target-feature-dictionary",
        type=Path,
        default=PROJECT_ROOT / common.DEFAULT_TARGET_FEATURE_DICTIONARY,
    )
    parser.add_argument(
        "--specialist-reference-root",
        type=Path,
        default=PROJECT_ROOT / common.DEFAULT_SPECIALIST_REFERENCE,
    )
    parser.add_argument(
        "--preprocessing-reference-root",
        type=Path,
        default=PROJECT_ROOT / common.DEFAULT_PREPROCESSING_REFERENCE,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.self_test:
        return
    require(args.method is not None, "--method is required")
    require(args.label_fraction is not None, "--label-fraction is required")
    require(
        0.0 < float(args.label_fraction) <= 1.0,
        "--label-fraction must be in (0, 1]",
    )
    require(args.fold is not None, "--fold is required")
    require(args.output_root is not None, "--output-root is required")
    require(args.workers > 0, "--workers must be positive")
    require(args.bootstrap_draws > 0, "--bootstrap-draws must be positive")
    require(
        args.gpu_staging_reserve_gib >= 12.0,
        "At least 12 GiB VRAM reserve is required",
    )
    require(not (args.resume and args.overwrite), "--resume and --overwrite conflict")
    method = normalize_method(args.method)
    if method != "ridge":
        require(
            args.device == "cuda",
            f"{method} formal execution requires CUDA; CPU fallback is forbidden",
        )


def normalize_paths(args: argparse.Namespace) -> None:
    for name in (
        "output_root",
        "sampling_manifest_root",
        "phase_cache",
        "exact_root",
        "target_table",
        "target_feature_dictionary",
        "specialist_reference_root",
        "preprocessing_reference_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, Path(value).expanduser().resolve())


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.self_test:
        run_self_test()
        return
    normalize_paths(args)
    method = normalize_method(args.method)
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / "low_label_result.json"
    if args.resume and result_path.is_file():
        result = read_json(result_path)
        require(
            result.get("schema_version") == RESULT_SCHEMA_VERSION
            and result.get("status") == "complete"
            and result.get("completed") is True,
            "Existing low-label result marker is invalid",
        )
        print(json.dumps(result, indent=2), flush=True)
        return

    manifest_path, manifest_payload = resolve_sampling_manifest(args)
    manifest_sha256 = common.file_sha256(manifest_path)
    x_state, x_reference_path, x_reference_sha256, full_y_references = (
        load_x_preprocessing(args)
    )
    frozen_table = pd.read_csv(args.target_table)
    require(
        len(frozen_table) == 52
        and frozen_table["reporter_slug"].nunique() == 52,
        "Frozen reporter registry changed",
    )
    indexed = frozen_table.set_index("reporter_slug", drop=False)
    require(
        all(slug in indexed.index for slug in common.PANEL12),
        "Frozen target table lacks panel12",
    )
    phase_cache = PhaseCache.open(args.phase_cache)
    common.validate_frozen_assets(phase_cache, frozen_table, args.exact_root)
    technical = frozen_engine.load_technical_core_features(
        args.target_feature_dictionary, frozen_table
    )
    if method != "ridge":
        gpu = common.configure_cuda(args.seed, tf32=True)
    else:
        gpu = {
            "device": "cpu",
            "execution_policy": "predeclared_closed_form_cpu_not_fallback",
        }
    atomic_json(args.output_root / "gpu_preflight.json", gpu)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "method_id": method,
        "experiment_arm": "target_only",
        "split": args.split,
        "fold": args.fold,
        "label_fraction": args.label_fraction,
        "seed": args.seed,
        "target_reporters": list(common.PANEL12),
        "donor_reporter_policy": "absent",
        "sampling_manifest": str(manifest_path),
        "sampling_manifest_sha256": manifest_sha256,
        "x_preprocessing_reference": str(x_reference_path),
        "x_preprocessing_reference_sha256": x_reference_sha256,
        "runner": str(Path(__file__).resolve()),
        "runner_sha256": common.file_sha256(Path(__file__).resolve()),
        "outer_test_y_access_during_training": "physically_sealed",
    }
    identity_path = args.output_root / "training_identity.json"
    if identity_path.is_file():
        require(
            read_json(identity_path) == identity,
            "Existing classical low-label training identity changed",
        )
    else:
        atomic_json(identity_path, identity)

    started = time.time()
    rows: list[dict[str, Any]] = []
    for reporter_index, slug in enumerate(common.PANEL12, start=1):
        print(
            f"[{reporter_index:02d}/12] {method} {slug} "
            f"fold={args.fold} fraction={args.label_fraction:g}",
            flush=True,
        )
        evaluation = run_reporter(
            args,
            method=method,
            phase_cache=phase_cache,
            x_state=x_state,
            x_reference_sha256=x_reference_sha256,
            full_y_references=full_y_references,
            manifest_path=manifest_path,
            manifest_payload=manifest_payload,
            reporter_row=indexed.loc[slug],
            technical_features=technical[slug],
        )
        rows.append(
            reporter_result_row(evaluation, method=method, args=args)
        )
    table = pd.DataFrame(rows)
    require(len(table) == 12, "Classical low-label evaluation is not panel12 complete")
    for metric in PRIMARY_METRICS:
        require(metric in table, f"Missing required primary metric: {metric}")
        require(
            pd.to_numeric(table[metric], errors="coerce").notna().all(),
            f"Nonfinite required primary metric: {metric}",
        )
    summary_path = args.output_root / "evaluation_states_summary.csv"
    frozen_engine.atomic_csv(summary_path, table)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "completed": True,
        "method_id": method,
        "experiment_arm": "target_only",
        "mode": "formal",
        "split": args.split,
        "fold": args.fold,
        "label_fraction": args.label_fraction,
        "seed": args.seed,
        "n_target_reporters": 12,
        "target_reporter_policy": "panel12_low_label",
        "donor_reporter_policy": "absent",
        "cohort_matches_frozen_reference": True,
        "sampling_manifest": str(manifest_path),
        "sampling_manifest_sha256": manifest_sha256,
        "x_preprocessing_policy": "frozen_full_unlabelled_phase_training_cohort",
        "target_y_preprocessing_policy": "selected_low_label_training_rows_only",
        "outer_test_used_for_selection": False,
        "outer_test_y_loaded_after_checkpoint_selection": True,
        "test_evaluated": True,
        "execution_device": "cpu" if method == "ridge" else "cuda",
        "cpu_fallback_allowed": False,
        "primary_state": "single",
        "macro_metrics": {
            metric: float(table[metric].mean()) for metric in PRIMARY_METRICS
        },
        "reporter_metrics": str(summary_path.resolve()),
        "reporter_metrics_sha256": common.file_sha256(summary_path),
        "runtime_seconds": time.time() - started,
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
