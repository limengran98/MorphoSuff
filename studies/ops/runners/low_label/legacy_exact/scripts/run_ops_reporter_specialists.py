#!/usr/bin/env python3
"""Run configurable OPS phase-to-fluorescence reporter specialists.

The runner is deliberately sequential at the reporter level.  It loads one
reporter, applies one frozen split, fits train-only preprocessing once, and
then runs the selected model families.  Shared truth and metadata are written
once per reporter/split/fold; model directories contain only model-specific
predictions, metrics, and hyperparameters.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ops_reporter_specialist_lib import (
    RUNNER_SCHEMA_VERSION,
    VALID_MODELS,
    VALID_SPLITS,
    PhaseCache,
    aggregate_gene_profiles,
    assert_split_integrity,
    atomic_json,
    atomic_npy,
    bootstrap_gene_profile_metrics,
    choose_torch_device,
    fit_catboost_multirmse,
    fit_gbdt,
    fit_knn,
    fit_mlp,
    fit_preprocessing,
    fit_ridge,
    hash_arrays,
    load_reporter_data,
    matrix_metrics,
    parse_csv_choice,
    partition_indices,
    profile_metrics,
)


DEFAULT_PHASE_CACHE = Path("data/processed/ops_phase172_indexed")
DEFAULT_EXACT_ROOT = Path("data/processed/ops_full_reporter_exact/reporters")
DEFAULT_TARGET_TABLE = Path("results/ops_phase0_asset_audit/reporter_targets.csv")
DEFAULT_TARGET_FEATURE_DICTIONARY = Path(
    "results/ops_phase0_asset_audit/target_feature_dictionary.csv"
)
DEFAULT_OUTPUT = Path("results/ops_reporter_specialists_v1")


def parse_float_list(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of nonnegative numbers")
    return result


def parse_int_list(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers")
    return result


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
        return table.copy()
    tokens = [token.strip().casefold() for token in requested.split(",") if token.strip()]
    selected_rows = []
    for token in tokens:
        match = table[
            table["reporter_slug"].astype(str).str.casefold().eq(token)
            | table["short_name"].astype(str).str.casefold().eq(token)
        ]
        if len(match) != 1:
            raise ValueError(
                f"Reporter {token!r} matched {len(match)} rows. Use an exact reporter_slug."
            )
        selected_rows.append(match.iloc[0])
    result = pd.DataFrame(selected_rows).drop_duplicates("reporter_slug")
    return result.reset_index(drop=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_technical_core_features(
    path: Path, target_table: pd.DataFrame
) -> dict[str, list[str]]:
    dictionary = pd.read_csv(path)
    required = {"reporter_slug", "target_feature_name", "selected_in_technical_core"}
    missing_columns = sorted(required - set(dictionary.columns))
    if missing_columns:
        raise RuntimeError(
            f"Target feature dictionary lacks required columns: {missing_columns}"
        )
    if dictionary.duplicated(["reporter_slug", "target_feature_name"]).any():
        raise RuntimeError("Target feature dictionary contains duplicate reporter-feature rows")
    selected_column = dictionary["selected_in_technical_core"]
    if pd.api.types.is_bool_dtype(selected_column):
        selected = selected_column.fillna(False)
    else:
        selected = selected_column.astype(str).str.strip().str.casefold().isin(
            {"true", "1", "yes"}
        )
    core = dictionary.loc[selected].copy()
    grouped = {
        str(slug): frame["target_feature_name"].astype(str).tolist()
        for slug, frame in core.groupby("reporter_slug", sort=False)
    }
    expected_reporters = set(target_table["reporter_slug"].astype(str))
    if set(grouped) != expected_reporters:
        missing = sorted(expected_reporters - set(grouped))
        extra = sorted(set(grouped) - expected_reporters)
        raise RuntimeError(
            f"Technical-core reporter mismatch; missing={missing}, extra={extra}"
        )
    for row in target_table.itertuples(index=False):
        slug = str(row.reporter_slug)
        expected = int(row.n_technical_core_features)
        observed = len(grouped[slug])
        if observed != expected or observed < 1:
            raise RuntimeError(
                f"Technical-core count mismatch for {slug}: expected {expected}, found {observed}"
            )
    return grouped


def safe_remove_model_dir(path: Path, output_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = output_root.resolve()
    if resolved_root not in resolved_path.parents:
        raise RuntimeError(f"Refusing to remove path outside output root: {path}")
    if path.exists():
        shutil.rmtree(path)


def write_shared_bundle(
    shared_dir: Path,
    reporter_row: pd.Series,
    split_name: str,
    outer_fold: int,
    validation_fold: int,
    n_folds: int,
    data: Any,
    test_indices: np.ndarray,
    y_test_raw: np.ndarray,
    feature_names: np.ndarray,
    preprocessing: Any,
    phase_metadata: dict[str, np.ndarray],
    phase_manifest: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    shared_dir.mkdir(parents=True, exist_ok=True)
    phase_rows = data.phase_rows[test_indices]
    split_fingerprint = hash_arrays(phase_rows, test_indices)
    manifest = {
        "schema_version": RUNNER_SCHEMA_VERSION,
        "reporter_slug": str(reporter_row.reporter_slug),
        "reporter_short_name": str(reporter_row.short_name),
        "biological_category": str(reporter_row.biological_category),
        "split": split_name,
        "outer_test_fold": outer_fold,
        "validation_fold": validation_fold,
        "n_folds": n_folds,
        "seed": seed,
        "n_test_observations": int(len(test_indices)),
        "n_test_targeting": int((~data.is_control[test_indices]).sum()),
        "n_test_controls": int(data.is_control[test_indices].sum()),
        "target_feature_policy": "locked_phase0_technical_core",
        "n_source_target_features": int(data.source_target_features),
        "n_selected_target_features_before_train_variance_filter": int(
            len(data.target_feature_names)
        ),
        "n_source_exact_pairs": int(
            len(data.phase_rows) + data.dropped_nonfinite_target_rows
        ),
        "n_finite_technical_core_pairs": int(len(data.phase_rows)),
        "n_dropped_nonfinite_technical_core_pairs": int(
            data.dropped_nonfinite_target_rows
        ),
        "n_target_features": int(len(feature_names)),
        "split_fingerprint_sha256": split_fingerprint,
        "source_reporter_cache": str(data.source_cache.resolve()),
        "source_reporter_cache_size_bytes": data.source_size_bytes,
        "phase172_schema_version": phase_manifest.get("schema_version"),
        "phase172_feature_manifest_sha256": phase_manifest.get("feature_manifest_sha256"),
        "preprocessing": preprocessing.to_json(),
    }
    existing_path = shared_dir / "manifest.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        keys = (
            "schema_version",
            "reporter_slug",
            "split",
            "outer_test_fold",
            "split_fingerprint_sha256",
            "phase172_feature_manifest_sha256",
        )
        if any(existing.get(key) != manifest.get(key) for key in keys):
            raise RuntimeError(
                f"Existing shared output is incompatible: {existing_path}. "
                "Use a new output root or --overwrite."
            )
    atomic_npy(shared_dir / "test_phase_row_index.npy", phase_rows)
    atomic_npy(shared_dir / "truth_raw.npy", y_test_raw)
    atomic_npy(shared_dir / "test_is_control.npy", data.is_control[test_indices])
    for name, values in phase_metadata.items():
        atomic_npy(shared_dir / f"test_{name}.npy", values[test_indices])
    (shared_dir / "target_feature_names.txt").write_text(
        "\n".join(feature_names.astype(str)) + "\n", encoding="utf-8"
    )
    atomic_json(shared_dir / "manifest.json", manifest)
    return manifest


def save_gene_shared_once(shared_dir: Path, profiles: Any) -> None:
    gene_dir = shared_dir / "gene_profiles"
    gene_dir.mkdir(parents=True, exist_ok=True)
    expected = {
        "truth_control_relative_scaled.npy": profiles.truth,
        "screen_code.npy": profiles.screen_code,
        "gene_code.npy": profiles.gene_code,
        "n_cells.npy": profiles.n_cells,
        "n_control_cells.npy": profiles.n_control_cells,
    }
    for name, values in expected.items():
        path = gene_dir / name
        if path.exists():
            existing = np.load(path, mmap_mode="r")
            if existing.shape != values.shape or not np.array_equal(existing, values):
                raise RuntimeError(f"Model-specific aggregation changed shared gene truth: {path}")
        else:
            atomic_npy(path, values)


def run_model(
    model_name: str,
    args: argparse.Namespace,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    resolved_device: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_name == "ridge":
        return fit_ridge(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            args.ridge_alphas,
        )
    if model_name == "knn":
        return fit_knn(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            args.knn_k,
            args.knn_backend,
            resolved_device,
            args.knn_query_batch_size,
            args.knn_train_block_size,
            args.workers,
            args.allow_slow_knn,
        )
    if model_name == "mlp":
        return fit_mlp(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            args.device,
            args.mlp_hidden,
            args.mlp_batch_size,
            args.mlp_epochs,
            args.mlp_patience,
            args.mlp_learning_rate,
            args.mlp_weight_decay,
            args.workers,
            seed,
        )
    if model_name == "gbdt":
        return fit_gbdt(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            args.gbdt_backend,
            resolved_device,
            args.workers,
            args.gbdt_max_depth,
            args.gbdt_estimators,
            args.gbdt_learning_rate,
            args.gbdt_patience,
            seed,
        )
    if model_name == "catboost_multirmse":
        return fit_catboost_multirmse(
            x_train,
            y_train,
            x_validation,
            y_validation,
            x_test,
            resolved_device,
            args.workers,
            args.catboost_iterations,
            args.catboost_depth,
            args.catboost_learning_rate,
            args.catboost_patience,
            args.catboost_l2_leaf_reg,
            args.catboost_border_count,
            args.catboost_devices,
            args.catboost_gpu_ram_part,
            seed,
        )
    raise ValueError(model_name)


def run_reporter_partition(
    args: argparse.Namespace,
    phase_cache: PhaseCache,
    reporter_row: pd.Series,
    data: Any,
    x_all: np.ndarray,
    split_name: str,
    outer_fold: int,
    models: list[str],
    resolved_device: str,
) -> None:
    n_folds = int(phase_cache.manifest["n_folds"])
    fold_values = np.asarray(
        phase_cache.folds[split_name][data.phase_rows], dtype=np.uint8
    )
    train, validation, test, validation_fold = partition_indices(
        fold_values, outer_fold, n_folds
    )
    phase_metadata_all = {
        name: np.asarray(values[data.phase_rows])
        for name, values in phase_cache.metadata.items()
        if name != "is_target"
    }
    assert_split_integrity(
        split_name,
        train,
        validation,
        test,
        data.is_control,
        phase_metadata_all["gene_code"],
        data.phase_rows,
        phase_metadata_all["screen_code"],
        phase_metadata_all["well_code"],
        phase_metadata_all["tile_code"],
    )

    partition_root = (
        args.output_root
        / split_name
        / f"fold_{outer_fold}"
        / str(reporter_row.reporter_slug)
    )
    shared_dir = partition_root / "shared"
    preprocessing = fit_preprocessing(
        x_all[train], data.y[train], args.min_x_finite_fraction
    )
    x_train = preprocessing.transform_x(x_all[train])
    x_validation = preprocessing.transform_x(x_all[validation])
    x_test = preprocessing.transform_x(x_all[test])
    y_train = preprocessing.transform_y(data.y[train])
    y_validation = preprocessing.transform_y(data.y[validation])
    y_test = preprocessing.transform_y(data.y[test])
    y_test_raw = np.asarray(data.y[test][:, preprocessing.y_keep], dtype=np.float32)
    feature_names = data.target_feature_names[preprocessing.y_keep]
    metadata_test = {
        name: values[test] for name, values in phase_metadata_all.items()
    }
    shared_manifest = write_shared_bundle(
        shared_dir,
        reporter_row,
        split_name,
        outer_fold,
        validation_fold,
        n_folds,
        data,
        test,
        y_test_raw,
        feature_names,
        preprocessing,
        phase_metadata_all,
        phase_cache.manifest,
        args.seed,
    )
    partition_counts = {
        "source_exact_pairs": int(
            len(data.phase_rows) + data.dropped_nonfinite_target_rows
        ),
        "finite_technical_core_pairs": int(len(data.phase_rows)),
        "dropped_nonfinite_technical_core_pairs": int(
            data.dropped_nonfinite_target_rows
        ),
        "train": int(len(train)),
        "validation": int(len(validation)),
        "test": int(len(test)),
        "train_targeting": int((~data.is_control[train]).sum()),
        "validation_targeting": int((~data.is_control[validation]).sum()),
        "test_targeting": int((~data.is_control[test]).sum()),
        "train_controls": int(data.is_control[train].sum()),
        "validation_controls": int(data.is_control[validation].sum()),
        "test_controls": int(data.is_control[test].sum()),
        "n_phase_features_after_train_filter": int(x_train.shape[1]),
        "n_target_features_after_train_filter": int(y_train.shape[1]),
    }
    atomic_json(shared_dir / "partition_counts.json", partition_counts)
    print(
        f"  {split_name} fold={outer_fold}: train={len(train):,}, "
        f"val={len(validation):,}, test={len(test):,}, "
        f"X={x_train.shape[1]}, Y={y_train.shape[1]}",
        flush=True,
    )

    for model_name in models:
        model_dir = partition_root / "models" / model_name
        if args.overwrite:
            safe_remove_model_dir(model_dir, args.output_root)
        complete_path = model_dir / "complete.json"
        if complete_path.exists() and not args.overwrite:
            print(f"    skip complete model: {model_name}", flush=True)
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        failure_path = model_dir / "failed.json"
        failure_path.unlink(missing_ok=True)
        started = time.time()
        reporter_seed = int.from_bytes(
            str(reporter_row.reporter_slug).encode("utf-8"), "little", signed=False
        ) % 1_000_003
        model_seed = (
            args.seed
            + outer_fold * 1009
            + VALID_MODELS.index(model_name) * 100_003
            + reporter_seed
        )
        model_device = "cpu" if model_name == "ridge" else resolved_device
        print(f"    start {model_name} on {model_device}", flush=True)
        try:
            prediction_scaled, hyperparameters = run_model(
                model_name,
                args,
                x_train,
                y_train,
                x_validation,
                y_validation,
                x_test,
                model_seed,
                resolved_device,
            )
            prediction_raw = preprocessing.inverse_y(prediction_scaled)
            target_test = ~data.is_control[test]
            baseline_raw = np.broadcast_to(
                preprocessing.y_mean, y_test_raw[target_test].shape
            )
            cell_summary, cell_features = matrix_metrics(
                y_test_raw[target_test],
                prediction_raw[target_test],
                baseline_raw,
                feature_names,
            )
            cell_summary["standardized_mse"] = float(
                np.mean((y_test[target_test] - prediction_scaled[target_test]) ** 2)
            )
            standardized_baseline_mse = float(np.mean(y_test[target_test] ** 2))
            cell_summary["standardized_baseline_mse"] = standardized_baseline_mse
            cell_summary["standardized_gain_vs_train_mean"] = float(
                1.0 - cell_summary["standardized_mse"] / standardized_baseline_mse
            )
            cell_summary["n_targeting_test_cells"] = int(target_test.sum())
            profiles = aggregate_gene_profiles(
                y_test,
                prediction_scaled,
                metadata_test["screen_code"],
                metadata_test["gene_code"],
                data.is_control[test],
            )
            gene_summary, gene_features = profile_metrics(profiles, feature_names)
            gene_summary.update(
                bootstrap_gene_profile_metrics(
                    profiles, args.bootstrap_draws, model_seed + 17
                )
            )
            save_gene_shared_once(shared_dir, profiles)
            atomic_npy(
                model_dir / "gene_prediction_control_relative_scaled.npy",
                profiles.prediction,
            )
            if args.save_predictions:
                atomic_npy(model_dir / "test_prediction_raw.npy", prediction_raw)
            cell_features.to_csv(model_dir / "cell_feature_metrics.csv", index=False)
            gene_features.to_csv(model_dir / "gene_feature_metrics.csv", index=False)
            runtime = time.time() - started
            complete = {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "reporter_slug": str(reporter_row.reporter_slug),
                "reporter_short_name": str(reporter_row.short_name),
                "split": split_name,
                "outer_fold": outer_fold,
                "model": model_name,
                "device_requested": args.device,
                "device_resolved": model_device,
                "runtime_seconds": runtime,
                "shared_split_fingerprint_sha256": shared_manifest[
                    "split_fingerprint_sha256"
                ],
                "hyperparameters": hyperparameters,
                "cell_metrics": cell_summary,
                "gene_metrics": gene_summary,
            }
            atomic_json(complete_path, complete)
            print(
                f"    complete {model_name}: cell gain={cell_summary['gain_vs_train_mean']:.4f}; "
                f"gene gain={gene_summary['gain_vs_train_mean']:.4f}; "
                f"{runtime / 60:.1f} min",
                flush=True,
            )
            del prediction_scaled, prediction_raw, profiles
        except Exception as error:
            failure = {
                "schema_version": RUNNER_SCHEMA_VERSION,
                "reporter_slug": str(reporter_row.reporter_slug),
                "split": split_name,
                "outer_fold": outer_fold,
                "model": model_name,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.time() - started,
            }
            atomic_json(failure_path, failure)
            print(f"    FAILED {model_name}: {error}", file=sys.stderr, flush=True)
            if not args.continue_on_error:
                raise
        finally:
            gc.collect()

    del x_train, x_validation, x_test, y_train, y_validation, y_test, y_test_raw
    gc.collect()


def rebuild_summary(output_root: Path) -> None:
    rows = []
    for path in output_root.glob("*/fold_*/*/models/*/complete.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = {
            "reporter_slug": payload["reporter_slug"],
            "reporter_short_name": payload["reporter_short_name"],
            "split": payload["split"],
            "outer_fold": payload["outer_fold"],
            "model": payload["model"],
            "device": payload["device_resolved"],
            "runtime_seconds": payload["runtime_seconds"],
        }
        row.update({f"cell_{key}": value for key, value in payload["cell_metrics"].items()})
        row.update({f"gene_{key}": value for key, value in payload["gene_metrics"].items()})
        rows.append(row)
    if rows:
        frame = pd.DataFrame(rows).sort_values(
            ["split", "outer_fold", "reporter_slug", "model"]
        )
        summary_path = output_root / "benchmark_summary.csv"
        temporary = summary_path.with_name(
            f".{summary_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        )
        frame.to_csv(temporary, index=False)
        os.replace(temporary, summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-cache", type=Path, default=DEFAULT_PHASE_CACHE)
    parser.add_argument("--exact-cache-root", type=Path, default=DEFAULT_EXACT_ROOT)
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGET_TABLE)
    parser.add_argument(
        "--target-feature-dictionary",
        type=Path,
        default=DEFAULT_TARGET_FEATURE_DICTIONARY,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reporters", default="all", help="all or comma-separated reporter_slug values")
    parser.add_argument(
        "--models",
        default="ridge,knn,mlp",
        help=(
            "Comma-separated models. GBDT and catboost_multirmse are implemented "
            "but intentionally not default."
        ),
    )
    parser.add_argument(
        "--splits", default="field_holdout_sanity,gene_holdout_main"
    )
    parser.add_argument("--folds", default="0", help="0-based fold list or all")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--min-x-finite-fraction", type=float, default=0.80)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    prediction_group = parser.add_mutually_exclusive_group()
    prediction_group.add_argument(
        "--save-predictions", dest="save_predictions", action="store_true"
    )
    prediction_group.add_argument(
        "--no-save-predictions", dest="save_predictions", action="store_false"
    )
    parser.set_defaults(save_predictions=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--suppress-root-run-plan",
        action="store_true",
        help=(
            "Do not publish output_root/run_plan.json. Intended for bounded "
            "multi-process supervisors that publish one consolidated plan."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-reporters", action="store_true")

    parser.add_argument("--ridge-alphas", type=parse_float_list, default=parse_float_list("0.1,1,10,100,1000"))

    parser.add_argument("--knn-k", type=parse_int_list, default=parse_int_list("5,15,50,100"))
    parser.add_argument(
        "--knn-backend", choices=("auto", "faiss", "torch", "sklearn"), default="auto"
    )
    parser.add_argument("--knn-query-batch-size", type=int, default=256)
    parser.add_argument("--knn-train-block-size", type=int, default=32768)
    parser.add_argument("--allow-slow-knn", action="store_true")

    parser.add_argument("--mlp-hidden", type=parse_int_list, default=parse_int_list("256,128"))
    parser.add_argument("--mlp-batch-size", type=int, default=1024)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-patience", type=int, default=10)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--mlp-weight-decay", type=float, default=1e-4)

    parser.add_argument("--gbdt-backend", choices=("auto", "xgboost", "sklearn"), default="auto")
    parser.add_argument("--gbdt-max-depth", type=int, default=6)
    parser.add_argument("--gbdt-estimators", type=int, default=300)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbdt-patience", type=int, default=20)

    parser.add_argument("--catboost-iterations", type=int, default=300)
    parser.add_argument("--catboost-depth", type=int, default=6)
    parser.add_argument("--catboost-learning-rate", type=float, default=0.05)
    parser.add_argument("--catboost-patience", type=int, default=20)
    parser.add_argument("--catboost-l2-leaf-reg", type=float, default=3.0)
    parser.add_argument("--catboost-border-count", type=int, default=128)
    parser.add_argument("--catboost-devices", default="0")
    parser.add_argument("--catboost-gpu-ram-part", type=float, default=0.80)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0 < args.min_x_finite_fraction <= 1:
        raise ValueError("--min-x-finite-fraction must be in (0, 1]")
    if args.bootstrap_draws < 100:
        raise ValueError("Use at least 100 gene bootstrap draws")
    if (
        args.knn_query_batch_size < 1
        or args.knn_train_block_size < 1
        or args.mlp_batch_size < 1
    ):
        raise ValueError("Batch sizes must be positive")
    if (
        args.catboost_iterations < 1
        or args.catboost_depth < 1
        or args.catboost_patience < 1
        or args.catboost_l2_leaf_reg <= 0
        or args.catboost_border_count < 1
    ):
        raise ValueError("CatBoost iteration, depth, patience, regularization, and border values must be positive")
    if not 0 < args.catboost_gpu_ram_part <= 1:
        raise ValueError("--catboost-gpu-ram-part must be in (0, 1]")
    if not args.catboost_devices.strip():
        raise ValueError("--catboost-devices must not be empty")


def dependency_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        name: importlib.util.find_spec(name) is not None
        for name in ("torch", "faiss", "xgboost", "catboost")
    }
    status.update(
        {
            "xgboost_cuda": False,
            "xgboost_version": None,
            "xgboost_path": None,
            "catboost_version": None,
            "catboost_path": None,
        }
    )
    if status["xgboost"]:
        import xgboost as xgb

        status["xgboost_version"] = xgb.__version__
        status["xgboost_path"] = str(Path(xgb.__file__).resolve())
        status["xgboost_cuda"] = bool(xgb.build_info().get("USE_CUDA", False))
    if status["catboost"]:
        import catboost

        status["catboost_version"] = catboost.__version__
        status["catboost_path"] = str(Path(catboost.__file__).resolve())
    return status


def preflight_dependencies(
    models: list[str], args: argparse.Namespace, status: dict[str, Any]
) -> None:
    if "mlp" in models and not status["torch"]:
        raise RuntimeError(
            "MLP was requested but PyTorch is not installed in this Python environment. "
            "Use the research environment that contains torch, or omit mlp."
        )
    if "knn" in models and args.knn_backend == "faiss" and not status["faiss"]:
        raise RuntimeError("--knn-backend faiss was requested but FAISS is not installed")
    if "knn" in models and args.knn_backend == "torch" and not status["torch"]:
        raise RuntimeError("--knn-backend torch was requested but PyTorch is not installed")
    if "gbdt" in models and args.gbdt_backend == "xgboost" and not status["xgboost"]:
        raise RuntimeError("--gbdt-backend xgboost was requested but XGBoost is not installed")
    if (
        "gbdt" in models
        and args.gbdt_backend == "xgboost"
        and args.device == "cuda"
        and not status["xgboost_cuda"]
    ):
        raise RuntimeError(
            "--device cuda with --gbdt-backend xgboost requires a CUDA-enabled "
            "XGBoost wheel; put .ops_xgboost_gpu_deps first in PYTHONPATH"
        )
    if "catboost_multirmse" in models and not status["catboost"]:
        raise RuntimeError(
            "catboost_multirmse was requested but CatBoost is not installed; "
            "put .ops_catboost_gpu_deps first in PYTHONPATH"
        )
    if (
        "catboost_multirmse" in models
        and args.device == "cuda"
        and status["catboost"]
    ):
        from catboost.utils import get_gpu_device_count

        if int(get_gpu_device_count()) < 1:
            raise RuntimeError(
                "--device cuda with catboost_multirmse requires a visible CatBoost GPU"
            )


def main() -> None:
    args = parse_args()
    validate_args(args)
    table = pd.read_csv(args.target_table)
    if len(table) != 52 or table["reporter_slug"].nunique() != 52:
        raise RuntimeError(f"Expected the frozen 52-reporter table, found {len(table)} rows")
    if args.list_reporters:
        print(
            table[
                [
                    "reporter_slug",
                    "short_name",
                    "biological_category",
                    "n_targeting_links",
                    "n_target_features",
                ]
            ].to_string(index=False)
        )
        return

    reporters = select_reporters(table, args.reporters)
    technical_core = load_technical_core_features(
        args.target_feature_dictionary, table
    )
    models = parse_csv_choice(args.models, VALID_MODELS, "model")
    splits = parse_csv_choice(args.splits, VALID_SPLITS, "split")
    dependencies = dependency_status()
    manifest = json.loads((args.phase_cache / "manifest.json").read_text(encoding="utf-8"))
    folds = parse_folds(args.folds, int(manifest["n_folds"]))
    job_count = len(reporters) * len(splits) * len(folds) * len(models)
    plan = {
        "reporters": reporters["reporter_slug"].astype(str).tolist(),
        "models": models,
        "splits": splits,
        "folds": folds,
        "n_model_jobs": job_count,
        "device_requested": args.device,
        "workers": args.workers,
        "gbdt_included": "gbdt" in models,
        "catboost_multirmse_included": "catboost_multirmse" in models,
        "sequential_reporters": True,
        "target_feature_policy": "locked_phase0_technical_core",
        "target_feature_dictionary": str(args.target_feature_dictionary.resolve()),
        "target_feature_dictionary_sha256": file_sha256(
            args.target_feature_dictionary
        ),
        "technical_core_feature_counts": {
            slug: len(technical_core[slug])
            for slug in reporters["reporter_slug"].astype(str)
        },
        "dependency_status": dependencies,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only; no phase features were loaded and no model was fit.")
        return

    preflight_dependencies(models, args, dependencies)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(args.workers))
    resolved_device = choose_torch_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if not args.suppress_root_run_plan:
        atomic_json(args.output_root / "run_plan.json", plan)
    phase_cache = PhaseCache.open(args.phase_cache)
    started = time.time()
    for reporter_number, (_, reporter_row) in enumerate(reporters.iterrows(), start=1):
        slug = str(reporter_row.reporter_slug)
        cache_path = args.exact_cache_root / f"all_cells_fluor_{slug}.exact.h5"
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        print(
            f"[{reporter_number}/{len(reporters)}] {slug}: "
            f"{int(reporter_row.n_targeting_links):,} targeting links, "
            f"{len(technical_core[slug])} locked technical-core targets",
            flush=True,
        )
        data = load_reporter_data(cache_path, slug, technical_core[slug])
        if np.any(data.phase_rows < 0) or np.any(data.phase_rows >= phase_cache.x.shape[0]):
            raise RuntimeError(f"Out-of-range phase_row_index in {cache_path}")
        x_all = np.asarray(phase_cache.x[data.phase_rows], dtype=np.float32)
        for split_name in splits:
            for outer_fold in folds:
                run_reporter_partition(
                    args,
                    phase_cache,
                    reporter_row,
                    data,
                    x_all,
                    split_name,
                    outer_fold,
                    models,
                    resolved_device,
                )
                rebuild_summary(args.output_root)
        del x_all, data
        gc.collect()
        elapsed = time.time() - started
        print(
            f"finished reporter {reporter_number}/{len(reporters)}; "
            f"elapsed {elapsed / 3600:.2f} h",
            flush=True,
        )
    rebuild_summary(args.output_root)
    print(f"All requested jobs complete in {(time.time() - started) / 3600:.2f} h")


if __name__ == "__main__":
    main()
