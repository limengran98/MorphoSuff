#!/usr/bin/env python3
"""Run one frozen OPS independent-specialist evaluation partition.

The runner has two deliberately separate modes:

* ``gene_holdout_main`` re-evaluates the already frozen specialist predictions
  with the common v2/cell-eval-adapted evaluator.  It does not retrain a model.
* ``strict_whole_screen`` and ``strict_gene_screen`` rebuild train/validation/
  test partitions and fit the five method-native specialists from scratch.

One invocation owns exactly one reporter/partition and writes a single
``training_result.json`` completion marker. This makes the task resumable by a
generic single-GPU scheduler without allowing two allocations to share a
result directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import ops_reporter_cell_eval_adapter as cell_eval_adapter
from ops_reporter_specialist_lib import (
    VALID_MODELS,
    PhaseCache,
    aggregate_gene_profiles,
    atomic_json,
    atomic_npy,
    bootstrap_gene_profile_metrics,
    choose_torch_device,
    fit_preprocessing,
    hash_arrays,
    load_reporter_data,
    matrix_metrics,
    parse_csv_choice,
    partition_indices,
    profile_metrics,
)
from run_ops_reporter_specialists import (
    dependency_status,
    load_technical_core_features,
    preflight_dependencies,
    run_model,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ops-independent-specialist-four-tier-task-v1"
TASK_SPEC_SCHEMA = "ops-independent-specialist-protocol-task-spec-v1"
TASK_RESULT_SCHEMA = "ops-independent-specialist-protocol-task-result-v1"
PROTOCOLS = (
    "gene_holdout_main",
    "strict_whole_screen",
    "strict_gene_screen",
)
DEFAULT_PHASE_CACHE = ROOT / "data/processed/ops_phase172_indexed"
DEFAULT_EXACT_ROOT = ROOT / "data/processed/ops_full_reporter_exact/reporters"
DEFAULT_TARGET_TABLE = ROOT / "results/ops_phase0_asset_audit/reporter_targets.csv"
DEFAULT_FEATURE_DICTIONARY = (
    ROOT / "results/ops_phase0_asset_audit/target_feature_dictionary.csv"
)
DEFAULT_LEGACY_ROOTS = {
    "ridge": ROOT / "results/ops_reporter_specialists_v1",
    "knn": ROOT / "results/ops_reporter_specialists_v1",
    "mlp": ROOT / "results/ops_reporter_specialists_v1",
    "gbdt": ROOT / "results/ops_reporter_specialists_gbdt_v1",
    "catboost_multirmse": (
        ROOT / "results/ops_reporter_specialists_catboost_multirmse_v1"
    ),
}


def stable_seed(*values: Any) -> int:
    digest = hashlib.sha256("\0".join(map(str, values)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31 - 1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def load_reporter_row(path: Path, slug: str) -> pd.Series:
    table = pd.read_csv(path)
    if len(table) != 52 or table["reporter_slug"].nunique() != 52:
        raise RuntimeError("Frozen reporter table must contain exactly 52 reporters")
    matched = table.loc[table["reporter_slug"].astype(str) == str(slug)]
    if len(matched) != 1:
        raise ValueError(f"Reporter {slug!r} matched {len(matched)} frozen rows")
    return matched.iloc[0]


def screen_code(categories: Mapping[str, Sequence[str]], screen_id: str) -> int:
    values = tuple(map(str, categories["screen_code"]))
    try:
        return values.index(str(screen_id))
    except ValueError as error:
        raise ValueError(f"Unknown frozen screen ID: {screen_id}") from error


def reporter_screen_ids(row: pd.Series) -> tuple[str, ...]:
    return tuple(token for token in str(row.screen_ids).split(";") if token)


def build_partition(
    *,
    protocol: str,
    phase_cache: PhaseCache,
    phase_rows: np.ndarray,
    is_control: np.ndarray,
    destination_screen: str | None,
    outer_gene_fold: int,
    source_validation_field_fold: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    n_folds = int(phase_cache.manifest["n_folds"])
    gene_fold = np.asarray(
        phase_cache.folds["gene_holdout_main"][phase_rows], dtype=np.int16
    )
    field_fold = np.asarray(
        phase_cache.folds["field_holdout_sanity"][phase_rows], dtype=np.int16
    )
    screens = np.asarray(phase_cache.metadata["screen_code"][phase_rows], dtype=np.int32)
    genes = np.asarray(phase_cache.metadata["gene_code"][phase_rows], dtype=np.int32)

    if protocol == "gene_holdout_main":
        if not 0 <= outer_gene_fold < n_folds:
            raise ValueError("gene_holdout_main requires --outer-gene-fold in [0,4]")
        train, validation, test, validation_fold = partition_indices(
            gene_fold, outer_gene_fold, n_folds
        )
        details = {
            "outer_gene_fold": int(outer_gene_fold),
            "validation_gene_fold": int(validation_fold),
            "destination_screen": None,
            "destination_screen_code": None,
            "whole_destination_screen_removed_from_train_and_validation": False,
        }
    else:
        if destination_screen is None:
            raise ValueError(f"{protocol} requires --destination-screen")
        destination_code = screen_code(phase_cache.categories, destination_screen)
        destination = screens == destination_code
        source = ~destination
        if protocol == "strict_whole_screen":
            if outer_gene_fold != -1:
                raise ValueError("strict_whole_screen requires outer_gene_fold=-1")
            if not 0 <= source_validation_field_fold < n_folds:
                raise ValueError("source validation field fold must be in [0,4]")
            train = np.flatnonzero(source & (field_fold != source_validation_field_fold))
            validation = np.flatnonzero(
                source & (field_fold == source_validation_field_fold)
            )
            test = np.flatnonzero(destination)
            details = {
                "outer_gene_fold": None,
                "validation_gene_fold": None,
                "source_validation_field_fold": int(source_validation_field_fold),
                "destination_screen": destination_screen,
                "destination_screen_code": int(destination_code),
                "whole_destination_screen_removed_from_train_and_validation": True,
                "test_gene_policy": "seen_in_source_training",
            }
        elif protocol == "strict_gene_screen":
            if not 0 <= outer_gene_fold < n_folds:
                raise ValueError("strict_gene_screen requires --outer-gene-fold in [0,4]")
            validation_fold = (outer_gene_fold + 1) % n_folds
            train = np.flatnonzero(
                source & (gene_fold != outer_gene_fold) & (gene_fold != validation_fold)
            )
            validation = np.flatnonzero(source & (gene_fold == validation_fold))
            test = np.flatnonzero(destination & (gene_fold == outer_gene_fold))
            details = {
                "outer_gene_fold": int(outer_gene_fold),
                "validation_gene_fold": int(validation_fold),
                "destination_screen": destination_screen,
                "destination_screen_code": int(destination_code),
                "whole_destination_screen_removed_from_train_and_validation": True,
                "test_gene_policy": "held_out_from_every_source_screen",
            }
        else:  # pragma: no cover - argparse closes this branch
            raise ValueError(protocol)

    for name, values in (("train", train), ("validation", validation), ("test", test)):
        if len(values) == 0:
            raise RuntimeError(f"Empty {name} partition for {protocol}")
    if set(train.tolist()) & set(validation.tolist()):
        raise RuntimeError("train/validation overlap")
    if set(train.tolist()) & set(test.tolist()):
        raise RuntimeError("train/test overlap")
    if set(validation.tolist()) & set(test.tolist()):
        raise RuntimeError("validation/test overlap")

    if protocol != "gene_holdout_main":
        destination_code = int(details["destination_screen_code"])
        if np.any(screens[train] == destination_code) or np.any(
            screens[validation] == destination_code
        ):
            raise RuntimeError("Destination screen leaked into train/validation")
        if not np.all(screens[test] == destination_code):
            raise RuntimeError("Screen-held-out test contains a source-screen row")

    target = ~np.asarray(is_control, dtype=bool)
    train_genes = set(genes[train][target[train]].tolist())
    validation_genes = set(genes[validation][target[validation]].tolist())
    test_genes = set(genes[test][target[test]].tolist())
    if protocol == "strict_whole_screen" and not test_genes.issubset(train_genes):
        missing = sorted(test_genes - train_genes)
        raise RuntimeError(
            f"Seen-gene screen protocol has {len(missing)} unseen destination genes"
        )
    if protocol in {"gene_holdout_main", "strict_gene_screen"}:
        if train_genes & test_genes or validation_genes & test_genes:
            raise RuntimeError("Held-out target genes leaked across partitions")

    details.update(
        {
            "n_train": int(len(train)),
            "n_validation": int(len(validation)),
            "n_test": int(len(test)),
            "n_train_targeting": int(target[train].sum()),
            "n_validation_targeting": int(target[validation].sum()),
            "n_test_targeting": int(target[test].sum()),
            "n_train_controls": int((~target[train]).sum()),
            "n_validation_controls": int((~target[validation]).sum()),
            "n_test_controls": int((~target[test]).sum()),
            "n_train_target_genes": int(len(train_genes)),
            "n_validation_target_genes": int(len(validation_genes)),
            "n_test_target_genes": int(len(test_genes)),
            "partition_sha256": hash_arrays(
                phase_rows[train], phase_rows[validation], phase_rows[test]
            ),
        }
    )
    return train, validation, test, details


def evaluate_prediction(
    *,
    output_dir: Path,
    truth_raw: np.ndarray,
    prediction_raw: np.ndarray,
    truth_scaled: np.ndarray,
    prediction_scaled: np.ndarray,
    is_control: np.ndarray,
    screen_codes: np.ndarray,
    gene_codes: np.ndarray,
    feature_names: np.ndarray,
    baseline_mean_raw: np.ndarray,
    seed: int,
    binding: Mapping[str, Any],
    save_predictions: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = ~np.asarray(is_control, dtype=bool)
    if not np.any(target) or not np.any(is_control):
        raise RuntimeError("Evaluation requires targeting and control cells")
    baseline_raw = np.broadcast_to(
        np.asarray(baseline_mean_raw, dtype=np.float32), truth_raw[target].shape
    )
    cell_summary, cell_features = matrix_metrics(
        truth_raw[target], prediction_raw[target], baseline_raw, feature_names
    )
    cell_summary["standardized_mse"] = float(
        np.mean((truth_scaled[target] - prediction_scaled[target]) ** 2)
    )
    standardized_baseline = float(np.mean(truth_scaled[target] ** 2))
    cell_summary["standardized_baseline_mse"] = standardized_baseline
    cell_summary["standardized_gain_vs_train_mean"] = float(
        1.0 - cell_summary["standardized_mse"] / standardized_baseline
    )
    cell_summary["n_targeting_test_cells"] = int(target.sum())

    profiles = aggregate_gene_profiles(
        truth_scaled, prediction_scaled, screen_codes, gene_codes, is_control
    )
    gene_summary, gene_features = profile_metrics(profiles, feature_names)
    gene_summary.update(bootstrap_gene_profile_metrics(profiles, 1000, seed + 17))
    adapted, group_metrics = cell_eval_adapter.phenotype_perturbation_metrics(profiles)
    gene_summary.update(adapted)
    ceiling_seed = seed + 31
    ceiling = cell_eval_adapter.bootstrap_data_ceiling_profiles(
        truth_scaled, screen_codes, gene_codes, is_control, ceiling_seed
    )
    if not (
        np.array_equal(ceiling.screen_code, profiles.screen_code)
        and np.array_equal(ceiling.gene_code, profiles.gene_code)
    ):
        raise RuntimeError("Data-ceiling profile order drifted")
    ceiling_summary, ceiling_groups = cell_eval_adapter.phenotype_perturbation_metrics(
        ceiling
    )
    gene_summary.update(
        {
            f"cell_eval_ceiling_{key.removeprefix('cell_eval_')}": value
            for key, value in ceiling_summary.items()
        }
    )
    adapter = cell_eval_adapter.adapter_manifest()
    adapter["binding"] = {
        **dict(binding),
        "ceiling_seed": int(ceiling_seed),
        "n_gene_screen_groups": int(len(profiles.truth)),
        "n_endpoints": int(len(feature_names)),
        "outer_test_only": True,
        "used_for_training_checkpoint_or_state_selection": False,
    }
    cell_eval_adapter.validate_manifest(adapter)

    atomic_npy(output_dir / "truth_raw.npy", truth_raw)
    atomic_npy(output_dir / "test_is_control.npy", is_control)
    atomic_npy(output_dir / "gene_truth_control_relative_scaled.npy", profiles.truth)
    atomic_npy(
        output_dir / "gene_prediction_control_relative_scaled.npy", profiles.prediction
    )
    atomic_npy(output_dir / "gene_screen_code.npy", profiles.screen_code)
    atomic_npy(output_dir / "gene_gene_code.npy", profiles.gene_code)
    atomic_npy(output_dir / "gene_n_cells.npy", profiles.n_cells)
    atomic_npy(output_dir / "gene_n_control_cells.npy", profiles.n_control_cells)
    atomic_npy(
        output_dir / "cell_eval_ceiling_half_a_control_relative_scaled.npy",
        ceiling.truth,
    )
    atomic_npy(
        output_dir / "cell_eval_ceiling_half_b_control_relative_scaled.npy",
        ceiling.prediction,
    )
    if save_predictions:
        atomic_npy(output_dir / "test_prediction_raw.npy", prediction_raw)
    atomic_text(
        output_dir / "target_feature_names.txt",
        "\n".join(feature_names.astype(str)) + "\n",
    )
    atomic_csv(output_dir / "cell_feature_metrics.csv", cell_features)
    atomic_csv(output_dir / "gene_feature_metrics.csv", gene_features)
    atomic_csv(output_dir / "gene_group_metrics.csv", group_metrics)
    atomic_csv(output_dir / "cell_eval_ceiling_group_metrics.csv", ceiling_groups)
    atomic_json(output_dir / "cell_eval_adapter_manifest.json", adapter)
    artifact_names = [
        "truth_raw.npy",
        "test_is_control.npy",
        "gene_truth_control_relative_scaled.npy",
        "gene_prediction_control_relative_scaled.npy",
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
    if save_predictions:
        artifact_names.append("test_prediction_raw.npy")
    return {
        "cell_metrics": cell_summary,
        "gene_metrics": gene_summary,
        "evaluator_id": cell_eval_adapter.EVALUATOR_ID,
        "evaluation_artifact_sha256": {
            name: file_sha256(output_dir / name) for name in artifact_names
        },
    }


def legacy_root(model: str, args: argparse.Namespace) -> Path:
    override = getattr(args, f"legacy_{model}_root", None)
    return Path(override) if override else DEFAULT_LEGACY_ROOTS[model]


def reevaluate_gene_model(
    *,
    args: argparse.Namespace,
    model: str,
    reporter: str,
    fold: int,
    expected_phase_rows: np.ndarray,
    destination: Path,
    seed: int,
) -> dict[str, Any]:
    root = legacy_root(model, args)
    partition = root / "gene_holdout_main" / f"fold_{fold}" / reporter
    shared = partition / "shared"
    model_dir = partition / "models" / model
    complete_path = model_dir / "complete.json"
    prediction_path = model_dir / "test_prediction_raw.npy"
    required = [
        shared / "manifest.json",
        shared / "test_phase_row_index.npy",
        shared / "truth_raw.npy",
        shared / "test_is_control.npy",
        shared / "test_screen_code.npy",
        shared / "test_gene_code.npy",
        shared / "target_feature_names.txt",
        complete_path,
        prediction_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Legacy gene artifacts missing for {model}: {missing}")
    phase_rows = np.load(shared / "test_phase_row_index.npy")
    if not np.array_equal(phase_rows, expected_phase_rows):
        raise RuntimeError(f"Frozen gene test cohort drift for {reporter}/{model}/fold{fold}")
    manifest = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))
    preprocessing = manifest["preprocessing"]
    y_mean = np.asarray(preprocessing["y_mean"], dtype=np.float32)
    y_scale = np.asarray(preprocessing["y_scale"], dtype=np.float32)
    truth_raw = np.asarray(np.load(shared / "truth_raw.npy"), dtype=np.float32)
    prediction_raw = np.asarray(np.load(prediction_path), dtype=np.float32)
    truth_scaled = (truth_raw - y_mean) / y_scale
    prediction_scaled = (prediction_raw - y_mean) / y_scale
    feature_names = np.asarray(
        (shared / "target_feature_names.txt").read_text(encoding="utf-8").splitlines(),
        dtype=str,
    )
    evaluated = evaluate_prediction(
        output_dir=destination,
        truth_raw=truth_raw,
        prediction_raw=prediction_raw,
        truth_scaled=truth_scaled,
        prediction_scaled=prediction_scaled,
        is_control=np.load(shared / "test_is_control.npy"),
        screen_codes=np.load(shared / "test_screen_code.npy"),
        gene_codes=np.load(shared / "test_gene_code.npy"),
        feature_names=feature_names,
        baseline_mean_raw=y_mean,
        seed=seed,
        binding={
            "reporter_slug": reporter,
            "protocol": "gene_holdout_main",
            "outer_gene_fold": int(fold),
            "source_model": model,
        },
        save_predictions=args.save_predictions,
    )
    source_complete = json.loads(complete_path.read_text(encoding="utf-8"))
    return {
        **evaluated,
        "execution_mode": "read_only_legacy_prediction_reevaluation",
        "source_complete": str(complete_path.resolve()),
        "source_complete_sha256": file_sha256(complete_path),
        "source_prediction_sha256": file_sha256(prediction_path),
        "source_hyperparameters": source_complete.get("hyperparameters"),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--task-json", type=Path)
    result.add_argument("--reporter", default="")
    result.add_argument("--protocol", choices=PROTOCOLS)
    result.add_argument("--destination-screen", default="")
    result.add_argument("--outer-gene-fold", type=int, default=-1)
    result.add_argument("--source-validation-field-fold", type=int, default=0)
    result.add_argument("--models", default=",".join(VALID_MODELS))
    result.add_argument(
        "--gene-mode", choices=("reevaluate", "retrain", "train"), default="reevaluate"
    )
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--phase-cache", type=Path, default=DEFAULT_PHASE_CACHE)
    result.add_argument("--exact-cache-root", type=Path, default=DEFAULT_EXACT_ROOT)
    result.add_argument("--target-table", type=Path, default=DEFAULT_TARGET_TABLE)
    result.add_argument(
        "--target-feature-dictionary", type=Path, default=DEFAULT_FEATURE_DICTIONARY
    )
    result.add_argument("--device", choices=("cuda", "auto", "cpu"), default="cuda")
    result.add_argument("--workers", type=int, default=16)
    result.add_argument("--seed", type=int, default=20260722)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument(
        "--partition-only",
        action="store_true",
        help="Build and validate the real partition, then stop before fitting/evaluation.",
    )
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--save-predictions", action="store_true", default=True)
    result.add_argument("--min-x-finite-fraction", type=float, default=0.80)
    result.add_argument("--ridge-alphas", default="0.1,1,10,100,1000")
    result.add_argument("--knn-k", default="5,15,50,100")
    result.add_argument("--knn-backend", choices=("faiss", "torch", "sklearn", "auto"), default="torch")
    result.add_argument("--knn-query-batch-size", type=int, default=1024)
    result.add_argument("--knn-train-block-size", type=int, default=65536)
    result.add_argument("--allow-slow-knn", action="store_true")
    result.add_argument("--mlp-hidden", default="256,128")
    result.add_argument("--mlp-batch-size", type=int, default=1024)
    result.add_argument("--mlp-epochs", type=int, default=100)
    result.add_argument("--mlp-patience", type=int, default=10)
    result.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    result.add_argument("--mlp-weight-decay", type=float, default=1e-4)
    result.add_argument("--gbdt-backend", choices=("auto", "xgboost", "sklearn"), default="xgboost")
    result.add_argument("--gbdt-max-depth", type=int, default=6)
    result.add_argument("--gbdt-estimators", type=int, default=300)
    result.add_argument("--gbdt-learning-rate", type=float, default=0.05)
    result.add_argument("--gbdt-patience", type=int, default=20)
    result.add_argument("--catboost-iterations", type=int, default=300)
    result.add_argument("--catboost-depth", type=int, default=6)
    result.add_argument("--catboost-learning-rate", type=float, default=0.05)
    result.add_argument("--catboost-patience", type=int, default=20)
    result.add_argument("--catboost-l2-leaf-reg", type=float, default=3.0)
    result.add_argument("--catboost-border-count", type=int, default=128)
    result.add_argument("--catboost-devices", default="0")
    result.add_argument("--catboost-gpu-ram-part", type=float, default=0.80)
    for model, default_root in DEFAULT_LEGACY_ROOTS.items():
        result.add_argument(
            f"--legacy-{model.replace('_', '-')}-root",
            dest=f"legacy_{model}_root",
            type=Path,
            default=default_root,
        )
    return result


def normalize_training_args(args: argparse.Namespace) -> None:
    args.ridge_alphas = [float(value) for value in args.ridge_alphas.split(",")]
    args.knn_k = [int(value) for value in args.knn_k.split(",")]
    args.mlp_hidden = [int(value) for value in args.mlp_hidden.split(",")]


def bind_task_spec(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.task_json is None:
        if not args.reporter or args.protocol is None:
            raise ValueError("Either --task-json or both --reporter/--protocol are required")
        return None
    payload = json.loads(args.task_json.read_text(encoding="utf-8"))
    if payload.get("schema_version") != TASK_SPEC_SCHEMA:
        raise ValueError(f"Wrong task-spec schema: {args.task_json}")
    scientific = payload.get("scientific_task")
    contract = payload.get("execution_contract")
    if not isinstance(scientific, dict) or not isinstance(contract, dict):
        raise ValueError("Task spec lacks scientific_task/execution_contract")
    if str(payload.get("task_payload_sha256")) != json_sha256(scientific):
        raise ValueError("Task spec scientific payload SHA256 mismatch")
    strict_plan = Path(str(payload.get("strict_plan", "")))
    if not strict_plan.is_file() or file_sha256(strict_plan) != str(
        payload.get("strict_plan_sha256")
    ):
        raise ValueError("Task spec strict plan path/SHA256 mismatch")
    task_type = scientific.get("task_type")
    protocol = {
        "gene": "gene_holdout_main",
        "screen": "strict_whole_screen",
        "joint": "strict_gene_screen",
    }.get(task_type)
    if protocol is None:
        raise ValueError(f"Unknown task_type in task spec: {task_type}")
    options = scientific.get("task_options", {})
    if not isinstance(options, dict):
        raise ValueError("Task spec task_options must be an object")
    if task_type in {"screen", "joint"} and options.get("calibration") != "zero_shot":
        raise ValueError("Strict screen tasks require zero-shot destination calibration")
    expected_models = ",".join(map(str, contract.get("models", [])))
    if expected_models and args.models != expected_models:
        raise ValueError("CLI --models drifted from frozen task execution contract")
    if args.gene_mode != str(contract.get("gene_mode")):
        raise ValueError("CLI --gene-mode drifted from frozen task execution contract")
    if args.device != contract.get("device") or args.workers != int(
        contract.get("workers", -1)
    ):
        raise ValueError("CLI device/workers drifted from frozen task execution contract")
    if contract.get("cpu_fallback_allowed") is not False:
        raise ValueError("Strict task contract must forbid CPU fallback")
    args.reporter = str(scientific["reporter_slug"])
    args.protocol = protocol
    args.destination_screen = str(scientific.get("destination_screen") or "")
    args.outer_gene_fold = int(
        scientific["gene_fold"] if scientific.get("gene_fold") is not None else -1
    )
    args.seed = int(scientific["seed"])
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    task_spec = bind_task_spec(args)
    if args.gene_mode == "train":
        args.gene_mode = "retrain"
    normalize_training_args(args)
    models = parse_csv_choice(args.models, VALID_MODELS, "model")
    if args.protocol != "gene_holdout_main" and args.gene_mode != "retrain":
        raise ValueError("Strict screen protocols require --gene-mode retrain")
    row = load_reporter_row(args.target_table, args.reporter)
    screens = reporter_screen_ids(row)
    destination = args.destination_screen or None
    if task_spec is not None:
        scientific = task_spec["scientific_task"]
        frozen_sources = tuple(sorted(map(str, scientific.get("source_screens", []))))
        expected_sources = (
            tuple()
            if args.protocol == "gene_holdout_main"
            else tuple(sorted(screen for screen in screens if screen != destination))
        )
        if frozen_sources != expected_sources:
            raise ValueError(
                "Frozen task source_screens drifted from reporter metadata: "
                f"{frozen_sources} != {expected_sources}"
            )
    if args.protocol != "gene_holdout_main":
        if len(screens) < 2:
            raise ValueError(f"Single-screen reporter {args.reporter} has no screen-transfer task")
        if destination not in screens:
            raise ValueError(
                f"Destination {destination!r} is not a frozen screen for {args.reporter}: {screens}"
            )

    external_gene_mode = "train" if args.gene_mode == "retrain" else args.gene_mode
    identity = {
        "schema_version": SCHEMA_VERSION,
        "reporter_slug": args.reporter,
        "protocol": args.protocol,
        "destination_screen": destination,
        "outer_gene_fold": (
            int(args.outer_gene_fold) if args.outer_gene_fold >= 0 else None
        ),
        "models": models,
        "gene_mode": external_gene_mode,
        "seed": int(args.seed),
        "evaluator_id": cell_eval_adapter.EVALUATOR_ID,
    }
    print(json.dumps(identity, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    final_path = args.output_root / "training_result.json"
    if final_path.exists() and not args.overwrite:
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in identity.items()) and existing.get(
            "completed"
        ) is True:
            print(f"SKIP complete task: {final_path}", flush=True)
            return 0
        raise RuntimeError(f"Incompatible completion marker: {final_path}")

    table = pd.read_csv(args.target_table)
    core = load_technical_core_features(args.target_feature_dictionary, table)
    cache_path = args.exact_cache_root / f"all_cells_fluor_{args.reporter}.exact.h5"
    data = load_reporter_data(cache_path, args.reporter, core[args.reporter])
    phase_cache = PhaseCache.open(args.phase_cache)
    phase_metadata = {
        name: np.asarray(values[data.phase_rows])
        for name, values in phase_cache.metadata.items()
        if name != "is_target"
    }
    train, validation, test, partition = build_partition(
        protocol=args.protocol,
        phase_cache=phase_cache,
        phase_rows=data.phase_rows,
        is_control=data.is_control,
        destination_screen=destination,
        outer_gene_fold=args.outer_gene_fold,
        source_validation_field_fold=args.source_validation_field_fold,
    )
    atomic_json(args.output_root / "partition_manifest.json", {**identity, **partition})
    atomic_npy(args.output_root / "test_phase_row_index.npy", data.phase_rows[test])
    if args.partition_only:
        print(json.dumps(partition, indent=2, sort_keys=True), flush=True)
        return 0
    model_results: dict[str, Any] = {}
    started = time.time()

    if args.protocol == "gene_holdout_main" and args.gene_mode == "reevaluate":
        for model in models:
            print(f"REEVALUATE {args.reporter} fold={args.outer_gene_fold} {model}", flush=True)
            model_seed = stable_seed(args.seed, args.reporter, args.protocol, args.outer_gene_fold, model)
            destination_dir = args.output_root / "models" / model
            result = reevaluate_gene_model(
                args=args,
                model=model,
                reporter=args.reporter,
                fold=args.outer_gene_fold,
                expected_phase_rows=data.phase_rows[test],
                destination=destination_dir,
                seed=model_seed,
            )
            model_complete = {**identity, "model": model, **result, "completed": True}
            atomic_json(destination_dir / "complete.json", model_complete)
            model_results[model] = {
                "complete": str((destination_dir / "complete.json").resolve()),
                "complete_sha256": file_sha256(destination_dir / "complete.json"),
                "execution_mode": result["execution_mode"],
            }
    else:
        dependencies = dependency_status()
        preflight_dependencies(models, args, dependencies)
        resolved_device = choose_torch_device(args.device)
        if any(model != "ridge" for model in models) and resolved_device != "cuda":
            raise RuntimeError("GPU specialist task forbids CPU fallback")
        x_all = np.asarray(phase_cache.x[data.phase_rows], dtype=np.float32)
        preprocessing = fit_preprocessing(
            x_all[train], data.y[train], args.min_x_finite_fraction
        )
        x_train = preprocessing.transform_x(x_all[train])
        x_validation = preprocessing.transform_x(x_all[validation])
        x_test = preprocessing.transform_x(x_all[test])
        y_train = preprocessing.transform_y(data.y[train])
        y_validation = preprocessing.transform_y(data.y[validation])
        truth_scaled = preprocessing.transform_y(data.y[test])
        truth_raw = np.asarray(data.y[test][:, preprocessing.y_keep], dtype=np.float32)
        feature_names = data.target_feature_names[preprocessing.y_keep]
        atomic_json(args.output_root / "preprocessing.json", preprocessing.to_json())
        for model in models:
            destination_dir = args.output_root / "models" / model
            complete_path = destination_dir / "complete.json"
            if complete_path.exists() and not args.overwrite:
                observed = json.loads(complete_path.read_text(encoding="utf-8"))
                if observed.get("completed") is True and all(
                    observed.get(key) == value for key, value in identity.items()
                ):
                    model_results[model] = {
                        "complete": str(complete_path.resolve()),
                        "complete_sha256": file_sha256(complete_path),
                        "execution_mode": "trained_strict_partition",
                    }
                    continue
                raise RuntimeError(f"Incompatible model marker: {complete_path}")
            model_seed = stable_seed(
                args.seed,
                args.reporter,
                args.protocol,
                destination,
                args.outer_gene_fold,
                model,
            )
            print(
                f"TRAIN {args.reporter} {args.protocol} destination={destination} "
                f"fold={args.outer_gene_fold} model={model}",
                flush=True,
            )
            model_started = time.time()
            prediction_scaled, hyperparameters = run_model(
                model,
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
            evaluated = evaluate_prediction(
                output_dir=destination_dir,
                truth_raw=truth_raw,
                prediction_raw=prediction_raw,
                truth_scaled=truth_scaled,
                prediction_scaled=prediction_scaled,
                is_control=data.is_control[test],
                screen_codes=phase_metadata["screen_code"][test],
                gene_codes=phase_metadata["gene_code"][test],
                feature_names=feature_names,
                baseline_mean_raw=preprocessing.y_mean,
                seed=model_seed,
                binding={
                    "reporter_slug": args.reporter,
                    "protocol": args.protocol,
                    "destination_screen": destination,
                    "outer_gene_fold": identity["outer_gene_fold"],
                    "source_model": model,
                },
                save_predictions=args.save_predictions,
            )
            complete = {
                **identity,
                "model": model,
                "completed": True,
                "execution_mode": "trained_strict_partition",
                "device_resolved": "cpu" if model == "ridge" else resolved_device,
                "runtime_seconds": time.time() - model_started,
                "hyperparameters": hyperparameters,
                **evaluated,
            }
            atomic_json(complete_path, complete)
            model_results[model] = {
                "complete": str(complete_path.resolve()),
                "complete_sha256": file_sha256(complete_path),
                "execution_mode": complete["execution_mode"],
            }

    result = {
        **identity,
        "completed": True,
        "runtime_seconds": time.time() - started,
        "partition_manifest": str((args.output_root / "partition_manifest.json").resolve()),
        "partition_manifest_sha256": file_sha256(args.output_root / "partition_manifest.json"),
        "model_results": model_results,
        "selection_used_outer_test": False,
        "test_evaluated": True,
    }
    if task_spec is not None:
        scientific = task_spec["scientific_task"]
        task_type = str(scientific["task_type"])
        import torch

        visible_cuda = int(torch.cuda.device_count())
        result.update(
            {
                "schema_version": TASK_RESULT_SCHEMA,
                "status": "complete",
                "method_id": "independent_specialists",
                "mode": "formal",
                "task_id": str(scientific["task_id"]),
                "task_type": task_type,
                "reporters_argument": args.reporter,
                "split": {
                    "gene": "gene_holdout_main",
                    "screen": "whole_screen_holdout",
                    "joint": "gene_x_screen_holdout",
                }[task_type],
                "fold": scientific.get("gene_fold"),
                "source_screens": list(scientific.get("source_screens", [])),
                "gene_fold": scientific.get("gene_fold"),
                "n_models": len(models),
                "device": args.device,
                "workers": int(args.workers),
                "cuda_available": visible_cuda == 1,
                "gpu_training_performed": task_type != "gene",
                "visible_cuda_device_count": visible_cuda,
                "cpu_fallback_allowed": False,
                "strict_plan_sha256": str(task_spec["strict_plan_sha256"]),
                "task_payload_sha256": str(task_spec["task_payload_sha256"]),
                "selection_used_test": False,
                "outer_test_used_for_training_or_selection": False,
            }
        )
    atomic_json(final_path, result)
    print(f"COMPLETE {final_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
