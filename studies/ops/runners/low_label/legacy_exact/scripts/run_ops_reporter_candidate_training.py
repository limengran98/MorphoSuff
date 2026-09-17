#!/usr/bin/env python3
"""Train one OPS reporter candidate under the common frozen harness.

This is the single-job runner used by the serial campaign launcher.  It owns
one method/split/fold/trial and deliberately has no process-level parallelism.
All methods share the same split construction, train-only preprocessing,
reporter-balanced masked loss, reporter exposure, validation schedule,
checkpoint objective, and downstream test metrics.

Formal jobs are CUDA-only and fail closed.  Development jobs never evaluate
the outer test set.  Single-checkpoint, EMA, and checkpoint-averaged states are
constructed only after the validation-selected checkpoint is frozen; test
metrics cannot choose among them.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import errno
import fcntl
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import h5py

import ops_reporter_candidate_models as candidate_models
import ops_reporter_masked_multitask_v2_lib as sampler_library
import ops_reporter_runtime_metrics as runtime_metrics
import ops_reporter_training_harness_lib as harness
import run_ops_reporter_masked_multitask_resmlp_v2 as frozen_engine
from ops_reporter_specialist_lib import (
    PhaseCache,
    atomic_json,
    hash_arrays,
)
from run_ops_same_cell_falsification import (
    _field_density_and_eccentricity as same_cell_covariates,
    _normal_or_residual_data as same_cell_arm_targets,
    _stable_seed as same_cell_stable_seed,
)


SCHEMA_VERSION = "ops-reporter-candidate-training-v2"
METHOD_MODEL_TYPES = {
    "resmlp_52head": "shared_resmlp",
    "tabm_specialists": "tabm",
    "mmoe_52head": "sparse_mmoe",
    "multitab_pilot": "multitab_column",
    "q_id_52": "endpoint_query_id",
    "q_semantic_52": "endpoint_query_semantic",
}
FULL52_METHODS = {
    "resmlp_52head",
    "tabm_specialists",
    "mmoe_52head",
    "multitab_pilot",
    "q_id_52",
    "q_semantic_52",
}
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
DEFAULT_PHASE_CACHE = Path("data/processed/ops_phase172_indexed")
DEFAULT_EXACT_ROOT = Path("data/processed/ops_full_reporter_exact/reporters")
DEFAULT_TARGET_TABLE = Path("results/ops_phase0_asset_audit/reporter_targets.csv")
DEFAULT_TARGET_FEATURE_DICTIONARY = Path(
    "results/ops_phase0_asset_audit/target_feature_dictionary.csv"
)
DEFAULT_SPECIALIST_REFERENCE = Path("results/ops_reporter_specialists_v1")
DEFAULT_PREPROCESSING_REFERENCE = Path(
    "results/ops_reporter_masked_multitask_resmlp_v1"
)
DEFAULT_CAMPAIGN_CONFIG = Path(
    "configs/ops_reporter_candidate_training_harness_v2.json"
)
DEFAULT_ENDPOINT_SEMANTICS = Path(
    "results/ops_reporter_endpoint_semantic_vocabulary_v1/endpoint_semantics.csv"
)
DEFAULT_SEMANTIC_VOCABULARY = Path(
    "results/ops_reporter_endpoint_semantic_vocabulary_v1/semantic_vocabulary.json"
)
DEFAULT_SEMANTIC_MANIFEST = Path(
    "results/ops_reporter_endpoint_semantic_vocabulary_v1/manifest.json"
)
DEFAULT_TABM_DEPENDENCY_WHEEL = Path(
    os.environ.get(
        "OPS_TABM_DEPENDENCY_WHEEL",
        Path(__file__).resolve().parents[1] / "external" / "original_methods" / "tabm" /
        "rtdl_num_embeddings-0.0.12-py3-none-any.whl",
    )
)
DEFAULT_TABM_DEPENDENCY_SHA256 = (
    "87fd61270118915cf40888f2164f3a1f0353edf702f15fd91e71a147d23c930d"
)
EXPECTED_PHASE_SHAPE = (8_410_291, 172)
EXPECTED_ENDPOINT_DISTRIBUTION = {20: 1, 24: 38, 30: 6, 60: 1, 72: 6}
SAME_CELL_ROBUSTNESS_ARMS = (
    "exact_pair",
    "gene_screen_deranged",
    "covariate_matched_deranged",
    "size_shape_only",
    "remove_size_shape",
)


def apply_same_cell_training_arm(
    args: argparse.Namespace,
    phase_cache: PhaseCache,
    heads: Sequence[frozen_engine.HeadPartition],
) -> dict[str, Any]:
    """Apply the frozen intervention to sealed train/validation labels only.

    Outer-test labels remain physically unopened until checkpoint selection.
    The destructive null therefore asks whether a model trained on broken
    phase--phenotype pairings can recover the unchanged exact outer test.
    """

    arm = str(args.same_cell_arm)
    audits: dict[str, Any] = {}
    if arm not in {"gene_screen_deranged", "covariate_matched_deranged"}:
        return {"arm": arm, "label_intervention": "none", "reporters": audits}
    for head in heads:
        rows = np.asarray(head.data.phase_rows, dtype=np.int64)
        gene = np.asarray(phase_cache.metadata["gene_code"][rows])
        screen = np.asarray(phase_cache.metadata["screen_code"][rows])
        _, _, _, density_audit, covariates = same_cell_covariates(
            np.asarray(phase_cache.x[rows], dtype=np.float32),
            screen,
            np.asarray(phase_cache.metadata["well_code"][rows]),
            np.asarray(phase_cache.metadata["tile_code"][rows]),
        )
        train = np.asarray(head.complete_train_indices, dtype=np.int64)
        validation = np.asarray(head.complete_validation_indices, dtype=np.int64)
        empty = np.empty(0, dtype=np.int64)
        # Reproduce the primary MLP's frozen intervention, regardless of the
        # candidate architecture being trained.
        seed = same_cell_stable_seed(head.slug, args.fold, arm, "mlp")
        y_train, y_validation, _, intervention_audit, residual = same_cell_arm_targets(
            arm,
            head.data.y,
            train,
            validation,
            empty,
            gene,
            screen,
            covariates,
            seed,
        )
        if residual:
            raise RuntimeError("Residual targets are not eligible for the candidate robustness harness")
        head.data.y[train] = y_train
        head.data.y[validation] = y_validation
        audits[head.slug] = {
            "intervention_seed": int(seed),
            "intervention": intervention_audit,
            "field_density": density_audit,
        }
    return {
        "arm": arm,
        "label_intervention": "sealed_train_and_validation_only",
        "outer_test_policy": "unchanged_exact_pairs_opened_only_after_selection",
        "reference_estimator_for_intervention_seed": "mlp",
        "reporters": audits,
    }


def apply_same_cell_phase_arm(
    args: argparse.Namespace,
    staging: frozen_engine.GPUTrainingStaging,
) -> dict[str, Any]:
    """Zero excluded standardized columns while preserving native input shape."""

    arm = str(args.same_cell_arm)
    if staging.phase is None:
        raise RuntimeError("Same-cell candidate robustness requires GPU-resident phase staging")
    if arm == "size_shape_only":
        staging.phase[:, :156].zero_()
        return {"arm": arm, "zeroed_standardized_columns": [0, 156], "retained_columns": [156, 172]}
    if arm == "remove_size_shape":
        staging.phase[:, 156:172].zero_()
        return {"arm": arm, "zeroed_standardized_columns": [156, 172], "retained_columns": [0, 156]}
    return {"arm": arm, "zeroed_standardized_columns": None, "retained_columns": [0, 172]}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def read_json_without_duplicate_keys(path: Path) -> dict[str, Any]:
    """Read a JSON object while rejecting duplicate keys at every depth."""

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    canonical_json(value)
    return value


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_torch_save(payload: Any, path: Path) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def strict_method_config(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("--method-config-json must be strict JSON") from error
    if not isinstance(value, dict):
        raise ValueError("--method-config-json must decode to a mapping")
    unknown = set(value) - {"model", "optimizer", "training"}
    if unknown:
        raise ValueError(f"Unknown method-config sections: {sorted(unknown)}")
    model = value.get("model", {})
    optimizer = value.get("optimizer", {})
    training = value.get("training", {})
    if not all(isinstance(item, dict) for item in (model, optimizer, training)):
        raise ValueError("model/optimizer/training config sections must be mappings")
    if set(optimizer) - {"name", "learning_rate", "weight_decay"}:
        raise ValueError("Unsupported optimizer option")
    if optimizer.get("name", "adamw").casefold() != "adamw":
        raise ValueError("The common harness currently freezes AdamW")
    learning_rate = float(optimizer.get("learning_rate", 5.0e-4))
    weight_decay = float(optimizer.get("weight_decay", 1.0e-4))
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    allowed_training = {
        "batch_heads",
        "observations_per_head",
        "epochs",
        "warmup_epochs",
        "minimum_learning_rate",
        "ema_decay",
        "gradient_clip_norm",
    }
    if set(training) - allowed_training:
        raise ValueError(
            f"Unsupported training options: {sorted(set(training) - allowed_training)}"
        )
    normalized = {
        "model": copy.deepcopy(model),
        "optimizer": {
            "name": "adamw",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        },
        "training": {
            "batch_heads": int(training.get("batch_heads", 4)),
            "observations_per_head": int(
                training.get("observations_per_head", 1024)
            ),
            "epochs": int(training.get("epochs", 100)),
            "warmup_epochs": int(training.get("warmup_epochs", 2)),
            "minimum_learning_rate": float(
                training.get("minimum_learning_rate", 5.0e-6)
            ),
            "ema_decay": float(training.get("ema_decay", 0.999)),
            "gradient_clip_norm": float(training.get("gradient_clip_norm", 1.0)),
        },
    }
    t = normalized["training"]
    if t["batch_heads"] != 4 or t["observations_per_head"] != 1024:
        raise ValueError(
            "Formal harness freezes batch_heads=4 and observations_per_head=1024"
        )
    if t["epochs"] != 100 or t["warmup_epochs"] != 2:
        raise ValueError("Formal harness freezes epochs=100 and warmup_epochs=2")
    if not 0.0 < t["ema_decay"] < 1.0:
        raise ValueError("ema_decay must be in (0, 1)")
    if t["minimum_learning_rate"] <= 0 or t["minimum_learning_rate"] > learning_rate:
        raise ValueError("minimum_learning_rate must be in (0, learning_rate]")
    if t["gradient_clip_norm"] < 0:
        raise ValueError("gradient_clip_norm must be nonnegative")
    canonical_json(normalized)
    return normalized


def bind_frozen_campaign_trial(
    args: argparse.Namespace, method_config: Mapping[str, Any]
) -> dict[str, Any]:
    if not args.campaign_config.is_file():
        raise FileNotFoundError(f"Missing frozen campaign config: {args.campaign_config}")
    payload = read_json_without_duplicate_keys(args.campaign_config)
    if payload.get("schema_version") != "ops-reporter-candidate-campaign-config-v2":
        raise RuntimeError("Unexpected candidate campaign schema")
    if payload.get("frozen") is not True:
        raise RuntimeError("Candidate campaign is not frozen")
    fairness = payload.get("fairness_contract", {})
    if fairness.get("trial_count_per_method") != 4:
        raise RuntimeError("Every method must retain four HPO trials")
    parameter_policy = fairness.get("parameter_policy")
    if parameter_policy != {
        "matching_required": False,
        "parameter_count_is_admission_or_selection_input": False,
        "parameter_limits": None,
        "native_capacity_selected_by_validation": True,
    }:
        raise RuntimeError(
            "Campaign must use validation-selected native capacity without "
            "parameter matching or limits"
        )
    resource_policy = fairness.get("resource_metrics", {})
    if (
        resource_policy.get("role") != "reporting_only"
        or resource_policy.get("used_for_admission_or_hpo_selection") is not False
    ):
        raise RuntimeError("Resource metrics must remain reporting-only")
    if fairness.get("search_seed") != args.seed:
        raise RuntimeError(
            f"Run seed {args.seed} differs from frozen search seed "
            f"{fairness.get('search_seed')}"
        )
    method = payload.get("methods", {}).get(args.method)
    if not isinstance(method, Mapping):
        raise RuntimeError(f"Campaign lacks method {args.method}")
    trials = method.get("trial_configs")
    if not isinstance(trials, list) or len(trials) != 4:
        raise RuntimeError(f"Campaign trial grid changed for {args.method}")
    if args.trial_index >= len(trials):
        raise RuntimeError("trial-index is outside the frozen four-trial grid")
    frozen = trials[args.trial_index]
    normalized_frozen = strict_method_config(
        canonical_json(
            {
                "model": frozen["model"]["options"],
                "optimizer": frozen["optimizer"],
                "training": frozen["training"],
            }
        )
    )
    if normalized_frozen != method_config:
        raise RuntimeError("Method config differs from frozen campaign trial")
    expected_reporters = (
        "all"
        if method["reporters"] == "all"
        else ",".join(str(value) for value in method["reporters"])
    )
    if args.reporters != expected_reporters:
        raise RuntimeError("Reporter argument differs from frozen method scope")
    development = {
        (str(item["split"]), int(item["fold"]))
        for item in fairness.get("development_folds", [])
    }
    if args.mode == "development" and (args.split, args.fold) not in development:
        raise RuntimeError("Development split/fold is outside the shared HPO budget")
    formal = payload.get("formal_evaluation", {})
    if args.mode == "formal":
        if args.method not in formal.get("eligible_methods", []):
            raise RuntimeError("Method is not eligible for formal outer-test evaluation")
        if args.split not in formal.get("splits", []) or args.fold not in formal.get(
            "folds", []
        ):
            raise RuntimeError("Formal split/fold is outside the frozen grid")
    binding = {
        "path": str(args.campaign_config.resolve()),
        "file_sha256": file_sha256(args.campaign_config),
        "semantic_sha256": json_sha256(payload),
        "trial_config_sha256": json_sha256(frozen),
        "method_config_sha256": json_sha256(method_config),
        "trial_count_per_method": 4,
        "shared_development_folds": [list(value) for value in sorted(development)],
        "search_seed": args.seed,
        "parameter_policy": copy.deepcopy(parameter_policy),
        "resource_metrics_policy": copy.deepcopy(resource_policy),
    }
    if args.mode == "formal":
        if len(args.output_root.parents) < 3:
            raise RuntimeError("Formal output root cannot locate method-level selection")
        method_root = args.output_root.parents[2]
        selection_candidates = (
            method_root / "fixed_default_selection_manifest.json",
            method_root / "splitwise_hpo_selection_manifest.json",
            method_root / "hpo_selection_manifest.json",
        )
        existing_selection_paths = [
            path for path in selection_candidates if path.is_file()
        ]
        if not existing_selection_paths:
            raise FileNotFoundError(
                "Formal outer-test access requires exactly one frozen selection "
                f"manifest under {method_root}"
            )
        if len(existing_selection_paths) != 1:
            raise RuntimeError(
                "Formal output has ambiguous selection manifests: "
                + ", ".join(str(path) for path in existing_selection_paths)
            )
        selection_path = existing_selection_paths[0]
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        declared = selection.get("selection_fingerprint_sha256")
        unhashed = dict(selection)
        unhashed.pop("selection_fingerprint_sha256", None)
        if declared != json_sha256(unhashed):
            raise RuntimeError("Selection manifest fingerprint mismatch")
        schema = selection.get("schema_version")
        if schema == "ops-reporter-candidate-fixed-default-selection-v1":
            required = {
                "method_id": args.method,
                "selected_trial_index": args.trial_index,
                "selected_method_config_sha256": json_sha256(method_config),
                "selection_basis": "predeclared_default_no_hpo",
                "fixed_before_outer_test": True,
                "outer_test_evaluated_before_selection": False,
                "outer_test_used_for_selection": False,
            }
            binding["selection_basis"] = "predeclared_default_no_hpo"
        elif schema == "ops-reporter-candidate-splitwise-hpo-selection-v1":
            selected_by_split = selection.get("selected_by_split")
            if not isinstance(selected_by_split, Mapping):
                raise RuntimeError("Splitwise selection lacks selected_by_split")
            split_selection = selected_by_split.get(args.split)
            if not isinstance(split_selection, Mapping):
                raise RuntimeError(
                    f"Splitwise selection lacks a record for {args.split}"
                )
            required = {
                "method_id": args.method,
                "selection_used_validation_only": True,
                "outer_test_evaluated_during_search": False,
                "outer_test_used_for_selection": False,
            }
            split_required = {
                "selected_trial_index": args.trial_index,
                "selected_method_config_sha256": json_sha256(method_config),
                "development_split": args.split,
                "development_fold": 0,
            }
            for key, expected in split_required.items():
                if split_selection.get(key) != expected:
                    raise RuntimeError(
                        f"Formal splitwise selection mismatch for {key}: "
                        f"{split_selection.get(key)!r} != {expected!r}"
                    )
            binding["selection_basis"] = "splitwise_validation_hpo"
            binding["split_selection"] = copy.deepcopy(dict(split_selection))
        elif schema == "ops-reporter-candidate-hpo-selection-v2" or (
            schema is None and selection_path.name == "hpo_selection_manifest.json"
        ):
            required = {
                "method_id": args.method,
                "selected_trial_index": args.trial_index,
                "selected_method_config_sha256": json_sha256(method_config),
                "selection_used_validation_only": True,
                "outer_test_evaluated_during_search": False,
                "outer_test_used_for_selection": False,
            }
            binding["selection_basis"] = "legacy_joint_validation_hpo"
        else:
            raise RuntimeError(f"Unsupported selection manifest schema: {schema!r}")
        for key, expected in required.items():
            if selection.get(key) != expected:
                raise RuntimeError(
                    f"Formal selection mismatch for {key}: "
                    f"{selection.get(key)!r} != {expected!r}"
                )
        binding["selection_manifest"] = str(selection_path.resolve())
        binding["selection_manifest_sha256"] = file_sha256(selection_path)
        binding["selection_manifest_schema"] = schema
        # Preserve the legacy fields for downstream readers of already-frozen
        # candidate artifacts.  They point to the actual immutable selection
        # file even when the selection was a predeclared default rather than HPO.
        binding["hpo_selection_manifest"] = str(selection_path.resolve())
        binding["hpo_selection_manifest_sha256"] = file_sha256(selection_path)
        binding["selection_fingerprint_sha256"] = declared
    return binding


def ensure_tabm_dependency(method: str, wheel: Path) -> None:
    if method != "tabm_specialists":
        return
    if not wheel.is_file():
        # Public installations may install the same released dependency from
        # PyPI instead of retaining the original wheel file.  Version checking
        # preserves the numerical dependency contract without tying execution
        # to the study workstation.
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version("rtdl_num_embeddings")
        except PackageNotFoundError as error:
            raise FileNotFoundError(
                "TabM requires rtdl_num_embeddings==0.0.12 or the original "
                f"checksummed wheel at {wheel}"
            ) from error
        if installed != "0.0.12":
            raise RuntimeError(
                "TabM requires rtdl_num_embeddings==0.0.12; "
                f"found {installed!r}"
            )
        return
    observed = file_sha256(wheel)
    if observed != DEFAULT_TABM_DEPENDENCY_SHA256:
        raise RuntimeError(
            "Frozen TabM dependency SHA256 mismatch: "
            f"observed={observed}, expected={DEFAULT_TABM_DEPENDENCY_SHA256}"
        )
    value = str(wheel.resolve())
    if value not in sys.path:
        sys.path.insert(0, value)


def create_runner_candidate_model(
    factory_config: candidate_models.CandidateModelConfig,
) -> Any:
    """Create a candidate, allowing the frozen MultiTab adapter to cover all52.

    The reusable model module deliberately retains the original 5--12 reporter
    pilot guard.  Formal OPS execution expands only that runner-local guard to
    the complete frozen 52-reporter registry; model construction, sparse
    active-reporter execution, state-dict layout, and all other methods remain
    unchanged.  The class constant is restored immediately after construction.
    """

    if str(factory_config.model_type).casefold() != "multitab_column":
        return candidate_models.create_candidate_model(factory_config)
    n_reporters = len(factory_config.head_dimensions)
    if n_reporters != 52 and not (
        candidate_models.MultiTabColumnPilot.MIN_PANEL_SIZE
        <= n_reporters
        <= candidate_models.MultiTabColumnPilot.MAX_PANEL_SIZE
    ):
        raise ValueError(
            "OPS MultiTab requires either its legacy 5-12 reporter pilot or "
            f"the frozen full52 registry; found {n_reporters}"
        )
    previous_maximum = candidate_models.MultiTabColumnPilot.MAX_PANEL_SIZE
    try:
        candidate_models.MultiTabColumnPilot.MAX_PANEL_SIZE = max(
            previous_maximum, n_reporters
        )
        return candidate_models.create_candidate_model(factory_config)
    finally:
        candidate_models.MultiTabColumnPilot.MAX_PANEL_SIZE = previous_maximum


def choose_reporters(
    method: str, requested: str, frozen_table: pd.DataFrame
) -> pd.DataFrame:
    expected = (
        tuple(frozen_table["reporter_slug"].astype(str))
        if method in FULL52_METHODS
        else PANEL12
    )
    if requested.strip().casefold() == "all":
        requested_slugs = expected
    else:
        requested_slugs = tuple(
            token.strip() for token in requested.split(",") if token.strip()
        )
    if requested_slugs != expected:
        scope = "all frozen 52 reporters" if method in FULL52_METHODS else "frozen panel12"
        raise ValueError(
            f"{method} requires {scope} in frozen order; found {requested_slugs}"
        )
    indexed = frozen_table.set_index("reporter_slug", drop=False)
    missing = [slug for slug in expected if slug not in indexed.index]
    if missing:
        raise RuntimeError(f"Frozen reporter table lacks required slugs: {missing}")
    return pd.DataFrame([indexed.loc[slug] for slug in expected]).reset_index(drop=True)


def validate_frozen_assets(
    phase_cache: PhaseCache,
    frozen_table: pd.DataFrame,
    exact_root: Path,
) -> None:
    if tuple(phase_cache.x.shape) != EXPECTED_PHASE_SHAPE:
        raise RuntimeError(
            f"Frozen phase matrix shape {phase_cache.x.shape} != {EXPECTED_PHASE_SHAPE}"
        )
    distribution = {
        int(key): int(value)
        for key, value in frozen_table["n_technical_core_features"]
        .value_counts()
        .sort_index()
        .items()
    }
    if distribution != EXPECTED_ENDPOINT_DISTRIBUTION:
        raise RuntimeError(
            f"Frozen endpoint distribution changed: {distribution}"
        )
    files = sorted(exact_root.glob("*.exact.h5"))
    if len(files) != 52:
        raise RuntimeError(f"Expected 52 exact H5 files, found {len(files)}")


def exact_cache_path(exact_root: Path, slug: str) -> Path:
    return exact_root / f"all_cells_fluor_{slug}.exact.h5"


def selected_feature_columns(
    source: h5py.File, selected_names: Sequence[str], slug: str
) -> tuple[np.ndarray, np.ndarray]:
    names = frozen_engine.decode(source["features/target_feature_names"][:])
    selected = tuple(str(value) for value in selected_names)
    available = {str(name): index for index, name in enumerate(names)}
    missing = [name for name in selected if name not in available]
    if missing:
        raise RuntimeError(f"Exact cache lacks frozen endpoints for {slug}: {missing}")
    indices = np.asarray([available[name] for name in selected], dtype=np.int64)
    # H5 column order, not dictionary order, remains the scientific schema.
    indices.sort()
    selected_in_h5_order = names[indices]
    if set(selected_in_h5_order.astype(str)) != set(selected):
        raise RuntimeError(f"Ambiguous endpoint selection for {slug}")
    return indices, selected_in_h5_order


def read_h5_rows_columns(
    dataset: h5py.Dataset,
    source_rows: np.ndarray,
    columns: np.ndarray,
    *,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Read only explicit source rows; excluded outer-test rows never enter RAM."""

    source_rows = np.asarray(source_rows, dtype=np.int64)
    if len(source_rows) and (
        np.any(np.diff(source_rows) <= 0)
        or source_rows[0] < 0
        or source_rows[-1] >= dataset.shape[0]
    ):
        raise RuntimeError("H5 source rows must be sorted, unique, and in range")
    result = np.empty((len(source_rows), len(columns)), dtype=np.float32)
    source_chunk = int(chunk_rows or (dataset.chunks or (8192,))[0])
    cursor = 0
    for source_start in range(0, dataset.shape[0], source_chunk):
        source_stop = min(source_start + source_chunk, dataset.shape[0])
        stop = int(np.searchsorted(source_rows, source_stop, side="left"))
        if stop == cursor:
            continue
        # h5py permits one fancy axis.  Read all source columns for the
        # selected non-test rows, then retain the frozen technical core.
        block = np.asarray(dataset[source_rows[cursor:stop], :], dtype=np.float32)
        result[cursor:stop] = block[:, columns]
        cursor = stop
    if cursor != len(source_rows):
        raise RuntimeError("H5 row reader did not consume the requested selection")
    return result


def load_sealed_training_data(
    cache_path: Path,
    slug: str,
    selected_names: Sequence[str],
    phase_cache: PhaseCache,
    split_name: str,
    outer_fold: int,
) -> frozen_engine.MaskedReporterData:
    """Load train+inner-validation Y while physically sealing outer-test Y."""

    with h5py.File(cache_path, "r") as source:
        phase_rows_all = np.asarray(source["phase_row_index"][:], dtype=np.int64)
        control_all = np.asarray(source["metadata/is_control"][:], dtype=bool)
        columns, names = selected_feature_columns(source, selected_names, slug)
        fold_values = np.asarray(
            phase_cache.folds[split_name][phase_rows_all], dtype=np.uint8
        )
        source_rows = np.flatnonzero(fold_values != outer_fold).astype(np.int64)
        y = read_h5_rows_columns(source["fluorescence"], source_rows, columns)
        source_target_features = int(source["fluorescence"].shape[1])
        n_source_rows = int(source["fluorescence"].shape[0])
    phase_rows = phase_rows_all[source_rows]
    is_control = control_all[source_rows]
    observed = np.isfinite(y)
    any_observed = observed.any(axis=1)
    complete = observed.all(axis=1)
    # The first architecture comparison is complete-core.  Drop partial rows
    # before GPU staging rather than carrying masked-but-inactive targets.
    retained = complete
    retained_source = source_rows[retained]
    if len(np.unique(phase_rows[retained])) != int(retained.sum()):
        raise RuntimeError(f"Reporter cache contains duplicate phase rows: {slug}")
    return frozen_engine.MaskedReporterData(
        slug=slug,
        source_h5_rows=retained_source,
        phase_rows=phase_rows[retained],
        y=y[retained],
        observed_mask=observed[retained],
        is_complete=np.ones(int(retained.sum()), dtype=bool),
        is_control=is_control[retained],
        target_feature_names=names,
        source_cache=cache_path,
        source_size_bytes=cache_path.stat().st_size,
        source_target_features=source_target_features,
        n_source_rows=n_source_rows,
        n_complete_rows=int(complete.sum()),
        n_partial_rows=int((any_observed & ~complete).sum()),
        n_all_missing_rows=int((~any_observed).sum()),
        all_missing_source_h5_rows=source_rows[~any_observed],
    )


def build_sealed_training_partition(
    args: argparse.Namespace,
    phase_cache: PhaseCache,
    reporter_row: pd.Series,
    data: frozen_engine.MaskedReporterData,
) -> frozen_engine.HeadPartition:
    """Construct complete-core train/validation without any outer-test label."""

    folds = np.asarray(phase_cache.folds[args.split][data.phase_rows], dtype=np.uint8)
    validation_fold = (args.fold + 1) % int(phase_cache.manifest["n_folds"])
    if np.any(folds == args.fold):
        raise RuntimeError("Outer-test row entered sealed training data")
    train_mask = folds != validation_fold
    validation_mask = folds == validation_fold
    complete = np.asarray(data.is_complete, dtype=bool)
    complete_train = np.flatnonzero(train_mask & complete).astype(np.int64)
    partial_train = np.flatnonzero(train_mask & ~complete).astype(np.int64)
    complete_validation = np.flatnonzero(validation_mask & complete).astype(np.int64)
    partial_validation = np.flatnonzero(validation_mask & ~complete).astype(np.int64)
    if min(len(complete_train), len(complete_validation)) <= 0:
        raise RuntimeError(f"Empty sealed train/validation partition for {data.slug}")
    # First architecture comparison is complete-core.  Endpoint-mask code is
    # still exercised by model unit tests and remains available for a later,
    # explicitly isolated data-efficiency experiment.
    active_train = complete_train.copy()
    if not np.all(data.observed_mask[active_train]):
        raise RuntimeError(f"Complete-core training contains missing endpoints: {data.slug}")
    empty = np.empty(0, dtype=np.int64)
    return frozen_engine.HeadPartition(
        slug=data.slug,
        reporter_row=reporter_row,
        data=data,
        complete_train_indices=complete_train,
        partial_train_indices=partial_train,
        active_train_indices=active_train,
        complete_validation_indices=complete_validation,
        excluded_partial_validation_indices=partial_validation,
        complete_test_indices=empty,
        excluded_partial_test_indices=empty,
        validation_fold=validation_fold,
        original_partition_counts={
            "train": int(len(complete_train)),
            "validation": int(len(complete_validation)),
            "test": 0,
            "partial_train_eligible": int(len(partial_train)),
            "partial_validation_excluded": int(len(partial_validation)),
            "partial_test_excluded": 0,
            "outer_test_y_physically_loaded": False,
        },
    )


def frozen_test_reference(
    args: argparse.Namespace, slug: str
) -> tuple[np.ndarray, np.ndarray, Path]:
    root = (
        args.specialist_reference_root
        / args.split
        / f"fold_{args.fold}"
        / slug
        / "shared"
    )
    row_path = root / "test_phase_row_index.npy"
    control_path = root / "test_is_control.npy"
    if not row_path.is_file() or not control_path.is_file():
        raise FileNotFoundError(f"Missing frozen specialist test reference: {root}")
    rows = np.asarray(np.load(row_path, mmap_mode="r"), dtype=np.int64)
    controls = np.asarray(np.load(control_path, mmap_mode="r"), dtype=bool)
    if len(rows) == 0 or len(rows) != len(controls) or len(np.unique(rows)) != len(rows):
        raise RuntimeError(f"Invalid frozen test reference for {slug}")
    return rows, controls, root


def load_reference_preprocessing(
    path: Path,
    heads: Sequence[frozen_engine.HeadPartition],
) -> frozen_engine.GlobalXPreprocessing:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen preprocessing reference: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "ops-reporter-masked-multitask-resmlp-v1":
        raise RuntimeError(f"Unexpected preprocessing schema in {path}")
    source_heads = payload.get("heads")
    if not isinstance(source_heads, Mapping):
        raise RuntimeError(f"Preprocessing reference lacks heads: {path}")
    for head in heads:
        if head.slug not in source_heads:
            raise RuntimeError(f"Preprocessing reference lacks {head.slug}")
        head.y_preprocessing = frozen_engine.HeadYPreprocessing.from_json(
            source_heads[head.slug]
        )
        if head.y_preprocessing.n_fit_rows != len(head.complete_train_indices):
            raise RuntimeError(
                f"Frozen preprocessing train cohort count changed for {head.slug}: "
                f"{head.y_preprocessing.n_fit_rows} != {len(head.complete_train_indices)}"
            )
        if not np.array_equal(
            head.y_preprocessing.kept_indices,
            np.arange(head.data.y.shape[1], dtype=np.int64),
        ):
            raise RuntimeError(f"Frozen preprocessing drops endpoints for {head.slug}")
    x_state = frozen_engine.GlobalXPreprocessing.from_json(payload["x"])
    if not np.array_equal(x_state.kept_indices, np.arange(172, dtype=np.int64)):
        raise RuntimeError("Frozen preprocessing does not retain all 172 phase features")
    return x_state


def preprocessing_manifest(
    reference_path: Path,
    x_state: frozen_engine.GlobalXPreprocessing,
    heads: Sequence[frozen_engine.HeadPartition],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": "frozen_complete_train_only_reference",
        "reference": str(reference_path.resolve()),
        "reference_sha256": file_sha256(reference_path),
        "partial_rows_used_for_fit": False,
        "partial_rows_allowed_for_masked_training": True,
        "x": x_state.to_json(),
        "heads": {
            head.slug: head.y_preprocessing.to_json()  # type: ignore[union-attr]
            for head in heads
        },
    }


def load_query_semantic_binding(
    args: argparse.Namespace,
    heads: Sequence[frozen_engine.HeadPartition],
) -> tuple[dict[str, dict[str, list[int]]], dict[str, int], dict[str, Any]]:
    """Bind query-model endpoint IDs to frozen H5 Y-column order."""

    for path in (
        args.endpoint_semantics,
        args.semantic_vocabulary,
        args.semantic_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen endpoint semantic asset: {path}")
    manifest = read_json_without_duplicate_keys(args.semantic_manifest)
    vocabulary = read_json_without_duplicate_keys(args.semantic_vocabulary)
    table = pd.read_csv(args.endpoint_semantics)
    expected_schema = "ops-reporter-endpoint-semantic-vocabulary-v1"
    if manifest.get("schema_version") != expected_schema or manifest.get("status") != "PASS":
        raise RuntimeError("Endpoint semantic manifest is not a frozen PASS asset")
    if vocabulary.get("schema_version") != expected_schema:
        raise RuntimeError("Endpoint semantic vocabulary schema changed")
    artifacts = manifest.get("artifacts", {})
    if artifacts.get("endpoint_semantics_sha256") != file_sha256(args.endpoint_semantics):
        raise RuntimeError("Endpoint semantic table SHA256 mismatch")
    if artifacts.get("semantic_vocabulary_sha256") != file_sha256(
        args.semantic_vocabulary
    ):
        raise RuntimeError("Endpoint semantic vocabulary SHA256 mismatch")
    fields = tuple(candidate_models.SEMANTIC_QUERY_FIELDS)
    vocabularies = vocabulary.get("vocabularies", {})
    vocab_sizes = {field: len(vocabularies[field]) for field in fields}
    if any(vocabularies[field][0] != "unknown" for field in fields):
        raise RuntimeError("Semantic unknown ID is not frozen at zero")
    expected_total = sum(int(head.data.y.shape[1]) for head in heads)
    if len(table) != expected_total:
        raise RuntimeError(
            f"Semantic endpoint count {len(table)} differs from active schema {expected_total}"
        )
    semantic_ids: dict[str, dict[str, list[int]]] = OrderedDict()
    cursor = 0
    for reporter_index, head in enumerate(heads):
        rows = table.loc[table["reporter_slug"].astype(str) == head.slug].copy()
        rows = rows.sort_values("endpoint_index_within_reporter", kind="mergesort")
        expected_names = head.data.target_feature_names.astype(str).tolist()
        observed_names = rows["target_feature_name"].astype(str).tolist()
        if observed_names != expected_names:
            raise RuntimeError(f"Semantic/H5 endpoint order mismatch for {head.slug}")
        expected_ids = list(range(cursor, cursor + len(expected_names)))
        if rows["global_endpoint_id"].astype(int).tolist() != expected_ids:
            raise RuntimeError(f"Global endpoint IDs drifted for {head.slug}")
        if rows["reporter_id"].astype(int).nunique() != 1 or int(
            rows["reporter_id"].iloc[0]
        ) != reporter_index:
            raise RuntimeError(f"Reporter ID drifted for {head.slug}")
        semantic_ids[head.slug] = {
            field: rows[f"{field}_id"].astype(int).tolist() for field in fields
        }
        cursor += len(expected_names)
    if cursor != expected_total:
        raise RuntimeError("Endpoint semantic cursor did not consume the active schema")
    binding = {
        "schema_version": expected_schema,
        "endpoint_semantics": str(args.endpoint_semantics.resolve()),
        "endpoint_semantics_sha256": file_sha256(args.endpoint_semantics),
        "semantic_vocabulary": str(args.semantic_vocabulary.resolve()),
        "semantic_vocabulary_sha256": file_sha256(args.semantic_vocabulary),
        "semantic_manifest": str(args.semantic_manifest.resolve()),
        "semantic_manifest_sha256": file_sha256(args.semantic_manifest),
        "n_reporters": len(heads),
        "n_endpoints": expected_total,
        "semantic_vocab_sizes": vocab_sizes,
        "unknown_counts": manifest.get("unknown_counts"),
        "endpoint_order": "frozen_reporter_order_then_h5_technical_core_order",
        "metadata_guessing_allowed": False,
    }
    return semantic_ids, vocab_sizes, binding


def cohort_fingerprint(
    head: frozen_engine.HeadPartition, indices: np.ndarray
) -> harness.CohortFingerprint:
    phase_rows = np.asarray(head.data.phase_rows[indices], dtype=np.int64)
    source_rows = np.asarray(head.data.source_h5_rows[indices], dtype=np.int64)
    return harness.CohortFingerprint(
        n_statistical_units=int(len(np.unique(phase_rows))),
        n_observations=int(len(indices)),
        unit_ids_sha256=hash_arrays(np.unique(phase_rows)),
        observation_ids_sha256=hash_arrays(phase_rows, source_rows),
    )


def build_split_contract(
    args: argparse.Namespace,
    heads: Sequence[frozen_engine.HeadPartition],
) -> harness.FrozenSplitContract:
    split_path = args.phase_cache / f"{args.split}.fold.npy"
    return harness.FrozenSplitContract(
        split_name=args.split,
        fold=args.fold,
        split_manifest_sha256=file_sha256(split_path),
        statistical_unit=harness.StatisticalUnitContract(
            unit_name="exact_phase_cell_reporter_observation",
            unit_id_field="phase_row_index",
            metric_unit="cell_then_gene",
            resampling_unit="gene",
            repeated_unit_policy="cluster_by_unit",
        ),
        reporter_cohorts=tuple(
            harness.ReporterCohortContract(
                reporter_slug=head.slug,
                train=cohort_fingerprint(head, head.train_indices),
                validation=cohort_fingerprint(head, head.validation_indices),
                test=(
                    lambda rows: harness.CohortFingerprint(
                        n_statistical_units=int(len(np.unique(rows))),
                        n_observations=int(len(rows)),
                        unit_ids_sha256=hash_arrays(np.unique(rows)),
                        observation_ids_sha256=hash_arrays(rows),
                    )
                )(frozen_test_reference(args, head.slug)[0]),
            )
            for head in heads
        ),
    )


def frozen_rounds_per_epoch(args: argparse.Namespace, n_reporters: int) -> int:
    """Recover V1 reporter rounds without reusing its 52-task step count.

    V1 has 13 physical steps per reporter round (52 reporters / 4 active
    reporters).  The number of rounds varies slightly by split/fold.  A
    12-reporter pilot keeps those rounds and therefore uses three physical
    steps per round, rather than inheriting the 52-task step count.
    """

    # A strict screen specialist keeps the reporter exposure frozen by the
    # gene/field harness, but it does not itself have a V1 split directory.
    # Its runner resolves and records that reference before entering this
    # shared loop.  Existing gene/field jobs never set the override and retain
    # the byte-for-byte path below.
    override = getattr(args, "reporter_rounds_per_epoch_override", None)
    if override is not None:
        rounds = int(override)
        if rounds <= 0:
            raise RuntimeError("Reporter-round override must be positive")
        return rounds

    source = (
        args.preprocessing_reference_root
        / args.split
        / f"fold_{args.fold}"
        / "train"
        / "training_complete.json"
    )
    if not source.is_file():
        raise FileNotFoundError(f"Missing frozen V1 budget reference: {source}")
    marker = json.loads(source.read_text(encoding="utf-8"))
    steps = int(marker["steps_per_epoch"])
    if steps <= 0 or steps % 13:
        raise RuntimeError(
            f"Frozen V1 steps_per_epoch={steps} is not whole 52-task rounds"
        )
    rounds = steps // 13
    if rounds <= 0:
        raise RuntimeError("Frozen V1 reporter rounds must be positive")
    return rounds


def build_update_budget(
    reporters: Sequence[str],
    epochs: int,
    observations_per_head: int,
    batch_heads: int,
    rounds_per_epoch: int,
) -> tuple[harness.OptimizerUpdateBudget, int]:
    if len(reporters) % batch_heads:
        raise RuntimeError(
            "Frozen scopes must divide evenly by batch_heads so every reporter "
            "has exactly one update exposure per epoch"
        )
    steps_per_round = len(reporters) // batch_heads
    steps_per_epoch = steps_per_round * rounds_per_epoch
    budget = harness.OptimizerUpdateBudget.create(
        reporters,
        total_optimizer_updates=steps_per_epoch * epochs,
        expected_example_exposures={
            slug: observations_per_head * rounds_per_epoch * epochs
            for slug in reporters
        },
        expected_update_exposures={
            slug: rounds_per_epoch * epochs for slug in reporters
        },
    )
    return budget, steps_per_epoch


def tensor_state_dict(model: Any, *, device: str = "cpu") -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {}
    for name, value in model.state_dict().items():
        clone = value.detach().clone()
        if device == "cpu":
            clone = clone.cpu()
        elif device != str(clone.device):
            clone = clone.to(device)
        result[name] = clone
    return result


def average_state_dicts(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch

    if len(states) < 2:
        raise ValueError("Checkpoint average requires at least two states")
    keys = tuple(states[0])
    if any(tuple(state) != keys for state in states[1:]):
        raise RuntimeError("Checkpoint state keys differ")
    result: dict[str, Any] = {}
    for key in keys:
        values = [state[key] for state in states]
        if values[0].is_floating_point():
            accumulator = values[0].to(torch.float64)
            for value in values[1:]:
                accumulator = accumulator + value.to(torch.float64)
            result[key] = (accumulator / len(values)).to(values[0].dtype)
        else:
            if any(not torch.equal(values[0], value) for value in values[1:]):
                # Integer counters are not learned parameters; retain the latest.
                result[key] = values[-1].clone()
            else:
                result[key] = values[0].clone()
    return result


def update_ema(ema_state: Mapping[str, Any], model: Any, decay: float) -> None:
    import torch

    with torch.no_grad():
        current = model.state_dict()
        for name, target in ema_state.items():
            value = current[name].detach()
            if target.is_floating_point():
                target.mul_(decay).add_(value.to(target.device), alpha=1.0 - decay)
            else:
                target.copy_(value.to(target.device))


def make_grad_scaler(enabled: bool) -> Any:
    import torch

    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


@contextlib.contextmanager
def autocast_cuda(enabled: bool) -> Iterable[None]:
    import torch

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=enabled
    ):
        yield


def load_batch(
    phase_cache: PhaseCache,
    x_state: frozen_engine.GlobalXPreprocessing,
    head: frozen_engine.HeadPartition,
    local_indices: np.ndarray,
    gpu_staging: frozen_engine.GPUTrainingStaging,
    device: str,
) -> tuple[Any, Any, Any]:
    import torch

    assert head.y_preprocessing is not None
    rows = np.asarray(head.data.phase_rows[local_indices], dtype=np.int64)
    if gpu_staging.phase is not None:
        phase_index = torch.as_tensor(rows, dtype=torch.long, device=device)
        x = gpu_staging.phase.index_select(0, phase_index)
        local_index = torch.as_tensor(
            local_indices, dtype=torch.long, device=device
        )
        y = gpu_staging.targets[head.slug].index_select(0, local_index)
        mask = gpu_staging.target_masks[head.slug].index_select(0, local_index)
    else:
        x = torch.from_numpy(
            x_state.transform(np.asarray(phase_cache.x[rows], dtype=np.float32))
        ).to(device, non_blocking=True)
        y_host, mask_host = head.y_preprocessing.transform_masked(
            head.data.y[local_indices], head.data.observed_mask[local_indices]
        )
        y = torch.from_numpy(y_host).to(device, non_blocking=True)
        mask = torch.from_numpy(mask_host).to(device, non_blocking=True)
    return x, y, mask


def predict_validation_head(
    model: Any,
    phase_cache: PhaseCache,
    x_state: frozen_engine.GlobalXPreprocessing,
    head: frozen_engine.HeadPartition,
    gpu_staging: frozen_engine.GPUTrainingStaging,
    device: str,
    batch_size: int,
    amp: bool,
) -> float:
    import torch

    assert head.y_preprocessing is not None
    total_squared = 0.0
    total_values = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(head.validation_indices), batch_size):
            local = head.validation_indices[start : start + batch_size]
            x, y, mask = load_batch(
                phase_cache, x_state, head, local, gpu_staging, device
            )
            if not bool(mask.all().item()):
                raise RuntimeError("Frozen validation cohort contains missing endpoints")
            with autocast_cuda(amp):
                prediction = model.predict(x, head.slug)
            difference = prediction.float() - y.float()
            total_squared += float(difference.square().sum().cpu())
            total_values += int(difference.numel())
    if total_values <= 0:
        raise RuntimeError(f"Empty validation cohort for {head.slug}")
    result = total_squared / total_values
    if not math.isfinite(result):
        raise RuntimeError(f"Nonfinite validation MSE for {head.slug}")
    return result


def null_validation_mse(
    heads: Sequence[frozen_engine.HeadPartition],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for head in heads:
        assert head.y_preprocessing is not None
        truth = head.y_preprocessing.transform(
            head.data.y[head.validation_indices]
        )
        value = float(np.mean(np.square(truth), dtype=np.float64))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(f"Invalid null validation MSE for {head.slug}")
        result[head.slug] = value
    return result


def checkpoint_payload(
    model_state: Mapping[str, Any],
    model_factory_config: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    optimizer_update: int,
    kind: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "optimizer_update": int(optimizer_update),
        "model_factory_config": copy.deepcopy(model_factory_config),
        "model_manifest": copy.deepcopy(model_manifest),
        "model_state": dict(model_state),
    }


def load_model_from_checkpoint(path: Path, device: str) -> tuple[Any, dict[str, Any]]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"Incompatible checkpoint: {path}")
    model = create_runner_candidate_model(
        candidate_models.CandidateModelConfig.from_mapping(
            payload["model_factory_config"]
        )
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return model, payload


class PredictionAdapter:
    """Expose candidate inference through the frozen evaluator's grouped API."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def eval(self) -> "PredictionAdapter":
        self.model.eval()
        return self

    def forward_grouped(
        self, x: Any, task_names: Sequence[str], group_sizes: Sequence[int]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        offset = 0
        for slug, size in zip(task_names, group_sizes):
            result[slug] = self.model.predict(x[offset : offset + size], slug)
            offset += size
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_grouped(*args, **kwargs)


def training_runtime_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    sampler: Any,
    ema_state: Mapping[str, Any],
    completed_epoch: int,
    global_step: int,
    history: Sequence[Mapping[str, Any]],
    trajectory: harness.ValidationTrajectory,
    ledger: harness.ExposureLedger,
    model_factory_config: Mapping[str, Any],
    model_manifest: Mapping[str, Any],
    identity_sha256: str,
    runtime_seconds: float,
    resource_meter_state: Mapping[str, Any],
    active_signature_ledger: Mapping[str, Any],
) -> None:
    import torch

    payload = {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "completed_epoch": completed_epoch,
        "global_step": global_step,
        "history": list(history),
        "trajectory": trajectory.to_manifest(),
        "ledger": ledger.to_manifest(),
        "model_factory_config": copy.deepcopy(model_factory_config),
        "model_manifest": copy.deepcopy(model_manifest),
        "model_state": tensor_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "sampler_state": sampler.state_dict(),
        "ema_state": {name: value.detach().cpu() for name, value in ema_state.items()},
        "runtime_seconds": float(runtime_seconds),
        "resource_meter_state": copy.deepcopy(resource_meter_state),
        "active_signature_ledger": copy.deepcopy(active_signature_ledger),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
    }
    atomic_torch_save(payload, path)


def eligible_updates(records: Sequence[Mapping[str, Any]], tolerance: float) -> set[int]:
    if not records:
        return set()
    minimum = min(float(row["reporter_normalized_score"]) for row in records)
    threshold = minimum * (1.0 + tolerance)
    return {
        int(row["optimizer_update"])
        for row in records
        if float(row["reporter_normalized_score"]) == minimum
        or float(row["reporter_normalized_score"]) < threshold
    }


def select_specialist_updates_by_reporter(
    records: Sequence[Mapping[str, Any]],
    objective: harness.ReporterNormalizedCheckpointObjective,
) -> dict[str, Any]:
    """Select one validation checkpoint independently for every specialist.

    A collection of disjoint reporter specialists must not inherit the single
    early-stopping time of a shared model.  Selection uses the same relative
    tie tolerance as the common harness and, within the eligible set, keeps the
    earliest update.  The returned macro score is therefore the score of the
    composite specialist state, not the score of any one joint checkpoint.
    """

    if not records:
        raise ValueError("Specialist checkpoint selection requires validation records")
    reporters = tuple(objective.reporters)
    nulls = objective.nulls
    tolerance = float(objective.tie_relative_tolerance)
    selected: dict[str, dict[str, Any]] = {}
    selected_losses: dict[str, float] = {}
    for reporter in reporters:
        rows: list[tuple[Mapping[str, Any], float, float]] = []
        for row in records:
            losses = row.get("validation_mse_by_reporter")
            if not isinstance(losses, Mapping) or reporter not in losses:
                raise RuntimeError(
                    f"Validation trajectory lacks reporter loss for {reporter}"
                )
            loss = float(losses[reporter])
            normalized = loss / max(
                float(nulls[reporter]), float(objective.denominator_floor)
            )
            if not math.isfinite(loss) or loss < 0.0 or not math.isfinite(normalized):
                raise RuntimeError(
                    f"Invalid specialist validation loss for {reporter}"
                )
            rows.append((row, loss, normalized))
        strict_minimum = min(value[2] for value in rows)
        threshold = strict_minimum * (1.0 + tolerance)
        eligible = [
            value
            for value in rows
            if value[2] == strict_minimum or value[2] < threshold
        ]
        chosen_row, chosen_loss, chosen_normalized = min(
            eligible, key=lambda value: int(value[0]["optimizer_update"])
        )
        selected_losses[reporter] = chosen_loss
        selected[reporter] = {
            "optimizer_update": int(chosen_row["optimizer_update"]),
            "checkpoint_sha256": str(chosen_row["checkpoint_sha256"]),
            "validation_mse": chosen_loss,
            "normalized_validation_mse": chosen_normalized,
            "strict_minimum_normalized_mse": strict_minimum,
            "eligibility_threshold": threshold,
        }
    return {
        "schema_version": "ops-specialist-composite-selection-v1",
        "selection_partition": "validation",
        "selection_unit": "reporter_specialist",
        "test_used_for_selection": False,
        "tie_relative_tolerance": tolerance,
        "reporter_selections": selected,
        "reporter_normalized_score": objective.score(selected_losses),
        "raw_reporter_macro_mse": math.fsum(selected_losses.values())
        / len(selected_losses),
        "trajectory_sha256": harness.json_sha256(list(records)),
    }


def validate_specialist_state_prefixes(
    state: Mapping[str, Any], reporter_prefixes: Mapping[str, str]
) -> dict[str, tuple[str, ...]]:
    """Prove that a specialist collection state is an exact prefix partition."""

    if not state:
        raise ValueError("Specialist state dict is empty")
    prefixes = {str(name): str(prefix) for name, prefix in reporter_prefixes.items()}
    if not prefixes or any(not value for value in prefixes.values()):
        raise ValueError("Specialist reporter prefixes must be non-empty")
    if len(set(prefixes.values())) != len(prefixes):
        raise ValueError("Specialist reporter prefixes are not unique")
    owned: dict[str, list[str]] = {name: [] for name in prefixes}
    for key in state:
        matches = [name for name, prefix in prefixes.items() if key.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(
                f"Specialist state key {key!r} belongs to {len(matches)} reporters"
            )
        owned[matches[0]].append(key)
    empty = [name for name, keys in owned.items() if not keys]
    if empty:
        raise RuntimeError(f"Specialist state prefixes own no keys: {empty}")
    return {name: tuple(keys) for name, keys in owned.items()}


def compose_specialist_state_dict(
    *,
    reference_state: Mapping[str, Any],
    source_states_by_update: Mapping[int, Mapping[str, Any]],
    reporter_selected_updates: Mapping[str, int],
    reporter_prefixes: Mapping[str, str],
) -> dict[str, Any]:
    """Compose a strict-loadable full state from reporter-local checkpoints."""

    ownership = validate_specialist_state_prefixes(reference_state, reporter_prefixes)
    if set(reporter_selected_updates) != set(ownership):
        raise RuntimeError("Reporter selection and state-prefix reporter sets differ")
    result: dict[str, Any] = {}
    reference_keys = tuple(reference_state)
    for reporter, keys in ownership.items():
        update = int(reporter_selected_updates[reporter])
        if update not in source_states_by_update:
            raise FileNotFoundError(
                f"Missing source checkpoint state for reporter {reporter} update {update}"
            )
        source = source_states_by_update[update]
        if tuple(source) != reference_keys:
            raise RuntimeError(
                f"Source state key order changed at specialist update {update}"
            )
        for key in keys:
            value = source[key]
            reference = reference_state[key]
            if value.shape != reference.shape or value.dtype != reference.dtype:
                raise RuntimeError(
                    f"Specialist state tensor contract changed for {reporter}/{key}"
                )
            result[key] = value.detach().cpu().clone()
    if set(result) != set(reference_state):
        raise RuntimeError("Composite specialist state does not cover the full model")
    return {key: result[key] for key in reference_keys}


def active_endpoint_query_labels(
    selected_reporters: Sequence[str],
    heads_by_slug: Mapping[str, frozen_engine.HeadPartition],
) -> tuple[str, ...]:
    """Return concrete query labels without materialising inactive reporters."""

    return tuple(
        f"{slug}::{feature_name}"
        for slug in selected_reporters
        for feature_name in heads_by_slug[slug].data.target_feature_names.astype(str)
    )


def measure_selected_checkpoint_inference(
    args: argparse.Namespace,
    *,
    model: Any,
    checkpoint_path: Path,
    phase_cache: PhaseCache,
    x_state: frozen_engine.GlobalXPreprocessing,
    heads: Sequence[frozen_engine.HeadPartition],
    gpu_staging: frozen_engine.GPUTrainingStaging,
) -> list[dict[str, Any]]:
    """Measure model-compute-only throughput for the validation-selected state.

    The fixed validation inputs are already resident on CUDA.  These records are
    reporting-only and are created strictly after checkpoint selection, so they
    cannot influence HPO or checkpoint choice.
    """

    import torch

    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(payload["model_state"], strict=True)
    model.to("cuda").eval()
    checkpoint_sha256 = file_sha256(checkpoint_path)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for head in heads:
            batch_size = min(
                int(args.throughput_batch_size),
                int(len(head.complete_validation_indices)),
            )
            if batch_size <= 0:
                raise RuntimeError(
                    f"No validation observations for throughput measurement: {head.slug}"
                )
            local = head.complete_validation_indices[:batch_size]
            batch_x, _, _ = load_batch(
                phase_cache, x_state, head, local, gpu_staging, "cuda"
            )

            def run_once() -> Any:
                with autocast_cuda(args.amp):
                    return model.predict(batch_x, head.slug)

            row = runtime_metrics.measure_cuda_inference_callable(
                run_once,
                cells_per_iteration=batch_size,
                endpoint_predictions_per_iteration=(
                    batch_size * int(head.data.y.shape[1])
                ),
                warmup_iterations=int(args.throughput_warmup_iterations),
                timed_iterations=int(args.throughput_timed_iterations),
                torch_module=torch,
                device="cuda",
            )
            row.update(
                {
                    "method_id": args.method,
                    "reporter_slug": head.slug,
                    "endpoint_dimension": int(head.data.y.shape[1]),
                    "evaluation_state": "validation_selected_single_checkpoint",
                    "checkpoint_sha256": checkpoint_sha256,
                    "amp_dtype": "bfloat16" if args.amp else "float32",
                }
            )
            rows.append(row)
    return rows


def train_model(
    args: argparse.Namespace,
    method_config: Mapping[str, Any],
    phase_cache: PhaseCache,
    heads: Sequence[frozen_engine.HeadPartition],
    x_state: frozen_engine.GlobalXPreprocessing,
    gpu_staging: frozen_engine.GPUTrainingStaging,
    split_contract: harness.FrozenSplitContract,
) -> tuple[dict[str, Any], dict[str, Path]]:
    import torch

    reporters = tuple(head.slug for head in heads)
    head_dimensions = OrderedDict(
        (head.slug, int(head.data.y.shape[1])) for head in heads
    )
    training = method_config["training"]
    optimizer_config = method_config["optimizer"]
    rounds_per_epoch = frozen_rounds_per_epoch(args, len(reporters))
    budget, steps_per_epoch = build_update_budget(
        reporters,
        int(training["epochs"]),
        int(training["observations_per_head"]),
        int(training["batch_heads"]),
        rounds_per_epoch,
    )
    model_options = copy.deepcopy(method_config["model"])
    endpoint_semantic_binding: dict[str, Any] | None = None
    if args.method in {"q_id_52", "q_semantic_52"}:
        semantic_ids, semantic_vocab_sizes, endpoint_semantic_binding = (
            load_query_semantic_binding(args, heads)
        )
        if args.method == "q_semantic_52":
            model_options["semantic_ids"] = semantic_ids
            model_options["semantic_vocab_sizes"] = semantic_vocab_sizes
    if args.method == "multitab_pilot":
        configured_panel = tuple(model_options.pop("reporter_panel", reporters))
        if configured_panel != reporters:
            raise RuntimeError("MultiTab config differs from the frozen full52 order")
        model_options["reporter_panel"] = list(reporters)
    factory_config = candidate_models.CandidateModelConfig(
        model_type=METHOD_MODEL_TYPES[args.method],
        head_dimensions=head_dimensions,
        input_dim=len(x_state.kept_indices),
        options=model_options,
    )
    model = create_runner_candidate_model(factory_config).to("cuda")
    model_manifest = model.config_manifest()
    parameter_inventory = runtime_metrics.ParameterInventory.from_model(
        model
    ).to_manifest()
    if tuple(model.reporter_names) != reporters:
        raise RuntimeError("Model reporter order differs from frozen data order")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        betas=(0.9, 0.999),
        eps=1.0e-8,
    )
    total_steps = budget.total_optimizer_updates
    warmup_steps = int(training["warmup_epochs"]) * steps_per_epoch
    minimum_ratio = float(training["minimum_learning_rate"]) / float(
        optimizer_config["learning_rate"]
    )

    def lr_multiplier(step_index: int) -> float:
        if warmup_steps and step_index < warmup_steps:
            return max(1.0 / warmup_steps, (step_index + 1) / warmup_steps)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step_index - warmup_steps) / denominator))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
    # BF16 has FP32-like exponent range and does not use dynamic loss scaling.
    scaler = make_grad_scaler(False)
    sampler = sampler_library.DeterministicCyclicTaskSampler(
        {head.slug: len(head.train_indices) for head in heads},
        tasks_per_batch=int(training["batch_heads"]),
        samples_per_task=int(training["observations_per_head"]),
        epoch_samples_per_task=(
            int(training["observations_per_head"]) * rounds_per_epoch
        ),
        seed=frozen_engine.stable_seed(args.seed, args.split, args.fold, args.method),
    )
    if len(sampler) != steps_per_epoch:
        raise RuntimeError(
            f"Sampler steps {len(sampler)} differ from frozen {steps_per_epoch}"
        )
    ledger = harness.ExposureLedger(budget)
    nulls = null_validation_mse(heads)
    requested_objective_reporters = getattr(
        args, "checkpoint_objective_reporters", None
    )
    objective_reporters = (
        reporters
        if requested_objective_reporters is None
        else tuple(str(value) for value in requested_objective_reporters)
    )
    if not objective_reporters or len(set(objective_reporters)) != len(
        objective_reporters
    ):
        raise RuntimeError("Checkpoint-objective reporters must be unique and nonempty")
    unknown_objective_reporters = set(objective_reporters) - set(reporters)
    if unknown_objective_reporters:
        raise RuntimeError(
            "Checkpoint objective contains reporters outside the training registry: "
            f"{sorted(unknown_objective_reporters)}"
        )
    objective = harness.ReporterNormalizedCheckpointObjective.create(
        objective_reporters,
        {reporter: nulls[reporter] for reporter in objective_reporters},
        denominator_floor=1.0e-3,
        tie_relative_tolerance=2.0e-3,
    )
    schedule = harness.ValidationSchedule(
        total_optimizer_updates=total_steps,
        validation_updates=tuple(
            epoch * steps_per_epoch for epoch in range(1, int(training["epochs"]) + 1)
        ),
    )
    trajectory = harness.ValidationTrajectory(objective, schedule)
    ema_state = tensor_state_dict(model, device="cuda")
    by_slug = {head.slug: head for head in heads}
    resource_meter = runtime_metrics.CudaTrainingResourceMeter(torch, device="cuda")
    resource_meter.begin()
    active_signature_ledger = runtime_metrics.CompactActiveSignatureLedger()
    checkpoint_dir = args.output_root / "checkpoints"
    state_dir = args.output_root / "evaluation_state_candidates"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "method": args.method,
        "method_config": method_config,
        "split": args.split,
        "fold": args.fold,
        "trial_index": args.trial_index,
        "seed": args.seed,
        "same_cell_arm": args.same_cell_arm,
        "reporters": list(reporters),
        "label_policy": "complete_core",
        "loss_policy": "observed_endpoints_only_never_zero_impute_missing_targets",
        "outer_test_y_access": "physically_sealed_during_training",
        "capacity_policy": {
            "parameter_matching_required": False,
            "parameter_count_is_admission_gate": False,
            "parameter_count_is_hpo_objective_or_tie_breaker": False,
            "native_capacity_selected_by_validation_within_family": True,
            "resource_metrics_role": "reporting_only",
        },
        "parameter_inventory": copy.deepcopy(parameter_inventory),
        "split_contract_sha256": harness.json_sha256(split_contract.to_manifest()),
        "update_budget_sha256": harness.json_sha256(budget.to_manifest()),
        "rounds_per_epoch": rounds_per_epoch,
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "candidate_model_sha256": file_sha256(Path(candidate_models.__file__).resolve()),
        "harness_library_sha256": file_sha256(Path(harness.__file__).resolve()),
        "campaign_binding": copy.deepcopy(args.campaign_binding),
        "endpoint_semantic_binding": copy.deepcopy(endpoint_semantic_binding),
    }
    identity_sha = json_sha256(identity)
    atomic_json(args.output_root / "training_identity.json", {**identity, "identity_sha256": identity_sha})

    initial_state = tensor_state_dict(model)
    initial_path = checkpoint_dir / "update_00000000.pt"
    if not (args.resume and initial_path.is_file()):
        atomic_torch_save(
            checkpoint_payload(
                initial_state,
                factory_config.to_manifest(),
                model_manifest,
                0,
                "initial",
            ),
            initial_path,
        )
    checkpoint_hashes: dict[int, str] = {0: file_sha256(initial_path)}
    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 1
    previous_runtime = 0.0
    last_path = args.output_root / "last.pt"
    # Resume is epoch-boundary only.  A partial epoch is intentionally replayed.
    if args.resume and last_path.exists():
        try:
            resume = torch.load(last_path, map_location="cpu", weights_only=False)
        except TypeError:
            resume = torch.load(last_path, map_location="cpu")
        if resume.get("identity_sha256") != identity_sha:
            raise RuntimeError("Resume checkpoint identity differs; use a new output root")
        model.load_state_dict(resume["model_state"], strict=True)
        model.to("cuda")
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        sampler.load_state_dict(resume["sampler_state"])
        ema_state = {
            name: value.to("cuda") for name, value in resume["ema_state"].items()
        }
        history = list(resume["history"])
        trajectory = harness.ValidationTrajectory.from_manifest(resume["trajectory"])
        checkpoint_hashes.update(
            {
                int(row["optimizer_update"]): str(row["checkpoint_sha256"])
                for row in trajectory.records
            }
        )
        ledger_payload = resume["ledger"]
        ledger = harness.ExposureLedger(budget)
        # Replay only integer exposure accounting, never data or gradients.
        completed_epochs = int(resume["completed_epoch"])
        for _epoch in range(completed_epochs):
            for _round in range(rounds_per_epoch):
                for offset in range(0, len(reporters), int(training["batch_heads"])):
                    ledger.record_update(
                        {
                            slug: int(training["observations_per_head"])
                            for slug in reporters[
                                offset : offset + int(training["batch_heads"])
                            ]
                        }
                    )
        if ledger.to_manifest() != ledger_payload:
            raise RuntimeError("Resume exposure ledger is not reproducible")
        global_step = int(resume["global_step"])
        start_epoch = completed_epochs + 1
        previous_runtime = float(resume.get("runtime_seconds", 0.0))
        resource_meter.load_state_dict(resume["resource_meter_state"])
        active_signature_ledger.load_state_dict(
            resume["active_signature_ledger"]
        )
        if active_signature_ledger.last_optimizer_update != global_step:
            raise RuntimeError(
                "Resume active-parameter ledger does not end at global_step"
            )
        random.setstate(resume["rng_state"]["python"])
        np.random.set_state(resume["rng_state"]["numpy"])
        torch.set_rng_state(resume["rng_state"]["torch_cpu"])
        torch.cuda.set_rng_state_all(resume["rng_state"]["torch_cuda"])

    started = time.time()
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        epoch_started = time.time()
        model.train()
        train_losses = {slug: [] for slug in reporters}
        resource_meter.start_training_region()
        for task_batch in sampler:
            selected = list(task_batch.task_names)
            x_parts = []
            targets: dict[str, Any] = {}
            masks: dict[str, Any] = {}
            group_sizes: list[int] = []
            exposure: dict[str, int] = {}
            observed_endpoint_predictions = 0
            for slug, positions in zip(selected, task_batch.task_indices):
                head = by_slug[slug]
                local = head.train_indices[np.asarray(positions, dtype=np.int64)]
                x, y, mask = load_batch(
                    phase_cache, x_state, head, local, gpu_staging, "cuda"
                )
                if not bool(mask.any(dim=1).all().item()):
                    raise RuntimeError(f"All-missing training row for {slug}")
                x_parts.append(x)
                targets[slug] = y
                masks[slug] = mask
                group_sizes.append(len(local))
                exposure[slug] = len(local)
                observed_endpoint_predictions += int(
                    head.data.observed_mask[local].sum()
                )
            batch_x = torch.cat(x_parts, dim=0)
            optimizer.zero_grad(set_to_none=True)
            with autocast_cuda(args.amp):
                predictions = model.forward_grouped(batch_x, selected, group_sizes)
                losses = {
                    slug: model.compute_loss(
                        predictions[slug], targets[slug], masks[slug]
                    )
                    for slug in selected
                }
                loss = torch.stack(tuple(losses.values())).mean()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Nonfinite loss at epoch={epoch}, step={global_step}")
            scaler.scale(loss).backward()
            active_row = runtime_metrics.active_parameter_snapshot_after_backward(
                model,
                optimizer_update=global_step + 1,
                active_reporters=selected,
                active_endpoint_queries=(
                    active_endpoint_query_labels(selected, by_slug)
                    if args.method in {"q_id_52", "q_semantic_52"}
                    else None
                ),
                cell_observations=sum(group_sizes),
                observed_endpoint_predictions=observed_endpoint_predictions,
            )
            active_signature_ledger.record(active_row)
            if float(training["gradient_clip_norm"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            update_ema(ema_state, model, float(training["ema_decay"]))
            ledger.record_update(exposure)
            global_step += 1
            for slug, value in losses.items():
                train_losses[slug].append(float(value.detach().float().cpu()))

        epoch_resource = resource_meter.stop_training_region()

        if global_step != epoch * steps_per_epoch:
            raise RuntimeError("Optimizer update count drifted from frozen schedule")
        validation = {
            head.slug: predict_validation_head(
                model,
                phase_cache,
                x_state,
                head,
                gpu_staging,
                "cuda",
                args.prediction_batch_size,
                args.amp,
            )
            for head in heads
        }
        current_state = tensor_state_dict(model)
        update_path = checkpoint_dir / f"update_{global_step:08d}.pt"
        atomic_torch_save(
            checkpoint_payload(
                current_state,
                factory_config.to_manifest(),
                model_manifest,
                global_step,
                "single",
            ),
            update_path,
        )
        checkpoint_hash = file_sha256(update_path)
        checkpoint_hashes[global_step] = checkpoint_hash
        selection_validation = {
            reporter: validation[reporter] for reporter in objective_reporters
        }
        trajectory.append(global_step, selection_validation, checkpoint_hash)
        row = trajectory.records[-1]
        history.append(
            {
                "epoch": epoch,
                "optimizer_update": global_step,
                "steps": steps_per_epoch,
                "train_mse_by_reporter": {
                    slug: float(np.mean(values)) for slug, values in train_losses.items()
                },
                "validation_mse_by_reporter": validation,
                "validation_macro_mse": row["raw_reporter_macro_mse"],
                "validation_reporter_normalized_mse": row[
                    "reporter_normalized_score"
                ],
                "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.time() - epoch_started,
                "training_gpu_seconds": epoch_resource["gpu_seconds"],
                "training_region_wall_seconds": epoch_resource["wall_seconds"],
            }
        )
        records = trajectory.records
        eligible = eligible_updates(records, objective.tie_relative_tolerance)
        if global_step in eligible:
            ema_path = state_dir / f"ema_update_{global_step:08d}.pt"
            atomic_torch_save(
                checkpoint_payload(
                    {name: value.detach().cpu() for name, value in ema_state.items()},
                    factory_config.to_manifest(),
                    model_manifest,
                    global_step,
                    "ema",
                ),
                ema_path,
            )
            previous_updates = sorted(checkpoint_hashes)
            position = previous_updates.index(global_step)
            average_updates = previous_updates[max(0, position - 2) : position + 1]
            if len(average_updates) < 2:
                raise RuntimeError("Checkpoint averaging lacks an initial/source state")
            average_states = []
            for update in average_updates:
                source_path = checkpoint_dir / f"update_{update:08d}.pt"
                try:
                    source = torch.load(source_path, map_location="cpu", weights_only=False)
                except TypeError:
                    source = torch.load(source_path, map_location="cpu")
                average_states.append(source["model_state"])
            average_path = state_dir / f"average_update_{global_step:08d}.pt"
            atomic_torch_save(
                {
                    **checkpoint_payload(
                        average_state_dicts(average_states),
                        factory_config.to_manifest(),
                        model_manifest,
                        global_step,
                        "checkpoint_average",
                    ),
                    "source_checkpoint_updates": average_updates,
                    "source_checkpoint_sha256": [
                        checkpoint_hashes[update] for update in average_updates
                    ],
                },
                average_path,
            )
        protected = eligible | set(sorted(checkpoint_hashes)[-3:]) | {0}
        if args.method == "tabm_specialists":
            # TabM is 52 disjoint specialists.  Keep every validation state
            # until the per-reporter composite has been materialised; the
            # global-macro eligible set is insufficient for local selection.
            protected.update(
                int(record["optimizer_update"]) for record in trajectory.records
            )
        for path in checkpoint_dir.glob("update_*.pt"):
            update = int(path.stem.split("_")[-1])
            if update not in protected:
                path.unlink(missing_ok=True)
        for prefix in ("ema_update_", "average_update_"):
            for path in state_dir.glob(f"{prefix}*.pt"):
                update = int(path.stem.split("_")[-1])
                if update not in eligible:
                    path.unlink(missing_ok=True)
        training_runtime_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            sampler=sampler,
            ema_state=ema_state,
            completed_epoch=epoch,
            global_step=global_step,
            history=history,
            trajectory=trajectory,
            ledger=ledger,
            model_factory_config=factory_config.to_manifest(),
            model_manifest=model_manifest,
            identity_sha256=identity_sha,
            runtime_seconds=previous_runtime + time.time() - started,
            resource_meter_state=resource_meter.state_dict(),
            active_signature_ledger=active_signature_ledger.state_dict(),
        )
        print(
            f"epoch {epoch:03d}/{training['epochs']}: "
            f"validation normalized={row['reporter_normalized_score']:.6f}; "
            f"raw={row['raw_reporter_macro_mse']:.6f}; update={global_step}",
            flush=True,
        )

    ledger.finalize()
    selection = trajectory.select_checkpoint()
    selected_update = selection.optimizer_update
    selected_path = checkpoint_dir / f"update_{selected_update:08d}.pt"
    ema_path = state_dir / f"ema_update_{selected_update:08d}.pt"
    average_path = state_dir / f"average_update_{selected_update:08d}.pt"
    for path in (selected_path, ema_path, average_path):
        if not path.is_file():
            raise RuntimeError(f"Selected read-only evaluation state is missing: {path}")
    try:
        average_payload = torch.load(
            average_path, map_location="cpu", weights_only=False
        )
    except TypeError:
        average_payload = torch.load(average_path, map_location="cpu")
    inventory = {
        int(row["optimizer_update"]): str(row["checkpoint_sha256"])
        for row in trajectory.records
    }
    inventory[0] = checkpoint_hashes[0]
    registry = harness.EvaluationStateRegistry(selection, inventory)
    registry.add_single()
    registry.add_ema(
        "ema_0999", file_sha256(ema_path), float(training["ema_decay"])
    )
    registry.add_checkpoint_average(
        "checkpoint_average_3",
        file_sha256(average_path),
        tuple(int(value) for value in average_payload["source_checkpoint_updates"]),
    )
    specialist_composite_selection: dict[str, Any] | None = None
    primary_single_path = selected_path
    if args.method == "tabm_specialists":
        specialist_composite_selection = select_specialist_updates_by_reporter(
            trajectory.records, objective
        )
        reporter_selected_updates = {
            reporter: int(row["optimizer_update"])
            for reporter, row in specialist_composite_selection[
                "reporter_selections"
            ].items()
        }
        source_states: dict[int, Mapping[str, Any]] = {}
        for update in sorted(set(reporter_selected_updates.values())):
            path = checkpoint_dir / f"update_{update:08d}.pt"
            if not path.is_file():
                raise FileNotFoundError(
                    f"TabM composite source checkpoint is missing: {path}"
                )
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if payload.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(f"TabM composite source schema changed: {path}")
            if file_sha256(path) != checkpoint_hashes[update]:
                raise RuntimeError(f"TabM composite source hash changed: {path}")
            source_states[update] = payload["model_state"]
        reference_state = tensor_state_dict(model)
        reporter_prefixes = {
            reporter: f"specialists.{model.reporter_module_key(reporter)}."
            for reporter in reporters
        }
        composite_state = compose_specialist_state_dict(
            reference_state=reference_state,
            source_states_by_update=source_states,
            reporter_selected_updates=reporter_selected_updates,
            reporter_prefixes=reporter_prefixes,
        )
        primary_single_path = state_dir / "specialist_composite_single.pt"
        composite_payload = checkpoint_payload(
            composite_state,
            factory_config.to_manifest(),
            model_manifest,
            max(reporter_selected_updates.values()),
            "specialist_composite_single",
        )
        composite_payload.update(
            {
                "read_only_evaluation_state": True,
                "primary_outer_evaluation_state": True,
                "reporter_selected_updates": reporter_selected_updates,
                "specialist_composite_selection": copy.deepcopy(
                    specialist_composite_selection
                ),
                "global_checkpoint_selection_is_audit_only": True,
            }
        )
        atomic_torch_save(composite_payload, primary_single_path)
        # Fail closed before any outer label can be opened.
        model.load_state_dict(composite_state, strict=True)
    training_resources = resource_meter.to_manifest()
    inference_throughput = measure_selected_checkpoint_inference(
        args,
        model=model,
        checkpoint_path=primary_single_path,
        phase_cache=phase_cache,
        x_state=x_state,
        heads=heads,
        gpu_staging=gpu_staging,
    )
    active_ledger_manifest = active_signature_ledger.to_manifest()
    runtime_manifest = runtime_metrics.build_runtime_metrics_manifest(
        parameter_inventory=parameter_inventory,
        training_resources=training_resources,
        inference_throughput=inference_throughput,
        active_signature_ledger=active_ledger_manifest,
    )
    parameter_inventory_path = args.output_root / "parameter_inventory.json"
    active_profiles_path = args.output_root / "active_parameter_profiles.json"
    throughput_path = args.output_root / "inference_throughput.csv"
    runtime_metrics_path = args.output_root / "runtime_metrics.json"
    atomic_json(parameter_inventory_path, parameter_inventory)
    atomic_json(
        active_profiles_path,
        active_ledger_manifest,
    )
    pd.DataFrame(inference_throughput).to_csv(throughput_path, index=False)
    atomic_json(runtime_metrics_path, runtime_manifest)
    runtime = previous_runtime + time.time() - started
    complete = {
        "schema_version": SCHEMA_VERSION,
        "method_id": args.method,
        "model_manifest": model_manifest,
        "model_factory_config": factory_config.to_manifest(),
        "split": args.split,
        "fold": args.fold,
        "trial_index": args.trial_index,
        "seed": args.seed,
        "device_resolved": "cuda",
        "n_heads": len(heads),
        "reporters": list(reporters),
        "label_policy": "complete_core",
        "loss_policy": "observed_endpoints_only_never_zero_impute_missing_targets",
        "outer_test_y_loaded_during_training": False,
        "runtime_seconds": runtime,
        "capacity_policy": {
            "parameter_matching_required": False,
            "parameter_count_is_admission_gate": False,
            "parameter_count_is_hpo_objective_or_tie_breaker": False,
            "native_capacity_selected_by_validation_within_family": True,
            "resource_metrics_role": "reporting_only",
        },
        "parameter_inventory": parameter_inventory,
        "training_resources": training_resources,
        "runtime_metrics_path": str(runtime_metrics_path.resolve()),
        "runtime_metrics_sha256": file_sha256(runtime_metrics_path),
        "parameter_inventory_sha256": file_sha256(parameter_inventory_path),
        "active_parameter_profiles_sha256": file_sha256(active_profiles_path),
        "inference_throughput_sha256": file_sha256(throughput_path),
        "epochs_completed": int(training["epochs"]),
        "steps_per_epoch": steps_per_epoch,
        "reporter_rounds_per_epoch": rounds_per_epoch,
        "global_steps_completed": global_step,
        "update_budget": budget.to_manifest(),
        "exposure_ledger": ledger.to_manifest(),
        "checkpoint_objective": objective.to_manifest(),
        "validation_trajectory": trajectory.to_manifest(),
        "checkpoint_selection": selection.to_manifest(),
        "global_checkpoint_selection": selection.to_manifest(),
        "specialist_composite_selection": copy.deepcopy(
            specialist_composite_selection
        ),
        "hpo_reporter_normalized_validation_score": (
            float(specialist_composite_selection["reporter_normalized_score"])
            if specialist_composite_selection is not None
            else float(selection.score)
        ),
        "evaluation_states": registry.to_manifest(),
        "split_contract": split_contract.to_manifest(),
        "identity_sha256": identity_sha,
        "endpoint_semantic_binding": copy.deepcopy(endpoint_semantic_binding),
        "checkpoint_path": str(primary_single_path.resolve()),
        "checkpoint_sha256": file_sha256(primary_single_path),
        "state_paths": {
            "single": str(primary_single_path.resolve()),
            "ema": str(ema_path.resolve()),
            "checkpoint_average": str(average_path.resolve()),
        },
        "global_audit_state_paths": {
            "single": str(selected_path.resolve()),
            "ema": str(ema_path.resolve()),
            "checkpoint_average": str(average_path.resolve()),
        },
        "evaluation_state_roles": {
            "single": (
                "primary_per_reporter_validation_selected_composite"
                if specialist_composite_selection is not None
                else "primary_global_validation_selected"
            ),
            "ema": "global_checkpoint_diagnostic_nonprimary",
            "checkpoint_average": "global_checkpoint_diagnostic_nonprimary",
        },
        "status": "complete",
    }
    # Round-trip every architecture-independent contract before publication.
    harness.FrozenSplitContract.from_manifest(complete["split_contract"])
    rebuilt_budget = harness.OptimizerUpdateBudget.from_manifest(
        complete["update_budget"]
    )
    harness.ExposureLedger.validate_manifest(
        complete["exposure_ledger"], rebuilt_budget
    )
    rebuilt_trajectory = harness.ValidationTrajectory.from_manifest(
        complete["validation_trajectory"]
    )
    harness.EvaluationStateRegistry.validate_manifest(
        complete["evaluation_states"], rebuilt_trajectory.select_checkpoint()
    )
    harness_manifest = {
        "schema_version": harness.HARNESS_SCHEMA_VERSION,
        "method_id": args.method,
        "method_config": copy.deepcopy(method_config),
        "method_config_sha256": json_sha256(method_config),
        "campaign_binding": copy.deepcopy(args.campaign_binding),
        "split_contract": complete["split_contract"],
        "update_budget": complete["update_budget"],
        "exposure_ledger": complete["exposure_ledger"],
        "validation_trajectory": complete["validation_trajectory"],
        "checkpoint_selection": complete["checkpoint_selection"],
        "global_checkpoint_selection": complete["global_checkpoint_selection"],
        "specialist_composite_selection": complete[
            "specialist_composite_selection"
        ],
        "hpo_reporter_normalized_validation_score": complete[
            "hpo_reporter_normalized_validation_score"
        ],
        "evaluation_states": complete["evaluation_states"],
        "evaluation_state_roles": complete["evaluation_state_roles"],
        "label_policy": "complete_core",
        "loss_policy": complete["loss_policy"],
        "capacity_policy": complete["capacity_policy"],
        "runtime_metrics": {
            "path": complete["runtime_metrics_path"],
            "sha256": complete["runtime_metrics_sha256"],
            "role": "reporting_only",
            "selection_inputs": [],
        },
        "outer_test_y_used_for_training_or_selection": False,
        "outer_test_y_access_during_training": "physically_sealed",
        "test_may_select_evaluation_state": False,
        "statistical_unit": complete["split_contract"]["statistical_unit"],
    }
    harness_manifest["manifest_sha256"] = json_sha256(harness_manifest)
    atomic_json(
        args.output_root / "training_harness_manifest.json", harness_manifest
    )
    complete["training_harness_manifest_sha256"] = file_sha256(
        args.output_root / "training_harness_manifest.json"
    )
    atomic_json(args.output_root / "training_complete.json", complete)
    pd.DataFrame(history).drop(
        columns=["train_mse_by_reporter", "validation_mse_by_reporter"]
    ).to_csv(args.output_root / "validation_trajectory.csv", index=False)
    atomic_json(
        args.output_root / "validation_trajectory_full.json",
        {"history": history},
    )
    return complete, {
        "single": primary_single_path,
        "ema": ema_path,
        "checkpoint_average": average_path,
    }


def evaluate_states(
    args: argparse.Namespace,
    training_complete: Mapping[str, Any],
    state_paths: Mapping[str, Path],
    phase_cache: PhaseCache,
    x_state: frozen_engine.GlobalXPreprocessing,
    heads: Sequence[frozen_engine.HeadPartition],
    gpu_staging: frozen_engine.GPUTrainingStaging,
    comparability: Mapping[str, Mapping[str, Any]],
    preprocessing_sha256: str,
) -> None:
    frozen_engine.SCHEMA_VERSION = SCHEMA_VERSION
    frozen_engine.MODEL_NAME = args.method
    args.arm = "training_harness"
    evaluation_rows: list[dict[str, Any]] = []
    for state_id, checkpoint_path in state_paths.items():
        model, _ = load_model_from_checkpoint(checkpoint_path, "cuda")
        adapter = PredictionAdapter(model)
        state_sha = file_sha256(checkpoint_path)
        marker = {
            "runtime_seconds": training_complete["runtime_seconds"],
            "n_heads": training_complete["n_heads"],
            "device_resolved": "cuda",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": state_sha,
            "head_schema_sha256": json_sha256(
                {head.slug: head.feature_names.astype(str).tolist() for head in heads}
            ),
            "job_manifest_sha256": training_complete["identity_sha256"],
            "preprocessing_sha256": preprocessing_sha256,
            "gpu_preflight_sha256": None,
            "run_fingerprint_sha256": training_complete["identity_sha256"],
        }
        for head in heads:
            destination = (
                args.output_root / "evaluation_states" / state_id / "reporters" / head.slug
            )
            complete = frozen_engine.evaluate_head(
                args,
                sampler_library,
                adapter,
                phase_cache,
                x_state,
                head,
                destination,
                args.split,
                args.fold,
                marker,
                comparability[head.slug],
                gpu_staging,
            )
            evaluation_rows.append(
                {
                    "evaluation_state": state_id,
                    "reporter_slug": head.slug,
                    "split": args.split,
                    "fold": args.fold,
                    "checkpoint_sha256": state_sha,
                    **{f"cell_{key}": value for key, value in complete["cell_metrics"].items()},
                    **{f"gene_{key}": value for key, value in complete["gene_metrics"].items()},
                }
            )
        del adapter, model
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    pd.DataFrame(evaluation_rows).to_csv(
        args.output_root / "evaluation_states_summary.csv", index=False
    )


def configure_cuda(seed: int, tf32: bool) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback is forbidden")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The frozen CUDA AMP policy requires BF16 support")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "Set CUBLAS_WORKSPACE_CONFIG=:4096:8 for deterministic CUDA training"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "tf32": tf32,
        "amp_dtype": "bfloat16",
        "cpu_fallback_allowed": False,
    }


def prepare_data(
    args: argparse.Namespace,
) -> tuple[
    PhaseCache,
    pd.DataFrame,
    list[frozen_engine.HeadPartition],
    frozen_engine.GlobalXPreprocessing,
    dict[str, dict[str, Any]],
    harness.FrozenSplitContract,
    str,
]:
    frozen_table = pd.read_csv(args.target_table)
    if len(frozen_table) != 52 or frozen_table["reporter_slug"].nunique() != 52:
        raise RuntimeError("Frozen target table must contain 52 unique reporters")
    reporter_table = choose_reporters(args.method, args.reporters, frozen_table)
    phase_cache = PhaseCache.open(args.phase_cache)
    validate_frozen_assets(phase_cache, frozen_table, args.exact_root)
    technical = frozen_engine.load_technical_core_features(
        args.target_feature_dictionary, frozen_table
    )
    reporter_data = {
        slug: load_sealed_training_data(
            exact_cache_path(args.exact_root, slug),
            slug,
            technical[slug],
            phase_cache,
            args.split,
            args.fold,
        )
        for slug in reporter_table["reporter_slug"].astype(str)
    }
    heads = [
        build_sealed_training_partition(
            args,
            phase_cache,
            row,
            reporter_data[str(row.reporter_slug)],
        )
        for _, row in reporter_table.iterrows()
    ]
    same_cell_audit = apply_same_cell_training_arm(args, phase_cache, heads)
    atomic_json(args.output_root / "same_cell_intervention_audit.json", same_cell_audit)
    reference_path = (
        args.preprocessing_reference_root
        / args.split
        / f"fold_{args.fold}"
        / "train"
        / "preprocessing.json"
    )
    x_state = load_reference_preprocessing(reference_path, heads)
    _, _ = frozen_engine.global_partition_integrity(heads)
    comparability = {
        head.slug: {
            "strict_specialist_comparable": True,
            "reference": str(frozen_test_reference(args, head.slug)[2].resolve()),
            "outer_test_y_physically_loaded": False,
        }
        for head in heads
    }
    split_contract = build_split_contract(args, heads)
    preprocessing_path = args.output_root / "preprocessing.json"
    atomic_json(
        preprocessing_path,
        preprocessing_manifest(reference_path, x_state, heads),
    )
    return (
        phase_cache,
        reporter_table,
        heads,
        x_state,
        comparability,
        split_contract,
        file_sha256(preprocessing_path),
    )


def build_staging_args(args: argparse.Namespace) -> argparse.Namespace:
    staging = copy.copy(args)
    staging.gpu_staging = args.gpu_staging
    staging.gpu_staging_reserve_gib = args.gpu_staging_reserve_gib
    staging.gpu_staging_chunk_rows = args.gpu_staging_chunk_rows
    return staging


def load_test_only_data(
    cache_path: Path,
    slug: str,
    selected_names: Sequence[str],
    reference_phase_rows: np.ndarray,
    reference_controls: np.ndarray,
) -> frozen_engine.MaskedReporterData:
    """Load only the frozen outer-test rows in the post-selection process."""

    with h5py.File(cache_path, "r") as source:
        phase_all = np.asarray(source["phase_row_index"][:], dtype=np.int64)
        order = np.argsort(phase_all, kind="mergesort")
        sorted_phase = phase_all[order]
        positions = np.searchsorted(sorted_phase, reference_phase_rows)
        if np.any(positions >= len(sorted_phase)):
            raise RuntimeError(f"Frozen test phase row is absent for {slug}")
        source_rows_in_reference_order = order[positions]
        if not np.array_equal(
            phase_all[source_rows_in_reference_order], reference_phase_rows
        ):
            raise RuntimeError(f"Frozen test phase-row identity changed for {slug}")
        columns, names = selected_feature_columns(source, selected_names, slug)
        # The low-level reader requires sorted source indices.  Restore frozen
        # reference order after the exact row-only H5 read.
        source_sort = np.argsort(source_rows_in_reference_order, kind="mergesort")
        sorted_source_rows = source_rows_in_reference_order[source_sort]
        sorted_y = read_h5_rows_columns(
            source["fluorescence"], sorted_source_rows, columns
        )
        inverse = np.empty(len(source_sort), dtype=np.int64)
        inverse[source_sort] = np.arange(len(source_sort), dtype=np.int64)
        y = sorted_y[inverse]
        controls_all = np.asarray(source["metadata/is_control"][:], dtype=bool)
        controls = controls_all[source_rows_in_reference_order]
        source_target_features = int(source["fluorescence"].shape[1])
        n_source_rows = int(source["fluorescence"].shape[0])
    if not np.array_equal(controls, reference_controls):
        raise RuntimeError(f"Frozen outer-test control mask changed for {slug}")
    observed = np.isfinite(y)
    if not bool(observed.all()):
        raise RuntimeError(f"Frozen outer-test cohort is not complete-core for {slug}")
    return frozen_engine.MaskedReporterData(
        slug=slug,
        source_h5_rows=np.asarray(source_rows_in_reference_order, dtype=np.int64),
        phase_rows=np.asarray(reference_phase_rows, dtype=np.int64),
        y=y,
        observed_mask=observed,
        is_complete=np.ones(len(y), dtype=bool),
        is_control=np.asarray(controls, dtype=bool),
        target_feature_names=names,
        source_cache=cache_path,
        source_size_bytes=cache_path.stat().st_size,
        source_target_features=source_target_features,
        n_source_rows=n_source_rows,
        n_complete_rows=len(y),
        n_partial_rows=0,
        n_all_missing_rows=0,
        all_missing_source_h5_rows=np.empty(0, dtype=np.int64),
    )


def test_only_partition(
    reporter_row: pd.Series,
    data: frozen_engine.MaskedReporterData,
    validation_fold: int,
) -> frozen_engine.HeadPartition:
    empty = np.empty(0, dtype=np.int64)
    return frozen_engine.HeadPartition(
        slug=data.slug,
        reporter_row=reporter_row,
        data=data,
        complete_train_indices=empty,
        partial_train_indices=empty,
        active_train_indices=empty,
        complete_validation_indices=empty,
        excluded_partial_validation_indices=empty,
        complete_test_indices=np.arange(len(data.y), dtype=np.int64),
        excluded_partial_test_indices=empty,
        validation_fold=validation_fold,
        original_partition_counts={
            "train": 0,
            "validation": 0,
            "test": len(data.y),
            "outer_test_y_physically_loaded_after_selection": True,
        },
    )


def build_phase_only_gpu_staging(
    args: argparse.Namespace,
    phase_cache: PhaseCache,
    x_state: frozen_engine.GlobalXPreprocessing,
) -> frozen_engine.GPUTrainingStaging:
    import torch

    shape = (len(phase_cache.x), len(x_state.kept_indices))
    estimated = int(np.prod(shape, dtype=np.int64)) * 4
    reserve = int(args.gpu_staging_reserve_gib * 1024**3)
    # A multi-slot scheduler can expose one GPU to multiple independent
    # Python processes.  Their ``mem_get_info`` checks must not race: without
    # this small cross-process lock, two children can both see the same free
    # memory and then simultaneously allocate a full 8.4M-cell phase cache.
    # The queue sets these two variables only for multi-slot launches.  The
    # historical single-slot workflow remains byte-for-byte unchanged.
    lock_directory = os.environ.get("OPS_GPU_STAGING_LOCK_DIR", "").strip()
    physical_gpu = os.environ.get("OPS_PHYSICAL_GPU_ID", "").strip()
    if bool(lock_directory) != bool(physical_gpu):
        raise RuntimeError(
            "GPU staging lock configuration is incomplete: both "
            "OPS_GPU_STAGING_LOCK_DIR and OPS_PHYSICAL_GPU_ID are required"
        )

    @contextlib.contextmanager
    def staging_admission_lock() -> Iterable[None]:
        if not lock_directory:
            yield
            return
        if not physical_gpu.isdecimal():
            raise RuntimeError(f"Invalid OPS_PHYSICAL_GPU_ID: {physical_gpu!r}")
        root = Path(lock_directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"physical_gpu_{physical_gpu}.staging.lock"
        with path.open("a+", encoding="utf-8") as handle:
            deadline = time.monotonic() + 900.0
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
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Timed out waiting for physical-GPU staging lock: "
                            f"{path}"
                        ) from error
                    time.sleep(0.25)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    # Hold the lock across the free-memory check and the actual allocation/
    # copy.  It is released before model training, so it never serializes the
    # independent jobs themselves.
    with staging_admission_lock():
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        if estimated + reserve > free_bytes:
            raise RuntimeError(
                "Outer-test phase staging cannot preserve the frozen VRAM reserve: "
                f"estimated={estimated}, reserve={reserve}, free={free_bytes}"
            )
        phase = torch.empty(shape, dtype=torch.float32, device="cuda")
        for start in range(0, shape[0], args.gpu_staging_chunk_rows):
            stop = min(start + args.gpu_staging_chunk_rows, shape[0])
            values = x_state.transform(
                np.asarray(phase_cache.x[start:stop], dtype=np.float32)
            )
            phase[start:stop].copy_(torch.from_numpy(values))
        torch.cuda.synchronize()
    return frozen_engine.GPUTrainingStaging(
        phase=phase,
        targets={},
        target_masks={},
        metadata={
            "requested": "required",
            "enabled": True,
            "data_path": "gpu_resident_phase_only_post_selection_evaluation",
            "outer_test_targets_staged": False,
            "estimated_bytes": estimated,
            "free_bytes_before": int(free_bytes),
            "total_bytes": int(total_bytes),
            "reserve_bytes": reserve,
        },
    )


def prepare_evaluation_data(
    args: argparse.Namespace,
) -> tuple[
    PhaseCache,
    list[frozen_engine.HeadPartition],
    frozen_engine.GlobalXPreprocessing,
    dict[str, dict[str, Any]],
    frozen_engine.GPUTrainingStaging,
    str,
]:
    frozen_table = pd.read_csv(args.target_table)
    reporter_table = choose_reporters(args.method, args.reporters, frozen_table)
    phase_cache = PhaseCache.open(args.phase_cache)
    validate_frozen_assets(phase_cache, frozen_table, args.exact_root)
    technical = frozen_engine.load_technical_core_features(
        args.target_feature_dictionary, frozen_table
    )
    heads: list[frozen_engine.HeadPartition] = []
    for _, row in reporter_table.iterrows():
        slug = str(row.reporter_slug)
        reference_rows, reference_controls, _ = frozen_test_reference(args, slug)
        data = load_test_only_data(
            exact_cache_path(args.exact_root, slug),
            slug,
            technical[slug],
            reference_rows,
            reference_controls,
        )
        heads.append(
            test_only_partition(
                row,
                data,
                (args.fold + 1) % int(phase_cache.manifest["n_folds"]),
            )
        )
    reference_path = (
        args.preprocessing_reference_root
        / args.split
        / f"fold_{args.fold}"
        / "train"
        / "preprocessing.json"
    )
    # Test-only heads have empty train indices, so deserialize directly rather
    # than asserting the train-fit count used in the training process.
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    x_state = frozen_engine.GlobalXPreprocessing.from_json(payload["x"])
    for head in heads:
        head.y_preprocessing = frozen_engine.HeadYPreprocessing.from_json(
            payload["heads"][head.slug]
        )
    comparability = {
        head.slug: frozen_engine.assert_specialist_comparability(
            args, head, args.split, args.fold
        )
        for head in heads
    }
    staging = build_phase_only_gpu_staging(args, phase_cache, x_state)
    phase_arm_audit = apply_same_cell_phase_arm(args, staging)
    atomic_json(
        args.output_root / "same_cell_evaluation_phase_intervention.json",
        phase_arm_audit,
    )
    preprocessing_path = args.output_root / "preprocessing.json"
    if not preprocessing_path.is_file():
        raise FileNotFoundError(f"Missing training preprocessing manifest: {preprocessing_path}")
    return (
        phase_cache,
        heads,
        x_state,
        comparability,
        staging,
        file_sha256(preprocessing_path),
    )


def evaluation_recipe(
    args: argparse.Namespace,
    method_config: Mapping[str, Any],
    complete: Mapping[str, Any],
    state_paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "post_selection_outer_test_evaluation",
        "method_id": args.method,
        "method_config_sha256": json_sha256(method_config),
        "split": args.split,
        "fold": args.fold,
        "trial_index": args.trial_index,
        "seed": args.seed,
        "checkpoint_selection_sha256": harness.json_sha256(
            complete["checkpoint_selection"]
        ),
        "training_complete_sha256": file_sha256(
            args.output_root / "training_complete.json"
        ),
        "states": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for name, path in state_paths.items()
        },
        "test_may_select_state": False,
        "training_or_checkpoint_mutation_allowed": False,
        "outer_test_y_loaded_by_training_process": False,
    }


def run_evaluation_child(
    args: argparse.Namespace,
    method_config: Mapping[str, Any],
    complete: Mapping[str, Any],
    state_paths: Mapping[str, Path],
) -> None:
    recipe = evaluation_recipe(args, method_config, complete, state_paths)
    recipe_path = args.output_root / "evaluation_recipe.json"
    atomic_json(recipe_path, recipe)
    recipe_sha = file_sha256(recipe_path)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--method",
        args.method,
        "--reporters",
        args.reporters,
        "--split",
        args.split,
        "--fold",
        str(args.fold),
        "--trial-index",
        str(args.trial_index),
        "--seed",
        str(args.seed),
        "--method-config-json",
        canonical_json(method_config),
        "--mode",
        "formal",
        "--device",
        "cuda",
        "--gpu-staging",
        "required",
        "--gpu-staging-reserve-gib",
        str(args.gpu_staging_reserve_gib),
        "--gpu-staging-chunk-rows",
        str(args.gpu_staging_chunk_rows),
        "--prediction-batch-size",
        str(args.prediction_batch_size),
        "--workers",
        str(args.workers),
        "--bootstrap-draws",
        str(args.bootstrap_draws),
        "--phase-cache",
        str(args.phase_cache),
        "--exact-root",
        str(args.exact_root),
        "--target-table",
        str(args.target_table),
        "--target-feature-dictionary",
        str(args.target_feature_dictionary),
        "--specialist-reference-root",
        str(args.specialist_reference_root),
        "--preprocessing-reference-root",
        str(args.preprocessing_reference_root),
        "--campaign-config",
        str(args.campaign_config),
        "--tabm-dependency-wheel",
        str(args.tabm_dependency_wheel),
        "--output-root",
        str(args.output_root),
        "--same-cell-arm",
        str(args.same_cell_arm),
        "--evaluation-only",
        "--evaluation-recipe-sha256",
        recipe_sha,
    ]
    if args.save_predictions:
        command.append("--save-predictions")
    if not args.amp:
        command.append("--no-amp")
    if not args.tf32:
        command.append("--no-tf32")
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


def evaluation_only(
    args: argparse.Namespace, method_config: Mapping[str, Any]
) -> None:
    recipe_path = args.output_root / "evaluation_recipe.json"
    if not recipe_path.is_file() or file_sha256(recipe_path) != args.evaluation_recipe_sha256:
        raise RuntimeError("Post-selection evaluation recipe hash mismatch")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("training_or_checkpoint_mutation_allowed") is not False:
        raise RuntimeError("Evaluation recipe permits training mutation")
    if recipe.get("test_may_select_state") is not False:
        raise RuntimeError("Evaluation recipe permits test-driven state selection")
    if recipe.get("method_config_sha256") != json_sha256(method_config):
        raise RuntimeError("Evaluation method config differs from frozen recipe")
    training_path = args.output_root / "training_complete.json"
    if file_sha256(training_path) != recipe["training_complete_sha256"]:
        raise RuntimeError("Training marker changed after evaluation recipe freeze")
    complete = json.loads(training_path.read_text(encoding="utf-8"))
    state_paths = {
        name: Path(item["path"])
        for name, item in recipe["states"].items()
    }
    for name, path in state_paths.items():
        if file_sha256(path) != recipe["states"][name]["sha256"]:
            raise RuntimeError(f"Frozen evaluation state changed: {name}")
    args.v1_reference_root = args.preprocessing_reference_root
    args.budget_policy = "v1_locked"
    (
        phase_cache,
        heads,
        x_state,
        comparability,
        staging,
        preprocessing_sha,
    ) = prepare_evaluation_data(args)
    evaluate_states(
        args,
        complete,
        state_paths,
        phase_cache,
        x_state,
        heads,
        staging,
        comparability,
        preprocessing_sha,
    )
    atomic_json(
        args.output_root / "evaluation_complete.json",
        {
            "schema_version": SCHEMA_VERSION,
            "evaluation_recipe_sha256": args.evaluation_recipe_sha256,
            "evaluation_states_summary_sha256": file_sha256(
                args.output_root / "evaluation_states_summary.csv"
            ),
            "outer_test_y_loaded_after_checkpoint_selection": True,
            "test_may_select_state": False,
            "completed": True,
        },
    )


def write_training_result(
    args: argparse.Namespace,
    method_config: Mapping[str, Any],
    complete: Mapping[str, Any],
    *,
    test_evaluated: bool,
) -> None:
    result = {
        "schema_version": SCHEMA_VERSION,
        "method_id": args.method,
        "method_config_sha256": json_sha256(method_config),
        "reporters_argument": args.reporters,
        "split": args.split,
        "fold": args.fold,
        "trial_index": args.trial_index,
        "seed": args.seed,
        "same_cell_arm": args.same_cell_arm,
        "mode": args.mode,
        "reporter_normalized_validation_score": complete[
            "hpo_reporter_normalized_validation_score"
        ],
        "selected_optimizer_update": complete["checkpoint_selection"][
            "optimizer_update"
        ],
        "selection_unit": (
            "reporter_specialist_composite"
            if complete["specialist_composite_selection"] is not None
            else "global_checkpoint"
        ),
        "specialist_composite_selection": complete[
            "specialist_composite_selection"
        ],
        "selection_used_test": False,
        "test_evaluated": bool(test_evaluated),
        "completed": True,
        "training_complete_sha256": file_sha256(
            args.output_root / "training_complete.json"
        ),
    }
    if test_evaluated:
        result["evaluation_states_summary_sha256"] = file_sha256(
            args.output_root / "evaluation_states_summary.csv"
        )
    atomic_json(args.output_root / "training_result.json", result)


def synthetic_smoke(args: argparse.Namespace, method_config: Mapping[str, Any]) -> None:
    import torch

    ensure_tabm_dependency(args.method, args.tabm_dependency_wheel)
    dimensions = OrderedDict((f"reporter_{index}", dim) for index, dim in enumerate((20, 24, 30, 60)))
    options = copy.deepcopy(method_config["model"])
    if args.method == "multitab_pilot":
        dimensions = OrderedDict(
            (f"reporter_{index}", 4 + index % 5) for index in range(52)
        )
        options["reporter_panel"] = list(dimensions)
    if args.method == "q_semantic_52":
        options["semantic_vocab_sizes"] = {
            field: 2 for field in candidate_models.SEMANTIC_QUERY_FIELDS
        }
        options["semantic_ids"] = {
            reporter: {
                field: [1] * dimension
                for field in candidate_models.SEMANTIC_QUERY_FIELDS
            }
            for reporter, dimension in dimensions.items()
        }
    factory_config = candidate_models.CandidateModelConfig(
        model_type=METHOD_MODEL_TYPES[args.method],
        head_dimensions=dimensions,
        input_dim=8,
        options=options,
    )
    model = create_runner_candidate_model(factory_config).to("cuda")
    names = list(dimensions)
    group_sizes = [3] * len(names)
    x = torch.randn(sum(group_sizes), 8, device="cuda")
    outputs = model.forward_grouped(x, names, group_sizes)
    losses = []
    for name, size in zip(names, group_sizes):
        target = torch.randn(size, dimensions[name], device="cuda")
        mask = torch.ones_like(target, dtype=torch.bool)
        losses.append(model.compute_loss(outputs[name], target, mask))
    torch.stack(losses).mean().backward()
    marker = {
        "schema_version": SCHEMA_VERSION,
        "method": args.method,
        "device": "cuda",
        "model_manifest": model.config_manifest(),
        "loss_finite": bool(torch.isfinite(torch.stack(losses)).all().item()),
        "status": "pass",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "synthetic_smoke.json", marker)
    print(json.dumps(marker, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=tuple(METHOD_MODEL_TYPES))
    parser.add_argument("--reporters", default="all")
    parser.add_argument("--split", required=True, choices=("field_holdout_sanity", "gene_holdout_main"))
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method-config-json", required=True)
    parser.add_argument("--mode", required=True, choices=("development", "formal"))
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--gpu-staging", default="required", choices=("required", "auto", "off"))
    parser.add_argument("--gpu-staging-reserve-gib", type=float, default=16.0)
    parser.add_argument("--gpu-staging-chunk-rows", type=int, default=131072)
    parser.add_argument("--prediction-batch-size", type=int, default=8192)
    parser.add_argument("--throughput-batch-size", type=int, default=8192)
    parser.add_argument("--throughput-warmup-iterations", type=int, default=3)
    parser.add_argument("--throughput-timed-iterations", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--require-specialist-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-observations-per-head", type=int, default=0)
    parser.add_argument("--max-validation-observations-per-head", type=int, default=0)
    parser.add_argument("--max-test-observations-per-head", type=int, default=0)
    parser.add_argument("--phase-cache", type=Path, default=DEFAULT_PHASE_CACHE)
    parser.add_argument("--exact-root", type=Path, default=DEFAULT_EXACT_ROOT)
    parser.add_argument("--target-table", type=Path, default=DEFAULT_TARGET_TABLE)
    parser.add_argument("--target-feature-dictionary", type=Path, default=DEFAULT_TARGET_FEATURE_DICTIONARY)
    parser.add_argument("--specialist-reference-root", type=Path, default=DEFAULT_SPECIALIST_REFERENCE)
    parser.add_argument("--preprocessing-reference-root", type=Path, default=DEFAULT_PREPROCESSING_REFERENCE)
    parser.add_argument("--campaign-config", type=Path, default=DEFAULT_CAMPAIGN_CONFIG)
    parser.add_argument("--endpoint-semantics", type=Path, default=DEFAULT_ENDPOINT_SEMANTICS)
    parser.add_argument("--semantic-vocabulary", type=Path, default=DEFAULT_SEMANTIC_VOCABULARY)
    parser.add_argument("--semantic-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST)
    parser.add_argument("--tabm-dependency-wheel", type=Path, default=DEFAULT_TABM_DEPENDENCY_WHEEL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--same-cell-arm",
        choices=("benchmark_exact",) + SAME_CELL_ROBUSTNESS_ARMS,
        default="benchmark_exact",
        help="Frozen information intervention for architecture-robustness jobs.",
    )
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--evaluation-recipe-sha256", default="", help=argparse.SUPPRESS)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fold < 0 or args.fold >= 5:
        raise ValueError("fold must be in [0, 4]")
    if args.trial_index < 0:
        raise ValueError("trial-index must be nonnegative")
    if args.mode == "development" and not args.train_only:
        raise ValueError("Development jobs must use --train-only; outer test is sealed")
    if args.mode == "formal" and args.train_only and not args.evaluation_only:
        raise ValueError("Formal jobs must evaluate the frozen outer test")
    if args.gpu_staging != "required":
        raise ValueError("Formal harness requires GPU-resident staging; no CPU data fallback")
    if args.gpu_staging_reserve_gib < 12.0:
        raise ValueError("Reserve at least 12 GiB VRAM for model/optimizer/EMA")
    if args.throughput_batch_size <= 0:
        raise ValueError("throughput-batch-size must be positive")
    if args.throughput_warmup_iterations < 0:
        raise ValueError("throughput-warmup-iterations must be nonnegative")
    if args.throughput_timed_iterations <= 0:
        raise ValueError("throughput-timed-iterations must be positive")
    if any(
        value != 0
        for value in (
            args.max_train_observations_per_head,
            args.max_validation_observations_per_head,
            args.max_test_observations_per_head,
        )
    ) and not args.synthetic_smoke:
        raise ValueError("Formal/development commands forbid data caps")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    method_config = strict_method_config(args.method_config_json)
    args.output_root = args.output_root.resolve()
    args.campaign_binding = bind_frozen_campaign_trial(args, method_config)
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    ensure_tabm_dependency(args.method, args.tabm_dependency_wheel)
    gpu = configure_cuda(args.seed, args.tf32)
    if args.evaluation_only:
        if not args.evaluation_recipe_sha256:
            raise ValueError("--evaluation-only requires a frozen recipe SHA256")
        evaluation_only(args, method_config)
        return
    atomic_json(args.output_root / "gpu_preflight.json", gpu)
    if args.synthetic_smoke:
        synthetic_smoke(args, method_config)
        return
    (
        phase_cache,
        _reporter_table,
        heads,
        x_state,
        comparability,
        split_contract,
        preprocessing_sha,
    ) = prepare_data(args)
    head_schema = {
        head.slug: head.data.target_feature_names.astype(str).tolist()
        for head in heads
    }
    atomic_json(
        args.output_root / "head_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "head_dimensions": {slug: len(names) for slug, names in head_schema.items()},
            "feature_names": head_schema,
            "sha256": json_sha256(head_schema),
        },
    )
    # Build model+EMA before staging in train_model would make the staging
    # memory check exact, but the existing proven staging routine runs here.
    # Its 16 GiB reserve covers model, optimizer, EMA, autocast, and workspace.
    staging_args = build_staging_args(args)
    gpu_staging = frozen_engine.build_gpu_training_staging(
        staging_args, phase_cache, x_state, heads, "cuda"
    )
    phase_arm_audit = apply_same_cell_phase_arm(args, gpu_staging)
    atomic_json(
        args.output_root / "same_cell_training_phase_intervention.json",
        phase_arm_audit,
    )
    complete, state_paths = train_model(
        args,
        method_config,
        phase_cache,
        heads,
        x_state,
        gpu_staging,
        split_contract,
    )
    if args.mode == "formal":
        # Release train/validation staging before a new process opens only the
        # frozen outer-test Y.  The immutable recipe binds checkpoints first.
        del gpu_staging, heads, phase_cache, x_state, comparability
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        run_evaluation_child(args, method_config, complete, state_paths)
        write_training_result(
            args, method_config, complete, test_evaluated=True
        )
    else:
        write_training_result(
            args, method_config, complete, test_evaluated=False
        )


if __name__ == "__main__":
    main()
