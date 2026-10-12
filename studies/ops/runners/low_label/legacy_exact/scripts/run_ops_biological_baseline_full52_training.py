#!/usr/bin/env python3
"""Run one fixed-configuration full52 biological baseline formal job.

This runner leaves the frozen panel12 development implementation untouched.  It
reuses its method-native CAPTAIN, scButterfly and MIDAS training loops, but binds
them to all 52 frozen reporters.  Each job trains with the requested outer fold
physically excluded, freezes the validation-selected checkpoint, and only then
spawns an evaluation-only child that opens the frozen outer-test labels and
calls the common OPS reporter evaluator.

There is deliberately no HPO mode: every method uses the single fixed default
configuration declared by ``ops_biological_baseline_full52_formal_v1.json``.
CUDA is mandatory and there is no CPU fallback.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

import ops_biological_baseline_harness_models as biological_models
import ops_reporter_masked_multitask_v2_lib as sampler_library
import ops_reporter_training_harness_lib as harness
import run_ops_biological_baseline_training as panel12_engine
import run_ops_reporter_candidate_training as common_runner
import run_ops_reporter_masked_multitask_resmlp_v2 as frozen_engine
from ops_reporter_specialist_lib import PhaseCache, atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ops-biological-baseline-full52-formal-training-v1"
CHECKPOINT_SCHEMA_VERSION = "ops-biological-baseline-full52-checkpoint-v1"
CAMPAIGN_SCHEMA_VERSION = "ops-biological-baseline-full52-formal-campaign-v1"
DEFAULT_CAMPAIGN_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "ops_biological_baseline_full52_formal_v1.json"
)
METHODS = ("ops_captain", "scbutterfly_ops_b", "midas_ops")
SPLITS = ("gene_holdout_main", "field_holdout_sanity")
FOLDS = (0, 1, 2, 3, 4)
FULL52 = tuple(biological_models.full_reporter_registry())


class Full52RunnerError(RuntimeError):
    """A fixed full52 formal-run invariant was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Full52RunnerError(message)


def _base_config_path(campaign: Mapping[str, Any]) -> Path:
    raw = Path(str(campaign["base_method_config"]["path"]))
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def load_campaign(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    path = path.expanduser().resolve()
    campaign = common_runner.read_json_without_duplicate_keys(path)
    _require(
        campaign.get("schema_version") == CAMPAIGN_SCHEMA_VERSION,
        "Unexpected full52 campaign schema",
    )
    _require(
        campaign.get("campaign_id")
        == "ops-biological-baseline-full52-formal-v1",
        "Unexpected full52 campaign id",
    )
    _require(campaign.get("frozen") is True, "Full52 campaign is not frozen")
    _require(
        campaign.get("reporter_scope")
        == "all52_in_frozen_reporter_targets_order",
        "Full52 reporter scope changed",
    )
    _require(
        campaign.get("endpoint_scope") == "all_1604_frozen_exact_endpoints",
        "Full52 endpoint scope changed",
    )
    _require(tuple(campaign.get("splits", ())) == SPLITS, "Formal splits changed")
    _require(tuple(campaign.get("folds", ())) == FOLDS, "Formal folds changed")
    _require(int(campaign.get("seed", -1)) == 20260721, "Formal seed changed")
    _require(tuple(campaign.get("methods", {})) == METHODS, "Method order changed")
    selection = campaign.get("selection_policy", {})
    _require(
        selection
        == {
            "hyperparameter_search": False,
            "validation_selects_architecture_or_hyperparameters": False,
            "outer_test_selects_anything": False,
            "fixed_default_configuration_per_method": True,
        },
        "Fixed-default/no-HPO policy changed",
    )
    for method in METHODS:
        _require(
            int(campaign["methods"][method]["fixed_trial_index"]) == 0,
            f"{method} is not bound to its fixed default trial",
        )
    base_path = _base_config_path(campaign).resolve()
    _require(base_path.is_file(), f"Missing base biological config: {base_path}")
    expected_sha = str(campaign["base_method_config"]["sha256"])
    _require(
        common_runner.file_sha256(base_path) == expected_sha,
        "Base biological method config SHA256 changed",
    )
    base = common_runner.read_json_without_duplicate_keys(base_path)
    panel12_engine.validate_frozen_campaign(base)
    for method in METHODS:
        trial = base["methods"][method]["trial_configs"][0]
        _require(
            trial["trial_role"]
            == campaign["methods"][method]["fixed_config_role"],
            f"{method} fixed default role changed",
        )
    return campaign, base, base_path


def bind_fixed_trial(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    campaign, base, base_path = load_campaign(args.campaign_config)
    _require(args.method in METHODS, f"Unknown method {args.method!r}")
    _require(args.reporters.strip().casefold() == "all", "Formal scope must be all52")
    _require(args.split in SPLITS and args.fold in FOLDS, "Formal split/fold changed")
    _require(args.seed == int(campaign["seed"]), "Run seed differs from campaign")
    trial_index = int(campaign["methods"][args.method]["fixed_trial_index"])
    _require(args.trial_index == trial_index, "Trial index differs from fixed default")
    trial = copy.deepcopy(base["methods"][args.method]["trial_configs"][trial_index])
    binding = {
        "schema_version": SCHEMA_VERSION,
        "campaign_path": str(args.campaign_config.resolve()),
        "campaign_sha256": common_runner.file_sha256(args.campaign_config),
        "campaign_semantic_sha256": common_runner.json_sha256(campaign),
        "base_method_config": str(base_path),
        "base_method_config_sha256": common_runner.file_sha256(base_path),
        "method": args.method,
        "fixed_trial_index": trial_index,
        "fixed_trial_sha256": common_runner.json_sha256(trial),
        "reporters": list(FULL52),
        "split": args.split,
        "fold": args.fold,
        "seed": args.seed,
        "mode": "formal",
        "hyperparameter_search": False,
        "outer_test_used_for_training_or_checkpoint_selection": False,
    }
    return base, trial, binding


def configure_reused_engine() -> None:
    """Bind the already-tested method-native loops to the full52 registry."""

    panel12_engine.PANEL12 = FULL52
    panel12_engine.SCHEMA_VERSION = SCHEMA_VERSION
    panel12_engine.CHECKPOINT_SCHEMA_VERSION = CHECKPOINT_SCHEMA_VERSION


def prepare_training_data(
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
    _require(
        tuple(frozen_table["reporter_slug"].astype(str)) == FULL52,
        "Frozen target table order differs from the full52 model registry",
    )
    phase_cache = PhaseCache.open(args.phase_cache)
    common_runner.validate_frozen_assets(phase_cache, frozen_table, args.exact_root)
    technical = frozen_engine.load_technical_core_features(
        args.target_feature_dictionary, frozen_table
    )
    reporter_data = {
        slug: common_runner.load_sealed_training_data(
            common_runner.exact_cache_path(args.exact_root, slug),
            slug,
            technical[slug],
            phase_cache,
            args.split,
            args.fold,
        )
        for slug in FULL52
    }
    indexed = frozen_table.set_index("reporter_slug", drop=False)
    heads = [
        common_runner.build_sealed_training_partition(
            args, phase_cache, indexed.loc[slug], reporter_data[slug]
        )
        for slug in FULL52
    ]
    reference_path = (
        args.preprocessing_reference_root
        / args.split
        / f"fold_{args.fold}"
        / "train"
        / "preprocessing.json"
    )
    x_state = common_runner.load_reference_preprocessing(reference_path, heads)
    frozen_engine.global_partition_integrity(heads)
    comparability = {
        head.slug: {
            "strict_specialist_comparable": True,
            "reference": str(
                common_runner.frozen_test_reference(args, head.slug)[2].resolve()
            ),
            "outer_test_y_physically_loaded": False,
        }
        for head in heads
    }
    split_contract = common_runner.build_split_contract(args, heads)
    preprocessing_path = args.output_root / "preprocessing.json"
    atomic_json(
        preprocessing_path,
        common_runner.preprocessing_manifest(reference_path, x_state, heads),
    )
    return (
        phase_cache,
        frozen_table,
        heads,
        x_state,
        comparability,
        split_contract,
        common_runner.file_sha256(preprocessing_path),
    )


class BiologicalPredictionAdapter:
    """Expose biological collections through the frozen evaluator API."""

    def __init__(self, model: biological_models.HarnessModelCollection) -> None:
        self.model = model

    def eval(self) -> "BiologicalPredictionAdapter":
        self.model.eval()
        return self

    def forward_grouped(
        self,
        x: torch.Tensor,
        task_names: Sequence[str],
        group_sizes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        offset = 0
        for slug, size in zip(task_names, group_sizes):
            result[slug] = self.model.predict_reporter(
                x[offset : offset + size], slug
            )
            offset += size
        _require(offset == len(x), "Evaluator grouped batch was not consumed")
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward_grouped(*args, **kwargs)


def load_biological_checkpoint(
    path: Path, device: str
) -> tuple[biological_models.HarnessModelCollection, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(
        payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        f"Incompatible full52 biological checkpoint: {path}",
    )
    _require(payload.get("method") in METHODS, "Checkpoint method is invalid")
    model = biological_models.load_harness_model(
        payload["model_manifest"], payload["model_state"], strict=True
    )
    _require(tuple(model.reporter_names) == FULL52, "Checkpoint is not full52")
    model.to(device).eval()
    return model, payload


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
    _require(
        tuple(frozen_table["reporter_slug"].astype(str)) == FULL52,
        "Evaluation target table order differs from full52",
    )
    phase_cache = PhaseCache.open(args.phase_cache)
    common_runner.validate_frozen_assets(phase_cache, frozen_table, args.exact_root)
    technical = frozen_engine.load_technical_core_features(
        args.target_feature_dictionary, frozen_table
    )
    indexed = frozen_table.set_index("reporter_slug", drop=False)
    heads: list[frozen_engine.HeadPartition] = []
    for slug in FULL52:
        reference_rows, reference_controls, _ = common_runner.frozen_test_reference(
            args, slug
        )
        data = common_runner.load_test_only_data(
            common_runner.exact_cache_path(args.exact_root, slug),
            slug,
            technical[slug],
            reference_rows,
            reference_controls,
        )
        heads.append(
            common_runner.test_only_partition(
                indexed.loc[slug],
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
    preprocessing = json.loads(reference_path.read_text(encoding="utf-8"))
    x_state = frozen_engine.GlobalXPreprocessing.from_json(preprocessing["x"])
    for head in heads:
        head.y_preprocessing = frozen_engine.HeadYPreprocessing.from_json(
            preprocessing["heads"][head.slug]
        )
    comparability = {
        head.slug: frozen_engine.assert_specialist_comparability(
            args, head, args.split, args.fold
        )
        for head in heads
    }
    staging = common_runner.build_phase_only_gpu_staging(args, phase_cache, x_state)
    training_preprocessing = args.output_root / "preprocessing.json"
    _require(
        training_preprocessing.is_file(),
        f"Missing training preprocessing manifest: {training_preprocessing}",
    )
    return (
        phase_cache,
        heads,
        x_state,
        comparability,
        staging,
        common_runner.file_sha256(training_preprocessing),
    )


def evaluation_recipe(
    args: argparse.Namespace,
    complete: Mapping[str, Any],
    state_paths: Mapping[str, Path],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": "post_checkpoint_full52_outer_test_evaluation",
        "method_id": args.method,
        "fixed_trial_index": args.trial_index,
        "split": args.split,
        "fold": args.fold,
        "seed": args.seed,
        "campaign_binding_sha256": common_runner.json_sha256(
            args.campaign_binding
        ),
        "training_complete_sha256": common_runner.file_sha256(
            args.output_root / "training_complete.json"
        ),
        "checkpoint_selection_sha256": common_runner.json_sha256(
            complete["checkpoint_selection"]
        ),
        "states": {
            name: {
                "path": str(path.resolve()),
                "sha256": common_runner.file_sha256(path),
            }
            for name, path in state_paths.items()
        },
        "test_may_select_state": False,
        "training_or_checkpoint_mutation_allowed": False,
        "outer_test_y_loaded_by_training_process": False,
    }


def run_evaluation_child(
    args: argparse.Namespace,
    complete: Mapping[str, Any],
    state_paths: Mapping[str, Path],
) -> None:
    recipe = evaluation_recipe(args, complete, state_paths)
    recipe_path = args.output_root / "evaluation_recipe.json"
    atomic_json(recipe_path, recipe)
    recipe_sha = common_runner.file_sha256(recipe_path)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--method",
        args.method,
        "--reporters",
        "all",
        "--split",
        args.split,
        "--fold",
        str(args.fold),
        "--trial-index",
        str(args.trial_index),
        "--seed",
        str(args.seed),
        "--campaign-config",
        str(args.campaign_config),
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
        "--output-root",
        str(args.output_root),
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
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def evaluate_frozen_states(
    args: argparse.Namespace,
    training_complete: Mapping[str, Any],
    state_paths: Mapping[str, Path],
    phase_cache: PhaseCache,
    heads: Sequence[frozen_engine.HeadPartition],
    x_state: frozen_engine.GlobalXPreprocessing,
    comparability: Mapping[str, Mapping[str, Any]],
    staging: frozen_engine.GPUTrainingStaging,
    preprocessing_sha256: str,
) -> None:
    frozen_engine.SCHEMA_VERSION = SCHEMA_VERSION
    frozen_engine.MODEL_NAME = args.method
    args.arm = "training_harness"
    args.v1_reference_root = args.preprocessing_reference_root
    args.budget_policy = "v1_locked"
    gpu_preflight = args.output_root / "gpu_preflight.json"
    rows: list[dict[str, Any]] = []
    for state_id in ("single", "ema", "checkpoint_average"):
        checkpoint_path = state_paths[state_id]
        model, _ = load_biological_checkpoint(checkpoint_path, "cuda")
        adapter = BiologicalPredictionAdapter(model)
        checkpoint_sha = common_runner.file_sha256(checkpoint_path)
        marker = {
            "runtime_seconds": float(training_complete["runtime_seconds"]),
            "n_heads": len(heads),
            "device_resolved": "cuda",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "head_schema_sha256": common_runner.json_sha256(
                {
                    head.slug: head.feature_names.astype(str).tolist()
                    for head in heads
                }
            ),
            "job_manifest_sha256": training_complete["identity_sha256"],
            "preprocessing_sha256": preprocessing_sha256,
            "gpu_preflight_sha256": (
                common_runner.file_sha256(gpu_preflight)
                if gpu_preflight.is_file()
                else None
            ),
            "run_fingerprint_sha256": training_complete["identity_sha256"],
        }
        for head in heads:
            destination = (
                args.output_root
                / "evaluation_states"
                / state_id
                / "reporters"
                / head.slug
            )
            result = frozen_engine.evaluate_head(
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
                staging,
            )
            rows.append(
                {
                    "evaluation_state": state_id,
                    "reporter_slug": head.slug,
                    "split": args.split,
                    "fold": args.fold,
                    "checkpoint_sha256": checkpoint_sha,
                    **{
                        f"cell_{key}": value
                        for key, value in result["cell_metrics"].items()
                    },
                    **{
                        f"gene_{key}": value
                        for key, value in result["gene_metrics"].items()
                    },
                }
            )
        del adapter, model
        gc.collect()
        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(
        args.output_root / "evaluation_states_summary.csv", index=False
    )


def evaluation_only(args: argparse.Namespace) -> None:
    recipe_path = args.output_root / "evaluation_recipe.json"
    _require(
        recipe_path.is_file()
        and common_runner.file_sha256(recipe_path)
        == args.evaluation_recipe_sha256,
        "Post-checkpoint evaluation recipe hash mismatch",
    )
    recipe = common_runner.read_json_without_duplicate_keys(recipe_path)
    _require(
        recipe.get("training_or_checkpoint_mutation_allowed") is False,
        "Evaluation recipe permits training mutation",
    )
    _require(
        recipe.get("test_may_select_state") is False,
        "Evaluation recipe permits test-driven state selection",
    )
    training_path = args.output_root / "training_complete.json"
    _require(
        common_runner.file_sha256(training_path)
        == recipe["training_complete_sha256"],
        "Training marker changed after recipe freeze",
    )
    complete = common_runner.read_json_without_duplicate_keys(training_path)
    state_paths = OrderedDict(
        (name, Path(recipe["states"][name]["path"]))
        for name in ("single", "ema", "checkpoint_average")
    )
    for name, path in state_paths.items():
        _require(
            common_runner.file_sha256(path) == recipe["states"][name]["sha256"],
            f"Frozen evaluation state changed: {name}",
        )
    (
        phase_cache,
        heads,
        x_state,
        comparability,
        staging,
        preprocessing_sha,
    ) = prepare_evaluation_data(args)
    evaluate_frozen_states(
        args,
        complete,
        state_paths,
        phase_cache,
        heads,
        x_state,
        comparability,
        staging,
        preprocessing_sha,
    )
    summary = args.output_root / "evaluation_states_summary.csv"
    atomic_json(
        args.output_root / "evaluation_complete.json",
        {
            "schema_version": SCHEMA_VERSION,
            "method_id": args.method,
            "split": args.split,
            "fold": args.fold,
            "n_reporters": len(FULL52),
            "n_evaluation_states": 3,
            "evaluation_recipe_sha256": args.evaluation_recipe_sha256,
            "evaluation_states_summary_sha256": common_runner.file_sha256(summary),
            "outer_test_y_loaded_after_checkpoint_selection": True,
            "test_may_select_state": False,
            "completed": True,
        },
    )


def write_formal_result(
    args: argparse.Namespace, complete: Mapping[str, Any]
) -> None:
    evaluation_path = args.output_root / "evaluation_complete.json"
    _require(evaluation_path.is_file(), "Formal outer-test evaluation did not complete")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed": True,
        "mode": "formal",
        "method_id": args.method,
        "reporters_argument": "all",
        "n_reporters": len(FULL52),
        "n_endpoints": sum(biological_models.full_reporter_registry().values()),
        "split": args.split,
        "fold": args.fold,
        "trial_index": args.trial_index,
        "seed": args.seed,
        "fixed_default_configuration": True,
        "hyperparameter_search": False,
        "checkpoint_selection_used_inner_validation_only": True,
        "outer_test_used_for_training_or_selection": False,
        "test_evaluated": True,
        "training_complete": str(
            (args.output_root / "training_complete.json").resolve()
        ),
        "training_complete_sha256": common_runner.file_sha256(
            args.output_root / "training_complete.json"
        ),
        "evaluation_complete": str(evaluation_path.resolve()),
        "evaluation_complete_sha256": common_runner.file_sha256(evaluation_path),
        "selected_optimizer_update": complete["checkpoint_selection"][
            "optimizer_update"
        ],
        "reporter_normalized_validation_score": complete["checkpoint_selection"][
            "score"
        ],
    }
    atomic_json(args.output_root / "training_result.json", result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--reporters", default="all", choices=("all",))
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--fold", required=True, type=int, choices=FOLDS)
    parser.add_argument("--trial-index", type=int, default=0, choices=(0,))
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--campaign-config", type=Path, default=DEFAULT_CAMPAIGN_CONFIG
    )
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--gpu-staging", default="required", choices=("required",))
    parser.add_argument("--gpu-staging-reserve-gib", type=float, default=16.0)
    parser.add_argument("--gpu-staging-chunk-rows", type=int, default=131072)
    parser.add_argument("--prediction-batch-size", type=int, default=8192)
    parser.add_argument("--throughput-batch-size", type=int, default=8192)
    parser.add_argument("--throughput-warmup-iterations", type=int, default=3)
    parser.add_argument("--throughput-timed-iterations", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--phase-cache", type=Path, default=common_runner.DEFAULT_PHASE_CACHE
    )
    parser.add_argument(
        "--exact-root", type=Path, default=common_runner.DEFAULT_EXACT_ROOT
    )
    parser.add_argument(
        "--target-table", type=Path, default=common_runner.DEFAULT_TARGET_TABLE
    )
    parser.add_argument(
        "--target-feature-dictionary",
        type=Path,
        default=common_runner.DEFAULT_TARGET_FEATURE_DICTIONARY,
    )
    parser.add_argument(
        "--specialist-reference-root",
        type=Path,
        default=common_runner.DEFAULT_SPECIALIST_REFERENCE,
    )
    parser.add_argument(
        "--preprocessing-reference-root",
        type=Path,
        default=common_runner.DEFAULT_PREPROCESSING_REFERENCE,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--evaluation-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--evaluation-recipe-sha256", default="", help=argparse.SUPPRESS
    )
    parser.add_argument("--contract-check-only", action="store_true")
    parser.set_defaults(
        max_train_observations_per_head=0,
        max_validation_observations_per_head=0,
        max_test_observations_per_head=0,
        require_specialist_reference=True,
        budget_policy="v1_locked",
        arm="training_harness",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    _require(args.device == "cuda", "CPU fallback is forbidden")
    _require(args.gpu_staging == "required", "GPU staging must be required")
    _require(
        args.gpu_staging_reserve_gib >= 12.0,
        "At least 12 GiB must remain reserved",
    )
    _require(args.gpu_staging_chunk_rows > 0, "GPU staging chunk must be positive")
    _require(args.prediction_batch_size > 0, "Prediction batch size must be positive")
    _require(args.throughput_batch_size > 0, "Throughput batch size must be positive")
    _require(
        args.throughput_timed_iterations > 0,
        "Throughput timed iterations must be positive",
    )
    _require(not (args.resume and args.overwrite), "--resume and --overwrite conflict")
    if args.evaluation_only:
        _require(
            bool(args.evaluation_recipe_sha256),
            "Evaluation-only mode requires a recipe SHA256",
        )


def main() -> None:
    args = build_parser().parse_args()
    args.campaign_config = args.campaign_config.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    for field in (
        "phase_cache",
        "exact_root",
        "target_table",
        "target_feature_dictionary",
        "specialist_reference_root",
        "preprocessing_reference_root",
    ):
        setattr(args, field, Path(getattr(args, field)).expanduser().resolve())
    validate_args(args)
    # Honour the campaign's per-job worker allocation in PyTorch and data code;
    # the launcher also exports the BLAS/OpenMP limits before Python starts.
    torch.set_num_threads(args.workers)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A resumed in-process test may already have initialised the inter-op
        # pool.  The per-job subprocess launcher still gives production runs a
        # fresh interpreter and therefore takes the setting above.
        pass
    configure_reused_engine()
    base, trial, binding = bind_fixed_trial(args)
    args.campaign_binding = binding
    panel12_engine.validate_trial_execution_policy(args, trial)
    if args.contract_check_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "method": args.method,
                    "n_reporters": len(FULL52),
                    "n_endpoints": sum(
                        biological_models.full_reporter_registry().values()
                    ),
                    "split": args.split,
                    "fold": args.fold,
                    "fixed_trial_index": args.trial_index,
                    "hyperparameter_search": False,
                },
                indent=2,
            )
        )
        return
    gpu = common_runner.configure_cuda(args.seed, args.tf32)
    if args.evaluation_only:
        evaluation_only(args)
        return
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "campaign_binding.json", binding)
    atomic_json(args.output_root / "gpu_preflight.json", gpu)
    (
        phase_cache,
        _reporter_table,
        heads,
        x_state,
        _comparability,
        split_contract,
        preprocessing_sha,
    ) = prepare_training_data(args)
    _require(tuple(head.slug for head in heads) == FULL52, "Prepared panel changed")
    panel12_engine.assert_development_heads_sealed(heads)
    identity_comparability = panel12_engine.verify_identity_only_outer_test_cohorts(
        args, phase_cache, heads
    )
    head_schema = OrderedDict(
        (
            head.slug,
            head.data.target_feature_names.astype(str).tolist(),
        )
        for head in heads
    )
    atomic_json(
        args.output_root / "head_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "reporters": list(FULL52),
            "head_dimensions": {
                slug: len(names) for slug, names in head_schema.items()
            },
            "feature_names": head_schema,
            "sha256": common_runner.json_sha256(head_schema),
        },
    )
    atomic_json(
        args.output_root / "sealed_training_data.json",
        {
            "schema_version": SCHEMA_VERSION,
            "n_reporters": len(FULL52),
            "n_endpoints": sum(
                biological_models.full_reporter_registry().values()
            ),
            "outer_test_label_access_during_training": "physically_sealed",
            "prepared_test_label_rows": 0,
            "preprocessing_sha256": preprocessing_sha,
            "comparability": identity_comparability,
            "split_contract_sha256": harness.json_sha256(
                split_contract.to_manifest()
            ),
        },
    )
    staging = frozen_engine.build_gpu_training_staging(
        common_runner.build_staging_args(args),
        phase_cache,
        x_state,
        heads,
        "cuda",
    )
    _require(staging.metadata.get("enabled") is True, "GPU staging was not enabled")
    if args.method in panel12_engine.SHARED_LOOP_METHODS:
        complete = panel12_engine.train_shared_method(
            args,
            campaign_payload=base,
            trial=trial,
            binding=binding,
            phase_cache=phase_cache,
            heads=heads,
            x_state=x_state,
            gpu_staging=staging,
            split_contract=split_contract,
        )
    else:
        complete = panel12_engine.train_scbutterfly_specialists(
            args,
            campaign_payload=base,
            trial=trial,
            binding=binding,
            phase_cache=phase_cache,
            heads=heads,
            x_state=x_state,
            gpu_staging=staging,
            split_contract=split_contract,
        )
    state_paths = OrderedDict(
        (name, Path(complete["state_paths"][name]))
        for name in ("single", "ema", "checkpoint_average")
    )
    del staging, heads, x_state, phase_cache
    gc.collect()
    torch.cuda.empty_cache()
    run_evaluation_child(args, complete, state_paths)
    write_formal_result(args, complete)
    print(
        json.dumps(
            {
                "status": "complete",
                "method": args.method,
                "split": args.split,
                "fold": args.fold,
                "reporters": len(FULL52),
                "outer_test_evaluated": True,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
