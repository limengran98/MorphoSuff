#!/usr/bin/env python3
"""Train and formally evaluate one full-52 OPS sciPENN split/fold job.

The frozen panel12 development entrypoint is deliberately left untouched.
This runner expands sciPENN's native dense endpoint union to the frozen 52
reporters (1,604 endpoints), trains one fixed predeclared recipe with outer-test
labels physically sealed, freezes the validation-selected checkpoint, and then
spawns a post-selection child which alone loads and evaluates the outer test.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

import ops_biological_baseline_scipenn_harness as scipenn_harness
import ops_reporter_masked_multitask_v2_lib as sampler_library
import ops_reporter_training_harness_lib as harness
import run_ops_biological_baseline_training as biological_runner
import run_ops_reporter_candidate_training as common_runner
import run_ops_reporter_masked_multitask_resmlp_v2 as frozen_engine
import run_ops_scipenn_development as development_binding
from ops_reporter_specialist_lib import atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ops_scipenn_full52_formal_v1.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "results" / "ops_scipenn_full52_formal_v1"
SCHEMA_VERSION = "ops-scipenn-full52-formal-training-v1"
EVALUATION_SCHEMA_VERSION = "ops-scipenn-full52-formal-evaluation-v1"
METHOD_ID = "scipenn_ops"
FULL52 = tuple(scipenn_harness.full_reporter_registry())
FULL52_ENDPOINTS = sum(scipenn_harness.full_reporter_registry().values())
SPLITS = ("gene_holdout_main", "field_holdout_sanity")
FOLDS = tuple(range(5))


class FormalRunnerError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormalRunnerError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_config(path: Path) -> dict[str, Any]:
    payload = common_runner.read_json_without_duplicate_keys(path)
    _require(
        payload.get("schema_version") == "ops-scipenn-full52-formal-campaign-v1",
        "Unexpected full52 sciPENN campaign schema",
    )
    _require(payload.get("campaign_id") == "ops-scipenn-full52-formal-v1", "Wrong campaign ID")
    _require(payload.get("frozen") is True, "Full52 sciPENN campaign is not frozen")
    _require(payload.get("method") == METHOD_ID, "Full52 campaign method changed")
    _require(payload.get("reporters") == "all52", "Full52 campaign reporter scope changed")
    _require(tuple(payload.get("splits", ())) == SPLITS, "Formal split set changed")
    _require(tuple(payload.get("folds", ())) == FOLDS, "Formal fold set changed")
    _require(payload.get("selection_policy") == {
        "hyperparameter_search_performed": False,
        "outer_test_used_for_model_or_checkpoint_selection": False,
        "recipe": "fixed_ops_default_declared_before_outer_test_access",
    }, "Fixed no-HPO selection policy changed")
    trial = payload.get("fixed_trial")
    _require(isinstance(trial, Mapping), "Fixed sciPENN trial is missing")
    _require(trial.get("trial_role") == "fixed_released_code_source_anchor_no_hpo", "Fixed trial role changed")
    _require(trial.get("model") == {
        "hidden_dim": 512,
        "dropout": 0.25,
        "quantile_levels": [0.1, 0.25, 0.75, 0.9],
        "loss_reduction": "source_dense_mean",
    }, "Fixed sciPENN model recipe changed")
    _require(trial.get("optimizer") == {
        "name": "adam",
        "learning_rate": 0.001,
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
    }, "Fixed sciPENN optimizer recipe changed")
    training = trial.get("training")
    _require(isinstance(training, Mapping), "Fixed sciPENN training recipe is missing")
    _require(training.get("joint_epochs") == 100, "Formal epoch budget changed")
    _require(training.get("batch_heads") == 4, "Formal batch-head count changed")
    _require(training.get("observations_per_head") == 1024, "Formal reporter batch size changed")
    _require(training.get("scheduler") == {"name": "fixed"}, "Formal scheduler changed")
    _require(training.get("precision_policy") == "float32_no_tf32", "Formal precision changed")
    boundary = payload.get("execution_boundary")
    _require(boundary == {
        "cuda_required": True,
        "cpu_fallback_allowed": False,
        "training_outer_test_y_physically_sealed": True,
        "checkpoint_selected_on_inner_validation_only": True,
        "outer_test_loaded_only_by_post_selection_child": True,
        "evaluation_states": ["single", "ema", "checkpoint_average"],
    }, "Formal execution boundary changed")
    return payload


def _configure_full52_binding() -> Any:
    """Reuse the audited sciPENN loop with process-local full52 constants."""

    runner = development_binding.configure_common_runner()
    # The existing wrapper intentionally binds panel12.  Override only in this
    # process; the frozen source files and panel12 command contract are unchanged.
    runner.PANEL12 = FULL52
    runner.DEVELOPMENT_FOLDS = tuple((split, fold) for split in SPLITS for fold in FOLDS)
    development_binding.campaign_contract.PANEL12 = FULL52
    development_binding.campaign_contract.DEVELOPMENT_FOLDS = runner.DEVELOPMENT_FOLDS
    common_runner.FULL52_METHODS = set(common_runner.FULL52_METHODS) | {METHOD_ID}
    _require(len(FULL52) == 52 and FULL52_ENDPOINTS == 1604, "Frozen full52 registry changed")
    return runner


def _binding(args: argparse.Namespace, payload: Mapping[str, Any]) -> dict[str, Any]:
    trial = payload["fixed_trial"]
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign_path": str(args.campaign_config.resolve()),
        "campaign_file_sha256": _file_sha256(args.campaign_config),
        "campaign_semantic_sha256": common_runner.json_sha256(payload),
        "campaign_id": payload["campaign_id"],
        "method": METHOD_ID,
        "fixed_recipe_no_hpo": True,
        "trial_index": 0,
        "trial_sha256": common_runner.json_sha256(trial),
        "split": args.split,
        "fold": args.fold,
        "reporters": list(FULL52),
        "n_reporters": len(FULL52),
        "n_endpoints": FULL52_ENDPOINTS,
        "seed": args.seed,
        "mode": "formal",
        "training_outer_test_y_access": "physically_sealed",
        "outer_test_evaluation_process": "post_selection_child_only",
        "outer_test_may_select_checkpoint_or_state": False,
        "wrapper_path": str(Path(__file__).resolve()),
        "wrapper_sha256": _file_sha256(Path(__file__).resolve()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=SPLITS)
    parser.add_argument("--fold", required=True, type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--campaign-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument("--gpu-staging", default="required", choices=("required",))
    parser.add_argument("--gpu-staging-reserve-gib", type=float, default=16.0)
    parser.add_argument("--gpu-staging-chunk-rows", type=int, default=131072)
    parser.add_argument("--prediction-batch-size", type=int, default=8192)
    parser.add_argument("--throughput-batch-size", type=int, default=8192)
    parser.add_argument("--throughput-warmup-iterations", type=int, default=3)
    parser.add_argument("--throughput-timed-iterations", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--evaluation-recipe-sha256")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--phase-cache",
        type=Path,
        default=(PROJECT_ROOT / common_runner.DEFAULT_PHASE_CACHE).resolve(),
    )
    parser.add_argument(
        "--exact-root",
        type=Path,
        default=(PROJECT_ROOT / common_runner.DEFAULT_EXACT_ROOT).resolve(),
    )
    parser.add_argument(
        "--target-table",
        type=Path,
        default=(PROJECT_ROOT / common_runner.DEFAULT_TARGET_TABLE).resolve(),
    )
    parser.add_argument(
        "--target-feature-dictionary",
        type=Path,
        default=(PROJECT_ROOT / common_runner.DEFAULT_TARGET_FEATURE_DICTIONARY).resolve(),
    )
    parser.add_argument(
        "--specialist-reference-root",
        type=Path,
        default=(PROJECT_ROOT / common_runner.DEFAULT_SPECIALIST_REFERENCE).resolve(),
    )
    parser.add_argument(
        "--preprocessing-reference-root",
        type=Path,
        default=(PROJECT_ROOT / common_runner.DEFAULT_PREPROCESSING_REFERENCE).resolve(),
    )
    # These are fixed evaluator-contract fields rather than user-facing sciPENN
    # options.  The common evaluator reads them when it verifies that the outer
    # test cohort is identical to the specialist benchmark.  Keep the formal
    # run uncapped and fail closed if the reference bundle is absent.
    parser.set_defaults(
        max_train_observations_per_head=0,
        max_validation_observations_per_head=0,
        max_test_observations_per_head=0,
        require_specialist_reference=True,
    )
    return parser


def _normalize_args(args: argparse.Namespace, payload: Mapping[str, Any]) -> None:
    _require(args.device == "cuda", "CPU fallback is forbidden")
    _require(args.gpu_staging == "required", "GPU staging must be required")
    _require(args.gpu_staging_reserve_gib >= 12.0, "At least 12 GiB VRAM reserve is required")
    _require(not args.amp and not args.tf32, "Released sciPENN source anchor requires FP32 without TF32")
    _require(args.seed == int(payload["seed"]), "Seed differs from frozen campaign")
    _require(not (args.resume and args.overwrite), "--resume and --overwrite conflict")
    _require(
        args.max_train_observations_per_head == 0
        and args.max_validation_observations_per_head == 0
        and args.max_test_observations_per_head == 0,
        "Formal sciPENN evaluation may not cap frozen observations",
    )
    _require(
        args.require_specialist_reference is True,
        "Formal sciPENN evaluation requires the frozen specialist cohort",
    )
    args.method = METHOD_ID
    args.reporters = "all"
    args.trial_index = 0
    args.mode = "formal"
    args.train_only = True
    args.arm = "training_harness"
    args.workers = 1
    args.v1_reference_root = args.preprocessing_reference_root
    args.budget_policy = "v1_locked"
    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOT / METHOD_ID / "formal" / args.split / f"fold_{args.fold}"
    args.output_root = args.output_root.expanduser().resolve()
    for name in (
        "campaign_config",
        "phase_cache",
        "exact_root",
        "target_table",
        "target_feature_dictionary",
        "specialist_reference_root",
        "preprocessing_reference_root",
    ):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())


class SciPENNPredictionAdapter:
    """Expose sciPENN prediction through the frozen grouped evaluator API."""

    def __init__(self, model: scipenn_harness.SciPENNHarnessCollection) -> None:
        self.model = model

    def eval(self) -> "SciPENNPredictionAdapter":
        self.model.eval()
        return self

    def forward_grouped(
        self, x: torch.Tensor, task_names: Sequence[str], group_sizes: Sequence[int]
    ) -> dict[str, torch.Tensor]:
        _require(len(task_names) == len(group_sizes), "Grouped prediction metadata differs")
        result: dict[str, torch.Tensor] = {}
        offset = 0
        for slug, size in zip(task_names, group_sizes):
            size = int(size)
            result[str(slug)] = self.model.predict_reporter(x[offset : offset + size], str(slug))
            offset += size
        _require(offset == len(x), "Grouped prediction sizes do not cover the batch")
        return result

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        return self.forward_grouped(*args, **kwargs)


def _load_scipenn_checkpoint(path: Path) -> tuple[scipenn_harness.SciPENNHarnessCollection, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema_version") == development_binding.CHECKPOINT_SCHEMA_VERSION, "Wrong sciPENN checkpoint schema")
    manifest = payload.get("model_manifest")
    _require(isinstance(manifest, Mapping), "Checkpoint lacks sciPENN model manifest")
    factory = manifest.get("factory_config")
    _require(isinstance(factory, Mapping), "Checkpoint lacks sciPENN factory config")
    _require(tuple(factory.get("active_reporters", ())) == FULL52, "Checkpoint is not full52")
    model = scipenn_harness.create_scipenn_harness_model(
        active_reporters=FULL52,
        options=factory["options"],
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to("cuda").eval()
    return model, payload


def _evaluation_recipe(args: argparse.Namespace, complete: Mapping[str, Any]) -> dict[str, Any]:
    state_paths = {name: Path(path) for name, path in complete["state_paths"].items()}
    _require(set(state_paths) == {"single", "ema", "checkpoint_average"}, "Formal state set changed")
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "role": "post_selection_outer_test_evaluation",
        "method": METHOD_ID,
        "split": args.split,
        "fold": args.fold,
        "seed": args.seed,
        "campaign_config_sha256": _file_sha256(args.campaign_config),
        "training_complete_sha256": _file_sha256(args.output_root / "training_complete.json"),
        "checkpoint_selection_sha256": harness.json_sha256(complete["checkpoint_selection"]),
        "states": {
            name: {"path": str(path.resolve()), "sha256": _file_sha256(path)}
            for name, path in state_paths.items()
        },
        "test_may_select_state": False,
        "training_or_checkpoint_mutation_allowed": False,
        "outer_test_y_loaded_by_training_process": False,
        "max_test_observations_per_head": None,
        "specialist_reference_required": True,
    }


def _load_completed_training_for_evaluation(
    args: argparse.Namespace, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate and reuse a completed formal training phase without retraining.

    The shared biological loop labels its train-only artifact as development,
    even when it is embedded in this formal wrapper.  This function trusts only
    the immutable ``training_complete.json`` plus its formal campaign binding;
    it never reads ``training_result.json``.  Consequently an interrupted
    post-selection evaluation can resume after wrapper-only evaluator fixes
    without either retraining or mistaking the intermediate development marker
    for a completed formal result.
    """

    path = args.output_root / "training_complete.json"
    _require(path.is_file(), "Completed training manifest is missing")
    complete = common_runner.read_json_without_duplicate_keys(path)
    _require(
        complete.get("schema_version") == development_binding.TRAINING_SCHEMA_VERSION,
        "Completed sciPENN training schema changed",
    )
    for key, expected in (
        ("status", "complete"),
        ("mode", "development"),
        ("train_only", True),
        ("method_id", METHOD_ID),
        ("split", args.split),
        ("fold", args.fold),
        ("trial_index", 0),
        ("seed", args.seed),
    ):
        _require(complete.get(key) == expected, f"Completed training violates {key}")
    _require(tuple(complete.get("reporters", ())) == FULL52, "Completed training is not full52")
    outer = complete.get("outer_test")
    _require(isinstance(outer, Mapping), "Completed training lacks the outer-test seal")
    _require(
        outer.get("label_access_during_training") is False
        and outer.get("evaluation_performed") is False
        and outer.get("used_for_training_or_selection") is False,
        "Completed training accessed or used outer-test labels",
    )
    selection = complete.get("checkpoint_selection")
    _require(isinstance(selection, Mapping), "Completed training lacks checkpoint selection")
    _require(
        selection.get("selection_partition") == "validation"
        and selection.get("test_used_for_selection") is False,
        "Completed checkpoint was not selected exclusively on validation",
    )
    binding = complete.get("campaign_binding")
    _require(isinstance(binding, Mapping), "Completed training lacks formal campaign binding")
    expected_binding = _binding(args, payload)
    # The wrapper digest may change for evaluator-only bug fixes.  Every field
    # capable of changing the data, model, split, or selection remains exact.
    for key, expected in expected_binding.items():
        if key == "wrapper_sha256":
            continue
        _require(binding.get(key) == expected, f"Completed formal binding violates {key}")
    state_paths = complete.get("state_paths")
    _require(
        isinstance(state_paths, Mapping)
        and set(state_paths) == {"single", "ema", "checkpoint_average"},
        "Completed training state set changed",
    )
    for name, raw_path in state_paths.items():
        state_path = Path(str(raw_path)).resolve()
        _require(state_path.is_file(), f"Completed training state is missing: {name}")
        _require(
            args.output_root in state_path.parents,
            f"Completed training state escapes the formal job root: {name}",
        )
    return complete


def _archive_intermediate_development_result(args: argparse.Namespace) -> None:
    """Move the shared loop's train-only marker out of the formal result slot."""

    source = args.output_root / "training_result.json"
    if not source.is_file():
        return
    row = common_runner.read_json_without_duplicate_keys(source)
    _require(
        row.get("schema_version") == development_binding.TRAINING_SCHEMA_VERSION
        and row.get("status") == "complete"
        and row.get("mode") == "development"
        and row.get("method_id") == METHOD_ID
        and row.get("split") == args.split
        and row.get("fold") == args.fold
        and row.get("trial_index") == 0
        and row.get("outer_test_evaluated") is False
        and row.get("outer_test_label_accessed") is False,
        "Refusing to archive an unexpected formal result marker",
    )
    training_path = args.output_root / "training_complete.json"
    _require(
        row.get("training_complete_sha256") == _file_sha256(training_path),
        "Intermediate result does not bind the completed training manifest",
    )
    destination = args.output_root / "training_phase_result.json"
    if destination.is_file():
        existing = common_runner.read_json_without_duplicate_keys(destination)
        _require(existing == row, "Archived training-phase result changed")
        source.unlink()
    else:
        os.replace(source, destination)


def _spawn_evaluation_child(args: argparse.Namespace, complete: Mapping[str, Any]) -> None:
    recipe = _evaluation_recipe(args, complete)
    recipe_path = args.output_root / "evaluation_recipe.json"
    atomic_json(recipe_path, recipe)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--split", args.split,
        "--fold", str(args.fold),
        "--seed", str(args.seed),
        "--campaign-config", str(args.campaign_config),
        "--output-root", str(args.output_root),
        "--device", "cuda",
        "--gpu-staging", "required",
        "--gpu-staging-reserve-gib", str(args.gpu_staging_reserve_gib),
        "--gpu-staging-chunk-rows", str(args.gpu_staging_chunk_rows),
        "--prediction-batch-size", str(args.prediction_batch_size),
        "--bootstrap-draws", str(args.bootstrap_draws),
        "--phase-cache", str(args.phase_cache),
        "--exact-root", str(args.exact_root),
        "--target-table", str(args.target_table),
        "--target-feature-dictionary", str(args.target_feature_dictionary),
        "--specialist-reference-root", str(args.specialist_reference_root),
        "--preprocessing-reference-root", str(args.preprocessing_reference_root),
        "--evaluation-only",
        "--evaluation-recipe-sha256", _file_sha256(recipe_path),
    ]
    if args.save_predictions:
        command.append("--save-predictions")
    command.extend(["--no-amp", "--no-tf32"])
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _evaluate_states(
    args: argparse.Namespace,
    complete: Mapping[str, Any],
    state_paths: Mapping[str, Path],
    phase_cache: Any,
    heads: Sequence[Any],
    x_state: Any,
    staging: Any,
    comparability: Mapping[str, Mapping[str, Any]],
    preprocessing_sha256: str,
) -> None:
    frozen_engine.SCHEMA_VERSION = SCHEMA_VERSION
    frozen_engine.MODEL_NAME = METHOD_ID
    rows: list[dict[str, Any]] = []
    for state_id in ("single", "ema", "checkpoint_average"):
        state_started = time.time()
        checkpoint_path = state_paths[state_id]
        model, _ = _load_scipenn_checkpoint(checkpoint_path)
        adapter = SciPENNPredictionAdapter(model)
        checkpoint_sha = _file_sha256(checkpoint_path)
        print(
            f"sciPENN outer-test state={state_id} START "
            f"reporters={len(heads)} checkpoint_sha256={checkpoint_sha}",
            flush=True,
        )
        marker = {
            "runtime_seconds": complete["runtime_seconds"],
            "n_heads": len(heads),
            "device_resolved": "cuda",
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "head_schema_sha256": common_runner.json_sha256(
                {head.slug: head.feature_names.astype(str).tolist() for head in heads}
            ),
            "job_manifest_sha256": complete["identity_sha256"],
            "preprocessing_sha256": preprocessing_sha256,
            "gpu_preflight_sha256": None,
            "run_fingerprint_sha256": complete["identity_sha256"],
        }
        for reporter_index, head in enumerate(heads, start=1):
            destination = args.output_root / "evaluation_states" / state_id / "reporters" / head.slug
            evaluated = frozen_engine.evaluate_head(
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
            rows.append({
                "evaluation_state": state_id,
                "reporter_slug": head.slug,
                "split": args.split,
                "fold": args.fold,
                "checkpoint_sha256": checkpoint_sha,
                **{f"cell_{key}": value for key, value in evaluated["cell_metrics"].items()},
                **{f"gene_{key}": value for key, value in evaluated["gene_metrics"].items()},
            })
            print(
                f"sciPENN outer-test state={state_id} "
                f"reporter={reporter_index:02d}/{len(heads):02d} slug={head.slug} "
                f"cell_standardized_gain="
                f"{float(evaluated['cell_metrics']['standardized_gain_vs_train_mean']):.6f} "
                f"gene_gain={float(evaluated['gene_metrics']['gain_vs_train_mean']):.6f} "
                f"gene_pearson="
                f"{float(evaluated['gene_metrics']['macro_feature_pearson']):.6f}",
                flush=True,
            )
        del adapter, model
        gc.collect()
        torch.cuda.empty_cache()
        print(
            f"sciPENN outer-test state={state_id} COMPLETE "
            f"elapsed_seconds={time.time() - state_started:.1f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(args.output_root / "evaluation_states_summary.csv", index=False)


def _evaluation_only(args: argparse.Namespace) -> None:
    recipe_path = args.output_root / "evaluation_recipe.json"
    _require(args.evaluation_recipe_sha256 is not None, "Evaluation recipe hash is required")
    _require(recipe_path.is_file(), "Evaluation recipe is missing")
    _require(_file_sha256(recipe_path) == args.evaluation_recipe_sha256, "Evaluation recipe hash changed")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    _require(recipe.get("schema_version") == EVALUATION_SCHEMA_VERSION, "Wrong evaluation recipe schema")
    _require(recipe.get("method") == METHOD_ID, "Evaluation recipe method changed")
    _require(recipe.get("split") == args.split and recipe.get("fold") == args.fold, "Evaluation recipe partition changed")
    _require(recipe.get("seed") == args.seed, "Evaluation recipe seed changed")
    _require(recipe.get("campaign_config_sha256") == _file_sha256(args.campaign_config), "Evaluation campaign changed")
    _require(recipe.get("test_may_select_state") is False, "Test may not select state")
    _require(recipe.get("training_or_checkpoint_mutation_allowed") is False, "Evaluation may not mutate training")
    _require(
        recipe.get("max_test_observations_per_head") is None,
        "Formal outer-test evaluation may not be capped",
    )
    _require(
        recipe.get("specialist_reference_required") is True,
        "Formal outer-test evaluation requires specialist cohort verification",
    )
    training_path = args.output_root / "training_complete.json"
    _require(_file_sha256(training_path) == recipe["training_complete_sha256"], "Training marker changed")
    complete = json.loads(training_path.read_text(encoding="utf-8"))
    _require(
        recipe.get("checkpoint_selection_sha256")
        == harness.json_sha256(complete["checkpoint_selection"]),
        "Validation-selected checkpoint changed",
    )
    _require(
        set(recipe.get("states", {})) == {"single", "ema", "checkpoint_average"},
        "Evaluation state set changed",
    )
    state_paths = {name: Path(row["path"]) for name, row in recipe["states"].items()}
    for name, path in state_paths.items():
        _require(path.is_file() and _file_sha256(path) == recipe["states"][name]["sha256"], f"Frozen state changed: {name}")
    args.v1_reference_root = args.preprocessing_reference_root
    args.budget_policy = "v1_locked"
    print(
        f"sciPENN outer-test LOAD split={args.split} fold={args.fold} "
        f"reporters={len(FULL52)} capped_test_rows=none",
        flush=True,
    )
    phase_cache, heads, x_state, comparability, staging, preprocessing_sha = common_runner.prepare_evaluation_data(args)
    _require(tuple(head.slug for head in heads) == FULL52, "Formal evaluation reporter order changed")
    print(
        f"sciPENN outer-test COHORT_VERIFIED split={args.split} fold={args.fold} "
        f"reporters={len(heads)} endpoints={sum(len(head.feature_names) for head in heads)} "
        f"observations={sum(len(head.test_indices) for head in heads)} "
        f"strict_specialist_comparable="
        f"{sum(bool(row['strict_specialist_comparable']) for row in comparability.values())}",
        flush=True,
    )
    _evaluate_states(
        args,
        complete,
        state_paths,
        phase_cache,
        heads,
        x_state,
        staging,
        comparability,
        preprocessing_sha,
    )
    summary_path = args.output_root / "evaluation_states_summary.csv"
    atomic_json(args.output_root / "evaluation_complete.json", {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "complete",
        "split": args.split,
        "fold": args.fold,
        "n_reporters": len(FULL52),
        "n_evaluation_states": 3,
        "outer_test_y_loaded_after_checkpoint_selection": True,
        "test_may_select_state": False,
        "evaluation_recipe_sha256": args.evaluation_recipe_sha256,
        "evaluation_states_summary_sha256": _file_sha256(summary_path),
    })


def _write_formal_result(args: argparse.Namespace) -> None:
    training_path = args.output_root / "training_complete.json"
    evaluation_path = args.output_root / "evaluation_complete.json"
    complete = common_runner.read_json_without_duplicate_keys(training_path)
    evaluation = common_runner.read_json_without_duplicate_keys(evaluation_path)
    for key, expected in (
        ("schema_version", EVALUATION_SCHEMA_VERSION),
        ("status", "complete"),
        ("split", args.split),
        ("fold", args.fold),
        ("n_reporters", len(FULL52)),
        ("n_evaluation_states", 3),
        ("outer_test_y_loaded_after_checkpoint_selection", True),
        ("test_may_select_state", False),
    ):
        _require(evaluation.get(key) == expected, f"Completed evaluation violates {key}")
    summary_path = args.output_root / "evaluation_states_summary.csv"
    _require(
        evaluation.get("evaluation_states_summary_sha256") == _file_sha256(summary_path),
        "Completed evaluation summary changed",
    )
    atomic_json(args.output_root / "training_result.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "mode": "formal",
        "method_id": METHOD_ID,
        "split": args.split,
        "fold": args.fold,
        "fixed_recipe_no_hpo": True,
        "n_reporters": len(FULL52),
        "n_endpoints": FULL52_ENDPOINTS,
        "reporter_normalized_validation_score": complete["checkpoint_selection"]["score"],
        "selected_optimizer_update": complete["checkpoint_selection"]["optimizer_update"],
        "selection_used_test": False,
        "outer_test_evaluated": True,
        "training_complete_sha256": _file_sha256(training_path),
        "evaluation_complete_sha256": _file_sha256(evaluation_path),
        "evaluation_states_summary_sha256": _file_sha256(summary_path),
    })


def _train(args: argparse.Namespace, payload: Mapping[str, Any], runner: Any) -> dict[str, Any]:
    trial = copy.deepcopy(payload["fixed_trial"])
    binding = _binding(args, payload)
    development_binding.validate_trial_execution_policy(args, trial)
    args.campaign_binding = binding
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "campaign_binding.json", binding)
    gpu = common_runner.configure_cuda(args.seed, args.tf32)
    atomic_json(args.output_root / "gpu_preflight.json", gpu)
    (
        phase_cache,
        _reporter_table,
        heads,
        x_state,
        _comparability,
        split_contract,
        preprocessing_sha256,
    ) = common_runner.prepare_data(args)
    _require(tuple(head.slug for head in heads) == FULL52, "Prepared training panel is not full52")
    runner.assert_development_heads_sealed(heads)
    comparability = runner.verify_identity_only_outer_test_cohorts(args, phase_cache, heads)
    head_schema = {head.slug: head.data.target_feature_names.astype(str).tolist() for head in heads}
    atomic_json(args.output_root / "head_schema.json", {
        "schema_version": SCHEMA_VERSION,
        "reporters": list(FULL52),
        "head_dimensions": {slug: len(names) for slug, names in head_schema.items()},
        "feature_names": head_schema,
        "sha256": common_runner.json_sha256(head_schema),
    })
    atomic_json(args.output_root / "formal_training_data_seal.json", {
        "schema_version": SCHEMA_VERSION,
        "outer_test_label_access_during_training": "physically_sealed",
        "prepared_test_label_rows": 0,
        "outer_test_identity_metadata_access": "frozen_cohort_fingerprint_only",
        "preprocessing_sha256": preprocessing_sha256,
        "comparability": comparability,
        "split_contract_sha256": harness.json_sha256(split_contract.to_manifest()),
    })
    staging = frozen_engine.build_gpu_training_staging(
        common_runner.build_staging_args(args), phase_cache, x_state, heads, "cuda"
    )
    _require(staging.metadata.get("enabled") is True, "Required GPU staging was not enabled")
    complete = runner.train_shared_method(
        args,
        campaign_payload=payload,
        trial=trial,
        binding=binding,
        phase_cache=phase_cache,
        heads=heads,
        x_state=x_state,
        gpu_staging=staging,
        split_contract=split_contract,
    )
    del staging, heads, x_state, phase_cache
    gc.collect()
    torch.cuda.empty_cache()
    return complete


def main() -> None:
    args = build_parser().parse_args()
    payload = _read_config(args.campaign_config.expanduser().resolve())
    _normalize_args(args, payload)
    runner = _configure_full52_binding()
    if args.evaluation_only:
        _evaluation_only(args)
        return
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    evaluation_complete = args.output_root / "evaluation_complete.json"
    if evaluation_complete.is_file():
        _write_formal_result(args)
        print(json.dumps({"status": "complete", "output_root": str(args.output_root)}, indent=2), flush=True)
        return
    training_complete = args.output_root / "training_complete.json"
    if args.resume and training_complete.is_file():
        complete = _load_completed_training_for_evaluation(args, payload)
        print(json.dumps({
            "status": "resuming_post_selection_evaluation",
            "method": METHOD_ID,
            "split": args.split,
            "fold": args.fold,
            "training_complete": str(training_complete),
            "training_reused": True,
            "development_training_result_used_as_formal_completion": False,
        }, indent=2), flush=True)
    else:
        complete = _train(args, payload, runner)
    _archive_intermediate_development_result(args)
    _spawn_evaluation_child(args, complete)
    _write_formal_result(args)
    print(json.dumps({
        "status": "complete",
        "method": METHOD_ID,
        "split": args.split,
        "fold": args.fold,
        "reporters": len(FULL52),
        "endpoints": FULL52_ENDPOINTS,
        "outer_test_evaluated": True,
        "output_root": str(args.output_root),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
