#!/usr/bin/env python3
"""Train one TabM or scButterfly specialist on a frozen OPS screen direction.

This is the deliberately narrow bridge between the already frozen 81
reporter-to-destination-screen tasks and the two neural specialist families.
It does not implement gene-by-screen tasks and it does not invent a shared
variant of either method.

The outer-screen labels are physically absent while preprocessing, training,
validation, and checkpoint selection run.  Only after all three evaluation
states have been frozen are the destination rows read and passed to the same
cell/gene evaluator used by the five classic specialists.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd

import ops_biological_baseline_harness_models as biological_models
import ops_reporter_training_harness_lib as harness
import run_ops_biological_baseline_training as biological_training
import run_ops_independent_specialist_protocol_task as strict_runner
import run_ops_reporter_candidate_training as candidate_training
import run_ops_reporter_masked_multitask_resmlp_v2 as frozen_engine
from ops_reporter_specialist_lib import PhaseCache, atomic_json, atomic_npy, hash_arrays


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ops-neural-specialist-screen-task-v1"
METHODS = ("tabm_specialists", "scbutterfly_ops_b")
# Keep the protocol identifier identical to the existing strict specialist
# task runner.  Presentation/aggregation code may label it whole-screen
# holdout, but the executable contract is ``strict_whole_screen``.
SPLIT = "strict_whole_screen"
EXPECTED_SCREEN_TASKS = 81
DEFAULT_STRICT_PLAN = (
    PROJECT_ROOT / "configs/ops_specialist_strict_screen_plan_v1/strict_tasks.json"
)
DEFAULT_PHASE_CACHE = PROJECT_ROOT / "data/processed/ops_phase172_indexed"
DEFAULT_EXACT_ROOT = PROJECT_ROOT / "data/processed/ops_full_reporter_exact/reporters"
DEFAULT_TARGET_TABLE = PROJECT_ROOT / "results/ops_phase0_asset_audit/reporter_targets.csv"
DEFAULT_FEATURE_DICTIONARY = (
    PROJECT_ROOT / "results/ops_phase0_asset_audit/target_feature_dictionary.csv"
)
DEFAULT_TABM_CAMPAIGN = PROJECT_ROOT / "configs/ops_reporter_candidate_training_harness_v2.json"
DEFAULT_BIOLOGICAL_CAMPAIGN = (
    PROJECT_ROOT / "configs/ops_biological_baseline_panel12_development_v2.json"
)
DEFAULT_ROUNDS_REFERENCE = (
    PROJECT_ROOT
    / "results/ops_reporter_masked_multitask_resmlp_v1"
    / "gene_holdout_main/fold_0/train/training_complete.json"
)
DEFAULT_TABM_WHEEL = candidate_training.DEFAULT_TABM_DEPENDENCY_WHEEL


class ScreenTaskError(RuntimeError):
    """Fail-closed error for a frozen screen-specialist job."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ScreenTaskError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_screen_task(plan_path: Path, task_id: str) -> tuple[dict[str, Any], int, str]:
    plan_path = plan_path.expanduser().resolve()
    require(plan_path.is_file(), f"Strict plan is missing: {plan_path}")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    require(
        payload.get("schema_version")
        == "ops-independent-specialists-strict-task-plan-v1",
        "Wrong strict task-plan schema",
    )
    tasks = payload.get("tasks")
    require(isinstance(tasks, list), "Strict plan tasks must be a list")
    screen_tasks = [row for row in tasks if row.get("task_type") == "screen"]
    require(
        len(screen_tasks) == EXPECTED_SCREEN_TASKS,
        f"Expected {EXPECTED_SCREEN_TASKS} screen tasks, found {len(screen_tasks)}",
    )
    require(
        len({str(row.get("task_id")) for row in screen_tasks}) == len(screen_tasks),
        "Strict screen task IDs are not unique",
    )
    matched = [row for row in screen_tasks if str(row.get("task_id")) == task_id]
    require(len(matched) == 1, f"Screen task {task_id!r} matched {len(matched)} rows")
    task = copy.deepcopy(matched[0])
    require(task.get("gene_fold") is None, "Screen-only task unexpectedly has a gene fold")
    require(bool(task.get("destination_screen")), "Screen task lacks destination_screen")
    require(bool(task.get("source_screens")), "Screen task lacks source_screens")
    require(
        task["destination_screen"] not in task["source_screens"],
        "Destination screen appears in frozen source_screens",
    )
    require(
        task.get("task_options", {}).get("calibration") == "zero_shot",
        "Screen task is not frozen as zero-shot destination calibration",
    )
    return task, screen_tasks.index(matched[0]), file_sha256(plan_path)


def selected_target_features(
    target_table: Path, dictionary: Path, reporter: str
) -> tuple[pd.Series, tuple[str, ...]]:
    table = pd.read_csv(target_table)
    require(
        len(table) == 52 and table["reporter_slug"].nunique() == 52,
        "Frozen reporter table must contain 52 unique reporters",
    )
    rows = table.loc[table["reporter_slug"].astype(str) == reporter]
    require(len(rows) == 1, f"Unknown frozen reporter: {reporter}")
    core = frozen_engine.load_technical_core_features(dictionary, table)
    require(reporter in core, f"Technical-core dictionary lacks {reporter}")
    return rows.iloc[0], tuple(map(str, core[reporter]))


def fingerprint(phase_rows: np.ndarray, source_rows: np.ndarray) -> harness.CohortFingerprint:
    phase_rows = np.asarray(phase_rows, dtype=np.int64)
    source_rows = np.asarray(source_rows, dtype=np.int64)
    require(len(phase_rows) > 0 and len(phase_rows) == len(source_rows), "Empty cohort")
    return harness.CohortFingerprint(
        n_statistical_units=int(len(np.unique(phase_rows))),
        n_observations=int(len(phase_rows)),
        unit_ids_sha256=hash_arrays(np.unique(phase_rows)),
        observation_ids_sha256=hash_arrays(phase_rows, source_rows),
    )


def load_accessible_training_head(
    *,
    args: argparse.Namespace,
    task: Mapping[str, Any],
    phase_cache: PhaseCache,
    reporter_row: pd.Series,
    feature_names: Sequence[str],
    strict_plan_sha256: str,
) -> tuple[
    frozen_engine.HeadPartition,
    frozen_engine.GlobalXPreprocessing,
    harness.FrozenSplitContract,
    dict[str, Any],
]:
    """Read source labels only and return a sealed train/validation head."""

    reporter = str(task["reporter_slug"])
    destination = str(task["destination_screen"])
    cache_path = args.exact_root / f"all_cells_fluor_{reporter}.exact.h5"
    require(cache_path.is_file(), f"Exact reporter cache is missing: {cache_path}")
    with h5py.File(cache_path, "r") as source:
        phase_all = np.asarray(source["phase_row_index"][:], dtype=np.int64)
        controls_all = np.asarray(source["metadata/is_control"][:], dtype=bool)
        columns, ordered_names = candidate_training.selected_feature_columns(
            source, feature_names, reporter
        )
        source_target_features = int(source["fluorescence"].shape[1])
        n_source_rows = int(source["fluorescence"].shape[0])

        train_h5, validation_h5, test_h5, partition = strict_runner.build_partition(
            protocol="strict_whole_screen",
            phase_cache=phase_cache,
            phase_rows=phase_all,
            is_control=controls_all,
            destination_screen=destination,
            outer_gene_fold=-1,
            source_validation_field_fold=args.source_validation_field_fold,
        )
        accessible_h5 = np.sort(np.concatenate((train_h5, validation_h5))).astype(
            np.int64
        )
        require(
            not np.intersect1d(accessible_h5, test_h5).size,
            "Destination H5 rows entered accessible training rows",
        )
        # This is the only fluorescence read before checkpoint selection.  The
        # row selector is exactly train U validation and excludes every test row.
        y_accessible = candidate_training.read_h5_rows_columns(
            source["fluorescence"], accessible_h5, columns
        )

    observed = np.isfinite(y_accessible)
    any_observed = observed.any(axis=1)
    complete = observed.all(axis=1)
    retained = complete
    retained_h5 = accessible_h5[retained]
    retained_phase = phase_all[retained_h5]
    retained_y = y_accessible[retained]
    retained_control = controls_all[retained_h5]
    retained_observed = observed[retained]
    is_train = np.isin(retained_h5, train_h5, assume_unique=True)
    is_validation = np.isin(retained_h5, validation_h5, assume_unique=True)
    require(np.all(is_train ^ is_validation), "Accessible row is not uniquely train/validation")
    train = np.flatnonzero(is_train).astype(np.int64)
    validation = np.flatnonzero(is_validation).astype(np.int64)
    require(len(train) > 0 and len(validation) > 0, "Empty complete train/validation cohort")

    data = frozen_engine.MaskedReporterData(
        slug=reporter,
        source_h5_rows=retained_h5,
        phase_rows=retained_phase,
        y=retained_y,
        observed_mask=retained_observed,
        is_complete=np.ones(len(retained_h5), dtype=bool),
        is_control=retained_control,
        target_feature_names=np.asarray(ordered_names, dtype=str),
        source_cache=cache_path,
        source_size_bytes=cache_path.stat().st_size,
        source_target_features=source_target_features,
        n_source_rows=n_source_rows,
        n_complete_rows=int(complete.sum()),
        n_partial_rows=int((any_observed & ~complete).sum()),
        n_all_missing_rows=int((~any_observed).sum()),
        all_missing_source_h5_rows=accessible_h5[~any_observed],
    )
    empty = np.empty(0, dtype=np.int64)
    head = frozen_engine.HeadPartition(
        slug=reporter,
        reporter_row=reporter_row,
        data=data,
        complete_train_indices=train,
        partial_train_indices=empty,
        active_train_indices=train.copy(),
        complete_validation_indices=validation,
        excluded_partial_validation_indices=empty,
        complete_test_indices=empty,
        excluded_partial_test_indices=empty,
        validation_fold=int(args.source_validation_field_fold),
        original_partition_counts={
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": 0,
            "outer_test_y_physically_loaded": False,
            "destination_h5_rows": int(len(test_h5)),
        },
    )
    head.y_preprocessing = frozen_engine.fit_head_y_preprocessing(data.y, train)
    x_state = frozen_engine.fit_global_x_preprocessing(
        phase_cache.x,
        np.unique(data.phase_rows[train]),
        args.min_x_finite_fraction,
    )
    require(
        np.array_equal(head.y_preprocessing.kept_indices, np.arange(data.y.shape[1])),
        "Strict screen source training unexpectedly drops a reporter endpoint",
    )
    require(
        np.array_equal(x_state.kept_indices, np.arange(172)),
        "Strict screen source training unexpectedly drops a phase feature",
    )

    contract = harness.FrozenSplitContract(
        split_name=SPLIT,
        fold=int(args.direction_index),
        split_manifest_sha256=strict_plan_sha256,
        statistical_unit=harness.StatisticalUnitContract(
            unit_name="exact_phase_cell_reporter_observation",
            unit_id_field="phase_row_index",
            metric_unit="cell_then_gene",
            resampling_unit="gene",
            repeated_unit_policy="cluster_by_unit",
        ),
        reporter_cohorts=(
            harness.ReporterCohortContract(
                reporter_slug=reporter,
                train=fingerprint(data.phase_rows[train], data.source_h5_rows[train]),
                validation=fingerprint(
                    data.phase_rows[validation], data.source_h5_rows[validation]
                ),
                test=fingerprint(phase_all[test_h5], test_h5),
            ),
        ),
    )
    seal = {
        "schema_version": SCHEMA_VERSION,
        "task_id": task["task_id"],
        "reporter_slug": reporter,
        "destination_screen": destination,
        "source_screens": list(task["source_screens"]),
        "source_validation_field_fold": int(args.source_validation_field_fold),
        "train_h5_rows_sha256": hash_arrays(train_h5),
        "validation_h5_rows_sha256": hash_arrays(validation_h5),
        "test_h5_rows_sha256": hash_arrays(test_h5),
        "train_phase_rows_sha256": hash_arrays(phase_all[train_h5]),
        "validation_phase_rows_sha256": hash_arrays(phase_all[validation_h5]),
        "test_phase_rows_sha256": hash_arrays(phase_all[test_h5]),
        "n_train_complete": int(len(train)),
        "n_validation_complete": int(len(validation)),
        "n_test_identity_rows": int(len(test_h5)),
        "destination_labels_read_before_checkpoint_freeze": False,
        "accessible_labels_read_from_h5_rows": "train_union_validation_only",
        "whole_destination_screen_removed_from_train_and_validation": True,
        "split_contract": contract.to_manifest(),
        "partition": partition,
    }
    test_identity = {
        "cache_path": str(cache_path),
        "test_h5_rows": test_h5,
        "test_phase_rows": phase_all[test_h5],
        "test_is_control": controls_all[test_h5],
        "columns": columns,
        "feature_names": np.asarray(ordered_names, dtype=str),
    }
    return head, x_state, contract, {"seal": seal, "test_identity": test_identity}


def load_rounds_per_epoch(path: Path) -> int:
    path = path.expanduser().resolve()
    require(path.is_file(), f"Reporter-round reference is missing: {path}")
    marker = json.loads(path.read_text(encoding="utf-8"))
    steps = int(marker.get("steps_per_epoch", 0))
    require(steps > 0 and steps % 13 == 0, "V1 reference steps are not 52-task rounds")
    return steps // 13


def tabm_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["methods"]["tabm_specialists"]["trial_configs"][0]
    normalized = candidate_training.strict_method_config(
        json.dumps(
            {
                "model": raw["model"]["options"],
                "optimizer": raw["optimizer"],
                "training": raw["training"],
            }
        )
    )
    adapted = copy.deepcopy(normalized)
    # One active specialist cannot form a four-reporter physical batch.  This
    # changes packing only: epochs, rounds, observations/reporter, optimizer,
    # architecture, and validation opportunities remain frozen.
    adapted["training"]["batch_heads"] = 1
    return normalized, adapted


def synthetic_gpu_smoke(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    endpoint_dim: int,
) -> None:
    """Exercise the method-native CUDA path without creating a formal marker.

    The smoke deliberately uses synthetic values: it checks construction,
    forward/backward, and optimizer ownership, but cannot read destination
    fluorescence or mutate a formal screen result directory.
    """

    import torch

    require(endpoint_dim > 0, "Synthetic smoke requires a positive endpoint width")
    gpu = candidate_training.configure_cuda(
        int(args.seed), args.method == "tabm_specialists"
    )
    reporter = str(task["reporter_slug"])
    batch_size = int(args.synthetic_smoke_batch_size)
    require(batch_size > 1, "Synthetic smoke batch size must exceed one")

    if args.method == "tabm_specialists":
        _, adapted = tabm_config(args.tabm_campaign)
        candidate_training.ensure_tabm_dependency(args.method, args.tabm_wheel)
        factory = candidate_training.candidate_models.CandidateModelConfig(
            model_type=candidate_training.METHOD_MODEL_TYPES[args.method],
            head_dimensions=OrderedDict(((reporter, endpoint_dim),)),
            input_dim=172,
            options=copy.deepcopy(adapted["model"]),
        )
        model = candidate_training.create_runner_candidate_model(factory).to("cuda")
        optimizer_config = adapted["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=(0.9, 0.999),
            eps=1.0e-8,
        )
        x = torch.randn(batch_size, 172, device="cuda")
        target = torch.randn(batch_size, endpoint_dim, device="cuda")
        observed = torch.ones_like(target, dtype=torch.bool)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            prediction = model.forward_grouped(x, [reporter], [batch_size])[reporter]
            loss = model.compute_loss(prediction, target, observed)
        require(bool(torch.isfinite(loss).item()), "TabM synthetic loss is non-finite")
        loss.backward()
        optimizer.step()
        details = {
            "optimizer_updates": 1,
            "losses": {"tabm_masked_member_mse": float(loss.detach().cpu())},
            "model_manifest": model.config_manifest(),
        }
    else:
        campaign = json.loads(args.biological_campaign.read_text(encoding="utf-8"))
        trial = copy.deepcopy(
            campaign["methods"]["scbutterfly_ops_b"]["trial_configs"][0]
        )
        biological_training.PANEL12 = (reporter,)
        dummy_head = SimpleNamespace(
            slug=reporter,
            data=SimpleNamespace(
                y=np.zeros((2, endpoint_dim), dtype=np.float32)
            ),
        )
        collection = biological_training.create_harness_collection(
            "scbutterfly_ops_b", trial, [dummy_head], device="cuda"
        )
        specialist = collection.specialist(reporter)
        controller = biological_training.ScButterflyTrainingController(
            specialist,
            biological_training.scbutterfly_controller_config(
                trial, rounds_per_epoch=1
            ),
        )
        x = torch.randn(batch_size, 172, device="cuda")
        target = torch.randn(batch_size, endpoint_dim, device="cuda")
        observed = torch.ones_like(target, dtype=torch.bool)
        paired = biological_training.ScButterflyPairedBatch(x, target, observed)
        previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            phase = controller.phase_pretrain_step(paired)
            controller.finish_phase_pretraining()
            phenotype = controller.phenotype_pretrain_step(paired)
            controller.finish_phenotype_pretraining()
            discriminator = controller.joint_discriminator_step(paired)
            generator = controller.joint_generator_step(paired)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        losses = {
            "phase_pretrain": float(phase.total.detach().cpu()),
            "phenotype_pretrain": float(phenotype.total.detach().cpu()),
            "joint_discriminator": float(discriminator.total.detach().cpu()),
            "joint_generator": float(generator.total.detach().cpu()),
        }
        require(all(np.isfinite(list(losses.values()))), "scButterfly smoke loss is non-finite")
        details = {
            "optimizer_updates": int(controller.counters.global_optimizer_steps),
            "losses": losses,
            "controller_counters": controller.counters.to_dict(),
            "model_manifest": collection.factory_manifest(),
        }

    marker = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "mode": "synthetic_gpu_smoke",
        "method_id": args.method,
        "task_id": args.task_id,
        "reporter_slug": reporter,
        "endpoint_dim": int(endpoint_dim),
        "device": "cuda",
        "gpu": gpu,
        "formal_marker_written": False,
        "destination_labels_read": False,
        **details,
    }
    atomic_json(args.output_root / "synthetic_gpu_smoke.json", marker)
    require(
        not (args.output_root / "training_result.json").exists(),
        "Synthetic smoke unexpectedly created a formal training marker",
    )
    print(json.dumps(marker, indent=2), flush=True)


def configure_runtime_args(args: argparse.Namespace, task: Mapping[str, Any]) -> None:
    args.method = str(args.method)
    args.split = SPLIT
    args.fold = int(args.direction_index)
    args.trial_index = 0
    args.seed = int(task["seed"])
    args.resume = not args.overwrite
    args.campaign_binding = {
        "schema_version": SCHEMA_VERSION,
        "strict_plan": str(args.strict_plan.resolve()),
        "strict_plan_sha256": args.strict_plan_sha256,
        "task": copy.deepcopy(dict(task)),
        "direction_index": int(args.direction_index),
        "hyperparameter_search": False,
        "outer_test_used_for_selection": False,
    }
    args.reporter_rounds_per_epoch_override = load_rounds_per_epoch(
        args.rounds_reference
    )
    args.gpu_staging = args.gpu_staging
    args.gpu_staging_reserve_gib = float(args.gpu_staging_reserve_gib)
    args.gpu_staging_chunk_rows = int(args.gpu_staging_chunk_rows)


def write_preprocessing(
    path: Path,
    head: frozen_engine.HeadPartition,
    x_state: frozen_engine.GlobalXPreprocessing,
    args: argparse.Namespace,
) -> str:
    assert head.y_preprocessing is not None
    payload = {
        "schema_version": SCHEMA_VERSION,
        "policy": "strict_source_train_only_fit",
        "strict_plan_sha256": args.strict_plan_sha256,
        "task_id": args.task_id,
        "destination_screen_used_for_fit": False,
        "x": x_state.to_json(),
        "heads": {head.slug: head.y_preprocessing.to_json()},
    }
    atomic_json(path, payload)
    return file_sha256(path)


def train_method(
    args: argparse.Namespace,
    task: Mapping[str, Any],
    phase_cache: PhaseCache,
    head: frozen_engine.HeadPartition,
    x_state: frozen_engine.GlobalXPreprocessing,
    split_contract: harness.FrozenSplitContract,
) -> tuple[dict[str, Any], OrderedDict[str, Path], dict[str, Any]]:
    staging = frozen_engine.build_gpu_training_staging(
        args, phase_cache, x_state, [head], "cuda"
    )
    require(
        staging.metadata.get("enabled") is True,
        "Strict neural screen task requires GPU-resident training data",
    )
    if args.method == "tabm_specialists":
        original, adapted = tabm_config(args.tabm_campaign)
        candidate_training.ensure_tabm_dependency(args.method, args.tabm_wheel)
        args.amp = True
        complete, states = candidate_training.train_model(
            args,
            adapted,
            phase_cache,
            [head],
            x_state,
            staging,
            split_contract,
        )
        binding = {
            "frozen_method_config": original,
            "execution_method_config": adapted,
            "single_reporter_packing_adaptation": {"batch_heads": [4, 1]},
        }
        state_paths = OrderedDict(
            (name, Path(states[name]))
            for name in ("single", "ema", "checkpoint_average")
        )
    else:
        campaign = json.loads(args.biological_campaign.read_text(encoding="utf-8"))
        trial = copy.deepcopy(campaign["methods"]["scbutterfly_ops_b"]["trial_configs"][0])
        # The reusable source-faithful controller binds its active panel through
        # this process-local constant.  No shared encoder or reporter query is
        # introduced.
        biological_training.PANEL12 = (head.slug,)
        args.amp = False
        args.tf32 = False
        binding = {
            "schema_version": SCHEMA_VERSION,
            "method": args.method,
            "trial_index": 0,
            "trial_sha256": json_sha256(trial),
            "strict_task_sha256": json_sha256(task),
            "active_reporters": [head.slug],
            "hyperparameter_search": False,
            "outer_test_used_for_selection": False,
        }
        complete = biological_training.train_scbutterfly_specialists(
            args,
            campaign_payload=campaign,
            trial=trial,
            binding=binding,
            phase_cache=phase_cache,
            heads=[head],
            x_state=x_state,
            gpu_staging=staging,
            split_contract=split_contract,
        )
        state_paths = OrderedDict(
            (name, Path(complete["state_paths"][name]))
            for name in ("single", "ema", "checkpoint_average")
        )
    del staging
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    return complete, state_paths, binding


def freeze_evaluation_recipe(
    args: argparse.Namespace,
    training_complete: Mapping[str, Any],
    state_paths: Mapping[str, Path],
) -> dict[str, Any]:
    training_path = args.output_root / "training_complete.json"
    require(training_path.is_file(), "Training loop did not freeze training_complete.json")
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "role": "post_checkpoint_destination_screen_evaluation",
        "method_id": args.method,
        "task_id": args.task_id,
        "strict_plan_sha256": args.strict_plan_sha256,
        "training_complete_sha256": file_sha256(training_path),
        "checkpoint_selection_sha256": json_sha256(
            training_complete["checkpoint_selection"]
        ),
        "states": {
            name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for name, path in state_paths.items()
        },
        "destination_labels_loaded_during_training": False,
        "destination_labels_may_be_loaded_after_this_recipe": True,
        "test_may_select_state": False,
        "training_or_checkpoint_mutation_allowed": False,
    }
    atomic_json(args.output_root / "evaluation_recipe.json", recipe)
    return recipe


def load_test_labels(
    identity: Mapping[str, Any], y_state: frozen_engine.HeadYPreprocessing
) -> dict[str, Any]:
    cache_path = Path(identity["cache_path"])
    rows = np.asarray(identity["test_h5_rows"], dtype=np.int64)
    with h5py.File(cache_path, "r") as source:
        y = candidate_training.read_h5_rows_columns(
            source["fluorescence"], rows, np.asarray(identity["columns"], dtype=np.int64)
        )
    complete = np.isfinite(y).all(axis=1)
    require(np.any(complete), "Destination screen has no complete test rows")
    y = y[complete][:, y_state.kept_indices]
    return {
        "truth_raw": np.asarray(y, dtype=np.float32),
        "truth_scaled": y_state.transform(y),
        "phase_rows": np.asarray(identity["test_phase_rows"], dtype=np.int64)[complete],
        "source_h5_rows": rows[complete],
        "is_control": np.asarray(identity["test_is_control"], dtype=bool)[complete],
        "feature_names": np.asarray(identity["feature_names"], dtype=str)[
            y_state.kept_indices
        ],
    }


def load_state_model(method: str, path: Path) -> Any:
    import torch

    if method == "tabm_specialists":
        model, _ = candidate_training.load_model_from_checkpoint(path, "cuda")
        return model
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(
        payload.get("schema_version") == biological_training.CHECKPOINT_SCHEMA_VERSION,
        f"Incompatible scButterfly checkpoint: {path}",
    )
    model = biological_models.load_harness_model(
        payload["model_manifest"], payload["model_state"], strict=True
    )
    model.to("cuda").eval()
    return model


def predict_scaled(
    *,
    args: argparse.Namespace,
    model: Any,
    reporter: str,
    phase_cache: PhaseCache,
    phase_rows: np.ndarray,
    x_state: frozen_engine.GlobalXPreprocessing,
) -> np.ndarray:
    import torch

    outputs: list[np.ndarray] = []
    for start in range(0, len(phase_rows), args.prediction_batch_size):
        stop = min(start + args.prediction_batch_size, len(phase_rows))
        x = x_state.transform(np.asarray(phase_cache.x[phase_rows[start:stop]], dtype=np.float32))
        tensor = torch.from_numpy(x).to("cuda")
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.amp)
        ):
            if args.method == "tabm_specialists":
                prediction = model.predict(tensor, reporter)
            else:
                prediction = model.predict_reporter(tensor, reporter)
        outputs.append(prediction.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evaluate_states(
    *,
    args: argparse.Namespace,
    task: Mapping[str, Any],
    recipe: Mapping[str, Any],
    state_paths: Mapping[str, Path],
    phase_cache: PhaseCache,
    x_state: frozen_engine.GlobalXPreprocessing,
    y_state: frozen_engine.HeadYPreprocessing,
    test_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    test = load_test_labels(test_identity, y_state)
    screen_codes = np.asarray(
        phase_cache.metadata["screen_code"][test["phase_rows"]], dtype=np.int32
    )
    gene_codes = np.asarray(
        phase_cache.metadata["gene_code"][test["phase_rows"]], dtype=np.int32
    )
    rows: list[dict[str, Any]] = []
    for state_id, path in state_paths.items():
        require(file_sha256(path) == recipe["states"][state_id]["sha256"], "State drift")
        model = load_state_model(args.method, path)
        prediction_scaled = predict_scaled(
            args=args,
            model=model,
            reporter=str(task["reporter_slug"]),
            phase_cache=phase_cache,
            phase_rows=test["phase_rows"],
            x_state=x_state,
        )
        prediction_raw = y_state.inverse(prediction_scaled)
        destination = (
            args.output_root
            / "evaluation_states"
            / state_id
            / "reporters"
            / str(task["reporter_slug"])
        )
        evaluated = strict_runner.evaluate_prediction(
            output_dir=destination,
            truth_raw=test["truth_raw"],
            prediction_raw=prediction_raw,
            truth_scaled=test["truth_scaled"],
            prediction_scaled=prediction_scaled,
            is_control=test["is_control"],
            screen_codes=screen_codes,
            gene_codes=gene_codes,
            feature_names=test["feature_names"],
            baseline_mean_raw=y_state.mean,
            seed=int(args.seed),
            binding={
                "method_id": args.method,
                "task_id": args.task_id,
                "reporter_slug": task["reporter_slug"],
                "destination_screen": task["destination_screen"],
                "evaluation_state": state_id,
                "strict_plan_sha256": args.strict_plan_sha256,
            },
            save_predictions=args.save_predictions,
        )
        rows.append(
            {
                "evaluation_state": state_id,
                "checkpoint_sha256": file_sha256(path),
                **{f"cell_{key}": value for key, value in evaluated["cell_metrics"].items()},
                **{f"gene_{key}": value for key, value in evaluated["gene_metrics"].items()},
            }
        )
        del model
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(args.output_root / "evaluation_states_summary.csv", index=False)
    return rows


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--method", required=True, choices=METHODS)
    result.add_argument("--strict-plan", type=Path, default=DEFAULT_STRICT_PLAN)
    result.add_argument("--task-id", required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--phase-cache", type=Path, default=DEFAULT_PHASE_CACHE)
    result.add_argument("--exact-root", type=Path, default=DEFAULT_EXACT_ROOT)
    result.add_argument("--target-table", type=Path, default=DEFAULT_TARGET_TABLE)
    result.add_argument("--target-feature-dictionary", type=Path, default=DEFAULT_FEATURE_DICTIONARY)
    result.add_argument("--tabm-campaign", type=Path, default=DEFAULT_TABM_CAMPAIGN)
    result.add_argument("--biological-campaign", type=Path, default=DEFAULT_BIOLOGICAL_CAMPAIGN)
    result.add_argument("--tabm-wheel", type=Path, default=DEFAULT_TABM_WHEEL)
    result.add_argument("--rounds-reference", type=Path, default=DEFAULT_ROUNDS_REFERENCE)
    result.add_argument("--source-validation-field-fold", type=int, default=0, choices=range(5))
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--prediction-batch-size", type=int, default=8192)
    result.add_argument("--throughput-batch-size", type=int, default=8192)
    result.add_argument("--throughput-warmup-iterations", type=int, default=3)
    result.add_argument("--throughput-timed-iterations", type=int, default=10)
    result.add_argument("--gpu-staging", choices=("required",), default="required")
    result.add_argument("--gpu-staging-reserve-gib", type=float, default=16.0)
    result.add_argument("--gpu-staging-chunk-rows", type=int, default=131072)
    result.add_argument("--min-x-finite-fraction", type=float, default=0.80)
    result.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--partition-only", action="store_true")
    result.add_argument("--contract-check-only", action="store_true")
    result.add_argument("--synthetic-gpu-smoke", action="store_true")
    result.add_argument("--synthetic-smoke-batch-size", type=int, default=32)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for field in (
        "strict_plan",
        "output_root",
        "phase_cache",
        "exact_root",
        "target_table",
        "target_feature_dictionary",
        "tabm_campaign",
        "biological_campaign",
        "tabm_wheel",
        "rounds_reference",
    ):
        setattr(args, field, Path(getattr(args, field)).expanduser().resolve())
    task, direction_index, plan_sha = load_screen_task(args.strict_plan, args.task_id)
    args.direction_index = direction_index
    args.strict_plan_sha256 = plan_sha
    configure_runtime_args(args, task)
    if args.contract_check_only:
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "method": args.method,
                    "task_id": args.task_id,
                    "task_type": "screen",
                    "direction_index": direction_index,
                    "strict_plan_sha256": plan_sha,
                    "gene_x_screen_supported": False,
                    "gpu_started": False,
                },
                indent=2,
            )
        )
        return 0

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / "training_result.json"
    if result_path.is_file() and not args.overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            existing.get("completed") is True
            and existing.get("method_id") == args.method
            and existing.get("task_id") == args.task_id
            and existing.get("strict_plan_sha256") == plan_sha
        ):
            print(f"SKIP complete task: {result_path}", flush=True)
            return 0
        raise ScreenTaskError(f"Incompatible completion marker: {result_path}")

    reporter_row, features = selected_target_features(
        args.target_table, args.target_feature_dictionary, str(task["reporter_slug"])
    )
    if args.synthetic_gpu_smoke:
        synthetic_gpu_smoke(args, task, len(features))
        return 0
    phase_cache = PhaseCache.open(args.phase_cache)
    head, x_state, split_contract, prepared = load_accessible_training_head(
        args=args,
        task=task,
        phase_cache=phase_cache,
        reporter_row=reporter_row,
        feature_names=features,
        strict_plan_sha256=plan_sha,
    )
    atomic_json(args.output_root / "sealed_training_data.json", prepared["seal"])
    atomic_npy(
        args.output_root / "test_phase_row_index.npy",
        np.asarray(prepared["test_identity"]["test_phase_rows"], dtype=np.int64),
    )
    preprocessing_sha = write_preprocessing(
        args.output_root / "preprocessing.json", head, x_state, args
    )
    if args.partition_only:
        atomic_json(
            args.output_root / "partition_only_complete.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "task_id": args.task_id,
                "method_id": args.method,
                "strict_plan_sha256": plan_sha,
                "destination_labels_read": False,
                "gpu_started": False,
            },
        )
        print(json.dumps(prepared["seal"], indent=2), flush=True)
        return 0

    import torch

    torch.set_num_threads(args.workers)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    # scButterfly source anchor is FP32/no-TF32; TabM uses the frozen BF16/TF32
    # common harness.  Both remain CUDA-only.
    args.tf32 = args.method == "tabm_specialists"
    args.amp = args.method == "tabm_specialists"
    gpu = candidate_training.configure_cuda(args.seed, args.tf32)
    atomic_json(args.output_root / "gpu_preflight.json", gpu)
    started = time.time()
    complete, state_paths, method_binding = train_method(
        args, task, phase_cache, head, x_state, split_contract
    )
    y_state = head.y_preprocessing
    assert y_state is not None
    # Drop every source label tensor before destination fluorescence is read.
    del head
    gc.collect()
    torch.cuda.empty_cache()
    recipe = freeze_evaluation_recipe(args, complete, state_paths)
    evaluation_rows = evaluate_states(
        args=args,
        task=task,
        recipe=recipe,
        state_paths=state_paths,
        phase_cache=phase_cache,
        x_state=x_state,
        y_state=y_state,
        test_identity=prepared["test_identity"],
    )
    summary_path = args.output_root / "evaluation_states_summary.csv"
    result = {
        "schema_version": SCHEMA_VERSION,
        "completed": True,
        "status": "complete",
        "mode": "formal",
        "method_id": args.method,
        "split": SPLIT,
        "task_id": args.task_id,
        "task_type": "screen",
        "direction_index": int(direction_index),
        "reporter_slug": task["reporter_slug"],
        "destination_screen": task["destination_screen"],
        "source_screens": list(task["source_screens"]),
        "strict_plan_sha256": plan_sha,
        "seed": int(args.seed),
        "trial_index": 0,
        "n_evaluation_states": 3,
        "evaluation_states": [row["evaluation_state"] for row in evaluation_rows],
        "evaluation_states_summary_sha256": file_sha256(summary_path),
        "preprocessing_sha256": preprocessing_sha,
        "method_binding": method_binding,
        "method_binding_sha256": json_sha256(method_binding),
        "destination_labels_loaded_during_training": False,
        "destination_labels_loaded_after_checkpoint_selection": True,
        "selection_used_test": False,
        "test_may_select_state": False,
        "test_evaluated": True,
        "cpu_fallback_allowed": False,
        "runtime_seconds": time.time() - started,
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
