#!/usr/bin/env python3
"""Run the decisive OPS low-label transfer experiment on gene-holdout fold 0.

The frozen panel12 reporters are the adaptation targets. Identifier-query and
semantic-query models train on all 52 reporters: donor40 retain their complete
labels while target12 train/validation labels are reduced to the requested
fraction. The independent MLP sees only the same reduced target12 labels.
Outer-test labels remain physically sealed until a validation-selected
checkpoint has been frozen.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import run_ops_reporter_candidate_training as base
from ops_reporter_specialist_lib import PhaseCache, atomic_json, hash_arrays


SCHEMA_VERSION = "ops-low-label-transfer-fivefold-v4"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_METHOD = {
    # The common harness already implements per-reporter checkpoint
    # composition for disjoint specialists under this internal method id.
    # The factory mapping is overridden below to instantiate the ordinary MLP,
    # not TabM.
    "mlp": "tabm_specialists",
    "q_id": "q_id_52",
    "q_semantic": "q_semantic_52",
    "q_semantic_permuted": "q_semantic_52",
}
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "mlp": {
        "model": {"hidden_dims": [256, 128], "dropout": 0.0},
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
        },
        "training": {"batch_heads": 4, "observations_per_head": 1024},
    },
    "q_id": {
        "model": {
            "hidden_dim": 512,
            "query_dim": 128,
            "reporter_embedding_dim": 64,
            "endpoint_embedding_dim": 128,
            "bilinear_rank": 128,
            "encoder_depth": 3,
            "head_hidden_dims": [256, 128],
            "dropout": 0.05,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
        },
        "training": {"batch_heads": 4, "observations_per_head": 1024},
    },
    "q_semantic": {
        "model": {
            "hidden_dim": 512,
            "query_dim": 128,
            "reporter_embedding_dim": 64,
            "endpoint_embedding_dim": 128,
            "semantic_embedding_dim": 16,
            "bilinear_rank": 128,
            "encoder_depth": 3,
            "head_hidden_dims": [256, 128],
            "dropout": 0.05,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
        },
        "training": {"batch_heads": 4, "observations_per_head": 1024},
    },
}
MODEL_CONFIGS["q_semantic_permuted"] = copy.deepcopy(MODEL_CONFIGS["q_semantic"])


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=tuple(MODEL_METHOD))
    p.add_argument("--label-fraction", required=True, type=float)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--seed", type=int, default=20260726)
    p.add_argument("--split", default="gene_holdout_main", choices=("gene_holdout_main",))
    p.add_argument("--fold", default=0, type=int, choices=range(5))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--evaluation-only", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--evaluation-recipe-sha256", default="", help=argparse.SUPPRESS)
    p.add_argument("--gpu-staging-reserve-gib", type=float, default=14.0)
    p.add_argument("--gpu-staging-chunk-rows", type=int, default=131072)
    p.add_argument("--prediction-batch-size", type=int, default=8192)
    p.add_argument("--throughput-batch-size", type=int, default=8192)
    p.add_argument("--throughput-warmup-iterations", type=int, default=3)
    p.add_argument("--throughput-timed-iterations", type=int, default=10)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--bootstrap-draws", type=int, default=1000)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--phase-cache", type=Path, default=base.DEFAULT_PHASE_CACHE)
    p.add_argument("--exact-root", type=Path, default=base.DEFAULT_EXACT_ROOT)
    p.add_argument("--target-table", type=Path, default=base.DEFAULT_TARGET_TABLE)
    p.add_argument(
        "--target-feature-dictionary",
        type=Path,
        default=base.DEFAULT_TARGET_FEATURE_DICTIONARY,
    )
    p.add_argument(
        "--specialist-reference-root",
        type=Path,
        default=base.DEFAULT_SPECIALIST_REFERENCE,
    )
    p.add_argument(
        "--preprocessing-reference-root",
        type=Path,
        default=base.DEFAULT_PREPROCESSING_REFERENCE,
    )
    p.add_argument("--endpoint-semantics", type=Path, default=base.DEFAULT_ENDPOINT_SEMANTICS)
    p.add_argument("--semantic-vocabulary", type=Path, default=base.DEFAULT_SEMANTIC_VOCABULARY)
    p.add_argument("--semantic-manifest", type=Path, default=base.DEFAULT_SEMANTIC_MANIFEST)
    return p


def normalize_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.label_fraction <= 1.0:
        raise ValueError("--label-fraction must be in (0, 1]")
    if args.gpu_staging_reserve_gib < 12.0:
        raise ValueError("At least 12 GiB VRAM must remain reserved")
    for name in (
        "output_root",
        "phase_cache",
        "exact_root",
        "target_table",
        "target_feature_dictionary",
        "specialist_reference_root",
        "preprocessing_reference_root",
        "endpoint_semantics",
        "semantic_vocabulary",
        "semantic_manifest",
    ):
        setattr(args, name, Path(getattr(args, name)).expanduser().resolve())
    args.method = MODEL_METHOD[args.model]
    args.reporters = "all"
    args.tabm_dependency_wheel = base.DEFAULT_TABM_DEPENDENCY_WHEEL
    args.trial_index = 0
    args.mode = "formal"
    args.device = "cuda"
    args.gpu_staging = "required"
    # The frozen evaluator reads this common-harness policy during the
    # post-selection evaluation subprocess. Low-label evaluation is strictly
    # resume-only and must never overwrite a completed endpoint result.
    args.overwrite = False
    args.require_specialist_reference = True
    args.max_train_observations_per_head = 0
    args.max_validation_observations_per_head = 0
    args.max_test_observations_per_head = 0
    args.v1_reference_root = args.preprocessing_reference_root
    args.budget_policy = "v1_locked"
    args.arm = "low_label_transfer"
    args.checkpoint_objective_reporters = tuple(base.PANEL12)
    args.campaign_binding = {
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "method_id": args.method,
        "split": args.split,
        "fold": args.fold,
        "label_fraction": args.label_fraction,
        "seed": args.seed,
        "target_reporters": list(base.PANEL12),
        "donor_policy": "other_40_full_labels_for_query_models_only",
        "target_label_policy": "same_fraction_of_train_and_inner_validation",
        "outer_test_policy": "physically_sealed_until_checkpoint_selection",
        "smoke": bool(args.smoke),
        "semantic_control": (
            {
                "name": "reporter_level_biology_compartment_permutation",
                "fields": [
                    "biology_category",
                    "subcellular_compartment",
                ],
                "permutation_seed": 2026072701,
                "preserved": [
                    "reporter_id_embedding",
                    "endpoint_id_embedding",
                    "target_semantic",
                    "measurement_family",
                    "statistic_type",
                    "architecture",
                    "parameter_count",
                    "optimizer",
                    "sampler",
                    "labels",
                    "split",
                ],
            }
            if args.model == "q_semantic_permuted"
            else None
        ),
    }


def install_semantic_permutation_control(args: argparse.Namespace) -> None:
    """Permute biology/compartment records between equal-width reporters.

    Equal endpoint width keeps each complete semantic vector intact and
    preserves the global endpoint-weighted ID histogram.  Groups with only one
    reporter (the 20D and 60D blocks) cannot be permuted and are recorded as
    fixed rather than silently altered.
    """

    if args.model != "q_semantic_permuted":
        return
    original_loader = base.load_query_semantic_binding
    permutation_seed = 2026072701
    permuted_fields = ("biology_category", "subcellular_compartment")

    def load_permuted(
        call_args: argparse.Namespace,
        heads: Sequence[base.frozen_engine.HeadPartition],
    ) -> tuple[dict[str, dict[str, list[int]]], dict[str, int], dict[str, Any]]:
        semantic_ids, vocab_sizes, binding = original_loader(call_args, heads)
        dimensions: dict[int, list[str]] = {}
        for head in heads:
            dimensions.setdefault(int(head.data.y.shape[1]), []).append(head.slug)
        rng = np.random.default_rng(permutation_seed)
        source_for_destination: dict[str, str] = {}
        unavoidable_fixed: list[str] = []
        for dimension in sorted(dimensions):
            reporters = dimensions[dimension]
            if len(reporters) == 1:
                source_for_destination[reporters[0]] = reporters[0]
                unavoidable_fixed.append(reporters[0])
                continue
            order = [reporters[index] for index in rng.permutation(len(reporters))]
            shift = int(rng.integers(1, len(order)))
            for index, destination in enumerate(order):
                source_for_destination[destination] = order[
                    (index + shift) % len(order)
                ]
        movable = [
            reporter
            for reporter, source in source_for_destination.items()
            if reporter != source
        ]
        expected_movable = len(heads) - len(unavoidable_fixed)
        if len(movable) != expected_movable:
            raise RuntimeError("Semantic permutation is not a width-stratified derangement")

        original_sha = base.json_sha256(semantic_ids)
        permuted = copy.deepcopy(semantic_ids)
        for destination, source in source_for_destination.items():
            for field in permuted_fields:
                source_values = semantic_ids[source][field]
                destination_values = semantic_ids[destination][field]
                if len(source_values) != len(destination_values):
                    raise RuntimeError("Width-stratified semantic permutation changed length")
                permuted[destination][field] = list(source_values)
        after_sha = base.json_sha256(permuted)
        if after_sha == original_sha:
            raise RuntimeError("Semantic permutation did not change the semantic registry")
        before_histograms = {
            field: np.bincount(
                np.concatenate(
                    [
                        np.asarray(semantic_ids[head.slug][field], dtype=np.int64)
                        for head in heads
                    ]
                ),
                minlength=vocab_sizes[field],
            ).tolist()
            for field in permuted_fields
        }
        after_histograms = {
            field: np.bincount(
                np.concatenate(
                    [
                        np.asarray(permuted[head.slug][field], dtype=np.int64)
                        for head in heads
                    ]
                ),
                minlength=vocab_sizes[field],
            ).tolist()
            for field in permuted_fields
        }
        if before_histograms != after_histograms:
            raise RuntimeError("Semantic permutation changed the global ID histogram")
        binding = {
            **binding,
            "semantic_permutation_control": {
                "name": "reporter_level_biology_compartment_permutation",
                "seed": permutation_seed,
                "fields": list(permuted_fields),
                "stratification": "reporter_endpoint_dimension",
                "source_for_destination": source_for_destination,
                "n_permuted_reporters": len(movable),
                "unavoidable_singleton_fixed_reporters": unavoidable_fixed,
                "original_semantic_ids_sha256": original_sha,
                "permuted_semantic_ids_sha256": after_sha,
                "global_id_histograms_preserved": True,
                "target_semantic_permuted": False,
                "measurement_and_statistic_semantics_permuted": False,
            },
        }
        return permuted, vocab_sizes, binding

    base.load_query_semantic_binding = load_permuted


def stratified_label_subset(
    head: base.frozen_engine.HeadPartition,
    indices: np.ndarray,
    phase_cache: PhaseCache,
    fraction: float,
    seed: int,
    partition_name: str,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if fraction >= 1.0 or len(indices) <= 1:
        return indices.copy()
    phase_rows = np.asarray(head.data.phase_rows[indices], dtype=np.int64)
    genes = np.asarray(phase_cache.metadata["gene_code"][phase_rows], dtype=np.int64)
    controls = np.asarray(head.data.is_control[indices], dtype=np.int8)
    keys = np.rec.fromarrays((controls, genes), names=("control", "gene"))
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    requested = max(1, int(round(fraction * len(indices))))
    target_count = min(len(indices), requested)

    # Hamilton apportionment gives an exact global label budget while retaining
    # proportional representation of control-status x gene strata.  In the
    # extreme-low-label regime it deliberately permits zero selected cells for
    # some genes; forcing one per gene would make 0.1% labels much larger than
    # the stated budget.
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
        raise RuntimeError("Stratified low-label quota allocation failed")

    selected: list[np.ndarray] = []
    stable_seed = base.frozen_engine.stable_seed(
        seed, head.slug, partition_name, f"{fraction:.8f}"
    )
    for group, quota in enumerate(quotas):
        positions = np.flatnonzero(inverse == group)
        rng = np.random.default_rng(stable_seed + group)
        chosen = positions[rng.permutation(len(positions))[: int(quota)]]
        selected.append(indices[chosen])
    result = np.sort(np.concatenate(selected).astype(np.int64, copy=False))
    if len(result) != target_count or len(np.unique(result)) != len(result):
        raise RuntimeError("Low-label subset identity is not exact and unique")
    return result


def prepare_training_data(
    args: argparse.Namespace,
) -> tuple[
    PhaseCache,
    list[base.frozen_engine.HeadPartition],
    base.frozen_engine.GlobalXPreprocessing,
    base.harness.FrozenSplitContract,
    str,
]:
    frozen_table = pd.read_csv(args.target_table)
    all_slugs = tuple(frozen_table["reporter_slug"].astype(str))
    if len(all_slugs) != 52 or len(set(all_slugs)) != 52:
        raise RuntimeError("Frozen reporter registry must contain exactly 52 reporters")
    target_set = set(base.PANEL12)
    selected_slugs = base.PANEL12 if args.model == "mlp" else all_slugs
    if args.smoke:
        selected_slugs = (
            tuple(base.PANEL12[:4])
            if args.model == "mlp"
            # The query semantic manifest is intentionally frozen against the
            # complete 52-reporter endpoint registry.  Keep that schema intact
            # in smoke mode and reduce only reporter rounds per epoch.
            else all_slugs
        )
        args.reporter_rounds_per_epoch_override = 1
        args.checkpoint_objective_reporters = tuple(
            slug for slug in selected_slugs if slug in target_set
        )

    indexed = frozen_table.set_index("reporter_slug", drop=False)
    reporter_table = pd.DataFrame([indexed.loc[slug] for slug in selected_slugs])
    phase_cache = PhaseCache.open(args.phase_cache)
    base.validate_frozen_assets(phase_cache, frozen_table, args.exact_root)
    technical = base.frozen_engine.load_technical_core_features(
        args.target_feature_dictionary, frozen_table
    )
    heads: list[base.frozen_engine.HeadPartition] = []
    for _, row in reporter_table.iterrows():
        slug = str(row.reporter_slug)
        data = base.load_sealed_training_data(
            base.exact_cache_path(args.exact_root, slug),
            slug,
            technical[slug],
            phase_cache,
            args.split,
            args.fold,
        )
        heads.append(base.build_sealed_training_partition(args, phase_cache, row, data))

    reference_path = (
        args.preprocessing_reference_root
        / args.split
        / f"fold_{args.fold}"
        / "train"
        / "preprocessing.json"
    )
    x_state = base.load_reference_preprocessing(reference_path, heads)
    sample_rows: list[dict[str, Any]] = []
    for head in heads:
        is_target = head.slug in target_set
        original_train = head.active_train_indices.copy()
        original_validation = head.complete_validation_indices.copy()
        if is_target:
            head.complete_train_indices = stratified_label_subset(
                head,
                head.complete_train_indices,
                phase_cache,
                args.label_fraction,
                args.seed,
                "train",
            )
            head.active_train_indices = head.complete_train_indices.copy()
            head.partial_train_indices = np.empty(0, dtype=np.int64)
            head.complete_validation_indices = stratified_label_subset(
                head,
                head.complete_validation_indices,
                phase_cache,
                args.label_fraction,
                args.seed,
                "validation",
            )
            head.y_preprocessing = base.frozen_engine.fit_head_y_preprocessing(
                head.data.y, head.complete_train_indices
            )
        sample_rows.append(
            {
                "reporter_slug": head.slug,
                "role": "target12" if is_target else "donor40",
                "requested_fraction": args.label_fraction if is_target else 1.0,
                "original_train": len(original_train),
                "selected_train": len(head.active_train_indices),
                "original_validation": len(original_validation),
                "selected_validation": len(head.complete_validation_indices),
                "train_phase_source_sha256": hash_arrays(
                    head.data.phase_rows[head.active_train_indices],
                    head.data.source_h5_rows[head.active_train_indices],
                ),
                "validation_phase_source_sha256": hash_arrays(
                    head.data.phase_rows[head.complete_validation_indices],
                    head.data.source_h5_rows[head.complete_validation_indices],
                ),
            }
        )
    _, _ = base.frozen_engine.global_partition_integrity(heads)
    split_contract = base.build_split_contract(args, heads)
    args.output_root.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_root / "low_label_cohort_manifest.csv"
    pd.DataFrame(sample_rows).to_csv(sample_path, index=False)
    preprocessing = {
        "schema_version": SCHEMA_VERSION,
        "policy": "target12_stratified_low_label_donor40_full_label",
        "reference": str(reference_path.resolve()),
        "reference_sha256": base.file_sha256(reference_path),
        "target_reporters": list(base.PANEL12),
        "donor_reporters": [slug for slug in all_slugs if slug not in target_set],
        "label_fraction": args.label_fraction,
        "train_and_validation_both_subsampled": True,
        "stratification": "exact_budget_hamilton_apportionment_control_status_x_gene_code",
        "minimum_one_per_gene_forbidden": True,
        "selection_uses_y_values": False,
        "x_fit_policy": "full_unlabelled_phase_training_cohort",
        "target_y_fit_policy": "selected_low_label_training_rows_only",
        "outer_test_y_physically_loaded": False,
        "cohort_manifest": str(sample_path.resolve()),
        "cohort_manifest_sha256": base.file_sha256(sample_path),
        "x": x_state.to_json(),
        "heads": {
            head.slug: head.y_preprocessing.to_json()  # type: ignore[union-attr]
            for head in heads
        },
    }
    preprocessing_path = args.output_root / "preprocessing.json"
    atomic_json(preprocessing_path, preprocessing)
    return (
        phase_cache,
        heads,
        x_state,
        split_contract,
        base.file_sha256(preprocessing_path),
    )


def prepare_evaluation_data(
    args: argparse.Namespace,
) -> tuple[
    PhaseCache,
    list[base.frozen_engine.HeadPartition],
    base.frozen_engine.GlobalXPreprocessing,
    dict[str, dict[str, Any]],
    base.frozen_engine.GPUTrainingStaging,
    str,
]:
    frozen_table = pd.read_csv(args.target_table)
    indexed = frozen_table.set_index("reporter_slug", drop=False)
    phase_cache = PhaseCache.open(args.phase_cache)
    base.validate_frozen_assets(phase_cache, frozen_table, args.exact_root)
    technical = base.frozen_engine.load_technical_core_features(
        args.target_feature_dictionary, frozen_table
    )
    heads: list[base.frozen_engine.HeadPartition] = []
    for slug in base.PANEL12:
        row = indexed.loc[slug]
        reference_rows, reference_controls, _ = base.frozen_test_reference(args, slug)
        data = base.load_test_only_data(
            base.exact_cache_path(args.exact_root, slug),
            slug,
            technical[slug],
            reference_rows,
            reference_controls,
        )
        heads.append(
            base.test_only_partition(
                row,
                data,
                (args.fold + 1) % int(phase_cache.manifest["n_folds"]),
            )
        )
    preprocessing_path = args.output_root / "preprocessing.json"
    payload = json.loads(preprocessing_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("Low-label preprocessing schema differs")
    x_state = base.frozen_engine.GlobalXPreprocessing.from_json(payload["x"])
    for head in heads:
        head.y_preprocessing = base.frozen_engine.HeadYPreprocessing.from_json(
            payload["heads"][head.slug]
        )
    comparability = {
        head.slug: base.frozen_engine.assert_specialist_comparability(
            args, head, args.split, args.fold
        )
        for head in heads
    }
    staging = base.build_phase_only_gpu_staging(args, phase_cache, x_state)
    return (
        phase_cache,
        heads,
        x_state,
        comparability,
        staging,
        base.file_sha256(preprocessing_path),
    )


def aggregate_primary_metrics(args: argparse.Namespace) -> None:
    path = args.output_root / "evaluation_states_summary.csv"
    table = pd.read_csv(path)
    primary = table.loc[table["evaluation_state"] == "single"].copy()
    if len(primary) != len(base.PANEL12):
        raise RuntimeError(f"Expected 12 target reporter evaluations, found {len(primary)}")
    preferred = (
        "cell_macro_feature_pearson",
        "cell_standardized_gain_vs_train_mean",
        "gene_macro_feature_pearson",
        "gene_gain_vs_train_mean",
        "gene_mean_profile_cosine",
        "gene_response_magnitude_spearman",
    )
    numeric = [
        name
        for name in preferred
        if name in primary.columns and pd.api.types.is_numeric_dtype(primary[name])
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "model": args.model,
        "method_id": args.method,
        "label_fraction": args.label_fraction,
        "split": args.split,
        "fold": args.fold,
        "n_target_reporters": len(primary),
        "primary_state": "single",
        "macro_metrics": {
            name: float(primary[name].mean()) for name in numeric
        },
        "reporter_metrics": str(path.resolve()),
        "reporter_metrics_sha256": base.file_sha256(path),
        "outer_test_used_for_selection": False,
    }
    atomic_json(args.output_root / "low_label_result.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def evaluation_only(args: argparse.Namespace) -> None:
    recipe_path = args.output_root / "evaluation_recipe.json"
    if (
        not recipe_path.is_file()
        or base.file_sha256(recipe_path) != args.evaluation_recipe_sha256
    ):
        raise RuntimeError("Frozen low-label evaluation recipe hash mismatch")
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    training_path = args.output_root / "training_complete.json"
    if base.file_sha256(training_path) != recipe["training_complete_sha256"]:
        raise RuntimeError("Low-label training marker changed after selection")
    complete = json.loads(training_path.read_text(encoding="utf-8"))
    checkpoint = Path(recipe["single_checkpoint"]["path"])
    if base.file_sha256(checkpoint) != recipe["single_checkpoint"]["sha256"]:
        raise RuntimeError("Validation-selected low-label checkpoint changed")
    (
        phase_cache,
        heads,
        x_state,
        comparability,
        staging,
        preprocessing_sha,
    ) = prepare_evaluation_data(args)
    base.evaluate_states(
        args,
        complete,
        {"single": checkpoint},
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
            "status": "complete",
            "evaluation_recipe_sha256": args.evaluation_recipe_sha256,
            "outer_test_y_loaded_after_checkpoint_selection": True,
            "test_may_select_checkpoint": False,
            "n_target_reporters": len(heads),
        },
    )
    aggregate_primary_metrics(args)


def spawn_evaluation(
    args: argparse.Namespace,
    complete: Mapping[str, Any],
    state_paths: Mapping[str, Path],
) -> None:
    checkpoint = state_paths["single"]
    recipe = {
        "schema_version": SCHEMA_VERSION,
        "role": "post_selection_outer_test_evaluation",
        "training_complete_sha256": base.file_sha256(
            args.output_root / "training_complete.json"
        ),
        "single_checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": base.file_sha256(checkpoint),
        },
        "selected_optimizer_update": complete["checkpoint_selection"]["optimizer_update"],
        "test_may_select_checkpoint": False,
        "training_or_checkpoint_mutation_allowed": False,
    }
    recipe_path = args.output_root / "evaluation_recipe.json"
    atomic_json(recipe_path, recipe)
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--label-fraction",
        str(args.label_fraction),
        "--fold",
        str(args.fold),
        "--output-root",
        str(args.output_root),
        "--seed",
        str(args.seed),
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
        "--endpoint-semantics",
        str(args.endpoint_semantics),
        "--semantic-vocabulary",
        str(args.semantic_vocabulary),
        "--semantic-manifest",
        str(args.semantic_manifest),
        "--evaluation-only",
        "--evaluation-recipe-sha256",
        base.file_sha256(recipe_path),
    ]
    if args.save_predictions:
        command.append("--save-predictions")
    if args.smoke:
        command.append("--smoke")
    if not args.amp:
        command.append("--no-amp")
    if not args.tf32:
        command.append("--no-tf32")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def resume_post_selection_evaluation(args: argparse.Namespace) -> bool:
    """Resume only the sealed outer-test step after a completed training run.

    A low-label job has two deliberately separate processes: training selects a
    checkpoint without opening outer-test labels, then a fresh process performs
    the outer-test evaluation.  If the second process is interrupted, repeating
    the parent command must *not* silently retrain or overwrite the selected
    checkpoint.  This helper validates the frozen identity/recipe pair and
    resumes precisely that evaluation stage.

    Returns ``True`` only when a valid training artifact was found and the
    evaluation has been completed or resumed.  It is intentionally strict:
    artifacts for another fold, model, fraction or seed are rejected rather
    than being reused.
    """

    training_path = args.output_root / "training_complete.json"
    recipe_path = args.output_root / "evaluation_recipe.json"
    identity_path = args.output_root / "training_identity.json"
    if not (training_path.is_file() and recipe_path.is_file() and identity_path.is_file()):
        return False

    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    binding = identity.get("campaign_binding", {})
    expected = {
        "model": args.model,
        "fold": args.fold,
        "split": args.split,
        "seed": args.seed,
    }
    for key, value in expected.items():
        if binding.get(key) != value:
            raise RuntimeError(
                "Refusing to resume low-label evaluation from a mismatched "
                f"training artifact: {key}={binding.get(key)!r}, expected {value!r}"
            )
    observed_fraction = float(binding.get("label_fraction", float("nan")))
    if not np.isclose(observed_fraction, float(args.label_fraction), rtol=0.0, atol=1e-12):
        raise RuntimeError(
            "Refusing to resume low-label evaluation from a mismatched "
            f"training artifact: label_fraction={observed_fraction!r}, "
            f"expected {args.label_fraction!r}"
        )

    args.evaluation_recipe_sha256 = base.file_sha256(recipe_path)
    print(
        json.dumps(
            {
                "status": "RESUME_POST_SELECTION_EVALUATION",
                "output_root": str(args.output_root),
                "evaluation_recipe_sha256": args.evaluation_recipe_sha256,
                "outer_test_used_for_selection": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    evaluation_only(args)
    complete = json.loads(training_path.read_text(encoding="utf-8"))
    method_config = base.strict_method_config(json.dumps(MODEL_CONFIGS[args.model]))
    base.write_training_result(args, method_config, complete, test_evaluated=True)
    return True


def main() -> None:
    args = parser().parse_args()
    normalize_args(args)
    if args.model == "mlp":
        base.METHOD_MODEL_TYPES["tabm_specialists"] = "independent_mlp"
    install_semantic_permutation_control(args)
    method_config = base.strict_method_config(json.dumps(MODEL_CONFIGS[args.model]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.evaluation_only:
        base.configure_cuda(args.seed, args.tf32)
        evaluation_only(args)
        return
    if (args.output_root / "low_label_result.json").is_file() and args.resume:
        print(
            json.dumps(
                {
                    "status": "already_complete",
                    "result": str(args.output_root / "low_label_result.json"),
                },
                indent=2,
            ),
            flush=True,
        )
        return

    gpu = base.configure_cuda(args.seed, args.tf32)
    if args.resume and resume_post_selection_evaluation(args):
        return
    atomic_json(args.output_root / "gpu_preflight.json", gpu)
    started = time.monotonic()
    phase_cache, heads, x_state, split_contract, preprocessing_sha = (
        prepare_training_data(args)
    )
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
            "sha256": base.json_sha256(head_schema),
        },
    )
    staging = base.frozen_engine.build_gpu_training_staging(
        args, phase_cache, x_state, heads, "cuda"
    )
    complete, state_paths = base.train_model(
        args,
        method_config,
        phase_cache,
        heads,
        x_state,
        staging,
        split_contract,
    )
    complete["low_label_protocol"] = copy.deepcopy(args.campaign_binding)
    complete["scientific_model"] = (
        "independent_mlp_specialists" if args.model == "mlp" else args.model
    )
    complete["checkpoint_objective_reporters"] = list(base.PANEL12)
    complete["low_label_preprocessing_sha256"] = preprocessing_sha
    complete["wall_seconds_through_training"] = time.monotonic() - started
    atomic_json(args.output_root / "training_complete.json", complete)
    base.write_training_result(args, method_config, complete, test_evaluated=False)
    if args.smoke:
        atomic_json(
            args.output_root / "real_gpu_smoke_complete.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "pass",
                "model": args.model,
                "label_fraction": args.label_fraction,
                "n_reporters": len(heads),
                "outer_test_loaded": False,
                "training_complete_sha256": base.file_sha256(
                    args.output_root / "training_complete.json"
                ),
            },
        )
        print("REAL_GPU_SMOKE_PASS", flush=True)
        return

    del staging, heads, phase_cache, x_state
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    spawn_evaluation(args, complete, state_paths)
    base.write_training_result(args, method_config, complete, test_evaluated=True)


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
