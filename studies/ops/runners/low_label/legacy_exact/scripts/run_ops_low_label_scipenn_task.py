#!/usr/bin/env python3
"""Run sciPENN-OPS with Donor40 full and Target12 low-label supervision."""

from __future__ import annotations

import copy
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

import ops_low_label_sampling_lib as sampling
import run_ops_scipenn_full52_formal as sci
from ops_reporter_specialist_lib import atomic_json


RESULT_SCHEMA = "ops-low-label-result-v1"


def _pop(argv: list[str], option: str) -> str | None:
    if option not in argv:
        return None
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise SystemExit(f"Missing value for {option}")
    value = argv[index + 1]
    del argv[index : index + 2]
    return value


def _peek(argv: list[str], option: str) -> str | None:
    if option not in argv:
        return None
    index = argv.index(option)
    if index + 1 >= len(argv):
        raise SystemExit(f"Missing value for {option}")
    return argv[index + 1]


def _configure_argv() -> tuple[float, Path, bool]:
    argv = list(sys.argv)
    method = _pop(argv, "--method") or "scipenn_ops"
    if method != "scipenn_ops":
        raise SystemExit("This runner accepts only scipenn_ops")
    raw_fraction = _pop(argv, "--label-fraction") or os.environ.get(
        "OPS_LOW_LABEL_FRACTION"
    )
    if raw_fraction is None:
        raise SystemExit("Missing --label-fraction")
    fraction = float(raw_fraction)
    sampling_root = Path(
        _pop(argv, "--sampling-manifest-root")
        or os.environ.get("OPS_LOW_LABEL_SAMPLING_ROOT", str(sampling.DEFAULT_ROOT))
    ).expanduser().resolve()
    workers = _pop(argv, "--workers")
    if workers is not None and int(workers) <= 0:
        raise SystemExit("--workers must be positive")
    end_to_end_smoke = "--end-to-end-smoke" in argv
    if end_to_end_smoke:
        argv.remove("--end-to-end-smoke")
    sys.argv = argv
    os.environ["OPS_LOW_LABEL_FRACTION"] = sampling.fraction_token(fraction)
    os.environ["OPS_LOW_LABEL_SAMPLING_ROOT"] = str(sampling_root)
    return fraction, sampling_root, end_to_end_smoke


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "usage: run_ops_low_label_scipenn_task.py --method scipenn_ops "
            "--label-fraction FRACTION --split gene_holdout_main "
            "--fold {0,1,2,3,4} --campaign-config JSON --output-root PATH "
            "--sampling-manifest-root PATH [delegated OPS asset options]\n\n"
            + (__doc__ or "")
        )
        return
    fraction, sampling_root, end_to_end_smoke = _configure_argv()
    target_table = Path(
        _peek(sys.argv, "--target-table")
        or os.environ.get(
            "OPS_LOW_LABEL_TARGET_TABLE", sci.common_runner.DEFAULT_TARGET_TABLE
        )
    ).expanduser().resolve()
    table = pd.read_csv(target_table)
    target12 = sampling.canonical_target12(table)

    original_read = sci._read_config
    original_normalize = sci._normalize_args
    original_binding = sci._binding
    original_prepare = sci.common_runner.prepare_data
    original_prepare_eval = sci.common_runner.prepare_evaluation_data
    original_eval_only = sci._evaluation_only
    original_write = sci._write_formal_result

    def read_config(path: Path) -> dict[str, Any]:
        payload = original_read(path)
        if end_to_end_smoke:
            payload = copy.deepcopy(payload)
            # Two trained states are the minimum needed by the immutable
            # checkpoint-averaging contract exercised after training.
            payload["fixed_trial"]["training"]["joint_epochs"] = 2
        return payload

    def normalize(args: Any, payload: Mapping[str, Any]) -> None:
        # Reuse every frozen execution check while allowing the common
        # low-label seed that also defines the central sampling manifest.
        requested_seed = int(args.seed)
        args.seed = int(payload["seed"])
        original_normalize(args, payload)
        args.seed = requested_seed
        if end_to_end_smoke:
            args.reporter_rounds_per_epoch_override = 1

    def binding(args: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
        result = original_binding(args, payload)
        result.update(
            {
                "schema_version": RESULT_SCHEMA,
                "seed": args.seed,
                "label_fraction": fraction,
                "target_reporters": list(target12),
                "target_reporter_policy": "panel12_low_label",
                "donor_reporter_policy": "donor40_full",
                "experiment_arm": "atlas_transfer",
                "central_sampling_manifest_required": True,
            }
        )
        return result

    def prepare(args: Any) -> Any:
        result = original_prepare(args)
        (
            phase_cache,
            reporter_table,
            heads,
            x_state,
            comparability,
            _split_contract,
            _preprocessing_sha,
        ) = result
        manifest_file = sampling.manifest_path(
            sampling_root,
            split=args.split,
            fold=args.fold,
            fraction=fraction,
        )
        sampling.apply_manifest_to_heads(
            heads=heads,
            manifest_file=manifest_file,
            split=args.split,
            fold=args.fold,
            fraction=fraction,
            seed=args.seed,
            target_reporters=target12,
        )
        reference_path = (
            args.preprocessing_reference_root
            / args.split
            / f"fold_{args.fold}"
            / "train"
            / "preprocessing.json"
        )
        preprocessing = sci.common_runner.preprocessing_manifest(
            reference_path, x_state, heads
        )
        preprocessing.update(
            {
                "central_sampling_manifest": str(manifest_file),
                "central_sampling_manifest_sha256": sampling.file_sha256(
                    manifest_file
                ),
                "target_reporter_policy": "panel12_low_label",
                "donor_reporter_policy": "donor40_full",
            }
        )
        preprocessing_path = args.output_root / "preprocessing.json"
        atomic_json(preprocessing_path, preprocessing)
        split_contract = sci.common_runner.build_split_contract(args, heads)
        return (
            phase_cache,
            reporter_table,
            heads,
            x_state,
            comparability,
            split_contract,
            sci.common_runner.file_sha256(preprocessing_path),
        )

    def prepare_eval(args: Any) -> Any:
        frozen_table = pd.read_csv(args.target_table)
        indexed = frozen_table.set_index("reporter_slug", drop=False)
        phase_cache = sci.common_runner.PhaseCache.open(args.phase_cache)
        sci.common_runner.validate_frozen_assets(
            phase_cache, frozen_table, args.exact_root
        )
        technical = sci.frozen_engine.load_technical_core_features(
            args.target_feature_dictionary, frozen_table
        )
        heads = []
        for slug in target12:
            reference_rows, reference_controls, _ = (
                sci.common_runner.frozen_test_reference(args, slug)
            )
            data = sci.common_runner.load_test_only_data(
                sci.common_runner.exact_cache_path(args.exact_root, slug),
                slug,
                technical[slug],
                reference_rows,
                reference_controls,
            )
            heads.append(
                sci.common_runner.test_only_partition(
                    indexed.loc[slug],
                    data,
                    (args.fold + 1) % int(phase_cache.manifest["n_folds"]),
                )
            )
        preprocessing_path = args.output_root / "preprocessing.json"
        payload = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        x_state = sci.frozen_engine.GlobalXPreprocessing.from_json(payload["x"])
        for head in heads:
            head.y_preprocessing = sci.frozen_engine.HeadYPreprocessing.from_json(
                payload["heads"][head.slug]
            )
        comparability = {
            head.slug: sci.frozen_engine.assert_specialist_comparability(
                args, head, args.split, args.fold
            )
            for head in heads
        }
        staging = sci.common_runner.build_phase_only_gpu_staging(
            args, phase_cache, x_state
        )
        return (
            phase_cache,
            heads,
            x_state,
            comparability,
            staging,
            sci.common_runner.file_sha256(preprocessing_path),
        )

    def evaluation_only(args: Any) -> None:
        recipe_path = args.output_root / "evaluation_recipe.json"
        sci._require(
            bool(args.evaluation_recipe_sha256)
            and recipe_path.is_file()
            and sci._file_sha256(recipe_path) == args.evaluation_recipe_sha256,
            "Low-label sciPENN evaluation recipe changed",
        )
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
        for key, expected in (
            ("schema_version", sci.EVALUATION_SCHEMA_VERSION),
            ("method", sci.METHOD_ID),
            ("split", args.split),
            ("fold", args.fold),
            ("seed", args.seed),
            ("test_may_select_state", False),
            ("training_or_checkpoint_mutation_allowed", False),
            ("outer_test_y_loaded_by_training_process", False),
            ("max_test_observations_per_head", None),
            ("specialist_reference_required", True),
        ):
            sci._require(
                recipe.get(key) == expected,
                f"Low-label sciPENN evaluation recipe violates {key}",
            )
        sci._require(
            recipe.get("campaign_config_sha256")
            == sci._file_sha256(args.campaign_config),
            "Low-label sciPENN evaluation campaign changed",
        )
        training_path = args.output_root / "training_complete.json"
        sci._require(
            sci._file_sha256(training_path) == recipe["training_complete_sha256"],
            "Low-label sciPENN training marker changed",
        )
        complete = sci.common_runner.read_json_without_duplicate_keys(
            training_path
        )
        sci._require(
            recipe.get("checkpoint_selection_sha256")
            == sci.harness.json_sha256(complete["checkpoint_selection"]),
            "Low-label sciPENN validation-selected checkpoint changed",
        )
        sci._require(
            set(recipe.get("states", {}))
            == {"single", "ema", "checkpoint_average"},
            "Low-label sciPENN evaluation state set changed",
        )
        state_paths = {
            name: Path(row["path"]) for name, row in recipe["states"].items()
        }
        for name, path in state_paths.items():
            sci._require(
                path.is_file()
                and sci._file_sha256(path) == recipe["states"][name]["sha256"],
                f"Frozen sciPENN state changed: {name}",
            )
        (
            phase_cache,
            heads,
            x_state,
            comparability,
            staging,
            preprocessing_sha,
        ) = prepare_eval(args)
        sci._require(
            tuple(head.slug for head in heads) == target12,
            "Low-label sciPENN evaluation reporter order changed",
        )
        sci._evaluate_states(
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
        summary = args.output_root / "evaluation_states_summary.csv"
        atomic_json(
            args.output_root / "evaluation_complete.json",
            {
                "schema_version": sci.EVALUATION_SCHEMA_VERSION,
                "status": "complete",
                "split": args.split,
                "fold": args.fold,
                "n_reporters": 12,
                "n_evaluation_states": 3,
                "outer_test_y_loaded_after_checkpoint_selection": True,
                "test_may_select_state": False,
                "evaluation_recipe_sha256": args.evaluation_recipe_sha256,
                "evaluation_states_summary_sha256": sci._file_sha256(summary),
            },
        )

    def write(args: Any) -> None:
        training_path = args.output_root / "training_complete.json"
        evaluation_path = args.output_root / "evaluation_complete.json"
        complete = sci.common_runner.read_json_without_duplicate_keys(training_path)
        evaluation = sci.common_runner.read_json_without_duplicate_keys(
            evaluation_path
        )
        for key, expected in (
            ("schema_version", sci.EVALUATION_SCHEMA_VERSION),
            ("status", "complete"),
            ("split", args.split),
            ("fold", args.fold),
            ("n_reporters", 12),
            ("n_evaluation_states", 3),
            ("outer_test_y_loaded_after_checkpoint_selection", True),
            ("test_may_select_state", False),
        ):
            sci._require(
                evaluation.get(key) == expected,
                f"sciPENN low-label completed evaluation violates {key}",
            )
        summary_path = args.output_root / "evaluation_states_summary.csv"
        sci._require(
            evaluation.get("evaluation_states_summary_sha256")
            == sci._file_sha256(summary_path),
            "sciPENN low-label evaluation summary changed",
        )
        rows = pd.read_csv(summary_path)
        primary = rows.loc[rows["evaluation_state"] == "single"].copy()
        sci._require(
            len(primary) == 12 and set(primary["reporter_slug"]) == set(target12),
            "sciPENN low-label summary is not Target12",
        )
        atomic_json(
            args.output_root / "training_result.json",
            {
                "schema_version": sci.SCHEMA_VERSION,
                "status": "complete",
                "mode": "formal",
                "method_id": sci.METHOD_ID,
                "split": args.split,
                "fold": args.fold,
                "n_reporters": 52,
                "n_endpoints": int(sci.FULL52_ENDPOINTS),
                "fixed_recipe_no_hpo": True,
                "reporter_normalized_validation_score": complete[
                    "checkpoint_selection"
                ]["score"],
                "selected_optimizer_update": complete["checkpoint_selection"][
                    "optimizer_update"
                ],
                "selection_used_test": False,
                "outer_test_evaluated": True,
                "training_complete_sha256": sci._file_sha256(training_path),
                "evaluation_complete_sha256": sci._file_sha256(evaluation_path),
            },
        )
        manifest_file = sampling.manifest_path(
            sampling_root,
            split=args.split,
            fold=args.fold,
            fraction=fraction,
        )
        preferred = (
            "cell_macro_feature_pearson",
            "cell_standardized_gain_vs_train_mean",
            "gene_macro_feature_pearson",
            "gene_gain_vs_train_mean",
            "gene_mean_profile_cosine",
            "gene_response_magnitude_spearman",
        )
        missing_metrics = [name for name in preferred if name not in primary]
        sci._require(
            not missing_metrics,
            f"sciPENN low-label summary lacks metrics: {missing_metrics}",
        )
        macro_metrics = {
            name: float(primary[name].mean()) for name in preferred
        }
        sci._require(
            all(math.isfinite(value) for value in macro_metrics.values()),
            "sciPENN low-label macro metrics are nonfinite",
        )
        atomic_json(
            args.output_root / "low_label_result.json",
            {
                "schema_version": RESULT_SCHEMA,
                "status": "complete",
                "completed": True,
                "mode": "formal",
                "method_id": "scipenn_ops",
                "experiment_arm": "atlas_transfer",
                "label_fraction": fraction,
                "split": args.split,
                "fold": args.fold,
                "seed": args.seed,
                "n_target_reporters": 12,
                "target_reporter_policy": "panel12_low_label",
                "donor_reporter_policy": "donor40_full",
                "cohort_matches_frozen_reference": True,
                "sampling_manifest": str(manifest_file),
                "sampling_manifest_sha256": sampling.file_sha256(manifest_file),
                "execution_device": "cuda",
                "cpu_fallback_allowed": False,
                "outer_test_used_for_selection": False,
                "outer_test_y_loaded_after_checkpoint_selection": True,
                "test_evaluated": True,
                "macro_metrics": macro_metrics,
                "reporter_metrics": str(summary_path.resolve()),
                "reporter_metrics_sha256": sci._file_sha256(summary_path),
            },
        )

    sci._read_config = read_config
    sci._normalize_args = normalize
    sci._binding = binding
    sci.common_runner.prepare_data = prepare
    sci.common_runner.prepare_evaluation_data = prepare_eval
    sci._evaluation_only = evaluation_only
    sci._write_formal_result = write
    sci.__file__ = str(Path(__file__).resolve())
    try:
        sci.main()
    finally:
        sci._read_config = original_read
        sci._normalize_args = original_normalize
        sci._binding = original_binding
        sci.common_runner.prepare_data = original_prepare
        sci.common_runner.prepare_evaluation_data = original_prepare_eval
        sci._evaluation_only = original_eval_only
        sci._write_formal_result = original_write
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
