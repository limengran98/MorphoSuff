#!/usr/bin/env python3
"""Run one generic neural OPS low-label benchmark task.

This isolated wrapper reuses the exercised low-label data/evaluation pipeline.
Shared ResMLP, MMoE and
MultiTab train on Donor40 full labels plus Target12 low labels.  TabM remains a
true target-specific baseline and sees only the same Target12 low-label rows.
"""

from __future__ import annotations

import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

import ops_low_label_sampling_lib as sampling
import run_ops_low_label_transfer_fold0 as low


RESULT_SCHEMA = "ops-low-label-result-v1"
METHOD_TO_MODEL = {
    "resmlp_52head": "resmlp",
    "tabm_specialists": "tabm",
    "mmoe_52head": "mmoe",
    "multitab_pilot": "multitab",
}
MODEL_TO_METHOD = {value: key for key, value in METHOD_TO_MODEL.items()}
CONFIGS: dict[str, dict[str, Any]] = {
    "resmlp": {
        "model": {
            "dropout": 0.05,
            "expansion_width": 1024,
            "head_width": 128,
            "residual_blocks": 4,
            "shared_width": 512,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
        },
        "training": {
            "batch_heads": 4,
            "observations_per_head": 1024,
            "epochs": 100,
            "warmup_epochs": 2,
            "minimum_learning_rate": 5.0e-6,
            "ema_decay": 0.999,
            "gradient_clip_norm": 1.0,
        },
    },
    "tabm": {
        "model": {
            "tabm_kwargs": {
                "arch_type": "tabm",
                "d_block": 256,
                "dropout": 0.05,
                "k": 16,
                "n_blocks": 3,
            }
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 1.0e-3,
            "weight_decay": 3.0e-4,
        },
        "training": {
            "batch_heads": 4,
            "observations_per_head": 1024,
            "epochs": 100,
            "warmup_epochs": 2,
            "minimum_learning_rate": 5.0e-6,
            "ema_decay": 0.999,
            "gradient_clip_norm": 1.0,
        },
    },
    "mmoe": {
        "model": {
            "dropout": 0.05,
            "expert_hidden_dims": [512, 256],
            "num_experts": 8,
            "tower_hidden_dims": [128],
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
        },
        "training": {
            "batch_heads": 4,
            "observations_per_head": 1024,
            "epochs": 100,
            "warmup_epochs": 2,
            "minimum_learning_rate": 5.0e-6,
            "ema_decay": 0.999,
            "gradient_clip_norm": 1.0,
        },
    },
    "multitab": {
        "model": {
            "dropout": 0.05,
            "feedforward_dim": 128,
            "head_hidden_dims": [64],
            "num_blocks": 2,
            "num_heads": 4,
            "token_dim": 32,
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 3.0e-4,
            "weight_decay": 1.0e-4,
        },
        "training": {
            "batch_heads": 4,
            "observations_per_head": 1024,
            "epochs": 100,
            "warmup_epochs": 2,
            "minimum_learning_rate": 5.0e-6,
            "ema_decay": 0.999,
            "gradient_clip_norm": 1.0,
        },
    },
}


def _pop_option(argv: list[str], name: str, *, required: bool = False) -> str | None:
    if name not in argv:
        if required:
            raise SystemExit(f"Missing required option {name}")
        return None
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise SystemExit(f"Missing value for {name}")
    value = argv[index + 1]
    del argv[index : index + 2]
    return value


def _peek_option(argv: list[str], name: str) -> str | None:
    """Read an ordinary delegated option without removing it from argv."""
    if name not in argv:
        return None
    index = argv.index(name)
    if index + 1 >= len(argv):
        raise SystemExit(f"Missing value for {name}")
    return argv[index + 1]


def _configure_argv() -> tuple[str, Path, bool]:
    argv = list(sys.argv)
    method = _pop_option(argv, "--method") or os.environ.get(
        "OPS_LOW_LABEL_GENERIC_METHOD"
    )
    if method is None:
        raise SystemExit("Missing required option --method")
    if method not in METHOD_TO_MODEL:
        raise SystemExit(f"Unsupported generic low-label method: {method}")
    sampling_root = Path(
        _pop_option(argv, "--sampling-manifest-root")
        or os.environ.get("OPS_LOW_LABEL_SAMPLING_MANIFEST_ROOT")
        or sampling.DEFAULT_ROOT
    ).expanduser().resolve()
    staging = _pop_option(argv, "--gpu-staging")
    if staging not in {None, "required"}:
        raise SystemExit("GPU staging must be required")
    end_to_end_smoke = "--end-to-end-smoke" in argv
    if end_to_end_smoke:
        argv.remove("--end-to-end-smoke")
    if "--model" not in argv:
        method_index = 1
        argv[method_index:method_index] = ["--model", METHOD_TO_MODEL[method]]
    sys.argv = argv
    return method, sampling_root, end_to_end_smoke


def main() -> None:
    if "--help" in sys.argv or "-h" in sys.argv:
        print(
            "usage: run_ops_low_label_generic_task.py --method "
            "{resmlp_52head,tabm_specialists,mmoe_52head,multitab_pilot} "
            "--label-fraction FRACTION --fold {0,1,2,3,4} --output-root PATH "
            "--sampling-manifest-root PATH [delegated OPS asset options]\n\n"
            + (__doc__ or "")
        )
        return
    method, sampling_root, end_to_end_smoke = _configure_argv()
    os.environ["OPS_LOW_LABEL_GENERIC_METHOD"] = method
    # ``spawn_evaluation`` deliberately launches a fresh Python process after
    # checkpoint selection.  Preserve the exact central sampling asset in that
    # child even though the reused command builder does not know this wrapper's
    # ``--sampling-manifest-root`` option.
    os.environ["OPS_LOW_LABEL_SAMPLING_MANIFEST_ROOT"] = str(sampling_root)
    model_name = METHOD_TO_MODEL[method]
    low.MODEL_METHOD.update(
        {
            "resmlp": "resmlp_52head",
            "tabm": "tabm_specialists",
            "mmoe": "mmoe_52head",
            "multitab": "multitab_pilot",
        }
    )
    # The reused low-label entry point calls ``train_model`` directly and
    # therefore bypasses the common runner's normal dependency gate.  Install
    # the frozen, hash-checked TabM wheel in both the training process and the
    # fresh post-selection evaluation process before model construction.
    low.base.ensure_tabm_dependency(
        method, low.base.DEFAULT_TABM_DEPENDENCY_WHEEL
    )
    configs = copy.deepcopy(CONFIGS)
    if method == "tabm_specialists":
        tabm_source = Path(
            os.environ.get(
                "OPS_TABM_SOURCE",
                low.base.candidate_models.DEFAULT_FROZEN_TABM_SOURCE,
            )
        ).expanduser().resolve()
        configs["tabm"]["model"]["provenance"] = {
            "mode": "source",
            "source_path": str(tabm_source),
            "expected_sha256": low.base.candidate_models.DEFAULT_FROZEN_TABM_SHA256,
            "expected_version": "0.0.3",
        }
    low.MODEL_CONFIGS.update(configs)
    # Keep the formal, hash-checked 100-epoch schedule even in the end-to-end
    # smoke.  The common harness intentionally rejects a different epoch
    # count.  Smoke runtime is reduced below by freezing one reporter round
    # per epoch, which still exercises optimizer/scheduler/checkpoint
    # selection and the fresh post-selection evaluation process.

    target_table = Path(
        _peek_option(sys.argv, "--target-table")
        or os.environ.get("OPS_LOW_LABEL_TARGET_TABLE", low.base.DEFAULT_TARGET_TABLE)
    ).expanduser().resolve()
    table = pd.read_csv(target_table)
    target12 = sampling.canonical_target12(table)
    # Keep one canonical Target12 order across every wrapper and evaluator.
    low.base.PANEL12 = target12

    original_normalize = low.normalize_args
    original_prepare = low.prepare_training_data
    original_rounds = low.base.frozen_rounds_per_epoch

    def normalize(args: Any) -> None:
        original_normalize(args)
        args.sampling_manifest_root = sampling_root
        manifest_file = sampling.manifest_path(
            sampling_root,
            split=args.split,
            fold=args.fold,
            fraction=args.label_fraction,
        )
        if not manifest_file.is_file():
            raise FileNotFoundError(
                f"Central low-label sampling manifest is missing: {manifest_file}"
            )
        args.campaign_binding.update(
            {
                "schema_version": RESULT_SCHEMA,
                "method_id": method,
                "experiment_arm": (
                    "target_only" if method == "tabm_specialists" else "atlas_transfer"
                ),
                "target_reporter_policy": "panel12_low_label",
                "donor_reporter_policy": (
                    "absent" if method == "tabm_specialists" else "donor40_full"
                ),
                "central_sampling_manifest_required": True,
                "central_sampling_manifest": str(manifest_file),
                "central_sampling_manifest_sha256": sampling.file_sha256(
                    manifest_file
                ),
                "generic_wrapper_sha256": low.base.file_sha256(
                    Path(__file__).resolve()
                ),
                "sampling_library_sha256": low.base.file_sha256(
                    Path(sampling.__file__).resolve()
                ),
            }
        )

    def prepare(args: Any) -> Any:
        # The live low-label runner uses its MLP branch to request only Target12.
        # Switching only this local selector leaves args.method/model/config as
        # TabM and therefore does not turn TabM into an MLP.
        if method == "tabm_specialists":
            saved = args.model
            args.model = "mlp"
            try:
                result = original_prepare(args)
            finally:
                args.model = saved
        else:
            result = original_prepare(args)
        phase_cache, heads, x_state, split_contract, _ = result
        manifest_file = sampling.manifest_path(
            sampling_root,
            split=args.split,
            fold=args.fold,
            fraction=args.label_fraction,
        )
        payload = sampling.apply_manifest_to_heads(
            heads=heads,
            manifest_file=manifest_file,
            split=args.split,
            fold=args.fold,
            fraction=args.label_fraction,
            seed=args.seed,
            target_reporters=target12,
        )
        preprocessing_path = args.output_root / "preprocessing.json"
        preprocessing = json.loads(preprocessing_path.read_text(encoding="utf-8"))
        preprocessing.update(
            {
                "central_sampling_manifest": str(manifest_file),
                "central_sampling_manifest_sha256": sampling.file_sha256(
                    manifest_file
                ),
                "central_sampling_schema": payload["schema_version"],
                "model_scope": (
                    "target12_only"
                    if method == "tabm_specialists"
                    else "donor40_full_plus_target12_low_label"
                ),
            }
        )
        low.atomic_json(preprocessing_path, preprocessing)
        split_contract = low.base.build_split_contract(args, heads)
        return (
            phase_cache,
            heads,
            x_state,
            split_contract,
            low.base.file_sha256(preprocessing_path),
        )

    def aggregate(args: Any) -> None:
        path = args.output_root / "evaluation_states_summary.csv"
        table_metrics = pd.read_csv(path)
        primary = table_metrics.loc[
            table_metrics["evaluation_state"] == "single"
        ].copy()
        if len(primary) != 12:
            raise RuntimeError(
                f"Expected 12 Target12 evaluations, found {len(primary)}"
            )
        observed_reporters = tuple(primary["reporter_slug"].astype(str))
        if len(set(observed_reporters)) != 12 or set(observed_reporters) != set(
            target12
        ):
            raise RuntimeError(
                "Final evaluation does not contain each frozen Target12 "
                "reporter exactly once"
            )
        preferred = (
            "cell_macro_feature_pearson",
            "cell_standardized_gain_vs_train_mean",
            "gene_macro_feature_pearson",
            "gene_gain_vs_train_mean",
            "gene_mean_profile_cosine",
            "gene_response_magnitude_spearman",
        )
        missing_metrics = set(preferred) - set(primary.columns)
        if missing_metrics:
            raise RuntimeError(
                f"Final evaluation lacks required metrics: {sorted(missing_metrics)}"
            )
        metrics: dict[str, float] = {}
        for name in preferred:
            values = pd.to_numeric(primary[name], errors="coerce")
            if values.isna().any() or not values.map(math.isfinite).all():
                raise RuntimeError(f"Final evaluation contains non-finite {name}")
            value = float(values.mean())
            if not math.isfinite(value):
                raise RuntimeError(f"Final macro metric is non-finite: {name}")
            metrics[name] = value
        manifest_file = sampling.manifest_path(
            sampling_root,
            split=args.split,
            fold=args.fold,
            fraction=args.label_fraction,
        )
        low.atomic_json(
            args.output_root / "low_label_result.json",
            {
                "schema_version": RESULT_SCHEMA,
                "status": "complete",
                "completed": True,
                "mode": "formal",
                "method_id": method,
                "experiment_arm": (
                    "target_only" if method == "tabm_specialists" else "atlas_transfer"
                ),
                "label_fraction": args.label_fraction,
                "split": args.split,
                "fold": args.fold,
                "seed": args.seed,
                "n_target_reporters": 12,
                "target_reporter_policy": "panel12_low_label",
                "donor_reporter_policy": (
                    "absent" if method == "tabm_specialists" else "donor40_full"
                ),
                "cohort_matches_frozen_reference": True,
                "sampling_manifest": str(manifest_file),
                "sampling_manifest_sha256": sampling.file_sha256(manifest_file),
                "execution_device": "cuda",
                "cpu_fallback_allowed": False,
                "outer_test_used_for_selection": False,
                "outer_test_y_loaded_after_checkpoint_selection": True,
                "test_evaluated": True,
                "macro_metrics": metrics,
                "reporter_metrics": str(path.resolve()),
                "reporter_metrics_sha256": low.base.file_sha256(path),
            },
        )

    low.normalize_args = normalize
    low.prepare_training_data = prepare
    low.aggregate_primary_metrics = aggregate
    if end_to_end_smoke:
        low.base.frozen_rounds_per_epoch = lambda _args, _n: 1
    # The reused evaluator spawns Path(module.__file__).  Point it back to this
    # wrapper so the method registry and central-manifest checks are installed
    # again in the fresh post-selection process.
    low.__file__ = str(Path(__file__).resolve())
    try:
        low.main()
    finally:
        low.base.frozen_rounds_per_epoch = original_rounds


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    main()
