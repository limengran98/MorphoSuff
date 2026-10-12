#!/usr/bin/env python3
"""Validate and freeze eight sciPENN development commands without executing.

The preparation boundary is intentionally strict: this module may inspect
configuration, source hashes, and path existence, but it has no subprocess or
training API.  It writes a serial command plan whose status is always
``READY_NOT_STARTED``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "ops_scipenn_panel12_development_v1.json"
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "run_ops_scipenn_development.py"
SCHEMA_VERSION = "ops-scipenn-development-campaign-v1"
COMMAND_SCHEMA_VERSION = "ops-scipenn-development-commands-v1"
METHODS = ("scipenn_ops",)
DEVELOPMENT_FOLDS = (
    ("gene_holdout_main", 0),
    ("field_holdout_sanity", 0),
)
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
FROZEN_ASSET_PATHS = {
    "phase_cache": (PROJECT_ROOT / "data/processed/ops_phase172_indexed").resolve(),
    "exact_root": (
        PROJECT_ROOT / "data/processed/ops_full_reporter_exact/reporters"
    ).resolve(),
    "target_table": (
        PROJECT_ROOT / "results/ops_phase0_asset_audit/reporter_targets.csv"
    ).resolve(),
    "target_feature_dictionary": (
        PROJECT_ROOT / "results/ops_phase0_asset_audit/target_feature_dictionary.csv"
    ).resolve(),
    "specialist_reference_root": (
        PROJECT_ROOT / "results/ops_reporter_specialists_v1"
    ).resolve(),
    "preprocessing_reference_root": (
        PROJECT_ROOT / "results/ops_reporter_masked_multitask_resmlp_v1"
    ).resolve(),
}


class CampaignError(RuntimeError):
    """The sciPENN campaign or command plan violates a frozen invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CampaignError("Campaign is not strict JSON") from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise CampaignError(f"Non-finite JSON constant is forbidden: {value}")


def read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_nonfinite,
            )
    except CampaignError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"Unable to read campaign {path}: {error}") from error
    _require(isinstance(payload, dict), "Campaign must be a JSON object")
    canonical_json(payload)
    return payload


def _exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(
        set(value) == keys,
        f"{label} keys changed: expected={sorted(keys)}, observed={sorted(value)}",
    )


def _positive_int(value: Any, label: str) -> int:
    _require(not isinstance(value, bool), f"{label} must be an integer")
    result = int(value)
    _require(result > 0 and result == value, f"{label} must be a positive integer")
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    result = float(value)
    _require(math.isfinite(result) and result >= 0.0, f"{label} is invalid")
    return result


def _validate_source_hashes(contract: Mapping[str, Any]) -> None:
    source_root = (PROJECT_ROOT / str(contract["upstream_repository"])).resolve()
    _require(source_root.is_dir(), "Frozen sciPENN source directory is missing")
    expected = contract["source_sha256"]
    expected_keys = {
        "src/sciPENN/sciPENN_API.py",
        "src/sciPENN/Network/Model.py",
        "src/sciPENN/Network/Layers.py",
        "src/sciPENN/Network/Losses.py",
        "src/sciPENN/Preprocessing.py",
        "src/sciPENN/Data_Infrastructure/DataLoader.py",
        "src/sciPENN/Data_Infrastructure/Samplers.py",
        "src/sciPENN/Data_Infrastructure/DataLoader_Constructor.py",
    }
    _require(
        isinstance(expected, Mapping) and set(expected) == expected_keys,
        "Source hash key set changed",
    )
    for relative, digest in expected.items():
        path = (source_root / str(relative)).resolve()
        _require(path.is_relative_to(source_root), "Source hash path escapes repository")
        _require(path.is_file(), f"Frozen sciPENN source is missing: {path}")
        _require(
            file_sha256(path) == str(digest),
            f"Frozen sciPENN source hash changed: {relative}",
        )


def validate_config(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "Wrong sciPENN schema")
    _require(
        payload.get("campaign_id") == "ops-scipenn-panel12-development-v1",
        "Wrong sciPENN campaign ID",
    )
    _require(payload.get("frozen") is True, "sciPENN campaign is not frozen")
    _require(tuple(payload.get("method_order", ())) == METHODS, "Method order changed")
    _require(tuple(payload.get("reporter_panel", ())) == PANEL12, "Panel12 changed")
    boundary = payload.get("execution_boundary")
    _require(isinstance(boundary, Mapping), "Execution boundary is missing")
    expected_boundary = {
        "development_only": True,
        "outer_test_label_access_allowed": False,
        "outer_test_identity_metadata_access_allowed": True,
        "outer_test_evaluation_allowed": False,
        "formal_training_authorized": False,
        "automatic_parallel_launch_allowed": False,
        "cuda_required": True,
        "cpu_fallback_allowed": False,
        "runner_entrypoint": "scripts/run_ops_scipenn_development.py",
        "result_filename": "training_result.json",
    }
    _require(dict(boundary) == expected_boundary, "Execution boundary changed")

    comparison = payload.get("comparison_contract")
    _require(isinstance(comparison, Mapping), "Comparison contract is missing")
    _require(comparison.get("trial_count_per_method") == 4, "Expected four trials")
    _require(comparison.get("search_seed") == 20260721, "Search seed changed")
    folds = tuple(
        (str(row.get("split")), int(row.get("fold")))
        for row in comparison.get("development_folds", ())
        if isinstance(row, Mapping)
    )
    _require(folds == DEVELOPMENT_FOLDS, "Development partitions changed")
    expected_selection = {
        "name": "reporter_normalized_validation_score",
        "direction": "minimize",
        "primary_development_job": {
            "split": "gene_holdout_main",
            "fold": 0,
            "role": "hpo_selection_primary",
        },
        "guardrail_development_job": {
            "split": "field_holdout_sanity",
            "fold": 0,
            "role": "reported_guardrail_not_selection",
        },
        "cross_fold_aggregation": "none",
        "tie_break": "lowest_trial_index",
        "validation_only": True,
        "outer_test_may_select_trial": False,
    }
    _require(comparison.get("selection_objective") == expected_selection, "HPO boundary changed")
    _require(
        comparison.get("parameter_policy")
        == {
            "matching_required": False,
            "parameter_count_is_admission_or_selection_input": False,
            "parameter_limits": None,
            "native_capacity_selected_by_validation": True,
        },
        "Parameter policy changed",
    )
    resources = comparison.get("resource_metrics", {})
    _require(resources.get("role") == "reporting_only", "Resources are not reporting-only")
    _require(
        resources.get("used_for_admission_or_hpo_selection") is False,
        "Resource metrics cannot select sciPENN trials",
    )
    _require(
        comparison.get("shared_primary_exposure")
        == {
            "reporters_per_primary_event": 4,
            "observations_per_reporter_per_event": 1024,
            "reporter_rounds_per_epoch": "frozen V1 reporter rounds for the same split/fold",
            "joint_epochs": 100,
            "validation_opportunities": 100,
            "early_stopping_changes_budget": False,
        },
        "Primary exposure contract changed",
    )

    methods = payload.get("methods")
    _require(isinstance(methods, Mapping) and set(methods) == set(METHODS), "Method registry changed")
    method = methods["scipenn_ops"]
    _require(
        method.get("scope")
        == "native_union_of_the_12_reporter_panels_participating_in_development",
        "Panel12 native union scope changed",
    )
    _require(
        "parameter_count_cap" in method.get("forbidden_adaptations", ()),
        "Parameter cap must remain forbidden",
    )
    source = method.get("source_fidelity_contract")
    _require(isinstance(source, Mapping), "Source fidelity contract is missing")
    _require(
        source.get("upstream_commit") == "34afb2008a076e13c40965a76d3dd31d0c331652",
        "sciPENN source commit changed",
    )
    _validate_source_hashes(source)
    anchor = source.get("trial0_source_anchor")
    _require(
        anchor
        == {
            "hidden_dim": 512,
            "dropout": 0.25,
            "quantile_levels": [0.1, 0.25, 0.75, 0.9],
            "loss_reduction": "source_dense_mean",
            "optimizer": "adam",
            "learning_rate": 0.001,
            "precision": "float32_no_tf32",
        },
        "Released-code anchor changed",
    )
    adaptations = source.get("ops_task_adaptations", {})
    _require(
        adaptations.get("panel12_training_union_contains_only_the_12_participating_panels") is True
        and adaptations.get("full52_union_reserved_for_formal_full52_training") is True,
        "Native active-panel union rule changed",
    )

    trials = method.get("trial_configs")
    _require(isinstance(trials, list) and len(trials) == 4, "Expected four sciPENN trials")
    roles = [trial.get("trial_role") for trial in trials]
    _require(
        roles
        == [
            "released_code_source_anchor",
            "source_architecture_observed_mean_stabilization",
            "validation_only_ops_capacity_variant",
            "validation_only_ops_wide_variant",
        ],
        "Trial roles changed",
    )
    expected_model_keys = {"hidden_dim", "dropout", "quantile_levels", "loss_reduction"}
    expected_optimizer_keys = {"name", "learning_rate", "epsilon", "weight_decay"}
    expected_training_keys = {
        "joint_epochs",
        "scheduler",
        "batch_heads",
        "observations_per_head",
        "gradient_clip_norm",
        "ema_decay",
        "precision_policy",
    }
    for index, trial in enumerate(trials):
        _exact_keys(trial, {"trial_role", "model", "optimizer", "training"}, f"trial {index}")
        model = trial["model"]
        optimizer = trial["optimizer"]
        training = trial["training"]
        _exact_keys(model, expected_model_keys, f"trial {index} model")
        _exact_keys(optimizer, expected_optimizer_keys, f"trial {index} optimizer")
        _exact_keys(training, expected_training_keys, f"trial {index} training")
        _positive_int(model["hidden_dim"], "hidden_dim")
        dropout = _finite_nonnegative(model["dropout"], "dropout")
        _require(dropout < 1.0, "Dropout must be below one")
        _require(model["quantile_levels"] == [0.1, 0.25, 0.75, 0.9], "Quantiles changed")
        _require(model["loss_reduction"] in {"source_dense_mean", "observed_mean"}, "Bad loss reduction")
        _require(optimizer["name"] == "adam", "sciPENN requires Adam")
        _require(_finite_nonnegative(optimizer["learning_rate"], "learning rate") > 0, "Bad learning rate")
        _require(float(optimizer["epsilon"]) == 1.0e-8, "Adam epsilon changed")
        _require(float(optimizer["weight_decay"]) == 0.0, "Source Adam has no weight decay")
        _require(
            training["joint_epochs"] == 100
            and training["batch_heads"] == 4
            and training["observations_per_head"] == 1024,
            "Shared training exposure changed",
        )
        _require(float(training["gradient_clip_norm"]) > 0.0, "Gradient clip must be positive")
        _require(0.0 < float(training["ema_decay"]) < 1.0, "EMA decay is invalid")
        policy = training["precision_policy"]
        _require(policy in {"float32_no_tf32", "amp_bfloat16_tf32"}, "Bad precision policy")
        scheduler = training["scheduler"]
        if scheduler.get("name") == "fixed":
            _exact_keys(scheduler, {"name"}, "fixed scheduler")
        else:
            _require(
                scheduler == {"name": "common_linear_warmup_cosine_decay", "warmup_epochs": 2},
                "OPS scheduler changed",
            )
    _require(trials[0]["model"] == {
        "hidden_dim": 512,
        "dropout": 0.25,
        "quantile_levels": [0.1, 0.25, 0.75, 0.9],
        "loss_reduction": "source_dense_mean",
    }, "Trial 0 model is not the source anchor")
    _require(
        trials[0]["optimizer"]["learning_rate"] == 0.001
        and trials[0]["training"]["precision_policy"] == "float32_no_tf32",
        "Trial 0 optimizer or precision changed",
    )
    _require(
        any(
            trial["model"]["hidden_dim"] == 512
            and trial["model"]["dropout"] == 0.25
            and trial["model"]["loss_reduction"] == "observed_mean"
            for trial in trials[1:]
        ),
        "Missing source-architecture observed-mean stabilization trial",
    )


@dataclass(frozen=True)
class DevelopmentJob:
    split: str
    fold: int
    trial_index: int
    seed: int
    output_root: Path

    @property
    def method(self) -> str:
        return METHODS[0]

    @property
    def job_id(self) -> str:
        return f"{self.method}__{self.split}__fold{self.fold}__trial{self.trial_index}"

    def command(self, *, python: Path, runner: Path, config: Path, precision: str) -> list[str]:
        command = [
            "env",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            str(python.resolve()),
            "-u",
            str(runner.resolve()),
            "--method",
            self.method,
            "--reporters",
            ",".join(PANEL12),
            "--split",
            self.split,
            "--fold",
            str(self.fold),
            "--trial-index",
            str(self.trial_index),
            "--seed",
            str(self.seed),
            "--campaign-config",
            str(config.resolve()),
            "--mode",
            "development",
            "--train-only",
            "--device",
            "cuda",
            "--gpu-staging",
            "required",
            "--gpu-staging-reserve-gib",
            "16",
        ]
        for flag, key in (
            ("--phase-cache", "phase_cache"),
            ("--exact-root", "exact_root"),
            ("--target-table", "target_table"),
            ("--target-feature-dictionary", "target_feature_dictionary"),
            ("--specialist-reference-root", "specialist_reference_root"),
            ("--preprocessing-reference-root", "preprocessing_reference_root"),
        ):
            command.extend((flag, str(FROZEN_ASSET_PATHS[key])))
        if precision == "float32_no_tf32":
            command.extend(("--no-amp", "--no-tf32"))
        elif precision == "amp_bfloat16_tf32":
            command.extend(("--amp", "--tf32"))
        else:
            raise CampaignError(f"Unsupported precision policy {precision!r}")
        command.extend(("--output-root", str(self.output_root.resolve())))
        return command


def development_jobs(payload: Mapping[str, Any], output_root: Path) -> list[DevelopmentJob]:
    validate_config(payload)
    seed = int(payload["comparison_contract"]["search_seed"])
    return [
        DevelopmentJob(
            split=split,
            fold=fold,
            trial_index=trial,
            seed=seed,
            output_root=(
                output_root
                / METHODS[0]
                / "development"
                / split
                / f"fold_{fold}"
                / f"trial_{trial}"
            ),
        )
        for trial in range(4)
        for split, fold in DEVELOPMENT_FOLDS
    ]


def _atomic_text(path: Path, value: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_commands(
    *, config: Path, runner: Path, python: Path, output_root: Path, artifact_root: Path
) -> dict[str, Any]:
    config = config.resolve()
    runner = runner.resolve()
    python = python.resolve()
    output_root = output_root.resolve()
    artifact_root = artifact_root.resolve()
    payload = read_config(config)
    validate_config(payload)
    _require(runner.is_file(), f"sciPENN runner is missing: {runner}")
    _require(python.is_file(), f"Persistent Python is missing: {python}")
    missing = [str(path) for path in FROZEN_ASSET_PATHS.values() if not path.exists()]
    _require(not missing, f"Frozen command assets are missing: {missing}")
    jobs = development_jobs(payload, output_root)
    trials = payload["methods"][METHODS[0]]["trial_configs"]
    commands = [
        job.command(
            python=python,
            runner=runner,
            config=config,
            precision=str(trials[job.trial_index]["training"]["precision_policy"]),
        )
        for job in jobs
    ]
    rows = [
        {
            "job_id": job.job_id,
            "method": job.method,
            "split": job.split,
            "fold": job.fold,
            "trial_index": job.trial_index,
            "seed": job.seed,
            "output_root": str(job.output_root.resolve()),
            "command": shlex.join(command),
        }
        for job, command in zip(jobs, commands)
    ]
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        tsv = handle.read()
    commands_tsv = artifact_root / "commands.tsv"
    commands_sh = artifact_root / "commands.sh"
    manifest_path = artifact_root / "manifest.json"
    _atomic_text(commands_tsv, tsv)
    _atomic_text(
        commands_sh,
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "# Generated only; this file was not executed by the preparation step.",
                *[shlex.join(command) for command in commands],
            ]
        )
        + "\n",
    )
    manifest = {
        "schema_version": COMMAND_SCHEMA_VERSION,
        "status": "READY_NOT_STARTED",
        "campaign_config": str(config),
        "campaign_config_sha256": file_sha256(config),
        "runner": str(runner),
        "runner_sha256": file_sha256(runner),
        "python": str(python),
        "n_methods": 1,
        "n_trials_per_method": 4,
        "n_development_folds": 2,
        "n_jobs": 8,
        "serial_order": True,
        "parallel_launch_performed": False,
        "training_started": False,
        "real_ops_data_opened": False,
        "outer_test_label_access_allowed": False,
        "outer_test_identity_metadata_access_allowed": True,
        "outer_test_evaluation_allowed": False,
        "commands_tsv": str(commands_tsv),
        "commands_tsv_sha256": file_sha256(commands_tsv),
        "commands_sh": str(commands_sh),
        "commands_sh_sha256": file_sha256(commands_sh),
        "frozen_asset_paths": {name: str(path) for name, path in FROZEN_ASSET_PATHS.items()},
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare_commands(
        config=args.config,
        runner=args.runner,
        python=args.python,
        output_root=args.output_root,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
