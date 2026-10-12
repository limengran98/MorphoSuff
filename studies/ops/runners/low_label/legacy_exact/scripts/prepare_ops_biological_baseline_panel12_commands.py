#!/usr/bin/env python3
"""Freeze serial development commands for OPS biological baselines.

The script only validates the frozen campaign and writes commands.  It cannot
execute them, spawn subprocesses, or access the outer test.  This makes the
user-approval boundary visible: generating ``commands.sh`` is not equivalent
to starting any training job.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "ops_biological_baseline_panel12_development_v2.json"
)
DEFAULT_RUNNER = PROJECT_ROOT / "scripts" / "run_ops_biological_baseline_training.py"
FROZEN_ASSET_PATHS = {
    "phase_cache": (PROJECT_ROOT / "data/processed/ops_phase172_indexed").resolve(),
    "exact_root": (
        PROJECT_ROOT / "data/processed/ops_full_reporter_exact/reporters"
    ).resolve(),
    "target_table": (
        PROJECT_ROOT / "results/ops_phase0_asset_audit/reporter_targets.csv"
    ).resolve(),
    "target_feature_dictionary": (
        PROJECT_ROOT
        / "results/ops_phase0_asset_audit/target_feature_dictionary.csv"
    ).resolve(),
    "specialist_reference_root": (
        PROJECT_ROOT / "results/ops_reporter_specialists_v1"
    ).resolve(),
    "preprocessing_reference_root": (
        PROJECT_ROOT / "results/ops_reporter_masked_multitask_resmlp_v1"
    ).resolve(),
}
SCHEMA_VERSION = "ops-biological-baseline-development-campaign-v2"
COMMAND_SCHEMA_VERSION = "ops-biological-baseline-development-commands-v2"
METHODS = ("ops_captain", "scbutterfly_ops_b", "midas_ops")
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


class CampaignError(RuntimeError):
    """A frozen campaign or command invariant was violated."""


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise CampaignError("Campaign contains a non-JSON value") from error


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object_without_duplicates(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CampaignError(f"Duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def read_config(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(
                handle, object_pairs_hook=_object_without_duplicates
            )
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"Unable to read campaign config {path}: {error}") from error
    if not isinstance(payload, dict):
        raise CampaignError("Campaign config must be a JSON object")
    canonical_json(payload)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CampaignError(message)


def _recursive_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key).casefold())
            keys.extend(_recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_recursive_keys(item))
    return keys


def validate_config(payload: Mapping[str, Any]) -> None:
    _require(payload.get("schema_version") == SCHEMA_VERSION, "Wrong schema")
    _require(
        payload.get("campaign_id")
        == "ops-biological-baseline-panel12-development-v2",
        "Wrong campaign ID",
    )
    _require(
        payload.get("supersedes_campaign_id")
        == "ops-biological-baseline-panel12-development-v1",
        "Wrong superseded campaign",
    )
    revision = str(payload.get("revision_reason", ""))
    _require(
        all(name in revision for name in ("CAPTAIN", "scButterfly", "MIDAS")),
        "v2 revision reason must disclose all three fidelity corrections",
    )
    _require(payload.get("frozen") is True, "Campaign is not frozen")
    _require(tuple(payload.get("method_order", ())) == METHODS, "Method order changed")
    _require(tuple(payload.get("reporter_panel", ())) == PANEL12, "Panel12 changed")
    boundary = payload.get("execution_boundary", {})
    _require(boundary.get("development_only") is True, "Not development-only")
    _require(
        boundary.get("outer_test_label_access_allowed") is False,
        "Outer-test label access must be false",
    )
    _require(
        boundary.get("outer_test_identity_metadata_access_allowed") is True,
        "Frozen outer-test identity metadata must remain available for cohort fingerprints",
    )
    _require(
        boundary.get("outer_test_evaluation_allowed") is False,
        "Development outer-test evaluation must be false",
    )
    _require(
        boundary.get("formal_training_authorized") is False,
        "Formal training must remain unauthorized",
    )
    _require(
        boundary.get("automatic_parallel_launch_allowed") is False,
        "Automatic parallel launch must remain forbidden",
    )
    comparison = payload.get("comparison_contract", {})
    _require(comparison.get("trial_count_per_method") == 4, "Expected four trials")
    _require(comparison.get("search_seed") == 20260721, "Search seed changed")
    observed_folds = tuple(
        (str(row.get("split")), int(row.get("fold")))
        for row in comparison.get("development_folds", [])
        if isinstance(row, Mapping)
    )
    _require(observed_folds == DEVELOPMENT_FOLDS, "Development folds changed")
    _require(
        comparison.get("selection_objective")
        == {
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
        },
        "Validation selection objective changed",
    )
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
    _require(
        comparison.get("shared_primary_exposure")
        == {
            "shared_model_reporters_per_primary_event": 4,
            "specialist_reporters_per_primary_event": 1,
            "observations_per_reporter_per_event": 1024,
            "reporter_rounds_per_epoch": (
                "frozen V1 reporter rounds for the same split/fold"
            ),
            "per_reporter_event_batch_size_matched": True,
            "total_joint_reporter_exposure_equalized": False,
            "scbutterfly_joint_reporter_exposure_multiplier_vs_standard": 2.0,
            "scbutterfly_total_target_bearing_exposure_multiplier_vs_standard": 3.0,
            "optimizer_step_calls_equalized": False,
            "event_and_optimizer_step_reporting_policy": (
                "method-native primary events, target-bearing exposures and "
                "owner optimizer.step calls are counted separately"
            ),
            "standard_vector_model_joint_epochs": 100,
            "standard_vector_validation_opportunities": 100,
            "scbutterfly_joint_epochs": (
                "OPS task-adapted per-trial schedule, reported separately and "
                "never described as an exact upstream schedule"
            ),
            "scbutterfly_validation_opportunities": 200,
            "early_stopping_changes_budget": False,
        },
        "Primary exposure/update-count contract changed",
    )
    resources = comparison.get("resource_metrics", {})
    _require(resources.get("role") == "reporting_only", "Resources are not reporting-only")
    _require(
        resources.get("used_for_admission_or_hpo_selection") is False,
        "Resource metrics may not select models",
    )
    methods = payload.get("methods", {})
    _require(set(methods) == set(METHODS), "Method registry changed")
    forbidden_capacity_keys = {
        "max_parameters",
        "parameter_limit",
        "parameter_budget",
        "target_parameter_count",
        "reference_parameter_count",
        "match_parameter_count",
    }
    for method in METHODS:
        record = methods.get(method)
        _require(isinstance(record, Mapping), f"Missing method {method}")
        trials = record.get("trial_configs")
        _require(isinstance(trials, list) and len(trials) == 4, f"{method} needs four trials")
        for index, trial in enumerate(trials):
            expected_trial_keys = {"trial_role", "model", "optimizer", "training"}
            _require(
                isinstance(trial, Mapping)
                and set(trial) == expected_trial_keys,
                f"Invalid {method} trial {index}",
            )
            observed_keys = set(_recursive_keys(trial))
            _require(
                observed_keys.isdisjoint(forbidden_capacity_keys),
                f"{method} trial {index} contains a parameter cap",
            )
            training = trial["training"]
            if method == "scbutterfly_ops_b":
                _require(
                    training.get("specialists_per_primary_event") == 1
                    and "batch_heads" not in training
                    and training.get("observations_per_head") == 1024,
                    f"{method} trial {index} changed specialist exposure/update semantics",
                )
            else:
                _require(
                    training.get("batch_heads") == 4
                    and training.get("observations_per_head") == 1024,
                    f"{method} trial {index} changed primary reporter exposure",
                )
            canonical_json(trial)
    captain = methods["ops_captain"]
    _require(
        "semantic_query_fields" in captain.get("forbidden_adaptations", []),
        "CAPTAIN semantic-query exclusion is missing",
    )
    source_anchor = captain.get("source_anchor_contract", {})
    _require(
        source_anchor.get("trial_index") == 0
        and source_anchor.get("upstream_repository")
        == "external/original_methods/captain"
        and source_anchor.get("upstream_commit")
        == "19a94cbe8d859ce9f89b7f480f7faaa234935a4b"
        and source_anchor.get("source_files")
        == ["pretrain/protein_model.py", "pretrain/torchrun.py"],
        "CAPTAIN source provenance anchor changed",
    )
    _require(
        source_anchor.get("source_shaped_endpoint_decoder")
        == {
            "model_dim": 512,
            "cross_attention_depth": 6,
            "num_heads": 8,
            "quantile_levels": [0.1, 0.25, 0.75, 0.9],
            "loss_weights": {
                "mean": 0.6,
                "quantile": 0.2,
                "phase_reconstruction": 0.2,
            },
        },
        "CAPTAIN source-shaped endpoint decoder changed",
    )
    _require(
        source_anchor.get("source_shaped_dropout")
        == {
            "embedding_dropout": 0.0,
            "attention_dropout": 0.1,
            "feedforward_dropout": 0.1,
        },
        "CAPTAIN source-shaped dropout anchor changed",
    )
    _require(
        source_anchor.get("source_shaped_optimizer")
        == {
            "name": "adam",
            "learning_rate": 1.0e-5,
            "epsilon": 1.0e-4,
            "weight_decay": 0.0,
        }
        and source_anchor.get("source_shaped_scheduler")
        == {"name": "fixed_step_lr", "step_size_epochs": 1, "gamma": 1.0},
        "CAPTAIN source-shaped optimizer/scheduler changed",
    )
    phase_adapter = source_anchor.get("ops_phase_task_adapter", {})
    _require(
        phase_adapter.get("input") == "frozen_172D_phase_morphology"
        and phase_adapter.get("phase_encoder_depth") == 2
        and phase_adapter.get("pretrained_checkpoint_transfer") is False
        and "RNA" in str(phase_adapter.get("reason", ""))
        and "172" in str(phase_adapter.get("implementation", "")),
        "CAPTAIN OPS phase task-adapter boundary changed",
    )
    captain_trials = captain["trial_configs"]
    _require(
        [trial.get("trial_role") for trial in captain_trials]
        == ["source_shaped_fixed_anchor"]
        + ["validation_only_ops_variant"] * 3,
        "CAPTAIN requires one source-shaped anchor and three OPS variants",
    )
    captain_model_keys = {
        "model_dim",
        "num_heads",
        "phase_encoder_depth",
        "cross_attention_depth",
        "feedforward_multiplier",
        "embedding_dropout",
        "attention_dropout",
        "feedforward_dropout",
        "quantile_levels",
        "phase_reconstruction_enabled",
        "loss_weights",
    }
    for trial_index, trial in enumerate(captain_trials):
        _require(
            set(trial["model"]) == captain_model_keys,
            "CAPTAIN split dropout/model fields changed",
        )
        _require(
            all(
                0.0 <= float(trial["model"][key]) < 1.0
                for key in (
                    "embedding_dropout",
                    "attention_dropout",
                    "feedforward_dropout",
                )
            ),
            "CAPTAIN dropout values must lie in [0, 1)",
        )
        _require(
            trial["model"].get("phase_reconstruction_enabled") is True
            and trial["model"].get("quantile_levels")
            == [0.1, 0.25, 0.75, 0.9]
            and trial["model"].get("loss_weights")
            == {"mean": 0.6, "quantile": 0.2, "phase_reconstruction": 0.2},
            "CAPTAIN native auxiliary heads were disabled",
        )
        _require(
            trial["optimizer"].get("name") == "adam"
            and trial["optimizer"].get("epsilon") == 1.0e-4,
            "CAPTAIN optimizer differs from the pinned public training code",
        )
        scheduler = trial["training"].get("scheduler", {})
        if trial_index == 0:
            _require(
                trial["model"].get("model_dim") == 512
                and trial["model"].get("cross_attention_depth") == 6
                and trial["model"].get("num_heads") == 8
                and trial["model"].get("phase_encoder_depth") == 2
                and trial["model"].get("embedding_dropout") == 0.0
                and trial["model"].get("attention_dropout") == 0.1
                and trial["model"].get("feedforward_dropout") == 0.1,
                "CAPTAIN source-shaped anchor architecture changed",
            )
            _require(
                trial["optimizer"]
                == {
                    "name": "adam",
                    "learning_rate": 1.0e-5,
                    "epsilon": 1.0e-4,
                    "weight_decay": 0.0,
                }
                and scheduler
                == {
                    "name": "fixed_step_lr",
                    "step_size_epochs": 1,
                    "gamma": 1.0,
                },
                "CAPTAIN source-shaped fixed optimization anchor changed",
            )
        else:
            _require(
                scheduler.get("name")
                == "common_linear_warmup_cosine_decay"
                and isinstance(scheduler.get("warmup_epochs"), int)
                and scheduler["warmup_epochs"] > 0,
                "CAPTAIN OPS variant scheduler changed",
            )
    butterfly = methods["scbutterfly_ops_b"]
    fidelity = butterfly.get("source_fidelity_contract", {})
    _require(
        fidelity.get("source_architecture_factory")
        == "ScButterflyOPSConfig.source_anchor"
        and fidelity.get("source_training_factory")
        == "ScButterflyTrainingConfig.source_anchor"
        and "no trial is an exact" in str(fidelity.get("claim_boundary", "")),
        "scButterfly source/task-adaptation boundary changed",
    )
    schedule = butterfly.get("training_schedule_contract", {})
    _require(
        schedule
        == {
            "phase_pretrain_epochs": 100,
            "phenotype_pretrain_epochs": 100,
            "joint_epochs": 200,
            "kl_warmup_epochs": 50,
            "discriminator_steps_per_generator_step": 1,
            "source_relationship": (
                "source-centered epoch counts under an OPS task-adapted batch "
                "and checkpoint protocol"
            ),
            "upstream_execution_exact": False,
        },
        "scButterfly task-adapted schedule contract changed",
    )
    _require(
        butterfly.get("primary_exposure_contract")
        == {
            "specialists_per_primary_event": 1,
            "observations_per_specialist": 1024,
            "per_reporter_event_batch_size_matches_shared_methods": True,
            "joint_reporter_exposure_multiplier_vs_standard": 2.0,
            "phenotype_pretraining_target_exposure_multiplier_vs_standard": 1.0,
            "phase_pretraining_target_exposure_multiplier_vs_standard": 0.0,
            "total_target_bearing_exposure_multiplier_vs_standard": 3.0,
            "validation_opportunities": 200,
            "paired_discriminator_generator_events": True,
            "owner_optimizer_step_calls_per_event": {
                "phase_pretrain": 3,
                "phenotype_pretrain": 3,
                "joint_discriminator": 2,
                "joint_generator": 5,
            },
            "reporting_rule": (
                "report primary events, target-bearing reporter exposure and "
                "owner optimizer.step calls separately"
            ),
        },
        "scButterfly specialist exposure/update contract changed",
    )
    butterfly_trials = butterfly["trial_configs"]
    _require(
        [trial.get("trial_role") for trial in butterfly_trials]
        == [
            "source_anchor",
            "source_mechanics_ops_tuned",
            "source_mechanics_ops_tuned",
            "ops_stabilized_sensitivity",
        ],
        "scButterfly trial roles changed",
    )
    source_losses = {
        "phase_pretrain_kl": 20.0 / 172.0,
        "phenotype_pretrain_kl": 20.0 / 172.0,
        "phase_reconstruction": 1.0,
        "phenotype_reconstruction": 2.0,
        "phase_joint_kl": 40.0 / 172.0,
        "phenotype_joint_kl": 40.0 / 172.0,
        "adversarial": 1.0,
    }
    model_keys = {
        "architecture_factory",
        "phase_encoder_widths",
        "phenotype_encoder_widths",
        "phase_decoder_widths",
        "phenotype_decoder_widths",
        "latent_dim",
        "discriminator_hidden_widths",
        "discriminator_output_batch_norm",
        "dropout",
        "phase_input_mask_rate",
        "phenotype_input_mask_rate",
    }
    training_keys = {
        "adversarial_protocol",
        "source_adversarial_threshold",
        "loss_weights",
        "phase_pretrain_epochs",
        "phenotype_pretrain_epochs",
        "joint_epochs",
        "kl_warmup_epochs",
        "discriminator_steps_per_generator_step",
        "precision_policy",
        "specialists_per_primary_event",
        "observations_per_head",
        "gradient_clip_norm",
        "ema_decay",
    }
    for index, trial in enumerate(butterfly_trials):
        _require(set(trial["model"]) == model_keys, "scButterfly model fields changed")
        _require(
            set(trial["training"]) == training_keys,
            "scButterfly protocol/noise/loss fields changed",
        )
        _require(
            trial["training"].get("discriminator_steps_per_generator_step") == 1,
            "scButterfly lost alternating adversarial training",
        )
        _require(
            trial["training"].get("specialists_per_primary_event") == 1
            and "batch_heads" not in trial["training"],
            "scButterfly must use one specialist per primary event",
        )
        _require(
            trial["optimizer"].get("generator_name") == "adam"
            and trial["optimizer"].get("discriminator_name") == "sgd",
            "scButterfly optimizer families differ from the method-native controller",
        )
        for key in (
            "phase_pretrain_epochs",
            "phenotype_pretrain_epochs",
            "joint_epochs",
            "kl_warmup_epochs",
            "discriminator_steps_per_generator_step",
        ):
            _require(
                trial["training"][key] == schedule[key],
                "scButterfly task-adapted stage schedule changed",
            )
        if index < 3:
            _require(
                trial["training"]["adversarial_protocol"] == "source_anchor"
                and trial["training"]["loss_weights"] == source_losses,
                "scButterfly source mechanics/loss anchor changed",
            )
        else:
            _require(
                trial["training"]["adversarial_protocol"] == "ops_stabilized",
                "OPS-stabilized trial must remain an explicit sensitivity analysis",
            )
        if index == 0:
            _require(
                trial["training"]["precision_policy"] == "float32_no_tf32"
                and trial["training"]["gradient_clip_norm"] is None,
                "scButterfly source anchor must use fp32 and no clipping",
            )
        else:
            _require(
                trial["training"]["precision_policy"]
                == "amp_bfloat16_tf32"
                and float(trial["training"]["gradient_clip_norm"]) > 0,
                "scButterfly OPS variants require their explicit precision/clip policy",
            )
    anchor_model = butterfly_trials[0]["model"]
    _require(
        anchor_model["architecture_factory"] == "source_anchor"
        and anchor_model["phase_encoder_widths"] == [256, 128]
        and anchor_model["phenotype_encoder_widths"] == [128, 128]
        and anchor_model["phase_decoder_widths"] == [256]
        and anchor_model["phenotype_decoder_widths"] == [128]
        and anchor_model["latent_dim"] == 128
        and anchor_model["phase_input_mask_rate"] == 0.5
        and anchor_model["phenotype_input_mask_rate"] == 0.0,
        "scButterfly source architecture/noise anchor changed",
    )

    midas = methods["midas_ops"]
    _require(
        midas.get("scope") == "phase_plus_52_independent_sparse_reporter_modalities"
        and "dense_1604_target_union" in midas.get("forbidden_adaptations", []),
        "MIDAS 53-modality scope changed",
    )
    batch_policy = midas.get("batch_policy", {})
    _require(
        batch_policy.get("batch_disentanglement") is False
        and batch_policy.get("batch_covariate_kind") == "none"
        and batch_policy.get("reporter_id_as_batch_forbidden") is True,
        "MIDAS batch policy changed",
    )
    midas_fidelity = midas.get("source_fidelity_contract", {})
    _require(
        midas_fidelity.get("upstream_commit")
        == "3ef7847c88c90583c05147cafef496d862986dd3"
        and midas_fidelity.get("technical_information_bottleneck_multiplier") == 5.0
        and midas_fidelity.get("source_modality_alignment_multiplier") == 50.0,
        "MIDAS source provenance or information bottleneck changed",
    )
    midas_trials = midas["trial_configs"]
    _require(
        [trial.get("trial_role") for trial in midas_trials]
        == ["source_fixed_adamw_anchor"] + ["validation_only_ops_variant"] * 3,
        "MIDAS trial roles changed",
    )
    midas_model_keys = {
        "biological_dim",
        "technical_dim",
        "modality_width",
        "shared_encoder_hidden_dims",
        "shared_decoder_hidden_dims",
        "dropout",
        "loss_weights",
        "technical_information_bottleneck_multiplier",
        "reporter_reconstruction_reduction",
        "batch_disentanglement",
        "batch_covariate_kind",
    }
    for index, trial in enumerate(midas_trials):
        model = trial["model"]
        _require(set(model) == midas_model_keys, "MIDAS model fields changed")
        _require(
            set(model["loss_weights"])
            == {
                "phase_reconstruction",
                "reporter_reconstruction",
                "kl_biological",
                "kl_technical",
                "modality_alignment",
            }
            and "phenotype_reconstruction" not in model["loss_weights"],
            "MIDAS reporter-modality loss keys changed",
        )
        _require(
            float(model["technical_information_bottleneck_multiplier"]) > 0
            and model["batch_disentanglement"] is False
            and model["batch_covariate_kind"] == "none",
            "MIDAS bottleneck or batch covariate fields changed",
        )
        _require(
            trial["optimizer"].get("name") == "adamw"
            and trial["optimizer"].get("epsilon") == 1.0e-8,
            "MIDAS optimizer differs from the pinned reproducibility code",
        )
        scheduler = trial["training"].get("scheduler", {})
        if index == 0:
            _require(
                trial["optimizer"]
                == {
                    "name": "adamw",
                    "learning_rate": 1.0e-4,
                    "epsilon": 1.0e-8,
                    "weight_decay": 0.01,
                }
                and model["loss_weights"]["modality_alignment"] == 50.0
                and scheduler == {"name": "fixed"},
                "MIDAS source fixed AdamW anchor changed",
            )
        else:
            _require(
                scheduler.get("name") == "common_linear_warmup_cosine_decay"
                and int(scheduler.get("warmup_epochs", 0)) > 0,
                "MIDAS OPS scheduler changed",
            )
    _require(
        [
            float(trial["model"]["loss_weights"]["modality_alignment"])
            for trial in midas_trials
        ]
        == [50.0, 10.0, 1.0, 0.1],
        "MIDAS modality-alignment HPO coverage changed",
    )


@dataclass(frozen=True)
class DevelopmentJob:
    method: str
    split: str
    fold: int
    trial_index: int
    seed: int
    output_root: Path

    @property
    def job_id(self) -> str:
        return f"{self.method}__{self.split}__fold{self.fold}__trial{self.trial_index}"

    def command(
        self,
        *,
        python: Path,
        runner: Path,
        config: Path,
    ) -> list[str]:
        python = python.expanduser().resolve()
        runner = runner.expanduser().resolve()
        config = config.expanduser().resolve()
        output_root = self.output_root.expanduser().resolve()
        command = [
            "env",
            "CUBLAS_WORKSPACE_CONFIG=:4096:8",
            str(python),
            "-u",
            str(runner),
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
            str(config),
            "--mode",
            "development",
            "--train-only",
            "--device",
            "cuda",
            "--gpu-staging",
            "required",
            "--gpu-staging-reserve-gib",
            "16",
            "--phase-cache",
            str(FROZEN_ASSET_PATHS["phase_cache"]),
            "--exact-root",
            str(FROZEN_ASSET_PATHS["exact_root"]),
            "--target-table",
            str(FROZEN_ASSET_PATHS["target_table"]),
            "--target-feature-dictionary",
            str(FROZEN_ASSET_PATHS["target_feature_dictionary"]),
            "--specialist-reference-root",
            str(FROZEN_ASSET_PATHS["specialist_reference_root"]),
            "--preprocessing-reference-root",
            str(FROZEN_ASSET_PATHS["preprocessing_reference_root"]),
        ]
        if self.method == "scbutterfly_ops_b" and self.trial_index == 0:
            command.extend(("--no-amp", "--no-tf32"))
        command.extend(("--output-root", str(output_root)))
        return command


def development_jobs(payload: Mapping[str, Any], output_root: Path) -> list[DevelopmentJob]:
    validate_config(payload)
    seed = int(payload["comparison_contract"]["search_seed"])
    jobs: list[DevelopmentJob] = []
    for method in METHODS:
        for trial_index in range(4):
            for split, fold in DEVELOPMENT_FOLDS:
                destination = (
                    output_root
                    / method
                    / "development"
                    / split
                    / f"fold_{fold}"
                    / f"trial_{trial_index}"
                )
                jobs.append(
                    DevelopmentJob(
                        method=method,
                        split=split,
                        fold=fold,
                        trial_index=trial_index,
                        seed=seed,
                        output_root=destination,
                    )
                )
    return jobs


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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
    config = args.config.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    python = args.python.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    payload = read_config(config)
    validate_config(payload)
    if not runner.is_file():
        raise FileNotFoundError(f"Training runner is not ready: {runner}")
    if not python.is_file():
        raise FileNotFoundError(f"Persistent environment Python is missing: {python}")
    missing_assets = [
        str(path) for path in FROZEN_ASSET_PATHS.values() if not path.exists()
    ]
    if missing_assets:
        raise FileNotFoundError(
            "Frozen command assets are missing: " + ", ".join(missing_assets)
        )
    jobs = development_jobs(payload, output_root)
    commands = [
        job.command(python=python, runner=runner, config=config) for job in jobs
    ]
    rows = []
    for job, command in zip(jobs, commands):
        rows.append(
            {
                "job_id": job.job_id,
                "method": job.method,
                "split": job.split,
                "fold": job.fold,
                "trial_index": job.trial_index,
                "seed": job.seed,
                "output_root": str(job.output_root),
                "command": shlex.join(command),
            }
        )
    tsv_lines: list[str] = []
    fieldnames = list(rows[0])
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        handle.seek(0)
        tsv_lines.append(handle.read())
    commands_tsv = artifact_root / "commands.tsv"
    commands_sh = artifact_root / "commands.sh"
    manifest_path = artifact_root / "manifest.json"
    _atomic_text(commands_tsv, "".join(tsv_lines))
    shell = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "# Generated only; this file was not executed by the preparation step.",
    ] + [shlex.join(command) for command in commands]
    _atomic_text(commands_sh, "\n".join(shell) + "\n")
    manifest = {
        "schema_version": COMMAND_SCHEMA_VERSION,
        "status": "READY_NOT_STARTED",
        "campaign_config": str(config),
        "campaign_config_sha256": file_sha256(config),
        "runner": str(runner),
        "runner_sha256": file_sha256(runner),
        "python": str(python),
        "frozen_asset_paths": {
            name: {
                "path": str(path),
                "kind": "file" if path.is_file() else "directory",
                "file_sha256": file_sha256(path) if path.is_file() else None,
            }
            for name, path in FROZEN_ASSET_PATHS.items()
        },
        "frozen_asset_path_contract_sha256": hashlib.sha256(
            canonical_json(
                {name: str(path) for name, path in FROZEN_ASSET_PATHS.items()}
            ).encode("utf-8")
        ).hexdigest(),
        "n_methods": len(METHODS),
        "n_trials_per_method": 4,
        "n_development_folds": len(DEVELOPMENT_FOLDS),
        "n_jobs": len(jobs),
        "serial_order": True,
        "parallel_launch_performed": False,
        "training_started": False,
        "outer_test_label_access_allowed": False,
        "outer_test_identity_metadata_access_allowed": True,
        "outer_test_evaluation_allowed": False,
        "commands_tsv": str(commands_tsv.resolve()),
        "commands_tsv_sha256": file_sha256(commands_tsv),
        "commands_sh": str(commands_sh.resolve()),
        "commands_sh_sha256": file_sha256(commands_sh),
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
