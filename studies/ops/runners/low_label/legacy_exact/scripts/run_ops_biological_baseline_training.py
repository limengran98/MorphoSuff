#!/usr/bin/env python3
"""Run one frozen development job for the OPS biological baselines.

The process owns exactly one method/split/fold/trial.  It is deliberately
development-only, CUDA-only, and train-only: outer-test fluorescence is never
opened by this module.  Data preparation, split construction, train-only
preprocessing, GPU staging, validation cohorts, and the common checkpoint
objective are reused from :mod:`run_ops_reporter_candidate_training`.

CAPTAIN and MIDAS use one reporter-balanced primary loop.  scButterfly remains
twelve independent specialists and uses its method-native phase pretraining,
phenotype pretraining, and alternating discriminator/generator controller.
Resource and capacity measurements are reporting-only.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import h5py
from torch import nn

import ops_biological_baseline_harness_models as biological_models
import ops_reporter_masked_multitask_v2_lib as sampler_library
import ops_reporter_runtime_metrics as runtime_metrics
import ops_reporter_training_harness_lib as harness
import prepare_ops_biological_baseline_panel12_commands as campaign_contract
import run_ops_reporter_candidate_training as common_runner
import run_ops_reporter_masked_multitask_resmlp_v2 as frozen_engine
from ops_biological_baseline_scbutterfly import (
    ScButterflyOPSConfig,
    ScButterflyOPSSpecialist,
)
from ops_biological_baseline_scbutterfly_training import (
    ScButterflyPairedBatch,
    ScButterflyTrainingConfig,
    ScButterflyTrainingController,
)
from ops_biological_baseline_captain_training import (
    CaptainReporterTrainingBatch,
)
from ops_reporter_specialist_lib import atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "ops-biological-baseline-development-training-v2"
CHECKPOINT_SCHEMA_VERSION = "ops-biological-baseline-checkpoint-v2"
DEFAULT_CAMPAIGN_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "ops_biological_baseline_panel12_development_v2.json"
)
METHODS = ("ops_captain", "scbutterfly_ops_b", "midas_ops")
SHARED_LOOP_METHODS = ("ops_captain", "midas_ops")
DEVELOPMENT_FOLDS = (
    ("gene_holdout_main", 0),
    ("field_holdout_sanity", 0),
)
PANEL12 = campaign_contract.PANEL12


class BiologicalRunnerError(RuntimeError):
    """A frozen development-run invariant was violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BiologicalRunnerError(message)


def _exact_keys(
    payload: Mapping[str, Any], expected: set[str], label: str
) -> None:
    observed = set(payload)
    _require(
        observed == expected,
        f"{label} keys changed: expected={sorted(expected)}, "
        f"observed={sorted(observed)}",
    )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise BiologicalRunnerError(f"{label} must be an integer")
    result = int(value)
    _require(result > 0 and result == value, f"{label} must be a positive integer")
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    result = float(value)
    _require(
        math.isfinite(result) and result >= 0.0,
        f"{label} must be finite and non-negative",
    )
    return result


def _positive_float(value: Any, label: str) -> float:
    result = float(value)
    _require(
        math.isfinite(result) and result > 0.0,
        f"{label} must be finite and positive",
    )
    return result


def _probability(value: Any, label: str, *, allow_zero: bool = True) -> float:
    result = float(value)
    lower = result >= 0.0 if allow_zero else result > 0.0
    _require(
        math.isfinite(result) and lower and result < 1.0,
        f"{label} must be in {'[0, 1)' if allow_zero else '(0, 1)'}",
    )
    return result


def _validate_common_training(
    training: Mapping[str, Any],
    *,
    expected_keys: set[str],
    label: str,
    primary_event_field: str = "batch_heads",
    expected_reporters_per_primary_event: int = 4,
) -> None:
    _exact_keys(training, expected_keys, label)
    _require(
        training[primary_event_field]
        == expected_reporters_per_primary_event,
        (
            f"{label} {primary_event_field} must be "
            f"{expected_reporters_per_primary_event}"
        ),
    )
    _require(
        training["observations_per_head"] == 1024,
        f"{label} observations_per_head must be 1024",
    )
    if training["gradient_clip_norm"] is not None:
        _positive_float(training["gradient_clip_norm"], f"{label} gradient clip")
    _probability(training["ema_decay"], f"{label} EMA decay", allow_zero=False)


def _validate_captain_trial(
    trial: Mapping[str, Any], shared_joint_epochs: int, trial_index: int
) -> None:
    _exact_keys(
        trial,
        {"trial_role", "model", "optimizer", "training"},
        "CAPTAIN trial",
    )
    model = trial["model"]
    optimizer = trial["optimizer"]
    training = trial["training"]
    _require(isinstance(model, Mapping), "CAPTAIN model config must be a mapping")
    _require(
        isinstance(optimizer, Mapping),
        "CAPTAIN optimizer config must be a mapping",
    )
    _require(
        isinstance(training, Mapping),
        "CAPTAIN training config must be a mapping",
    )
    expected_role = (
        "source_shaped_fixed_anchor"
        if trial_index == 0
        else "validation_only_ops_variant"
    )
    _require(
        trial["trial_role"] == expected_role,
        "CAPTAIN trial role/index binding changed",
    )
    _exact_keys(
        model,
        {
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
        },
        "CAPTAIN model",
    )
    model_dim = _positive_int(model["model_dim"], "CAPTAIN model_dim")
    num_heads = _positive_int(model["num_heads"], "CAPTAIN num_heads")
    _require(model_dim % num_heads == 0, "CAPTAIN model_dim/head count mismatch")
    _positive_int(model["phase_encoder_depth"], "CAPTAIN phase depth")
    _positive_int(model["cross_attention_depth"], "CAPTAIN cross depth")
    _positive_int(model["feedforward_multiplier"], "CAPTAIN FF multiplier")
    for key in (
        "embedding_dropout",
        "attention_dropout",
        "feedforward_dropout",
    ):
        _probability(model[key], f"CAPTAIN {key}")
    quantiles = model["quantile_levels"]
    _require(
        quantiles == [0.1, 0.25, 0.75, 0.9],
        "CAPTAIN quantile levels changed",
    )
    _require(
        model["phase_reconstruction_enabled"] is True,
        "CAPTAIN phase reconstruction must remain enabled",
    )
    loss_weights = model["loss_weights"]
    _require(isinstance(loss_weights, Mapping), "CAPTAIN loss_weights must map")
    _exact_keys(
        loss_weights,
        {"mean", "quantile", "phase_reconstruction"},
        "CAPTAIN loss_weights",
    )
    _require(
        loss_weights
        == {"mean": 0.6, "quantile": 0.2, "phase_reconstruction": 0.2},
        "CAPTAIN source-shaped loss weights changed",
    )
    _exact_keys(
        optimizer,
        {"name", "learning_rate", "epsilon", "weight_decay"},
        "CAPTAIN optimizer",
    )
    _require(optimizer["name"] == "adam", "CAPTAIN pinned optimizer must be Adam")
    _positive_float(optimizer["learning_rate"], "CAPTAIN learning rate")
    _require(
        float(optimizer["epsilon"]) == 1.0e-4,
        "CAPTAIN Adam epsilon must match the AMP public code",
    )
    _nonnegative_float(optimizer["weight_decay"], "CAPTAIN weight decay")
    _validate_common_training(
        training,
        expected_keys={
            "joint_epochs",
            "scheduler",
            "phase_mask_rate",
            "batch_heads",
            "observations_per_head",
            "gradient_clip_norm",
            "ema_decay",
        },
        label="CAPTAIN training",
    )
    _require(
        _positive_int(training["joint_epochs"], "CAPTAIN epochs")
        == shared_joint_epochs,
        "CAPTAIN joint epochs differ from shared primary exposure",
    )
    scheduler = training["scheduler"]
    _require(isinstance(scheduler, Mapping), "CAPTAIN scheduler must be a mapping")
    if trial_index == 0:
        _require(
            model["model_dim"] == 512
            and model["cross_attention_depth"] == 6
            and model["num_heads"] == 8
            and model["phase_encoder_depth"] == 2,
            "CAPTAIN source-shaped anchor architecture changed",
        )
        _require(
            model["embedding_dropout"] == 0.0
            and model["attention_dropout"] == 0.1
            and model["feedforward_dropout"] == 0.1,
            "CAPTAIN source-shaped dropout anchor changed",
        )
        _require(
            optimizer
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
        _exact_keys(
            scheduler,
            {"name", "warmup_epochs"},
            "CAPTAIN OPS scheduler",
        )
        _require(
            scheduler["name"] == "common_linear_warmup_cosine_decay",
            "CAPTAIN OPS variants require the common cosine scheduler",
        )
        warmup = _positive_int(
            scheduler["warmup_epochs"], "CAPTAIN OPS warmup"
        )
        _require(warmup <= shared_joint_epochs, "CAPTAIN warmup exceeds training")
    _probability(
        training["phase_mask_rate"],
        "CAPTAIN phase mask rate",
        allow_zero=False,
    )


def _validate_midas_trial(
    trial: Mapping[str, Any], shared_joint_epochs: int, trial_index: int
) -> None:
    _exact_keys(
        trial,
        {"trial_role", "model", "optimizer", "training"},
        "MIDAS trial",
    )
    expected_role = (
        "source_fixed_adamw_anchor"
        if trial_index == 0
        else "validation_only_ops_variant"
    )
    _require(trial["trial_role"] == expected_role, "MIDAS trial role changed")
    model = trial["model"]
    optimizer = trial["optimizer"]
    training = trial["training"]
    _require(isinstance(model, Mapping), "MIDAS model config must be a mapping")
    _require(isinstance(optimizer, Mapping), "MIDAS optimizer must be a mapping")
    _require(isinstance(training, Mapping), "MIDAS training must be a mapping")
    _exact_keys(
        model,
        {
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
        },
        "MIDAS model",
    )
    for key in ("biological_dim", "technical_dim", "modality_width"):
        _positive_int(model[key], f"MIDAS {key}")
    for key in ("shared_encoder_hidden_dims", "shared_decoder_hidden_dims"):
        widths = model[key]
        _require(
            isinstance(widths, list) and widths,
            f"MIDAS {key} must be a non-empty list",
        )
        for value in widths:
            _positive_int(value, f"MIDAS {key}")
    _probability(model["dropout"], "MIDAS dropout")
    _require(
        model["batch_disentanglement"] is False,
        "MIDAS batch disentanglement must remain disabled",
    )
    _require(
        model["batch_covariate_kind"] == "none",
        "MIDAS has no frozen technical batch covariate",
    )
    _positive_float(
        model["technical_information_bottleneck_multiplier"],
        "MIDAS technical information-bottleneck multiplier",
    )
    _require(
        model["reporter_reconstruction_reduction"]
        in {"modality_mean", "observed_scalar_mean"},
        "MIDAS reporter reconstruction reduction changed",
    )
    weights = model["loss_weights"]
    _require(isinstance(weights, Mapping), "MIDAS loss_weights must map")
    _exact_keys(
        weights,
        {
            "phase_reconstruction",
            "reporter_reconstruction",
            "kl_biological",
            "kl_technical",
            "modality_alignment",
        },
        "MIDAS loss_weights",
    )
    for name, value in weights.items():
        _nonnegative_float(value, f"MIDAS {name} weight")
    _require(
        sum(float(value) for value in weights.values()) > 0.0,
        "MIDAS loss weights are empty",
    )
    _exact_keys(
        optimizer,
        {"name", "learning_rate", "epsilon", "weight_decay"},
        "MIDAS optimizer",
    )
    _require(
        optimizer["name"] == "adamw",
        "MIDAS pinned optimizer must be AdamW",
    )
    _positive_float(optimizer["learning_rate"], "MIDAS learning rate")
    _require(
        float(optimizer["epsilon"]) == 1.0e-8,
        "MIDAS AdamW epsilon must remain 1e-8",
    )
    _nonnegative_float(optimizer["weight_decay"], "MIDAS weight decay")
    _validate_common_training(
        training,
        expected_keys={
            "joint_epochs",
            "scheduler",
            "batch_heads",
            "observations_per_head",
            "gradient_clip_norm",
            "ema_decay",
        },
        label="MIDAS training",
    )
    _require(
        _positive_int(training["joint_epochs"], "MIDAS epochs")
        == shared_joint_epochs,
        "MIDAS joint epochs differ from shared primary exposure",
    )
    scheduler = training["scheduler"]
    _require(isinstance(scheduler, Mapping), "MIDAS scheduler must map")
    if trial_index == 0:
        _require(
            optimizer
            == {
                "name": "adamw",
                "learning_rate": 1.0e-4,
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
            }
            and scheduler == {"name": "fixed"},
            "MIDAS source fixed AdamW anchor changed",
        )
        _require(
            float(weights["modality_alignment"]) == 50.0,
            "MIDAS source modality-alignment multiplier must remain 50",
        )
    else:
        _exact_keys(
            scheduler,
            {"name", "warmup_epochs"},
            "MIDAS OPS scheduler",
        )
        _require(
            scheduler["name"] == "common_linear_warmup_cosine_decay",
            "MIDAS OPS variants require cosine decay",
        )
        warmup = _positive_int(
            scheduler["warmup_epochs"], "MIDAS warmup"
        )
        _require(warmup <= shared_joint_epochs, "MIDAS warmup exceeds training")


def _validate_scbutterfly_trial(
    trial: Mapping[str, Any], schedule: Mapping[str, Any], trial_index: int
) -> None:
    _exact_keys(
        trial,
        {"trial_role", "model", "optimizer", "training"},
        "scButterfly trial",
    )
    expected_roles = (
        "source_anchor",
        "source_mechanics_ops_tuned",
        "source_mechanics_ops_tuned",
        "ops_stabilized_sensitivity",
    )
    _require(
        trial["trial_role"] == expected_roles[trial_index],
        "scButterfly role/index binding changed",
    )
    model = trial["model"]
    optimizer = trial["optimizer"]
    training = trial["training"]
    _require(isinstance(model, Mapping), "scButterfly model must be a mapping")
    _require(
        isinstance(optimizer, Mapping),
        "scButterfly optimizer must be a mapping",
    )
    _require(
        isinstance(training, Mapping),
        "scButterfly training must be a mapping",
    )
    _exact_keys(
        model,
        {
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
        },
        "scButterfly model",
    )
    _require(
        model["architecture_factory"] in {"source_anchor", "ops_explicit"},
        "scButterfly architecture factory changed",
    )
    for key in (
        "phase_encoder_widths",
        "phenotype_encoder_widths",
        "phase_decoder_widths",
        "phenotype_decoder_widths",
    ):
        widths = model[key]
        _require(
            isinstance(widths, list) and widths,
            f"scButterfly {key} must be a non-empty list",
        )
        for value in widths:
            _positive_int(value, f"scButterfly {key}")
    discriminator_widths = model["discriminator_hidden_widths"]
    _require(
        isinstance(discriminator_widths, list),
        "scButterfly discriminator widths must be a list",
    )
    for value in discriminator_widths:
        _positive_int(value, "scButterfly discriminator width")
    _require(
        isinstance(model["discriminator_output_batch_norm"], bool),
        "scButterfly discriminator output BatchNorm flag must be boolean",
    )
    _positive_int(model["latent_dim"], "scButterfly latent_dim")
    _probability(model["dropout"], "scButterfly dropout")
    _probability(model["phase_input_mask_rate"], "scButterfly phase noise")
    _probability(
        model["phenotype_input_mask_rate"], "scButterfly phenotype noise"
    )
    _exact_keys(
        optimizer,
        {
            "generator_name",
            "generator_learning_rate",
            "generator_weight_decay",
            "discriminator_name",
            "discriminator_learning_rate",
        },
        "scButterfly optimizer",
    )
    _require(
        optimizer["generator_name"] == "adam",
        "scButterfly method-native generator optimizer must be Adam",
    )
    _require(
        optimizer["discriminator_name"] == "sgd",
        "scButterfly method-native discriminator optimizer must be SGD",
    )
    _positive_float(
        optimizer["generator_learning_rate"],
        "scButterfly generator learning rate",
    )
    _nonnegative_float(
        optimizer["generator_weight_decay"],
        "scButterfly generator weight decay",
    )
    _positive_float(
        optimizer["discriminator_learning_rate"],
        "scButterfly discriminator learning rate",
    )
    _validate_common_training(
        training,
        expected_keys={
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
        },
        label="scButterfly training",
        primary_event_field="specialists_per_primary_event",
        expected_reporters_per_primary_event=1,
    )
    _require(
        training["adversarial_protocol"] in {"source_anchor", "ops_stabilized"},
        "scButterfly adversarial protocol changed",
    )
    _positive_float(
        training["source_adversarial_threshold"],
        "scButterfly source adversarial threshold",
    )
    losses = training["loss_weights"]
    _require(isinstance(losses, Mapping), "scButterfly loss weights must map")
    _exact_keys(
        losses,
        {
            "phase_pretrain_kl",
            "phenotype_pretrain_kl",
            "phase_reconstruction",
            "phenotype_reconstruction",
            "phase_joint_kl",
            "phenotype_joint_kl",
            "adversarial",
        },
        "scButterfly loss weights",
    )
    for name, value in losses.items():
        _positive_float(value, f"scButterfly {name} weight")
    for key in (
        "phase_pretrain_epochs",
        "phenotype_pretrain_epochs",
        "joint_epochs",
        "kl_warmup_epochs",
        "discriminator_steps_per_generator_step",
    ):
        _positive_int(training[key], f"scButterfly {key}")
        _require(
            training[key] == schedule[key],
            f"scButterfly {key} differs from the task-adapted schedule",
        )
    _require(
        training["discriminator_steps_per_generator_step"] == 1,
        "scButterfly requires one discriminator step per generator step",
    )
    if trial_index < 3:
        _require(
            training["adversarial_protocol"] == "source_anchor",
            "Source-mechanics scButterfly trial lost source adversarial mechanics",
        )
    else:
        _require(
            training["adversarial_protocol"] == "ops_stabilized",
            "OPS-stabilized scButterfly must remain sensitivity-only",
        )
    if trial_index == 0:
        _require(
            training["precision_policy"] == "float32_no_tf32"
            and training["gradient_clip_norm"] is None,
            "scButterfly source anchor requires fp32 and no clipping",
        )
    else:
        _require(
            training["precision_policy"] == "amp_bfloat16_tf32"
            and training["gradient_clip_norm"] is not None,
            "scButterfly OPS trials require explicit AMP/clip policy",
        )
    if trial_index == 0:
        _require(
            model["architecture_factory"] == "source_anchor"
            and model["phase_encoder_widths"] == [256, 128]
            and model["phenotype_encoder_widths"] == [128, 128]
            and model["phase_decoder_widths"] == [256]
            and model["phenotype_decoder_widths"] == [128]
            and model["latent_dim"] == 128
            and model["phase_input_mask_rate"] == 0.5
            and model["phenotype_input_mask_rate"] == 0.0,
            "scButterfly source architecture/noise anchor changed",
        )


def validate_frozen_campaign(payload: Mapping[str, Any]) -> None:
    try:
        campaign_contract.validate_config(payload)
    except campaign_contract.CampaignError as error:
        raise BiologicalRunnerError(str(error)) from error
    comparison = payload["comparison_contract"]
    boundary = payload["execution_boundary"]
    _require(boundary["development_only"] is True, "Runner is not development-only")
    _require(
        boundary["outer_test_label_access_allowed"] is False,
        "Outer-test label access must be physically sealed",
    )
    _require(
        boundary["outer_test_identity_metadata_access_allowed"] is True,
        "Frozen outer-test identity fingerprints must remain available",
    )
    _require(
        boundary["outer_test_evaluation_allowed"] is False,
        "Development outer-test evaluation must remain forbidden",
    )
    _require(
        boundary["runner_entrypoint"]
        == "scripts/run_ops_biological_baseline_training.py",
        "Frozen runner entrypoint changed",
    )
    _require(
        boundary["result_filename"] == "training_result.json",
        "Frozen result filename changed",
    )
    shared = comparison["shared_primary_exposure"]
    _exact_keys(
        shared,
        {
            "shared_model_reporters_per_primary_event",
            "specialist_reporters_per_primary_event",
            "observations_per_reporter_per_event",
            "reporter_rounds_per_epoch",
            "per_reporter_event_batch_size_matched",
            "total_joint_reporter_exposure_equalized",
            "scbutterfly_joint_reporter_exposure_multiplier_vs_standard",
            "scbutterfly_total_target_bearing_exposure_multiplier_vs_standard",
            "optimizer_step_calls_equalized",
            "event_and_optimizer_step_reporting_policy",
            "standard_vector_model_joint_epochs",
            "standard_vector_validation_opportunities",
            "scbutterfly_joint_epochs",
            "scbutterfly_validation_opportunities",
            "early_stopping_changes_budget",
        },
        "shared primary exposure",
    )
    _require(
        shared["shared_model_reporters_per_primary_event"] == 4
        and shared["specialist_reporters_per_primary_event"] == 1
        and shared["observations_per_reporter_per_event"] == 1024,
        "Primary-event or per-reporter batch cardinality changed",
    )
    _require(
        shared["per_reporter_event_batch_size_matched"] is True
        and shared["total_joint_reporter_exposure_equalized"] is False
        and shared[
            "scbutterfly_joint_reporter_exposure_multiplier_vs_standard"
        ]
        == 2.0
        and shared[
            "scbutterfly_total_target_bearing_exposure_multiplier_vs_standard"
        ]
        == 3.0
        and shared["optimizer_step_calls_equalized"] is False
        and shared["event_and_optimizer_step_reporting_policy"]
        == (
            "method-native primary events, target-bearing exposures and owner "
            "optimizer.step calls are counted separately"
        ),
        "Primary exposure/update-count comparison policy changed",
    )
    _require(
        shared["standard_vector_validation_opportunities"] == 100
        and shared["scbutterfly_validation_opportunities"] == 200,
        "Validation-opportunity counts changed",
    )
    _require(
        shared["early_stopping_changes_budget"] is False,
        "Early stopping cannot change exposure",
    )
    shared_epochs = _positive_int(
        shared["standard_vector_model_joint_epochs"],
        "shared standard-vector joint epochs",
    )
    _require(
        shared["scbutterfly_joint_epochs"]
        == (
            "OPS task-adapted per-trial schedule, reported separately and "
            "never described as an exact upstream schedule"
        ),
        "scButterfly primary schedule declaration changed",
    )
    methods = payload["methods"]
    for trial_index, trial in enumerate(methods["ops_captain"]["trial_configs"]):
        _validate_captain_trial(trial, shared_epochs, trial_index)
    butterfly_schedule = methods["scbutterfly_ops_b"][
        "training_schedule_contract"
    ]
    _require(
        butterfly_schedule["upstream_execution_exact"] is False,
        "scButterfly schedule must remain explicitly task-adapted",
    )
    _require(
        methods["scbutterfly_ops_b"]["primary_exposure_contract"]
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
    for trial_index, trial in enumerate(
        methods["scbutterfly_ops_b"]["trial_configs"]
    ):
        _validate_scbutterfly_trial(
            trial, butterfly_schedule, trial_index
        )
    for trial_index, trial in enumerate(methods["midas_ops"]["trial_configs"]):
        _validate_midas_trial(trial, shared_epochs, trial_index)
    _require(
        [
            float(trial["model"]["loss_weights"]["modality_alignment"])
            for trial in methods["midas_ops"]["trial_configs"]
        ]
        == [50.0, 10.0, 1.0, 0.1],
        "MIDAS modality-alignment HPO coverage changed",
    )


def bind_development_trial(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not args.campaign_config.is_file():
        raise FileNotFoundError(
            f"Missing frozen biological campaign: {args.campaign_config}"
        )
    payload = common_runner.read_json_without_duplicate_keys(args.campaign_config)
    validate_frozen_campaign(payload)
    _require(args.method in METHODS, f"Unknown method {args.method!r}")
    _require(
        (args.split, args.fold) in DEVELOPMENT_FOLDS,
        "Split/fold is outside the two frozen development jobs",
    )
    _require(
        args.seed == payload["comparison_contract"]["search_seed"],
        "Seed differs from the frozen development search seed",
    )
    requested_reporters = tuple(
        value.strip() for value in args.reporters.split(",") if value.strip()
    )
    _require(requested_reporters == PANEL12, "Reporter panel/order differs from panel12")
    trials = payload["methods"][args.method]["trial_configs"]
    _require(
        0 <= args.trial_index < len(trials),
        "trial-index is outside the frozen four-trial grid",
    )
    trial = copy.deepcopy(trials[args.trial_index])
    selection = payload["comparison_contract"]["selection_objective"]
    primary = selection["primary_development_job"]
    guardrail = selection["guardrail_development_job"]
    if (args.split, args.fold) == (primary["split"], primary["fold"]):
        development_job_role = primary["role"]
        hpo_selection_eligible = True
    elif (args.split, args.fold) == (guardrail["split"], guardrail["fold"]):
        development_job_role = guardrail["role"]
        hpo_selection_eligible = False
    else:  # Already rejected above; kept fail-closed if the config drifts.
        raise BiologicalRunnerError("Development job has no selection role")
    binding = {
        "schema_version": SCHEMA_VERSION,
        "campaign_path": str(args.campaign_config.resolve()),
        "campaign_file_sha256": common_runner.file_sha256(args.campaign_config),
        "campaign_semantic_sha256": common_runner.json_sha256(payload),
        "campaign_id": payload["campaign_id"],
        "method": args.method,
        "trial_index": args.trial_index,
        "trial_sha256": common_runner.json_sha256(trial),
        "split": args.split,
        "fold": args.fold,
        "development_folds": [list(value) for value in DEVELOPMENT_FOLDS],
        "development_job_role": development_job_role,
        "hpo_selection_eligible": hpo_selection_eligible,
        "cross_fold_hpo_aggregation": selection["cross_fold_aggregation"],
        "reporters": list(PANEL12),
        "seed": args.seed,
        "mode": "development",
        "train_only": True,
        "outer_test_label_access_allowed": False,
        "outer_test_identity_metadata_access_allowed": True,
        "outer_test_evaluation_allowed": False,
        "parameter_policy": copy.deepcopy(
            payload["comparison_contract"]["parameter_policy"]
        ),
        "resource_metrics_policy": copy.deepcopy(
            payload["comparison_contract"]["resource_metrics"]
        ),
    }
    return payload, trial, binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--reporters", required=True)
    parser.add_argument(
        "--split",
        required=True,
        choices=tuple(value[0] for value in DEVELOPMENT_FOLDS),
    )
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--trial-index", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--campaign-config", type=Path, default=DEFAULT_CAMPAIGN_CONFIG)
    parser.add_argument("--mode", required=True, choices=("development",))
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    parser.add_argument(
        "--gpu-staging",
        default="required",
        choices=("required",),
    )
    parser.add_argument("--gpu-staging-reserve-gib", type=float, default=16.0)
    parser.add_argument("--gpu-staging-chunk-rows", type=int, default=131072)
    parser.add_argument("--prediction-batch-size", type=int, default=8192)
    parser.add_argument("--throughput-batch-size", type=int, default=8192)
    parser.add_argument("--throughput-warmup-iterations", type=int, default=3)
    parser.add_argument("--throughput-timed-iterations", type=int, default=10)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--tf32", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overwrite", action="store_true")
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
    return parser


def validate_args(args: argparse.Namespace) -> None:
    _require(args.mode == "development", "Only development mode is implemented")
    _require(args.train_only is True, "Development jobs require --train-only")
    _require(args.device == "cuda", "CPU fallback is forbidden")
    _require(args.gpu_staging == "required", "GPU staging must be required")
    _require(
        args.gpu_staging_reserve_gib >= 12.0,
        "At least 12 GiB must remain reserved for model training",
    )
    _require(args.gpu_staging_chunk_rows > 0, "GPU staging chunk must be positive")
    _require(args.prediction_batch_size > 0, "Prediction batch size must be positive")
    _require(args.throughput_batch_size > 0, "Throughput batch size must be positive")
    _require(
        args.throughput_warmup_iterations >= 0,
        "Throughput warmup iterations must be non-negative",
    )
    _require(
        args.throughput_timed_iterations > 0,
        "Throughput timed iterations must be positive",
    )
    _require(not (args.resume and args.overwrite), "--resume and --overwrite conflict")


def validate_trial_execution_policy(
    args: argparse.Namespace, trial: Mapping[str, Any]
) -> None:
    """Bind declared precision to CLI flags before any CUDA/model work."""

    if args.method != "scbutterfly_ops_b":
        return
    policy = str(trial["training"]["precision_policy"])
    if policy == "float32_no_tf32":
        _require(
            args.amp is False and args.tf32 is False,
            "scButterfly source anchor requires --no-amp --no-tf32",
        )
    elif policy == "amp_bfloat16_tf32":
        _require(
            args.amp is True and args.tf32 is True,
            "scButterfly OPS trial requires --amp --tf32",
        )
    else:
        raise BiologicalRunnerError(
            f"Unsupported scButterfly precision policy {policy!r}"
        )


@dataclass(frozen=True)
class GroupedReporterBatch:
    reporter_slug: str
    phase_x: torch.Tensor
    target_y: torch.Tensor
    observed_mask: torch.Tensor
    phase_row_indices: torch.Tensor


def load_reporter_groups(
    *,
    task_batch: Any,
    heads_by_slug: Mapping[str, frozen_engine.HeadPartition],
    phase_cache: Any,
    x_state: Any,
    gpu_staging: frozen_engine.GPUTrainingStaging,
    device: str,
) -> tuple[list[GroupedReporterBatch], dict[str, int], int]:
    """Reuse the frozen candidate loader for one reporter-balanced batch."""

    groups: list[GroupedReporterBatch] = []
    exposure: dict[str, int] = {}
    observed_endpoint_predictions = 0
    for slug, positions in zip(task_batch.task_names, task_batch.task_indices):
        head = heads_by_slug[str(slug)]
        local = head.train_indices[np.asarray(positions, dtype=np.int64)]
        phase_x, target_y, observed_mask = common_runner.load_batch(
            phase_cache,
            x_state,
            head,
            local,
            gpu_staging,
            device,
        )
        if not bool(observed_mask.any(dim=1).all().item()):
            raise BiologicalRunnerError(f"All-missing training row for {slug}")
        groups.append(
            GroupedReporterBatch(
                reporter_slug=str(slug),
                phase_x=phase_x,
                target_y=target_y,
                observed_mask=observed_mask,
                phase_row_indices=torch.as_tensor(
                    np.asarray(head.data.phase_rows[local], dtype=np.int64),
                    dtype=torch.long,
                    device=phase_x.device,
                ),
            )
        )
        exposure[str(slug)] = len(local)
        observed_endpoint_predictions += int(observed_mask.sum().item())
    return groups, exposure, observed_endpoint_predictions


def captain_active_endpoint_query_labels(
    groups: Sequence[GroupedReporterBatch],
    heads_by_slug: Mapping[str, frozen_engine.HeadPartition],
) -> list[str]:
    """Concrete global endpoint-query labels active in one CAPTAIN event."""

    labels = [
        f"{group.reporter_slug}::{endpoint}"
        for group in groups
        for endpoint in heads_by_slug[
            group.reporter_slug
        ].data.target_feature_names.astype(str)
    ]
    _require(len(labels) == len(set(labels)), "CAPTAIN endpoint-query labels repeat")
    return labels


def reporter_macro_loss(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Equal reporter weighting, independent of endpoint dimensionality."""

    if not losses:
        raise ValueError("Reporter-balanced loss requires at least one reporter")
    values = tuple(losses.values())
    if any(value.ndim != 0 for value in values):
        raise ValueError("Each reporter loss must be scalar")
    result = torch.stack(values).mean()
    if not bool(torch.isfinite(result).item()):
        raise FloatingPointError("Reporter-balanced loss is non-finite")
    return result


def make_phase_mge_mask(phase_x: torch.Tensor, mask_rate: float) -> torch.Tensor:
    """Sample a nonempty per-cell CAPTAIN MGE mask on the input device."""

    rate = _probability(mask_rate, "phase mask rate", allow_zero=False)
    mask = torch.rand_like(phase_x) < rate
    empty = ~mask.any(dim=1)
    if bool(empty.any().item()):
        fallback = torch.randint(
            phase_x.shape[1],
            (int(empty.sum().item()),),
            device=phase_x.device,
        )
        rows = torch.nonzero(empty, as_tuple=False).flatten()
        mask[rows, fallback] = True
    return mask


def create_primary_optimizer(
    method: str,
    model: nn.Module,
    optimizer_config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Create only the optimizer pinned by each public method provenance."""

    learning_rate = float(optimizer_config["learning_rate"])
    weight_decay = float(optimizer_config["weight_decay"])
    common = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
        "betas": (0.9, 0.999),
    }
    if method == "ops_captain":
        _require(optimizer_config["name"] == "adam", "CAPTAIN requires Adam")
        epsilon = float(optimizer_config["epsilon"])
        _require(
            epsilon == 1.0e-4,
            "CAPTAIN Adam epsilon must remain 1e-4",
        )
        return torch.optim.Adam(model.parameters(), eps=epsilon, **common)
    if method == "midas_ops":
        _require(optimizer_config["name"] == "adamw", "MIDAS requires AdamW")
        epsilon = float(optimizer_config["epsilon"])
        _require(epsilon == 1.0e-8, "MIDAS AdamW epsilon must remain 1e-8")
        return torch.optim.AdamW(model.parameters(), eps=epsilon, **common)
    raise BiologicalRunnerError(f"No shared primary optimizer for {method!r}")


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = _positive_int(total_steps, "total scheduler steps")
    _require(0 <= warmup_steps <= total_steps, "Invalid warmup step count")

    def multiplier(step_index: int) -> float:
        if warmup_steps and step_index < warmup_steps:
            return max(1.0 / warmup_steps, (step_index + 1) / warmup_steps)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(
            1.0,
            max(0.0, (step_index - warmup_steps) / denominator),
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def create_shared_primary_scheduler(
    method: str,
    optimizer: torch.optim.Optimizer,
    training: Mapping[str, Any],
    *,
    total_steps: int,
    steps_per_epoch: int,
) -> tuple[Any, dict[str, Any]]:
    """Create the per-trial scheduler and an explicit provenance manifest.

    CAPTAIN trial 0 follows the public code's StepLR(gamma=1) shape.  The
    upstream code advances that scheduler on source-file boundaries, which do
    not exist in OPS; one OPS epoch is therefore the declared task-adapter
    clock.  Because gamma is exactly one this remains a fixed learning rate.
    CAPTAIN trials 1--3 are validation-only OPS variants using the common
    warmup/cosine schedule.  MIDAS trial 0 preserves the public code's fixed
    AdamW learning rate; its remaining trials are OPS validation variants.
    """

    total_steps = _positive_int(total_steps, "total scheduler steps")
    steps_per_epoch = _positive_int(steps_per_epoch, "scheduler steps per epoch")
    if method == "ops_captain":
        scheduler_config = training["scheduler"]
        _require(
            isinstance(scheduler_config, Mapping),
            "CAPTAIN scheduler config must be a mapping",
        )
        name = scheduler_config.get("name")
        if name == "fixed_step_lr":
            _exact_keys(
                scheduler_config,
                {"name", "step_size_epochs", "gamma"},
                "CAPTAIN fixed scheduler",
            )
            step_size_epochs = _positive_int(
                scheduler_config["step_size_epochs"],
                "CAPTAIN StepLR epoch interval",
            )
            gamma = _positive_float(
                scheduler_config["gamma"], "CAPTAIN StepLR gamma"
            )
            _require(gamma == 1.0, "CAPTAIN source anchor requires StepLR gamma=1")
            step_size_updates = step_size_epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size_updates,
                gamma=gamma,
            )
            manifest = {
                "name": "fixed_step_lr",
                "declared_step_size_epochs": step_size_epochs,
                "translated_step_size_optimizer_updates": step_size_updates,
                "gamma": gamma,
                "effective_learning_rate_policy": "fixed",
                "source_relationship": (
                    "source-shaped CAPTAIN StepLR(gamma=1), translated from "
                    "upstream source-file boundaries to the OPS epoch clock"
                ),
                "source_shaped": True,
                "upstream_execution_exact": False,
                "scheduler_is_upstream_code_faithful": False,
            }
            return scheduler, manifest
        if name == "common_linear_warmup_cosine_decay":
            _exact_keys(
                scheduler_config,
                {"name", "warmup_epochs"},
                "CAPTAIN common scheduler",
            )
            warmup_epochs = _positive_int(
                scheduler_config["warmup_epochs"], "CAPTAIN warmup epochs"
            )
            warmup_steps = warmup_epochs * steps_per_epoch
            scheduler = cosine_warmup_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
            manifest = {
                "name": "common_linear_warmup_cosine_decay",
                "warmup_epochs": warmup_epochs,
                "warmup_optimizer_updates": warmup_steps,
                "total_optimizer_updates": total_steps,
                "source_relationship": "validation-only OPS HPO variant",
                "source_shaped": False,
                "upstream_execution_exact": False,
                "scheduler_is_upstream_code_faithful": False,
            }
            return scheduler, manifest
        raise BiologicalRunnerError(
            f"Unsupported CAPTAIN scheduler {name!r}"
        )
    if method == "midas_ops":
        scheduler_config = training["scheduler"]
        _require(
            isinstance(scheduler_config, Mapping),
            "MIDAS scheduler config must map",
        )
        name = scheduler_config.get("name")
        if name == "fixed":
            _exact_keys(scheduler_config, {"name"}, "MIDAS fixed scheduler")
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda _step: 1.0
            )
            return scheduler, {
                "name": "fixed",
                "effective_learning_rate_policy": "fixed",
                "source_relationship": "source AdamW fixed-learning-rate anchor",
                "source_shaped": True,
                "upstream_execution_exact": False,
                "scheduler_is_upstream_code_faithful": True,
            }
        if name == "common_linear_warmup_cosine_decay":
            _exact_keys(
                scheduler_config,
                {"name", "warmup_epochs"},
                "MIDAS OPS scheduler",
            )
            warmup_epochs = _positive_int(
                scheduler_config["warmup_epochs"], "MIDAS warmup epochs"
            )
            warmup_steps = warmup_epochs * steps_per_epoch
            scheduler = cosine_warmup_scheduler(
                optimizer,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
            )
            return scheduler, {
                "name": "common_linear_warmup_cosine_decay",
                "warmup_epochs": warmup_epochs,
                "warmup_optimizer_updates": warmup_steps,
                "total_optimizer_updates": total_steps,
                "source_relationship": "validation-only OPS HPO variant",
                "source_shaped": False,
                "upstream_execution_exact": False,
                "scheduler_is_upstream_code_faithful": False,
            }
        raise BiologicalRunnerError(f"Unsupported MIDAS scheduler {name!r}")
    raise BiologicalRunnerError(f"No shared primary scheduler for {method!r}")


def captain_provenance_manifest(
    trial: Mapping[str, Any], source_anchor_contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Describe the CAPTAIN fidelity/adapter boundary without overclaiming."""

    role = str(trial["trial_role"])
    return {
        "trial_role": role,
        "this_trial_is_source_shaped_anchor": role
        == "source_shaped_fixed_anchor",
        "upstream_repository": source_anchor_contract["upstream_repository"],
        "upstream_commit": source_anchor_contract["upstream_commit"],
        "source_files": list(source_anchor_contract["source_files"]),
        "source_shaped_dropout": copy.deepcopy(
            source_anchor_contract["source_shaped_dropout"]
        ),
        "ops_phase_task_adapter": copy.deepcopy(
            source_anchor_contract["ops_phase_task_adapter"]
        ),
        "pretrained_checkpoint_transfer": False,
        "end_to_end_upstream_execution_exact": False,
        "claim_boundary": source_anchor_contract["claim_boundary"],
    }


def method_implementation_manifest(method: str) -> dict[str, dict[str, str]]:
    """Hash the actual runner, factory, core, and training bridge in use."""

    method_files = {
        "ops_captain": {
            "core": "ops_biological_baseline_captain.py",
            "bridge": "ops_biological_baseline_captain_training.py",
        },
        "midas_ops": {
            "core": "ops_biological_baseline_midas_multimodal.py",
            "bridge": "ops_biological_baseline_midas_multimodal_training.py",
        },
        "scbutterfly_ops_b": {
            "core": "ops_biological_baseline_scbutterfly.py",
            "bridge": "ops_biological_baseline_scbutterfly_training.py",
        },
    }
    _require(method in method_files, f"Unknown implementation method {method!r}")
    paths = {
        "runner": Path(__file__).resolve(),
        "factory": Path(biological_models.__file__).resolve(),
        "common_runner": Path(common_runner.__file__).resolve(),
        "frozen_engine": Path(frozen_engine.__file__).resolve(),
        "sampler_library": Path(sampler_library.__file__).resolve(),
        "training_harness_library": Path(harness.__file__).resolve(),
        "runtime_metrics": Path(runtime_metrics.__file__).resolve(),
        **{
            role: (PROJECT_ROOT / "scripts" / filename).resolve()
            for role, filename in method_files[method].items()
        },
    }
    result: dict[str, dict[str, str]] = {}
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {method} {role} implementation: {path}")
        result[role] = {
            "path": str(path),
            "sha256": common_runner.file_sha256(path),
        }
    return result


def bind_training_identity(
    path: Path,
    identity: Mapping[str, Any],
    *,
    resume_payload: Mapping[str, Any] | None,
) -> str:
    """Write a fresh identity or verify an existing resume without mutation."""

    identity_sha = common_runner.json_sha256(identity)
    expected = json.loads(
        common_runner.canonical_json(
            {**copy.deepcopy(dict(identity)), "identity_sha256": identity_sha}
        )
    )
    if resume_payload is None:
        _require(
            not path.exists(),
            "Fresh training identity path already exists",
        )
        atomic_json(path, expected)
        return identity_sha
    _require(
        path.is_file(),
        "Resume checkpoint exists without training_identity.json",
    )
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BiologicalRunnerError(
            "Existing resume training identity is unreadable"
        ) from error
    _require(
        existing == expected,
        "Existing training identity differs from this frozen job",
    )
    _require(
        resume_payload.get("identity_sha256") == identity_sha,
        "Resume checkpoint identity differs from training_identity.json",
    )
    return identity_sha


def biological_checkpoint_payload(
    *,
    model: nn.Module,
    model_manifest: Mapping[str, Any],
    method: str,
    optimizer_update: int,
    kind: str,
    stage_updates: Mapping[str, int],
    model_state: Mapping[str, torch.Tensor] | None = None,
    source_updates: Sequence[int] | None = None,
) -> dict[str, Any]:
    state = (
        common_runner.tensor_state_dict(model)
        if model_state is None
        else {name: value.detach().cpu().clone() for name, value in model_state.items()}
    )
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "method": method,
        "kind": kind,
        "optimizer_update": int(optimizer_update),
        "stage_updates": {key: int(value) for key, value in stage_updates.items()},
        "model_manifest": copy.deepcopy(model_manifest),
        "model_state": state,
        "read_only_evaluation_state": kind in {"single", "ema", "checkpoint_average"},
        "outer_test_selection_allowed": False,
    }
    if source_updates is not None:
        payload["source_checkpoint_updates"] = [int(value) for value in source_updates]
    return payload


@dataclass(frozen=True)
class PrimaryForwardResult:
    total_loss: torch.Tensor
    reporter_diagnostics: OrderedDict[str, torch.Tensor]
    loss_components: dict[str, torch.Tensor]


def create_harness_collection(
    method: str,
    trial: Mapping[str, Any],
    heads: Sequence[Any],
    *,
    device: str | torch.device,
) -> biological_models.HarnessModelCollection:
    """Bind a method-native collection to the frozen registry and panel12."""

    dimensions = OrderedDict(
        (str(head.slug), int(head.data.y.shape[1])) for head in heads
    )
    _require(tuple(dimensions) == PANEL12, "Head order differs from panel12")
    registry = biological_models.full_reporter_registry()
    _require(
        dimensions == OrderedDict((name, registry[name]) for name in PANEL12),
        "Data endpoint dimensions differ from the frozen 52-reporter registry",
    )
    factory = biological_models.HarnessModelConfig(
        method=method,
        active_reporters=PANEL12,
        options=trial["model"],
    )
    model = biological_models.create_harness_model(factory).to(device)
    _require(
        tuple(model.reporter_names) == PANEL12,
        "Harness collection reporter order differs from panel12",
    )
    return model


def forward_shared_primary(
    method: str,
    model: biological_models.HarnessModelCollection,
    groups: Sequence[GroupedReporterBatch],
    *,
    phase_mask_rate: float | None = None,
    sample: bool = True,
) -> PrimaryForwardResult:
    """Run one shared reporter-balanced CAPTAIN or MIDAS optimizer batch."""

    if not groups:
        raise ValueError("Primary forward requires at least one reporter group")
    if len({group.reporter_slug for group in groups}) != len(groups):
        raise ValueError("Primary forward received duplicate reporter groups")
    if method == "ops_captain":
        if not isinstance(model, biological_models.CaptainHarnessCollection):
            raise TypeError("CAPTAIN method requires CaptainHarnessCollection")
        if phase_mask_rate is None:
            raise ValueError("CAPTAIN training requires a phase mask rate")
        batches = [
            CaptainReporterTrainingBatch(
                reporter=group.reporter_slug,
                phase_x=group.phase_x,
                endpoint_target=group.target_y,
                endpoint_observed_mask=group.observed_mask,
                phase_mge_mask=make_phase_mge_mask(
                    group.phase_x, phase_mask_rate
                ),
            )
            for group in groups
        ]
        rich = model.forward_grouped(batches)
        reporter_losses = OrderedDict(
            (name, result.reporter_loss) for name, result in rich.items()
        )
        total = reporter_macro_loss(reporter_losses)
        components = {
            "reporter_macro_total": total,
            "mean_mse_macro": torch.stack(
                tuple(value.loss_components.mean_mse for value in rich.values())
            ).mean(),
            "quantile_pinball_macro": torch.stack(
                tuple(
                    value.loss_components.quantile_pinball
                    for value in rich.values()
                )
            ).mean(),
            "phase_reconstruction_mse_macro": torch.stack(
                tuple(
                    value.loss_components.phase_reconstruction_mse
                    for value in rich.values()
                )
            ).mean(),
        }
        return PrimaryForwardResult(total, reporter_losses, components)
    if method == "midas_ops":
        if not isinstance(model, biological_models.MIDASHarnessCollection):
            raise TypeError("MIDAS method requires MIDASHarnessCollection")
        if phase_mask_rate is not None:
            raise ValueError("MIDAS does not use a CAPTAIN phase MGE mask")
        batches = [
            biological_models.MIDASReporterTrainingBatch(
                reporter=group.reporter_slug,
                phase_x=group.phase_x,
                endpoint_target=group.target_y,
                endpoint_observed_mask=group.observed_mask,
                phase_row_indices=group.phase_row_indices,
            )
            for group in groups
        ]
        rich = model.forward_grouped(batches, sample=sample)
        diagnostics: OrderedDict[str, torch.Tensor] = OrderedDict()
        by_name = {group.reporter_slug: group for group in groups}
        for name, prediction in rich.local_predictions.items():
            group = by_name[name]
            safe_target = torch.where(
                group.observed_mask,
                group.target_y,
                torch.zeros_like(group.target_y),
            )
            safe_prediction = torch.where(
                group.observed_mask,
                prediction,
                torch.zeros_like(prediction),
            )
            diagnostics[name] = (
                (safe_prediction - safe_target).square().sum()
                / group.observed_mask.sum().to(safe_prediction.dtype)
            )
        if not bool(torch.isfinite(rich.total_loss).item()):
            raise FloatingPointError(
                "MIDAS native multimodal objective is non-finite"
            )
        return PrimaryForwardResult(
            total_loss=rich.total_loss,
            reporter_diagnostics=diagnostics,
            loss_components=dict(rich.loss_components),
        )
    raise BiologicalRunnerError(f"Method {method!r} has no shared primary loop")


def synthetic_shared_optimizer_step(
    method: str,
    model: biological_models.HarnessModelCollection,
    groups: Sequence[GroupedReporterBatch],
    optimizer: torch.optim.Optimizer,
    *,
    phase_mask_rate: float | None = None,
) -> PrimaryForwardResult:
    """One device-agnostic step used by focused unit tests and smoke checks."""

    model.train()
    optimizer.zero_grad(set_to_none=True)
    result = forward_shared_primary(
        method,
        model,
        groups,
        phase_mask_rate=phase_mask_rate,
        sample=True,
    )
    result.total_loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(
        bool(torch.isfinite(value).all().item()) for value in gradients
    ):
        raise FloatingPointError("Primary optimizer step produced invalid gradients")
    optimizer.step()
    return result


def backward_captain_reporter_accumulation(
    model: biological_models.CaptainHarnessCollection,
    groups: Sequence[GroupedReporterBatch],
    *,
    phase_mask_rate: float,
    backward: Callable[[torch.Tensor], None] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Backward CAPTAIN one reporter graph at a time, then let caller step once.

    Dividing every local loss by ``len(groups)`` is algebraically identical to
    the reporter-macro grouped objective.  Immediate backward releases each
    172-token cross-attention graph before the next reporter forward.
    """

    if not groups:
        raise ValueError("CAPTAIN accumulation requires reporter groups")
    apply_backward = backward or (lambda value: value.backward())
    diagnostics: OrderedDict[str, torch.Tensor] = OrderedDict()
    for group in groups:
        result = forward_shared_primary(
            "ops_captain",
            model,
            [group],
            phase_mask_rate=phase_mask_rate,
        )
        apply_backward(result.total_loss / len(groups))
        # Do not retain the just-backpropagated cross-attention graph through
        # diagnostics.  CAPTAIN's production path intentionally has only one
        # reporter graph resident at a time.
        diagnostics.update(
            (name, value.detach())
            for name, value in result.reporter_diagnostics.items()
        )
        del result
    return diagnostics


def predict_validation_reporter(
    model: biological_models.HarnessModelCollection,
    *,
    phase_cache: Any,
    x_state: Any,
    head: frozen_engine.HeadPartition,
    gpu_staging: frozen_engine.GPUTrainingStaging,
    device: str,
    batch_size: int,
    amp: bool,
) -> float:
    """Candidate-harness validation cohort with method-native prediction."""

    total_squared = 0.0
    total_values = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(head.validation_indices), batch_size):
            local = head.validation_indices[start : start + batch_size]
            phase_x, target_y, observed_mask = common_runner.load_batch(
                phase_cache,
                x_state,
                head,
                local,
                gpu_staging,
                device,
            )
            if not bool(observed_mask.all().item()):
                raise BiologicalRunnerError(
                    "Frozen validation cohort contains missing endpoints"
                )
            with common_runner.autocast_cuda(amp):
                prediction = model.predict_reporter(phase_x, head.slug)
            difference = prediction.float() - target_y.float()
            total_squared += float(difference.square().sum().cpu())
            total_values += int(difference.numel())
    _require(total_values > 0, f"Empty validation cohort for {head.slug}")
    result = total_squared / total_values
    _require(
        math.isfinite(result) and result >= 0.0,
        f"Invalid validation MSE for {head.slug}",
    )
    return result


def evaluate_validation_state_candidates(
    args: argparse.Namespace,
    *,
    model: biological_models.HarnessModelCollection,
    state_paths: Mapping[str, Path],
    objective: harness.ReporterNormalizedCheckpointObjective,
    phase_cache: Any,
    x_state: Any,
    heads: Sequence[frozen_engine.HeadPartition],
    gpu_staging: frozen_engine.GPUTrainingStaging,
) -> dict[str, Any]:
    """Compare frozen read-only states on validation, never select among them."""

    expected_states = {"single", "ema", "checkpoint_average"}
    _require(
        set(state_paths) == expected_states,
        "Validation state comparison requires single, EMA and average",
    )
    state_rows: dict[str, dict[str, Any]] = {}
    for state_name in ("single", "ema", "checkpoint_average"):
        path = state_paths[state_name]
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(
            payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
            f"Validation state schema changed for {state_name}",
        )
        model.load_state_dict(payload["model_state"], strict=True)
        model.to("cuda")
        validation = {
            head.slug: predict_validation_reporter(
                model,
                phase_cache=phase_cache,
                x_state=x_state,
                head=head,
                gpu_staging=gpu_staging,
                device="cuda",
                batch_size=args.prediction_batch_size,
                amp=args.amp,
            )
            for head in heads
        }
        state_rows[state_name] = {
            "state": state_name,
            "payload_kind": payload["kind"],
            "optimizer_update": int(payload["optimizer_update"]),
            "path": str(path.resolve()),
            "sha256": common_runner.file_sha256(path),
            "validation_mse_by_reporter": validation,
            "validation_macro_mse": math.fsum(validation.values())
            / len(validation),
            "validation_reporter_normalized_mse": objective.score(validation),
            "used_for_checkpoint_selection": False,
            "used_for_outer_test_selection": False,
        }
    # Leave the model in the already selected single-checkpoint state.  The
    # comparison above is diagnostic only and cannot alter state selection.
    selected_single = torch.load(
        state_paths["single"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(selected_single["model_state"], strict=True)
    model.to("cuda")
    return {
        "schema_version": SCHEMA_VERSION,
        "partition": "validation",
        "selection_inputs": [],
        "state_selection_performed": False,
        "outer_test_accessed": False,
        "states": state_rows,
    }


def assert_development_heads_sealed(
    heads: Sequence[frozen_engine.HeadPartition],
) -> None:
    """Fail if any materialised development head contains outer-test labels."""

    for head in heads:
        _require(
            len(head.complete_test_indices) == 0
            and len(head.excluded_partial_test_indices) == 0,
            f"Development head materialised test labels for {head.slug}",
        )
        _require(
            head.original_partition_counts.get("outer_test_y_physically_loaded")
            is False,
            f"Outer-test Y entered the training partition for {head.slug}",
        )


def _identity_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(tuple(array.shape)).encode("utf-8"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def verify_identity_only_outer_test_cohorts(
    args: argparse.Namespace,
    phase_cache: Any,
    heads: Sequence[frozen_engine.HeadPartition],
) -> dict[str, dict[str, Any]]:
    """Verify frozen outer cohorts without opening fluorescence datasets."""

    fold_map = phase_cache.folds[args.split]
    result: dict[str, dict[str, Any]] = {}
    for head in heads:
        slug = head.slug
        cache_path = common_runner.exact_cache_path(args.exact_root, slug)
        with h5py.File(cache_path, "r") as source:
            _require(
                "phase_row_index" in source
                and "metadata/is_control" in source,
                f"Exact cache lacks identity-only metadata for {slug}",
            )
            phase_rows = np.asarray(source["phase_row_index"][:], dtype=np.int64)
            controls = np.asarray(source["metadata/is_control"][:], dtype=bool)
        _require(
            len(phase_rows) == len(controls),
            f"Exact identity arrays differ in length for {slug}",
        )
        fold_values = np.asarray(fold_map[phase_rows], dtype=np.uint8)
        outer = fold_values == int(args.fold)
        observed_rows = phase_rows[outer]
        observed_controls = controls[outer]
        reference_rows, reference_controls, reference_root = (
            common_runner.frozen_test_reference(args, slug)
        )
        _require(
            len(reference_rows) == len(reference_controls),
            f"Frozen reference identity arrays differ in length for {slug}",
        )
        _require(
            len(np.unique(reference_rows)) == len(reference_rows),
            f"Frozen reference phase rows repeat for {slug}",
        )
        order = np.argsort(observed_rows, kind="mergesort")
        sorted_rows = observed_rows[order]
        positions = np.searchsorted(sorted_rows, reference_rows)
        in_bounds = positions < len(sorted_rows)
        matched = np.zeros(len(reference_rows), dtype=bool)
        matched[in_bounds] = (
            sorted_rows[positions[in_bounds]] == reference_rows[in_bounds]
        )
        _require(
            bool(matched.all()),
            f"Frozen reference phase row is absent from exact outer identity for {slug}",
        )
        # Every frozen complete-core reference cell must map to exactly one
        # exact identity row.  Extra outer-fold identity rows are expected:
        # they are cells excluded later by complete-core/technical rules.
        left = np.searchsorted(sorted_rows, reference_rows, side="left")
        right = np.searchsorted(sorted_rows, reference_rows, side="right")
        _require(
            bool(np.all((right - left) == 1)),
            f"Frozen reference phase row is not unique in exact outer identity for {slug}",
        )
        matched_source_rows = order[positions]
        matched_controls = observed_controls[matched_source_rows]
        _require(
            np.array_equal(matched_controls, reference_controls),
            f"Identity-only outer control mask changed for {slug}",
        )
        n_outer_identity_all = int(len(observed_rows))
        n_reference_complete_core = int(len(reference_rows))
        n_identity_only_excluded = (
            n_outer_identity_all - n_reference_complete_core
        )
        _require(
            n_identity_only_excluded >= 0,
            f"Frozen complete-core reference exceeds exact outer identity for {slug}",
        )
        result[slug] = {
            "strict_specialist_comparable": True,
            "identity_only_outer_verification": "PASS",
            "reference": str(reference_root.resolve()),
            "n_outer_identity_all": n_outer_identity_all,
            "n_reference_complete_core": n_reference_complete_core,
            "n_identity_only_excluded": n_identity_only_excluded,
            "reference_phase_row_index_sha256": _identity_array_sha256(
                reference_rows
            ),
            "reference_is_control_sha256": _identity_array_sha256(
                reference_controls
            ),
            "outer_identity_all_phase_row_index_sha256": (
                _identity_array_sha256(observed_rows)
            ),
            "outer_identity_all_is_control_sha256": _identity_array_sha256(
                observed_controls
            ),
            "outer_test_y_physically_loaded": False,
            "fluorescence_dataset_accessed": False,
        }
    return result


class ValidationStateManager:
    """Persist single/EMA/average candidates without mutating training state."""

    def __init__(
        self,
        output_root: Path,
        *,
        method: str,
        model_manifest: Mapping[str, Any],
        stage_updates: Mapping[str, int],
    ) -> None:
        self.output_root = output_root
        self.method = method
        self.model_manifest = copy.deepcopy(model_manifest)
        self.checkpoint_dir = output_root / "checkpoints"
        self.state_dir = output_root / "evaluation_state_candidates"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_hashes: dict[int, str] = {}
        self.stage_updates = {key: int(value) for key, value in stage_updates.items()}

    def checkpoint_path(self, update: int) -> Path:
        return self.checkpoint_dir / f"update_{int(update):08d}.pt"

    def ema_path(self, update: int) -> Path:
        return self.state_dir / f"ema_update_{int(update):08d}.pt"

    def average_path(self, update: int) -> Path:
        return self.state_dir / f"average_update_{int(update):08d}.pt"

    def save_initial(self, model: nn.Module) -> None:
        path = self.checkpoint_path(0)
        if not path.is_file():
            common_runner.atomic_torch_save(
                biological_checkpoint_payload(
                    model=model,
                    model_manifest=self.model_manifest,
                    method=self.method,
                    optimizer_update=0,
                    kind="initial",
                    stage_updates=self.stage_updates,
                ),
                path,
            )
        self.checkpoint_hashes[0] = common_runner.file_sha256(path)

    def restore_inventory(
        self, trajectory: harness.ValidationTrajectory
    ) -> None:
        initial = self.checkpoint_path(0)
        if not initial.is_file():
            raise FileNotFoundError("Resume is missing the initialization checkpoint")
        self.checkpoint_hashes = {0: common_runner.file_sha256(initial)}
        self.checkpoint_hashes.update(
            {
                int(row["optimizer_update"]): str(row["checkpoint_sha256"])
                for row in trajectory.records
            }
        )
        required_updates = {0}
        recorded_updates = [
            int(row["optimizer_update"]) for row in trajectory.records
        ]
        required_updates.update(recorded_updates[-3:])
        required_updates.update(
            common_runner.eligible_updates(
                trajectory.records,
                trajectory.objective.tie_relative_tolerance,
            )
        )
        if self.method == "scbutterfly_ops_b":
            required_updates.update(recorded_updates)
        for update in sorted(required_updates):
            self._verified_checkpoint_path(update)

    def _verified_checkpoint_path(self, update: int) -> Path:
        value = int(update)
        _require(
            value in self.checkpoint_hashes,
            f"Checkpoint update {value} has no registered SHA256",
        )
        path = self.checkpoint_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Registered checkpoint is missing: {path}")
        observed = common_runner.file_sha256(path)
        _require(
            observed == self.checkpoint_hashes[value],
            f"Checkpoint SHA256 changed for update {value}",
        )
        return path

    def _write_checkpoint_average(
        self,
        *,
        model: nn.Module,
        optimizer_update: int,
        source_updates: Sequence[int],
    ) -> Path:
        """Materialise a read-only average from at least two trained states."""

        update = int(optimizer_update)
        sources = tuple(sorted({int(value) for value in source_updates}))
        _require(
            len(sources) >= 2,
            "Checkpoint averaging requires at least two trained states",
        )
        _require(
            all(value > 0 for value in sources),
            "Checkpoint averaging may not use the update-0 initialization",
        )
        source_states = []
        for source_update in sources:
            source_path = self._verified_checkpoint_path(source_update)
            source = torch.load(
                source_path,
                map_location="cpu",
                weights_only=False,
            )
            source_states.append(source["model_state"])
        averaged = common_runner.average_state_dicts(source_states)
        average_payload = biological_checkpoint_payload(
            model=model,
            model_manifest=self.model_manifest,
            method=self.method,
            optimizer_update=update,
            kind="checkpoint_average",
            stage_updates=self.stage_updates,
            model_state=averaged,
            source_updates=sources,
        )
        average_payload["source_checkpoint_sha256"] = [
            self.checkpoint_hashes[value] for value in sources
        ]
        path = self.average_path(update)
        common_runner.atomic_torch_save(average_payload, path)
        return path

    def capture(
        self,
        *,
        model: nn.Module,
        ema_state: Mapping[str, torch.Tensor],
        optimizer_update: int,
        validation: Mapping[str, float],
        trajectory: harness.ValidationTrajectory,
        tie_relative_tolerance: float,
        stage_updates: Mapping[str, int],
    ) -> dict[str, Any]:
        update = int(optimizer_update)
        self.stage_updates = {key: int(value) for key, value in stage_updates.items()}
        current = common_runner.tensor_state_dict(model)
        single_path = self.checkpoint_path(update)
        common_runner.atomic_torch_save(
            biological_checkpoint_payload(
                model=model,
                model_manifest=self.model_manifest,
                method=self.method,
                optimizer_update=update,
                kind="single",
                stage_updates=self.stage_updates,
                model_state=current,
            ),
            single_path,
        )
        digest = common_runner.file_sha256(single_path)
        self.checkpoint_hashes[update] = digest
        trajectory.append(update, validation, digest)
        eligible = common_runner.eligible_updates(
            trajectory.records, tie_relative_tolerance
        )
        if update in eligible:
            common_runner.atomic_torch_save(
                biological_checkpoint_payload(
                    model=model,
                    model_manifest=self.model_manifest,
                    method=self.method,
                    optimizer_update=update,
                    kind="ema",
                    stage_updates=self.stage_updates,
                    model_state=ema_state,
                ),
                self.ema_path(update),
            )
            trained_updates = sorted(
                value
                for value in self.checkpoint_hashes
                if 0 < value <= update
            )
            sources = trained_updates[-3:]
            # Epoch/update one has only one trained checkpoint.  Defer its
            # diagnostic average until finalization, when a second trained
            # state exists; update zero is never treated as a trained source.
            if len(sources) >= 2:
                self._write_checkpoint_average(
                    model=model,
                    optimizer_update=update,
                    source_updates=sources,
                )
        protected = eligible | set(sorted(self.checkpoint_hashes)[-3:]) | {0}
        if self.method == "scbutterfly_ops_b":
            # Independent specialists need their own validation-selected
            # source epochs.  Retain all joint-stage validation checkpoints
            # until the strict-loadable composite is materialised.
            protected.update(
                int(record["optimizer_update"]) for record in trajectory.records
            )
        for eligible_update in eligible:
            average_path = self.average_path(eligible_update)
            if average_path.is_file():
                average_payload = torch.load(
                    average_path, map_location="cpu", weights_only=False
                )
                protected.update(
                    int(value)
                    for value in average_payload.get(
                        "source_checkpoint_updates", []
                    )
                )
        for path in self.checkpoint_dir.glob("update_*.pt"):
            path_update = int(path.stem.split("_")[-1])
            if path_update not in protected:
                path.unlink(missing_ok=True)
        for prefix in ("ema_update_", "average_update_"):
            for path in self.state_dir.glob(f"{prefix}*.pt"):
                path_update = int(path.stem.split("_")[-1])
                if path_update not in eligible:
                    path.unlink(missing_ok=True)
        return copy.deepcopy(trajectory.records[-1])

    def selected_registry(
        self,
        trajectory: harness.ValidationTrajectory,
        *,
        ema_decay: float,
        model: nn.Module,
    ) -> tuple[harness.CheckpointSelection, harness.EvaluationStateRegistry, dict[str, Path]]:
        selection = trajectory.select_checkpoint()
        update = selection.optimizer_update
        paths = {
            "single": self.checkpoint_path(update),
            "ema": self.ema_path(update),
            "checkpoint_average": self.average_path(update),
        }
        for path in (paths["single"], paths["ema"]):
            if not path.is_file():
                raise FileNotFoundError(f"Selected read-only state is missing: {path}")
        if common_runner.file_sha256(paths["single"]) != selection.checkpoint_sha256:
            raise BiologicalRunnerError("Selected single checkpoint hash changed")
        if not paths["checkpoint_average"].is_file():
            available_trained_updates = []
            for candidate_update in sorted(self.checkpoint_hashes):
                if candidate_update <= 0:
                    continue
                if self.checkpoint_path(candidate_update).is_file():
                    self._verified_checkpoint_path(candidate_update)
                    available_trained_updates.append(candidate_update)
            _require(
                update in available_trained_updates,
                "Selected checkpoint is absent from retained trained states",
            )
            other_updates = [
                value for value in available_trained_updates if value != update
            ]
            source_updates = tuple(sorted({update, *other_updates[-2:]}))
            _require(
                len(source_updates) >= 2,
                "Final checkpoint average needs a second retained trained state",
            )
            self._write_checkpoint_average(
                model=model,
                optimizer_update=update,
                source_updates=source_updates,
            )
        average = torch.load(
            paths["checkpoint_average"],
            map_location="cpu",
            weights_only=False,
        )
        source_updates = tuple(
            int(value) for value in average["source_checkpoint_updates"]
        )
        _require(
            len(source_updates) >= 2
            and 0 not in source_updates
            and update in source_updates,
            "Selected checkpoint average violates the trained-source invariant",
        )
        _require(
            all(value in self.checkpoint_hashes for value in source_updates),
            "Checkpoint-average source is absent from the registered inventory",
        )
        _require(
            list(average.get("source_checkpoint_sha256", ()))
            == [self.checkpoint_hashes[value] for value in source_updates],
            "Checkpoint-average source SHA256 registry changed",
        )
        for source_update in source_updates:
            self._verified_checkpoint_path(source_update)
        registry = harness.EvaluationStateRegistry(
            selection, self.checkpoint_hashes
        )
        registry.add_single()
        registry.add_ema(
            "ema",
            common_runner.file_sha256(paths["ema"]),
            float(ema_decay),
        )
        registry.add_checkpoint_average(
            f"checkpoint_average_{len(source_updates)}",
            common_runner.file_sha256(paths["checkpoint_average"]),
            source_updates,
            allow_post_selection_sources=any(
                value > update for value in source_updates
            ),
        )
        return selection, registry, paths


def measure_selected_inference(
    args: argparse.Namespace,
    *,
    model: biological_models.HarnessModelCollection,
    checkpoint_path: Path,
    phase_cache: Any,
    x_state: Any,
    heads: Sequence[frozen_engine.HeadPartition],
    gpu_staging: frozen_engine.GPUTrainingStaging,
    checkpoint_already_loaded: bool = False,
) -> list[dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(
        payload.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        "Selected checkpoint schema changed",
    )
    if not checkpoint_already_loaded:
        model.load_state_dict(payload["model_state"], strict=True)
    model.to("cuda").eval()
    checkpoint_sha = common_runner.file_sha256(checkpoint_path)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for head in heads:
            batch_size = min(
                int(args.throughput_batch_size),
                int(len(head.validation_indices)),
            )
            _require(batch_size > 0, f"No throughput cohort for {head.slug}")
            local = head.validation_indices[:batch_size]
            phase_x, _, _ = common_runner.load_batch(
                phase_cache,
                x_state,
                head,
                local,
                gpu_staging,
                "cuda",
            )

            def run_once() -> torch.Tensor:
                with common_runner.autocast_cuda(args.amp):
                    return model.predict_reporter(phase_x, head.slug)

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
                    "checkpoint_sha256": checkpoint_sha,
                    "amp_dtype": "bfloat16" if args.amp else "float32",
                }
            )
            rows.append(row)
    return rows


def finalize_development_training(
    args: argparse.Namespace,
    *,
    campaign_payload: Mapping[str, Any],
    trial: Mapping[str, Any],
    binding: Mapping[str, Any],
    model: biological_models.HarnessModelCollection,
    model_manifest: Mapping[str, Any],
    split_contract: harness.FrozenSplitContract,
    budget: harness.OptimizerUpdateBudget,
    ledger: harness.ExposureLedger,
    objective: harness.ReporterNormalizedCheckpointObjective,
    trajectory: harness.ValidationTrajectory,
    state_manager: ValidationStateManager,
    ema_decay: float,
    stage_updates: Mapping[str, int],
    history: Sequence[Mapping[str, Any]],
    parameter_inventory: Mapping[str, Any],
    resource_meter: runtime_metrics.CudaTrainingResourceMeter,
    active_signature_ledger: runtime_metrics.CompactActiveSignatureLedger,
    phase_cache: Any,
    x_state: Any,
    heads: Sequence[frozen_engine.HeadPartition],
    gpu_staging: frozen_engine.GPUTrainingStaging,
    identity_sha256: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    ledger.finalize()
    selection, registry, state_paths = state_manager.selected_registry(
        trajectory, ema_decay=ema_decay, model=model
    )
    validation_state_comparison = evaluate_validation_state_candidates(
        args,
        model=model,
        state_paths=state_paths,
        objective=objective,
        phase_cache=phase_cache,
        x_state=x_state,
        heads=heads,
        gpu_staging=gpu_staging,
    )
    validation_state_comparison_path = (
        args.output_root / "validation_state_comparison.json"
    )
    atomic_json(
        validation_state_comparison_path, validation_state_comparison
    )
    global_state_paths = dict(state_paths)
    specialist_composite_selection: dict[str, Any] | None = None
    primary_single_path = state_paths["single"]
    if args.method == "scbutterfly_ops_b":
        specialist_composite_selection = (
            common_runner.select_specialist_updates_by_reporter(
                trajectory.records, objective
            )
        )
        reporter_selected_updates = {
            reporter: int(row["optimizer_update"])
            for reporter, row in specialist_composite_selection[
                "reporter_selections"
            ].items()
        }
        source_states: dict[int, Mapping[str, torch.Tensor]] = {}
        for update in sorted(set(reporter_selected_updates.values())):
            source_path = state_manager._verified_checkpoint_path(update)
            source = torch.load(
                source_path, map_location="cpu", weights_only=False
            )
            _require(
                source.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
                f"scButterfly composite source schema changed: {source_path}",
            )
            source_states[update] = source["model_state"]
        reference_state = common_runner.tensor_state_dict(model)
        reporter_prefixes = {
            head.slug: f"specialists.{head.slug}." for head in heads
        }
        composite_state = common_runner.compose_specialist_state_dict(
            reference_state=reference_state,
            source_states_by_update=source_states,
            reporter_selected_updates=reporter_selected_updates,
            reporter_prefixes=reporter_prefixes,
        )
        primary_single_path = (
            state_manager.state_dir / "specialist_composite_single.pt"
        )
        composite_payload = biological_checkpoint_payload(
            model=model,
            model_manifest=model_manifest,
            method=args.method,
            optimizer_update=max(reporter_selected_updates.values()),
            kind="specialist_composite_single",
            stage_updates=stage_updates,
            model_state=composite_state,
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
        common_runner.atomic_torch_save(composite_payload, primary_single_path)
        # Fail closed on prefix or tensor-shape drift before outer evaluation.
        model.load_state_dict(composite_state, strict=True)
        state_paths = {
            "single": primary_single_path,
            "ema": global_state_paths["ema"],
            "checkpoint_average": global_state_paths["checkpoint_average"],
        }
    training_resources = resource_meter.to_manifest()
    active_manifest = active_signature_ledger.to_manifest()
    throughput = measure_selected_inference(
        args,
        model=model,
        checkpoint_path=primary_single_path,
        phase_cache=phase_cache,
        x_state=x_state,
        heads=heads,
        gpu_staging=gpu_staging,
        checkpoint_already_loaded=True,
    )
    runtime_manifest = runtime_metrics.build_runtime_metrics_manifest(
        parameter_inventory=parameter_inventory,
        training_resources=training_resources,
        inference_throughput=throughput,
        active_signature_ledger=active_manifest,
    )
    if args.method == "scbutterfly_ops_b":
        observations = int(trial["training"]["observations_per_head"])
        reporter_count = len(heads)
        phase_events = int(stage_updates["phase_pretrain"])
        phenotype_events = int(stage_updates["phenotype_pretrain"])
        discriminator_events = int(stage_updates["joint_discriminator"])
        generator_events = int(stage_updates["primary_joint_generator"])
        _require(
            discriminator_events == generator_events,
            "scButterfly final D/G event counts are not paired",
        )
        for name, value in (
            ("phase_pretrain", phase_events),
            ("phenotype_pretrain", phenotype_events),
            ("joint_discriminator", discriminator_events),
            ("joint_generator", generator_events),
        ):
            _require(
                value % reporter_count == 0,
                f"scButterfly {name} events are not reporter-balanced",
            )
        per_reporter_target_batch_events = (
            phenotype_events + generator_events
        ) // reporter_count
        event_accounting = {
            "accounting_unit": "method_native_controller_event",
            "stage_events": {
                "phase_pretrain": phase_events,
                "phenotype_pretrain": phenotype_events,
                "joint_discriminator": discriminator_events,
                "joint_generator": generator_events,
                "paired_joint_discriminator_generator": generator_events,
            },
            "per_reporter_stage_events": {
                "phase_pretrain": phase_events // reporter_count,
                "phenotype_pretrain": phenotype_events // reporter_count,
                "joint_discriminator": discriminator_events // reporter_count,
                "joint_generator": generator_events // reporter_count,
            },
            "owner_optimizer_step_calls": {
                "phase_pretrain": phase_events * 3,
                "phenotype_pretrain": phenotype_events * 3,
                "joint_discriminator": discriminator_events * 2,
                "joint_generator": generator_events * 5,
                "total": (
                    phase_events * 3
                    + phenotype_events * 3
                    + discriminator_events * 2
                    + generator_events * 5
                ),
            },
            "target_bearing_exposure": {
                "definition": (
                    "phenotype-pretrain batches plus paired joint D/G batches; "
                    "a reused paired batch is counted once"
                ),
                "per_reporter_batch_events": per_reporter_target_batch_events,
                "per_reporter_target_vectors": (
                    per_reporter_target_batch_events * observations
                ),
                "multiplier_vs_100_epoch_standard_primary_exposure": 3.0,
            },
            "target_consuming_forward_events": (
                phenotype_events + discriminator_events + generator_events
            ),
            "cross_method_optimizer_step_calls_equalized": False,
        }
    else:
        primary_events = int(stage_updates["primary_joint_generator"])
        event_accounting = {
            "accounting_unit": "shared_primary_event",
            "stage_events": {"primary_joint": primary_events},
            "owner_optimizer_step_calls": {
                "primary_joint": primary_events,
                "total": primary_events,
            },
            "target_bearing_exposure": {
                "definition": "primary observed target-vector batches",
                "per_reporter_target_vectors": dict(ledger.example_exposures),
            },
            "cross_method_optimizer_step_calls_equalized": False,
        }
    parameter_path = args.output_root / "parameter_inventory.json"
    active_path = args.output_root / "active_parameter_profiles.json"
    throughput_path = args.output_root / "inference_throughput.csv"
    runtime_path = args.output_root / "runtime_metrics.json"
    atomic_json(parameter_path, dict(parameter_inventory))
    atomic_json(active_path, active_manifest)
    pd.DataFrame(throughput).to_csv(throughput_path, index=False)
    atomic_json(runtime_path, runtime_manifest)
    complete = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "mode": "development",
        "train_only": True,
        "method_id": args.method,
        "split": args.split,
        "fold": args.fold,
        "trial_index": args.trial_index,
        "seed": args.seed,
        "device_resolved": "cuda",
        "reporters": list(PANEL12),
        "model_manifest": copy.deepcopy(model_manifest),
        "factory_config": model.factory_config.to_manifest(),
        "trial_config": copy.deepcopy(trial),
        "campaign_binding": copy.deepcopy(binding),
        "identity_sha256": identity_sha256,
        "runtime_seconds": float(runtime_seconds),
        "label_policy": "complete_core_explicit_observed_mask",
        "outer_test": {
            "label_access_during_training": False,
            "identity_metadata_access_for_fingerprint": True,
            "evaluation_performed": False,
            "used_for_training_or_selection": False,
        },
        "capacity_policy": copy.deepcopy(
            campaign_payload["comparison_contract"]["parameter_policy"]
        ),
        "resource_metrics_policy": {
            "role": "reporting_only",
            "selection_inputs": [],
        },
        "parameter_inventory": copy.deepcopy(parameter_inventory),
        "training_resources": training_resources,
        "stage_controller_events": {
            key: int(value) for key, value in stage_updates.items()
        },
        "method_native_event_accounting": event_accounting,
        "primary_event_definition": (
            "scbutterfly_joint_generator_controller_event"
            if args.method == "scbutterfly_ops_b"
            else "shared_model_optimizer_step"
        ),
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
        "validation_state_comparison": validation_state_comparison,
        "validation_state_comparison_path": str(
            validation_state_comparison_path.resolve()
        ),
        "validation_state_comparison_sha256": common_runner.file_sha256(
            validation_state_comparison_path
        ),
        "split_contract": split_contract.to_manifest(),
        "checkpoint_path": str(primary_single_path.resolve()),
        "checkpoint_sha256": common_runner.file_sha256(primary_single_path),
        "state_paths": {
            key: str(value.resolve()) for key, value in state_paths.items()
        },
        "global_audit_state_paths": {
            key: str(value.resolve()) for key, value in global_state_paths.items()
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
        "runtime_metrics_path": str(runtime_path.resolve()),
        "runtime_metrics_sha256": common_runner.file_sha256(runtime_path),
        "parameter_inventory_sha256": common_runner.file_sha256(parameter_path),
        "active_parameter_profiles_sha256": common_runner.file_sha256(active_path),
        "inference_throughput_sha256": common_runner.file_sha256(throughput_path),
    }
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
        "trial_config_sha256": common_runner.json_sha256(trial),
        "campaign_binding": copy.deepcopy(binding),
        "split_contract": complete["split_contract"],
        "update_budget": complete["update_budget"],
        "exposure_ledger": complete["exposure_ledger"],
        "validation_trajectory": complete["validation_trajectory"],
        "checkpoint_selection": complete["checkpoint_selection"],
        "global_checkpoint_selection": complete[
            "global_checkpoint_selection"
        ],
        "specialist_composite_selection": complete[
            "specialist_composite_selection"
        ],
        "hpo_reporter_normalized_validation_score": complete[
            "hpo_reporter_normalized_validation_score"
        ],
        "evaluation_states": complete["evaluation_states"],
        "evaluation_state_roles": complete["evaluation_state_roles"],
        "stage_controller_events": complete["stage_controller_events"],
        "method_native_event_accounting": complete[
            "method_native_event_accounting"
        ],
        "primary_event_definition": complete["primary_event_definition"],
        "outer_test_label_access_during_training": False,
        "outer_test_identity_metadata_access_for_fingerprint": True,
        "outer_test_evaluation_performed": False,
        "outer_test_used_for_selection": False,
        "test_may_select_evaluation_state": False,
        "capacity_policy": complete["capacity_policy"],
        "runtime_metrics": {
            "path": complete["runtime_metrics_path"],
            "sha256": complete["runtime_metrics_sha256"],
            "role": "reporting_only",
            "selection_inputs": [],
        },
        "optimizer_schedule": {
            "role": (
                "method_native_multistage_fixed_learning_rates"
                if args.method == "scbutterfly_ops_b"
                else str(trial["training"]["scheduler"]["name"])
            ),
            "trial_role": str(trial["trial_role"]),
            "claim_of_exact_upstream_schedule_fidelity": False,
            "task_adapted_batch_validation_and_stopping_clock": True,
        },
    }
    harness_manifest["manifest_sha256"] = common_runner.json_sha256(
        harness_manifest
    )
    harness_path = args.output_root / "training_harness_manifest.json"
    atomic_json(harness_path, harness_manifest)
    complete["training_harness_manifest_sha256"] = common_runner.file_sha256(
        harness_path
    )
    atomic_json(args.output_root / "training_complete.json", complete)
    flattened_history = []
    for row in history:
        flattened_history.append(
            {
                key: value
                for key, value in row.items()
                if not isinstance(value, Mapping)
            }
        )
    pd.DataFrame(flattened_history).to_csv(
        args.output_root / "validation_trajectory.csv", index=False
    )
    atomic_json(
        args.output_root / "validation_trajectory_full.json",
        {"history": list(history)},
    )
    training_result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "mode": "development",
        "method_id": args.method,
        "split": args.split,
        "fold": args.fold,
        "trial_index": args.trial_index,
        "training_complete": str(
            (args.output_root / "training_complete.json").resolve()
        ),
        "training_complete_sha256": common_runner.file_sha256(
            args.output_root / "training_complete.json"
        ),
        "reporter_normalized_validation_score": complete[
            "hpo_reporter_normalized_validation_score"
        ],
        "selection_unit": (
            "reporter_specialist_composite"
            if complete["specialist_composite_selection"] is not None
            else "global_checkpoint"
        ),
        "outer_test_evaluated": False,
        "outer_test_label_accessed": False,
    }
    atomic_json(args.output_root / "training_result.json", training_result)
    return complete


def _replay_shared_exposure_ledger(
    ledger: harness.ExposureLedger,
    *,
    reporters: Sequence[str],
    completed_epochs: int,
    rounds_per_epoch: int,
    batch_heads: int,
    observations_per_head: int,
) -> None:
    for _epoch in range(completed_epochs):
        for _round in range(rounds_per_epoch):
            for offset in range(0, len(reporters), batch_heads):
                ledger.record_update(
                    {
                        reporter: observations_per_head
                        for reporter in reporters[offset : offset + batch_heads]
                    }
                )


def validate_shared_resume_consistency(
    *,
    completed_epochs: int,
    total_epochs: int,
    steps_per_epoch: int,
    global_step: int,
    stage_updates: Mapping[str, int],
    trajectory: harness.ValidationTrajectory,
    ledger: harness.ExposureLedger,
    history: Sequence[Mapping[str, Any]],
    sampler_state: Mapping[str, Any],
) -> None:
    """Cross-check independently persisted shared-loop progress counters."""

    _require(
        0 <= completed_epochs <= total_epochs,
        "Resume completed_epoch is outside the frozen schedule",
    )
    expected_steps = completed_epochs * steps_per_epoch
    _require(
        global_step == expected_steps,
        "Resume global_step disagrees with completed_epoch",
    )
    _require(
        ledger.completed_optimizer_updates == expected_steps,
        "Resume exposure ledger update count disagrees with global_step",
    )
    expected_stage_updates = {
        "phase_pretrain": 0,
        "phenotype_pretrain": 0,
        "joint_discriminator": 0,
        "primary_joint_generator": expected_steps,
    }
    _require(
        {key: int(value) for key, value in stage_updates.items()}
        == expected_stage_updates,
        "Resume shared stage counters disagree with global_step",
    )
    records = trajectory.records
    _require(
        len(records) == completed_epochs,
        "Resume validation trajectory length disagrees with completed_epoch",
    )
    _require(
        [int(row["optimizer_update"]) for row in records]
        == [epoch * steps_per_epoch for epoch in range(1, completed_epochs + 1)],
        "Resume validation trajectory updates disagree with epoch clock",
    )
    _require(
        len(history) == completed_epochs
        and [int(row.get("epoch", -1)) for row in history]
        == list(range(1, completed_epochs + 1)),
        "Resume history disagrees with completed_epoch",
    )
    _require(
        int(sampler_state.get("epoch", -1)) == completed_epochs
        and int(sampler_state.get("next_batch_index", -1)) == 0,
        "Resume sampler cursor disagrees with completed_epoch",
    )


def train_shared_method(
    args: argparse.Namespace,
    *,
    campaign_payload: Mapping[str, Any],
    trial: Mapping[str, Any],
    binding: Mapping[str, Any],
    phase_cache: Any,
    heads: Sequence[frozen_engine.HeadPartition],
    x_state: Any,
    gpu_staging: frozen_engine.GPUTrainingStaging,
    split_contract: harness.FrozenSplitContract,
) -> dict[str, Any]:
    _require(args.method in SHARED_LOOP_METHODS, "Shared loop method is invalid")
    reporters = tuple(head.slug for head in heads)
    training = trial["training"]
    optimizer_config = trial["optimizer"]
    epochs = int(training["joint_epochs"])
    batch_heads = int(training["batch_heads"])
    observations_per_head = int(training["observations_per_head"])
    rounds_per_epoch = common_runner.frozen_rounds_per_epoch(args, len(reporters))
    budget, steps_per_epoch = common_runner.build_update_budget(
        reporters,
        epochs,
        observations_per_head,
        batch_heads,
        rounds_per_epoch,
    )
    model = create_harness_collection(
        args.method, trial, heads, device="cuda"
    )
    model_manifest = model.factory_manifest()
    parameter_inventory = runtime_metrics.ParameterInventory.from_model(
        model
    ).to_manifest()
    optimizer = create_primary_optimizer(
        args.method, model, optimizer_config
    )
    total_steps = budget.total_optimizer_updates
    scheduler, scheduler_manifest = create_shared_primary_scheduler(
        args.method,
        optimizer,
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        training=training,
    )
    scaler = common_runner.make_grad_scaler(False)
    sampler = sampler_library.DeterministicCyclicTaskSampler(
        {head.slug: len(head.train_indices) for head in heads},
        tasks_per_batch=batch_heads,
        samples_per_task=observations_per_head,
        epoch_samples_per_task=observations_per_head * rounds_per_epoch,
        seed=frozen_engine.stable_seed(
            args.seed, args.split, args.fold, args.method, "primary"
        ),
    )
    _require(len(sampler) == steps_per_epoch, "Primary sampler budget drifted")
    ledger = harness.ExposureLedger(budget)
    nulls = common_runner.null_validation_mse(heads)
    objective = harness.ReporterNormalizedCheckpointObjective.create(
        reporters,
        nulls,
        denominator_floor=1.0e-3,
        tie_relative_tolerance=2.0e-3,
    )
    schedule = harness.ValidationSchedule(
        total_optimizer_updates=total_steps,
        validation_updates=tuple(
            epoch * steps_per_epoch for epoch in range(1, epochs + 1)
        ),
    )
    trajectory = harness.ValidationTrajectory(objective, schedule)
    ema_state = common_runner.tensor_state_dict(model, device="cuda")
    heads_by_slug = {head.slug: head for head in heads}
    resource_meter = runtime_metrics.CudaTrainingResourceMeter(
        torch, device="cuda"
    )
    resource_meter.begin()
    active_ledger = runtime_metrics.CompactActiveSignatureLedger()
    stage_updates = {
        "phase_pretrain": 0,
        "phenotype_pretrain": 0,
        "joint_discriminator": 0,
        "primary_joint_generator": 0,
    }
    state_manager = ValidationStateManager(
        args.output_root,
        method=args.method,
        model_manifest=model_manifest,
        stage_updates=stage_updates,
    )
    last_path = args.output_root / "last.pt"
    if args.resume and last_path.is_file():
        if not state_manager.checkpoint_path(0).is_file():
            raise FileNotFoundError(
                "Resume checkpoint exists without its frozen initialization"
            )
    else:
        state_manager.save_initial(model)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "method": args.method,
        "trial": copy.deepcopy(trial),
        "binding": copy.deepcopy(binding),
        "split_contract_sha256": harness.json_sha256(split_contract.to_manifest()),
        "update_budget_sha256": harness.json_sha256(budget.to_manifest()),
        "model_manifest": copy.deepcopy(model_manifest),
        "optimizer": {
            "name": optimizer_config["name"],
            "learning_rate": float(optimizer_config["learning_rate"]),
            "weight_decay": float(optimizer_config["weight_decay"]),
            "epsilon": (
                float(optimizer_config["epsilon"])
            ),
            "scheduler": copy.deepcopy(scheduler_manifest),
        },
        "outer_test_label_access": "physically_sealed",
        "precision_policy": {
            "amp_enabled": bool(args.amp),
            "tf32_enabled": bool(args.tf32),
            "autocast_dtype": "bfloat16" if args.amp else "float32",
        },
        "implementation_files": method_implementation_manifest(args.method),
        "implementation_sha256": common_runner.file_sha256(Path(__file__)),
        "harness_models_sha256": common_runner.file_sha256(
            Path(biological_models.__file__)
        ),
    }
    if args.method == "ops_captain":
        identity["captain_provenance"] = captain_provenance_manifest(
            trial,
            campaign_payload["methods"]["ops_captain"][
                "source_anchor_contract"
            ],
        )
    elif args.method == "midas_ops":
        identity["midas_provenance"] = {
            "trial_role": trial["trial_role"],
            "source_fidelity_contract": copy.deepcopy(
                campaign_payload["methods"]["midas_ops"][
                    "source_fidelity_contract"
                ]
            ),
            "scientific_scope": (
                "phase plus 52 independent sparse reporter modalities"
            ),
            "same_cell_coalescing_key": "frozen_phase_row_index",
            "batch_correction_claim": False,
        }
    if last_path.exists() and not args.resume:
        raise BiologicalRunnerError(
            "Output contains last.pt; use --resume or a new output root"
        )
    resume_payload: Mapping[str, Any] | None = None
    if args.resume and last_path.is_file():
        resume_payload = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        _require(
            resume_payload.get("schema_version") == SCHEMA_VERSION,
            "Resume checkpoint schema changed",
        )
    identity_sha = bind_training_identity(
        args.output_root / "training_identity.json",
        identity,
        resume_payload=resume_payload,
    )
    history: list[dict[str, Any]] = []
    global_step = 0
    start_epoch = 1
    previous_runtime = 0.0
    if resume_payload is not None:
        resume = resume_payload
        completed_epochs = int(resume["completed_epoch"])
        _replay_shared_exposure_ledger(
            ledger,
            reporters=reporters,
            completed_epochs=completed_epochs,
            rounds_per_epoch=rounds_per_epoch,
            batch_heads=batch_heads,
            observations_per_head=observations_per_head,
        )
        _require(
            ledger.to_manifest() == resume["ledger"],
            "Resume exposure ledger cannot be reproduced",
        )
        trajectory = harness.ValidationTrajectory.from_manifest(
            resume["trajectory"]
        )
        history = list(resume["history"])
        global_step = int(resume["global_step"])
        start_epoch = completed_epochs + 1
        previous_runtime = float(resume.get("runtime_seconds", 0.0))
        stage_updates = {
            key: int(value) for key, value in resume["stage_updates"].items()
        }
        validate_shared_resume_consistency(
            completed_epochs=completed_epochs,
            total_epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            global_step=global_step,
            stage_updates=stage_updates,
            trajectory=trajectory,
            ledger=ledger,
            history=history,
            sampler_state=resume["sampler_state"],
        )
        state_manager.restore_inventory(trajectory)
        model.load_state_dict(resume["model_state"], strict=True)
        model.to("cuda")
        optimizer.load_state_dict(resume["optimizer_state"])
        scheduler.load_state_dict(resume["scheduler_state"])
        scaler.load_state_dict(resume["scaler_state"])
        sampler.load_state_dict(resume["sampler_state"])
        ema_state = {
            name: value.to("cuda") for name, value in resume["ema_state"].items()
        }
        resource_meter.load_state_dict(resume["resource_meter_state"])
        active_ledger.load_state_dict(resume["active_signature_ledger"])
        random.setstate(resume["rng_state"]["python"])
        np.random.set_state(resume["rng_state"]["numpy"])
        torch.set_rng_state(resume["rng_state"]["torch_cpu"])
        torch.cuda.set_rng_state_all(resume["rng_state"]["torch_cuda"])
    started = time.time()
    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.time()
        model.train()
        train_diagnostics: dict[str, list[float]] = {
            reporter: [] for reporter in reporters
        }
        resource_meter.start_training_region()
        for task_batch in sampler:
            groups, exposure, observed_predictions = load_reporter_groups(
                task_batch=task_batch,
                heads_by_slug=heads_by_slug,
                phase_cache=phase_cache,
                x_state=x_state,
                gpu_staging=gpu_staging,
                device="cuda",
            )
            optimizer.zero_grad(set_to_none=True)
            diagnostics: OrderedDict[str, torch.Tensor] = OrderedDict()
            if args.method == "ops_captain":
                # Keep only one reporter's 172-token cross-attention graph in
                # memory at a time.  The 1/N scaling is exactly the grouped
                # reporter-macro objective and still forms one physical update.
                with common_runner.autocast_cuda(args.amp):
                    diagnostics = backward_captain_reporter_accumulation(
                        model,
                        groups,
                        phase_mask_rate=float(training["phase_mask_rate"]),
                        backward=lambda value: scaler.scale(value).backward(),
                    )
            else:
                with common_runner.autocast_cuda(args.amp):
                    result = forward_shared_primary(
                        args.method,
                        model,
                        groups,
                        phase_mask_rate=None,
                        sample=True,
                    )
                scaler.scale(result.total_loss).backward()
                diagnostics.update(result.reporter_diagnostics)
            active_endpoint_queries = None
            if args.method == "ops_captain":
                active_endpoint_queries = captain_active_endpoint_query_labels(
                    groups, heads_by_slug
                )
            active_row = runtime_metrics.active_parameter_snapshot_after_backward(
                model,
                optimizer_update=global_step + 1,
                active_reporters=[group.reporter_slug for group in groups],
                active_endpoint_queries=active_endpoint_queries,
                cell_observations=sum(group.phase_x.shape[0] for group in groups),
                observed_endpoint_predictions=observed_predictions,
            )
            active_ledger.record(active_row)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            common_runner.update_ema(
                ema_state, model, float(training["ema_decay"])
            )
            ledger.record_update(exposure)
            global_step += 1
            stage_updates["primary_joint_generator"] += 1
            for reporter, value in diagnostics.items():
                train_diagnostics[reporter].append(
                    float(value.detach().float().cpu())
                )
        epoch_resource = resource_meter.stop_training_region()
        _require(
            global_step == epoch * steps_per_epoch,
            "Primary optimizer update count drifted",
        )
        validation = {
            head.slug: predict_validation_reporter(
                model,
                phase_cache=phase_cache,
                x_state=x_state,
                head=head,
                gpu_staging=gpu_staging,
                device="cuda",
                batch_size=args.prediction_batch_size,
                amp=args.amp,
            )
            for head in heads
        }
        trajectory_row = state_manager.capture(
            model=model,
            ema_state=ema_state,
            optimizer_update=global_step,
            validation=validation,
            trajectory=trajectory,
            tie_relative_tolerance=objective.tie_relative_tolerance,
            stage_updates=stage_updates,
        )
        history.append(
            {
                "stage": "primary_joint",
                "epoch": epoch,
                "optimizer_update": global_step,
                "train_diagnostic_by_reporter": {
                    reporter: float(np.mean(values))
                    for reporter, values in train_diagnostics.items()
                },
                "validation_mse_by_reporter": validation,
                "validation_macro_mse": trajectory_row["raw_reporter_macro_mse"],
                "validation_reporter_normalized_mse": trajectory_row[
                    "reporter_normalized_score"
                ],
                "learning_rate_end": float(optimizer.param_groups[0]["lr"]),
                "seconds": time.time() - epoch_started,
                "training_gpu_seconds": epoch_resource["gpu_seconds"],
            }
        )
        runtime_payload = {
            "schema_version": SCHEMA_VERSION,
            "identity_sha256": identity_sha,
            "completed_epoch": epoch,
            "global_step": global_step,
            "stage_updates": copy.deepcopy(stage_updates),
            "history": copy.deepcopy(history),
            "trajectory": trajectory.to_manifest(),
            "ledger": ledger.to_manifest(),
            "model_state": common_runner.tensor_state_dict(model),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "sampler_state": sampler.state_dict(),
            "ema_state": {
                name: value.detach().cpu() for name, value in ema_state.items()
            },
            "runtime_seconds": previous_runtime + time.time() - started,
            "resource_meter_state": resource_meter.state_dict(),
            "active_signature_ledger": active_ledger.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
        }
        common_runner.atomic_torch_save(runtime_payload, last_path)
        print(
            f"{args.method} epoch {epoch:03d}/{epochs}: "
            f"validation normalized="
            f"{trajectory_row['reporter_normalized_score']:.6f}; "
            f"update={global_step}",
            flush=True,
        )
    return finalize_development_training(
        args,
        campaign_payload=campaign_payload,
        trial=trial,
        binding=binding,
        model=model,
        model_manifest=model_manifest,
        split_contract=split_contract,
        budget=budget,
        ledger=ledger,
        objective=objective,
        trajectory=trajectory,
        state_manager=state_manager,
        ema_decay=float(training["ema_decay"]),
        stage_updates=stage_updates,
        history=history,
        parameter_inventory=parameter_inventory,
        resource_meter=resource_meter,
        active_signature_ledger=active_ledger,
        phase_cache=phase_cache,
        x_state=x_state,
        heads=heads,
        gpu_staging=gpu_staging,
        identity_sha256=identity_sha,
        runtime_seconds=previous_runtime + time.time() - started,
    )


def scbutterfly_controller_config(
    trial: Mapping[str, Any], *, rounds_per_epoch: int
) -> ScButterflyTrainingConfig:
    """Build the declared source-anchor or OPS-stabilized controller config."""

    optimizer = trial["optimizer"]
    training = trial["training"]
    _require(
        optimizer["generator_name"] == "adam",
        "scButterfly controller requires Adam",
    )
    _require(
        optimizer["discriminator_name"] == "sgd",
        "scButterfly controller requires SGD discriminators",
    )
    generator_lr = float(optimizer["generator_learning_rate"])
    losses = training["loss_weights"]
    overrides = dict(
        phase_encoder_lr=generator_lr,
        phenotype_encoder_lr=generator_lr,
        phase_decoder_lr=generator_lr,
        phenotype_decoder_lr=generator_lr,
        phase_pretrain_bridge_lr=generator_lr,
        phenotype_pretrain_bridge_lr=generator_lr,
        translator_lr=generator_lr,
        discriminator_lr=float(optimizer["discriminator_learning_rate"]),
        adam_weight_decay=float(optimizer["generator_weight_decay"]),
        source_adversarial_threshold=float(
            training["source_adversarial_threshold"]
        ),
        phase_pretrain_kl_weight=float(losses["phase_pretrain_kl"]),
        phenotype_pretrain_kl_weight=float(
            losses["phenotype_pretrain_kl"]
        ),
        phase_reconstruction_weight=float(losses["phase_reconstruction"]),
        phenotype_reconstruction_weight=float(
            losses["phenotype_reconstruction"]
        ),
        phase_joint_kl_weight=float(losses["phase_joint_kl"]),
        phenotype_joint_kl_weight=float(losses["phenotype_joint_kl"]),
        adversarial_weight=float(losses["adversarial"]),
        phase_pretrain_kl_warmup_steps=(
            int(training["kl_warmup_epochs"]) * rounds_per_epoch
        ),
        phenotype_pretrain_kl_warmup_steps=(
            int(training["kl_warmup_epochs"]) * rounds_per_epoch
        ),
        joint_kl_warmup_steps=(
            int(training["kl_warmup_epochs"]) * rounds_per_epoch
        ),
        max_gradient_norm=(
            None
            if training["gradient_clip_norm"] is None
            else float(training["gradient_clip_norm"])
        ),
    )
    protocol = str(training["adversarial_protocol"])
    if protocol == "source_anchor":
        return ScButterflyTrainingConfig.source_anchor(
            phase_dim=172,
            **overrides,
        )
    if protocol == "ops_stabilized":
        return ScButterflyTrainingConfig.ops_stabilized(**overrides)
    raise BiologicalRunnerError(
        f"Unsupported scButterfly adversarial protocol {protocol!r}"
    )


def aggregate_scbutterfly_stage_updates(
    controllers: Mapping[str, ScButterflyTrainingController]
) -> dict[str, int]:
    return {
        "phase_pretrain": sum(
            value.counters.phase_pretrain_steps for value in controllers.values()
        ),
        "phenotype_pretrain": sum(
            value.counters.phenotype_pretrain_steps
            for value in controllers.values()
        ),
        "joint_discriminator": sum(
            value.counters.joint_discriminator_steps
            for value in controllers.values()
        ),
        "primary_joint_generator": sum(
            value.counters.joint_generator_steps for value in controllers.values()
        ),
    }


def update_scbutterfly_reporter_ema(
    ema_state: Mapping[str, torch.Tensor],
    model: biological_models.ScButterflySpecialistCollection,
    reporter: str,
    decay: float,
) -> None:
    """Update EMA only for the specialist that received this G step.

    A full-collection EMA call after every reporter-local generator update
    would decay each inactive specialist eleven extra times per panel round.
    That silently changes its effective decay from ``d`` to ``d**12``.  State
    keys include buffers, matching the common full-model EMA semantics.
    """

    if not isinstance(model, biological_models.ScButterflySpecialistCollection):
        raise TypeError("reporter-local EMA requires a scButterfly collection")
    name = str(reporter)
    model.specialist(name)  # Public active-reporter validation.
    value = _probability(decay, "scButterfly EMA decay", allow_zero=False)
    prefix = f"specialists.{name}."
    current = model.state_dict()
    keys = tuple(key for key in current if key.startswith(prefix))
    _require(bool(keys), f"No scButterfly EMA state found for {name}")
    _require(
        all(key in ema_state for key in keys),
        f"scButterfly EMA state is incomplete for {name}",
    )
    with torch.no_grad():
        for key in keys:
            target = ema_state[key]
            source = current[key].detach().to(target.device)
            if target.is_floating_point():
                target.mul_(value).add_(source, alpha=1.0 - value)
            else:
                target.copy_(source)


def _make_specialist_sampler(
    args: argparse.Namespace,
    heads: Sequence[frozen_engine.HeadPartition],
    *,
    rounds_per_epoch: int,
    observations_per_head: int,
    stage: str,
) -> sampler_library.DeterministicCyclicTaskSampler:
    return sampler_library.DeterministicCyclicTaskSampler(
        {head.slug: len(head.train_indices) for head in heads},
        tasks_per_batch=1,
        samples_per_task=observations_per_head,
        epoch_samples_per_task=observations_per_head * rounds_per_epoch,
        seed=frozen_engine.stable_seed(
            args.seed,
            args.split,
            args.fold,
            args.method,
            stage,
        ),
    )


def _save_scbutterfly_runtime(
    path: Path,
    *,
    identity_sha256: str,
    current_stage: str,
    completed_epochs: Mapping[str, int],
    controllers: Mapping[str, ScButterflyTrainingController],
    samplers: Mapping[str, Any],
    ema_state: Mapping[str, torch.Tensor] | None,
    trajectory: harness.ValidationTrajectory,
    ledger: harness.ExposureLedger,
    history: Sequence[Mapping[str, Any]],
    runtime_seconds: float,
    resource_meter: runtime_metrics.CudaTrainingResourceMeter,
    active_ledger: runtime_metrics.CompactActiveSignatureLedger,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity_sha256,
        "current_stage": current_stage,
        "completed_epochs": {
            key: int(value) for key, value in completed_epochs.items()
        },
        "controller_states": {
            slug: controller.state_dict()
            for slug, controller in controllers.items()
        },
        "sampler_states": {
            name: sampler.state_dict() for name, sampler in samplers.items()
        },
        "ema_state": (
            None
            if ema_state is None
            else {
                name: value.detach().cpu() for name, value in ema_state.items()
            }
        ),
        "trajectory": trajectory.to_manifest(),
        "ledger": ledger.to_manifest(),
        "history": copy.deepcopy(list(history)),
        "stage_updates": aggregate_scbutterfly_stage_updates(controllers),
        "runtime_seconds": float(runtime_seconds),
        "resource_meter_state": resource_meter.state_dict(),
        "active_signature_ledger": (
            None if active_ledger.n_observed_batches == 0 else active_ledger.state_dict()
        ),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
        },
    }
    common_runner.atomic_torch_save(payload, path)


def _replay_specialist_primary_ledger(
    ledger: harness.ExposureLedger,
    *,
    reporters: Sequence[str],
    completed_joint_epochs: int,
    rounds_per_epoch: int,
    observations_per_head: int,
) -> None:
    for _epoch in range(completed_joint_epochs):
        for _round in range(rounds_per_epoch):
            for reporter in reporters:
                ledger.record_update({reporter: observations_per_head})


def validate_scbutterfly_resume_consistency(
    *,
    current_stage: str,
    completed_epochs: Mapping[str, int],
    scheduled_epochs: Mapping[str, int],
    rounds_per_epoch: int,
    reporters: Sequence[str],
    controllers: Mapping[str, ScButterflyTrainingController],
    stage_updates: Mapping[str, int],
    trajectory: harness.ValidationTrajectory,
    ledger: harness.ExposureLedger,
    history: Sequence[Mapping[str, Any]],
    ema_state_present: bool,
    sampler_states: Mapping[str, Mapping[str, Any]],
) -> None:
    """Cross-check stage, controller, trajectory, ledger and EMA progress."""

    stages = ("phase_pretrain", "phenotype_pretrain", "joint")
    _require(set(completed_epochs) == set(stages), "Resume epoch-stage keys changed")
    _require(set(scheduled_epochs) == set(stages), "Scheduled epoch-stage keys changed")
    for stage in stages:
        _require(
            0 <= int(completed_epochs[stage]) <= int(scheduled_epochs[stage]),
            f"Resume {stage} epoch is outside the frozen schedule",
        )
    _require(current_stage in stages, "Resume current_stage is invalid")
    if current_stage == "phase_pretrain":
        _require(
            int(completed_epochs["phenotype_pretrain"]) == 0
            and int(completed_epochs["joint"]) == 0,
            "Phase-pretrain resume contains later-stage progress",
        )
    elif current_stage == "phenotype_pretrain":
        _require(
            int(completed_epochs["phase_pretrain"])
            == int(scheduled_epochs["phase_pretrain"])
            and int(completed_epochs["joint"]) == 0,
            "Phenotype-pretrain resume has inconsistent stage progress",
        )
    else:
        _require(
            int(completed_epochs["phase_pretrain"])
            == int(scheduled_epochs["phase_pretrain"])
            and int(completed_epochs["phenotype_pretrain"])
            == int(scheduled_epochs["phenotype_pretrain"]),
            "Joint resume started before both pretraining stages completed",
        )
    _require(
        ema_state_present == (current_stage == "joint"),
        "scButterfly resume EMA presence disagrees with current_stage",
    )
    reporter_count = len(tuple(reporters))
    _require(reporter_count > 0, "scButterfly resume reporter set is empty")
    expected_per_controller = {
        "phase_pretrain": int(completed_epochs["phase_pretrain"])
        * rounds_per_epoch,
        "phenotype_pretrain": int(completed_epochs["phenotype_pretrain"])
        * rounds_per_epoch,
        "joint_discriminator": int(completed_epochs["joint"])
        * rounds_per_epoch,
        "primary_joint_generator": int(completed_epochs["joint"])
        * rounds_per_epoch,
    }
    for reporter in reporters:
        controller = controllers[reporter]
        _require(
            controller.stage.value == current_stage,
            f"scButterfly resume controller stage drifted for {reporter}",
        )
        counters = controller.counters
        _require(
            {
                "phase_pretrain": counters.phase_pretrain_steps,
                "phenotype_pretrain": counters.phenotype_pretrain_steps,
                "joint_discriminator": counters.joint_discriminator_steps,
                "primary_joint_generator": counters.joint_generator_steps,
            }
            == expected_per_controller,
            f"scButterfly resume controller counters drifted for {reporter}",
        )
    expected_aggregate = {
        key: value * reporter_count
        for key, value in expected_per_controller.items()
    }
    _require(
        {key: int(value) for key, value in stage_updates.items()}
        == expected_aggregate,
        "scButterfly resume aggregate stage counters drifted",
    )
    joint_events = expected_aggregate["primary_joint_generator"]
    _require(
        ledger.completed_optimizer_updates == joint_events,
        "scButterfly resume exposure ledger disagrees with joint G events",
    )
    records = trajectory.records
    completed_joint = int(completed_epochs["joint"])
    primary_events_per_epoch = reporter_count * rounds_per_epoch
    _require(
        len(records) == completed_joint
        and [int(row["optimizer_update"]) for row in records]
        == [
            epoch * primary_events_per_epoch
            for epoch in range(1, completed_joint + 1)
        ],
        "scButterfly resume validation trajectory drifted",
    )
    expected_history_counts = {
        stage: int(completed_epochs[stage]) for stage in stages
    }
    observed_history_counts = {
        stage: sum(row.get("stage") == stage for row in history)
        for stage in stages
    }
    _require(
        observed_history_counts == expected_history_counts
        and len(history) == sum(expected_history_counts.values()),
        "scButterfly resume history drifted from completed epochs",
    )
    _require(
        set(sampler_states) == set(stages),
        "scButterfly resume sampler-stage keys changed",
    )
    for stage in stages:
        state = sampler_states[stage]
        _require(
            int(state.get("epoch", -1)) == int(completed_epochs[stage])
            and int(state.get("next_batch_index", -1)) == 0,
            f"scButterfly resume sampler cursor drifted for {stage}",
        )


def train_scbutterfly_specialists(
    args: argparse.Namespace,
    *,
    campaign_payload: Mapping[str, Any],
    trial: Mapping[str, Any],
    binding: Mapping[str, Any],
    phase_cache: Any,
    heads: Sequence[frozen_engine.HeadPartition],
    x_state: Any,
    gpu_staging: frozen_engine.GPUTrainingStaging,
    split_contract: harness.FrozenSplitContract,
) -> dict[str, Any]:
    _require(
        args.method == "scbutterfly_ops_b",
        "Specialist loop is only for scButterfly",
    )
    reporters = tuple(head.slug for head in heads)
    training = trial["training"]
    specialists_per_primary_event = int(
        training["specialists_per_primary_event"]
    )
    _require(
        specialists_per_primary_event == 1,
        "scButterfly requires one specialist per primary event",
    )
    observations_per_head = int(training["observations_per_head"])
    rounds_per_epoch = common_runner.frozen_rounds_per_epoch(args, len(reporters))
    phase_epochs = int(training["phase_pretrain_epochs"])
    phenotype_epochs = int(training["phenotype_pretrain_epochs"])
    joint_epochs = int(training["joint_epochs"])
    primary_steps_per_epoch = len(reporters) * rounds_per_epoch
    budget = harness.OptimizerUpdateBudget.create(
        reporters,
        total_optimizer_updates=primary_steps_per_epoch * joint_epochs,
        expected_example_exposures={
            reporter: observations_per_head * rounds_per_epoch * joint_epochs
            for reporter in reporters
        },
        expected_update_exposures={
            reporter: rounds_per_epoch * joint_epochs for reporter in reporters
        },
    )
    model = create_harness_collection(
        args.method, trial, heads, device="cuda"
    )
    if not isinstance(model, biological_models.ScButterflySpecialistCollection):
        raise TypeError("Expected ScButterflySpecialistCollection")
    model_manifest = model.factory_manifest()
    parameter_inventory = runtime_metrics.ParameterInventory.from_model(
        model
    ).to_manifest()
    controller_config = scbutterfly_controller_config(
        trial, rounds_per_epoch=rounds_per_epoch
    )
    controllers = OrderedDict(
        (
            reporter,
            ScButterflyTrainingController(
                model.specialist(reporter), controller_config
            ),
        )
        for reporter in reporters
    )
    samplers = {
        stage: _make_specialist_sampler(
            args,
            heads,
            rounds_per_epoch=rounds_per_epoch,
            observations_per_head=observations_per_head,
            stage=stage,
        )
        for stage in ("phase_pretrain", "phenotype_pretrain", "joint")
    }
    nulls = common_runner.null_validation_mse(heads)
    objective = harness.ReporterNormalizedCheckpointObjective.create(
        reporters,
        nulls,
        denominator_floor=1.0e-3,
        tie_relative_tolerance=2.0e-3,
    )
    schedule = harness.ValidationSchedule(
        total_optimizer_updates=budget.total_optimizer_updates,
        validation_updates=tuple(
            epoch * primary_steps_per_epoch
            for epoch in range(1, joint_epochs + 1)
        ),
    )
    trajectory = harness.ValidationTrajectory(objective, schedule)
    ledger = harness.ExposureLedger(budget)
    heads_by_slug = {head.slug: head for head in heads}
    resource_meter = runtime_metrics.CudaTrainingResourceMeter(
        torch, device="cuda"
    )
    resource_meter.begin()
    active_ledger = runtime_metrics.CompactActiveSignatureLedger()
    state_manager = ValidationStateManager(
        args.output_root,
        method=args.method,
        model_manifest=model_manifest,
        stage_updates=aggregate_scbutterfly_stage_updates(controllers),
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "method": args.method,
        "trial": copy.deepcopy(trial),
        "binding": copy.deepcopy(binding),
        "split_contract_sha256": harness.json_sha256(split_contract.to_manifest()),
        "primary_update_budget_sha256": harness.json_sha256(budget.to_manifest()),
        "model_manifest": copy.deepcopy(model_manifest),
        "controller_config": asdict(controller_config),
        "controller_variant": next(iter(controllers.values())).controller_manifest()[
            "variant_label"
        ],
        "trial_role": trial["trial_role"],
        "source_fidelity_contract": copy.deepcopy(
            campaign_payload["methods"]["scbutterfly_ops_b"][
                "source_fidelity_contract"
            ]
        ),
        "training_schedule_contract": copy.deepcopy(
            campaign_payload["methods"]["scbutterfly_ops_b"][
                "training_schedule_contract"
            ]
        ),
        "primary_exposure_contract": copy.deepcopy(
            campaign_payload["methods"]["scbutterfly_ops_b"][
                "primary_exposure_contract"
            ]
        ),
        "primary_event_and_optimizer_step_semantics": {
            "specialists_per_primary_event": specialists_per_primary_event,
            "per_reporter_event_batch_size_matches_shared_methods": True,
            "joint_reporter_exposure_multiplier_vs_standard": 2.0,
            "total_target_bearing_exposure_multiplier_vs_standard": 3.0,
            "joint_generator_events": budget.total_optimizer_updates,
            "joint_discriminator_events": budget.total_optimizer_updates,
            "paired_discriminator_generator_events": budget.total_optimizer_updates,
            "owner_optimizer_step_calls_per_event": {
                "phase_pretrain": 3,
                "phenotype_pretrain": 3,
                "joint_discriminator": 2,
                "joint_generator": 5,
            },
            "optimizer_step_calls_equalized_across_model_families": False,
        },
        "optimizer_schedule": "method_native_multistage_fixed_learning_rates",
        "claim_of_exact_upstream_schedule_fidelity": False,
        "kl_warmup_fidelity": (
            "OPS task-adapted warmup clock; not an exact upstream schedule"
        ),
        "ema_scope": "reporter_local_after_current_specialist_generator_step",
        "primary_update_definition": "joint_generator_update_only",
        "auxiliary_updates_reported_separately": True,
        "outer_test_label_access": "physically_sealed",
        "precision_policy": {
            "amp_enabled": bool(args.amp),
            "tf32_enabled": bool(args.tf32),
            "autocast_dtype": "bfloat16" if args.amp else "float32",
            "declared_trial_policy": training["precision_policy"],
        },
        "implementation_files": method_implementation_manifest(args.method),
        "implementation_sha256": common_runner.file_sha256(Path(__file__)),
        "controller_sha256": common_runner.file_sha256(
            PROJECT_ROOT
            / "scripts"
            / "ops_biological_baseline_scbutterfly_training.py"
        ),
    }
    last_path = args.output_root / "last.pt"
    if last_path.exists() and not args.resume:
        raise BiologicalRunnerError(
            "Output contains last.pt; use --resume or a new output root"
        )
    resume_payload: Mapping[str, Any] | None = None
    if args.resume and last_path.is_file():
        resume_payload = torch.load(
            last_path, map_location="cpu", weights_only=False
        )
        _require(
            resume_payload.get("schema_version") == SCHEMA_VERSION,
            "scButterfly resume schema changed",
        )
    identity_sha = bind_training_identity(
        args.output_root / "training_identity.json",
        identity,
        resume_payload=resume_payload,
    )
    completed_epochs = {
        "phase_pretrain": 0,
        "phenotype_pretrain": 0,
        "joint": 0,
    }
    current_stage = "phase_pretrain"
    history: list[dict[str, Any]] = []
    ema_state: dict[str, torch.Tensor] | None = None
    previous_runtime = 0.0
    if resume_payload is not None:
        resume = resume_payload
        for reporter, controller in controllers.items():
            controller.load_state_dict(
                resume["controller_states"][reporter],
                restore_rng_state=False,
            )
        for name, sampler in samplers.items():
            sampler.load_state_dict(resume["sampler_states"][name])
        current_stage = str(resume["current_stage"])
        completed_epochs = {
            key: int(value) for key, value in resume["completed_epochs"].items()
        }
        trajectory = harness.ValidationTrajectory.from_manifest(
            resume["trajectory"]
        )
        _replay_specialist_primary_ledger(
            ledger,
            reporters=reporters,
            completed_joint_epochs=completed_epochs["joint"],
            rounds_per_epoch=rounds_per_epoch,
            observations_per_head=observations_per_head,
        )
        _require(
            ledger.to_manifest() == resume["ledger"],
            "scButterfly resume exposure ledger cannot be reproduced",
        )
        history = list(resume["history"])
        previous_runtime = float(resume.get("runtime_seconds", 0.0))
        saved_ema = resume.get("ema_state")
        if saved_ema is not None:
            ema_state = {
                name: value.to("cuda") for name, value in saved_ema.items()
            }
        resume_stage_updates = {
            key: int(value) for key, value in resume["stage_updates"].items()
        }
        validate_scbutterfly_resume_consistency(
            current_stage=current_stage,
            completed_epochs=completed_epochs,
            scheduled_epochs={
                "phase_pretrain": phase_epochs,
                "phenotype_pretrain": phenotype_epochs,
                "joint": joint_epochs,
            },
            rounds_per_epoch=rounds_per_epoch,
            reporters=reporters,
            controllers=controllers,
            stage_updates=resume_stage_updates,
            trajectory=trajectory,
            ledger=ledger,
            history=history,
            ema_state_present=ema_state is not None,
            sampler_states=resume["sampler_states"],
        )
        resource_meter.load_state_dict(resume["resource_meter_state"])
        if resume.get("active_signature_ledger") is not None:
            active_ledger.load_state_dict(resume["active_signature_ledger"])
        if completed_epochs["joint"] > 0:
            state_manager.restore_inventory(trajectory)
        random.setstate(resume["rng_state"]["python"])
        np.random.set_state(resume["rng_state"]["numpy"])
        torch.set_rng_state(resume["rng_state"]["torch_cpu"])
        torch.cuda.set_rng_state_all(resume["rng_state"]["torch_cuda"])
    started = time.time()

    if current_stage == "phase_pretrain":
        for epoch in range(completed_epochs["phase_pretrain"] + 1, phase_epochs + 1):
            epoch_started = time.time()
            losses: list[float] = []
            resource_meter.start_training_region()
            for task_batch in samplers["phase_pretrain"]:
                groups, _, _ = load_reporter_groups(
                    task_batch=task_batch,
                    heads_by_slug=heads_by_slug,
                    phase_cache=phase_cache,
                    x_state=x_state,
                    gpu_staging=gpu_staging,
                    device="cuda",
                )
                group = groups[0]
                model.zero_grad(set_to_none=True)
                with common_runner.autocast_cuda(args.amp):
                    loss = controllers[group.reporter_slug].phase_pretrain_step(
                        ScButterflyPairedBatch(
                            group.phase_x, group.target_y, group.observed_mask
                        )
                    )
                losses.append(float(loss.total.detach().float().cpu()))
            epoch_resource = resource_meter.stop_training_region()
            completed_epochs["phase_pretrain"] = epoch
            history.append(
                {
                    "stage": "phase_pretrain",
                    "epoch": epoch,
                    "mean_loss": float(np.mean(losses)),
                    "aggregate_stage_controller_events": aggregate_scbutterfly_stage_updates(
                        controllers
                    )["phase_pretrain"],
                    "seconds": time.time() - epoch_started,
                    "training_gpu_seconds": epoch_resource["gpu_seconds"],
                }
            )
            _save_scbutterfly_runtime(
                last_path,
                identity_sha256=identity_sha,
                current_stage=current_stage,
                completed_epochs=completed_epochs,
                controllers=controllers,
                samplers=samplers,
                ema_state=ema_state,
                trajectory=trajectory,
                ledger=ledger,
                history=history,
                runtime_seconds=previous_runtime + time.time() - started,
                resource_meter=resource_meter,
                active_ledger=active_ledger,
            )
        for controller in controllers.values():
            controller.finish_phase_pretraining()
        current_stage = "phenotype_pretrain"

    if current_stage == "phenotype_pretrain":
        for epoch in range(
            completed_epochs["phenotype_pretrain"] + 1,
            phenotype_epochs + 1,
        ):
            epoch_started = time.time()
            losses = []
            resource_meter.start_training_region()
            for task_batch in samplers["phenotype_pretrain"]:
                groups, _, _ = load_reporter_groups(
                    task_batch=task_batch,
                    heads_by_slug=heads_by_slug,
                    phase_cache=phase_cache,
                    x_state=x_state,
                    gpu_staging=gpu_staging,
                    device="cuda",
                )
                group = groups[0]
                model.zero_grad(set_to_none=True)
                with common_runner.autocast_cuda(args.amp):
                    loss = controllers[
                        group.reporter_slug
                    ].phenotype_pretrain_step(
                        ScButterflyPairedBatch(
                            group.phase_x, group.target_y, group.observed_mask
                        )
                    )
                losses.append(float(loss.total.detach().float().cpu()))
            epoch_resource = resource_meter.stop_training_region()
            completed_epochs["phenotype_pretrain"] = epoch
            history.append(
                {
                    "stage": "phenotype_pretrain",
                    "epoch": epoch,
                    "mean_loss": float(np.mean(losses)),
                    "aggregate_stage_controller_events": aggregate_scbutterfly_stage_updates(
                        controllers
                    )["phenotype_pretrain"],
                    "seconds": time.time() - epoch_started,
                    "training_gpu_seconds": epoch_resource["gpu_seconds"],
                }
            )
            _save_scbutterfly_runtime(
                last_path,
                identity_sha256=identity_sha,
                current_stage=current_stage,
                completed_epochs=completed_epochs,
                controllers=controllers,
                samplers=samplers,
                ema_state=ema_state,
                trajectory=trajectory,
                ledger=ledger,
                history=history,
                runtime_seconds=previous_runtime + time.time() - started,
                resource_meter=resource_meter,
                active_ledger=active_ledger,
            )
        for controller in controllers.values():
            controller.finish_phenotype_pretraining()
        current_stage = "joint"
        ema_state = common_runner.tensor_state_dict(model, device="cuda")
        state_manager.stage_updates = aggregate_scbutterfly_stage_updates(
            controllers
        )
        state_manager.save_initial(model)

    if current_stage != "joint":
        raise BiologicalRunnerError(f"Invalid scButterfly stage {current_stage!r}")
    if ema_state is None:
        raise BiologicalRunnerError("Joint scButterfly stage lacks EMA state")
    global_step = completed_epochs["joint"] * primary_steps_per_epoch
    for epoch in range(completed_epochs["joint"] + 1, joint_epochs + 1):
        epoch_started = time.time()
        generator_losses: dict[str, list[float]] = {
            reporter: [] for reporter in reporters
        }
        discriminator_losses: list[float] = []
        resource_meter.start_training_region()
        for task_batch in samplers["joint"]:
            groups, _, observed_predictions = load_reporter_groups(
                task_batch=task_batch,
                heads_by_slug=heads_by_slug,
                phase_cache=phase_cache,
                x_state=x_state,
                gpu_staging=gpu_staging,
                device="cuda",
            )
            group = groups[0]
            paired = ScButterflyPairedBatch(
                group.phase_x, group.target_y, group.observed_mask
            )
            controller = controllers[group.reporter_slug]
            model.zero_grad(set_to_none=True)
            with common_runner.autocast_cuda(args.amp):
                discriminator_loss = controller.joint_discriminator_step(paired)
            model.zero_grad(set_to_none=True)
            with common_runner.autocast_cuda(args.amp):
                generator_loss = controller.joint_generator_step(paired)
            global_step += 1
            active_row = runtime_metrics.active_parameter_snapshot_after_backward(
                model,
                optimizer_update=global_step,
                active_reporters=[group.reporter_slug],
                # scButterfly is a specialist collection, not a query model.
                active_endpoint_queries=None,
                cell_observations=len(group.phase_x),
                observed_endpoint_predictions=observed_predictions,
            )
            active_ledger.record(active_row)
            update_scbutterfly_reporter_ema(
                ema_state,
                model,
                group.reporter_slug,
                float(training["ema_decay"]),
            )
            ledger.record_update(
                {group.reporter_slug: len(group.phase_x)}
            )
            generator_losses[group.reporter_slug].append(
                float(generator_loss.total.detach().float().cpu())
            )
            discriminator_losses.append(
                float(discriminator_loss.total.detach().float().cpu())
            )
        epoch_resource = resource_meter.stop_training_region()
        _require(
            global_step == epoch * primary_steps_per_epoch,
            "scButterfly primary generator update count drifted",
        )
        completed_epochs["joint"] = epoch
        validation = {
            head.slug: predict_validation_reporter(
                model,
                phase_cache=phase_cache,
                x_state=x_state,
                head=head,
                gpu_staging=gpu_staging,
                device="cuda",
                batch_size=args.prediction_batch_size,
                amp=args.amp,
            )
            for head in heads
        }
        stage_updates = aggregate_scbutterfly_stage_updates(controllers)
        trajectory_row = state_manager.capture(
            model=model,
            ema_state=ema_state,
            optimizer_update=global_step,
            validation=validation,
            trajectory=trajectory,
            tie_relative_tolerance=objective.tie_relative_tolerance,
            stage_updates=stage_updates,
        )
        history.append(
            {
                "stage": "joint",
                "epoch": epoch,
                "optimizer_update": global_step,
                "generator_loss_by_reporter": {
                    reporter: float(np.mean(values))
                    for reporter, values in generator_losses.items()
                },
                "discriminator_loss_macro": float(
                    np.mean(discriminator_losses)
                ),
                "validation_mse_by_reporter": validation,
                "validation_macro_mse": trajectory_row["raw_reporter_macro_mse"],
                "validation_reporter_normalized_mse": trajectory_row[
                    "reporter_normalized_score"
                ],
                "stage_controller_events": stage_updates,
                "seconds": time.time() - epoch_started,
                "training_gpu_seconds": epoch_resource["gpu_seconds"],
            }
        )
        _save_scbutterfly_runtime(
            last_path,
            identity_sha256=identity_sha,
            current_stage=current_stage,
            completed_epochs=completed_epochs,
            controllers=controllers,
            samplers=samplers,
            ema_state=ema_state,
            trajectory=trajectory,
            ledger=ledger,
            history=history,
            runtime_seconds=previous_runtime + time.time() - started,
            resource_meter=resource_meter,
            active_ledger=active_ledger,
        )
        print(
            f"scButterfly joint epoch {epoch:03d}/{joint_epochs}: "
            f"validation normalized="
            f"{trajectory_row['reporter_normalized_score']:.6f}; "
            f"primary G update={global_step}",
            flush=True,
        )
    for controller in controllers.values():
        controller.finish_joint_training()
    stage_updates = aggregate_scbutterfly_stage_updates(controllers)
    return finalize_development_training(
        args,
        campaign_payload=campaign_payload,
        trial=trial,
        binding=binding,
        model=model,
        model_manifest=model_manifest,
        split_contract=split_contract,
        budget=budget,
        ledger=ledger,
        objective=objective,
        trajectory=trajectory,
        state_manager=state_manager,
        ema_decay=float(training["ema_decay"]),
        stage_updates=stage_updates,
        history=history,
        parameter_inventory=parameter_inventory,
        resource_meter=resource_meter,
        active_signature_ledger=active_ledger,
        phase_cache=phase_cache,
        x_state=x_state,
        heads=heads,
        gpu_staging=gpu_staging,
        identity_sha256=identity_sha,
        runtime_seconds=previous_runtime + time.time() - started,
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    payload, trial, binding = bind_development_trial(args)
    validate_trial_execution_policy(args, trial)
    args.output_root = args.output_root.expanduser().resolve()
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    elif args.output_root.exists() and any(args.output_root.iterdir()):
        if not (args.resume and (args.output_root / "last.pt").is_file()):
            raise BiologicalRunnerError(
                "Non-empty output root is neither an exact resume nor overwrite"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root / "campaign_binding.json", binding)
    gpu = common_runner.configure_cuda(args.seed, args.tf32)
    atomic_json(args.output_root / "gpu_preflight.json", gpu)
    args.campaign_binding = binding
    (
        phase_cache,
        _reporter_table,
        heads,
        x_state,
        comparability,
        split_contract,
        preprocessing_sha256,
    ) = common_runner.prepare_data(args)
    _require(tuple(head.slug for head in heads) == PANEL12, "Prepared panel changed")
    assert_development_heads_sealed(heads)
    comparability = verify_identity_only_outer_test_cohorts(
        args, phase_cache, heads
    )
    head_schema = {
        head.slug: head.data.target_feature_names.astype(str).tolist()
        for head in heads
    }
    atomic_json(
        args.output_root / "head_schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "reporters": list(PANEL12),
            "head_dimensions": {
                slug: len(names) for slug, names in head_schema.items()
            },
            "feature_names": head_schema,
            "sha256": common_runner.json_sha256(head_schema),
        },
    )
    atomic_json(
        args.output_root / "development_data_seal.json",
        {
            "schema_version": SCHEMA_VERSION,
            "outer_test_label_access": "physically_sealed",
            "outer_test_identity_metadata_access": (
                "allowed_for_frozen_cohort_fingerprint_only"
            ),
            "outer_test_evaluation_allowed": False,
            "prepared_test_label_rows": 0,
            "preprocessing_sha256": preprocessing_sha256,
            "comparability": comparability,
            "split_contract_sha256": harness.json_sha256(
                split_contract.to_manifest()
            ),
        },
    )
    gpu_staging = frozen_engine.build_gpu_training_staging(
        common_runner.build_staging_args(args),
        phase_cache,
        x_state,
        heads,
        "cuda",
    )
    _require(
        gpu_staging.metadata.get("enabled") is True,
        "Required GPU staging was not enabled",
    )
    if args.method in SHARED_LOOP_METHODS:
        complete = train_shared_method(
            args,
            campaign_payload=payload,
            trial=trial,
            binding=binding,
            phase_cache=phase_cache,
            heads=heads,
            x_state=x_state,
            gpu_staging=gpu_staging,
            split_contract=split_contract,
        )
    else:
        complete = train_scbutterfly_specialists(
            args,
            campaign_payload=payload,
            trial=trial,
            binding=binding,
            phase_cache=phase_cache,
            heads=heads,
            x_state=x_state,
            gpu_staging=gpu_staging,
            split_contract=split_contract,
        )
    print(
        json.dumps(
            {
                "status": complete["status"],
                "method": args.method,
                "split": args.split,
                "fold": args.fold,
                "trial_index": args.trial_index,
                "checkpoint": complete["checkpoint_path"],
                "outer_test_evaluated": False,
            },
            indent=2,
        ),
        flush=True,
    )
    del gpu_staging, heads, x_state, phase_cache
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
