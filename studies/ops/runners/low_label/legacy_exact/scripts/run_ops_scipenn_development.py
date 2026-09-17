#!/usr/bin/env python3
"""Dedicated sciPENN development entrypoint over the audited common harness.

This file adds sciPENN by binding method-specific construction, masked-union
forward loss, Adam, and scheduler policy to the already-audited biological
shared loop.  It does not edit or copy the frozen three-method v2 runner.

Importing this module performs no binding, data access, CUDA initialization, or
training.  Binding occurs only in :func:`configure_common_runner`; execution
occurs only through :func:`main`.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

import ops_biological_baseline_scipenn_harness as scipenn_harness
import prepare_ops_scipenn_panel12_commands as campaign_contract
import run_ops_biological_baseline_training as common_biological_runner
from ops_biological_baseline_scipenn_training import SciPENNReporterTrainingBatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHOD_ID = "scipenn_ops"
TRAINING_SCHEMA_VERSION = "ops-scipenn-development-training-v1"
CHECKPOINT_SCHEMA_VERSION = "ops-scipenn-checkpoint-v1"
DEFAULT_CAMPAIGN_CONFIG = (
    PROJECT_ROOT / "configs" / "ops_scipenn_panel12_development_v1.json"
)
EXPECTED_SHARED_RUNNER_SHA256 = (
    "6efafe8eef480e094a758456cdd5920d33fc6864c2d8d54ec634d98cfef1c4ac"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise common_biological_runner.BiologicalRunnerError(message)


def validate_trial_execution_policy(args: Any, trial: Mapping[str, Any]) -> None:
    """Fail closed on source-anchor versus OPS precision declarations."""

    _require(args.method == METHOD_ID, "Dedicated runner accepts only sciPENN")
    policy = str(trial["training"]["precision_policy"])
    if policy == "float32_no_tf32":
        _require(
            args.amp is False and args.tf32 is False,
            "sciPENN source anchor requires --no-amp --no-tf32",
        )
    elif policy == "amp_bfloat16_tf32":
        _require(
            args.amp is True and args.tf32 is True,
            "sciPENN OPS variants require --amp --tf32",
        )
    else:
        raise common_biological_runner.BiologicalRunnerError(
            f"Unsupported sciPENN precision policy {policy!r}"
        )


def create_harness_collection(
    method: str,
    trial: Mapping[str, Any],
    heads: Sequence[Any],
    *,
    device: str | torch.device,
) -> scipenn_harness.SciPENNHarnessCollection:
    _require(method == METHOD_ID, "Dedicated collection accepts only sciPENN")
    names = tuple(str(head.slug) for head in heads)
    _require(names == campaign_contract.PANEL12, "Prepared reporter order changed")
    full_registry = scipenn_harness.full_reporter_registry()
    dimensions = OrderedDict(
        (str(head.slug), int(head.data.y.shape[1])) for head in heads
    )
    _require(
        dimensions == OrderedDict((name, full_registry[name]) for name in names),
        "Prepared endpoint dimensions differ from the frozen registry",
    )
    model = scipenn_harness.create_scipenn_harness_model(
        active_reporters=names,
        options=trial["model"],
    ).to(device)
    # Native sciPENN's source-dense denominator is the union over participating
    # panels.  Never admit 40 all-missing full52 blocks into panel12 training.
    _require(
        tuple(model.model.reporter_names) == campaign_contract.PANEL12,
        "sciPENN training union differs from the campaign reporter panel",
    )
    _require(
        model.model.total_endpoints == sum(dimensions.values()),
        "sciPENN training union contains endpoints outside panel12",
    )
    return model


def forward_shared_primary(
    method: str,
    model: scipenn_harness.SciPENNHarnessCollection,
    groups: Sequence[common_biological_runner.GroupedReporterBatch],
    *,
    phase_mask_rate: float | None = None,
    sample: bool = True,
) -> common_biological_runner.PrimaryForwardResult:
    """Run sciPENN's native dense-union masked loss for one balanced event."""

    del sample
    _require(method == METHOD_ID, "Dedicated forward accepts only sciPENN")
    if not isinstance(model, scipenn_harness.SciPENNHarnessCollection):
        raise TypeError("sciPENN method requires SciPENNHarnessCollection")
    if phase_mask_rate is not None:
        raise ValueError("sciPENN does not use a phase MGE mask")
    if not groups or len({group.reporter_slug for group in groups}) != len(groups):
        raise ValueError("sciPENN primary event needs unique reporter groups")
    batches = [
        SciPENNReporterTrainingBatch(
            reporter=group.reporter_slug,
            phase_x=group.phase_x,
            endpoint_target=group.target_y,
            endpoint_observed_mask=group.observed_mask,
        )
        for group in groups
    ]
    rich = model.forward_grouped(batches)
    by_name = {group.reporter_slug: group for group in groups}
    diagnostics: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, prediction in rich.local_predictions.items():
        group = by_name[name]
        clean_target = torch.where(
            group.observed_mask, group.target_y, torch.zeros_like(group.target_y)
        )
        clean_prediction = torch.where(
            group.observed_mask, prediction, torch.zeros_like(prediction)
        )
        diagnostics[name] = (
            (clean_prediction - clean_target).square().sum()
            / group.observed_mask.sum().to(clean_prediction.dtype)
        )
    losses = rich.loss_components
    _require(bool(torch.isfinite(losses.total).item()), "sciPENN loss is non-finite")
    return common_biological_runner.PrimaryForwardResult(
        total_loss=losses.total,
        reporter_diagnostics=diagnostics,
        loss_components={
            "native_total": losses.total,
            "masked_mse": losses.mse,
            "masked_quantile_pinball": losses.quantile_pinball,
        },
    )


def create_primary_optimizer(
    method: str,
    model: nn.Module,
    optimizer_config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    _require(method == METHOD_ID, "Dedicated optimizer accepts only sciPENN")
    _require(optimizer_config["name"] == "adam", "sciPENN requires Adam")
    _require(float(optimizer_config["epsilon"]) == 1.0e-8, "Adam epsilon changed")
    _require(float(optimizer_config["weight_decay"]) == 0.0, "Adam weight decay changed")
    return torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )


def create_shared_primary_scheduler(
    method: str,
    optimizer: torch.optim.Optimizer,
    training: Mapping[str, Any],
    *,
    total_steps: int,
    steps_per_epoch: int,
) -> tuple[Any, dict[str, Any]]:
    _require(method == METHOD_ID, "Dedicated scheduler accepts only sciPENN")
    scheduler = training["scheduler"]
    name = str(scheduler["name"])
    if name == "fixed":
        _require(set(scheduler) == {"name"}, "Fixed scheduler fields changed")
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0), {
            "name": "fixed",
            "effective_learning_rate_policy": "fixed",
            "source_relationship": (
                "released Adam learning-rate anchor under the shared fixed-budget "
                "OPS clock; upstream validation-triggered StepLR/early stopping is "
                "not claimed"
            ),
            "source_shaped": True,
            "upstream_execution_exact": False,
            "scheduler_is_upstream_code_faithful": False,
        }
    if name == "common_linear_warmup_cosine_decay":
        _require(
            set(scheduler) == {"name", "warmup_epochs"},
            "Warmup scheduler fields changed",
        )
        warmup_epochs = int(scheduler["warmup_epochs"])
        value = common_biological_runner.cosine_warmup_scheduler(
            optimizer,
            total_steps=int(total_steps),
            warmup_steps=warmup_epochs * int(steps_per_epoch),
        )
        return value, {
            "name": name,
            "warmup_epochs": warmup_epochs,
            "warmup_optimizer_updates": warmup_epochs * int(steps_per_epoch),
            "total_optimizer_updates": int(total_steps),
            "source_relationship": "validation-only OPS stabilization variant",
            "source_shaped": False,
            "upstream_execution_exact": False,
            "scheduler_is_upstream_code_faithful": False,
        }
    raise common_biological_runner.BiologicalRunnerError(
        f"Unsupported sciPENN scheduler {name!r}"
    )


def method_implementation_manifest(method: str) -> dict[str, dict[str, str]]:
    _require(method == METHOD_ID, "Dedicated manifest accepts only sciPENN")
    common = common_biological_runner
    files = {
        "dedicated_runner": Path(__file__).resolve(),
        "shared_runner": Path(common.__file__).resolve(),
        "scipenn_harness": Path(scipenn_harness.__file__).resolve(),
        "scipenn_core": PROJECT_ROOT / "scripts/ops_biological_baseline_scipenn.py",
        "scipenn_bridge": PROJECT_ROOT / "scripts/ops_biological_baseline_scipenn_training.py",
        "campaign_contract": Path(campaign_contract.__file__).resolve(),
        "common_runner": Path(common.common_runner.__file__).resolve(),
        "frozen_engine": Path(common.frozen_engine.__file__).resolve(),
        "sampler_library": Path(common.sampler_library.__file__).resolve(),
        "training_harness_library": Path(common.harness.__file__).resolve(),
        "runtime_metrics": Path(common.runtime_metrics.__file__).resolve(),
    }
    result: dict[str, dict[str, str]] = {}
    for role, path in files.items():
        path = path.resolve()
        _require(path.is_file(), f"Missing sciPENN implementation file: {path}")
        result[role] = {
            "path": str(path),
            "sha256": common.common_runner.file_sha256(path),
        }
    return result


def configure_common_runner() -> Any:
    """Install a narrow sciPENN binding into the imported shared runner."""

    runner = common_biological_runner
    shared_runner_path = Path(runner.__file__).resolve()
    observed_shared_sha = hashlib.sha256(shared_runner_path.read_bytes()).hexdigest()
    _require(
        observed_shared_sha == EXPECTED_SHARED_RUNNER_SHA256,
        "Audited shared biological runner changed; sciPENN binding must be re-audited",
    )
    runner.__doc__ = __doc__
    runner.SCHEMA_VERSION = TRAINING_SCHEMA_VERSION
    runner.CHECKPOINT_SCHEMA_VERSION = CHECKPOINT_SCHEMA_VERSION
    runner.DEFAULT_CAMPAIGN_CONFIG = DEFAULT_CAMPAIGN_CONFIG
    runner.METHODS = (METHOD_ID,)
    runner.SHARED_LOOP_METHODS = (METHOD_ID,)
    runner.DEVELOPMENT_FOLDS = campaign_contract.DEVELOPMENT_FOLDS
    runner.PANEL12 = campaign_contract.PANEL12
    runner.campaign_contract = campaign_contract
    runner.biological_models = scipenn_harness
    runner.validate_frozen_campaign = campaign_contract.validate_config
    runner.validate_trial_execution_policy = validate_trial_execution_policy
    runner.create_harness_collection = create_harness_collection
    runner.forward_shared_primary = forward_shared_primary
    runner.create_primary_optimizer = create_primary_optimizer
    runner.create_shared_primary_scheduler = create_shared_primary_scheduler
    runner.method_implementation_manifest = method_implementation_manifest
    return runner


def main() -> None:
    configure_common_runner().main()


if __name__ == "__main__":
    main()
