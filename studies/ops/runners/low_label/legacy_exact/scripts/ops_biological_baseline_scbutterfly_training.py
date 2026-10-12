#!/usr/bin/env python3
"""Method-native training controller for one scButterfly-OPS-B specialist.

This module owns *only* optimization mechanics.  It accepts an already
standardised, exact-paired batch for one reporter and deliberately contains no
data loading, split construction, evaluation, checkpoint selection, launcher,
or experiment side effects.

The controller preserves the native scButterfly stage order and optimizer
topology:

1. phase autoencoder/variational-bridge pretraining;
2. phenotype autoencoder/variational-bridge pretraining;
3. joint training with a discriminator update followed by a generator update.

Each model parameter has exactly one named optimizer owner.  Encoder/decoder
owners are reused across pretraining and joint training, matching the upstream
method, while the two pretraining bridges never enter the joint optimizer
path.  The alternating joint boundary is explicit and serialised, so a
checkpoint taken after D cannot silently resume with another D update.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn

from ops_biological_baseline_scbutterfly import (
    DiscriminatorLoss,
    JointGeneratorLoss,
    JointLossWeights,
    OPS_STABILIZED_PROTOCOL,
    SOURCE_ANCHOR_PROTOCOL,
    SOURCE_DISCRIMINATOR_WEIGHT,
    SOURCE_GENERATOR_GATE_THRESHOLD,
    SOURCE_JOINT_KL_WEIGHT_PHASE172,
    SOURCE_PHASE_RECONSTRUCTION_WEIGHT,
    SOURCE_PHENOTYPE_RECONSTRUCTION_WEIGHT,
    SOURCE_PRETRAIN_KL_WEIGHT_PHASE172,
    PretrainLoss,
    ScButterflyOPSSpecialist,
    prepare_observed_phenotype,
    validate_adversarial_protocol,
)


TRAINING_SCHEMA_VERSION = "ops-scbutterfly-specialist-training-v2"
TRAJECTORY_SCHEMA_VERSION = "ops-scbutterfly-specialist-trajectory-v2"


class TrainingStage(str, Enum):
    PHASE_PRETRAIN = "phase_pretrain"
    PHENOTYPE_PRETRAIN = "phenotype_pretrain"
    JOINT = "joint"
    COMPLETE = "complete"


class JointTurn(str, Enum):
    DISCRIMINATOR = "discriminator"
    GENERATOR = "generator"


class TrainingStageError(RuntimeError):
    """Raised when a stage or alternating-step boundary is violated."""


class OptimizerOwnershipError(RuntimeError):
    """Raised when a gradient escapes the active optimizer owners."""


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive, found {value}")
    return value


def _nonnegative_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, found {value}")
    return value


def _nonnegative_int(value: int, name: str) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, found {value}")
    return value


@dataclass(frozen=True)
class ScButterflyTrainingConfig:
    """Native optimizer and loss settings, independent of a data harness."""

    phase_encoder_lr: float = 1.0e-3
    phenotype_encoder_lr: float = 1.0e-3
    phase_decoder_lr: float = 1.0e-3
    phenotype_decoder_lr: float = 1.0e-3
    phase_pretrain_bridge_lr: float = 1.0e-3
    phenotype_pretrain_bridge_lr: float = 1.0e-3
    translator_lr: float = 1.0e-3
    discriminator_lr: float = 5.0e-3
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1.0e-8
    adam_weight_decay: float = 0.0
    discriminator_momentum: float = 0.0
    adversarial_protocol: str = OPS_STABILIZED_PROTOCOL
    source_adversarial_threshold: float = SOURCE_GENERATOR_GATE_THRESHOLD
    phase_pretrain_kl_weight: float = 1.0 / 150.0
    phenotype_pretrain_kl_weight: float = 1.0 / 150.0
    phase_reconstruction_weight: float = 1.0
    phenotype_reconstruction_weight: float = 1.0
    phase_joint_kl_weight: float = 1.0 / 150.0
    phenotype_joint_kl_weight: float = 1.0 / 150.0
    adversarial_weight: float = 1.0
    phase_pretrain_kl_warmup_steps: int = 0
    phenotype_pretrain_kl_warmup_steps: int = 0
    joint_kl_warmup_steps: int = 0
    lock_encoders_and_decoders_in_joint: bool = False
    max_gradient_norm: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "adversarial_protocol",
            validate_adversarial_protocol(self.adversarial_protocol),
        )
        for name in (
            "phase_encoder_lr",
            "phenotype_encoder_lr",
            "phase_decoder_lr",
            "phenotype_decoder_lr",
            "phase_pretrain_bridge_lr",
            "phenotype_pretrain_bridge_lr",
            "translator_lr",
            "discriminator_lr",
            "adam_eps",
            "phase_pretrain_kl_weight",
            "phenotype_pretrain_kl_weight",
            "phase_reconstruction_weight",
            "phenotype_reconstruction_weight",
            "phase_joint_kl_weight",
            "phenotype_joint_kl_weight",
            "adversarial_weight",
            "source_adversarial_threshold",
        ):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        for name in ("adam_beta1", "adam_beta2"):
            value = float(getattr(self, name))
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1), found {value}")
            object.__setattr__(self, name, value)
        for name in ("adam_weight_decay", "discriminator_momentum"):
            object.__setattr__(
                self, name, _nonnegative_float(getattr(self, name), name)
            )
        if self.discriminator_momentum >= 1.0:
            raise ValueError("discriminator_momentum must be smaller than 1")
        for name in (
            "phase_pretrain_kl_warmup_steps",
            "phenotype_pretrain_kl_warmup_steps",
            "joint_kl_warmup_steps",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        if self.max_gradient_norm is not None:
            object.__setattr__(
                self,
                "max_gradient_norm",
                _positive_float(self.max_gradient_norm, "max_gradient_norm"),
            )

    @classmethod
    def source_anchor(
        cls,
        *,
        phase_dim: int = 172,
        **overrides: object,
    ) -> "ScButterflyTrainingConfig":
        """Public RNA--ADT loss/optimizer anchor mapped to phase -> Y.

        The source defines both modality-pretrain KL coefficients as
        ``20 / source_dim`` and the joint coefficient on each KL term as
        ``40 / source_dim``.  Reconstruction weights are 1 (source/phase) and
        2 (target/phenotype); discriminator weight is 1 and gate is 1.35.
        """

        phase_dim = int(phase_dim)
        if phase_dim <= 0:
            raise ValueError("phase_dim must be positive")
        pretrain_kl = 20.0 / phase_dim
        joint_kl = 40.0 / phase_dim
        anchor: dict[str, object] = {
            "adversarial_protocol": SOURCE_ANCHOR_PROTOCOL,
            "source_adversarial_threshold": SOURCE_GENERATOR_GATE_THRESHOLD,
            "phase_pretrain_kl_weight": pretrain_kl,
            "phenotype_pretrain_kl_weight": pretrain_kl,
            "phase_reconstruction_weight": SOURCE_PHASE_RECONSTRUCTION_WEIGHT,
            "phenotype_reconstruction_weight": SOURCE_PHENOTYPE_RECONSTRUCTION_WEIGHT,
            "phase_joint_kl_weight": joint_kl,
            "phenotype_joint_kl_weight": joint_kl,
            "adversarial_weight": SOURCE_DISCRIMINATOR_WEIGHT,
        }
        anchor.update(overrides)
        return cls(**anchor)

    @classmethod
    def ops_stabilized(
        cls, **overrides: object
    ) -> "ScButterflyTrainingConfig":
        """Reproduce the earlier OPS-stabilized continuous adversarial path."""

        stabilized: dict[str, object] = {
            "adversarial_protocol": OPS_STABILIZED_PROTOCOL,
            "phase_pretrain_kl_weight": 1.0 / 150.0,
            "phenotype_pretrain_kl_weight": 1.0 / 150.0,
            "phase_reconstruction_weight": 1.0,
            "phenotype_reconstruction_weight": 1.0,
            "phase_joint_kl_weight": 1.0 / 150.0,
            "phenotype_joint_kl_weight": 1.0 / 150.0,
            "adversarial_weight": 1.0,
        }
        stabilized.update(overrides)
        return cls(**stabilized)


@dataclass(frozen=True)
class ScButterflyPairedBatch:
    """An already standardised exact-paired single-reporter batch."""

    phase_x: torch.Tensor
    phenotype_y: torch.Tensor
    endpoint_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.phase_x.shape[0])

    @property
    def observed_endpoint_count(self) -> int:
        return int(self.endpoint_mask.sum().item())


@dataclass
class TrainingCounters:
    global_optimizer_steps: int = 0
    phase_pretrain_steps: int = 0
    phenotype_pretrain_steps: int = 0
    joint_discriminator_steps: int = 0
    joint_generator_steps: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(value)
            for name, value in asdict(self).items()
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TrainingCounters":
        expected = set(asdict(cls()))
        if set(payload) != expected:
            raise ValueError(
                f"Counter keys must be {sorted(expected)}, found {sorted(payload)}"
            )
        values = {
            name: _nonnegative_int(payload[name], name)
            for name in expected
        }
        result = cls(**values)
        if result.global_optimizer_steps != (
            result.phase_pretrain_steps
            + result.phenotype_pretrain_steps
            + result.joint_discriminator_steps
            + result.joint_generator_steps
        ):
            raise ValueError("global_optimizer_steps does not match stage counters")
        return result


@dataclass(frozen=True)
class JointAlternatingLoss:
    discriminator: DiscriminatorLoss
    generator: JointGeneratorLoss


def _module_parameters(module: nn.Module) -> tuple[nn.Parameter, ...]:
    return tuple(module.parameters())


@contextmanager
def _temporarily_frozen(modules: Sequence[nn.Module]) -> Iterator[None]:
    states: list[tuple[nn.Parameter, bool]] = []
    for module in modules:
        for parameter in module.parameters():
            states.append((parameter, bool(parameter.requires_grad)))
            parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in states:
            parameter.requires_grad_(requires_grad)


class ScButterflyTrainingController:
    """Stateful optimizer controller with enforced method-native boundaries."""

    def __init__(
        self,
        model: ScButterflyOPSSpecialist,
        config: ScButterflyTrainingConfig | None = None,
    ) -> None:
        if not isinstance(model, ScButterflyOPSSpecialist):
            raise TypeError("model must be a ScButterflyOPSSpecialist")
        self.model = model
        self.config = config or ScButterflyTrainingConfig()
        self.stage = TrainingStage.PHASE_PRETRAIN
        self.joint_turn: JointTurn | None = None
        self.counters = TrainingCounters()
        self._trajectory: list[dict[str, Any]] = []

        self._owner_parameters: dict[str, tuple[nn.Parameter, ...]] = {
            "phase_encoder": _module_parameters(model.phase_encoder),
            "phenotype_encoder": _module_parameters(model.phenotype_encoder),
            "phase_decoder": _module_parameters(model.phase_decoder),
            "phenotype_decoder": _module_parameters(model.phenotype_decoder),
            "phase_pretrain_bridge": _module_parameters(
                model.phase_pretrain_bridge
            ),
            "phenotype_pretrain_bridge": _module_parameters(
                model.phenotype_pretrain_bridge
            ),
            "translator": _module_parameters(model.translator),
            "phase_discriminator": _module_parameters(model.phase_discriminator),
            "phenotype_discriminator": _module_parameters(
                model.phenotype_discriminator
            ),
        }
        self._validate_complete_exclusive_ownership()
        adam_kwargs = {
            "betas": (self.config.adam_beta1, self.config.adam_beta2),
            "eps": self.config.adam_eps,
            "weight_decay": self.config.adam_weight_decay,
        }
        self.optimizers: dict[str, torch.optim.Optimizer] = {
            "phase_encoder": torch.optim.Adam(
                self._owner_parameters["phase_encoder"],
                lr=self.config.phase_encoder_lr,
                **adam_kwargs,
            ),
            "phenotype_encoder": torch.optim.Adam(
                self._owner_parameters["phenotype_encoder"],
                lr=self.config.phenotype_encoder_lr,
                **adam_kwargs,
            ),
            "phase_decoder": torch.optim.Adam(
                self._owner_parameters["phase_decoder"],
                lr=self.config.phase_decoder_lr,
                **adam_kwargs,
            ),
            "phenotype_decoder": torch.optim.Adam(
                self._owner_parameters["phenotype_decoder"],
                lr=self.config.phenotype_decoder_lr,
                **adam_kwargs,
            ),
            "phase_pretrain_bridge": torch.optim.Adam(
                self._owner_parameters["phase_pretrain_bridge"],
                lr=self.config.phase_pretrain_bridge_lr,
                **adam_kwargs,
            ),
            "phenotype_pretrain_bridge": torch.optim.Adam(
                self._owner_parameters["phenotype_pretrain_bridge"],
                lr=self.config.phenotype_pretrain_bridge_lr,
                **adam_kwargs,
            ),
            "translator": torch.optim.Adam(
                self._owner_parameters["translator"],
                lr=self.config.translator_lr,
                **adam_kwargs,
            ),
            "phase_discriminator": torch.optim.SGD(
                self._owner_parameters["phase_discriminator"],
                lr=self.config.discriminator_lr,
                momentum=self.config.discriminator_momentum,
            ),
            "phenotype_discriminator": torch.optim.SGD(
                self._owner_parameters["phenotype_discriminator"],
                lr=self.config.discriminator_lr,
                momentum=self.config.discriminator_momentum,
            ),
        }
        self._validate_optimizer_bindings()

    @property
    def trajectory(self) -> tuple[dict[str, Any], ...]:
        return tuple(copy.deepcopy(self._trajectory))

    def _validate_complete_exclusive_ownership(self) -> None:
        model_parameters = tuple(self.model.parameters())
        model_ids = {id(value) for value in model_parameters}
        owned_ids: set[int] = set()
        for owner, parameters in self._owner_parameters.items():
            if not parameters:
                raise OptimizerOwnershipError(f"Owner {owner!r} has no parameters")
            local_ids = [id(value) for value in parameters]
            if len(local_ids) != len(set(local_ids)):
                raise OptimizerOwnershipError(
                    f"Owner {owner!r} contains duplicate parameters"
                )
            overlap = owned_ids.intersection(local_ids)
            if overlap:
                raise OptimizerOwnershipError(
                    f"Owner {owner!r} overlaps an earlier optimizer owner"
                )
            owned_ids.update(local_ids)
        if owned_ids != model_ids:
            missing = len(model_ids - owned_ids)
            extra = len(owned_ids - model_ids)
            raise OptimizerOwnershipError(
                f"Optimizer ownership is incomplete: missing={missing}, extra={extra}"
            )

    def _validate_optimizer_bindings(self) -> None:
        if set(self.optimizers) != set(self._owner_parameters):
            raise OptimizerOwnershipError("Optimizer names do not match parameter owners")
        for owner, optimizer in self.optimizers.items():
            optimizer_ids = {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group["params"]
            }
            owner_ids = {id(value) for value in self._owner_parameters[owner]}
            if optimizer_ids != owner_ids:
                raise OptimizerOwnershipError(
                    f"Optimizer {owner!r} is not bound exactly to its owner"
                )

    def _validate_batch(self, batch: ScButterflyPairedBatch) -> None:
        if not isinstance(batch, ScButterflyPairedBatch):
            raise TypeError("batch must be a ScButterflyPairedBatch")
        phase = batch.phase_x
        phenotype = batch.phenotype_y
        mask = batch.endpoint_mask
        if not isinstance(phase, torch.Tensor) or phase.ndim != 2:
            raise ValueError("phase_x must be a rank-2 torch.Tensor")
        if phase.shape[1] != self.model.phase_dim:
            raise ValueError(
                f"Expected phase_x width {self.model.phase_dim}, found {phase.shape[1]}"
            )
        if not phase.is_floating_point() or not torch.isfinite(phase).all():
            raise ValueError("phase_x must be finite floating point")
        prepare_observed_phenotype(
            phenotype,
            mask,
            expected_dim=self.model.phenotype_dim,
        )
        if phase.shape[0] != phenotype.shape[0]:
            raise ValueError("phase_x and phenotype_y batch sizes must match")
        if phase.shape[0] < 2:
            raise ValueError("Training batches require at least two rows for BatchNorm")
        if phase.device != phenotype.device or phase.device != mask.device:
            raise ValueError("phase_x, phenotype_y, and endpoint_mask must share a device")
        model_parameter = next(self.model.parameters())
        if phase.device != model_parameter.device:
            raise ValueError("batch and model must be on the same device")
        if phase.dtype != model_parameter.dtype or phenotype.dtype != model_parameter.dtype:
            raise ValueError("batch floating dtype must match the model dtype")

    def _require_stage(self, expected: TrainingStage) -> None:
        if self.stage is not expected:
            raise TrainingStageError(
                f"Expected stage {expected.value!r}, found {self.stage.value!r}"
            )

    def _require_joint_turn(self, expected: JointTurn) -> None:
        self._require_stage(TrainingStage.JOINT)
        if self.joint_turn is not expected:
            actual = None if self.joint_turn is None else self.joint_turn.value
            raise TrainingStageError(
                f"Expected joint {expected.value!r} step, found {actual!r}"
            )

    @staticmethod
    def _warmup_factor(completed_steps: int, warmup_steps: int) -> float:
        if warmup_steps <= 0:
            return 1.0
        return min(float(completed_steps + 1) / float(warmup_steps), 1.0)

    def _clear_gradients(self, active_owners: Sequence[str]) -> None:
        self.model.zero_grad(set_to_none=True)
        for owner in active_owners:
            self.optimizers[owner].zero_grad(set_to_none=True)

    def _assert_gradient_ownership(self, active_owners: Sequence[str]) -> None:
        allowed = {
            id(parameter)
            for owner in active_owners
            for parameter in self._owner_parameters[owner]
        }
        escaped = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None and id(parameter) not in allowed
        ]
        if escaped:
            raise OptimizerOwnershipError(
                "Gradients escaped active optimizer ownership: " + ", ".join(escaped)
            )
        active_gradients = [
            parameter.grad
            for owner in active_owners
            for parameter in self._owner_parameters[owner]
            if parameter.grad is not None
        ]
        if not active_gradients:
            raise OptimizerOwnershipError("The active optimizer owners received no gradients")
        if not all(bool(torch.isfinite(value).all().item()) for value in active_gradients):
            raise FloatingPointError("A non-finite gradient was produced")

    def _step_optimizers(self, active_owners: Sequence[str]) -> None:
        parameters = tuple(
            parameter
            for owner in active_owners
            for parameter in self._owner_parameters[owner]
            if parameter.requires_grad
        )
        if self.config.max_gradient_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, self.config.max_gradient_norm)
        for owner in active_owners:
            self.optimizers[owner].step()

    @staticmethod
    def _loss_scalar(value: torch.Tensor, name: str) -> float:
        result = float(value.detach().cpu().item())
        if not math.isfinite(result):
            raise FloatingPointError(f"Loss {name!r} is not finite")
        return result

    def _record_step(
        self,
        *,
        stage: TrainingStage,
        substep: str,
        stage_step: int,
        active_owners: Sequence[str],
        batch: ScButterflyPairedBatch,
        losses: Mapping[str, torch.Tensor],
        scalars: Mapping[str, float] | None = None,
    ) -> None:
        self.counters.global_optimizer_steps += 1
        row: dict[str, Any] = {
            "event_index": len(self._trajectory),
            "event": "optimizer_step",
            "stage": stage.value,
            "substep": str(substep),
            "global_optimizer_step": self.counters.global_optimizer_steps,
            "stage_step": int(stage_step),
            "active_optimizers": list(active_owners),
            "batch_size": batch.batch_size,
            "observed_endpoint_count": batch.observed_endpoint_count,
            "losses": {
                name: self._loss_scalar(value, name)
                for name, value in losses.items()
            },
        }
        if scalars:
            checked: dict[str, float] = {}
            for name, value in scalars.items():
                number = float(value)
                if not math.isfinite(number):
                    raise FloatingPointError(f"Scalar {name!r} is not finite")
                checked[name] = number
            row["scalars"] = checked
        self._trajectory.append(row)

    def _transition(self, destination: TrainingStage) -> None:
        source = self.stage
        self.stage = destination
        self._trajectory.append(
            {
                "event_index": len(self._trajectory),
                "event": "stage_transition",
                "from_stage": source.value,
                "to_stage": destination.value,
                "global_optimizer_step": self.counters.global_optimizer_steps,
            }
        )

    def phase_pretrain_step(self, batch: ScButterflyPairedBatch) -> PretrainLoss:
        self._require_stage(TrainingStage.PHASE_PRETRAIN)
        self._validate_batch(batch)
        self.model.train()
        owners = ("phase_encoder", "phase_decoder", "phase_pretrain_bridge")
        kl_factor = self._warmup_factor(
            self.counters.phase_pretrain_steps,
            self.config.phase_pretrain_kl_warmup_steps,
        )
        kl_weight = self.config.phase_pretrain_kl_weight * kl_factor
        self._clear_gradients(owners)
        loss = self.model.phase_pretrain_loss(
            batch.phase_x,
            kl_weight=kl_weight,
            sample=True,
        )
        self._loss_scalar(loss.total, "total")
        loss.total.backward()
        self._assert_gradient_ownership(owners)
        self._step_optimizers(owners)
        self.counters.phase_pretrain_steps += 1
        self._record_step(
            stage=TrainingStage.PHASE_PRETRAIN,
            substep="phase_autoencoder",
            stage_step=self.counters.phase_pretrain_steps,
            active_owners=owners,
            batch=batch,
            losses={
                "total": loss.total,
                "reconstruction": loss.reconstruction,
                "kl": loss.kl,
            },
            scalars={"kl_weight": kl_weight, "kl_warmup_factor": kl_factor},
        )
        return loss

    def finish_phase_pretraining(self) -> None:
        self._require_stage(TrainingStage.PHASE_PRETRAIN)
        if self.counters.phase_pretrain_steps == 0:
            raise TrainingStageError("Phase pretraining cannot finish before one step")
        self._transition(TrainingStage.PHENOTYPE_PRETRAIN)

    def phenotype_pretrain_step(
        self, batch: ScButterflyPairedBatch
    ) -> PretrainLoss:
        self._require_stage(TrainingStage.PHENOTYPE_PRETRAIN)
        self._validate_batch(batch)
        self.model.train()
        owners = (
            "phenotype_encoder",
            "phenotype_decoder",
            "phenotype_pretrain_bridge",
        )
        kl_factor = self._warmup_factor(
            self.counters.phenotype_pretrain_steps,
            self.config.phenotype_pretrain_kl_warmup_steps,
        )
        kl_weight = self.config.phenotype_pretrain_kl_weight * kl_factor
        self._clear_gradients(owners)
        loss = self.model.phenotype_pretrain_loss(
            batch.phenotype_y,
            batch.endpoint_mask,
            kl_weight=kl_weight,
            sample=True,
        )
        self._loss_scalar(loss.total, "total")
        loss.total.backward()
        self._assert_gradient_ownership(owners)
        self._step_optimizers(owners)
        self.counters.phenotype_pretrain_steps += 1
        self._record_step(
            stage=TrainingStage.PHENOTYPE_PRETRAIN,
            substep="phenotype_autoencoder",
            stage_step=self.counters.phenotype_pretrain_steps,
            active_owners=owners,
            batch=batch,
            losses={
                "total": loss.total,
                "reconstruction": loss.reconstruction,
                "kl": loss.kl,
            },
            scalars={"kl_weight": kl_weight, "kl_warmup_factor": kl_factor},
        )
        return loss

    def finish_phenotype_pretraining(self) -> None:
        self._require_stage(TrainingStage.PHENOTYPE_PRETRAIN)
        if self.counters.phenotype_pretrain_steps == 0:
            raise TrainingStageError(
                "Phenotype pretraining cannot finish before one step"
            )
        self.joint_turn = JointTurn.DISCRIMINATOR
        self._transition(TrainingStage.JOINT)

    def joint_discriminator_step(
        self, batch: ScButterflyPairedBatch
    ) -> DiscriminatorLoss:
        self._require_joint_turn(JointTurn.DISCRIMINATOR)
        self._validate_batch(batch)
        self.model.train()
        owners = ("phase_discriminator", "phenotype_discriminator")
        self._clear_gradients(owners)
        loss = self.model.discriminator_step_loss(
            batch.phase_x,
            batch.phenotype_y,
            batch.endpoint_mask,
            adversarial_protocol=self.config.adversarial_protocol,
        )
        self._loss_scalar(loss.total, "total")
        loss.total.backward()
        self._assert_gradient_ownership(owners)
        self._step_optimizers(owners)
        self.counters.joint_discriminator_steps += 1
        self._record_step(
            stage=TrainingStage.JOINT,
            substep="discriminator",
            stage_step=self.counters.joint_discriminator_steps,
            active_owners=owners,
            batch=batch,
            losses={
                "total": loss.total,
                "phase_real": loss.phase_real,
                "phase_fake": loss.phase_fake,
                "phenotype_real": loss.phenotype_real,
                "phenotype_fake": loss.phenotype_fake,
            },
            scalars={
                "source_soft_label_mean": (
                    0.0
                    if loss.soft_labels is None
                    else float(loss.soft_labels.mean().item())
                ),
                "source_real_fraction": (
                    0.0
                    if loss.soft_labels is None
                    else float((loss.soft_labels > 0.5).float().mean().item())
                ),
            },
        )
        self.joint_turn = JointTurn.GENERATOR
        return loss

    def _joint_generator_owners(self) -> tuple[str, ...]:
        if self.config.lock_encoders_and_decoders_in_joint:
            return ("translator",)
        return (
            "translator",
            "phase_encoder",
            "phenotype_encoder",
            "phase_decoder",
            "phenotype_decoder",
        )

    def joint_generator_step(
        self, batch: ScButterflyPairedBatch
    ) -> JointGeneratorLoss:
        self._require_joint_turn(JointTurn.GENERATOR)
        self._validate_batch(batch)
        self.model.train()
        owners = self._joint_generator_owners()
        kl_factor = self._warmup_factor(
            self.counters.joint_generator_steps,
            self.config.joint_kl_warmup_steps,
        )
        weights = JointLossWeights(
            phase_reconstruction=self.config.phase_reconstruction_weight,
            phenotype_reconstruction=self.config.phenotype_reconstruction_weight,
            phase_kl=self.config.phase_joint_kl_weight * kl_factor,
            phenotype_kl=self.config.phenotype_joint_kl_weight * kl_factor,
            adversarial=self.config.adversarial_weight,
        )
        self._clear_gradients(owners)
        modules_to_lock: tuple[nn.Module, ...] = ()
        if self.config.lock_encoders_and_decoders_in_joint:
            modules_to_lock = (
                self.model.phase_encoder,
                self.model.phenotype_encoder,
                self.model.phase_decoder,
                self.model.phenotype_decoder,
            )
        with _temporarily_frozen(modules_to_lock):
            loss = self.model.joint_generator_loss(
                batch.phase_x,
                batch.phenotype_y,
                batch.endpoint_mask,
                weights=weights,
                sample=True,
                adversarial_protocol=self.config.adversarial_protocol,
                source_adversarial_threshold=(
                    self.config.source_adversarial_threshold
                ),
            )
            self._loss_scalar(loss.total, "total")
            loss.total.backward()
        self._assert_gradient_ownership(owners)
        self._step_optimizers(owners)
        self.counters.joint_generator_steps += 1
        self._record_step(
            stage=TrainingStage.JOINT,
            substep="generator",
            stage_step=self.counters.joint_generator_steps,
            active_owners=owners,
            batch=batch,
            losses={
                "total": loss.total,
                "phase_to_phase": loss.phase_to_phase,
                "phase_to_phenotype": loss.phase_to_phenotype,
                "phenotype_to_phase": loss.phenotype_to_phase,
                "phenotype_to_phenotype": loss.phenotype_to_phenotype,
                "phase_kl": loss.phase_kl,
                "phenotype_kl": loss.phenotype_kl,
                "adversarial": loss.adversarial,
                "discriminator_probe": loss.discriminator_probe,
            },
            scalars={
                "phase_kl_weight": weights.phase_kl,
                "phenotype_kl_weight": weights.phenotype_kl,
                "kl_warmup_factor": kl_factor,
                "adversarial_gate_active": float(
                    loss.adversarial_gate_active
                ),
                "source_adversarial_threshold": (
                    self.config.source_adversarial_threshold
                ),
            },
        )
        self.joint_turn = JointTurn.DISCRIMINATOR
        return loss

    def joint_alternating_step(
        self, batch: ScButterflyPairedBatch
    ) -> JointAlternatingLoss:
        """Run exactly one native D-then-G update pair for the same batch."""

        discriminator = self.joint_discriminator_step(batch)
        generator = self.joint_generator_step(batch)
        return JointAlternatingLoss(
            discriminator=discriminator,
            generator=generator,
        )

    def finish_joint_training(self) -> None:
        self._require_stage(TrainingStage.JOINT)
        if self.joint_turn is not JointTurn.DISCRIMINATOR:
            raise TrainingStageError(
                "Joint training cannot finish between discriminator and generator"
            )
        if self.counters.joint_generator_steps == 0:
            raise TrainingStageError("Joint training cannot finish before one D/G pair")
        if (
            self.counters.joint_discriminator_steps
            != self.counters.joint_generator_steps
        ):
            raise TrainingStageError("Joint D/G counters are not balanced")
        self._transition(TrainingStage.COMPLETE)

    def optimizer_ownership_manifest(self) -> dict[str, Any]:
        names_by_id = {
            id(parameter): name
            for name, parameter in self.model.named_parameters()
        }
        owners: dict[str, Any] = {}
        for owner, parameters in self._owner_parameters.items():
            optimizer = self.optimizers[owner]
            owners[owner] = {
                "optimizer": type(optimizer).__name__,
                "parameter_count": int(sum(value.numel() for value in parameters)),
                "parameter_names": [names_by_id[id(value)] for value in parameters],
                "learning_rates": [
                    float(group["lr"]) for group in optimizer.param_groups
                ],
            }
        return {
            "exclusive": True,
            "complete": True,
            "owners": owners,
            "phase_pretrain_active": [
                "phase_encoder",
                "phase_decoder",
                "phase_pretrain_bridge",
            ],
            "phenotype_pretrain_active": [
                "phenotype_encoder",
                "phenotype_decoder",
                "phenotype_pretrain_bridge",
            ],
            "joint_discriminator_active": [
                "phase_discriminator",
                "phenotype_discriminator",
            ],
            "joint_generator_active": list(self._joint_generator_owners()),
        }

    def trajectory_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "stage": self.stage.value,
            "joint_turn": (
                None if self.joint_turn is None else self.joint_turn.value
            ),
            "counters": self.counters.to_dict(),
            "records": copy.deepcopy(self._trajectory),
        }

    def _source_anchor_match(self) -> dict[str, bool]:
        phase_dim = self.model.config.phase_dim
        expected_pretrain_kl = 20.0 / phase_dim
        expected_joint_kl = 40.0 / phase_dim
        loss_match = all(
            (
                math.isclose(
                    self.config.phase_pretrain_kl_weight,
                    expected_pretrain_kl,
                ),
                math.isclose(
                    self.config.phenotype_pretrain_kl_weight,
                    expected_pretrain_kl,
                ),
                math.isclose(
                    self.config.phase_joint_kl_weight,
                    expected_joint_kl,
                ),
                math.isclose(
                    self.config.phenotype_joint_kl_weight,
                    expected_joint_kl,
                ),
                math.isclose(self.config.phase_reconstruction_weight, 1.0),
                math.isclose(self.config.phenotype_reconstruction_weight, 2.0),
                math.isclose(self.config.adversarial_weight, 1.0),
                math.isclose(
                    self.config.source_adversarial_threshold,
                    SOURCE_GENERATOR_GATE_THRESHOLD,
                ),
            )
        )
        architecture = self.model.config
        architecture_match = all(
            (
                architecture.phase_dim == 172,
                architecture.phase_encoder_widths == (256, 128),
                architecture.phenotype_encoder_widths == (128, 128),
                architecture.phase_decoder_widths == (256,),
                architecture.phenotype_decoder_widths == (128,),
                architecture.latent_dim == 128,
                architecture.discriminator_hidden_widths == (),
                architecture.discriminator_output_batch_norm,
                math.isclose(architecture.dropout, 0.1),
                math.isclose(architecture.phase_input_mask_rate, 0.5),
                math.isclose(architecture.phenotype_input_mask_rate, 0.0),
            )
        )
        return {
            "adversarial_mechanics": (
                self.config.adversarial_protocol == SOURCE_ANCHOR_PROTOCOL
            ),
            "loss_weights": loss_match,
            "architecture": architecture_match,
            "complete_config_anchor": (
                self.config.adversarial_protocol == SOURCE_ANCHOR_PROTOCOL
                and loss_match
                and architecture_match
            ),
        }

    def controller_manifest(self) -> dict[str, Any]:
        protocol = self.config.adversarial_protocol
        source_match = self._source_anchor_match()
        if protocol == OPS_STABILIZED_PROTOCOL:
            variant_label = "scButterfly-OPS-B OPS-stabilized"
        elif source_match["complete_config_anchor"]:
            variant_label = "scButterfly-OPS-B source-anchor"
        else:
            variant_label = "scButterfly-OPS-B source-mechanics OPS-tuned"
        return {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "scope": "single_reporter_exact_paired_training_controller",
            "input_contract": "prestandardized_single_reporter_exact_paired_batch",
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),
            "adversarial_protocol": protocol,
            "variant_label": variant_label,
            "source_anchor_match": source_match,
            "source_anchor_adversarial_mechanics": (
                protocol == SOURCE_ANCHOR_PROTOCOL
            ),
            "may_be_described_as_source_anchor": (
                source_match["complete_config_anchor"]
            ),
            "may_be_described_as_source_faithful": False,
            "ops_stabilized_is_source_faithful": False,
            "trajectory": self.trajectory_manifest(),
            "optimizer_ownership": self.optimizer_ownership_manifest(),
            "contains_data_loader": False,
            "contains_evaluator": False,
            "contains_launcher": False,
        }

    def state_dict(self) -> dict[str, Any]:
        model_device = next(self.model.parameters()).device
        cuda_rng_state: list[torch.Tensor] | None = None
        if model_device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state_all()
        return {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),
            "stage": self.stage.value,
            "joint_turn": (
                None if self.joint_turn is None else self.joint_turn.value
            ),
            "counters": self.counters.to_dict(),
            "trajectory": copy.deepcopy(self._trajectory),
            "model_state": copy.deepcopy(self.model.state_dict()),
            "optimizer_states": {
                name: copy.deepcopy(optimizer.state_dict())
                for name, optimizer in self.optimizers.items()
            },
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_state_all": cuda_rng_state,
        }

    def _validate_loaded_control_state(
        self,
        *,
        stage: TrainingStage,
        joint_turn: JointTurn | None,
        counters: TrainingCounters,
        trajectory: Sequence[Mapping[str, Any]],
    ) -> None:
        if stage in (TrainingStage.PHASE_PRETRAIN, TrainingStage.PHENOTYPE_PRETRAIN):
            if joint_turn is not None:
                raise ValueError("Pretraining state cannot have a joint turn")
            if counters.joint_discriminator_steps or counters.joint_generator_steps:
                raise ValueError("Pretraining state cannot have joint counters")
        if stage in (TrainingStage.JOINT, TrainingStage.COMPLETE):
            if joint_turn is None:
                raise ValueError("Joint/complete state must retain its next joint turn")
            discriminator_steps = counters.joint_discriminator_steps
            generator_steps = counters.joint_generator_steps
            if joint_turn is JointTurn.DISCRIMINATOR and discriminator_steps != generator_steps:
                raise ValueError("Discriminator-turn state must have balanced D/G counters")
            if joint_turn is JointTurn.GENERATOR and discriminator_steps != generator_steps + 1:
                raise ValueError("Generator-turn state must contain one unmatched D step")
            if stage is TrainingStage.COMPLETE and joint_turn is not JointTurn.DISCRIMINATOR:
                raise ValueError("Complete state cannot contain an unmatched D step")
        for index, row in enumerate(trajectory):
            if not isinstance(row, Mapping) or row.get("event_index") != index:
                raise ValueError("Trajectory event indices are not contiguous")
        optimizer_rows = [row for row in trajectory if row.get("event") == "optimizer_step"]
        if len(optimizer_rows) != counters.global_optimizer_steps:
            raise ValueError("Trajectory optimizer-step count does not match counters")
        for index, row in enumerate(optimizer_rows, start=1):
            if row.get("global_optimizer_step") != index:
                raise ValueError("Trajectory optimizer-step indices are not contiguous")

    def load_state_dict(
        self,
        payload: Mapping[str, Any],
        *,
        restore_rng_state: bool = True,
    ) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("Controller state must be a mapping")
        if payload.get("schema_version") != TRAINING_SCHEMA_VERSION:
            raise ValueError("Wrong scButterfly training state schema")
        if payload.get("model_config") != asdict(self.model.config):
            raise ValueError("Saved model config does not match this specialist")
        if payload.get("training_config") != asdict(self.config):
            raise ValueError("Saved training config does not match this controller")
        try:
            stage = TrainingStage(payload["stage"])
            raw_turn = payload.get("joint_turn")
            joint_turn = None if raw_turn is None else JointTurn(raw_turn)
            counters = TrainingCounters.from_mapping(payload["counters"])
            trajectory = copy.deepcopy(payload["trajectory"])
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed scButterfly controller state") from error
        if not isinstance(trajectory, Sequence):
            raise ValueError("Saved trajectory must be a sequence")
        self._validate_loaded_control_state(
            stage=stage,
            joint_turn=joint_turn,
            counters=counters,
            trajectory=trajectory,
        )
        optimizer_states = payload.get("optimizer_states")
        if not isinstance(optimizer_states, Mapping):
            raise ValueError("Saved optimizer_states must be a mapping")
        if set(optimizer_states) != set(self.optimizers):
            raise ValueError("Saved optimizer names do not match this controller")
        self.model.load_state_dict(payload["model_state"], strict=True)
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(optimizer_states[name])
        self.stage = stage
        self.joint_turn = joint_turn
        self.counters = counters
        self._trajectory = [dict(row) for row in trajectory]
        if restore_rng_state:
            rng_state = payload.get("torch_rng_state")
            if not isinstance(rng_state, torch.Tensor):
                raise ValueError("Saved torch_rng_state must be a tensor")
            torch.set_rng_state(rng_state.cpu())
            cuda_state = payload.get("cuda_rng_state_all")
            if cuda_state is not None:
                if next(self.model.parameters()).device.type != "cuda":
                    raise ValueError("CUDA RNG state cannot be restored into a CPU controller")
                torch.cuda.set_rng_state_all(cuda_state)


__all__ = [
    "JointAlternatingLoss",
    "JointTurn",
    "OptimizerOwnershipError",
    "ScButterflyPairedBatch",
    "ScButterflyTrainingConfig",
    "ScButterflyTrainingController",
    "TRAINING_SCHEMA_VERSION",
    "TRAJECTORY_SCHEMA_VERSION",
    "TrainingCounters",
    "TrainingStage",
    "TrainingStageError",
]
