#!/usr/bin/env python3
"""Training-time bridge from the common OPS harness to CAPTAIN.

This module deliberately does not load data, construct splits, create an
optimizer, select a device, checkpoint, or run a training loop.  It gives the
common harness a small reporter-grouped API around :class:`OPSCaptainAdapter`:

* one group contains cells from exactly one reporter;
* only that reporter's concrete global endpoint tokens are instantiated;
* endpoint supervision always requires an explicit boolean observed mask;
* a phase MGE mask is accepted only when the wrapped model has a non-zero
  phase-reconstruction objective;
* the scalar loss and all auditable loss components are returned per reporter.

The bridge introduces no query parameters.  In particular, it cannot add
semantic metadata or factorise a concrete reporter-endpoint token into
reporter and endpoint components.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from typing import Any, Sequence

import torch
from torch import nn

from ops_biological_baseline_captain import (
    CaptainLoss,
    CaptainPrediction,
    OPSCaptainAdapter,
)


ReporterReference = str | int | torch.Tensor


@dataclass(frozen=True)
class CaptainReporterTrainingBatch:
    """One reporter-homogeneous minibatch supplied by the common harness.

    ``endpoint_target`` and ``endpoint_observed_mask`` both have shape
    ``[batch, endpoints_for_reporter]``.  A numerical target value of zero is
    valid whenever the corresponding mask entry is ``True``.

    ``phase_mge_mask`` has shape ``[batch, 172]`` and selects originally
    observed phase values that are hidden from the encoder and reconstructed.
    It is intentionally independent of ``endpoint_observed_mask``.
    """

    reporter: ReporterReference
    phase_x: torch.Tensor
    endpoint_target: torch.Tensor
    endpoint_observed_mask: torch.Tensor
    phase_observed_mask: torch.Tensor | None = None
    phase_mge_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class CaptainReporterPredictionBatch:
    """One reporter-homogeneous inference minibatch, without labels or MGE."""

    reporter: ReporterReference
    phase_x: torch.Tensor
    phase_observed_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class CaptainReporterTrainingResult:
    """Prediction and differentiable loss for one canonical reporter."""

    reporter_name: str
    reporter_index: int
    prediction: CaptainPrediction
    loss_components: CaptainLoss

    @property
    def reporter_loss(self) -> torch.Tensor:
        """Scalar loss consumed by the common optimizer step."""

        return self.loss_components.total

    @property
    def endpoint_ids(self) -> torch.Tensor:
        return self.prediction.endpoint_ids


@dataclass(frozen=True)
class CaptainReporterPredictionResult:
    """Label-free prediction for one canonical reporter."""

    reporter_name: str
    reporter_index: int
    prediction: CaptainPrediction

    @property
    def endpoint_ids(self) -> torch.Tensor:
        return self.prediction.endpoint_ids


class OPSCaptainTrainingAdapter(nn.Module):
    """Thin, parameter-free training API around :class:`OPSCaptainAdapter`.

    The wrapped model remains the sole owner of all learned parameters.  This
    class only canonicalises reporter groups, validates group-local endpoint
    shapes, and delegates forward/loss computation to the code-faithful model.
    """

    family = "ops_captain_training_bridge"

    def __init__(self, model: OPSCaptainAdapter) -> None:
        super().__init__()
        if not isinstance(model, OPSCaptainAdapter):
            raise TypeError("model must be an OPSCaptainAdapter")
        self.model = model

    @property
    def phase_mge_loss_enabled(self) -> bool:
        return self.model.phase_reconstruction_loss_weight > 0.0

    def _canonical_reporter(
        self, reporter: ReporterReference
    ) -> tuple[int, str]:
        index = self.model.reporter_index(reporter)
        return index, self.model.reporter_names[index]

    def _validate_group_shapes(
        self,
        *,
        reporter_index: int,
        phase_x: torch.Tensor,
        endpoint_target: torch.Tensor,
        endpoint_observed_mask: torch.Tensor,
    ) -> None:
        if not isinstance(phase_x, torch.Tensor):
            raise TypeError("phase_x must be a torch.Tensor")
        if phase_x.ndim != 2 or phase_x.shape[1] != self.model.input_dim:
            raise ValueError(
                f"phase_x must have shape [batch, {self.model.input_dim}], "
                f"found {tuple(phase_x.shape)}"
            )
        expected = (
            phase_x.shape[0],
            self.model.output_dimensions[reporter_index],
        )
        if not isinstance(endpoint_target, torch.Tensor):
            raise TypeError("endpoint_target must be a torch.Tensor")
        if tuple(endpoint_target.shape) != expected:
            raise ValueError(
                "endpoint_target must be reporter-local with shape "
                f"{expected}, found {tuple(endpoint_target.shape)}"
            )
        if not isinstance(endpoint_observed_mask, torch.Tensor):
            raise TypeError(
                "endpoint_observed_mask must be an explicit torch.Tensor"
            )
        if endpoint_observed_mask.dtype is not torch.bool:
            raise TypeError("endpoint_observed_mask must have dtype torch.bool")
        if tuple(endpoint_observed_mask.shape) != expected:
            raise ValueError(
                "endpoint_observed_mask must match the reporter-local target "
                f"shape {expected}, found {tuple(endpoint_observed_mask.shape)}"
            )

    def _validate_mge_contract(
        self, phase_mge_mask: torch.Tensor | None
    ) -> None:
        if self.phase_mge_loss_enabled and phase_mge_mask is None:
            raise ValueError(
                "The wrapped CAPTAIN model has a non-zero phase reconstruction "
                "loss weight, so training requires an explicit phase_mge_mask"
            )
        if not self.phase_mge_loss_enabled and phase_mge_mask is not None:
            raise ValueError(
                "phase_mge_mask was supplied while the wrapped model has no "
                "active phase reconstruction loss; refusing to corrupt phase "
                "inputs without an MGE objective"
            )

    def forward_reporter(
        self,
        phase_x: torch.Tensor,
        reporter: ReporterReference,
        endpoint_target: torch.Tensor,
        endpoint_observed_mask: torch.Tensor,
        *,
        phase_observed_mask: torch.Tensor | None = None,
        phase_mge_mask: torch.Tensor | None = None,
    ) -> CaptainReporterTrainingResult:
        """Run and score one reporter-homogeneous training group."""

        reporter_index, reporter_name = self._canonical_reporter(reporter)
        self._validate_group_shapes(
            reporter_index=reporter_index,
            phase_x=phase_x,
            endpoint_target=endpoint_target,
            endpoint_observed_mask=endpoint_observed_mask,
        )
        self._validate_mge_contract(phase_mge_mask)

        prediction = self.model.forward_reporter(
            phase_x,
            reporter_index,
            phase_observed_mask=phase_observed_mask,
            phase_reconstruction_mask=phase_mge_mask,
        )
        loss = self.model.loss_components(
            prediction,
            endpoint_target,
            endpoint_observed_mask,
            phase_target=phase_x if self.phase_mge_loss_enabled else None,
            phase_reconstruction_mask=(
                phase_mge_mask if self.phase_mge_loss_enabled else None
            ),
        )
        return CaptainReporterTrainingResult(
            reporter_name=reporter_name,
            reporter_index=reporter_index,
            prediction=prediction,
            loss_components=loss,
        )

    def forward_batch(
        self, batch: CaptainReporterTrainingBatch
    ) -> CaptainReporterTrainingResult:
        """Dataclass form of :meth:`forward_reporter` for harness adapters."""

        if not isinstance(batch, CaptainReporterTrainingBatch):
            raise TypeError("batch must be CaptainReporterTrainingBatch")
        return self.forward_reporter(
            batch.phase_x,
            batch.reporter,
            batch.endpoint_target,
            batch.endpoint_observed_mask,
            phase_observed_mask=batch.phase_observed_mask,
            phase_mge_mask=batch.phase_mge_mask,
        )

    def forward_grouped(
        self, batches: Sequence[CaptainReporterTrainingBatch]
    ) -> OrderedDict[str, CaptainReporterTrainingResult]:
        """Return one differentiable result per unique reporter group.

        No cross-reporter reduction is imposed here.  The common harness can
        choose reporter-macro, sum, accumulation, or another documented
        training reduction from the returned ``reporter_loss`` tensors.
        """

        if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
            raise TypeError("batches must be a sequence of reporter batches")
        if not batches:
            raise ValueError("batches must contain at least one reporter group")
        results: OrderedDict[str, CaptainReporterTrainingResult] = OrderedDict()
        for batch in batches:
            result = self.forward_batch(batch)
            if result.reporter_name in results:
                raise ValueError(
                    "forward_grouped requires one group per reporter; duplicate "
                    f"reporter {result.reporter_name!r}"
                )
            results[result.reporter_name] = result
        return results

    def forward(
        self, batches: Sequence[CaptainReporterTrainingBatch]
    ) -> OrderedDict[str, CaptainReporterTrainingResult]:
        return self.forward_grouped(batches)

    def predict_reporter(
        self,
        phase_x: torch.Tensor,
        reporter: ReporterReference,
        *,
        phase_observed_mask: torch.Tensor | None = None,
    ) -> CaptainReporterPredictionResult:
        """Predict one reporter without labels and without training-time MGE."""

        reporter_index, reporter_name = self._canonical_reporter(reporter)
        prediction = self.model.forward_reporter(
            phase_x,
            reporter_index,
            phase_observed_mask=phase_observed_mask,
            phase_reconstruction_mask=None,
        )
        return CaptainReporterPredictionResult(
            reporter_name=reporter_name,
            reporter_index=reporter_index,
            prediction=prediction,
        )

    def predict_batch(
        self, batch: CaptainReporterPredictionBatch
    ) -> CaptainReporterPredictionResult:
        if not isinstance(batch, CaptainReporterPredictionBatch):
            raise TypeError("batch must be CaptainReporterPredictionBatch")
        return self.predict_reporter(
            batch.phase_x,
            batch.reporter,
            phase_observed_mask=batch.phase_observed_mask,
        )

    def predict_grouped(
        self, batches: Sequence[CaptainReporterPredictionBatch]
    ) -> OrderedDict[str, CaptainReporterPredictionResult]:
        """Return sparse reporter-local predictions for unique groups."""

        if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
            raise TypeError("batches must be a sequence of prediction batches")
        if not batches:
            raise ValueError("batches must contain at least one reporter group")
        results: OrderedDict[str, CaptainReporterPredictionResult] = OrderedDict()
        for batch in batches:
            result = self.predict_batch(batch)
            if result.reporter_name in results:
                raise ValueError(
                    "predict_grouped requires one group per reporter; duplicate "
                    f"reporter {result.reporter_name!r}"
                )
            results[result.reporter_name] = result
        return results

    def config_manifest(self) -> dict[str, Any]:
        """Return JSON-safe bridge metadata without duplicating model config."""

        payload: dict[str, Any] = {
            "schema_version": "ops-biological-baseline-captain-training-v1",
            "family": self.family,
            "wrapped_model_family": self.model.family,
            "interface": {
                "unit": "reporter_homogeneous_group",
                "forward": "forward_reporter_or_forward_grouped",
                "predict": "predict_reporter_or_predict_grouped",
                "loss_return": "differentiable_per_reporter_loss_and_components",
                "cross_reporter_reduction_owned_by_common_harness": True,
            },
            "supervision": {
                "endpoint_mask": "explicit_boolean_observed_mask",
                "zero_endpoint_target_is_valid": True,
                "phase_mge_mask": (
                    "explicit_required_when_reconstruction_weight_is_nonzero"
                ),
                "endpoint_and_phase_masks_are_independent": True,
            },
            "query": {
                "unit": "concrete_global_reporter_endpoint_id",
                "factorized_reporter_plus_endpoint": False,
                "semantic_metadata_used": False,
                "reporter_local_sparse_queries": True,
            },
            "ownership": {
                "learned_parameters": "wrapped_model_only",
                "data_loading": False,
                "split_construction": False,
                "optimizer": False,
                "scheduler": False,
                "checkpointing": False,
                "device_selection": False,
                "training_loop": False,
            },
            "phase_mge_loss_enabled": self.phase_mge_loss_enabled,
        }
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload


__all__ = [
    "CaptainReporterPredictionBatch",
    "CaptainReporterPredictionResult",
    "CaptainReporterTrainingBatch",
    "CaptainReporterTrainingResult",
    "OPSCaptainTrainingAdapter",
    "ReporterReference",
]

