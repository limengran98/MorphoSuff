#!/usr/bin/env python3
"""Reporter-grouped training bridge for the OPS sciPENN adapter.

The common OPS harness naturally yields one exact-paired reporter block at a
time.  Native sciPENN instead concatenates cells from datasets with different
protein panels, predicts the complete protein union for every row, and applies
a row-wise dataset-panel mask.  This bridge performs that exact structural
translation for heterogeneous OPS reporters:

* every reporter remains a distinct 20/24/30/60/72D measured panel;
* batches are concatenated as separate observations, matching sciPENN (same
  phase cell IDs are not coalesced across reporter panels);
* the complete endpoint union of the reporter panels participating in the
  current run is predicted for every row (12/470D in panel12 development and
  52/1,604D only in a future full52 run);
* a structural panel mask and a within-panel endpoint mask are kept separate;
* only their intersection contributes to loss.

The bridge owns no learned parameters, optimizer, data loader, split policy,
checkpointing, device selection, or launch logic.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ops_biological_baseline_scipenn import (
    OPSSciPENNAdapter,
    SciPENNLoss,
    SciPENNOutput,
)


TRAINING_ADAPTER_SCHEMA = "ops-scipenn-training-adapter-v1"


@dataclass(frozen=True)
class SciPENNReporterTrainingBatch:
    """One reporter-homogeneous OPS minibatch.

    ``endpoint_observed_mask`` captures missing endpoints *inside* the
    reporter panel.  The reporter name itself determines the separate
    structural panel-availability mask.  Numerical target zero remains valid.
    """

    reporter: str
    phase_x: torch.Tensor
    endpoint_target: torch.Tensor
    endpoint_observed_mask: torch.Tensor


@dataclass(frozen=True)
class SciPENNAssembledBatch:
    """Dense sciPENN union representation and both missingness mechanisms."""

    phase_x: torch.Tensor
    union_target: torch.Tensor
    panel_availability_mask: torch.Tensor
    endpoint_observed_mask: torch.Tensor
    effective_observed_mask: torch.Tensor
    reporter_row_slices: OrderedDict[str, slice]


@dataclass(frozen=True)
class SciPENNGroupedTrainingResult:
    output: SciPENNOutput
    loss_components: SciPENNLoss
    local_predictions: OrderedDict[str, torch.Tensor]
    local_quantiles: OrderedDict[str, torch.Tensor | None]
    assembled: SciPENNAssembledBatch

    @property
    def total_loss(self) -> torch.Tensor:
        return self.loss_components.total


class OPSSciPENNTrainingAdapter(nn.Module):
    """Parameter-free bridge around one unified :class:`OPSSciPENNAdapter`."""

    family = "scipenn_ops_heterogeneous_panel_training_adapter"

    def __init__(
        self,
        reporter_dimensions: Mapping[str, int],
        *,
        model: OPSSciPENNAdapter | None = None,
        model_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if model is not None and model_options is not None:
            raise ValueError("Supply model or model_options, not both")
        expected = OrderedDict(
            (str(name), int(dimension))
            for name, dimension in reporter_dimensions.items()
        )
        if not expected:
            raise ValueError("reporter_dimensions must not be empty")
        self.model = model or OPSSciPENNAdapter(
            expected, **dict(model_options or {})
        )
        if self.model.reporter_dimensions != expected:
            raise ValueError("Model reporter registry differs from bridge registry")
        self.reporter_dimensions = expected

    def _validate_local_batch(self, batch: SciPENNReporterTrainingBatch) -> None:
        if not isinstance(batch, SciPENNReporterTrainingBatch):
            raise TypeError("Every group must be SciPENNReporterTrainingBatch")
        name = str(batch.reporter)
        if name not in self.reporter_dimensions:
            raise KeyError(f"Unknown reporter panel {name!r}")
        self.model._validate_phase(batch.phase_x)
        expected = (len(batch.phase_x), self.reporter_dimensions[name])
        if not isinstance(batch.endpoint_target, torch.Tensor):
            raise TypeError("endpoint_target must be a torch.Tensor")
        if tuple(batch.endpoint_target.shape) != expected:
            raise ValueError(
                f"{name} endpoint_target must have shape {expected}, found "
                f"{tuple(batch.endpoint_target.shape)}"
            )
        if not batch.endpoint_target.is_floating_point():
            raise TypeError("endpoint_target must be floating point")
        if batch.endpoint_target.device != batch.phase_x.device:
            raise ValueError("phase_x and endpoint_target must share a device")
        if not isinstance(batch.endpoint_observed_mask, torch.Tensor):
            raise TypeError("endpoint_observed_mask must be a torch.Tensor")
        if batch.endpoint_observed_mask.dtype is not torch.bool:
            raise TypeError("endpoint_observed_mask must have dtype torch.bool")
        if tuple(batch.endpoint_observed_mask.shape) != expected:
            raise ValueError("endpoint_observed_mask must match endpoint_target")
        if batch.endpoint_observed_mask.device != batch.phase_x.device:
            raise ValueError("phase_x and endpoint_observed_mask must share a device")
        if not bool(batch.endpoint_observed_mask.any()):
            raise ValueError(f"Reporter panel {name!r} has no observed endpoint")
        observed = batch.endpoint_target.masked_select(
            batch.endpoint_observed_mask
        )
        if not bool(torch.isfinite(observed).all()):
            raise ValueError("Every observed endpoint target must be finite")

    def assemble_grouped(
        self, batches: Sequence[SciPENNReporterTrainingBatch]
    ) -> SciPENNAssembledBatch:
        if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
            raise TypeError("batches must be a sequence")
        if not batches:
            raise ValueError("batches must not be empty")

        seen: set[str] = set()
        phases: list[torch.Tensor] = []
        row_slices: OrderedDict[str, slice] = OrderedDict()
        cursor = 0
        for batch in batches:
            self._validate_local_batch(batch)
            name = str(batch.reporter)
            if name in seen:
                raise ValueError(
                    f"Duplicate reporter group {name!r} in one sciPENN union batch"
                )
            seen.add(name)
            phases.append(batch.phase_x)
            row_slices[name] = slice(cursor, cursor + len(batch.phase_x))
            cursor += len(batch.phase_x)

        phase_x = torch.cat(phases, dim=0)
        union_target = phase_x.new_zeros((cursor, self.model.total_endpoints))
        panel_mask = torch.zeros(
            (cursor, self.model.total_endpoints),
            dtype=torch.bool,
            device=phase_x.device,
        )
        endpoint_mask = torch.zeros_like(panel_mask)

        for batch in batches:
            name = str(batch.reporter)
            rows = row_slices[name]
            endpoints = self.model.reporter_slice(name)
            # Missing storage may be NaN or another sentinel; sanitize it
            # before placement.  A real observed zero remains untouched.
            local_target = torch.where(
                batch.endpoint_observed_mask,
                batch.endpoint_target,
                torch.zeros_like(batch.endpoint_target),
            )
            union_target[rows, endpoints] = local_target
            panel_mask[rows, endpoints] = True
            endpoint_mask[rows, endpoints] = batch.endpoint_observed_mask

        effective = panel_mask & endpoint_mask
        if not bool(effective.any()):
            raise ValueError("Assembled sciPENN batch has no observed endpoint")
        return SciPENNAssembledBatch(
            phase_x=phase_x,
            union_target=union_target,
            panel_availability_mask=panel_mask,
            endpoint_observed_mask=endpoint_mask,
            effective_observed_mask=effective,
            reporter_row_slices=row_slices,
        )

    def forward_grouped(
        self, batches: Sequence[SciPENNReporterTrainingBatch]
    ) -> SciPENNGroupedTrainingResult:
        assembled = self.assemble_grouped(batches)
        output = self.model(assembled.phase_x)
        losses = self.model.compute_loss(
            output,
            assembled.union_target,
            assembled.effective_observed_mask,
        )
        local_predictions: OrderedDict[str, torch.Tensor] = OrderedDict()
        local_quantiles: OrderedDict[str, torch.Tensor | None] = OrderedDict()
        for name, rows in assembled.reporter_row_slices.items():
            endpoints = self.model.reporter_slice(name)
            local_predictions[name] = output.mean[rows, endpoints]
            local_quantiles[name] = (
                None
                if output.quantiles is None
                else output.quantiles[rows, endpoints, :]
            )
        return SciPENNGroupedTrainingResult(
            output=output,
            loss_components=losses,
            local_predictions=local_predictions,
            local_quantiles=local_quantiles,
            assembled=assembled,
        )

    def predict_reporter(
        self, phase_x: torch.Tensor, reporter: str
    ) -> torch.Tensor:
        return self.model.forward_reporter(phase_x, reporter).mean

    def predict_grouped(
        self, phase_by_reporter: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        if not isinstance(phase_by_reporter, Mapping) or not phase_by_reporter:
            raise ValueError("phase_by_reporter must be a non-empty mapping")
        names = tuple(str(name) for name in phase_by_reporter)
        if len(set(names)) != len(names):
            raise ValueError("phase_by_reporter contains duplicate reporters")
        phases: list[torch.Tensor] = []
        sizes: list[int] = []
        for name in names:
            if name not in self.reporter_dimensions:
                raise KeyError(f"Unknown reporter panel {name!r}")
            phase = phase_by_reporter[name]
            self.model._validate_phase(phase)
            phases.append(phase)
            sizes.append(len(phase))
        combined = torch.cat(phases, dim=0)
        output = self.model(combined)
        predictions: OrderedDict[str, torch.Tensor] = OrderedDict()
        cursor = 0
        for name, size in zip(names, sizes):
            endpoints = self.model.reporter_slice(name)
            predictions[name] = output.mean[
                cursor : cursor + size, endpoints
            ]
            cursor += size
        return predictions

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return self.model.parameter_count(trainable_only=trainable_only)

    def config_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_ADAPTER_SCHEMA,
            "family": self.family,
            "grouped_contract": {
                "source": "phase172_always_observed",
                "target": "dense_union_with_reporter_panel_mask",
                "same_phase_cell_across_panels": (
                    "separate_rows_as_in_native_scipenn_dataset_concatenation"
                ),
                "panel_availability_mask": "reporter_block_structural",
                "endpoint_observed_mask": "within_reporter_label_availability",
                "effective_loss_mask": "intersection",
                "numeric_zero_is_valid": True,
                "missing_target_storage_sanitized_before_arithmetic": True,
            },
            "model": self.model.config_manifest(),
            "parameter_count": self.parameter_count(),
            "parameter_policy": {
                "matching_required": False,
                "explicit_limit": None,
                "role": "reporting_only",
            },
        }


__all__ = [
    "OPSSciPENNTrainingAdapter",
    "SciPENNAssembledBatch",
    "SciPENNGroupedTrainingResult",
    "SciPENNReporterTrainingBatch",
    "TRAINING_ADAPTER_SCHEMA",
]
