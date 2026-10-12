#!/usr/bin/env python3
"""Training bridge for the OPS-native 53-modality MIDAS adapter.

The common harness supplies one exact-paired local reporter batch at a time.
This bridge concatenates those rows while preserving every reporter as a
separate sparse modality with its native endpoint dimensionality.

No optimizer, data loader, split policy, checkpoint I/O, or launch logic lives
here.  Technical batch IDs are rejected because no trustworthy OPS technical
batch covariate has been frozen; reporter identity is a modality key only.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from ops_biological_baseline_midas_multimodal import (
    MIDASMultimodalOPS,
    MIDASMultimodalOPSConfig,
    MIDASMultimodalOPSOutput,
    ReporterObservation,
)


TRAINING_ADAPTER_SCHEMA = "ops-midas-53-modality-training-adapter-v2"


@dataclass(frozen=True)
class MIDASMultimodalReporterTrainingBatch:
    reporter: str
    phase_x: torch.Tensor
    endpoint_target: torch.Tensor
    endpoint_observed_mask: torch.Tensor
    phase_row_indices: torch.Tensor
    technical_batch_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class MIDASMultimodalAssembledBatch:
    phase_x: torch.Tensor
    phase_row_indices: torch.Tensor
    reporter_observations: OrderedDict[str, ReporterObservation]
    reporter_row_indices: OrderedDict[str, torch.Tensor]


@dataclass(frozen=True)
class MIDASMultimodalGroupedTrainingResult:
    output: MIDASMultimodalOPSOutput
    loss_components: dict[str, torch.Tensor]
    local_predictions: OrderedDict[str, torch.Tensor]
    reporter_row_indices: OrderedDict[str, torch.Tensor]

    @property
    def total_loss(self) -> torch.Tensor:
        return self.loss_components["total"]


class MIDASMultimodalOPSTrainingAdapter(nn.Module):
    """Reporter-local harness boundary for one shared multimodal MIDAS model."""

    family = "midas_ops_53_modality_training_adapter"

    def __init__(
        self,
        reporter_dimensions: Mapping[str, int],
        *,
        model: MIDASMultimodalOPS | None = None,
        config: MIDASMultimodalOPSConfig | None = None,
    ) -> None:
        super().__init__()
        if model is not None and config is not None:
            raise ValueError("Supply model or config, not both")
        self.model = model or MIDASMultimodalOPS(reporter_dimensions, config)
        expected = OrderedDict((str(k), int(v)) for k, v in reporter_dimensions.items())
        if self.model.reporter_dimensions != expected:
            raise ValueError("Model reporter modalities differ from bridge registry")
        self.reporter_dimensions = expected

    def _validate_local_batch(
        self, batch: MIDASMultimodalReporterTrainingBatch
    ) -> None:
        if not isinstance(batch, MIDASMultimodalReporterTrainingBatch):
            raise TypeError(
                "Every group must be MIDASMultimodalReporterTrainingBatch"
            )
        name = str(batch.reporter)
        if name not in self.reporter_dimensions:
            raise KeyError(f"Unknown reporter modality {name!r}")
        if batch.technical_batch_ids is not None:
            raise ValueError(
                "technical_batch_ids are disabled: no trustworthy OPS "
                "technical-batch variable is frozen, and reporter ID is forbidden"
            )
        self.model._validate_phase(batch.phase_x)
        dimension = self.reporter_dimensions[name]
        if batch.endpoint_target.shape != (len(batch.phase_x), dimension):
            raise ValueError(
                f"{name} target must have shape [{len(batch.phase_x)}, {dimension}]"
            )
        if not batch.endpoint_target.is_floating_point():
            raise TypeError("endpoint_target must be floating point")
        if batch.endpoint_target.device != batch.phase_x.device:
            raise ValueError("phase and endpoint target must share a device")
        if batch.endpoint_observed_mask.dtype is not torch.bool:
            raise TypeError("endpoint_observed_mask must have dtype torch.bool")
        if batch.endpoint_observed_mask.shape != batch.endpoint_target.shape:
            raise ValueError("endpoint_observed_mask must match endpoint_target")
        if batch.endpoint_observed_mask.device != batch.phase_x.device:
            raise ValueError("endpoint mask and phase must share a device")
        phase_rows = batch.phase_row_indices
        if (
            not isinstance(phase_rows, torch.Tensor)
            or phase_rows.dtype is not torch.long
            or phase_rows.ndim != 1
            or len(phase_rows) != len(batch.phase_x)
        ):
            raise TypeError(
                "phase_row_indices must be a 1D torch.long tensor matching phase"
            )
        if phase_rows.device != batch.phase_x.device:
            raise ValueError("phase row IDs and phase must share a device")
        if not bool(batch.endpoint_observed_mask.any()):
            raise ValueError(
                f"Reporter modality {name!r} has no observed endpoint in the group"
            )
        observed = batch.endpoint_target.masked_select(
            batch.endpoint_observed_mask
        )
        if not bool(torch.isfinite(observed).all()):
            raise ValueError("Observed reporter targets must be finite")
        # The deterministic cyclic sampler can cross from the end of one
        # shuffled cycle into the start of the next within a single optimizer
        # batch.  A cell may consequently be sampled twice even though the
        # frozen exact cache itself is unique by phase row.  Repeated exposure
        # is valid, but a repeated identity must still carry the identical
        # frozen phase vector, endpoint mask, and observed target.
        if len(torch.unique(phase_rows)) != len(phase_rows):
            order = torch.argsort(phase_rows, stable=True)
            sorted_rows = phase_rows.index_select(0, order)
            adjacent_same = sorted_rows[1:] == sorted_rows[:-1]
            sorted_phase = batch.phase_x.index_select(0, order)
            sorted_mask = batch.endpoint_observed_mask.index_select(0, order)
            clean_target = torch.where(
                batch.endpoint_observed_mask,
                batch.endpoint_target,
                torch.zeros_like(batch.endpoint_target),
            ).index_select(0, order)
            phase_consistent = torch.all(
                sorted_phase[1:] == sorted_phase[:-1], dim=1
            )
            mask_consistent = torch.all(
                sorted_mask[1:] == sorted_mask[:-1], dim=1
            )
            target_consistent = torch.all(
                clean_target[1:] == clean_target[:-1], dim=1
            )
            if not bool(
                (
                    (~adjacent_same)
                    | (phase_consistent & mask_consistent & target_consistent)
                ).all()
            ):
                raise ValueError(
                    f"Repeated phase row inside reporter modality {name!r} "
                    "mapped to inconsistent frozen data"
                )

    def assemble_grouped(
        self,
        batches: Sequence[MIDASMultimodalReporterTrainingBatch],
    ) -> MIDASMultimodalAssembledBatch:
        if isinstance(batches, (str, bytes)) or not isinstance(batches, Sequence):
            raise TypeError("batches must be a sequence")
        if not batches:
            raise ValueError("batches must not be empty")
        names: list[str] = []
        local_slices: OrderedDict[str, slice] = OrderedDict()
        phases: list[torch.Tensor] = []
        phase_rows: list[torch.Tensor] = []
        occurrence_indices: list[torch.Tensor] = []
        cursor = 0
        for batch in batches:
            self._validate_local_batch(batch)
            name = str(batch.reporter)
            if name in local_slices:
                raise ValueError(f"Duplicate reporter modality group {name!r}")
            names.append(name)
            phases.append(batch.phase_x)
            phase_rows.append(batch.phase_row_indices)
            # Pair a physical phase-row identity with its within-reporter
            # occurrence number.  Occurrence 0 still coalesces the same cell
            # across reporter modalities, while occurrence 1+ preserves a
            # cyclic sampler's repeated exposure instead of overwriting it.
            local_order = torch.argsort(batch.phase_row_indices, stable=True)
            local_sorted = batch.phase_row_indices.index_select(0, local_order)
            local_first = torch.ones_like(local_sorted, dtype=torch.bool)
            if len(local_first) > 1:
                local_first[1:] = local_sorted[1:] != local_sorted[:-1]
            local_positions = torch.arange(
                len(local_sorted),
                dtype=torch.long,
                device=local_sorted.device,
            )
            local_starts = torch.where(
                local_first, local_positions, torch.zeros_like(local_positions)
            )
            local_starts = torch.cummax(local_starts, dim=0).values
            local_sorted_occurrence = local_positions - local_starts
            local_occurrence = torch.empty_like(local_sorted_occurrence)
            local_occurrence.index_copy_(
                0, local_order, local_sorted_occurrence
            )
            occurrence_indices.append(local_occurrence)
            local_slices[name] = slice(cursor, cursor + len(batch.phase_x))
            cursor += len(batch.phase_x)
        concatenated_phase = torch.cat(phases, dim=0)
        concatenated_rows = torch.cat(phase_rows, dim=0)
        concatenated_occurrence = torch.cat(occurrence_indices, dim=0)
        occurrence_stride = int(concatenated_occurrence.max().item()) + 1
        composite_rows = (
            concatenated_rows * occurrence_stride + concatenated_occurrence
        )
        order = torch.argsort(composite_rows, stable=True)
        sorted_composite = composite_rows.index_select(0, order)
        first = torch.ones_like(sorted_composite, dtype=torch.bool)
        if len(first) > 1:
            first[1:] = sorted_composite[1:] != sorted_composite[:-1]
        first_positions = order.masked_select(first)
        unique_composite = sorted_composite.masked_select(first)
        unique_rows = concatenated_rows.index_select(0, first_positions)
        inverse = torch.searchsorted(unique_composite, composite_rows)
        phase = concatenated_phase.index_select(0, first_positions)
        # The composite is a collision-free encoding of identity and explicit
        # occurrence, never a numerical hash.  Equal keys must still point to
        # exactly the same frozen phase172 vector.
        if not torch.equal(
            concatenated_phase,
            phase.index_select(0, inverse),
        ):
            raise ValueError("One frozen phase row ID mapped to different phase values")
        reporter_rows: OrderedDict[str, torch.Tensor] = OrderedDict(
            (name, inverse[local_slices[name]]) for name in names
        )

        observations: OrderedDict[str, ReporterObservation] = OrderedDict()
        total_rows = len(phase)
        for batch, name in zip(batches, names):
            rows = reporter_rows[name]
            dimension = self.reporter_dimensions[name]
            clean_local = torch.where(
                batch.endpoint_observed_mask,
                batch.endpoint_target,
                torch.zeros_like(batch.endpoint_target),
            )
            values = batch.endpoint_target.new_zeros((total_rows, dimension))
            values = values.index_copy(0, rows, clean_local)
            mask = torch.zeros(
                (total_rows, dimension),
                dtype=torch.bool,
                device=batch.endpoint_target.device,
            )
            mask = mask.index_copy(0, rows, batch.endpoint_observed_mask)
            observations[name] = ReporterObservation(values, mask)
        return MIDASMultimodalAssembledBatch(
            phase_x=phase,
            phase_row_indices=unique_rows,
            reporter_observations=observations,
            reporter_row_indices=reporter_rows,
        )

    def forward_grouped(
        self,
        batches: Sequence[MIDASMultimodalReporterTrainingBatch],
        *,
        sample: bool | None = None,
    ) -> MIDASMultimodalGroupedTrainingResult:
        assembled = self.assemble_grouped(batches)
        names = tuple(assembled.reporter_observations)
        output = self.model(
            assembled.phase_x,
            assembled.reporter_observations,
            decode_reporters=names,
            sample=sample,
        )
        losses = self.model.compute_loss(
            output,
            assembled.phase_x,
            assembled.reporter_observations,
        )
        local: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, rows in assembled.reporter_row_indices.items():
            local[name] = output.reporters[name].mean.index_select(0, rows)
        return MIDASMultimodalGroupedTrainingResult(
            output=output,
            loss_components=losses,
            local_predictions=local,
            reporter_row_indices=assembled.reporter_row_indices,
        )

    def predict_reporter(self, phase_x: torch.Tensor, reporter: str) -> torch.Tensor:
        name = str(reporter)
        if name not in self.reporter_dimensions:
            raise KeyError(f"Unknown reporter modality {name!r}")
        return self.model.predict_reporter(phase_x, name)

    def predict_grouped(
        self, phase_by_reporter: Mapping[str, torch.Tensor]
    ) -> OrderedDict[str, torch.Tensor]:
        if not isinstance(phase_by_reporter, Mapping) or not phase_by_reporter:
            raise ValueError("phase_by_reporter must be a non-empty mapping")
        names = tuple(str(name) for name in phase_by_reporter)
        if len(set(names)) != len(names):
            raise ValueError("phase_by_reporter contains duplicate modalities")
        phases: list[torch.Tensor] = []
        sizes: list[int] = []
        for name in names:
            if name not in self.reporter_dimensions:
                raise KeyError(f"Unknown reporter modality {name!r}")
            phase = phase_by_reporter[name]
            self.model._validate_phase(phase)
            phases.append(phase)
            sizes.append(len(phase))
        combined = torch.cat(phases, dim=0)
        all_predictions = self.model.predict_reporters(combined, names)
        result: OrderedDict[str, torch.Tensor] = OrderedDict()
        cursor = 0
        for name, size in zip(names, sizes):
            result[name] = all_predictions[name][cursor : cursor + size]
            cursor += size
        return result

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return self.model.parameter_count(trainable_only=trainable_only)

    def config_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": TRAINING_ADAPTER_SCHEMA,
            "family": self.family,
            "grouped_contract": {
                "phase": "frozen_phase_row_always_observed_source",
                "phase_row_coalescing": (
                    "collision-free (frozen phase_row_index, within-reporter "
                    "sampled occurrence); matching occurrences coalesce across "
                    "modalities; never phase-value hashing"
                ),
                "cyclic_sampler_repeat_exposure": "preserved",
                "repeated_identity_consistency": (
                    "exact phase, observed mask, and observed target required"
                ),
                "targets": "independent_sparse_reporter_modalities",
                "non_group_reporter_rows": "absent_modality_false_mask",
                "numeric_zero_is_valid": True,
            },
            "batch_policy": {
                "enabled": False,
                "trusted_technical_batch_available": False,
                "reporter_id_allowed_as_batch": False,
                "latent_split_identifiable": False,
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
    "MIDASMultimodalAssembledBatch",
    "MIDASMultimodalGroupedTrainingResult",
    "MIDASMultimodalOPSTrainingAdapter",
    "MIDASMultimodalReporterTrainingBatch",
    "TRAINING_ADAPTER_SCHEMA",
]
