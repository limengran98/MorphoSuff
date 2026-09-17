#!/usr/bin/env python3
"""Code-faithful sciPENN architecture adapted to frozen OPS phase172.

This module is deliberately limited to model and loss semantics.  It does not
load OPS arrays, construct splits, choose a device, create an optimizer, write
checkpoints, or start training.

The public sciPENN 0.9.6 implementation represents the union of proteins from
heterogeneous CITE-seq panels with one shared output matrix.  A row-wise panel
availability mask removes proteins that were not measured in the originating
dataset from the MSE/quantile loss.  Its source encoder is:

``BatchNorm -> Dropout -> Linear -> BatchNorm -> PReLU -> Dropout`` followed
by one shared ``RNNCell`` applied four times.  Three feed-forward blocks create
the inputs to recurrent updates two through four.  This adapter preserves
those mechanics.  It makes only the task-required substitutions:

* the common RNA input becomes the frozen 172D phase morphology vector;
* the protein union becomes the union of concrete ``reporter::endpoint``
  outputs, preserving each reporter's native 20/24/30/60/72 dimensions;
* panel availability and endpoint observation are explicit boolean masks;
* no value, including numerical zero, is interpreted as missing.

There are no reporter queries, semantic embeddings, reporter-specific experts,
or per-reporter heads.  The unified output matrix is a defining sciPENN
mechanism and keeps this baseline distinct from query-conditioned models.

Frozen upstream reference:

* repository commit ``34afb2008a076e13c40965a76d3dd31d0c331652``;
* ``src/sciPENN/Network/Model.py`` SHA256
  ``373213be9c4f985e32c76a8a40a58732833c522860365c46412d85d858cafc02``;
* ``src/sciPENN/Network/Layers.py`` SHA256
  ``411dc984348b2002d9d86d183c0c4f5c55a1fa4c5af6649879c1c150c000fdee``;
* ``src/sciPENN/Network/Losses.py`` SHA256
  ``b834105e3cce9e6698356456355b42cfefe37391637b9306e7c337923d40882d``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_PHASE_FEATURES = 172
DEFAULT_HIDDEN_DIM = 512
DEFAULT_DROPOUT = 0.25
DEFAULT_QUANTILES = (0.10, 0.25, 0.75, 0.90)
DEFAULT_RECURRENT_UPDATES = 4
SCIPENN_UPSTREAM_COMMIT = "34afb2008a076e13c40965a76d3dd31d0c331652"
SCIPENN_UPSTREAM_VERSION = "0.9.6"
SCIPENN_MODEL_SHA256 = (
    "373213be9c4f985e32c76a8a40a58732833c522860365c46412d85d858cafc02"
)
SCIPENN_LAYERS_SHA256 = (
    "411dc984348b2002d9d86d183c0c4f5c55a1fa4c5af6649879c1c150c000fdee"
)
SCIPENN_LOSSES_SHA256 = (
    "b834105e3cce9e6698356456355b42cfefe37391637b9306e7c337923d40882d"
)
SCIPENN_PREPROCESSING_SHA256 = (
    "127911a791ac9d8e7bb393e72925d81f20dd503565da9a0d68ad3ff8fdc2a677"
)
SCIPENN_DATALOADER_SHA256 = (
    "384ecafd554086a4af4e5dbda9fb6deba76e112d142316a8fed2bf48261941c2"
)
SCIPENN_SAMPLERS_SHA256 = (
    "b382401909f815687838345a080ee0ca405a7858cb39acc3b2727c7a96244b4d"
)
SCIPENN_DATALOADER_CONSTRUCTOR_SHA256 = (
    "cb4e779acb35639d8d9d1b9606e11bc80b0cf499781a00a76ff3ac9c91743fe9"
)
SCIPENN_API_SHA256 = (
    "a3333420758bc908b955c0e47e6b1a39a51ec6278b7a7f363d4d1660a43fdc8a"
)
FROZEN_SCIPENN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "original_methods"
    / "scipenn"
)


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive, found {result}")
    return result


def _dropout_probability(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"dropout must be finite and in [0, 1), found {result}")
    return result


def _normalise_dimensions(
    reporter_dimensions: Mapping[str, int],
) -> OrderedDict[str, int]:
    if not isinstance(reporter_dimensions, Mapping) or not reporter_dimensions:
        raise ValueError("reporter_dimensions must be a non-empty mapping")
    result: OrderedDict[str, int] = OrderedDict()
    for raw_name, raw_dimension in reporter_dimensions.items():
        name = str(raw_name)
        if not name:
            raise ValueError("Reporter names must be non-empty")
        if name in result:
            raise ValueError(f"Duplicate reporter name {name!r}")
        result[name] = _positive_int(
            raw_dimension, f"endpoint count for reporter {name!r}"
        )
    return result


def _normalise_quantiles(
    values: Sequence[float] | None,
) -> tuple[float, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise TypeError("quantile_levels must be a numerical sequence or None")
    result = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in result):
        raise ValueError("Every quantile level must be finite and in (0, 1)")
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError("quantile_levels must be strictly increasing and unique")
    return result


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Return scalar parameters for reporting, never as an admission gate."""

    values = module.parameters()
    if trainable_only:
        values = (value for value in values if value.requires_grad)
    return int(sum(value.numel() for value in values))


def verify_frozen_source() -> dict[str, str]:
    """Fail closed if any architecture/loss source used for fidelity moved."""

    expected = {
        "src/sciPENN/Network/Model.py": SCIPENN_MODEL_SHA256,
        "src/sciPENN/Network/Layers.py": SCIPENN_LAYERS_SHA256,
        "src/sciPENN/Network/Losses.py": SCIPENN_LOSSES_SHA256,
        "src/sciPENN/Preprocessing.py": SCIPENN_PREPROCESSING_SHA256,
        "src/sciPENN/Data_Infrastructure/DataLoader.py": (
            SCIPENN_DATALOADER_SHA256
        ),
        "src/sciPENN/Data_Infrastructure/Samplers.py": SCIPENN_SAMPLERS_SHA256,
        "src/sciPENN/Data_Infrastructure/DataLoader_Constructor.py": (
            SCIPENN_DATALOADER_CONSTRUCTOR_SHA256
        ),
        "src/sciPENN/sciPENN_API.py": SCIPENN_API_SHA256,
    }
    observed: dict[str, str] = {}
    for relative, digest in expected.items():
        path = FROZEN_SCIPENN_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen sciPENN source is missing: {path}")
        current = hashlib.sha256(path.read_bytes()).hexdigest()
        if current != digest:
            raise RuntimeError(
                f"Frozen sciPENN source hash mismatch for {relative}: "
                f"expected {digest}, found {current}"
            )
        observed[relative] = current
    return observed


class SciPENNInputBlock(nn.Module):
    """Exact layer order of upstream ``Input_Block``."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.bnorm_in = nn.BatchNorm1d(input_dim)
        self.dropout_in = nn.Dropout(dropout)
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.bnorm_out = nn.BatchNorm1d(hidden_dim)
        self.act = nn.PReLU()
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.bnorm_in(values)
        values = self.dropout_in(values)
        values = self.dense(values)
        values = self.bnorm_out(values)
        values = self.act(values)
        return self.dropout_out(values)


class SciPENNFeedForwardBlock(nn.Module):
    """Exact layer order of upstream ``FF_Block``."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.bnorm = nn.BatchNorm1d(hidden_dim)
        self.act = nn.PReLU()
        # Preserve the upstream public attribute spelling.
        self.Dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.dense(values)
        values = self.bnorm(values)
        values = self.act(values)
        return self.Dropout(values)


@dataclass(frozen=True)
class SciPENNOutput:
    """Dense predictions over the complete reporter-endpoint union."""

    mean: torch.Tensor
    quantiles: torch.Tensor | None
    embedding: torch.Tensor


@dataclass(frozen=True)
class SciPENNReporterOutput:
    """A reporter-local view sliced from one dense sciPENN evaluation."""

    reporter_name: str
    reporter_index: int
    mean: torch.Tensor
    quantiles: torch.Tensor | None
    embedding: torch.Tensor
    endpoint_offset: int


@dataclass(frozen=True)
class SciPENNLoss:
    """Differentiable observed-only loss plus auditable denominators."""

    total: torch.Tensor
    mse: torch.Tensor
    quantile_pinball: torch.Tensor
    observed_count: torch.Tensor
    dense_union_count: int
    reduction: str

    def as_dict(self) -> dict[str, torch.Tensor | int | str]:
        return {
            "total": self.total,
            "mse": self.mse,
            "quantile_pinball": self.quantile_pinball,
            "observed_count": self.observed_count,
            "dense_union_count": self.dense_union_count,
            "reduction": self.reduction,
        }


class OPSSciPENNAdapter(nn.Module):
    """sciPENN's shared recurrent trunk and masked union output for OPS.

    ``source_dense_mean`` reproduces upstream ``(loss * bools).mean()``:
    missing outputs have zero gradient, while the denominator is the complete
    dense union tensor.  ``observed_mean`` is exposed only as a clearly named
    OPS stabilization option; it changes normalization, not which values are
    supervised.  The source-faithful anchor must use ``source_dense_mean``.
    """

    family = "scipenn_ops_heterogeneous_panel_union"

    def __init__(
        self,
        reporter_dimensions: Mapping[str, int],
        *,
        input_dim: int = DEFAULT_PHASE_FEATURES,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        dropout: float = DEFAULT_DROPOUT,
        quantile_levels: Sequence[float] | None = DEFAULT_QUANTILES,
        loss_reduction: str = "source_dense_mean",
    ) -> None:
        super().__init__()
        dimensions = _normalise_dimensions(reporter_dimensions)
        self.reporter_dimensions = dimensions
        self.reporter_names = tuple(dimensions)
        self.output_dimensions = tuple(dimensions.values())
        self._reporter_to_index = {
            name: index for index, name in enumerate(self.reporter_names)
        }
        offsets = [0]
        for dimension in self.output_dimensions:
            offsets.append(offsets[-1] + dimension)
        self.endpoint_offsets = tuple(offsets)
        self.total_endpoints = int(offsets[-1])

        self.input_dim = _positive_int(input_dim, "input_dim")
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.dropout = _dropout_probability(dropout)
        self.quantile_levels = _normalise_quantiles(quantile_levels)
        self.loss_reduction = str(loss_reduction)
        if self.loss_reduction not in {"source_dense_mean", "observed_mean"}:
            raise ValueError(
                "loss_reduction must be source_dense_mean or observed_mean"
            )

        self.input_block = SciPENNInputBlock(
            self.input_dim, self.hidden_dim, self.dropout
        )
        # One and the same RNNCell owns every recurrent update, as upstream.
        self.rnn_cell = nn.RNNCell(self.hidden_dim, self.hidden_dim)
        self.skip_blocks = nn.ModuleList(
            SciPENNFeedForwardBlock(self.hidden_dim, self.dropout)
            for _ in range(DEFAULT_RECURRENT_UPDATES - 1)
        )
        self.mean_output = nn.Linear(self.hidden_dim, self.total_endpoints)
        if self.quantile_levels:
            self.quantile_output: nn.Linear | None = nn.Linear(
                self.hidden_dim,
                self.total_endpoints * len(self.quantile_levels),
            )
        else:
            self.quantile_output = None
        self.register_buffer(
            "_quantile_tensor",
            torch.tensor(self.quantile_levels, dtype=torch.float32),
            persistent=True,
        )

    def reporter_index(self, reporter: str | int) -> int:
        if isinstance(reporter, bool):
            raise TypeError("reporter must be a name or integer index")
        if isinstance(reporter, int):
            if not 0 <= reporter < len(self.reporter_names):
                raise IndexError(f"Reporter index outside registry: {reporter}")
            return reporter
        name = str(reporter)
        if name not in self._reporter_to_index:
            raise KeyError(f"Unknown reporter {name!r}")
        return self._reporter_to_index[name]

    def reporter_slice(self, reporter: str | int) -> slice:
        index = self.reporter_index(reporter)
        return slice(self.endpoint_offsets[index], self.endpoint_offsets[index + 1])

    def _validate_phase(self, phase_x: torch.Tensor) -> None:
        if not isinstance(phase_x, torch.Tensor):
            raise TypeError("phase_x must be a torch.Tensor")
        if phase_x.ndim != 2 or phase_x.shape[1] != self.input_dim:
            raise ValueError(
                f"phase_x must have shape [batch, {self.input_dim}], "
                f"found {tuple(phase_x.shape)}"
            )
        if not phase_x.is_floating_point():
            raise TypeError("phase_x must be floating point")
        if len(phase_x) == 0:
            raise ValueError("phase_x must contain at least one row")
        if not bool(torch.isfinite(phase_x).all()):
            raise ValueError(
                "phase172 is the always-observed source modality and must be finite"
            )

    def encode(self, phase_x: torch.Tensor) -> torch.Tensor:
        self._validate_phase(phase_x)
        values = self.input_block(phase_x)
        hidden = self.rnn_cell(values, torch.zeros_like(values))
        for block in self.skip_blocks:
            values = block(values)
            hidden = self.rnn_cell(values, hidden)
        return hidden

    def forward(self, phase_x: torch.Tensor) -> SciPENNOutput:
        embedding = self.encode(phase_x)
        mean = self.mean_output(embedding)
        if self.quantile_output is None:
            quantiles = None
        else:
            quantiles = self.quantile_output(embedding).reshape(
                len(phase_x), self.total_endpoints, len(self.quantile_levels)
            )
        return SciPENNOutput(mean=mean, quantiles=quantiles, embedding=embedding)

    def forward_reporter(
        self, phase_x: torch.Tensor, reporter: str | int
    ) -> SciPENNReporterOutput:
        """Evaluate the dense source model, then return one reporter slice."""

        index = self.reporter_index(reporter)
        name = self.reporter_names[index]
        endpoint_slice = self.reporter_slice(index)
        output = self(phase_x)
        quantiles = (
            None
            if output.quantiles is None
            else output.quantiles[:, endpoint_slice, :]
        )
        return SciPENNReporterOutput(
            reporter_name=name,
            reporter_index=index,
            mean=output.mean[:, endpoint_slice],
            quantiles=quantiles,
            embedding=output.embedding,
            endpoint_offset=self.endpoint_offsets[index],
        )

    def compute_loss(
        self,
        output: SciPENNOutput,
        target: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> SciPENNLoss:
        """Compute a mask-explicit sciPENN MSE plus optional quantile loss."""

        if not isinstance(output, SciPENNOutput):
            raise TypeError("output must be a SciPENNOutput")
        if target.shape != output.mean.shape:
            raise ValueError("target must match the dense union mean output")
        if not target.is_floating_point():
            raise TypeError("target must be floating point")
        if target.device != output.mean.device:
            raise ValueError("target and predictions must share a device")
        if observed_mask.dtype is not torch.bool:
            raise TypeError("observed_mask must have dtype torch.bool")
        if observed_mask.shape != target.shape:
            raise ValueError("observed_mask must match target")
        if observed_mask.device != target.device:
            raise ValueError("observed_mask and target must share a device")
        observed_count = observed_mask.sum()
        if int(observed_count.item()) == 0:
            raise ValueError("At least one reporter endpoint must be observed")
        observed_values = target.masked_select(observed_mask)
        if not bool(torch.isfinite(observed_values).all()):
            raise ValueError("Every observed endpoint target must be finite")

        # NaN/sentinel storage in missing positions is removed before any
        # arithmetic.  torch.where also makes their target gradients exactly 0.
        clean_target = torch.where(
            observed_mask, target, torch.zeros_like(target)
        )
        squared_error = (output.mean - clean_target).square()
        masked_mse_sum = torch.where(
            observed_mask, squared_error, torch.zeros_like(squared_error)
        ).sum()

        if output.quantiles is None:
            if self.quantile_levels or self.quantile_output is not None:
                raise ValueError("Quantile configuration and model output disagree")
            masked_quantile_sum = masked_mse_sum * 0.0
        else:
            expected = (*target.shape, len(self.quantile_levels))
            if output.quantiles.shape != expected:
                raise ValueError(
                    f"quantile output must have shape {expected}, found "
                    f"{tuple(output.quantiles.shape)}"
                )
            if not self.quantile_levels:
                raise ValueError("Quantile output exists without quantile levels")
            quantile_tensor = self._quantile_tensor.to(
                device=output.quantiles.device, dtype=output.quantiles.dtype
            )
            bias = output.quantiles - clean_target.unsqueeze(-1)
            over = bias.detach() > 0.0
            weight = torch.where(over, 1.0 - quantile_tensor, quantile_tensor)
            # Upstream quantile_loss averages quantiles before applying the
            # protein/panel mask.
            per_endpoint = (bias.abs() * weight).mean(dim=-1)
            masked_quantile_sum = torch.where(
                observed_mask, per_endpoint, torch.zeros_like(per_endpoint)
            ).sum()

        dense_union_count = int(target.numel())
        if self.loss_reduction == "source_dense_mean":
            denominator = target.new_tensor(float(dense_union_count))
        else:
            denominator = observed_count.to(dtype=target.dtype)
        mse = masked_mse_sum / denominator
        quantile_pinball = masked_quantile_sum / denominator
        return SciPENNLoss(
            total=mse + quantile_pinball,
            mse=mse,
            quantile_pinball=quantile_pinball,
            observed_count=observed_count,
            dense_union_count=dense_union_count,
            reduction=self.loss_reduction,
        )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return parameter_count(self, trainable_only=trainable_only)

    def config_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "ops-scipenn-adapter-v1",
            "family": self.family,
            "upstream": {
                "method": "sciPENN",
                "version": SCIPENN_UPSTREAM_VERSION,
                "commit": SCIPENN_UPSTREAM_COMMIT,
                "source_sha256": {
                    "Network/Model.py": SCIPENN_MODEL_SHA256,
                    "Network/Layers.py": SCIPENN_LAYERS_SHA256,
                    "Network/Losses.py": SCIPENN_LOSSES_SHA256,
                    "Preprocessing.py": SCIPENN_PREPROCESSING_SHA256,
                    "Data_Infrastructure/DataLoader.py": SCIPENN_DATALOADER_SHA256,
                    "Data_Infrastructure/Samplers.py": SCIPENN_SAMPLERS_SHA256,
                    "Data_Infrastructure/DataLoader_Constructor.py": (
                        SCIPENN_DATALOADER_CONSTRUCTOR_SHA256
                    ),
                    "sciPENN_API.py": SCIPENN_API_SHA256,
                },
                "checkpoint_loaded": False,
            },
            "architecture": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "input_block": (
                    "BatchNorm-Dropout-Linear-BatchNorm-PReLU-Dropout"
                ),
                "shared_rnn_cell": True,
                "recurrent_updates": DEFAULT_RECURRENT_UPDATES,
                "feedforward_blocks": len(self.skip_blocks),
                "output": "single_dense_union_linear_layer",
                "quantile_levels": list(self.quantile_levels),
                "celltype_transfer_head": False,
            },
            "ops_adaptation": {
                "source_modality": "frozen_phase172_always_observed",
                "target_union_unit": "concrete_reporter::endpoint",
                "reporter_dimensions": dict(self.reporter_dimensions),
                "total_endpoints": self.total_endpoints,
                "reporter_query": False,
                "semantic_metadata": False,
                "reporter_specific_experts": False,
                "reporter_specific_heads": False,
                "heterogeneous_panel_mask": True,
                "within_reporter_endpoint_mask": True,
                "numeric_zero_is_valid": True,
            },
            "loss": {
                "components": "MSE_plus_mean_quantile_pinball",
                "reduction": self.loss_reduction,
                "source_anchor_reduction": "source_dense_mean",
                "missing_entries_have_gradient": False,
            },
            "claim_boundary": {
                "exact_upstream_training_pipeline": False,
                "reason": (
                    "RNA/protein preprocessing and cell labels do not map to "
                    "frozen OPS phase172/reporter phenotypes"
                ),
                "architecture_and_panel_mask_mechanics_preserved": True,
            },
            "parameter_count": self.parameter_count(),
            "parameter_policy": {
                "matching_required": False,
                "explicit_limit": None,
                "role": "reporting_only",
            },
        }


__all__ = [
    "DEFAULT_DROPOUT",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_PHASE_FEATURES",
    "DEFAULT_QUANTILES",
    "DEFAULT_RECURRENT_UPDATES",
    "FROZEN_SCIPENN_ROOT",
    "OPSSciPENNAdapter",
    "SCIPENN_LAYERS_SHA256",
    "SCIPENN_LOSSES_SHA256",
    "SCIPENN_MODEL_SHA256",
    "SCIPENN_PREPROCESSING_SHA256",
    "SCIPENN_DATALOADER_SHA256",
    "SCIPENN_SAMPLERS_SHA256",
    "SCIPENN_DATALOADER_CONSTRUCTOR_SHA256",
    "SCIPENN_API_SHA256",
    "SCIPENN_UPSTREAM_COMMIT",
    "SCIPENN_UPSTREAM_VERSION",
    "SciPENNFeedForwardBlock",
    "SciPENNInputBlock",
    "SciPENNLoss",
    "SciPENNOutput",
    "SciPENNReporterOutput",
    "parameter_count",
    "verify_frozen_source",
]
