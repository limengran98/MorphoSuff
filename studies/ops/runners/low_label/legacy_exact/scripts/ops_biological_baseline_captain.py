#!/usr/bin/env python3
"""OPS-native, code-faithful adapter of CAPTAIN's protein query module.

This module is intentionally architecture-only.  It does not load OPS data,
construct splits, own a training loop, write artifacts, or select a device.

The executable CAPTAIN v1.0.0 protein module uses learned protein tokens as
queries and RNA tokens as keys/values in every attention block.  When context
is present, there is no protein-query self-attention.  The OPS adapter keeps
that behaviour while making only the modality/schema changes required here:

* each of the 172 continuous phase features becomes one
  ``feature-ID embedding + value embedding`` token;
* each concrete reporter-endpoint is one global learned token (there is no
  reporter-ID/endpoint-ID factorisation and no semantic metadata);
* endpoint tokens query phase-feature tokens through cross-attention;
* one scalar mean head, and optionally one quantile head, is shared by every
  endpoint token;
* an independently switchable masked phase-feature reconstruction objective is
  the OPS analogue of CAPTAIN's masked gene-expression (MGE) objective;
* labels are supervised only through an explicit observed mask.  A numerical
  target value of zero is valid and is never interpreted as missing.

Because all endpoint operations are token-wise and there is deliberately no
query self-attention, evaluating a sparse query subset is mathematically the
same as selecting those columns from a dense query evaluation (up to ordinary
stochastic dropout while the module is in training mode).

Upstream executable references frozen in this workspace:

``external/original_methods/captain/pretrain/protein_model.py``
    SHA256 ``a191d2c8...6dd8618``
``external/original_methods/captain/pretrain/torchrun.py``
    SHA256 ``e8026be4...15bf03``

No upstream checkpoint is loaded: CAPTAIN's gene/protein vocabularies are not
aligned to OPS morphology features or fluorescence phenotype endpoints.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


DEFAULT_PHASE_FEATURES = 172
DEFAULT_QUANTILES = (0.10, 0.25, 0.75, 0.90)
UPSTREAM_RELEASE = "v1.0.0"
UPSTREAM_RELEASE_COMMIT = "19a94cb"
UPSTREAM_PROTEIN_MODEL_SHA256 = (
    "a191d2c8b9240ffe01c707eddd7c334bfbf150fab8e6fe3ea3821b2456dd8618"
)
UPSTREAM_TORCHRUN_SHA256 = (
    "e8026be4abab09a968f94bf5ffabcbf1d9c0f7f21eff9b8c7883713b4f15bf03"
)
FROZEN_CAPTAIN_ROOT = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "original_methods"
    / "captain"
)


def _positive_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{label} must be positive, found {value}")
    return value


def _nonnegative_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer, not bool")
    value = int(value)
    if value < 0:
        raise ValueError(f"{label} must be nonnegative, found {value}")
    return value


def _probability(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1), found {value}")
    return value


def _nonnegative_float(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative, found {value}")
    return value


def _normalise_head_dimensions(
    head_dimensions: Mapping[str, int],
) -> OrderedDict[str, int]:
    if not isinstance(head_dimensions, Mapping) or not head_dimensions:
        raise ValueError("head_dimensions must be a non-empty ordered mapping")
    result: OrderedDict[str, int] = OrderedDict()
    for raw_name, raw_dimension in head_dimensions.items():
        name = str(raw_name)
        if not name:
            raise ValueError("Reporter names must be non-empty")
        if name in result:
            raise ValueError(f"Duplicate reporter name: {name!r}")
        result[name] = _positive_int(
            raw_dimension, f"endpoint count for reporter {name!r}"
        )
    return result


def _normalise_quantiles(
    levels: Sequence[float] | None,
) -> tuple[float, ...]:
    if levels is None:
        return ()
    if isinstance(levels, (str, bytes)):
        raise TypeError("quantile_levels must be a numerical sequence or None")
    result = tuple(float(level) for level in levels)
    if not result:
        return ()
    if any(not math.isfinite(level) or not 0.0 < level < 1.0 for level in result):
        raise ValueError("Every quantile level must be finite and in (0, 1)")
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError("quantile_levels must be strictly increasing and unique")
    return result


def parameter_count(module: nn.Module, *, trainable_only: bool = False) -> int:
    """Return the scalar parameter count; never use it as an admission gate."""

    values = module.parameters()
    if trainable_only:
        values = (value for value in values if value.requires_grad)
    return int(sum(value.numel() for value in values))


@dataclass(frozen=True)
class CaptainPrediction:
    """Predictions for one shared one-dimensional endpoint query set.

    ``mean`` has shape ``[batch, queries]``.  ``quantiles`` is either ``None``
    or ``[batch, queries, n_quantiles]``.  ``endpoint_ids`` is the exact global
    query ID vector, shape ``[queries]``, shared by all rows in the batch.
    """

    mean: torch.Tensor
    quantiles: torch.Tensor | None
    endpoint_ids: torch.Tensor
    phase_reconstruction: torch.Tensor | None = None
    phase_reconstruction_mask: torch.Tensor | None = None


@dataclass(frozen=True)
class CaptainLoss:
    """Observed-only loss components returned without detaching gradients."""

    total: torch.Tensor
    mean_mse: torch.Tensor
    quantile_pinball: torch.Tensor
    phase_reconstruction_mse: torch.Tensor
    observed_count: torch.Tensor
    reconstructed_feature_count: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total,
            "mean_mse": self.mean_mse,
            "quantile_pinball": self.quantile_pinball,
            "phase_reconstruction_mse": self.phase_reconstruction_mse,
            "observed_count": self.observed_count,
            "reconstructed_feature_count": self.reconstructed_feature_count,
        }


class _PositionwiseFeedForward(nn.Module):
    """CAPTAIN-code-style residual feed-forward block."""

    def __init__(self, model_dim: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = model_dim * multiplier
        self.network = nn.Sequential(
            nn.Linear(model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values + self.network(values))


class _PhaseSelfAttentionBlock(nn.Module):
    """Within-cell self-attention over named phase-feature tokens only."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_multiplier: int,
        attention_dropout: float,
        feedforward_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feedforward = _PositionwiseFeedForward(
            model_dim, feedforward_multiplier, feedforward_dropout
        )

    def forward(
        self, values: torch.Tensor, key_padding_mask: torch.Tensor | None
    ) -> torch.Tensor:
        attended, _ = self.attention(
            query=values,
            key=values,
            value=values,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        values = self.attention_norm(values + self.attention_dropout(attended))
        return self.feedforward(values)


class CaptainCrossAttentionBlock(nn.Module):
    """Endpoint-query to phase-context attention with no query self-attention."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_multiplier: int,
        attention_dropout: float,
        feedforward_dropout: float,
    ) -> None:
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.attention_norm = nn.LayerNorm(model_dim)
        self.feedforward = _PositionwiseFeedForward(
            model_dim, feedforward_multiplier, feedforward_dropout
        )

    def forward(
        self,
        endpoint_queries: torch.Tensor,
        phase_context: torch.Tensor,
        phase_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # This direction is the behaviour in CAPTAIN's public protein_model.py:
        # query=x (protein/endpoint), key=context, value=context (RNA/phase).
        attended, _ = self.cross_attention(
            query=endpoint_queries,
            key=phase_context,
            value=phase_context,
            key_padding_mask=phase_key_padding_mask,
            need_weights=False,
        )
        endpoint_queries = self.attention_norm(
            endpoint_queries + self.attention_dropout(attended)
        )
        return self.feedforward(endpoint_queries)


def _shared_head(
    model_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = model_dim
    for raw_width in hidden_dims:
        width = _positive_int(raw_width, "prediction-head hidden width")
        layers.extend([nn.Linear(current, width), nn.ReLU()])
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class OPSCaptainAdapter(nn.Module):
    """CAPTAIN-style global endpoint-token model for frozen OPS phase172.

    Parameters are deliberately not capped or matched to the 4.31M ResMLP.
    Width, depth, head count, and head capacity are ordinary model-family
    hyperparameters to be selected from validation data by the common harness.

    The default endpoint module retains CAPTAIN's native 512D, six-layer,
    eight-head shape.  The two-layer phase encoder is the necessary OPS input
    adaptation and is separately configurable.
    """

    family = "ops_captain_endpoint_cross_attention"

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        *,
        input_dim: int = DEFAULT_PHASE_FEATURES,
        model_dim: int = 512,
        num_heads: int = 8,
        phase_encoder_depth: int = 2,
        cross_attention_depth: int = 6,
        feedforward_multiplier: int = 4,
        mean_head_hidden_dims: Sequence[int] = (128, 64),
        quantile_head_hidden_dims: Sequence[int] = (128, 64),
        quantile_levels: Sequence[float] | None = DEFAULT_QUANTILES,
        phase_reconstruction_enabled: bool = True,
        phase_reconstruction_head_hidden_dims: Sequence[int] = (128, 64),
        embedding_dropout: float = 0.0,
        attention_dropout: float = 0.1,
        feedforward_dropout: float = 0.1,
        mean_loss_weight: float | None = None,
        quantile_loss_weight: float | None = None,
        phase_reconstruction_loss_weight: float | None = None,
    ) -> None:
        super().__init__()
        dimensions = _normalise_head_dimensions(head_dimensions)
        self.reporter_names = tuple(dimensions)
        self.output_dimensions = tuple(dimensions.values())
        self._reporter_to_index = {
            name: index for index, name in enumerate(self.reporter_names)
        }
        offsets = [0]
        for dimension in self.output_dimensions:
            offsets.append(offsets[-1] + dimension)
        self.endpoint_offsets = tuple(offsets)
        self.total_endpoints = offsets[-1]

        self.input_dim = _positive_int(input_dim, "input_dim")
        self.model_dim = _positive_int(model_dim, "model_dim")
        self.num_heads = _positive_int(num_heads, "num_heads")
        if self.model_dim % self.num_heads != 0:
            raise ValueError(
                f"model_dim={self.model_dim} must be divisible by "
                f"num_heads={self.num_heads}"
            )
        self.phase_encoder_depth = _nonnegative_int(
            phase_encoder_depth, "phase_encoder_depth"
        )
        self.cross_attention_depth = _positive_int(
            cross_attention_depth, "cross_attention_depth"
        )
        self.feedforward_multiplier = _positive_int(
            feedforward_multiplier, "feedforward_multiplier"
        )
        self.mean_head_hidden_dims = tuple(
            _positive_int(value, "mean head hidden width")
            for value in mean_head_hidden_dims
        )
        self.quantile_head_hidden_dims = tuple(
            _positive_int(value, "quantile head hidden width")
            for value in quantile_head_hidden_dims
        )
        if not isinstance(phase_reconstruction_enabled, bool):
            raise TypeError("phase_reconstruction_enabled must be bool")
        self.phase_reconstruction_enabled = phase_reconstruction_enabled
        self.phase_reconstruction_head_hidden_dims = tuple(
            _positive_int(value, "phase reconstruction head hidden width")
            for value in phase_reconstruction_head_hidden_dims
        )
        self.quantile_levels_tuple = _normalise_quantiles(quantile_levels)
        self.embedding_dropout_probability = _probability(
            embedding_dropout, "embedding_dropout"
        )
        self.attention_dropout_probability = _probability(
            attention_dropout, "attention_dropout"
        )
        self.feedforward_dropout_probability = _probability(
            feedforward_dropout, "feedforward_dropout"
        )

        quantiles_enabled = bool(self.quantile_levels_tuple)
        if mean_loss_weight is None:
            mean_loss_weight = 1.0
            if quantiles_enabled:
                mean_loss_weight -= 0.2
            if self.phase_reconstruction_enabled:
                mean_loss_weight -= 0.2
        if quantile_loss_weight is None:
            quantile_loss_weight = 0.2 if quantiles_enabled else 0.0
        if phase_reconstruction_loss_weight is None:
            phase_reconstruction_loss_weight = (
                0.2 if self.phase_reconstruction_enabled else 0.0
            )
        self.mean_loss_weight = _nonnegative_float(
            mean_loss_weight, "mean_loss_weight"
        )
        self.quantile_loss_weight = _nonnegative_float(
            quantile_loss_weight, "quantile_loss_weight"
        )
        self.phase_reconstruction_loss_weight = _nonnegative_float(
            phase_reconstruction_loss_weight,
            "phase_reconstruction_loss_weight",
        )
        if not quantiles_enabled and self.quantile_loss_weight != 0.0:
            raise ValueError(
                "quantile_loss_weight must be zero when quantile_levels is disabled"
            )
        if (
            not self.phase_reconstruction_enabled
            and self.phase_reconstruction_loss_weight != 0.0
        ):
            raise ValueError(
                "phase_reconstruction_loss_weight must be zero when phase "
                "reconstruction is disabled"
            )
        if (
            self.mean_loss_weight == 0.0
            and self.quantile_loss_weight == 0.0
            and self.phase_reconstruction_loss_weight == 0.0
        ):
            raise ValueError("At least one loss weight must be positive")

        self.feature_id_embedding = nn.Embedding(self.input_dim, self.model_dim)
        self.feature_value_embedding = nn.Linear(1, self.model_dim)
        self.phase_token_norm = nn.LayerNorm(self.model_dim)
        self.phase_token_dropout = nn.Dropout(self.embedding_dropout_probability)
        if self.phase_reconstruction_enabled:
            self.phase_mask_token: nn.Parameter | None = nn.Parameter(
                torch.zeros(self.model_dim)
            )
        else:
            self.register_parameter("phase_mask_token", None)
        self.register_buffer(
            "phase_feature_ids",
            torch.arange(self.input_dim, dtype=torch.long),
            persistent=True,
        )
        self.phase_encoder = nn.ModuleList(
            [
                _PhaseSelfAttentionBlock(
                    self.model_dim,
                    self.num_heads,
                    self.feedforward_multiplier,
                    self.attention_dropout_probability,
                    self.feedforward_dropout_probability,
                )
                for _ in range(self.phase_encoder_depth)
            ]
        )
        self.phase_context_norm = nn.LayerNorm(self.model_dim)
        if self.phase_reconstruction_enabled:
            self.phase_reconstruction_head: nn.Module | None = _shared_head(
                self.model_dim,
                self.phase_reconstruction_head_hidden_dims,
                1,
            )
        else:
            self.phase_reconstruction_head = None

        # A single embedding table over concrete global reporter-endpoint IDs.
        # There is intentionally no reporter embedding and no semantic input.
        self.endpoint_token_embedding = nn.Embedding(
            self.total_endpoints, self.model_dim
        )
        self.endpoint_token_dropout = nn.Dropout(self.embedding_dropout_probability)
        self.register_buffer(
            "all_endpoint_ids",
            torch.arange(self.total_endpoints, dtype=torch.long),
            persistent=True,
        )
        self.cross_attention_layers = nn.ModuleList(
            [
                CaptainCrossAttentionBlock(
                    self.model_dim,
                    self.num_heads,
                    self.feedforward_multiplier,
                    self.attention_dropout_probability,
                    self.feedforward_dropout_probability,
                )
                for _ in range(self.cross_attention_depth)
            ]
        )
        self.endpoint_norm = nn.LayerNorm(self.model_dim)
        self.mean_head = _shared_head(
            self.model_dim, self.mean_head_hidden_dims, 1
        )
        if quantiles_enabled:
            self.quantile_head: nn.Module | None = _shared_head(
                self.model_dim,
                self.quantile_head_hidden_dims,
                len(self.quantile_levels_tuple),
            )
        else:
            self.quantile_head = None
        self.register_buffer(
            "quantile_levels",
            torch.tensor(self.quantile_levels_tuple, dtype=torch.float32),
            persistent=True,
        )

    @property
    def head_dimensions(self) -> OrderedDict[str, int]:
        return OrderedDict(zip(self.reporter_names, self.output_dimensions))

    @property
    def quantiles_enabled(self) -> bool:
        return self.quantile_head is not None

    def reporter_index(self, reporter: str | int | torch.Tensor) -> int:
        if isinstance(reporter, torch.Tensor):
            if reporter.ndim != 0:
                raise TypeError("reporter tensor must be scalar")
            reporter = int(reporter.item())
        if isinstance(reporter, bool):
            raise TypeError("Boolean is not a reporter index")
        if isinstance(reporter, int):
            if not 0 <= reporter < len(self.reporter_names):
                raise IndexError(
                    f"Reporter index {reporter} outside "
                    f"[0, {len(self.reporter_names)})"
                )
            return reporter
        name = str(reporter)
        try:
            return self._reporter_to_index[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown reporter {name!r}; available={self.reporter_names}"
            ) from error

    def endpoint_ids_for_reporter(
        self,
        reporter: str | int | torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        index = self.reporter_index(reporter)
        return torch.arange(
            self.endpoint_offsets[index],
            self.endpoint_offsets[index + 1],
            dtype=torch.long,
            device=device,
        )

    def _validate_phase(
        self,
        phase_x: torch.Tensor,
        phase_observed_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(phase_x, torch.Tensor):
            raise TypeError("phase_x must be a torch.Tensor")
        if phase_x.ndim != 2 or phase_x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected phase_x [batch, {self.input_dim}], "
                f"found {tuple(phase_x.shape)}"
            )
        if not phase_x.is_floating_point():
            raise TypeError("phase_x must be floating point")
        if phase_observed_mask is None:
            if not bool(torch.isfinite(phase_x).all()):
                raise ValueError(
                    "Non-finite phase_x requires an explicit phase_observed_mask"
                )
            return phase_x, None
        if not isinstance(phase_observed_mask, torch.Tensor):
            raise TypeError("phase_observed_mask must be a torch.Tensor")
        if phase_observed_mask.dtype is not torch.bool:
            raise TypeError("phase_observed_mask must have dtype torch.bool")
        if phase_observed_mask.shape != phase_x.shape:
            raise ValueError(
                "phase_observed_mask must match phase_x shape, found "
                f"{tuple(phase_observed_mask.shape)} and {tuple(phase_x.shape)}"
            )
        if phase_observed_mask.device != phase_x.device:
            raise ValueError("phase_observed_mask and phase_x must share a device")
        if bool((phase_observed_mask.sum(dim=1) == 0).any()):
            raise ValueError("Every cell must retain at least one observed phase feature")
        if not bool(torch.isfinite(phase_x[phase_observed_mask]).all()):
            raise ValueError("Observed phase features must be finite")
        safe_phase = torch.where(
            phase_observed_mask, phase_x, torch.zeros_like(phase_x)
        )
        return safe_phase, ~phase_observed_mask

    def _validate_phase_reconstruction_mask(
        self,
        phase_x: torch.Tensor,
        phase_observed_mask: torch.Tensor | None,
        phase_reconstruction_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if phase_reconstruction_mask is None:
            return None
        if not self.phase_reconstruction_enabled:
            raise ValueError(
                "phase_reconstruction_mask was supplied while phase "
                "reconstruction is disabled"
            )
        if not isinstance(phase_reconstruction_mask, torch.Tensor):
            raise TypeError("phase_reconstruction_mask must be a torch.Tensor")
        if phase_reconstruction_mask.dtype is not torch.bool:
            raise TypeError("phase_reconstruction_mask must have dtype torch.bool")
        if phase_reconstruction_mask.shape != phase_x.shape:
            raise ValueError(
                "phase_reconstruction_mask must match phase_x shape, found "
                f"{tuple(phase_reconstruction_mask.shape)} and "
                f"{tuple(phase_x.shape)}"
            )
        if phase_reconstruction_mask.device != phase_x.device:
            raise ValueError(
                "phase_reconstruction_mask and phase_x must share a device"
            )
        if phase_observed_mask is not None and bool(
            (phase_reconstruction_mask & ~phase_observed_mask).any()
        ):
            raise ValueError(
                "Only genuinely observed phase features may be selected for "
                "masked reconstruction"
            )
        return phase_reconstruction_mask

    def encode_phase(
        self,
        phase_x: torch.Tensor,
        phase_observed_mask: torch.Tensor | None = None,
        phase_reconstruction_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return phase tokens ``[batch, 172, model_dim]`` and padding mask."""

        safe_phase, key_padding_mask = self._validate_phase(
            phase_x, phase_observed_mask
        )
        phase_reconstruction_mask = self._validate_phase_reconstruction_mask(
            phase_x, phase_observed_mask, phase_reconstruction_mask
        )
        batch_size = safe_phase.shape[0]
        feature_ids = self.phase_feature_ids.unsqueeze(0).expand(batch_size, -1)
        tokens = self.feature_id_embedding(feature_ids)
        value_tokens = self.feature_value_embedding(safe_phase.unsqueeze(-1))
        if phase_reconstruction_mask is not None:
            assert self.phase_mask_token is not None
            masked_value_tokens = self.phase_mask_token.view(1, 1, -1).expand_as(
                value_tokens
            )
            value_tokens = torch.where(
                phase_reconstruction_mask.unsqueeze(-1),
                masked_value_tokens,
                value_tokens,
            )
        tokens = tokens + value_tokens
        tokens = self.phase_token_dropout(self.phase_token_norm(tokens))
        for layer in self.phase_encoder:
            tokens = layer(tokens, key_padding_mask)
        return self.phase_context_norm(tokens), key_padding_mask

    def _normalise_endpoint_ids(
        self,
        endpoint_ids: Sequence[int] | torch.Tensor | None,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if endpoint_ids is None:
            return self.all_endpoint_ids
        if isinstance(endpoint_ids, torch.Tensor):
            if endpoint_ids.ndim != 1:
                raise ValueError("endpoint_ids must have shape [queries]")
            integer_dtypes = {
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            }
            if endpoint_ids.dtype not in integer_dtypes:
                raise TypeError("endpoint_ids must contain integers")
            ids = endpoint_ids.to(device=device, dtype=torch.long)
        else:
            if isinstance(endpoint_ids, (str, bytes)):
                raise TypeError("endpoint_ids must be a numerical sequence")
            values = list(endpoint_ids)
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise TypeError("endpoint_ids must contain integers")
            ids = torch.tensor(values, dtype=torch.long, device=device)
        if ids.numel() == 0:
            raise ValueError("endpoint_ids must contain at least one query")
        if bool((ids < 0).any()) or bool((ids >= self.total_endpoints).any()):
            raise IndexError(
                f"endpoint_ids must be in [0, {self.total_endpoints})"
            )
        return ids

    def decode_endpoints(
        self,
        phase_context: torch.Tensor,
        phase_key_padding_mask: torch.Tensor | None,
        endpoint_ids: Sequence[int] | torch.Tensor | None = None,
    ) -> CaptainPrediction:
        """Decode dense or sparse global endpoint queries from encoded phase."""

        if phase_context.ndim != 3 or phase_context.shape[1:] != (
            self.input_dim,
            self.model_dim,
        ):
            raise ValueError(
                "phase_context must have shape "
                f"[batch, {self.input_dim}, {self.model_dim}], found "
                f"{tuple(phase_context.shape)}"
            )
        if phase_key_padding_mask is not None:
            expected = phase_context.shape[:2]
            if (
                phase_key_padding_mask.dtype is not torch.bool
                or tuple(phase_key_padding_mask.shape) != expected
                or phase_key_padding_mask.device != phase_context.device
            ):
                raise ValueError(
                    "phase_key_padding_mask must be bool [batch, input_dim] "
                    "on the phase_context device"
                )
        ids = self._normalise_endpoint_ids(
            endpoint_ids, device=phase_context.device
        )
        batch_size = phase_context.shape[0]
        queries = self.endpoint_token_embedding(ids)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.endpoint_token_dropout(queries)
        for layer in self.cross_attention_layers:
            queries = layer(queries, phase_context, phase_key_padding_mask)
        queries = self.endpoint_norm(queries)
        mean = self.mean_head(queries).squeeze(-1)
        quantiles = None
        if self.quantile_head is not None:
            quantiles = self.quantile_head(queries)
        return CaptainPrediction(mean=mean, quantiles=quantiles, endpoint_ids=ids)

    def forward(
        self,
        phase_x: torch.Tensor,
        endpoint_ids: Sequence[int] | torch.Tensor | None = None,
        *,
        phase_observed_mask: torch.Tensor | None = None,
        phase_reconstruction_mask: torch.Tensor | None = None,
    ) -> CaptainPrediction:
        """Predict all endpoints or a shared sparse endpoint subset.

        Passing ``endpoint_ids=None`` evaluates the full registered vocabulary.
        A one-dimensional ID vector evaluates only those concrete global
        reporter-endpoint queries and does not materialise inactive queries.
        """

        context, key_padding_mask = self.encode_phase(
            phase_x,
            phase_observed_mask,
            phase_reconstruction_mask,
        )
        endpoint_prediction = self.decode_endpoints(
            context, key_padding_mask, endpoint_ids=endpoint_ids
        )
        phase_reconstruction = None
        if self.phase_reconstruction_head is not None:
            phase_reconstruction = self.phase_reconstruction_head(context).squeeze(-1)
        return CaptainPrediction(
            mean=endpoint_prediction.mean,
            quantiles=endpoint_prediction.quantiles,
            endpoint_ids=endpoint_prediction.endpoint_ids,
            phase_reconstruction=phase_reconstruction,
            phase_reconstruction_mask=phase_reconstruction_mask,
        )

    def forward_reporter(
        self,
        phase_x: torch.Tensor,
        reporter: str | int | torch.Tensor,
        *,
        phase_observed_mask: torch.Tensor | None = None,
        phase_reconstruction_mask: torch.Tensor | None = None,
    ) -> CaptainPrediction:
        ids = self.endpoint_ids_for_reporter(reporter, device=phase_x.device)
        return self(
            phase_x,
            endpoint_ids=ids,
            phase_observed_mask=phase_observed_mask,
            phase_reconstruction_mask=phase_reconstruction_mask,
        )

    @staticmethod
    def _validate_supervision(
        prediction: CaptainPrediction,
        target: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(target, torch.Tensor) or not target.is_floating_point():
            raise TypeError("target must be a floating-point torch.Tensor")
        if not isinstance(observed_mask, torch.Tensor):
            raise TypeError("observed_mask must be an explicit torch.Tensor")
        if observed_mask.dtype is not torch.bool:
            raise TypeError("observed_mask must have dtype torch.bool")
        if target.shape != prediction.mean.shape:
            raise ValueError(
                "target must match prediction mean shape, found "
                f"{tuple(target.shape)} and {tuple(prediction.mean.shape)}"
            )
        if observed_mask.shape != target.shape:
            raise ValueError(
                "observed_mask must match target shape, found "
                f"{tuple(observed_mask.shape)} and {tuple(target.shape)}"
            )
        if (
            target.device != prediction.mean.device
            or observed_mask.device != prediction.mean.device
        ):
            raise ValueError("prediction, target, and observed_mask must share a device")
        observed_count = observed_mask.sum()
        if int(observed_count.detach().cpu()) == 0:
            raise ValueError("observed_mask contains no observed endpoint labels")
        if not bool(torch.isfinite(target[observed_mask]).all()):
            raise ValueError("Observed endpoint targets must be finite")
        safe_target = torch.where(observed_mask, target, torch.zeros_like(target))
        safe_target = safe_target.to(dtype=prediction.mean.dtype)
        return safe_target, observed_mask, observed_count

    def loss_components(
        self,
        prediction: CaptainPrediction,
        target: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        phase_target: torch.Tensor | None = None,
        phase_reconstruction_mask: torch.Tensor | None = None,
    ) -> CaptainLoss:
        """Compute CAPTAIN-style losses over explicit observed entries only.

        The quantile reduction follows the public CAPTAIN code: pinball losses
        are summed over configured quantiles for each observed scalar, then
        divided by the number of observed scalar labels (not by Q again).
        """

        safe_target, mask, observed_count = self._validate_supervision(
            prediction, target, observed_mask
        )
        safe_mean = torch.where(
            mask, prediction.mean, torch.zeros_like(prediction.mean)
        )
        squared_error = (safe_mean - safe_target).square()
        mean_mse = squared_error.sum() / observed_count.to(squared_error.dtype)

        quantile_pinball = prediction.mean.new_zeros(())
        if prediction.quantiles is not None:
            expected_shape = (*prediction.mean.shape, len(self.quantile_levels_tuple))
            if tuple(prediction.quantiles.shape) != expected_shape:
                raise ValueError(
                    "quantile prediction shape mismatch: expected "
                    f"{expected_shape}, found {tuple(prediction.quantiles.shape)}"
                )
            expanded_mask = mask.unsqueeze(-1)
            safe_quantiles = torch.where(
                expanded_mask,
                prediction.quantiles,
                torch.zeros_like(prediction.quantiles),
            )
            bias = safe_quantiles - safe_target.unsqueeze(-1)
            levels = self.quantile_levels.to(
                device=bias.device, dtype=bias.dtype
            ).view(1, 1, -1)
            weights = torch.where(bias.detach() > 0.0, 1.0 - levels, levels)
            pinball = bias.abs() * weights
            pinball = torch.where(expanded_mask, pinball, torch.zeros_like(pinball))
            quantile_pinball = pinball.sum() / observed_count.to(pinball.dtype)
        elif self.quantile_loss_weight != 0.0:
            raise RuntimeError(
                "A nonzero quantile loss weight requires quantile predictions"
            )

        phase_reconstruction_mse = prediction.mean.new_zeros(())
        reconstructed_feature_count = torch.zeros(
            (), dtype=torch.long, device=prediction.mean.device
        )
        if self.phase_reconstruction_loss_weight != 0.0:
            if prediction.phase_reconstruction is None:
                raise RuntimeError(
                    "A nonzero phase reconstruction weight requires phase "
                    "reconstruction predictions"
                )
            if phase_target is None or phase_reconstruction_mask is None:
                raise ValueError(
                    "phase_target and explicit phase_reconstruction_mask are "
                    "required when phase reconstruction loss is enabled"
                )
            if prediction.phase_reconstruction_mask is None:
                raise ValueError(
                    "The prediction was produced without a phase reconstruction "
                    "mask; reconstruction loss cannot be attached retrospectively"
                )
            if not isinstance(phase_target, torch.Tensor) or not phase_target.is_floating_point():
                raise TypeError("phase_target must be a floating-point torch.Tensor")
            if not isinstance(phase_reconstruction_mask, torch.Tensor):
                raise TypeError("phase_reconstruction_mask must be a torch.Tensor")
            if phase_reconstruction_mask.dtype is not torch.bool:
                raise TypeError(
                    "phase_reconstruction_mask must have dtype torch.bool"
                )
            expected_phase_shape = prediction.phase_reconstruction.shape
            if phase_target.shape != expected_phase_shape:
                raise ValueError(
                    "phase_target must match phase reconstruction shape, found "
                    f"{tuple(phase_target.shape)} and {tuple(expected_phase_shape)}"
                )
            if phase_reconstruction_mask.shape != phase_target.shape:
                raise ValueError(
                    "phase_reconstruction_mask must match phase_target shape"
                )
            if not torch.equal(
                phase_reconstruction_mask, prediction.phase_reconstruction_mask
            ):
                raise ValueError(
                    "The loss phase_reconstruction_mask must exactly match the "
                    "mask used to construct the prediction"
                )
            if (
                phase_target.device != prediction.mean.device
                or phase_reconstruction_mask.device != prediction.mean.device
            ):
                raise ValueError(
                    "phase prediction, phase_target, and reconstruction mask "
                    "must share a device"
                )
            reconstructed_feature_count = phase_reconstruction_mask.sum()
            if int(reconstructed_feature_count.detach().cpu()) == 0:
                raise ValueError(
                    "phase_reconstruction_mask contains no reconstruction targets"
                )
            if not bool(
                torch.isfinite(phase_target[phase_reconstruction_mask]).all()
            ):
                raise ValueError("Reconstructed phase targets must be finite")
            safe_phase_target = torch.where(
                phase_reconstruction_mask,
                phase_target,
                torch.zeros_like(phase_target),
            ).to(dtype=prediction.phase_reconstruction.dtype)
            safe_phase_prediction = torch.where(
                phase_reconstruction_mask,
                prediction.phase_reconstruction,
                torch.zeros_like(prediction.phase_reconstruction),
            )
            phase_reconstruction_mse = (
                safe_phase_prediction - safe_phase_target
            ).square().sum() / reconstructed_feature_count.to(
                safe_phase_prediction.dtype
            )
        total = (
            self.mean_loss_weight * mean_mse
            + self.quantile_loss_weight * quantile_pinball
            + self.phase_reconstruction_loss_weight
            * phase_reconstruction_mse
        )
        return CaptainLoss(
            total=total,
            mean_mse=mean_mse,
            quantile_pinball=quantile_pinball,
            phase_reconstruction_mse=phase_reconstruction_mse,
            observed_count=observed_count,
            reconstructed_feature_count=reconstructed_feature_count,
        )

    def compute_loss(
        self,
        prediction: CaptainPrediction,
        target: torch.Tensor,
        observed_mask: torch.Tensor,
        *,
        phase_target: torch.Tensor | None = None,
        phase_reconstruction_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.loss_components(
            prediction,
            target,
            observed_mask,
            phase_target=phase_target,
            phase_reconstruction_mask=phase_reconstruction_mask,
        ).total

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return parameter_count(self, trainable_only=trainable_only)

    def config_manifest(self) -> dict[str, Any]:
        """Return JSON-safe architecture/provenance metadata for later runners."""

        payload: dict[str, Any] = {
            "schema_version": "ops-biological-baseline-captain-v1",
            "family": self.family,
            "upstream": {
                "name": "CAPTAIN",
                "release": UPSTREAM_RELEASE,
                "release_commit": UPSTREAM_RELEASE_COMMIT,
                "local_root": str(FROZEN_CAPTAIN_ROOT),
                "protein_model_sha256": UPSTREAM_PROTEIN_MODEL_SHA256,
                "torchrun_sha256": UPSTREAM_TORCHRUN_SHA256,
                "behavioral_reference": "public_code_not_dimensionally_ambiguous_paper_equation",
                "checkpoint_loaded": False,
            },
            "input": {
                "phase_shape": ["batch", self.input_dim],
                "token_shape": ["batch", self.input_dim, self.model_dim],
                "token_components": [
                    "learned_phase_feature_id",
                    "continuous_feature_value_projection",
                ],
            },
            "query": {
                "unit": "concrete_global_reporter_endpoint_id",
                "factorized_reporter_plus_endpoint": False,
                "semantic_metadata_used": False,
                "total_endpoints": self.total_endpoints,
                "reporter_endpoint_offsets": list(self.endpoint_offsets),
                "direction": "endpoint_Q_to_phase_KV",
                "query_self_attention": False,
                "sparse_dense_equivalence": "exact_in_deterministic_eval",
            },
            "architecture": {
                "model_dim": self.model_dim,
                "num_heads": self.num_heads,
                "phase_encoder_depth": self.phase_encoder_depth,
                "cross_attention_depth": self.cross_attention_depth,
                "feedforward_multiplier": self.feedforward_multiplier,
                "mean_head_hidden_dims": list(self.mean_head_hidden_dims),
                "quantile_head_hidden_dims": list(
                    self.quantile_head_hidden_dims
                ),
                "quantile_levels": list(self.quantile_levels_tuple),
                "phase_reconstruction_enabled": self.phase_reconstruction_enabled,
                "phase_reconstruction_head_hidden_dims": list(
                    self.phase_reconstruction_head_hidden_dims
                ),
                "embedding_dropout": self.embedding_dropout_probability,
                "attention_dropout": self.attention_dropout_probability,
                "feedforward_dropout": self.feedforward_dropout_probability,
                "mean_head_sharing": "one_scalar_head_for_all_endpoints",
                "quantile_head_sharing": (
                    "one_head_for_all_endpoints" if self.quantiles_enabled else None
                ),
            },
            "loss": {
                "mask_source": "explicit_observed_mask",
                "zero_target_is_observed_when_mask_true": True,
                "missing_inferred_from_target_value": False,
                "observed_only": True,
                "mean_loss_weight": self.mean_loss_weight,
                "quantile_loss_weight": self.quantile_loss_weight,
                "phase_reconstruction_loss_weight": (
                    self.phase_reconstruction_loss_weight
                ),
                "phase_reconstruction_mask_source": (
                    "explicit_independent_mask"
                ),
                "endpoint_and_phase_masks_are_independent": True,
                "quantile_reduction": "sum_over_quantiles_then_mean_over_observed_scalars",
            },
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": self.parameter_count(trainable_only=True),
            "parameter_policy": {
                "explicit_limit": None,
                "matched_to_resmlp_4_31m": False,
                "admission_or_primary_optimization_target": False,
                "role": "reporting_only",
                "capacity_selection": "validation_within_model_family",
            },
            "device_policy": "caller_selected_no_cpu_or_cuda_assumption",
        }
        # Fail here if a future edit introduces non-serialisable provenance.
        json.dumps(payload, sort_keys=True, allow_nan=False)
        return payload

    @staticmethod
    def frozen_upstream_hashes_match() -> bool:
        """Read-only provenance check; never invoked implicitly by forward."""

        expected = {
            FROZEN_CAPTAIN_ROOT / "pretrain" / "protein_model.py": (
                UPSTREAM_PROTEIN_MODEL_SHA256
            ),
            FROZEN_CAPTAIN_ROOT / "pretrain" / "torchrun.py": (
                UPSTREAM_TORCHRUN_SHA256
            ),
        }
        for path, digest in expected.items():
            if not path.is_file():
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return False
        return True


__all__ = [
    "CaptainCrossAttentionBlock",
    "CaptainLoss",
    "CaptainPrediction",
    "DEFAULT_PHASE_FEATURES",
    "DEFAULT_QUANTILES",
    "FROZEN_CAPTAIN_ROOT",
    "OPSCaptainAdapter",
    "UPSTREAM_PROTEIN_MODEL_SHA256",
    "UPSTREAM_RELEASE",
    "UPSTREAM_RELEASE_COMMIT",
    "UPSTREAM_TORCHRUN_SHA256",
    "parameter_count",
]
