"""Frozen OPS-native PyTorch model identities.

This is a package-facing migration of the study-authored implementations in
``ops_reporter_candidate_models.py`` and
``ops_reporter_masked_multitask_v2_lib.py``.  The source anchors are recorded
below so architecture parity can be audited without importing the historical
runner.  MMoE and endpoint-query models are intentionally outside this module.

The licensing status of the historical files is not inferred from the package
license; see ``docs/ops_model_code_provenance.md`` before redistribution.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "OPS neural models require PyTorch; install measurement-sufficiency[torch]."
    ) from exc


# The source anchors live in the dependency-free provenance module so that the
# digest audit runs in environments without PyTorch.  They are re-exported here
# because every historical import site refers to them through this module.
from .ops_provenance import (  # noqa: F401
    CANDIDATE_MODEL_SOURCE_SHA256,
    DEFAULT_PHASE_INPUT_DIM,
    MASKED_MULTITASK_SOURCE_SHA256,
)


def _positive(value: int, label: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive, found {result}")
    return result


def _dropout(value: float) -> float:
    result = float(value)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"dropout must be in [0, 1), found {result}")
    return result


def _dimensions(values: Mapping[str, int]) -> OrderedDict[str, int]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("head_dimensions must be a non-empty ordered mapping")
    result: OrderedDict[str, int] = OrderedDict()
    for raw_name, raw_width in values.items():
        name = str(raw_name)
        if not name or name in result:
            raise ValueError("reporter names must be unique and nonempty")
        result[name] = _positive(raw_width, f"output dimension for {name}")
    return result


def _widths(values: Sequence[int], label: str) -> tuple[int, ...]:
    result = tuple(_positive(value, label) for value in values)
    if not result:
        raise ValueError(f"{label} must contain at least one width")
    return result


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Historical Linear -> GELU -> LayerNorm hidden-layer ordering."""

    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.extend((nn.Linear(current, width), nn.GELU(), nn.LayerNorm(width)))
        if dropout:
            layers.append(nn.Dropout(dropout))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class OPSReporterModel(nn.Module):
    """Common active-reporter API without a dense all-reporter output."""

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
    ) -> None:
        super().__init__()
        dimensions = _dimensions(head_dimensions)
        self.input_dim = _positive(input_dim, "input_dim")
        self.reporter_names = tuple(dimensions)
        self.output_dimensions = tuple(dimensions.values())
        self._reporter_to_index = {
            name: index for index, name in enumerate(self.reporter_names)
        }
        self._reporter_module_keys = tuple(
            f"reporter_{index:03d}" for index in range(len(self.reporter_names))
        )

    @property
    def head_dimensions(self) -> dict[str, int]:
        return dict(zip(self.reporter_names, self.output_dimensions))

    def reporter_index(self, reporter: str | int | torch.Tensor) -> int:
        if isinstance(reporter, torch.Tensor):
            if reporter.ndim:
                raise TypeError("reporter tensor must be scalar")
            reporter = int(reporter.item())
        if isinstance(reporter, bool):
            raise TypeError("boolean is not a reporter id")
        if isinstance(reporter, int):
            if not 0 <= reporter < len(self.reporter_names):
                raise IndexError(f"reporter index outside [0, {len(self.reporter_names)})")
            return reporter
        try:
            return self._reporter_to_index[str(reporter)]
        except KeyError as exc:
            raise KeyError(f"unknown reporter {reporter!r}") from exc

    def reporter_name(self, reporter: str | int | torch.Tensor) -> str:
        return self.reporter_names[self.reporter_index(reporter)]

    def _validate_x(self, values: torch.Tensor) -> None:
        if not isinstance(values, torch.Tensor):
            raise TypeError("phase features must be a torch.Tensor")
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"expected phase features [batch, {self.input_dim}], found {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError("phase features must be floating point")

    def forward_grouped(
        self,
        values: torch.Tensor,
        task_names: Sequence[str | int | torch.Tensor],
        group_sizes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        self._validate_x(values)
        names = tuple(self.reporter_name(name) for name in task_names)
        sizes = tuple(_positive(size, "group size") for size in group_sizes)
        if len(names) != len(sizes) or not names or len(set(names)) != len(names):
            raise ValueError("grouped requests require unique tasks and matching sizes")
        if sum(sizes) != len(values):
            raise ValueError("group sizes do not cover the concatenated feature rows")
        result: dict[str, torch.Tensor] = {}
        start = 0
        for name, size in zip(names, sizes):
            stop = start + size
            result[name] = self(values[start:stop], name)
            start = stop
        return result


class IndependentMLPSpecialists(OPSReporterModel):
    """One independent GELU/LayerNorm MLP for every reporter."""

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__(head_dimensions, input_dim)
        self.hidden_dims = _widths(hidden_dims, "MLP hidden width")
        self.dropout = _dropout(dropout)
        self.specialists = nn.ModuleDict(
            {
                key: _mlp(self.input_dim, self.hidden_dims, output_dim, self.dropout)
                for key, output_dim in zip(
                    self._reporter_module_keys, self.output_dimensions
                )
            }
        )

    def specialist(self, reporter: str | int | torch.Tensor) -> nn.Module:
        key = self._reporter_module_keys[self.reporter_index(reporter)]
        return self.specialists[key]

    def forward(
        self, values: torch.Tensor, reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        self._validate_x(values)
        return self.specialist(reporter)(values)


class PreNormResidualFFN(nn.Module):
    """LayerNorm-first residual feed-forward block used by frozen ResMLP."""

    def __init__(self, width: int, expansion_width: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear_in = nn.Linear(width, expansion_width)
        self.activation = nn.GELU()
        self.dropout_in = nn.Dropout(dropout)
        self.linear_out = nn.Linear(expansion_width, width)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.dropout_in(self.activation(self.linear_in(self.norm(values))))
        return residual + self.dropout_out(self.linear_out(values))


class ReporterHead(nn.Module):
    def __init__(self, shared_width: int, head_width: int, output_dim: int) -> None:
        super().__init__()
        self.linear_in = nn.Linear(shared_width, head_width)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(head_width)
        self.linear_out = nn.Linear(head_width, output_dim)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.norm(self.activation(self.linear_in(encoded))))


class _ResMLPBackbone(nn.Module):
    """State-layout-compatible migration of the frozen shared backbone."""

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int,
        shared_width: int,
        expansion_width: int,
        residual_blocks: int,
        head_width: int | Mapping[str, int],
        dropout: float,
    ) -> None:
        super().__init__()
        dimensions = _dimensions(head_dimensions)
        self.input_dim = _positive(input_dim, "input_dim")
        self.shared_width = _positive(shared_width, "shared_width")
        self.expansion_width = _positive(expansion_width, "expansion_width")
        self.residual_blocks = _positive(residual_blocks, "residual_blocks")
        self.dropout = _dropout(dropout)
        self.task_names = tuple(dimensions)
        self.output_dimensions = tuple(dimensions.values())
        self._task_to_index = {name: i for i, name in enumerate(self.task_names)}
        if isinstance(head_width, Mapping):
            normalized_widths = {str(name): int(value) for name, value in head_width.items()}
            if set(normalized_widths) != set(self.task_names):
                raise ValueError("head_width mapping must match reporters exactly")
            self.head_widths = tuple(
                _positive(normalized_widths[name], f"head width for {name}")
                for name in self.task_names
            )
        else:
            width = _positive(head_width, "head_width")
            self.head_widths = tuple(width for _ in self.task_names)

        self.input_projection = nn.Linear(self.input_dim, self.shared_width)
        self.blocks = nn.ModuleList(
            PreNormResidualFFN(
                self.shared_width, self.expansion_width, self.dropout
            )
            for _ in range(self.residual_blocks)
        )
        self.final_norm = nn.LayerNorm(self.shared_width)
        self.heads = nn.ModuleList(
            ReporterHead(self.shared_width, width, output_dim)
            for width, output_dim in zip(self.head_widths, self.output_dimensions)
        )

    def task_index(self, name: str) -> int:
        try:
            return self._task_to_index[str(name)]
        except KeyError as exc:
            raise KeyError(f"unknown task {name!r}") from exc

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(f"expected [batch, {self.input_dim}] input")
        encoded = self.input_projection(values)
        for block in self.blocks:
            encoded = block(encoded)
        return self.final_norm(encoded)

    def predict_encoded(self, encoded: torch.Tensor, name: str) -> torch.Tensor:
        return self.heads[self.task_index(name)](encoded)

    def forward(self, values: torch.Tensor, name: str) -> torch.Tensor:
        return self.predict_encoded(self.encode(values), name)

    def forward_grouped(
        self, values: torch.Tensor, names: Sequence[str], sizes: Sequence[int]
    ) -> dict[str, torch.Tensor]:
        encoded = self.encode(values)
        result: dict[str, torch.Tensor] = {}
        start = 0
        for name, size in zip(names, sizes):
            stop = start + size
            result[name] = self.predict_encoded(encoded[start:stop], name)
            start = stop
        return result


class SharedResMLP(OPSReporterModel):
    """Shared pre-normalized ResMLP plus sparse reporter vector heads."""

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
        shared_width: int = 512,
        expansion_width: int = 1024,
        residual_blocks: int = 4,
        head_width: int | Mapping[str, int] = 56,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(head_dimensions, input_dim)
        self.backbone = _ResMLPBackbone(
            self.head_dimensions,
            self.input_dim,
            shared_width,
            expansion_width,
            residual_blocks,
            head_width,
            dropout,
        )

    def forward(
        self, values: torch.Tensor, reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        self._validate_x(values)
        return self.backbone(values, self.reporter_name(reporter))

    def forward_grouped(
        self,
        values: torch.Tensor,
        task_names: Sequence[str | int | torch.Tensor],
        group_sizes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        self._validate_x(values)
        names = tuple(self.reporter_name(name) for name in task_names)
        sizes = tuple(_positive(size, "group size") for size in group_sizes)
        if len(names) != len(sizes) or not names or len(set(names)) != len(names):
            raise ValueError("grouped requests require unique tasks and matching sizes")
        if sum(sizes) != len(values):
            raise ValueError("group sizes do not cover the concatenated feature rows")
        return self.backbone.forward_grouped(values, names, sizes)


class _ContinuousFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, token_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, token_dim))
        self.bias = nn.Parameter(torch.zeros(n_features, token_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class MultiTab(OPSReporterModel):
    """Within-cell feature-column attention with one active reporter token.

    The frozen public method accepts either the original 5--12 reporter pilot
    or the complete 52-reporter OPS registry.  ``batch_first=True`` makes the
    attention axis columns within each cell; samples never attend to one
    another.
    """

    MIN_PANEL_SIZE = 5
    MAX_PILOT_SIZE = 12
    FULL_REGISTRY_SIZE = 52

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        reporter_panel: Sequence[str] | None = None,
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
        token_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        feedforward_dim: int = 128,
        head_hidden_dims: Sequence[int] = (64,),
        dropout: float = 0.0,
    ) -> None:
        available = _dimensions(head_dimensions)
        panel = tuple(available) if reporter_panel is None else tuple(map(str, reporter_panel))
        valid_size = self.MIN_PANEL_SIZE <= len(panel) <= self.MAX_PILOT_SIZE
        if not valid_size and len(panel) != self.FULL_REGISTRY_SIZE:
            raise ValueError("MultiTab requires a 5-12 reporter pilot or the full 52 registry")
        if len(set(panel)) != len(panel):
            raise ValueError("reporter_panel contains duplicates")
        missing = [name for name in panel if name not in available]
        if missing:
            raise KeyError(f"unknown reporters in reporter_panel: {missing}")
        super().__init__(OrderedDict((name, available[name]) for name in panel), input_dim)
        self.reporter_panel = panel
        self.token_dim = _positive(token_dim, "token_dim")
        self.num_heads = _positive(num_heads, "num_heads")
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.num_blocks = _positive(num_blocks, "num_blocks")
        self.feedforward_dim = _positive(feedforward_dim, "feedforward_dim")
        self.head_hidden_dims = _widths(head_hidden_dims, "head hidden width")
        self.dropout = _dropout(dropout)

        self.feature_tokenizer = _ContinuousFeatureTokenizer(self.input_dim, self.token_dim)
        self.reporter_tokens = nn.Embedding(len(self.reporter_names), self.token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feedforward_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.column_encoder = nn.TransformerEncoder(
            layer, num_layers=self.num_blocks, enable_nested_tensor=False
        )
        attention_mask = torch.zeros(
            self.input_dim + 1, self.input_dim + 1, dtype=torch.bool
        )
        attention_mask[1:, 0] = True
        self.register_buffer("column_attention_mask", attention_mask, persistent=False)
        self.heads = nn.ModuleList(
            _mlp(self.token_dim, self.head_hidden_dims, output_dim, self.dropout)
            for output_dim in self.output_dimensions
        )
        self.inter_sample_attention = False

    def forward(
        self, values: torch.Tensor, reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        self._validate_x(values)
        index = self.reporter_index(reporter)
        feature_tokens = self.feature_tokenizer(values)
        ids = torch.full((len(values),), index, dtype=torch.long, device=values.device)
        reporter_token = self.reporter_tokens(ids).unsqueeze(1)
        encoded = self.column_encoder(
            torch.cat((reporter_token, feature_tokens), dim=1),
            mask=self.column_attention_mask,
        )
        return self.heads[index](encoded[:, 0, :])


__all__ = [
    "CANDIDATE_MODEL_SOURCE_SHA256",
    "DEFAULT_PHASE_INPUT_DIM",
    "IndependentMLPSpecialists",
    "MASKED_MULTITASK_SOURCE_SHA256",
    "MultiTab",
    "OPSReporterModel",
    "PreNormResidualFFN",
    "ReporterHead",
    "SharedResMLP",
]
