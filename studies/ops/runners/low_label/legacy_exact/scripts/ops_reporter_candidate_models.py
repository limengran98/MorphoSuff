#!/usr/bin/env python3
"""Architecture-only candidate models for OPS phase-to-reporter regression.

This module deliberately contains no data loading, split construction, training
loop, filesystem output, or experiment launch code.  Every candidate has the
same narrow input contract:

``forward(phase_x, active_reporter)``

``phase_x`` is a two-dimensional numerical phase-feature tensor and
``active_reporter`` is one reporter name/index shared by the batch.  The API is
intentionally sparse: heterogeneous reporter endpoints are never materialised
as an 8.4M x 1,604 dense target, and KO/gene identity is never accepted as a
model input.

TabM is loaded from the unmodified official wheel or an explicitly checksummed
source file.  The MMoE, column-only MultiTab pilot, Q-ID, and Q-Semantic models
are OPS-native adaptations rather than copies of upstream runners.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import nn

import ops_reporter_masked_multitask_v2_lib as resmlp_library


MODEL_MANIFEST_SCHEMA = "ops-reporter-candidate-model-v1"
FACTORY_CONFIG_SCHEMA = "ops-reporter-candidate-factory-config-v1"
DEFAULT_PHASE_INPUT_DIM = 172
DEFAULT_FROZEN_TABM_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "external"
    / "original_methods"
    / "tabm"
    / "tabm.py"
)
DEFAULT_FROZEN_TABM_SHA256 = (
    "fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade"
)


def _positive_int(value: int, label: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{label} must be positive, found {value}")
    return value


def _dropout(value: float) -> float:
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"dropout must be in [0, 1), found {value}")
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
        result[name] = _positive_int(raw_dimension, f"output dimension for {name}")
    return result


def _normalise_widths(values: Sequence[int], label: str) -> tuple[int, ...]:
    widths = tuple(_positive_int(value, label) for value in values)
    if not widths:
        raise ValueError(f"{label} must contain at least one width")
    return widths


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Value is not JSON serialisable: {type(value).__name__}")


def parameter_count(module: nn.Module, trainable_only: bool = False) -> int:
    """Return scalar parameter count, optionally restricted to trainable values."""

    parameters = module.parameters()
    if trainable_only:
        parameters = (value for value in parameters if value.requires_grad)
    return int(sum(value.numel() for value in parameters))


def masked_endpoint_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    endpoint_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean squared error over observed endpoints only.

    Masked target storage is replaced with zero *before* subtraction, so NaNs or
    arbitrary sentinels below a false mask cannot affect either the value or the
    gradient.  An all-missing batch is rejected instead of yielding a silent
    zero-loss optimizer step.
    """

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have equal shape, found "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.ndim != 2:
        raise ValueError(
            f"Expected [batch, endpoints] tensors, found {tuple(prediction.shape)}"
        )
    if endpoint_mask is None:
        if not torch.isfinite(target).all():
            raise ValueError("Non-finite target requires an explicit endpoint_mask")
        endpoint_mask = torch.ones_like(target, dtype=torch.bool)
    elif endpoint_mask.shape != target.shape:
        raise ValueError(
            f"endpoint_mask shape {tuple(endpoint_mask.shape)} does not match target"
        )
    else:
        endpoint_mask = endpoint_mask.to(device=target.device, dtype=torch.bool)
    observed = endpoint_mask.sum()
    if int(observed.detach().cpu()) == 0:
        raise ValueError("endpoint_mask contains no observed values")
    clean_target = torch.where(endpoint_mask, target, torch.zeros_like(target))
    squared_error = (prediction - clean_target).square()
    squared_error = torch.where(
        endpoint_mask, squared_error, torch.zeros_like(squared_error)
    )
    return squared_error.sum() / observed.to(dtype=squared_error.dtype)


class CandidateModelBase(nn.Module):
    """Shared sparse reporter registry and loss API."""

    family = "candidate_base"

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
    ) -> None:
        super().__init__()
        dimensions = _normalise_head_dimensions(head_dimensions)
        self.input_dim = _positive_int(input_dim, "phase input dimension")
        self.reporter_names = tuple(dimensions)
        self.output_dimensions = tuple(dimensions.values())
        self._reporter_to_index = {
            name: index for index, name in enumerate(self.reporter_names)
        }
        # ModuleDict keys are deliberately independent of reporter slugs: PyTorch
        # forbids dots in module names, while frozen reporter identifiers should
        # remain unrestricted external data.
        self._reporter_module_keys = tuple(
            f"reporter_{index:03d}" for index in range(len(self.reporter_names))
        )

    @property
    def head_dimensions(self) -> dict[str, int]:
        return dict(zip(self.reporter_names, self.output_dimensions))

    def reporter_index(self, active_reporter: str | int | torch.Tensor) -> int:
        if isinstance(active_reporter, torch.Tensor):
            if active_reporter.ndim != 0:
                raise TypeError(
                    "active_reporter must be one scalar id for the whole sparse batch"
                )
            active_reporter = int(active_reporter.item())
        if isinstance(active_reporter, bool):
            raise TypeError("Boolean is not a valid reporter id")
        if isinstance(active_reporter, int):
            if not 0 <= active_reporter < len(self.reporter_names):
                raise IndexError(
                    f"Reporter index {active_reporter} outside "
                    f"[0, {len(self.reporter_names)})"
                )
            return active_reporter
        name = str(active_reporter)
        try:
            return self._reporter_to_index[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown reporter {name!r}; available={self.reporter_names}"
            ) from error

    def reporter_name(self, active_reporter: str | int | torch.Tensor) -> str:
        return self.reporter_names[self.reporter_index(active_reporter)]

    def output_dim(self, active_reporter: str | int | torch.Tensor) -> int:
        return self.output_dimensions[self.reporter_index(active_reporter)]

    def reporter_module_key(
        self, active_reporter: str | int | torch.Tensor
    ) -> str:
        return self._reporter_module_keys[self.reporter_index(active_reporter)]

    def _validate_phase_x(self, phase_x: torch.Tensor) -> None:
        if not isinstance(phase_x, torch.Tensor):
            raise TypeError("phase_x must be a torch.Tensor")
        if phase_x.ndim != 2 or phase_x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected phase_x [batch, {self.input_dim}], "
                f"found {tuple(phase_x.shape)}"
            )
        if not phase_x.is_floating_point():
            raise TypeError("phase_x must be floating point")

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        endpoint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return masked_endpoint_mse(prediction, target, endpoint_mask)

    def predict(
        self, phase_x: torch.Tensor, active_reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        return self(phase_x, active_reporter)

    def _normalise_grouped_request(
        self,
        phase_x: torch.Tensor,
        task_names: Sequence[str | int | torch.Tensor],
        group_sizes: Sequence[int],
    ) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
        self._validate_phase_x(phase_x)
        names = tuple(self.reporter_name(name) for name in task_names)
        sizes = tuple(_positive_int(size, "group size") for size in group_sizes)
        if len(names) != len(sizes):
            raise ValueError("task_names and group_sizes must have equal length")
        if not names:
            raise ValueError("A grouped sparse batch must contain at least one task")
        if len(set(names)) != len(names):
            raise ValueError("A grouped sparse batch may contain each reporter once")
        if sum(sizes) != len(phase_x):
            raise ValueError(
                f"group_sizes sum to {sum(sizes)}, but phase_x has {len(phase_x)} rows"
            )
        indices = tuple(self.reporter_index(name) for name in names)
        return names, sizes, indices

    def forward_grouped(
        self,
        phase_x: torch.Tensor,
        task_names: Sequence[str | int | torch.Tensor],
        group_sizes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Run several disjoint reporter groups in one physical optimizer step.

        The concatenated rows must follow ``task_names``/``group_sizes`` order.
        Outputs remain a sparse dictionary and may have different final widths.
        Subclasses can override this method to share encoder work across groups.
        """

        names, sizes, _ = self._normalise_grouped_request(
            phase_x, task_names, group_sizes
        )
        result: dict[str, torch.Tensor] = {}
        start = 0
        for name, size in zip(names, sizes):
            stop = start + size
            result[name] = self(phase_x[start:stop], name)
            start = stop
        return result

    def parameter_count(self, trainable_only: bool = False) -> int:
        return parameter_count(self, trainable_only=trainable_only)

    def _manifest(self, architecture: Mapping[str, Any]) -> dict[str, Any]:
        manifest = {
            "schema_version": MODEL_MANIFEST_SCHEMA,
            "family": self.family,
            "input_contract": {
                "tensor": "phase_features_only",
                "shape": ["batch", self.input_dim],
                "active_reporter": "one_name_or_index_per_sparse_batch",
                "ko_or_gene_identity_allowed": False,
                "dense_all_reporter_target_allowed": False,
            },
            "head_dimensions": self.head_dimensions,
            "n_reporters": len(self.reporter_names),
            "total_endpoints_schema_only": int(sum(self.output_dimensions)),
            "parameter_count": self.parameter_count(),
            "trainable_parameter_count": self.parameter_count(trainable_only=True),
            "architecture": _jsonable(architecture),
        }
        json.dumps(manifest, sort_keys=True)
        return manifest

    def config_manifest(self) -> dict[str, Any]:
        raise NotImplementedError


def _mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    dropout: float,
    *,
    layer_norm: bool = True,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current = input_dim
    for width in hidden_dims:
        layers.append(nn.Linear(current, width))
        layers.append(nn.GELU())
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        if dropout:
            layers.append(nn.Dropout(dropout))
        current = width
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


class IndependentMLPSpecialists(CandidateModelBase):
    """Independent MLP per reporter behind the common active-head API."""

    family = "independent_mlp_specialists"

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.0,
    ) -> None:
        super().__init__(head_dimensions, input_dim)
        self.hidden_dims = _normalise_widths(hidden_dims, "MLP hidden width")
        self.dropout = _dropout(dropout)
        self.specialists = nn.ModuleDict(
            {
                key: _mlp(
                    self.input_dim,
                    self.hidden_dims,
                    output_dim,
                    self.dropout,
                )
                for key, output_dim in zip(
                    self._reporter_module_keys, self.output_dimensions
                )
            }
        )

    def specialist(
        self, active_reporter: str | int | torch.Tensor
    ) -> nn.Module:
        return self.specialists[self.reporter_module_key(active_reporter)]

    def forward(
        self, phase_x: torch.Tensor, active_reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        self._validate_phase_x(phase_x)
        return self.specialist(active_reporter)(phase_x)

    def config_manifest(self) -> dict[str, Any]:
        return self._manifest(
            {
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "sharing": "none_between_reporters",
                "sparse_active_head_only": True,
            }
        )


class SharedResMLP52Head(CandidateModelBase):
    """OPS shared phase ResMLP with one sparse vector head per reporter.

    The implementation reuses the already validated V2 ResMLP module, but the
    width/depth/head capacity are ordinary hyperparameters.  No parameter-count
    matching rule is applied here or in the factory.
    """

    family = "ops_shared_resmlp_52head"

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
        self.backbone = resmlp_library.MaskedMultiTaskResMLP(
            self.head_dimensions,
            input_dim=self.input_dim,
            shared_width=shared_width,
            expansion_width=expansion_width,
            residual_blocks=residual_blocks,
            head_width=head_width,
            dropout=dropout,
        )

    def forward(
        self, phase_x: torch.Tensor, active_reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        self._validate_phase_x(phase_x)
        return self.backbone(phase_x, self.reporter_name(active_reporter))

    def forward_grouped(
        self,
        phase_x: torch.Tensor,
        task_names: Sequence[str | int | torch.Tensor],
        group_sizes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        names, sizes, _ = self._normalise_grouped_request(
            phase_x, task_names, group_sizes
        )
        return self.backbone.forward_grouped(phase_x, names, sizes)

    def config_manifest(self) -> dict[str, Any]:
        return self._manifest(
            {
                "shared_width": self.backbone.shared_width,
                "expansion_width": self.backbone.expansion_width,
                "residual_blocks": self.backbone.residual_blocks,
                "head_width_by_reporter": self.backbone.head_width_by_task,
                "dropout": self.backbone.dropout,
                "shared": "phase_encoder",
                "reporter_specific": "sparse_vector_head",
                "parameter_matching_required": False,
                "parameter_count_is_admission_or_selection_input": False,
            }
        )


@dataclass(frozen=True)
class TabMProvenance:
    """Fail-closed description of the official TabM backend."""

    mode: Literal["source", "wheel"] = "source"
    source_path: str | Path | None = DEFAULT_FROZEN_TABM_SOURCE
    expected_sha256: str | None = DEFAULT_FROZEN_TABM_SHA256
    expected_version: str | None = "0.0.3"

    def __post_init__(self) -> None:
        if self.mode not in {"source", "wheel"}:
            raise ValueError("TabM provenance mode must be 'source' or 'wheel'")
        if self.mode == "source" and self.source_path is None:
            raise ValueError("source provenance requires source_path")
        if self.expected_sha256 is not None:
            digest = str(self.expected_sha256).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("expected_sha256 must be a 64-character hex digest")

    @classmethod
    def from_value(
        cls, value: "TabMProvenance | Mapping[str, Any] | None"
    ) -> "TabMProvenance":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("tabm_provenance must be TabMProvenance or mapping")
        return cls(**dict(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_official_tabm(
    provenance: TabMProvenance,
) -> tuple[type[nn.Module], dict[str, Any]]:
    if provenance.mode == "wheel":
        module = importlib.import_module("tabm")
        version = str(getattr(module, "__version__", "unknown"))
        if (
            provenance.expected_version is not None
            and version != provenance.expected_version
        ):
            raise RuntimeError(
                f"TabM wheel version {version!r} differs from frozen "
                f"{provenance.expected_version!r}"
            )
        module_path = getattr(module, "__file__", None)
        resolved = {
            "mode": "wheel",
            "module": "tabm",
            "version": version,
            "module_path": None if module_path is None else str(module_path),
            "expected_version": provenance.expected_version,
            "upstream_modified": False,
        }
    else:
        path = Path(provenance.source_path).expanduser().resolve()  # type: ignore[arg-type]
        if path.is_dir():
            path = path / "tabm.py"
        if not path.is_file():
            raise FileNotFoundError(f"Official TabM source not found: {path}")
        digest = _sha256_file(path)
        if (
            provenance.expected_sha256 is not None
            and digest != provenance.expected_sha256.lower()
        ):
            raise RuntimeError(
                "Official TabM source SHA256 mismatch: "
                f"observed={digest}, expected={provenance.expected_sha256}"
            )
        module_name = f"_ops_frozen_tabm_{digest[:16]}"
        module = sys.modules.get(module_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Unable to load TabM source: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
        version = str(getattr(module, "__version__", "unknown"))
        if (
            provenance.expected_version is not None
            and version != provenance.expected_version
        ):
            raise RuntimeError(
                f"TabM source version {version!r} differs from frozen "
                f"{provenance.expected_version!r}"
            )
        resolved = {
            "mode": "source",
            "source_path": str(path),
            "source_sha256": digest,
            "expected_sha256": provenance.expected_sha256,
            "version": version,
            "expected_version": provenance.expected_version,
            "upstream_modified": False,
        }
    tabm_class = getattr(module, "TabM", None)
    if not isinstance(module, ModuleType) or tabm_class is None:
        raise ImportError("Loaded official tabm module does not expose TabM")
    return tabm_class, resolved


class TabMSpecialistWrapper(CandidateModelBase):
    """One official, unmodified TabM specialist per OPS reporter.

    Forward returns member predictions ``[batch, k, endpoints]``.  Training must
    call :meth:`compute_loss`, which computes each ensemble member's masked loss
    first and only then averages over members.  :meth:`predict` averages members
    for evaluation/inference.
    """

    family = "official_tabm_specialists"

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
        provenance: TabMProvenance | Mapping[str, Any] | None = None,
        tabm_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(head_dimensions, input_dim)
        self.provenance = TabMProvenance.from_value(provenance)
        self.tabm_kwargs = dict(tabm_kwargs or {})
        forbidden = {"n_num_features", "cat_cardinalities", "d_out"}.intersection(
            self.tabm_kwargs
        )
        if forbidden:
            raise ValueError(
                "tabm_kwargs may not override OPS data/output arguments: "
                f"{sorted(forbidden)}"
            )
        tabm_class, self.resolved_provenance = _load_official_tabm(self.provenance)
        self.specialists = nn.ModuleDict(
            {
                key: tabm_class.make(
                    n_num_features=self.input_dim,
                    d_out=output_dim,
                    **self.tabm_kwargs,
                )
                for key, output_dim in zip(
                    self._reporter_module_keys, self.output_dimensions
                )
            }
        )

    def specialist(
        self, active_reporter: str | int | torch.Tensor
    ) -> nn.Module:
        return self.specialists[self.reporter_module_key(active_reporter)]

    def forward(
        self, phase_x: torch.Tensor, active_reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        self._validate_phase_x(phase_x)
        index = self.reporter_index(active_reporter)
        output = self.specialist(index)(phase_x)
        expected_dim = self.output_dimensions[index]
        if output.ndim != 3 or output.shape[0] != len(phase_x) or output.shape[2] != expected_dim:
            raise RuntimeError(
                "Official TabM violated [batch, members, endpoints] contract: "
                f"found {tuple(output.shape)}, endpoints={expected_dim}"
            )
        return output

    def member_losses(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        endpoint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prediction.ndim != 3:
            raise ValueError("TabM prediction must be [batch, members, endpoints]")
        if target.ndim != 2 or (
            prediction.shape[0] != target.shape[0]
            or prediction.shape[2] != target.shape[1]
        ):
            raise ValueError(
                f"TabM target {tuple(target.shape)} incompatible with "
                f"prediction {tuple(prediction.shape)}"
            )
        if endpoint_mask is None:
            if not torch.isfinite(target).all():
                raise ValueError("Non-finite target requires an explicit endpoint_mask")
            endpoint_mask = torch.ones_like(target, dtype=torch.bool)
        elif endpoint_mask.shape != target.shape:
            raise ValueError("endpoint_mask must match TabM target")
        else:
            endpoint_mask = endpoint_mask.to(target.device, dtype=torch.bool)
        observed = endpoint_mask.sum()
        if int(observed.detach().cpu()) == 0:
            raise ValueError("endpoint_mask contains no observed values")
        clean_target = torch.where(endpoint_mask, target, torch.zeros_like(target))
        expanded_mask = endpoint_mask[:, None, :]
        squared = (prediction - clean_target[:, None, :]).square()
        squared = torch.where(expanded_mask, squared, torch.zeros_like(squared))
        return squared.sum(dim=(0, 2)) / observed.to(squared.dtype)

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        endpoint_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.member_losses(prediction, target, endpoint_mask).mean()

    def predict(
        self, phase_x: torch.Tensor, active_reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        return self(phase_x, active_reporter).mean(dim=1)

    def config_manifest(self) -> dict[str, Any]:
        return self._manifest(
            {
                "backend": "official_unmodified_tabm",
                "provenance": self.resolved_provenance,
                "tabm_kwargs": self.tabm_kwargs,
                "training_reduction": "masked_loss_per_member_then_mean_members",
                "inference_reduction": "mean_members",
                "sharing": "within_reporter_parameter_efficient_ensemble_only",
                "sparse_active_specialist_only": True,
            }
        )


class _RepresentationMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: Sequence[int], dropout: float
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(current, width), nn.GELU(), nn.LayerNorm(width)])
            if dropout:
                layers.append(nn.Dropout(dropout))
            current = width
        self.network = nn.Sequential(*layers)
        self.output_dim = current

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class SparseMMoE(CandidateModelBase):
    """OPS-native sparse multi-gate mixture of shared phase experts."""

    family = "ops_native_sparse_mmoe"

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
        num_experts: int = 8,
        expert_hidden_dims: Sequence[int] = (256, 128),
        tower_hidden_dims: Sequence[int] = (64,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__(head_dimensions, input_dim)
        self.num_experts = _positive_int(num_experts, "num_experts")
        if self.num_experts < 2:
            raise ValueError("MMoE requires at least two experts")
        self.expert_hidden_dims = _normalise_widths(
            expert_hidden_dims, "expert hidden width"
        )
        self.tower_hidden_dims = tuple(
            _positive_int(value, "tower hidden width") for value in tower_hidden_dims
        )
        self.dropout = _dropout(dropout)
        self.experts = nn.ModuleList(
            [
                _RepresentationMLP(
                    self.input_dim, self.expert_hidden_dims, self.dropout
                )
                for _ in range(self.num_experts)
            ]
        )
        expert_dim = self.expert_hidden_dims[-1]
        self.gates = nn.ModuleList(
            [
                nn.Linear(self.input_dim, self.num_experts)
                for _ in self.reporter_names
            ]
        )
        self.towers = nn.ModuleList(
            [
                _mlp(
                    expert_dim,
                    self.tower_hidden_dims,
                    output_dim,
                    self.dropout,
                )
                for output_dim in self.output_dimensions
            ]
        )

    def _expert_values(self, phase_x: torch.Tensor) -> torch.Tensor:
        return torch.stack([expert(phase_x) for expert in self.experts], dim=1)

    def _route_active(
        self,
        phase_x: torch.Tensor,
        expert_values: torch.Tensor,
        reporter_index: int,
        *,
        detach_diagnostics: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gate_weights = torch.softmax(self.gates[reporter_index](phase_x), dim=-1)
        mixed = torch.einsum("be,beh->bh", gate_weights, expert_values)
        prediction = self.towers[reporter_index](mixed)
        entropy = -(gate_weights.clamp_min(1e-12).log() * gate_weights).sum(dim=1)
        diagnostics = {
            "gate_weights": gate_weights,
            "mean_expert_usage": gate_weights.mean(dim=0),
            "gate_entropy": entropy.mean(),
            "normalised_gate_entropy": entropy.mean() / math.log(self.num_experts),
        }
        if detach_diagnostics:
            diagnostics = {key: value.detach() for key, value in diagnostics.items()}
        return prediction, diagnostics

    def forward_with_diagnostics(
        self,
        phase_x: torch.Tensor,
        active_reporter: str | int | torch.Tensor,
        *,
        detach_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_phase_x(phase_x)
        index = self.reporter_index(active_reporter)
        return self._route_active(
            phase_x,
            self._expert_values(phase_x),
            index,
            detach_diagnostics=detach_diagnostics,
        )

    def forward(
        self, phase_x: torch.Tensor, active_reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        prediction, _ = self.forward_with_diagnostics(
            phase_x, active_reporter, detach_diagnostics=True
        )
        return prediction

    def forward_grouped(
        self,
        phase_x: torch.Tensor,
        task_names: Sequence[str | int | torch.Tensor],
        group_sizes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        names, sizes, indices = self._normalise_grouped_request(
            phase_x, task_names, group_sizes
        )
        # All shared experts run exactly once on the concatenated physical batch.
        expert_values = self._expert_values(phase_x)
        result: dict[str, torch.Tensor] = {}
        start = 0
        for name, size, index in zip(names, sizes, indices):
            stop = start + size
            result[name], _ = self._route_active(
                phase_x[start:stop],
                expert_values[start:stop],
                index,
                detach_diagnostics=True,
            )
            start = stop
        return result

    def config_manifest(self) -> dict[str, Any]:
        return self._manifest(
            {
                "num_experts": self.num_experts,
                "expert_hidden_dims": self.expert_hidden_dims,
                "tower_hidden_dims": self.tower_hidden_dims,
                "dropout": self.dropout,
                "shared": "experts",
                "reporter_specific": ["gate", "tower"],
                "sparse_active_gate_and_tower_only": True,
                "diagnostics": [
                    "gate_weights",
                    "mean_expert_usage",
                    "gate_entropy",
                    "normalised_gate_entropy",
                ],
            }
        )


class _ContinuousFeatureTokenizer(nn.Module):
    """Feature-specific affine tokenisation without categorical casts."""

    def __init__(self, n_features: int, token_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, token_dim))
        self.bias = nn.Parameter(torch.zeros(n_features, token_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class MultiTabColumnPilot(CandidateModelBase):
    """Small-panel, within-cell column-attention pilot inspired by MultiTab-Net.

    A call inserts only the active reporter token alongside the 172 feature
    tokens.  Attention is therefore over columns *within each cell*; the batch
    dimension is never reinterpreted as a token axis and inter-sample attention
    is structurally absent.
    """

    family = "multitab_column_only_pilot"
    MIN_PANEL_SIZE = 5
    MAX_PANEL_SIZE = 12

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        reporter_panel: Sequence[str],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
        token_dim: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
        feedforward_dim: int = 128,
        head_hidden_dims: Sequence[int] = (64,),
        dropout: float = 0.0,
    ) -> None:
        all_dimensions = _normalise_head_dimensions(head_dimensions)
        panel = tuple(str(name) for name in reporter_panel)
        if not self.MIN_PANEL_SIZE <= len(panel) <= self.MAX_PANEL_SIZE:
            raise ValueError(
                "MultiTab column pilot requires an explicit 5-12 reporter panel"
            )
        if len(set(panel)) != len(panel):
            raise ValueError("reporter_panel contains duplicates")
        missing = [name for name in panel if name not in all_dimensions]
        if missing:
            raise KeyError(f"reporter_panel contains unknown reporters: {missing}")
        selected = OrderedDict((name, all_dimensions[name]) for name in panel)
        super().__init__(selected, input_dim)
        self.reporter_panel = panel
        self.token_dim = _positive_int(token_dim, "token_dim")
        self.num_heads = _positive_int(num_heads, "num_heads")
        if self.token_dim % self.num_heads:
            raise ValueError("token_dim must be divisible by num_heads")
        self.num_blocks = _positive_int(num_blocks, "num_blocks")
        self.feedforward_dim = _positive_int(feedforward_dim, "feedforward_dim")
        self.head_hidden_dims = tuple(
            _positive_int(value, "head hidden width") for value in head_hidden_dims
        )
        self.dropout = _dropout(dropout)
        self.feature_tokenizer = _ContinuousFeatureTokenizer(
            self.input_dim, self.token_dim
        )
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
        # FxT_TxT sparse-safe mask.  The active task token may read all feature
        # tokens, while feature tokens cannot read the task token.  With one
        # active task token per call there is no path through which an inactive
        # reporter token can alter the feature representation.
        attention_mask = torch.zeros(
            self.input_dim + 1, self.input_dim + 1, dtype=torch.bool
        )
        attention_mask[1:, 0] = True
        self.register_buffer(
            "column_attention_mask", attention_mask, persistent=False
        )
        self.heads = nn.ModuleList(
            [
                _mlp(
                    self.token_dim,
                    self.head_hidden_dims,
                    output_dim,
                    self.dropout,
                )
                for output_dim in self.output_dimensions
            ]
        )
        self.inter_sample_attention = False

    def forward(
        self, phase_x: torch.Tensor, active_reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        self._validate_phase_x(phase_x)
        index = self.reporter_index(active_reporter)
        feature_tokens = self.feature_tokenizer(phase_x)
        reporter_ids = torch.full(
            (len(phase_x),), index, dtype=torch.long, device=phase_x.device
        )
        reporter_token = self.reporter_tokens(reporter_ids).unsqueeze(1)
        # batch_first=True: attention axis is reporter/features within each cell.
        encoded = self.column_encoder(
            torch.cat([reporter_token, feature_tokens], dim=1),
            mask=self.column_attention_mask,
        )
        return self.heads[index](encoded[:, 0, :])

    def config_manifest(self) -> dict[str, Any]:
        return self._manifest(
            {
                "upstream_inspiration": "MultiTab-Net column transformer",
                "upstream_code_copied": False,
                "reporter_panel": self.reporter_panel,
                "panel_size": len(self.reporter_panel),
                "token_dim": self.token_dim,
                "num_heads": self.num_heads,
                "num_blocks": self.num_blocks,
                "feedforward_dim": self.feedforward_dim,
                "head_hidden_dims": self.head_hidden_dims,
                "dropout": self.dropout,
                "attention_scope": "within_cell_columns_only",
                "inter_sample_attention": False,
                "attention_mask_policy": "FxT_TxT_sparse_safe",
                "feature_tokens_read_reporter_token": False,
                "tokens_per_active_call": self.input_dim + 1,
                "sparse_active_reporter_token_and_head_only": True,
            }
        )


SEMANTIC_QUERY_FIELDS = (
    "target",
    "biology_category",
    "subcellular_compartment",
    "measurement_family",
    "statistic_type",
)


class _EndpointQueryBase(CandidateModelBase):
    """Shared phase encoder and scalar decoder for concrete endpoint queries.

    The registered endpoint vocabulary is ragged by reporter.  A sparse call
    constructs exactly ``D_r`` query vectors for the active reporter and never
    materialises queries or predictions for any inactive reporter.
    """

    family = "endpoint_query_base"

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
        hidden_dim: int = 256,
        query_dim: int = 64,
        reporter_embedding_dim: int | None = None,
        endpoint_embedding_dim: int | None = None,
        bilinear_rank: int = 64,
        encoder_depth: int = 2,
        head_hidden_dims: Sequence[int] = (64,),
        dropout: float = 0.0,
        direct_readout_rank: int = 0,
        reporter_adapter_rank: int = 0,
        *,
        extra_query_input_dim: int = 0,
    ) -> None:
        super().__init__(head_dimensions, input_dim)
        self.hidden_dim = _positive_int(hidden_dim, "hidden_dim")
        self.query_dim = _positive_int(query_dim, "query_dim")
        self.reporter_embedding_dim = _positive_int(
            self.query_dim if reporter_embedding_dim is None else reporter_embedding_dim,
            "reporter_embedding_dim",
        )
        self.endpoint_embedding_dim = _positive_int(
            self.query_dim if endpoint_embedding_dim is None else endpoint_embedding_dim,
            "endpoint_embedding_dim",
        )
        self.bilinear_rank = _positive_int(bilinear_rank, "bilinear_rank")
        self.encoder_depth = _positive_int(encoder_depth, "encoder_depth")
        self.head_hidden_dims = tuple(
            _positive_int(value, "decoder hidden width")
            for value in head_hidden_dims
        )
        self.dropout = _dropout(dropout)
        self.direct_readout_rank = int(direct_readout_rank)
        self.reporter_adapter_rank = int(reporter_adapter_rank)
        if self.direct_readout_rank < 0:
            raise ValueError("direct_readout_rank cannot be negative")
        if self.reporter_adapter_rank < 0:
            raise ValueError("reporter_adapter_rank cannot be negative")
        self.extra_query_input_dim = int(extra_query_input_dim)
        if self.extra_query_input_dim < 0:
            raise ValueError("extra_query_input_dim cannot be negative")

        offsets = [0]
        for dimension in self.output_dimensions:
            offsets.append(offsets[-1] + dimension)
        self.endpoint_offsets = tuple(offsets)
        self.total_concrete_endpoints = offsets[-1]

        encoder_layers: list[nn.Module] = []
        current = self.input_dim
        for _ in range(self.encoder_depth):
            encoder_layers.extend(
                [
                    nn.Linear(current, self.hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(self.hidden_dim),
                ]
            )
            if self.dropout:
                encoder_layers.append(nn.Dropout(self.dropout))
            current = self.hidden_dim
        self.phase_encoder = nn.Sequential(*encoder_layers)

        self.reporter_id_embeddings = nn.Embedding(
            len(self.reporter_names), self.reporter_embedding_dim
        )
        self.endpoint_id_embeddings = nn.Embedding(
            self.total_concrete_endpoints, self.endpoint_embedding_dim
        )
        query_input_dim = (
            self.reporter_embedding_dim
            + self.endpoint_embedding_dim
            + self.extra_query_input_dim
        )
        self.query_projection = nn.Sequential(
            nn.Linear(query_input_dim, self.query_dim),
            nn.GELU(),
            nn.LayerNorm(self.query_dim),
        )
        self.film = nn.Linear(self.query_dim, 2 * self.hidden_dim)
        self.phase_bilinear = nn.Linear(
            self.hidden_dim, self.bilinear_rank, bias=False
        )
        self.query_bilinear = nn.Linear(
            self.query_dim, self.bilinear_rank, bias=False
        )
        self.bilinear_output = nn.Linear(
            self.bilinear_rank, self.hidden_dim, bias=False
        )
        self.condition_norm = nn.LayerNorm(self.hidden_dim)
        # One decoder is shared by all registered concrete reporter-endpoint targets.
        self.scalar_decoder = _mlp(
            self.hidden_dim,
            self.head_hidden_dims,
            1,
            self.dropout,
        )
        # Optional low-rank hypernetwork readouts address a concrete failure
        # mode of the first query model: all 1,604 concrete endpoints otherwise
        # have to pass through one shared scalar decoder.  The global branch is
        # fully query generated.  The reporter-adapter branch adds a small
        # reporter-local phase projection while retaining endpoint queries as
        # the only output weights; it is not an independent vector head.
        self.direct_phase_readout = (
            nn.Linear(self.hidden_dim, self.direct_readout_rank, bias=False)
            if self.direct_readout_rank
            else None
        )
        self.direct_query_readout = (
            nn.Linear(self.query_dim, self.direct_readout_rank, bias=False)
            if self.direct_readout_rank
            else None
        )
        self.reporter_phase_adapters = (
            nn.ModuleList(
                [
                    nn.Linear(
                        self.hidden_dim, self.reporter_adapter_rank, bias=False
                    )
                    for _ in self.reporter_names
                ]
            )
            if self.reporter_adapter_rank
            else None
        )
        self.reporter_adapter_queries = (
            nn.Linear(self.query_dim, self.reporter_adapter_rank, bias=False)
            if self.reporter_adapter_rank
            else None
        )
        self.query_output_bias = (
            nn.Linear(self.query_dim, 1, bias=False)
            if self.direct_readout_rank or self.reporter_adapter_rank
            else None
        )

    @property
    def reporter_queries(self) -> nn.Embedding:
        """Compatibility view of the former reporter-only query embedding."""

        return self.reporter_id_embeddings

    def active_endpoint_ids(
        self,
        active_reporter: str | int | torch.Tensor,
        *,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        index = self.reporter_index(active_reporter)
        return torch.arange(
            self.endpoint_offsets[index],
            self.endpoint_offsets[index + 1],
            dtype=torch.long,
            device=device,
        )

    def _extra_query_components(
        self, reporter_index: int, endpoint_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        del reporter_index, endpoint_ids
        return ()

    def _active_queries(
        self, reporter_index: int, *, device: torch.device
    ) -> torch.Tensor:
        endpoint_ids = self.active_endpoint_ids(reporter_index, device=device)
        reporter_ids = torch.full(
            (len(endpoint_ids),), reporter_index, dtype=torch.long, device=device
        )
        components = (
            self.reporter_id_embeddings(reporter_ids),
            self.endpoint_id_embeddings(endpoint_ids),
            *self._extra_query_components(reporter_index, endpoint_ids),
        )
        query_input = torch.cat(components, dim=-1)
        return self.query_projection(query_input)

    def _decode_active(
        self, encoded: torch.Tensor, reporter_index: int
    ) -> torch.Tensor:
        query = self._active_queries(reporter_index, device=encoded.device)
        gamma, beta = self.film(query).chunk(2, dim=-1)
        film_value = (
            encoded[:, None, :] * (1.0 + torch.tanh(gamma)[None, :, :])
            + beta[None, :, :]
        )
        bilinear_value = self.bilinear_output(
            self.phase_bilinear(encoded)[:, None, :]
            * self.query_bilinear(query)[None, :, :]
        )
        conditioned = self.condition_norm(film_value + bilinear_value)
        output = self.scalar_decoder(conditioned).squeeze(-1)
        if self.direct_readout_rank:
            direct = (
                self.direct_phase_readout(encoded)[:, None, :]
                * self.direct_query_readout(query)[None, :, :]
            ).sum(dim=-1) / math.sqrt(self.direct_readout_rank)
            output = output + direct
        if self.reporter_adapter_rank:
            adapted = self.reporter_phase_adapters[reporter_index](encoded)
            adapter_query = self.reporter_adapter_queries(query)
            residual = (adapted[:, None, :] * adapter_query[None, :, :]).sum(
                dim=-1
            ) / math.sqrt(self.reporter_adapter_rank)
            output = output + residual
        if self.query_output_bias is not None:
            output = output + self.query_output_bias(query).squeeze(-1)[None, :]
        return output

    def forward(
        self, phase_x: torch.Tensor, active_reporter: str | int | torch.Tensor
    ) -> torch.Tensor:
        self._validate_phase_x(phase_x)
        index = self.reporter_index(active_reporter)
        return self._decode_active(self.phase_encoder(phase_x), index)

    def forward_grouped(
        self,
        phase_x: torch.Tensor,
        task_names: Sequence[str | int | torch.Tensor],
        group_sizes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        names, sizes, indices = self._normalise_grouped_request(
            phase_x, task_names, group_sizes
        )
        encoded = self.phase_encoder(phase_x)
        result: dict[str, torch.Tensor] = {}
        start = 0
        for name, size, index in zip(names, sizes, indices):
            stop = start + size
            result[name] = self._decode_active(encoded[start:stop], index)
            start = stop
        return result

    def _query_architecture_manifest(self) -> dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "query_dim": self.query_dim,
            "reporter_embedding_dim": self.reporter_embedding_dim,
            "endpoint_embedding_dim": self.endpoint_embedding_dim,
            "bilinear_rank": self.bilinear_rank,
            "encoder_depth": self.encoder_depth,
            "decoder_hidden_dims": self.head_hidden_dims,
            # Retain the old option name in the manifest for checkpoint/config
            # provenance while making its scalar-decoder meaning explicit.
            "head_hidden_dims_compatibility_name": self.head_hidden_dims,
            "dropout": self.dropout,
            "direct_readout_rank": self.direct_readout_rank,
            "reporter_adapter_rank": self.reporter_adapter_rank,
            "query_unit": "concrete_reporter_endpoint",
            "total_registered_endpoint_queries": self.total_concrete_endpoints,
            "active_query_count": "D_r_for_current_reporter_only",
            "output_per_query": "one_scalar",
            "decoder_sharing": "shared_scalar_plus_optional_query_hyperreadouts",
            "conditioning": [
                "FiLM",
                "bilinear",
                *(["global_query_low_rank_readout"] if self.direct_readout_rank else []),
                *(
                    ["reporter_local_query_low_rank_adapter"]
                    if self.reporter_adapter_rank
                    else []
                ),
            ],
            "observed_only_loss_supported": True,
            "dense_all_reporter_target_allowed": False,
            "parameter_matching_required": False,
            "parameter_count_is_admission_gate": False,
            "parameter_count_role": "reporting_only",
            "capacity_selection_basis": "validation_within_model_family",
            "captain_code_or_weights_used": False,
        }


class EndpointQueryIDModel(_EndpointQueryBase):
    """Q-ID: reporter-ID plus concrete endpoint-ID conditioned scalar model."""

    family = "endpoint_query_id"

    def config_manifest(self) -> dict[str, Any]:
        architecture = self._query_architecture_manifest()
        architecture.update(
            {
                "query_components": [
                    "learned_reporter_id_embedding",
                    "learned_concrete_endpoint_id_embedding",
                ],
                "semantic_metadata_used": False,
            }
        )
        return self._manifest(architecture)


def _normalise_semantic_ids(
    reporter_names: Sequence[str],
    output_dimensions: Sequence[int],
    semantic_ids: (
        Mapping[str, Mapping[str, Sequence[int] | torch.Tensor]] | torch.Tensor
    ),
    semantic_vocab_sizes: Mapping[str, int],
) -> tuple[torch.Tensor, dict[str, int], dict[str, int]]:
    """Validate caller-supplied semantics without inferring any missing value."""

    if not isinstance(semantic_vocab_sizes, Mapping):
        raise TypeError("semantic_vocab_sizes must be a field -> size mapping")
    expected_fields = set(SEMANTIC_QUERY_FIELDS)
    observed_vocab_fields = {str(field) for field in semantic_vocab_sizes}
    if observed_vocab_fields != expected_fields:
        raise ValueError(
            "semantic_vocab_sizes must contain exactly "
            f"{SEMANTIC_QUERY_FIELDS}; found={sorted(observed_vocab_fields)}"
        )
    vocab_sizes = {
        field: _positive_int(semantic_vocab_sizes[field], f"{field} vocabulary size")
        for field in SEMANTIC_QUERY_FIELDS
    }
    total_endpoints = int(sum(output_dimensions))
    if isinstance(semantic_ids, torch.Tensor):
        if semantic_ids.ndim != 2 or tuple(semantic_ids.shape) != (
            total_endpoints,
            len(SEMANTIC_QUERY_FIELDS),
        ):
            raise ValueError(
                "semantic_ids tensor must have shape "
                f"[{total_endpoints}, {len(SEMANTIC_QUERY_FIELDS)}], found "
                f"{list(semantic_ids.shape)}"
            )
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }
        if semantic_ids.dtype not in integer_dtypes:
            raise TypeError("semantic_ids tensor must contain integer IDs")
        tensor = semantic_ids.detach().to(device="cpu", dtype=torch.long).clone()
        unknown_counts: dict[str, int] = {}
        for field_index, field in enumerate(SEMANTIC_QUERY_FIELDS):
            column = tensor[:, field_index]
            if bool((column < 0).any()) or bool(
                (column >= vocab_sizes[field]).any()
            ):
                raise ValueError(
                    f"semantic_ids tensor column {field!r} contains IDs outside "
                    f"[0, {vocab_sizes[field]})"
                )
            unknown_counts[field] = int((column == 0).sum())
        return tensor, vocab_sizes, unknown_counts
    if not isinstance(semantic_ids, Mapping):
        raise TypeError(
            "semantic_ids must be either a frozen integer tensor or a "
            "reporter -> field -> integer IDs mapping"
        )
    supplied_reporters = {str(name) for name in semantic_ids}
    expected_reporters = set(reporter_names)
    if supplied_reporters != expected_reporters:
        raise ValueError(
            "semantic_ids reporter keys must exactly match head_dimensions; "
            f"missing={sorted(expected_reporters - supplied_reporters)}, "
            f"extra={sorted(supplied_reporters - expected_reporters)}"
        )

    rows: list[list[int]] = []
    unknown_counts = {field: 0 for field in SEMANTIC_QUERY_FIELDS}
    for reporter, dimension in zip(reporter_names, output_dimensions):
        reporter_payload = semantic_ids[reporter]
        if not isinstance(reporter_payload, Mapping):
            raise TypeError(f"semantic_ids[{reporter!r}] must be a field mapping")
        supplied_fields = {str(field) for field in reporter_payload}
        if supplied_fields != expected_fields:
            raise ValueError(
                f"semantic_ids[{reporter!r}] must contain exactly "
                f"{SEMANTIC_QUERY_FIELDS}; missing={sorted(expected_fields - supplied_fields)}, "
                f"extra={sorted(supplied_fields - expected_fields)}"
            )
        field_values: dict[str, list[int]] = {}
        for field in SEMANTIC_QUERY_FIELDS:
            raw_values = reporter_payload[field]
            if isinstance(raw_values, torch.Tensor):
                if raw_values.ndim != 1:
                    raise ValueError(
                        f"semantic_ids[{reporter!r}][{field!r}] must be one-dimensional"
                    )
                raw_values = raw_values.detach().cpu().tolist()
            if isinstance(raw_values, (str, bytes)) or not isinstance(
                raw_values, Sequence
            ):
                raise TypeError(
                    f"semantic_ids[{reporter!r}][{field!r}] must be integer IDs"
                )
            if len(raw_values) != dimension:
                raise ValueError(
                    f"semantic_ids[{reporter!r}][{field!r}] has {len(raw_values)} "
                    f"IDs, expected {dimension}"
                )
            values: list[int] = []
            for endpoint_index, raw_value in enumerate(raw_values):
                if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                    raise TypeError(
                        f"semantic ID for {reporter}/{field}/{endpoint_index} "
                        "must be an integer; 0 is the only unknown ID"
                    )
                value = int(raw_value)
                if not 0 <= value < vocab_sizes[field]:
                    raise ValueError(
                        f"semantic ID {value} for {reporter}/{field}/{endpoint_index} "
                        f"outside [0, {vocab_sizes[field]})"
                    )
                values.append(value)
                unknown_counts[field] += int(value == 0)
            field_values[field] = values
        for endpoint_index in range(dimension):
            rows.append(
                [field_values[field][endpoint_index] for field in SEMANTIC_QUERY_FIELDS]
            )
    return torch.tensor(rows, dtype=torch.long), vocab_sizes, unknown_counts


class EndpointQuerySemanticModel(_EndpointQueryBase):
    """Q-Semantic with caller-supplied, frozen structured endpoint semantics.

    This class never parses names or guesses biology.  Every field must be
    supplied as an integer ID for every concrete endpoint; ID 0 is reserved for
    unknown/unresolved metadata and receives a fixed zero embedding.
    """

    family = "endpoint_query_semantic"

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        semantic_ids: (
            Mapping[str, Mapping[str, Sequence[int] | torch.Tensor]] | torch.Tensor
        ),
        semantic_vocab_sizes: Mapping[str, int],
        input_dim: int = DEFAULT_PHASE_INPUT_DIM,
        hidden_dim: int = 256,
        query_dim: int = 64,
        reporter_embedding_dim: int | None = None,
        endpoint_embedding_dim: int | None = None,
        semantic_embedding_dim: int = 16,
        bilinear_rank: int = 64,
        encoder_depth: int = 2,
        head_hidden_dims: Sequence[int] = (64,),
        dropout: float = 0.0,
        direct_readout_rank: int = 0,
        reporter_adapter_rank: int = 0,
    ) -> None:
        dimensions = _normalise_head_dimensions(head_dimensions)
        semantic_id_tensor, vocab_sizes, unknown_counts = _normalise_semantic_ids(
            tuple(dimensions),
            tuple(dimensions.values()),
            semantic_ids,
            semantic_vocab_sizes,
        )
        self.semantic_embedding_dim = _positive_int(
            semantic_embedding_dim, "semantic_embedding_dim"
        )
        self.semantic_vocab_sizes = vocab_sizes
        self.semantic_unknown_counts = unknown_counts
        super().__init__(
            dimensions,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            query_dim=query_dim,
            reporter_embedding_dim=reporter_embedding_dim,
            endpoint_embedding_dim=endpoint_embedding_dim,
            bilinear_rank=bilinear_rank,
            encoder_depth=encoder_depth,
            head_hidden_dims=head_hidden_dims,
            dropout=dropout,
            direct_readout_rank=direct_readout_rank,
            reporter_adapter_rank=reporter_adapter_rank,
            extra_query_input_dim=(
                len(SEMANTIC_QUERY_FIELDS) * self.semantic_embedding_dim
            ),
        )
        self.register_buffer("semantic_id_tensor", semantic_id_tensor, persistent=True)
        self.semantic_embeddings = nn.ModuleDict(
            {
                field: nn.Embedding(
                    self.semantic_vocab_sizes[field],
                    self.semantic_embedding_dim,
                    padding_idx=0,
                )
                for field in SEMANTIC_QUERY_FIELDS
            }
        )

    def _extra_query_components(
        self, reporter_index: int, endpoint_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        del reporter_index
        active_semantics = self.semantic_id_tensor[endpoint_ids]
        components: list[torch.Tensor] = []
        for field_index, field in enumerate(SEMANTIC_QUERY_FIELDS):
            semantic_ids = active_semantics[:, field_index]
            embedded = self.semantic_embeddings[field](semantic_ids)
            # ``padding_idx`` prevents ordinary training updates, while this
            # explicit mask additionally keeps unknown=0 exact after loading a
            # checkpoint whose padding row was accidentally non-zero.
            embedded = embedded.masked_fill(semantic_ids[:, None] == 0, 0.0)
            components.append(embedded)
        return tuple(components)

    def config_manifest(self) -> dict[str, Any]:
        architecture = self._query_architecture_manifest()
        semantic_payload = self.semantic_id_tensor.detach().cpu().tolist()
        semantic_sha256 = hashlib.sha256(
            json.dumps(semantic_payload, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        architecture.update(
            {
                "query_components": [
                    "learned_reporter_id_embedding",
                    "learned_concrete_endpoint_id_embedding",
                    *[f"learned_{field}_embedding" for field in SEMANTIC_QUERY_FIELDS],
                ],
                "semantic_fields": SEMANTIC_QUERY_FIELDS,
                "semantic_embedding_dim": self.semantic_embedding_dim,
                "semantic_vocab_sizes_including_unknown": self.semantic_vocab_sizes,
                "semantic_unknown_id": 0,
                "semantic_unknown_embedding": "hard_zero_in_forward",
                "semantic_unknown_counts": self.semantic_unknown_counts,
                "semantic_id_tensor_shape": list(self.semantic_id_tensor.shape),
                "semantic_id_tensor_sha256": semantic_sha256,
                "semantic_ids_source": "caller_supplied_frozen_no_model_inference",
                "name_or_metadata_guessing_allowed": False,
            }
        )
        return self._manifest(architecture)


class ReporterQueryModel(EndpointQueryIDModel):
    """Backward-compatible class name for Q-ID (not the former vector-head model)."""


_FORBIDDEN_CONTEXT_OPTION_FRAGMENTS = (
    "ko_input",
    "ko_embedding",
    "gene_id",
    "gene_embedding",
    "perturbation_input",
    "perturbation_embedding",
)


@dataclass(frozen=True)
class CandidateModelConfig:
    """JSON-compatible factory configuration for one architecture candidate."""

    model_type: str
    head_dimensions: Mapping[str, int]
    input_dim: int = DEFAULT_PHASE_INPUT_DIM
    options: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = FACTORY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FACTORY_CONFIG_SCHEMA:
            raise ValueError(f"Unsupported factory schema: {self.schema_version}")
        normalised = _normalise_head_dimensions(self.head_dimensions)
        object.__setattr__(self, "head_dimensions", dict(normalised))
        object.__setattr__(self, "input_dim", _positive_int(self.input_dim, "input_dim"))
        options = dict(self.options)
        for raw_key in options:
            key = str(raw_key).lower()
            if any(fragment in key for fragment in _FORBIDDEN_CONTEXT_OPTION_FRAGMENTS):
                raise ValueError(
                    f"Candidate models forbid KO/gene/perturbation input option {raw_key!r}"
                )
        _jsonable(options)
        object.__setattr__(self, "options", options)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CandidateModelConfig":
        payload = dict(value)
        return cls(
            model_type=payload["model_type"],
            head_dimensions=payload["head_dimensions"],
            input_dim=payload.get("input_dim", DEFAULT_PHASE_INPUT_DIM),
            options=payload.get("options", {}),
            schema_version=payload.get("schema_version", FACTORY_CONFIG_SCHEMA),
        )

    def to_manifest(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "model_type": str(self.model_type),
            "head_dimensions": dict(self.head_dimensions),
            "input_dim": self.input_dim,
            "options": _jsonable(self.options),
            "ko_or_gene_identity_allowed": False,
            "dense_all_reporter_target_allowed": False,
        }
        json.dumps(result, sort_keys=True)
        return result


_MODEL_ALIASES = {
    "independent_mlp": "independent_mlp",
    "independent_mlp_specialists": "independent_mlp",
    "resmlp": "shared_resmlp",
    "shared_resmlp": "shared_resmlp",
    "resmlp_52head": "shared_resmlp",
    "tabm": "tabm",
    "tabm_specialists": "tabm",
    "sparse_mmoe": "sparse_mmoe",
    "mmoe": "sparse_mmoe",
    "multitab_column": "multitab_column",
    "multitab_column_pilot": "multitab_column",
    "reporter_query": "endpoint_query_id",
    "lightweight_reporter_query": "endpoint_query_id",
    "q_id": "endpoint_query_id",
    "qid": "endpoint_query_id",
    "endpoint_query_id": "endpoint_query_id",
    "q_semantic": "endpoint_query_semantic",
    "qsemantic": "endpoint_query_semantic",
    "endpoint_query_semantic": "endpoint_query_semantic",
}


def create_candidate_model(
    config: CandidateModelConfig | Mapping[str, Any] | str,
    head_dimensions: Mapping[str, int] | None = None,
    *,
    input_dim: int = DEFAULT_PHASE_INPUT_DIM,
    **options: Any,
) -> CandidateModelBase:
    """Construct one candidate without importing a training runner."""

    if isinstance(config, str):
        if head_dimensions is None:
            raise ValueError("head_dimensions is required with a model type string")
        model_config = CandidateModelConfig(
            model_type=config,
            head_dimensions=head_dimensions,
            input_dim=input_dim,
            options=options,
        )
    else:
        if head_dimensions is not None or options:
            raise ValueError(
                "Do not pass head_dimensions/options alongside a config object"
            )
        model_config = (
            config
            if isinstance(config, CandidateModelConfig)
            else CandidateModelConfig.from_mapping(config)
        )
    raw_kind = str(model_config.model_type).lower()
    try:
        kind = _MODEL_ALIASES[raw_kind]
    except KeyError as error:
        raise KeyError(
            f"Unknown candidate model {model_config.model_type!r}; "
            f"available={sorted(_MODEL_ALIASES)}"
        ) from error
    kwargs = dict(model_config.options)
    common = {
        "head_dimensions": model_config.head_dimensions,
        "input_dim": model_config.input_dim,
    }
    if kind == "independent_mlp":
        return IndependentMLPSpecialists(**common, **kwargs)
    if kind == "shared_resmlp":
        return SharedResMLP52Head(**common, **kwargs)
    if kind == "tabm":
        if "provenance" in kwargs:
            kwargs["provenance"] = TabMProvenance.from_value(kwargs["provenance"])
        return TabMSpecialistWrapper(**common, **kwargs)
    if kind == "sparse_mmoe":
        return SparseMMoE(**common, **kwargs)
    if kind == "multitab_column":
        return MultiTabColumnPilot(**common, **kwargs)
    if kind == "endpoint_query_id":
        return EndpointQueryIDModel(**common, **kwargs)
    if kind == "endpoint_query_semantic":
        return EndpointQuerySemanticModel(**common, **kwargs)
    raise AssertionError(f"Unreachable candidate kind: {kind}")


build_candidate_model = create_candidate_model


__all__ = [
    "CandidateModelBase",
    "CandidateModelConfig",
    "DEFAULT_FROZEN_TABM_SHA256",
    "DEFAULT_FROZEN_TABM_SOURCE",
    "DEFAULT_PHASE_INPUT_DIM",
    "FACTORY_CONFIG_SCHEMA",
    "EndpointQueryIDModel",
    "EndpointQuerySemanticModel",
    "IndependentMLPSpecialists",
    "MODEL_MANIFEST_SCHEMA",
    "MultiTabColumnPilot",
    "ReporterQueryModel",
    "SEMANTIC_QUERY_FIELDS",
    "SharedResMLP52Head",
    "SparseMMoE",
    "TabMProvenance",
    "TabMSpecialistWrapper",
    "build_candidate_model",
    "create_candidate_model",
    "masked_endpoint_mse",
    "parameter_count",
]
