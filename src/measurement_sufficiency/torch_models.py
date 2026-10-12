"""Generic portable PyTorch adapters and the pinned official TabM bridge.

The ResMLP and MultiTab implementations in this module satisfy the reusable
``SparsePredictor`` protocol. They share broad architecture families with the
study models, but they are not the frozen OPS benchmark identities and have no
state/forward parity claim. Those identities live in :mod:`ops_models` and are
trained through :mod:`ops_training`. TabM uses one verified official
specialist per reporter. Unobserved endpoints never enter any adapter loss.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import numpy as np

from .models import ModelDependencyError, SparseTargetError, SparseTargets


FROZEN_TABM_VERSION = "0.0.3"
FROZEN_TABM_SOURCE_SHA256 = "fc654af6a16bac53d893a8265c79d7af4ebddcb95ad0d600cc6b6bc6b7317ade"


def _torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise ModelDependencyError(
            "ResMLP, MultiTab and TabM adapters require PyTorch; install "
            "measurement-sufficiency[torch]."
        ) from exc
    return torch, nn


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TabMProvenance:
    """Pinned official TabM source or wheel declaration.

    Source mode requires an explicit file and SHA-256. Wheel mode requires an
    exact version. Both modes fail closed before model construction.
    """

    mode: str = "source"
    source_path: str | Path | None = None
    expected_sha256: str | None = FROZEN_TABM_SOURCE_SHA256
    expected_version: str | None = FROZEN_TABM_VERSION
    #: Opt-in escape hatch for smoke tests against a stand-in source. Passing the
    #: expectations as None used to disable verification silently; now it must be
    #: stated, and the resolved provenance records that it was.
    allow_unpinned: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"source", "wheel"}:
            raise ValueError("TabM provenance mode must be 'source' or 'wheel'")
        if self.mode == "source" and self.source_path is None:
            raise ValueError("TabM source mode requires source_path")
        if not self.allow_unpinned:
            if self.mode == "source" and self.expected_sha256 is None:
                raise ValueError(
                    "TabM source mode without expected_sha256 performs no verification; "
                    "pass allow_unpinned=True to state that deliberately"
                )
            if self.mode == "wheel" and self.expected_version is None:
                raise ValueError(
                    "TabM wheel mode without expected_version performs no verification; "
                    "pass allow_unpinned=True to state that deliberately"
                )
        if self.expected_sha256 is not None:
            digest = self.expected_sha256.lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")

    @classmethod
    def from_value(cls, value: "TabMProvenance | Mapping[str, Any]") -> "TabMProvenance":
        return value if isinstance(value, cls) else cls(**dict(value))


def load_official_tabm(provenance: TabMProvenance | Mapping[str, Any]) -> tuple[type, dict[str, Any]]:
    """Load official TabM only after exact provenance verification."""
    declared = TabMProvenance.from_value(provenance)
    if declared.mode == "wheel":
        module = importlib.import_module("tabm")
        version = str(getattr(module, "__version__", "unknown"))
        if declared.expected_version is not None and version != declared.expected_version:
            raise RuntimeError(f"TabM wheel version {version!r} != {declared.expected_version!r}")
        resolved = {"mode": "wheel", "version": version, "expected_version": declared.expected_version,
                    "unpinned": bool(declared.allow_unpinned)}
    else:
        path = Path(declared.source_path).expanduser().resolve()  # type: ignore[arg-type]
        if path.is_dir():
            path = path / "tabm.py"
        if not path.is_file():
            raise FileNotFoundError(f"official TabM source not found: {path}")
        digest = _file_sha256(path)
        if declared.expected_sha256 is not None and digest != declared.expected_sha256.lower():
            raise RuntimeError("official TabM source SHA-256 mismatch")
        name = f"_measurement_sufficiency_tabm_{digest[:16]}"
        module = sys.modules.get(name)
        if module is None:
            spec = importlib.util.spec_from_file_location(name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot load TabM source: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                # Without this, a first call that fails midway leaves a partly
                # initialised module cached under the source digest, and the next
                # call short-circuits on it and reports it as verified.
                sys.modules.pop(name, None)
                raise
        version = str(getattr(module, "__version__", "unknown"))
        if declared.expected_version is not None and version != declared.expected_version:
            raise RuntimeError(f"TabM source version {version!r} != {declared.expected_version!r}")
        resolved = {
            "mode": "source", "source_sha256": digest, "version": version,
            "expected_sha256": declared.expected_sha256, "expected_version": declared.expected_version,
            "unpinned": bool(declared.allow_unpinned),
        }
    if not isinstance(module, ModuleType) or not hasattr(module, "TabM"):
        raise ImportError("verified official module does not expose TabM")
    return module.TabM, resolved


def _task_layout(endpoint_ids: tuple[object, ...], groups: Mapping[str, list[object] | tuple[object, ...]] | None) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    lookup = {endpoint: index for index, endpoint in enumerate(endpoint_ids)}
    if groups is None:
        return tuple(map(str, endpoint_ids)), tuple((index,) for index in range(len(endpoint_ids)))
    names, indices, seen = [], [], set()
    for raw_name, members in groups.items():
        name = str(raw_name)
        if not name or name in names:
            raise SparseTargetError("endpoint_groups must use unique nonempty task names")
        missing = set(members).difference(lookup)
        if missing:
            raise SparseTargetError(f"endpoint_groups references unknown endpoints: {sorted(map(str, missing))}")
        task_indices = tuple(lookup[member] for member in members)
        if not task_indices or seen.intersection(task_indices):
            raise SparseTargetError("endpoint_groups must be nonempty and disjoint")
        names.append(name)
        indices.append(task_indices)
        seen.update(task_indices)
    if seen != set(range(len(endpoint_ids))):
        raise SparseTargetError("endpoint_groups must cover every endpoint exactly once")
    return tuple(names), tuple(indices)


def _build_resmlp(torch: Any, nn: Any, input_dim: int, task_names: tuple[str, ...], output_dims: tuple[int, ...], kwargs: Mapping[str, Any]) -> Any:
    width = int(kwargs.get("shared_width", 512))
    expansion = int(kwargs.get("expansion_width", 1024))
    blocks = int(kwargs.get("residual_blocks", 4))
    raw_head_width = kwargs.get("head_width", 56)
    if isinstance(raw_head_width, Mapping):
        head_widths = tuple(int(raw_head_width[name]) for name in task_names)
    else:
        head_widths = tuple(int(raw_head_width) for _ in task_names)
    dropout = float(kwargs.get("dropout", 0.0))

    class ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, expansion), nn.GELU(), nn.Dropout(dropout), nn.Linear(expansion, width), nn.Dropout(dropout))
        def forward(self, x: Any) -> Any:
            return x + self.net(x)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input_projection = nn.Linear(input_dim, width)
            self.blocks = nn.ModuleList([ResidualBlock() for _ in range(blocks)])
            self.norm = nn.LayerNorm(width)
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(width, head_width), nn.GELU(), nn.LayerNorm(head_width), nn.Linear(head_width, dim)) for head_width, dim in zip(head_widths, output_dims)])
        def forward(self, x: Any, task: int) -> Any:
            encoded = self.input_projection(x)
            for block in self.blocks:
                encoded = block(encoded)
            return self.heads[task](self.norm(encoded))
    return Model()


def _build_multitab(torch: Any, nn: Any, input_dim: int, output_dims: tuple[int, ...], kwargs: Mapping[str, Any]) -> Any:
    token_dim = int(kwargs.get("token_dim", 32))
    heads = int(kwargs.get("num_heads", 4))
    blocks = int(kwargs.get("num_blocks", 2))
    feedforward = int(kwargs.get("feedforward_dim", 128))
    head_hidden_dims = tuple(int(value) for value in kwargs.get("head_hidden_dims", (64,)))
    dropout = float(kwargs.get("dropout", 0.0))
    if token_dim % heads:
        raise ValueError("MultiTab token_dim must be divisible by num_heads")

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_weight = nn.Parameter(torch.empty(input_dim, token_dim))
            self.feature_bias = nn.Parameter(torch.zeros(input_dim, token_dim))
            nn.init.normal_(self.feature_weight, std=0.02)
            self.task_tokens = nn.Embedding(len(output_dims), token_dim)
            layer = nn.TransformerEncoderLayer(token_dim, heads, feedforward, dropout, activation="gelu", batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
            mask = torch.zeros(input_dim + 1, input_dim + 1, dtype=torch.bool)
            mask[1:, 0] = True
            self.register_buffer("attention_mask", mask, persistent=False)
            output_heads = []
            for output_dim in output_dims:
                layers, current = [], token_dim
                for width in head_hidden_dims:
                    layers.extend([nn.Linear(current, width), nn.GELU(), nn.LayerNorm(width)])
                    if dropout:
                        layers.append(nn.Dropout(dropout))
                    current = width
                layers.append(nn.Linear(current, output_dim))
                output_heads.append(nn.Sequential(*layers))
            self.output_heads = nn.ModuleList(output_heads)
        def forward(self, x: Any, task: int) -> Any:
            feature_tokens = x.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)
            ids = torch.full((len(x),), task, device=x.device, dtype=torch.long)
            task_token = self.task_tokens(ids).unsqueeze(1)
            encoded = self.encoder(torch.cat([task_token, feature_tokens], dim=1), mask=self.attention_mask)
            return self.output_heads[task](encoded[:, 0])
    return Model()


def _build_tabm(torch: Any, nn: Any, input_dim: int, output_dims: tuple[int, ...], kwargs: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    if "provenance" not in kwargs:
        raise ValueError("TabM requires an explicit provenance declaration")
    tabm_class, resolved = load_official_tabm(kwargs["provenance"])
    tabm_kwargs = dict(kwargs.get("tabm_kwargs", {}))
    if {"n_num_features", "cat_cardinalities", "d_out"}.intersection(tabm_kwargs):
        raise ValueError("tabm_kwargs may not override input or output dimensions")

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.specialists = nn.ModuleList([tabm_class.make(n_num_features=input_dim, d_out=dim, **tabm_kwargs) for dim in output_dims])
        def forward(self, x: Any, task: int) -> Any:
            return self.specialists[task](x)
    return Model(), resolved


@dataclass
class TorchGroupedSparseRegressor:
    """Observed-endpoint-only trainer for ResMLP, MultiTab or official TabM."""

    architecture: str
    seed: int = 0
    endpoint_groups: Mapping[str, list[object] | tuple[object, ...]] | None = None
    epochs: int = 100
    batch_size: int = 512
    tasks_per_batch: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    device: str = "cpu"
    architecture_kwargs: Mapping[str, Any] = field(default_factory=dict)
    endpoint_ids: tuple[object, ...] = ()
    task_names_: tuple[str, ...] = ()
    task_indices_: tuple[tuple[int, ...], ...] = ()
    model_: Any = None
    resolved_provenance_: dict[str, Any] | None = None

    def fit(self, features: np.ndarray, targets: SparseTargets) -> "TorchGroupedSparseRegressor":
        torch, nn = _torch()
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or len(x) != len(targets.observation_ids):
            raise SparseTargetError("features must have one row per target observation")
        self.endpoint_ids = targets.endpoint_ids
        self.task_names_, self.task_indices_ = _task_layout(self.endpoint_ids, self.endpoint_groups)
        output_dims = tuple(map(len, self.task_indices_))
        torch.manual_seed(int(self.seed))
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ModelDependencyError("CUDA was requested but is unavailable")
        if self.architecture == "resmlp":
            model = _build_resmlp(torch, nn, x.shape[1], self.task_names_, output_dims, self.architecture_kwargs)
        elif self.architecture == "multitab":
            model = _build_multitab(torch, nn, x.shape[1], output_dims, self.architecture_kwargs)
        elif self.architecture == "tabm":
            model, self.resolved_provenance_ = _build_tabm(torch, nn, x.shape[1], output_dims, self.architecture_kwargs)
        else:
            raise ValueError(f"unknown torch architecture: {self.architecture}")
        device = torch.device(self.device)
        self.model_ = model.to(device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))
        task_rows = [np.flatnonzero(targets.observed[:, list(indices)].any(axis=1)) for indices in self.task_indices_]
        if any(len(rows) == 0 for rows in task_rows):
            missing = [self.task_names_[i] for i, rows in enumerate(task_rows) if len(rows) == 0]
            raise SparseTargetError(f"tasks have no observed targets: {missing}")
        if not task_rows:
            raise SparseTargetError("torch model requires at least one observed target")
        samples_per_task = int(self.batch_size)
        if samples_per_task <= 0 or int(self.tasks_per_batch) <= 0:
            raise ValueError("batch_size and tasks_per_batch must be positive")
        rounds = max(1, int(np.ceil(np.median([len(rows) for rows in task_rows]) / samples_per_task)))
        cycles = [0 for _ in task_rows]
        cursors = [0 for _ in task_rows]
        permutations: list[np.ndarray | None] = [None for _ in task_rows]

        def stable_seed(*parts: object) -> int:
            digest = hashlib.blake2b("|".join(map(str, (self.seed, *parts))).encode(), digest_size=8).digest()
            return int.from_bytes(digest, "little")

        def draw(task: int) -> np.ndarray:
            pieces, remaining = [], samples_per_task
            while remaining:
                if permutations[task] is None:
                    generator = np.random.default_rng(stable_seed("task", self.task_names_[task], "cycle", cycles[task]))
                    permutations[task] = generator.permutation(task_rows[task])
                available = len(permutations[task]) - cursors[task]
                take = min(remaining, available)
                pieces.append(permutations[task][cursors[task]:cursors[task] + take])
                cursors[task] += take
                remaining -= take
                if cursors[task] == len(permutations[task]):
                    cycles[task] += 1
                    cursors[task] = 0
                    permutations[task] = None
            return np.concatenate(pieces)

        self.model_.train()
        for epoch in range(int(self.epochs)):
            for round_index in range(rounds):
                order = np.random.default_rng(stable_seed("task-order", epoch, round_index)).permutation(len(task_rows))
                for start in range(0, len(order), int(self.tasks_per_batch)):
                    active_tasks = order[start:start + int(self.tasks_per_batch)]
                    optimizer.zero_grad(set_to_none=True)
                    task_losses = []
                    for raw_task in active_tasks:
                        task = int(raw_task)
                        rows = draw(task)
                        indices = list(self.task_indices_[task])
                        prediction = self.model_(torch.as_tensor(x[rows], device=device), task)
                        truth = torch.as_tensor(targets.values[np.ix_(rows, indices)], dtype=torch.float32, device=device)
                        mask = torch.as_tensor(targets.observed[np.ix_(rows, indices)], dtype=torch.bool, device=device)
                        clean = torch.where(mask, truth, torch.zeros_like(truth))
                        counts = mask.sum(dim=0)
                        valid_features = counts > 0
                        if prediction.ndim == 3:  # official TabM members
                            squared = (prediction - clean[:, None, :]).square()
                            feature_mse = torch.where(mask[:, None, :], squared, torch.zeros_like(squared)).sum(dim=0) / counts.clamp_min(1)[None, :]
                            task_loss = feature_mse[:, valid_features].mean()
                        else:
                            squared = (prediction - clean).square()
                            feature_mse = torch.where(mask, squared, torch.zeros_like(squared)).sum(dim=0) / counts.clamp_min(1)
                            task_loss = feature_mse[valid_features].mean()
                        task_losses.append(task_loss)
                    loss = torch.stack(task_losses).mean()
                    loss.backward()
                    optimizer.step()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("model must be fit before prediction")
        torch, _ = _torch()
        x = np.asarray(features, dtype=np.float32)
        output = np.full((len(x), len(self.endpoint_ids)), np.nan, dtype=float)
        device = torch.device(self.device)
        self.model_.eval()
        with torch.no_grad():
            for start in range(0, len(x), int(self.batch_size)):
                stop = min(start + int(self.batch_size), len(x))
                tensor = torch.as_tensor(x[start:stop], device=device)
                for task, indices in enumerate(self.task_indices_):
                    prediction = self.model_(tensor, task)
                    if prediction.ndim == 3:
                        prediction = prediction.mean(dim=1)
                    output[start:stop, list(indices)] = prediction.detach().cpu().numpy()
        return output
