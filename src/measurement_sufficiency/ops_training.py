"""Frozen sparse-training primitives for OPS-native PyTorch models.

The objective is deliberately hierarchical: cells are averaged within each
observed endpoint, endpoints are averaged within each active reporter, and
active reporters are then averaged equally.  Target storage below a false mask
is never read.  This is distinct from a dense element-weighted MSE.

The implementation is provenance-bound to the source anchors exported by
``measurement_sufficiency.ops_models``.  See the accompanying provenance note
before redistributing code derived from the historical study files.
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "OPS neural training requires PyTorch; install measurement-sufficiency[torch]."
    ) from exc

from .ops_models import IndependentMLPSpecialists, OPSReporterModel


SAMPLER_SCHEMA_VERSION = "ops-balanced-cyclic-sampler-v1"
CHECKPOINT_SCHEMA_VERSION = "measurement-sufficiency-ops-training-v1"


def _positive(value: int, label: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive, found {result}")
    return result


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


@dataclass(frozen=True)
class ReporterBatch:
    """One reporter's phase rows, ragged endpoint values, and availability."""

    features: np.ndarray
    targets: np.ndarray
    observed: np.ndarray | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.features, dtype=np.float32)
        y = np.asarray(self.targets, dtype=np.float32)
        mask = np.isfinite(y) if self.observed is None else np.asarray(self.observed, dtype=bool)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
            raise ValueError("features and targets must be two-dimensional with equal rows")
        if mask.shape != y.shape:
            raise ValueError("observed must have the same shape as targets")
        if not len(x) or not mask.any():
            raise ValueError("a reporter batch requires observed target elements")
        if not mask.any(axis=1).all():
            raise ValueError("every reporter row must contain an observed endpoint")
        if not np.isfinite(x).all() or not np.isfinite(y[mask]).all():
            raise ValueError("features and observed target values must be finite")
        object.__setattr__(self, "features", x)
        object.__setattr__(self, "targets", y)
        object.__setattr__(self, "observed", mask)


@dataclass(frozen=True)
class TaskBatch:
    epoch: int
    batch_index: int
    round_index: int
    task_names: tuple[str, ...]
    task_indices: tuple[np.ndarray, ...]

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(int(len(indices)) for indices in self.task_indices)

    def indices_by_task(self) -> dict[str, np.ndarray]:
        return dict(zip(self.task_names, self.task_indices))


class ReporterBalancedSampler:
    """Exact deterministic shuffled cyclic sampler with equal reporter exposure."""

    def __init__(
        self,
        task_sizes: Mapping[str, int],
        tasks_per_batch: int = 4,
        samples_per_task: int = 1024,
        epoch_samples_per_task: int | None = None,
        seed: int = 0,
    ) -> None:
        if not isinstance(task_sizes, Mapping) or not task_sizes:
            raise ValueError("task_sizes must be a non-empty ordered mapping")
        self.task_names = tuple(map(str, task_sizes))
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task names must be unique")
        self.task_sizes = {
            str(name): _positive(size, f"task size for {name}")
            for name, size in task_sizes.items()
        }
        self.tasks_per_batch = _positive(tasks_per_batch, "tasks_per_batch")
        self.samples_per_task = _positive(samples_per_task, "samples_per_task")
        self.seed = int(seed)
        quota = (
            int(math.ceil(float(np.median(tuple(self.task_sizes.values())))))
            if epoch_samples_per_task is None
            else _positive(epoch_samples_per_task, "epoch_samples_per_task")
        )
        self.rounds_per_epoch = int(math.ceil(quota / self.samples_per_task))
        self.epoch_samples_per_task = self.rounds_per_epoch * self.samples_per_task
        self.task_groups_per_round = int(
            math.ceil(len(self.task_names) / self.tasks_per_batch)
        )
        self.batches_per_epoch = self.rounds_per_epoch * self.task_groups_per_round
        self.epoch = 0
        self.next_batch_index = 0
        self._cycles = {name: 0 for name in self.task_names}
        self._cursors = {name: 0 for name in self.task_names}
        self._permutations: dict[str, tuple[int, np.ndarray]] = {}
        self._fingerprint = self._configuration_fingerprint()

    def _configuration_fingerprint(self) -> str:
        value = (
            SAMPLER_SCHEMA_VERSION,
            self.task_names,
            tuple(self.task_sizes[name] for name in self.task_names),
            self.tasks_per_batch,
            self.samples_per_task,
            self.epoch_samples_per_task,
            self.seed,
        )
        return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _permutation(self, name: str) -> np.ndarray:
        cycle = self._cycles[name]
        cached = self._permutations.get(name)
        if cached is not None and cached[0] == cycle:
            return cached[1]
        generator = np.random.default_rng(
            _stable_seed(self.seed, "task", name, "cycle", cycle)
        )
        permutation = generator.permutation(self.task_sizes[name]).astype(
            np.int64, copy=False
        )
        self._permutations[name] = (cycle, permutation)
        return permutation

    def _draw(self, name: str) -> np.ndarray:
        pieces: list[np.ndarray] = []
        remaining = self.samples_per_task
        while remaining:
            permutation = self._permutation(name)
            cursor = self._cursors[name]
            take = min(remaining, len(permutation) - cursor)
            pieces.append(permutation[cursor : cursor + take])
            cursor += take
            remaining -= take
            if cursor == len(permutation):
                self._cycles[name] += 1
                self._cursors[name] = 0
                self._permutations.pop(name, None)
            else:
                self._cursors[name] = cursor
        return pieces[0].copy() if len(pieces) == 1 else np.concatenate(pieces)

    def _task_order(self, epoch: int, round_index: int) -> tuple[str, ...]:
        generator = np.random.default_rng(
            _stable_seed(self.seed, "task-order", epoch, round_index)
        )
        order = generator.permutation(len(self.task_names))
        return tuple(self.task_names[int(index)] for index in order)

    def _advance_epoch(self) -> None:
        if self.next_batch_index >= self.batches_per_epoch:
            self.epoch += 1
            self.next_batch_index = 0

    def next_batch(self) -> TaskBatch:
        self._advance_epoch()
        batch_index = self.next_batch_index
        round_index = batch_index // self.task_groups_per_round
        group_index = batch_index % self.task_groups_per_round
        order = self._task_order(self.epoch, round_index)
        start = group_index * self.tasks_per_batch
        names = order[start : start + self.tasks_per_batch]
        indices = tuple(self._draw(name) for name in names)
        result = TaskBatch(
            self.epoch, batch_index, round_index, names, indices
        )
        self.next_batch_index += 1
        return result

    def __iter__(self) -> Iterator[TaskBatch]:
        starting_epoch = self.epoch
        while self.epoch == starting_epoch and self.next_batch_index < len(self):
            yield self.next_batch()
        self._advance_epoch()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SAMPLER_SCHEMA_VERSION,
            "configuration_fingerprint": self._fingerprint,
            "task_names": list(self.task_names),
            "task_sizes": dict(self.task_sizes),
            "tasks_per_batch": self.tasks_per_batch,
            "samples_per_task": self.samples_per_task,
            "epoch_samples_per_task": self.epoch_samples_per_task,
            "rounds_per_epoch": self.rounds_per_epoch,
            "batches_per_epoch": self.batches_per_epoch,
            "seed": self.seed,
            "epoch": self.epoch,
            "next_batch_index": self.next_batch_index,
            "cycles": dict(self._cycles),
            "cursors": dict(self._cursors),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema_version") != SAMPLER_SCHEMA_VERSION:
            raise RuntimeError(f"unsupported sampler state: {state.get('schema_version')}")
        if state.get("configuration_fingerprint") != self._fingerprint:
            raise RuntimeError("sampler checkpoint is incompatible with its configuration")
        epoch = int(state["epoch"])
        next_batch = int(state["next_batch_index"])
        cycles = {str(name): int(value) for name, value in state["cycles"].items()}
        cursors = {str(name): int(value) for name, value in state["cursors"].items()}
        if epoch < 0 or not 0 <= next_batch <= self.batches_per_epoch:
            raise RuntimeError("invalid sampler epoch or batch cursor")
        if set(cycles) != set(self.task_names) or set(cursors) != set(self.task_names):
            raise RuntimeError("sampler task cursor names differ")
        for name in self.task_names:
            if cycles[name] < 0 or not 0 <= cursors[name] < self.task_sizes[name]:
                raise RuntimeError(f"invalid sampler cursor for {name}")
        self.epoch = epoch
        self.next_batch_index = next_batch
        self._cycles = cycles
        self._cursors = cursors
        self._permutations.clear()


def endpoint_balanced_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average cells within endpoint, then observed endpoints equally."""

    if prediction.ndim != 2 or prediction.shape != target.shape:
        raise ValueError("prediction and target must be equal-shape rank-two tensors")
    if observed is None:
        valid = torch.isfinite(target)
    else:
        if observed.shape != target.shape:
            raise ValueError("observed mask shape differs from target")
        valid = observed.to(device=target.device, dtype=torch.bool)
        if not bool(torch.isfinite(target[valid]).all().item()):
            raise ValueError("observed target values must be finite")
    prediction32 = prediction.float()
    clean_target = torch.where(valid, target, torch.zeros_like(target)).float()
    squared = torch.where(
        valid, (prediction32 - clean_target).square(), torch.zeros_like(prediction32)
    )
    counts = valid.sum(dim=0)
    available = counts > 0
    if not bool(available.any().item()):
        raise ValueError("reporter has no observed target elements")
    endpoint_mse = squared.sum(dim=0) / counts.clamp_min(1).float()
    return endpoint_mse[available].mean()


def reporter_balanced_mse(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    observed: Mapping[str, torch.Tensor] | None = None,
    *,
    return_per_reporter: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Macro-average endpoint-balanced observed-only MSE across reporters."""

    if not predictions or set(predictions) != set(targets):
        raise ValueError("prediction and target reporter keys must match and be nonempty")
    if observed is not None and not set(predictions).issubset(observed):
        raise ValueError("observed masks are missing active reporters")
    losses = {
        name: endpoint_balanced_mse(
            prediction,
            targets[name],
            None if observed is None else observed[name],
        )
        for name, prediction in predictions.items()
    }
    loss = torch.stack(tuple(losses.values())).mean()
    return (loss, losses) if return_per_reporter else loss


def configure_deterministic_torch(*, cuda: bool = True, warn_only: bool = False) -> None:
    """Put PyTorch into deterministic mode without touching the generators.

    Seeding alone does not make a CUDA run reproducible: cuDNN autotuning and
    non-deterministic kernels still vary between runs. The frozen study script
    this module migrates configured all of this, and dropping it made the
    reproducibility claim weaker than the one the study actually ran under.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=warn_only)
    if cuda and torch.cuda.is_available():
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace not in {":4096:8", ":16:8"}:
            # Setting it here would be a silent no-op: cuBLAS reads the variable
            # when the CUDA context is created, which has already happened.
            raise RuntimeError(
                "deterministic CUDA training requires CUBLAS_WORKSPACE_CONFIG to be "
                f"':4096:8' or ':16:8' in the environment before the process starts; "
                f"found {workspace!r}"
            )


def seed_all(seed: int, *, cuda: bool = True, deterministic: bool = True, warn_only: bool = False) -> None:
    """Initialize Python, NumPy, CPU torch, and available CUDA generators."""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        configure_deterministic_torch(cuda=cuda, warn_only=warn_only)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def atomic_training_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    sampler: ReporterBalancedSampler | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save model, optimizer, sampler, and complete RNG state."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "sampler_state": None if sampler is None else sampler.state_dict(),
        "rng_state": capture_rng_state(),
        "training_state": dict(training_state or {}),
    }
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_training_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    sampler: ReporterBalancedSampler | None = None,
    map_location: Any = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint, failing closed when requested state is absent."""

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch before weights_only
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported OPS training checkpoint")
    model.load_state_dict(payload["model_state"], strict=True)
    for target, key in ((optimizer, "optimizer_state"), (sampler, "sampler_state")):
        if target is not None:
            if payload.get(key) is None:
                raise RuntimeError(f"checkpoint has no {key}")
            target.load_state_dict(payload[key])
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


@dataclass(frozen=True)
class IndependentMLPFit:
    model: nn.Module
    best_validation_mse: float
    best_epoch: int
    epochs_completed: int
    history: tuple[dict[str, float | int], ...]


def fit_independent_mlp(
    train: ReporterBatch,
    validation: ReporterBatch,
    *,
    hidden_dims: Sequence[int] = (256, 128),
    dropout: float = 0.0,
    batch_size: int = 1024,
    epochs: int = 100,
    patience: int = 10,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    min_delta: float = 1e-6,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> IndependentMLPFit:
    """Fit one true independent OPS MLP and restore its best validation state."""

    if train.features.shape[1] != validation.features.shape[1]:
        raise ValueError("train and validation feature dimensions differ")
    if train.targets.shape[1] != validation.targets.shape[1]:
        raise ValueError("train and validation endpoint dimensions differ")
    batch_size = _positive(batch_size, "batch_size")
    epochs = _positive(epochs, "epochs")
    patience = _positive(patience, "patience")
    seed_all(seed)
    resolved = torch.device(device)
    collection = IndependentMLPSpecialists(
        {"reporter": train.targets.shape[1]},
        input_dim=train.features.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    model = collection.specialist("reporter").to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    generator = torch.Generator().manual_seed(int(seed))
    dataset = TensorDataset(
        torch.from_numpy(train.features),
        torch.from_numpy(train.targets),
        torch.from_numpy(np.asarray(train.observed, dtype=bool)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=resolved.type == "cuda",
        generator=generator,
    )
    validation_x = torch.from_numpy(validation.features).to(resolved)
    validation_y = torch.from_numpy(validation.targets).to(resolved)
    validation_mask = torch.from_numpy(np.asarray(validation.observed, dtype=bool)).to(resolved)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for batch_x, batch_y, batch_mask in loader:
            batch_x = batch_x.to(resolved, non_blocking=True)
            batch_y = batch_y.to(resolved, non_blocking=True)
            batch_mask = batch_mask.to(resolved, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = endpoint_balanced_mse(model(batch_x), batch_y, batch_mask)
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            batches += 1
        model.eval()
        with torch.no_grad():
            validation_loss = float(
                endpoint_balanced_mse(model(validation_x), validation_y, validation_mask)
                .detach()
                .cpu()
            )
        if not math.isfinite(validation_loss):
            raise RuntimeError("MLP training produced non-finite validation loss")
        history.append(
            {
                "epoch": epoch,
                "train_mse": running / max(batches, 1),
                "validation_mse": validation_loss,
            }
        )
        if validation_loss < best_loss - float(min_delta):
            best_loss, best_epoch, best_state, stale = (
                validation_loss,
                epoch,
                _cpu_state(model),
                0,
            )
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return IndependentMLPFit(
        model, best_loss, best_epoch, len(history), tuple(history)
    )


@dataclass(frozen=True)
class BalancedFit:
    model: OPSReporterModel
    sampler: ReporterBalancedSampler
    best_validation_mse: float
    best_epoch: int
    history: tuple[dict[str, Any], ...]


def _validation_loss(
    model: OPSReporterModel,
    validation: Mapping[str, ReporterBatch],
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    per_reporter: dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for name, batch in validation.items():
            prediction = model(torch.from_numpy(batch.features).to(device), name)
            value = endpoint_balanced_mse(
                prediction,
                torch.from_numpy(batch.targets).to(device),
                torch.from_numpy(np.asarray(batch.observed, dtype=bool)).to(device),
            )
            per_reporter[name] = float(value.detach().cpu())
    return float(np.mean(tuple(per_reporter.values()))), per_reporter


def fit_reporter_balanced(
    model: OPSReporterModel,
    train: Mapping[str, ReporterBatch],
    validation: Mapping[str, ReporterBatch],
    *,
    epochs: int = 100,
    tasks_per_batch: int = 4,
    samples_per_task: int = 1024,
    epoch_samples_per_task: int | None = None,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    gradient_clip_norm: float = 0.0,
    min_delta: float = 0.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> BalancedFit:
    """AdamW fit with equal reporter exposure and validation-best restoration."""

    names = tuple(model.reporter_names)
    if tuple(map(str, train)) != names or tuple(map(str, validation)) != names:
        raise ValueError("train/validation reporter order must match the model exactly")
    for name, width in zip(names, model.output_dimensions):
        if train[name].features.shape[1] != model.input_dim:
            raise ValueError(f"feature width differs for {name}")
        if train[name].targets.shape[1] != width or validation[name].targets.shape[1] != width:
            raise ValueError(f"endpoint width differs for {name}")
    epochs = _positive(epochs, "epochs")
    seed_all(seed)
    resolved = torch.device(device)
    model.to(resolved)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    sampler = ReporterBalancedSampler(
        {name: len(train[name].features) for name in names},
        tasks_per_batch=tasks_per_batch,
        samples_per_task=samples_per_task,
        epoch_samples_per_task=epoch_samples_per_task,
        seed=seed,
    )
    best_loss, best_epoch, best_state = math.inf, 0, None
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        sampled: dict[str, list[float]] = {name: [] for name in names}
        for task_batch in sampler:
            active = task_batch.task_names
            x_parts: list[torch.Tensor] = []
            targets: dict[str, torch.Tensor] = {}
            masks: dict[str, torch.Tensor] = {}
            for name, indices in zip(active, task_batch.task_indices):
                batch = train[name]
                x_parts.append(torch.from_numpy(batch.features[indices]).to(resolved))
                targets[name] = torch.from_numpy(batch.targets[indices]).to(resolved)
                masks[name] = torch.from_numpy(
                    np.asarray(batch.observed, dtype=bool)[indices]
                ).to(resolved)
            optimizer.zero_grad(set_to_none=True)
            predictions = model.forward_grouped(
                torch.cat(x_parts), active, task_batch.group_sizes
            )
            loss, losses = reporter_balanced_mse(
                predictions, targets, masks, return_per_reporter=True
            )
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("training produced a non-finite loss")
            loss.backward()
            if float(gradient_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(gradient_clip_norm))
            optimizer.step()
            for name, value in losses.items():
                sampled[name].append(float(value.detach().cpu()))
        validation_loss, validation_by_reporter = _validation_loss(
            model, validation, resolved
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse_by_reporter": {
                    name: float(np.mean(values)) for name, values in sampled.items()
                },
                "validation_mse_by_reporter": validation_by_reporter,
                "validation_macro_mse": validation_loss,
            }
        )
        if validation_loss < best_loss - float(min_delta):
            best_loss, best_epoch, best_state = validation_loss, epoch, _cpu_state(model)
    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return BalancedFit(model, sampler, best_loss, best_epoch, tuple(history))


__all__ = [
    "BalancedFit",
    "CHECKPOINT_SCHEMA_VERSION",
    "IndependentMLPFit",
    "ReporterBalancedSampler",
    "ReporterBatch",
    "SAMPLER_SCHEMA_VERSION",
    "TaskBatch",
    "atomic_training_checkpoint",
    "capture_rng_state",
    "endpoint_balanced_mse",
    "fit_independent_mlp",
    "fit_reporter_balanced",
    "load_training_checkpoint",
    "reporter_balanced_mse",
    "restore_rng_state",
    "seed_all",
]
