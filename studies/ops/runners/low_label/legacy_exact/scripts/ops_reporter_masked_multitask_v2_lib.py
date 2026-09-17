"""Reusable primitives for the OPS capacity-aware truly-masked ResMLP V2.

This module deliberately contains no OPS path assumptions and no experiment
orchestration.  A runner supplies already-frozen task schemas, split-local
arrays, and output locations.  The primitives here provide:

* the parameter-matched shared ResMLP and reporter-specific heads;
* exact parameter-budget accounting against independent MLP specialists;
* sparse, task-balanced masked regression loss;
* a deterministic cyclic task sampler with compact resumable state;
* AMP policy and atomic training-checkpoint helpers; and
* a synthetic 52-head end-to-end self-test.

The module supports both the frozen V1 uniform 56-wide heads and V2's ordered
reporter-specific head widths.  V2 assigns width 128 only to 60/72D reporters
and reduces the shared expansion from 640 to 568, yielding 4,310,636 total
parameters versus V1's 4,310,852 (-216; -0.005%).
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = "ops-masked-multitask-resmlp-checkpoint-v2"
SAMPLER_SCHEMA_VERSION = "ops-balanced-cyclic-sampler-v1"
MODEL_SCHEMA_VERSION = "ops-masked-multitask-resmlp-v2"

DEFAULT_INPUT_DIM = 172
DEFAULT_SHARED_WIDTH = 512
DEFAULT_EXPANSION_WIDTH = 640
DEFAULT_RESIDUAL_BLOCKS = 4
DEFAULT_HEAD_WIDTH = 56
DEFAULT_DROPOUT = 0.05


def _positive_integer(value: int, label: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{label} must be positive, found {value}")
    return value


def _normalise_head_dimensions(
    head_dimensions: Mapping[str, int],
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if not isinstance(head_dimensions, Mapping) or not head_dimensions:
        raise ValueError("head_dimensions must be a non-empty ordered mapping")
    names: list[str] = []
    dimensions: list[int] = []
    for raw_name, raw_dimension in head_dimensions.items():
        name = str(raw_name)
        if not name:
            raise ValueError("Task names must be non-empty strings")
        if name in names:
            raise ValueError(f"Duplicate task name: {name!r}")
        names.append(name)
        dimensions.append(_positive_integer(raw_dimension, f"output dimension for {name}"))
    return tuple(names), tuple(dimensions)


def _normalise_head_widths(
    task_names: Sequence[str], head_width: int | Mapping[str, int]
) -> tuple[int, ...]:
    if isinstance(head_width, Mapping):
        widths = {str(name): int(value) for name, value in head_width.items()}
        missing = sorted(set(task_names) - set(widths))
        extra = sorted(set(widths) - set(task_names))
        if missing or extra:
            raise ValueError(
                f"head_width mapping differs from tasks: missing={missing}, extra={extra}"
            )
        return tuple(
            _positive_integer(widths[name], f"head width for {name}")
            for name in task_names
        )
    width = _positive_integer(head_width, "head_width")
    return tuple(width for _ in task_names)


class PreNormResidualFFN(nn.Module):
    """Pre-LayerNorm residual feed-forward block for tabular phase features."""

    def __init__(
        self,
        width: int = DEFAULT_SHARED_WIDTH,
        expansion_width: int = DEFAULT_EXPANSION_WIDTH,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        width = _positive_integer(width, "width")
        expansion_width = _positive_integer(expansion_width, "expansion_width")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.norm = nn.LayerNorm(width)
        self.linear_in = nn.Linear(width, expansion_width)
        self.activation = nn.GELU()
        self.dropout_in = nn.Dropout(float(dropout))
        self.linear_out = nn.Linear(expansion_width, width)
        self.dropout_out = nn.Dropout(float(dropout))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.norm(values)
        values = self.linear_in(values)
        values = self.activation(values)
        values = self.dropout_in(values)
        values = self.linear_out(values)
        values = self.dropout_out(values)
        return residual + values


class ReporterHead(nn.Module):
    """Reporter-specific nonlinear readout from the shared phase encoding."""

    def __init__(self, shared_width: int, head_width: int, output_dim: int) -> None:
        super().__init__()
        shared_width = _positive_integer(shared_width, "shared_width")
        head_width = _positive_integer(head_width, "head_width")
        output_dim = _positive_integer(output_dim, "output_dim")
        self.linear_in = nn.Linear(shared_width, head_width)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(head_width)
        self.linear_out = nn.Linear(head_width, output_dim)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.linear_out(self.norm(self.activation(self.linear_in(encoded))))


class MaskedMultiTaskResMLP(nn.Module):
    """One shared phase encoder with a sparse reporter-specific head collection.

    ``head_dimensions`` is ordered.  The order is preserved in checkpoints and
    must remain stable across resume and evaluation.  A ``ModuleList`` is used
    instead of a ``ModuleDict`` so arbitrary reporter slugs remain valid names.
    """

    def __init__(
        self,
        head_dimensions: Mapping[str, int],
        input_dim: int = DEFAULT_INPUT_DIM,
        shared_width: int = DEFAULT_SHARED_WIDTH,
        expansion_width: int = DEFAULT_EXPANSION_WIDTH,
        residual_blocks: int = DEFAULT_RESIDUAL_BLOCKS,
        head_width: int | Mapping[str, int] = DEFAULT_HEAD_WIDTH,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        task_names, output_dimensions = _normalise_head_dimensions(head_dimensions)
        self.input_dim = _positive_integer(input_dim, "input_dim")
        self.shared_width = _positive_integer(shared_width, "shared_width")
        self.expansion_width = _positive_integer(expansion_width, "expansion_width")
        self.residual_blocks = _positive_integer(residual_blocks, "residual_blocks")
        self.head_widths = _normalise_head_widths(task_names, head_width)
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.dropout = float(dropout)
        self.task_names = task_names
        self.output_dimensions = output_dimensions
        self._task_to_index = {name: index for index, name in enumerate(task_names)}

        self.input_projection = nn.Linear(self.input_dim, self.shared_width)
        self.blocks = nn.ModuleList(
            [
                PreNormResidualFFN(
                    width=self.shared_width,
                    expansion_width=self.expansion_width,
                    dropout=self.dropout,
                )
                for _ in range(self.residual_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(self.shared_width)
        self.heads = nn.ModuleList(
            [
                ReporterHead(self.shared_width, width, output_dim)
                for width, output_dim in zip(
                    self.head_widths, self.output_dimensions
                )
            ]
        )

    @property
    def head_dimensions(self) -> dict[str, int]:
        return dict(zip(self.task_names, self.output_dimensions))

    @property
    def head_width_by_task(self) -> dict[str, int]:
        return dict(zip(self.task_names, self.head_widths))

    def task_index(self, task_name: str) -> int:
        try:
            return self._task_to_index[str(task_name)]
        except KeyError as error:
            raise KeyError(f"Unknown task {task_name!r}; available={self.task_names}") from error

    def task_output_dim(self, task_name: str) -> int:
        return self.output_dimensions[self.task_index(task_name)]

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, {self.input_dim}] input, found {tuple(values.shape)}"
            )
        encoded = self.input_projection(values)
        for block in self.blocks:
            encoded = block(encoded)
        return self.final_norm(encoded)

    def predict_encoded(self, encoded: torch.Tensor, task_name: str) -> torch.Tensor:
        return self.heads[self.task_index(task_name)](encoded)

    def forward(self, values: torch.Tensor, task_name: str) -> torch.Tensor:
        return self.predict_encoded(self.encode(values), task_name)

    def forward_all(self, values: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encode(values)
        return {
            name: self.heads[index](encoded)
            for index, name in enumerate(self.task_names)
        }

    def forward_grouped(
        self,
        values: torch.Tensor,
        task_names: Sequence[str],
        group_sizes: Sequence[int],
    ) -> dict[str, torch.Tensor]:
        """Encode a concatenated sparse batch once, then route groups to heads.

        ``values`` must be concatenated in the same order as ``task_names`` and
        ``group_sizes``.  Task names must be distinct within one optimizer step.
        """

        names = tuple(str(name) for name in task_names)
        sizes = tuple(_positive_integer(size, "group size") for size in group_sizes)
        if len(names) != len(sizes):
            raise ValueError("task_names and group_sizes must have equal length")
        if len(set(names)) != len(names):
            raise ValueError("A grouped batch may contain each task at most once")
        if sum(sizes) != len(values):
            raise ValueError(
                f"Grouped sizes sum to {sum(sizes)}, but input has {len(values)} rows"
            )
        encoded = self.encode(values)
        result: dict[str, torch.Tensor] = {}
        start = 0
        for name, size in zip(names, sizes):
            stop = start + size
            result[name] = self.predict_encoded(encoded[start:stop], name)
            start = stop
        return result

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = (
            parameter for parameter in self.parameters() if parameter.requires_grad
        ) if trainable_only else self.parameters()
        return int(sum(parameter.numel() for parameter in parameters))

    def config_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "head_dimensions": self.head_dimensions,
            "input_dim": self.input_dim,
            "shared_width": self.shared_width,
            "expansion_width": self.expansion_width,
            "residual_blocks": self.residual_blocks,
            "head_width_by_task": self.head_width_by_task,
            "dropout": self.dropout,
        }

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MaskedMultiTaskResMLP":
        if config.get("schema_version") not in {None, MODEL_SCHEMA_VERSION}:
            raise RuntimeError(f"Unsupported model schema: {config.get('schema_version')}")
        return cls(
            head_dimensions=config["head_dimensions"],
            input_dim=int(config.get("input_dim", DEFAULT_INPUT_DIM)),
            shared_width=int(config.get("shared_width", DEFAULT_SHARED_WIDTH)),
            expansion_width=int(
                config.get("expansion_width", DEFAULT_EXPANSION_WIDTH)
            ),
            residual_blocks=int(
                config.get("residual_blocks", DEFAULT_RESIDUAL_BLOCKS)
            ),
            head_width=(
                config["head_width_by_task"]
                if "head_width_by_task" in config
                else int(config.get("head_width", DEFAULT_HEAD_WIDTH))
            ),
            dropout=float(config.get("dropout", DEFAULT_DROPOUT)),
        )


def independent_mlp_parameter_count(
    input_dim: int,
    output_dimensions: Sequence[int],
    hidden: Sequence[int] = (256, 128),
    layer_norm_after_hidden: bool = True,
) -> int:
    """Count all independent specialist parameters, including biases and LN."""

    input_dim = _positive_integer(input_dim, "input_dim")
    outputs = tuple(_positive_integer(value, "output dimension") for value in output_dimensions)
    hidden_widths = tuple(_positive_integer(value, "hidden width") for value in hidden)
    if not outputs or not hidden_widths:
        raise ValueError("At least one output task and hidden layer are required")
    total = 0
    for output_dim in outputs:
        current = input_dim
        for width in hidden_widths:
            total += (current + 1) * width
            if layer_norm_after_hidden:
                total += 2 * width
            current = width
        total += (current + 1) * output_dim
    return int(total)


def expected_resmlp_parameter_count(
    input_dim: int,
    output_dimensions: Sequence[int],
    shared_width: int = DEFAULT_SHARED_WIDTH,
    expansion_width: int = DEFAULT_EXPANSION_WIDTH,
    residual_blocks: int = DEFAULT_RESIDUAL_BLOCKS,
    head_width: int = DEFAULT_HEAD_WIDTH,
    head_widths: Sequence[int] | None = None,
) -> int:
    """Closed-form count for :class:`MaskedMultiTaskResMLP`."""

    input_dim = _positive_integer(input_dim, "input_dim")
    shared_width = _positive_integer(shared_width, "shared_width")
    expansion_width = _positive_integer(expansion_width, "expansion_width")
    residual_blocks = _positive_integer(residual_blocks, "residual_blocks")
    outputs = tuple(_positive_integer(value, "output dimension") for value in output_dimensions)
    if not outputs:
        raise ValueError("At least one output task is required")
    if head_widths is None:
        width = _positive_integer(head_width, "head_width")
        widths = tuple(width for _ in outputs)
    else:
        widths = tuple(
            _positive_integer(value, "head_widths entry") for value in head_widths
        )
        if len(widths) != len(outputs):
            raise ValueError("head_widths and output_dimensions must have equal length")

    input_projection = (input_dim + 1) * shared_width
    one_block = (
        2 * shared_width
        + (shared_width + 1) * expansion_width
        + (expansion_width + 1) * shared_width
    )
    final_norm = 2 * shared_width
    heads = sum(
        (shared_width + 1) * width
        + 2 * width
        + (width + 1) * output_dim
        for width, output_dim in zip(widths, outputs)
    )
    return int(input_projection + residual_blocks * one_block + final_norm + heads)


def parameter_budget_report(
    model: MaskedMultiTaskResMLP,
    independent_hidden: Sequence[int] = (256, 128),
) -> dict[str, Any]:
    actual = model.parameter_count()
    formula = expected_resmlp_parameter_count(
        input_dim=model.input_dim,
        output_dimensions=model.output_dimensions,
        shared_width=model.shared_width,
        expansion_width=model.expansion_width,
        residual_blocks=model.residual_blocks,
        head_widths=model.head_widths,
    )
    reference = independent_mlp_parameter_count(
        model.input_dim, model.output_dimensions, hidden=independent_hidden
    )
    return {
        "actual_joint_parameters": actual,
        "closed_form_joint_parameters": formula,
        "independent_specialist_parameters": reference,
        "absolute_difference_vs_independent": actual - reference,
        "relative_difference_vs_independent": actual / reference - 1.0,
        "joint_to_independent_ratio": actual / reference,
        "n_tasks": len(model.task_names),
        "total_output_dimensions": int(sum(model.output_dimensions)),
        "head_width_by_task": model.head_width_by_task,
        "independent_hidden": [int(value) for value in independent_hidden],
    }


def assert_parameter_budget(
    model: MaskedMultiTaskResMLP,
    independent_hidden: Sequence[int] = (256, 128),
    relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Verify the implementation formula and total-parameter fairness budget."""

    if not 0.0 <= float(relative_tolerance) < 1.0:
        raise ValueError("relative_tolerance must be in [0, 1)")
    report = parameter_budget_report(model, independent_hidden=independent_hidden)
    if report["actual_joint_parameters"] != report["closed_form_joint_parameters"]:
        raise AssertionError(
            "ResMLP parameter implementation differs from its closed-form count: "
            f"{report}"
        )
    if abs(report["relative_difference_vs_independent"]) > relative_tolerance:
        raise AssertionError(
            "Joint model falls outside the independent-specialist parameter budget: "
            f"tolerance={relative_tolerance}, report={report}"
        )
    return report


@dataclass(frozen=True)
class TaskBatch:
    """One sparse optimizer batch with task-local row indices."""

    epoch: int
    batch_index: int
    round_index: int
    task_names: tuple[str, ...]
    task_indices: tuple[np.ndarray, ...]

    @property
    def group_sizes(self) -> tuple[int, ...]:
        return tuple(int(len(indices)) for indices in self.task_indices)

    @property
    def total_samples(self) -> int:
        return int(sum(self.group_sizes))

    def indices_by_task(self) -> dict[str, np.ndarray]:
        return dict(zip(self.task_names, self.task_indices))


def _stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(int(base_seed)), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


class DeterministicCyclicTaskSampler:
    """Equal-task sampler with deterministic shuffled cyclic streams.

    Each round visits every task exactly once in a deterministic shuffled
    order.  Every visited task contributes ``samples_per_task`` local row
    indices.  A task stream traverses a shuffled permutation without
    replacement, then starts a new independently shuffled cycle.  Large tasks
    therefore continue across epochs, while small tasks are balanced without
    repeatedly sampling arbitrary rows with replacement.

    The compact state stores only cycle numbers and cursors; permutations are
    deterministically reconstructed from ``seed``, task name, and cycle.
    """

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
        self.task_names = tuple(str(name) for name in task_sizes)
        if len(set(self.task_names)) != len(self.task_names):
            raise ValueError("task_sizes contains duplicate task names")
        self.task_sizes = {
            str(name): _positive_integer(size, f"task size for {name}")
            for name, size in task_sizes.items()
        }
        self.tasks_per_batch = _positive_integer(tasks_per_batch, "tasks_per_batch")
        self.samples_per_task = _positive_integer(samples_per_task, "samples_per_task")
        self.seed = int(seed)

        if epoch_samples_per_task is None:
            median_size = float(np.median(list(self.task_sizes.values())))
            requested_quota = int(math.ceil(median_size))
        else:
            requested_quota = _positive_integer(
                epoch_samples_per_task, "epoch_samples_per_task"
            )
        self.rounds_per_epoch = int(
            math.ceil(requested_quota / self.samples_per_task)
        )
        self.epoch_samples_per_task = self.rounds_per_epoch * self.samples_per_task
        self.task_groups_per_round = int(
            math.ceil(len(self.task_names) / self.tasks_per_batch)
        )
        self.batches_per_epoch = self.rounds_per_epoch * self.task_groups_per_round

        self.epoch = 0
        self.next_batch_index = 0
        self._cycles = {name: 0 for name in self.task_names}
        self._cursors = {name: 0 for name in self.task_names}
        self._permutation_cache: dict[str, tuple[int, np.ndarray]] = {}
        self._fingerprint = self._configuration_fingerprint()

    def _configuration_fingerprint(self) -> str:
        payload = repr(
            (
                SAMPLER_SCHEMA_VERSION,
                self.task_names,
                tuple(self.task_sizes[name] for name in self.task_names),
                self.tasks_per_batch,
                self.samples_per_task,
                self.epoch_samples_per_task,
                self.seed,
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _permutation(self, task_name: str) -> np.ndarray:
        cycle = self._cycles[task_name]
        cached = self._permutation_cache.get(task_name)
        if cached is not None and cached[0] == cycle:
            return cached[1]
        generator = np.random.default_rng(
            _stable_seed(self.seed, "task", task_name, "cycle", cycle)
        )
        permutation = generator.permutation(self.task_sizes[task_name]).astype(
            np.int64, copy=False
        )
        self._permutation_cache[task_name] = (cycle, permutation)
        return permutation

    def _draw_task_indices(self, task_name: str) -> np.ndarray:
        pieces: list[np.ndarray] = []
        remaining = self.samples_per_task
        while remaining > 0:
            permutation = self._permutation(task_name)
            cursor = self._cursors[task_name]
            available = len(permutation) - cursor
            take = min(remaining, available)
            pieces.append(permutation[cursor : cursor + take])
            cursor += take
            remaining -= take
            if cursor == len(permutation):
                self._cycles[task_name] += 1
                self._cursors[task_name] = 0
                self._permutation_cache.pop(task_name, None)
            else:
                self._cursors[task_name] = cursor
        if len(pieces) == 1:
            return pieces[0].copy()
        return np.concatenate(pieces).astype(np.int64, copy=False)

    def _round_task_order(self, epoch: int, round_index: int) -> tuple[str, ...]:
        generator = np.random.default_rng(
            _stable_seed(self.seed, "task-order", epoch, round_index)
        )
        order = generator.permutation(len(self.task_names))
        return tuple(self.task_names[int(index)] for index in order)

    def _advance_completed_epoch(self) -> None:
        if self.next_batch_index >= self.batches_per_epoch:
            self.epoch += 1
            self.next_batch_index = 0

    def next_batch(self) -> TaskBatch:
        self._advance_completed_epoch()
        batch_index = self.next_batch_index
        round_index = batch_index // self.task_groups_per_round
        group_index = batch_index % self.task_groups_per_round
        order = self._round_task_order(self.epoch, round_index)
        start = group_index * self.tasks_per_batch
        names = order[start : start + self.tasks_per_batch]
        if not names:
            raise RuntimeError("Internal sampler produced an empty task group")
        indices = tuple(self._draw_task_indices(name) for name in names)
        batch = TaskBatch(
            epoch=self.epoch,
            batch_index=batch_index,
            round_index=round_index,
            task_names=names,
            task_indices=indices,
        )
        # State now points to the next unprocessed batch.  A checkpoint written
        # after the optimizer step resumes without replaying this batch.
        self.next_batch_index += 1
        return batch

    def __iter__(self) -> Iterator[TaskBatch]:
        starting_epoch = self.epoch
        while self.epoch == starting_epoch and self.next_batch_index < len(self):
            yield self.next_batch()
        self._advance_completed_epoch()

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
            raise RuntimeError(f"Unsupported sampler state: {state.get('schema_version')}")
        if state.get("configuration_fingerprint") != self._fingerprint:
            raise RuntimeError(
                "Sampler checkpoint is incompatible with task sizes or sampler config"
            )
        epoch = int(state["epoch"])
        next_batch_index = int(state["next_batch_index"])
        if epoch < 0 or not 0 <= next_batch_index <= self.batches_per_epoch:
            raise RuntimeError("Invalid sampler epoch/batch cursor")
        cycles = {str(name): int(value) for name, value in state["cycles"].items()}
        cursors = {str(name): int(value) for name, value in state["cursors"].items()}
        if set(cycles) != set(self.task_names) or set(cursors) != set(self.task_names):
            raise RuntimeError("Sampler task cursor names do not match current tasks")
        for name in self.task_names:
            if cycles[name] < 0 or not 0 <= cursors[name] < self.task_sizes[name]:
                raise RuntimeError(f"Invalid sampler cursor for {name}")
        self.epoch = epoch
        self.next_batch_index = next_batch_index
        self._cycles = cycles
        self._cursors = cursors
        self._permutation_cache.clear()


def masked_mse_by_task(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Return equal-feature masked MSE for each active reporter task.

    Missing reporters are represented sparsely by absent mapping keys.  Masks
    only describe available elements inside an active task.  A feature is
    averaged over its available cells, then features are averaged equally.
    """

    if not predictions:
        raise ValueError("At least one active task prediction is required")
    if set(predictions) != set(targets):
        raise ValueError("Prediction and target task keys must match exactly")
    if masks is not None and not set(predictions).issubset(masks):
        missing = sorted(set(predictions) - set(masks))
        raise ValueError(f"Missing masks for active tasks: {missing}")

    task_losses: dict[str, torch.Tensor] = {}
    for task_name, prediction in predictions.items():
        target = targets[task_name]
        if prediction.ndim != 2 or target.shape != prediction.shape:
            raise ValueError(
                f"Shape mismatch for {task_name}: prediction={tuple(prediction.shape)}, "
                f"target={tuple(target.shape)}"
            )
        if masks is None:
            valid = torch.isfinite(target)
        else:
            mask = masks[task_name]
            if mask.shape != target.shape:
                raise ValueError(
                    f"Mask shape mismatch for {task_name}: {tuple(mask.shape)} "
                    f"vs {tuple(target.shape)}"
                )
            valid = mask.to(device=target.device, dtype=torch.bool) & torch.isfinite(target)

        # Do not multiply NaN targets by zero: torch.where removes unavailable
        # elements before subtraction.  Loss reduction remains FP32 under AMP.
        prediction_fp32 = prediction.float()
        target_fp32 = torch.where(valid, target, torch.zeros_like(target)).float()
        squared_error = torch.where(
            valid,
            (prediction_fp32 - target_fp32).square(),
            torch.zeros_like(prediction_fp32),
        )
        counts = valid.sum(dim=0)
        valid_features = counts > 0
        if not bool(valid_features.any().item()):
            raise ValueError(f"Task {task_name} has no observed target elements")
        feature_mse = squared_error.sum(dim=0) / counts.clamp_min(1).float()
        task_losses[task_name] = feature_mse[valid_features].mean()
    return task_losses


def masked_balanced_mse(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor] | None = None,
    return_per_task: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Macro-average masked MSE equally across active reporter tasks."""

    task_losses = masked_mse_by_task(predictions, targets, masks=masks)
    loss = torch.stack(tuple(task_losses.values())).mean()
    if return_per_task:
        return loss, task_losses
    return loss


@dataclass(frozen=True)
class AMPPolicy:
    device_type: str
    precision: str
    enabled: bool
    use_grad_scaler: bool

    @property
    def torch_dtype(self) -> torch.dtype | None:
        if self.precision == "bfloat16":
            return torch.bfloat16
        if self.precision == "float16":
            return torch.float16
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_amp_policy(
    device: str | torch.device,
    precision: str = "auto",
) -> AMPPolicy:
    """Resolve an AMP policy; ``auto`` prefers BF16 on supported CUDA GPUs."""

    resolved_device = torch.device(device)
    requested = str(precision).lower()
    aliases = {"fp32": "float32", "bf16": "bfloat16", "fp16": "float16"}
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "float32", "bfloat16", "float16"}:
        raise ValueError("precision must be auto, float32, bfloat16, or float16")

    if requested == "auto":
        if resolved_device.type == "cuda":
            requested = (
                "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            )
        else:
            requested = "float32"
    if requested == "float32":
        return AMPPolicy(resolved_device.type, requested, False, False)
    if requested == "float16" and resolved_device.type != "cuda":
        raise RuntimeError("float16 autocast is supported here only for CUDA")
    if requested == "bfloat16" and resolved_device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA device does not support BF16")
    if requested == "bfloat16" and resolved_device.type not in {"cuda", "cpu"}:
        raise RuntimeError(f"bfloat16 autocast is unsupported for {resolved_device.type}")
    return AMPPolicy(
        resolved_device.type,
        requested,
        True,
        requested == "float16" and resolved_device.type == "cuda",
    )


def autocast_context(policy: AMPPolicy) -> Any:
    if not policy.enabled:
        return contextlib.nullcontext()
    return torch.autocast(
        device_type=policy.device_type,
        dtype=policy.torch_dtype,
        enabled=True,
    )


def create_grad_scaler(policy: AMPPolicy) -> Any:
    """Create a CUDA GradScaler only when FP16 (not BF16) requires it."""

    return torch.cuda.amp.GradScaler(enabled=policy.use_grad_scaler)


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
    # A checkpoint loaded with ``map_location='cuda'`` also maps the serialized
    # RNG ByteTensors to CUDA.  Both CPU and CUDA generator-state setters expect
    # CPU ByteTensors, so normalize them explicitly before restoration.
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_state])


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> Path:
    """Write a torch payload in the destination directory, then atomically replace."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def save_training_checkpoint(
    path: str | os.PathLike[str],
    model: MaskedMultiTaskResMLP,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    grad_scaler: Any | None = None,
    sampler: DeterministicCyclicTaskSampler | None = None,
    training_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically save complete model/training/sampler/RNG state."""

    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "created_unix_time": time.time(),
        "model_config": model.config_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "grad_scaler_state": (
            grad_scaler.state_dict() if grad_scaler is not None else None
        ),
        "sampler_state": sampler.state_dict() if sampler is not None else None,
        "rng_state": capture_rng_state(),
        "training_state": dict(training_state or {}),
        "extra": dict(extra or {}),
    }
    return atomic_torch_save(payload, path)


def _torch_load(path: str | os.PathLike[str], map_location: Any) -> dict[str, Any]:
    # Explicit weights_only=False is needed by newer PyTorch versions because a
    # full training checkpoint intentionally contains RNG tuples and metadata.
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict):
        raise RuntimeError("Training checkpoint payload must be a dictionary")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint schema: {payload.get('schema_version')}"
        )
    return payload


def load_training_checkpoint(
    path: str | os.PathLike[str],
    model: MaskedMultiTaskResMLP,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    grad_scaler: Any | None = None,
    sampler: DeterministicCyclicTaskSampler | None = None,
    map_location: Any = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore supplied objects and return the complete checkpoint payload."""

    payload = _torch_load(path, map_location=map_location)
    expected_config = model.config_dict()
    if payload.get("model_config") != expected_config:
        raise RuntimeError("Checkpoint model_config does not match the supplied model")
    model.load_state_dict(payload["model_state"], strict=strict)

    restore_pairs = (
        (optimizer, "optimizer_state"),
        (scheduler, "scheduler_state"),
        (grad_scaler, "grad_scaler_state"),
    )
    for target, key in restore_pairs:
        saved = payload.get(key)
        if target is not None:
            if saved is None:
                raise RuntimeError(f"Checkpoint has no {key} for requested restoration")
            target.load_state_dict(saved)
    if sampler is not None:
        saved_sampler = payload.get("sampler_state")
        if saved_sampler is None:
            raise RuntimeError("Checkpoint has no sampler_state")
        sampler.load_state_dict(saved_sampler)
    if restore_rng:
        restore_rng_state(payload["rng_state"])
    return payload


def load_model_checkpoint(
    path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> tuple[MaskedMultiTaskResMLP, dict[str, Any]]:
    """Instantiate a model from checkpoint ``model_config`` and load weights."""

    payload = _torch_load(path, map_location=device)
    model = MaskedMultiTaskResMLP.from_config(payload["model_config"])
    model.load_state_dict(payload["model_state"], strict=strict)
    model.to(torch.device(device))
    model.eval()
    return model, payload


def predict_head_batches(
    model: MaskedMultiTaskResMLP,
    x_preprocessed: np.ndarray | torch.Tensor,
    task_name: str,
    batch_size: int = 8192,
    device: str | torch.device | None = None,
    amp_policy: AMPPolicy | None = None,
) -> np.ndarray:
    """Predict one head from an already-preprocessed dense phase matrix."""

    batch_size = _positive_integer(batch_size, "batch_size")
    if device is None:
        try:
            resolved_device = next(model.parameters()).device
        except StopIteration as error:
            raise RuntimeError("Model has no parameters") from error
    else:
        resolved_device = torch.device(device)
    policy = amp_policy or resolve_amp_policy(resolved_device, "auto")
    n_rows = int(len(x_preprocessed))
    output = np.empty((n_rows, model.task_output_dim(task_name)), dtype=np.float32)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            if torch.is_tensor(x_preprocessed):
                batch = x_preprocessed[start:stop].to(
                    resolved_device, dtype=torch.float32, non_blocking=True
                )
            else:
                batch = torch.from_numpy(
                    np.asarray(x_preprocessed[start:stop], dtype=np.float32)
                ).to(resolved_device, non_blocking=True)
            with autocast_context(policy):
                prediction = model(batch, task_name)
            output[start:stop] = prediction.float().cpu().numpy()
    model.train(was_training)
    return output


def _synthetic_ops_head_dimensions() -> dict[str, int]:
    # Same dimensionality distribution as the frozen 52-reporter technical
    # core: 38x24, 6x30, 1x60, 6x72, and 1x20 = 1,604 outputs.
    dimensions = [20] + [30] * 6 + [60] + [72] * 6 + [24] * 38
    if len(dimensions) != 52 or sum(dimensions) != 1604:
        raise AssertionError("Synthetic OPS head schema is malformed")
    return {f"task_{index:02d}": dimension for index, dimension in enumerate(dimensions)}


def run_synthetic_self_test(
    device: str | torch.device = "cpu",
    precision: str = "auto",
    seed: int = 314159,
    checkpoint_directory: str | os.PathLike[str] | None = None,
    capacity_variant: str = "v1",
) -> dict[str, Any]:
    """Exercise all 52 heads, gradients, sampler resume, and checkpoint reload.

    The default is a CPU-only test.  A runner may call the same function with
    ``device='cuda'`` as a real GPU/AMP smoke probe.
    """

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA self-test requested but torch.cuda.is_available() is false")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    head_dimensions = _synthetic_ops_head_dimensions()
    if capacity_variant not in {"v1", "high_dim"}:
        raise ValueError("capacity_variant must be 'v1' or 'high_dim'")
    if capacity_variant == "high_dim":
        head_width = {
            slug: (128 if dimensions >= 60 else 56)
            for slug, dimensions in head_dimensions.items()
        }
        model = MaskedMultiTaskResMLP(
            head_dimensions, expansion_width=568, head_width=head_width
        ).to(resolved_device)
        expected_parameters = 4_310_636
    else:
        model = MaskedMultiTaskResMLP(head_dimensions).to(resolved_device)
        expected_parameters = 4_310_852
    budget = assert_parameter_budget(model)
    if budget["actual_joint_parameters"] != expected_parameters:
        raise AssertionError(f"Unexpected primary parameter count: {budget}")
    if budget["independent_specialist_parameters"] != 4_260_420:
        raise AssertionError(f"Unexpected reference parameter count: {budget}")

    policy = resolve_amp_policy(resolved_device, precision)
    scaler = create_grad_scaler(policy)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    rows_per_task = 2
    total_rows = rows_per_task * len(head_dimensions)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    batch_x = torch.randn(total_rows, DEFAULT_INPUT_DIM, generator=generator).to(
        resolved_device
    )
    targets: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for task_name, output_dim in head_dimensions.items():
        target = torch.randn(rows_per_task, output_dim, generator=generator)
        mask = torch.ones(rows_per_task, output_dim, dtype=torch.bool)
        mask[0, 0] = False
        targets[task_name] = target.to(resolved_device)
        masks[task_name] = mask.to(resolved_device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(policy):
        predictions = model.forward_grouped(
            batch_x,
            tuple(head_dimensions),
            [rows_per_task] * len(head_dimensions),
        )
        loss = masked_balanced_mse(predictions, targets, masks=masks)
    if not bool(torch.isfinite(loss).item()):
        raise AssertionError("Synthetic masked loss is non-finite")
    scaler.scale(loss).backward()
    if scaler.is_enabled():
        scaler.unscale_(optimizer)

    gradient_norms = []
    for task_name, head in zip(model.task_names, model.heads):
        gradient_sum = sum(
            float(parameter.grad.detach().abs().sum().cpu())
            for parameter in head.parameters()
            if parameter.grad is not None
        )
        if not gradient_sum > 0.0:
            raise AssertionError(f"Head {task_name} received no gradient")
        gradient_norms.append(gradient_sum)
    scaler.step(optimizer)
    scaler.update()

    task_sizes = {name: 7 + (index % 5) for index, name in enumerate(head_dimensions)}
    sampler = DeterministicCyclicTaskSampler(
        task_sizes,
        tasks_per_batch=4,
        samples_per_task=4,
        epoch_samples_per_task=12,
        seed=seed + 2,
    )
    for _ in range(5):
        sampler.next_batch()
    sampler_state = sampler.state_dict()
    restored_sampler = DeterministicCyclicTaskSampler(
        task_sizes,
        tasks_per_batch=4,
        samples_per_task=4,
        epoch_samples_per_task=12,
        seed=seed + 2,
    )
    restored_sampler.load_state_dict(sampler_state)
    original_future = [sampler.next_batch() for _ in range(6)]
    restored_future = [restored_sampler.next_batch() for _ in range(6)]
    for original, restored in zip(original_future, restored_future):
        if original.task_names != restored.task_names or any(
            not np.array_equal(left, right)
            for left, right in zip(original.task_indices, restored.task_indices)
        ):
            raise AssertionError("Cyclic sampler state did not resume exactly")

    probe_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    probe_x = torch.randn(3, DEFAULT_INPUT_DIM, generator=probe_generator).to(
        resolved_device
    )
    model.eval()
    with torch.no_grad(), autocast_context(policy):
        expected_prediction = model(probe_x, model.task_names[0]).float().cpu().clone()

    # Prepare the next expected sampler batch from the exact state written to
    # checkpoint without mutating the sampler being checkpointed.
    checkpoint_sampler_clone = DeterministicCyclicTaskSampler(
        task_sizes,
        tasks_per_batch=4,
        samples_per_task=4,
        epoch_samples_per_task=12,
        seed=seed + 2,
    )
    checkpoint_sampler_clone.load_state_dict(sampler.state_dict())
    expected_next_batch = checkpoint_sampler_clone.next_batch()

    temporary_context: Any
    if checkpoint_directory is None:
        temporary_context = tempfile.TemporaryDirectory(
            prefix="ops_masked_multitask_selftest_"
        )
    else:
        Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)
        temporary_context = contextlib.nullcontext(str(checkpoint_directory))

    with temporary_context as directory:
        checkpoint_path = Path(directory) / "synthetic_checkpoint.pt"
        save_training_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            grad_scaler=scaler,
            sampler=sampler,
            training_state={"epoch": 0, "global_step": 1, "best_validation_mse": 1.0},
            extra={"self_test": True},
        )
        expected_rng = (
            random.random(),
            float(np.random.random()),
            torch.rand(1).cpu().item(),
        )
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        sampler.next_batch()
        payload = load_training_checkpoint(
            checkpoint_path,
            model,
            optimizer=optimizer,
            grad_scaler=scaler,
            sampler=sampler,
            map_location=resolved_device,
            restore_rng=True,
        )
        actual_rng = (
            random.random(),
            float(np.random.random()),
            torch.rand(1).cpu().item(),
        )
        if actual_rng != expected_rng:
            raise AssertionError("RNG state did not resume exactly")
        actual_next_batch = sampler.next_batch()
        if actual_next_batch.task_names != expected_next_batch.task_names or any(
            not np.array_equal(left, right)
            for left, right in zip(
                actual_next_batch.task_indices, expected_next_batch.task_indices
            )
        ):
            raise AssertionError("Checkpoint sampler restoration is not exact")
        with torch.no_grad(), autocast_context(policy):
            restored_prediction = model(probe_x, model.task_names[0]).float().cpu()
        if not torch.equal(restored_prediction, expected_prediction):
            maximum_difference = float(
                (restored_prediction - expected_prediction).abs().max()
            )
            raise AssertionError(
                f"Checkpoint prediction changed after exact reload: {maximum_difference}"
            )

    return {
        "passed": True,
        "device": str(resolved_device),
        "amp_policy": policy.to_dict(),
        "n_heads": len(head_dimensions),
        "total_output_dimensions": sum(head_dimensions.values()),
        "masked_loss": float(loss.detach().cpu()),
        "minimum_head_gradient_l1": min(gradient_norms),
        "all_heads_received_gradient": len(gradient_norms) == 52,
        "parameter_budget": budget,
        "capacity_variant": capacity_variant,
        "sampler_resume_exact": True,
        "checkpoint_prediction_exact": True,
        "rng_resume_exact": True,
        "checkpoint_schema_version": payload["schema_version"],
    }


def run_cpu_synthetic_self_test(
    seed: int = 314159,
    checkpoint_directory: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Explicit CPU entry point requested by validation and unit-test runners."""

    return run_synthetic_self_test(
        device="cpu",
        precision="float32",
        seed=seed,
        checkpoint_directory=checkpoint_directory,
    )


__all__ = [
    "AMPPolicy",
    "CHECKPOINT_SCHEMA_VERSION",
    "DEFAULT_DROPOUT",
    "DEFAULT_EXPANSION_WIDTH",
    "DEFAULT_HEAD_WIDTH",
    "DEFAULT_INPUT_DIM",
    "DEFAULT_RESIDUAL_BLOCKS",
    "DEFAULT_SHARED_WIDTH",
    "DeterministicCyclicTaskSampler",
    "MODEL_SCHEMA_VERSION",
    "MaskedMultiTaskResMLP",
    "PreNormResidualFFN",
    "ReporterHead",
    "SAMPLER_SCHEMA_VERSION",
    "TaskBatch",
    "assert_parameter_budget",
    "atomic_torch_save",
    "autocast_context",
    "capture_rng_state",
    "create_grad_scaler",
    "expected_resmlp_parameter_count",
    "independent_mlp_parameter_count",
    "load_model_checkpoint",
    "load_training_checkpoint",
    "masked_balanced_mse",
    "masked_mse_by_task",
    "parameter_budget_report",
    "predict_head_batches",
    "resolve_amp_policy",
    "restore_rng_state",
    "run_cpu_synthetic_self_test",
    "run_synthetic_self_test",
    "save_training_checkpoint",
]
