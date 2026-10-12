#!/usr/bin/env python3
"""Reporting-only runtime and capacity measurements for OPS candidates.

The measurements in this module are deliberately separated from validation
losses and HPO selection.  They describe the computational footprint of an
already specified trial; they are never an admission rule, an optimisation
objective, or a tie breaker.

``active_parameter_count`` has one architecture-independent definition: the
number of scalar values in trainable parameter tensors whose ``.grad`` is not
``None`` immediately after the current batch's backward pass.  The training
loop must therefore use ``optimizer.zero_grad(set_to_none=True)`` before the
forward pass.  This definition counts a parameter tensor once even when it is
shared or invoked repeatedly.  It intentionally does not count non-zero
gradient entries, because those depend on activations and numerical
cancellation rather than graph participation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence


RUNTIME_METRICS_SCHEMA = "ops-reporter-runtime-metrics-v1"
PARAMETER_INVENTORY_SCHEMA = "ops-reporter-parameter-inventory-v1"
ACTIVE_BATCH_SCHEMA = "ops-reporter-active-batch-parameters-v1"
ACTIVE_LEDGER_SCHEMA = "ops-reporter-active-signature-ledger-v1"
THROUGHPUT_SCHEMA = "ops-reporter-inference-throughput-v1"

REPORTING_ONLY_CONTRACT = {
    "role": "reporting_only",
    "eligible_for_admission": False,
    "eligible_for_hyperparameter_or_architecture_selection": False,
    "eligible_for_checkpoint_selection": False,
    "eligible_for_tie_breaking": False,
    "parameter_count_is_optimization_target": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _nonnegative_int(value: Any, label: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{label} must be an integer")
    _require(value >= 0, f"{label} must be nonnegative")
    return int(value)


def _positive_finite(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{label} must be numeric")
    result = float(value)
    _require(math.isfinite(result) and result > 0.0, f"{label} must be positive and finite")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ParameterInventory:
    """Model-wide parameter and buffer storage inventory."""

    total_parameter_count: int
    trainable_parameter_count: int
    frozen_parameter_count: int
    parameter_tensor_count: int
    trainable_parameter_tensor_count: int
    parameter_bytes: int
    trainable_parameter_bytes: int
    buffer_count: int
    buffer_bytes: int

    @classmethod
    def from_model(cls, model: Any) -> "ParameterInventory":
        # ``parameters()`` and ``buffers()`` deduplicate shared objects in
        # supported PyTorch versions.  Explicit id sets make that invariant
        # visible and protect against custom iterators yielding aliases.
        parameters = []
        seen: set[int] = set()
        for value in model.parameters():
            if id(value) not in seen:
                parameters.append(value)
                seen.add(id(value))
        buffers = []
        seen_buffers: set[int] = set()
        for value in model.buffers():
            if id(value) not in seen_buffers:
                buffers.append(value)
                seen_buffers.add(id(value))
        total = sum(int(value.numel()) for value in parameters)
        trainable = sum(int(value.numel()) for value in parameters if value.requires_grad)
        return cls(
            total_parameter_count=total,
            trainable_parameter_count=trainable,
            frozen_parameter_count=total - trainable,
            parameter_tensor_count=len(parameters),
            trainable_parameter_tensor_count=sum(value.requires_grad for value in parameters),
            parameter_bytes=sum(int(value.numel() * value.element_size()) for value in parameters),
            trainable_parameter_bytes=sum(
                int(value.numel() * value.element_size())
                for value in parameters
                if value.requires_grad
            ),
            buffer_count=len(buffers),
            buffer_bytes=sum(int(value.numel() * value.element_size()) for value in buffers),
        )

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "schema_version": PARAMETER_INVENTORY_SCHEMA,
            **asdict(self),
            "reporting_contract": dict(REPORTING_ONLY_CONTRACT),
        }
        validate_parameter_inventory(payload)
        return payload


def validate_parameter_inventory(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == PARAMETER_INVENTORY_SCHEMA, "parameter inventory schema changed")
    total = _nonnegative_int(payload.get("total_parameter_count"), "total_parameter_count")
    trainable = _nonnegative_int(payload.get("trainable_parameter_count"), "trainable_parameter_count")
    frozen = _nonnegative_int(payload.get("frozen_parameter_count"), "frozen_parameter_count")
    _require(total == trainable + frozen, "parameter partition does not sum to total")
    for key in (
        "parameter_tensor_count",
        "trainable_parameter_tensor_count",
        "parameter_bytes",
        "trainable_parameter_bytes",
        "buffer_count",
        "buffer_bytes",
    ):
        _nonnegative_int(payload.get(key), key)
    _require(payload.get("reporting_contract") == REPORTING_ONLY_CONTRACT, "parameter metrics are not reporting-only")
    return dict(payload)


def _normalise_labels(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    labels = tuple(str(value) for value in values)
    _require(all(labels), "active reporter/endpoint labels must be non-empty")
    _require(len(labels) == len(set(labels)), "active reporter/endpoint labels must be unique")
    return labels


def active_parameter_snapshot_after_backward(
    model: Any,
    *,
    optimizer_update: int,
    active_reporters: Sequence[str],
    active_endpoint_queries: Sequence[str] | None = None,
    cell_observations: int,
    observed_endpoint_predictions: int,
) -> dict[str, Any]:
    """Measure graph-active trainable storage after one backward pass.

    Call after ``loss.backward()`` and before ``optimizer.step()``.  The
    reporter and endpoint labels describe the sparse query set instantiated by
    that physical batch.  Endpoint labels may be omitted for vector-head
    models; ``observed_endpoint_predictions`` still records the exact masked
    loss denominator.
    """

    update = _nonnegative_int(optimizer_update, "optimizer_update")
    # Active capacity is a set property.  Canonical sorting prevents the same
    # sparse reporter/query set from producing multiple signatures solely due
    # to grouped-batch order.
    reporters = tuple(sorted(_normalise_labels(active_reporters)))
    endpoints = tuple(sorted(_normalise_labels(active_endpoint_queries)))
    _require(reporters, "at least one active reporter is required")
    cells = _nonnegative_int(cell_observations, "cell_observations")
    predictions = _nonnegative_int(
        observed_endpoint_predictions, "observed_endpoint_predictions"
    )
    _require(cells > 0, "cell_observations must be positive")
    _require(predictions > 0, "observed_endpoint_predictions must be positive")

    parameters = []
    seen: set[int] = set()
    for value in model.parameters():
        if value.requires_grad and id(value) not in seen:
            parameters.append(value)
            seen.add(id(value))
    active = [value for value in parameters if value.grad is not None]
    total_count = sum(int(value.numel()) for value in parameters)
    active_count = sum(int(value.numel()) for value in active)
    _require(
        active_count > 0,
        "no graph-active parameters found; snapshot must be taken after backward",
    )

    signature_payload = {
        "active_reporters": list(reporters),
        "active_endpoint_queries": list(endpoints),
    }
    payload = {
        "schema_version": ACTIVE_BATCH_SCHEMA,
        "optimizer_update": update,
        "active_signature_sha256": _sha256_json(signature_payload),
        **signature_payload,
        "active_reporter_count": len(reporters),
        "active_endpoint_query_count": len(endpoints),
        "cell_observations": cells,
        "observed_endpoint_predictions": predictions,
        "active_parameter_count": active_count,
        "active_parameter_tensor_count": len(active),
        "total_trainable_parameter_count": total_count,
        "active_trainable_fraction": active_count / total_count,
        "definition": "trainable scalar storage with grad_not_none_after_current_backward",
        "requires_zero_grad_set_to_none": True,
        "nonzero_gradient_entry_count_is_not_used": True,
        "reporting_contract": dict(REPORTING_ONLY_CONTRACT),
    }
    validate_active_parameter_snapshot(payload)
    return payload


def validate_active_parameter_snapshot(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == ACTIVE_BATCH_SCHEMA, "active batch schema changed")
    _nonnegative_int(payload.get("optimizer_update"), "optimizer_update")
    active = _nonnegative_int(payload.get("active_parameter_count"), "active_parameter_count")
    total = _nonnegative_int(payload.get("total_trainable_parameter_count"), "total_trainable_parameter_count")
    _require(0 < active <= total, "active_parameter_count must be in (0, total]")
    _require(payload.get("definition") == "trainable scalar storage with grad_not_none_after_current_backward", "active parameter definition changed")
    _require(payload.get("requires_zero_grad_set_to_none") is True, "active count requires set_to_none gradients")
    _require(payload.get("nonzero_gradient_entry_count_is_not_used") is True, "active count cannot depend on gradient values")
    _require(payload.get("reporting_contract") == REPORTING_ONLY_CONTRACT, "active metrics are not reporting-only")
    signature = {
        "active_reporters": list(_normalise_labels(payload.get("active_reporters"))),
        "active_endpoint_queries": list(_normalise_labels(payload.get("active_endpoint_queries"))),
    }
    _require(
        signature["active_reporters"] == sorted(signature["active_reporters"])
        and signature["active_endpoint_queries"]
        == sorted(signature["active_endpoint_queries"]),
        "active signature labels are not canonical",
    )
    _require(
        payload.get("active_reporter_count") == len(signature["active_reporters"]),
        "active_reporter_count mismatch",
    )
    _require(
        payload.get("active_endpoint_query_count")
        == len(signature["active_endpoint_queries"]),
        "active_endpoint_query_count mismatch",
    )
    _require(payload.get("active_signature_sha256") == _sha256_json(signature), "active signature fingerprint mismatch")
    fraction = float(payload.get("active_trainable_fraction"))
    _require(math.isclose(fraction, active / total, rel_tol=0.0, abs_tol=1e-15), "active fraction is inconsistent")
    return dict(payload)


class CompactActiveSignatureLedger:
    """Resume-safe occurrence ledger without one verbose row per update.

    Reporter/endpoint query labels are stored once per signature.  Every
    physical optimizer update contributes only aggregate occurrence, cell,
    endpoint-prediction and active-capacity statistics.  Contiguous update
    enforcement detects a missing or duplicate measurement at write time.

    Active counts are allowed to vary within one reporter/endpoint signature,
    which keeps the contract valid for future sparse conditional-execution
    models.  A compact histogram preserves that variation exactly.
    """

    def __init__(self) -> None:
        self._vocabulary: dict[str, dict[str, Any]] = {}
        self._occurrences: dict[str, dict[str, Any]] = {}
        self._first_optimizer_update: int | None = None
        self._last_optimizer_update = 0
        self._n_observed_batches = 0
        self._total_cell_observations = 0
        self._total_observed_endpoint_predictions = 0

    def record(self, snapshot: Mapping[str, Any]) -> None:
        row = validate_active_parameter_snapshot(snapshot)
        update = int(row["optimizer_update"])
        expected = 1 if self._first_optimizer_update is None else self._last_optimizer_update + 1
        _require(
            update == expected,
            f"active ledger expected optimizer_update={expected}, found {update}",
        )
        signature = str(row["active_signature_sha256"])
        vocabulary = {
            "active_signature_sha256": signature,
            "active_reporters": list(row["active_reporters"]),
            "active_endpoint_queries": list(row["active_endpoint_queries"]),
            "active_reporter_count": int(row["active_reporter_count"]),
            "active_endpoint_query_count": int(row["active_endpoint_query_count"]),
        }
        existing_vocabulary = self._vocabulary.get(signature)
        if existing_vocabulary is None:
            self._vocabulary[signature] = vocabulary
        else:
            _require(
                existing_vocabulary == vocabulary,
                "active signature hash maps to inconsistent reporter/endpoint labels",
            )

        active_count = int(row["active_parameter_count"])
        tensor_count = int(row["active_parameter_tensor_count"])
        total_trainable = int(row["total_trainable_parameter_count"])
        occurrence = self._occurrences.get(signature)
        if occurrence is None:
            occurrence = {
                "active_signature_sha256": signature,
                "batch_occurrences": 0,
                "first_optimizer_update": update,
                "last_optimizer_update": update,
                "total_cell_observations": 0,
                "total_observed_endpoint_predictions": 0,
                "active_parameter_count_min": active_count,
                "active_parameter_count_max": active_count,
                "active_parameter_count_sum": 0,
                "active_parameter_count_histogram": {},
                "active_parameter_tensor_count_histogram": {},
                "total_trainable_parameter_count": total_trainable,
            }
            self._occurrences[signature] = occurrence
        else:
            _require(
                occurrence["total_trainable_parameter_count"] == total_trainable,
                "total trainable parameter count changed within one run",
            )
            occurrence["last_optimizer_update"] = update
            occurrence["active_parameter_count_min"] = min(
                occurrence["active_parameter_count_min"], active_count
            )
            occurrence["active_parameter_count_max"] = max(
                occurrence["active_parameter_count_max"], active_count
            )
        occurrence["batch_occurrences"] += 1
        occurrence["total_cell_observations"] += int(row["cell_observations"])
        occurrence["total_observed_endpoint_predictions"] += int(
            row["observed_endpoint_predictions"]
        )
        occurrence["active_parameter_count_sum"] += active_count
        for key, value in (
            ("active_parameter_count_histogram", active_count),
            ("active_parameter_tensor_count_histogram", tensor_count),
        ):
            histogram = occurrence[key]
            label = str(value)
            histogram[label] = int(histogram.get(label, 0)) + 1

        if self._first_optimizer_update is None:
            self._first_optimizer_update = update
        self._last_optimizer_update = update
        self._n_observed_batches += 1
        self._total_cell_observations += int(row["cell_observations"])
        self._total_observed_endpoint_predictions += int(
            row["observed_endpoint_predictions"]
        )

    @property
    def last_optimizer_update(self) -> int:
        return self._last_optimizer_update

    @property
    def n_observed_batches(self) -> int:
        return self._n_observed_batches

    def to_manifest(self) -> dict[str, Any]:
        _require(self._n_observed_batches > 0, "active signature ledger is empty")
        core = {
            "schema_version": ACTIVE_LEDGER_SCHEMA,
            "definition": "trainable scalar storage with grad_not_none_after_current_backward",
            "requires_zero_grad_set_to_none": True,
            "nonzero_gradient_entry_count_is_not_used": True,
            "first_optimizer_update": self._first_optimizer_update,
            "last_optimizer_update": self._last_optimizer_update,
            "n_observed_batches": self._n_observed_batches,
            "n_active_signatures": len(self._vocabulary),
            "total_cell_observations": self._total_cell_observations,
            "total_observed_endpoint_predictions": self._total_observed_endpoint_predictions,
            "signature_vocabulary": {
                key: self._vocabulary[key] for key in sorted(self._vocabulary)
            },
            "occurrence_by_signature": {
                key: self._occurrences[key] for key in sorted(self._occurrences)
            },
            "storage_policy": "signature_vocabulary_once_plus_aggregated_occurrences",
            "per_optimizer_update_measurement_required": True,
            "selection_inputs": [],
            "reporting_contract": dict(REPORTING_ONLY_CONTRACT),
        }
        core["manifest_sha256"] = _sha256_json(core)
        validate_active_signature_ledger(core)
        return core

    def state_dict(self) -> dict[str, Any]:
        return self.to_manifest()

    @classmethod
    def from_manifest(
        cls, payload: Mapping[str, Any]
    ) -> "CompactActiveSignatureLedger":
        validated = validate_active_signature_ledger(payload)
        result = cls()
        result._first_optimizer_update = int(validated["first_optimizer_update"])
        result._last_optimizer_update = int(validated["last_optimizer_update"])
        result._n_observed_batches = int(validated["n_observed_batches"])
        result._total_cell_observations = int(validated["total_cell_observations"])
        result._total_observed_endpoint_predictions = int(
            validated["total_observed_endpoint_predictions"]
        )
        result._vocabulary = {
            str(key): dict(value)
            for key, value in validated["signature_vocabulary"].items()
        }
        result._occurrences = {
            str(key): {
                **dict(value),
                "active_parameter_count_histogram": dict(
                    value["active_parameter_count_histogram"]
                ),
                "active_parameter_tensor_count_histogram": dict(
                    value["active_parameter_tensor_count_histogram"]
                ),
            }
            for key, value in validated["occurrence_by_signature"].items()
        }
        return result

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        restored = type(self).from_manifest(payload)
        self._first_optimizer_update = restored._first_optimizer_update
        self._last_optimizer_update = restored._last_optimizer_update
        self._n_observed_batches = restored._n_observed_batches
        self._total_cell_observations = restored._total_cell_observations
        self._total_observed_endpoint_predictions = (
            restored._total_observed_endpoint_predictions
        )
        self._vocabulary = restored._vocabulary
        self._occurrences = restored._occurrences


def validate_active_signature_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("schema_version") == ACTIVE_LEDGER_SCHEMA,
        "active signature ledger schema changed",
    )
    _require(payload.get("selection_inputs") == [], "active ledger cannot be a selection input")
    _require(
        payload.get("reporting_contract") == REPORTING_ONLY_CONTRACT,
        "active ledger is not reporting-only",
    )
    _require(
        payload.get("definition")
        == "trainable scalar storage with grad_not_none_after_current_backward",
        "active ledger definition changed",
    )
    _require(
        payload.get("requires_zero_grad_set_to_none") is True,
        "active ledger requires set_to_none gradients",
    )
    _require(
        payload.get("nonzero_gradient_entry_count_is_not_used") is True,
        "active ledger cannot depend on gradient values",
    )
    unhashed = dict(payload)
    declared = unhashed.pop("manifest_sha256", None)
    _require(declared == _sha256_json(unhashed), "active signature ledger hash mismatch")
    first = _nonnegative_int(payload.get("first_optimizer_update"), "first_optimizer_update")
    last = _nonnegative_int(payload.get("last_optimizer_update"), "last_optimizer_update")
    batches = _nonnegative_int(payload.get("n_observed_batches"), "n_observed_batches")
    _require(first == 1 and last >= first, "active ledger must begin at optimizer update 1")
    _require(last - first + 1 == batches, "active ledger updates are not contiguous")
    vocabulary = payload.get("signature_vocabulary")
    occurrences = payload.get("occurrence_by_signature")
    _require(isinstance(vocabulary, Mapping) and vocabulary, "signature vocabulary is empty")
    _require(isinstance(occurrences, Mapping), "signature occurrences are invalid")
    _require(set(vocabulary) == set(occurrences), "signature vocabulary/occurrence keys differ")
    _require(
        _nonnegative_int(payload.get("n_active_signatures"), "n_active_signatures")
        == len(vocabulary),
        "n_active_signatures mismatch",
    )
    total_batches = 0
    total_cells = 0
    total_predictions = 0
    total_trainable_counts: set[int] = set()
    for signature, definition in vocabulary.items():
        _require(
            isinstance(signature, str)
            and len(signature) == 64
            and all(character in "0123456789abcdef" for character in signature)
            and definition.get("active_signature_sha256") == signature,
            "invalid active signature vocabulary entry",
        )
        signature_payload = {
            "active_reporters": list(_normalise_labels(definition.get("active_reporters"))),
            "active_endpoint_queries": list(
                _normalise_labels(definition.get("active_endpoint_queries"))
            ),
        }
        _require(
            signature_payload["active_reporters"]
            == sorted(signature_payload["active_reporters"])
            and signature_payload["active_endpoint_queries"]
            == sorted(signature_payload["active_endpoint_queries"]),
            "signature vocabulary labels are not canonical",
        )
        _require(
            definition.get("active_reporter_count")
            == len(signature_payload["active_reporters"])
            and definition.get("active_endpoint_query_count")
            == len(signature_payload["active_endpoint_queries"]),
            "signature vocabulary label counts mismatch",
        )
        _require(_sha256_json(signature_payload) == signature, "signature vocabulary hash mismatch")
        occurrence = occurrences[signature]
        _require(
            occurrence.get("active_signature_sha256") == signature,
            "occurrence signature mismatch",
        )
        count = _nonnegative_int(occurrence.get("batch_occurrences"), "batch_occurrences")
        _require(count > 0, "signature occurrence must be positive")
        occurrence_first = _nonnegative_int(
            occurrence.get("first_optimizer_update"), "signature first_optimizer_update"
        )
        occurrence_last = _nonnegative_int(
            occurrence.get("last_optimizer_update"), "signature last_optimizer_update"
        )
        _require(
            first <= occurrence_first <= occurrence_last <= last
            and count <= occurrence_last - occurrence_first + 1,
            "signature occurrence update range is invalid",
        )
        histogram = occurrence.get("active_parameter_count_histogram")
        tensor_histogram = occurrence.get("active_parameter_tensor_count_histogram")
        _require(isinstance(histogram, Mapping) and histogram, "active count histogram is empty")
        _require(isinstance(tensor_histogram, Mapping) and tensor_histogram, "active tensor histogram is empty")
        histogram_count = sum(_nonnegative_int(value, "histogram occurrence") for value in histogram.values())
        tensor_histogram_count = sum(
            _nonnegative_int(value, "tensor histogram occurrence")
            for value in tensor_histogram.values()
        )
        _require(histogram_count == count == tensor_histogram_count, "active histograms do not sum to occurrences")
        weighted = sum(int(key) * int(value) for key, value in histogram.items())
        _require(weighted == occurrence.get("active_parameter_count_sum"), "active parameter sum mismatch")
        values = [int(key) for key in histogram]
        _require(min(values) == occurrence.get("active_parameter_count_min"), "active parameter minimum mismatch")
        _require(max(values) == occurrence.get("active_parameter_count_max"), "active parameter maximum mismatch")
        total_trainable = _nonnegative_int(
            occurrence.get("total_trainable_parameter_count"),
            "total_trainable_parameter_count",
        )
        _require(total_trainable > 0, "total trainable parameter count must be positive")
        _require(
            int(occurrence["active_parameter_count_max"]) <= total_trainable,
            "active parameter count exceeds total trainable parameters",
        )
        total_trainable_counts.add(total_trainable)
        total_batches += count
        signature_cells = _nonnegative_int(
            occurrence.get("total_cell_observations"), "total_cell_observations"
        )
        signature_predictions = _nonnegative_int(
            occurrence.get("total_observed_endpoint_predictions"),
            "total_observed_endpoint_predictions",
        )
        _require(
            signature_cells >= count and signature_predictions >= count,
            "each active occurrence must contain cells and endpoint predictions",
        )
        total_cells += signature_cells
        total_predictions += signature_predictions
    _require(
        len(total_trainable_counts) == 1,
        "total trainable parameter count changed across signatures",
    )
    _require(total_batches == batches, "signature occurrences do not sum to batches")
    _require(total_cells == payload.get("total_cell_observations"), "total cell observations mismatch")
    _require(
        total_predictions == payload.get("total_observed_endpoint_predictions"),
        "total observed endpoint predictions mismatch",
    )
    return dict(payload)


def summarize_active_signature_ledger(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    ledger = validate_active_signature_ledger(payload)
    occurrences = ledger["occurrence_by_signature"]
    batches = int(ledger["n_observed_batches"])
    weighted = sum(
        int(row["active_parameter_count_sum"])
        for row in occurrences.values()
    )
    return {
        "definition": ledger["definition"],
        "n_observed_batches": batches,
        "n_active_signatures": int(ledger["n_active_signatures"]),
        "active_parameter_count_min": min(
            int(row["active_parameter_count_min"])
            for row in occurrences.values()
        ),
        "active_parameter_count_max": max(
            int(row["active_parameter_count_max"])
            for row in occurrences.values()
        ),
        "active_parameter_count_mean": weighted / batches,
        "batch_count_by_active_signature": {
            str(signature): int(row["batch_occurrences"])
            for signature, row in sorted(occurrences.items())
        },
        "reporting_contract": dict(REPORTING_ONLY_CONTRACT),
    }


class CudaTrainingResourceMeter:
    """Accumulate CUDA-event train time and peak memory for one process.

    ``begin`` should be called after model and GPU-resident dataset staging.
    One region should surround each epoch's *training updates only*.  Validation,
    checkpoint serialization, and CPU data preparation are intentionally
    excluded from ``training_gpu_seconds`` and remain represented by the
    runner's end-to-end wall time.
    """

    def __init__(self, torch_module: Any, device: Any = None) -> None:
        self.torch = torch_module
        self.device = device
        self._begun = False
        self._open: tuple[Any, float] | None = None
        self._gpu_milliseconds = 0.0
        self._wall_seconds = 0.0
        self._regions = 0
        self._baseline_allocated = 0
        self._baseline_reserved = 0
        self._prior_peak_allocated = 0
        self._prior_peak_reserved = 0
        self._meter_sessions = 0

    def begin(self) -> None:
        _require(not self._begun, "resource meter already begun")
        cuda = self.torch.cuda
        cuda.synchronize(self.device)
        cuda.reset_peak_memory_stats(self.device)
        self._baseline_allocated = int(cuda.memory_allocated(self.device))
        self._baseline_reserved = int(cuda.memory_reserved(self.device))
        self._meter_sessions = 1
        self._begun = True

    def start_training_region(self) -> None:
        _require(self._begun, "resource meter must begin before a region")
        _require(self._open is None, "training regions cannot overlap")
        event = self.torch.cuda.Event(enable_timing=True)
        event.record()
        self._open = (event, time.perf_counter())

    def stop_training_region(self) -> dict[str, float]:
        _require(self._open is not None, "no training region is open")
        start_event, wall_start = self._open
        end_event = self.torch.cuda.Event(enable_timing=True)
        end_event.record()
        end_event.synchronize()
        gpu_ms = float(start_event.elapsed_time(end_event))
        wall = time.perf_counter() - wall_start
        _require(math.isfinite(gpu_ms) and gpu_ms >= 0.0, "invalid CUDA event duration")
        self._gpu_milliseconds += gpu_ms
        self._wall_seconds += wall
        self._regions += 1
        self._open = None
        return {"gpu_seconds": gpu_ms / 1000.0, "wall_seconds": wall}

    def state_dict(self) -> dict[str, Any]:
        """Serializable epoch-boundary state for an exact resume ledger."""

        _require(self._begun, "resource meter did not begin")
        _require(self._open is None, "resource meter state requires a closed region")
        cuda = self.torch.cuda
        return {
            "gpu_milliseconds": self._gpu_milliseconds,
            "training_region_wall_seconds": self._wall_seconds,
            "training_regions": self._regions,
            "meter_sessions": self._meter_sessions,
            "memory_initial_baseline_allocated_bytes": self._baseline_allocated,
            "memory_initial_baseline_reserved_bytes": self._baseline_reserved,
            "peak_allocated_bytes": max(
                self._prior_peak_allocated,
                int(cuda.max_memory_allocated(self.device)),
            ),
            "peak_reserved_bytes": max(
                self._prior_peak_reserved,
                int(cuda.max_memory_reserved(self.device)),
            ),
        }

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        """Restore cumulative times and prior-process peaks after ``begin``."""

        _require(self._begun, "resource meter must begin before resume restore")
        _require(self._open is None, "cannot restore during an open region")
        _require(
            self._regions == 0
            and self._gpu_milliseconds == 0.0
            and self._wall_seconds == 0.0,
            "resume state must be restored before recording a new region",
        )
        milliseconds = payload.get("gpu_milliseconds")
        wall = payload.get("training_region_wall_seconds")
        _require(
            isinstance(milliseconds, (int, float))
            and math.isfinite(float(milliseconds))
            and float(milliseconds) >= 0.0,
            "gpu_milliseconds must be finite and nonnegative",
        )
        _require(
            isinstance(wall, (int, float))
            and math.isfinite(float(wall))
            and float(wall) >= 0.0,
            "training_region_wall_seconds must be finite and nonnegative",
        )
        self._gpu_milliseconds = float(milliseconds)
        self._wall_seconds = float(wall)
        self._regions = _nonnegative_int(
            payload.get("training_regions"), "training_regions"
        )
        self._meter_sessions = (
            _nonnegative_int(payload.get("meter_sessions"), "meter_sessions") + 1
        )
        self._baseline_allocated = _nonnegative_int(
            payload.get("memory_initial_baseline_allocated_bytes"),
            "memory_initial_baseline_allocated_bytes",
        )
        self._baseline_reserved = _nonnegative_int(
            payload.get("memory_initial_baseline_reserved_bytes"),
            "memory_initial_baseline_reserved_bytes",
        )
        self._prior_peak_allocated = _nonnegative_int(
            payload.get("peak_allocated_bytes"), "peak_allocated_bytes"
        )
        self._prior_peak_reserved = _nonnegative_int(
            payload.get("peak_reserved_bytes"), "peak_reserved_bytes"
        )

    def to_manifest(self) -> dict[str, Any]:
        _require(self._begun, "resource meter did not begin")
        _require(self._open is None, "cannot finalize an open training region")
        cuda = self.torch.cuda
        cuda.synchronize(self.device)
        peak_allocated = max(
            self._prior_peak_allocated,
            int(cuda.max_memory_allocated(self.device)),
        )
        peak_reserved = max(
            self._prior_peak_reserved,
            int(cuda.max_memory_reserved(self.device)),
        )
        payload = {
            "training_gpu_seconds": self._gpu_milliseconds / 1000.0,
            "training_region_wall_seconds": self._wall_seconds,
            "training_regions": self._regions,
            "meter_sessions": self._meter_sessions,
            "memory_baseline_allocated_bytes": self._baseline_allocated,
            "memory_baseline_reserved_bytes": self._baseline_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_increment_bytes": max(0, peak_allocated - self._baseline_allocated),
            "peak_reserved_increment_bytes": max(0, peak_reserved - self._baseline_reserved),
            "memory_scope": "model_plus_optimizer_plus_ema_plus_gpu_staged_data_and_training_workspace",
            "memory_backend": "torch_cuda_allocator",
            "non_pytorch_or_external_process_memory_included": False,
            "gpu_time_scope": "forward_backward_optimizer_updates_only",
            "reporting_contract": dict(REPORTING_ONLY_CONTRACT),
        }
        validate_resource_metrics(payload)
        return payload


def validate_resource_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    for key in (
        "training_gpu_seconds",
        "training_region_wall_seconds",
    ):
        value = payload.get(key)
        _require(isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0, f"{key} must be finite and nonnegative")
    for key in (
        "training_regions",
        "meter_sessions",
        "memory_baseline_allocated_bytes",
        "memory_baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_allocated_increment_bytes",
        "peak_reserved_increment_bytes",
    ):
        _nonnegative_int(payload.get(key), key)
    _require(int(payload.get("meter_sessions")) > 0, "meter_sessions must be positive")
    _require(
        payload.get("memory_scope")
        == "model_plus_optimizer_plus_ema_plus_gpu_staged_data_and_training_workspace"
        and payload.get("memory_backend") == "torch_cuda_allocator"
        and payload.get("non_pytorch_or_external_process_memory_included") is False,
        "VRAM measurement scope changed",
    )
    _require(
        payload.get("gpu_time_scope")
        == "forward_backward_optimizer_updates_only",
        "GPU training-time scope changed",
    )
    _require(
        int(payload.get("peak_allocated_bytes"))
        >= int(payload.get("memory_baseline_allocated_bytes")),
        "allocated peak is below baseline",
    )
    _require(
        int(payload.get("peak_reserved_bytes"))
        >= int(payload.get("memory_baseline_reserved_bytes")),
        "reserved peak is below baseline",
    )
    _require(
        int(payload.get("peak_allocated_increment_bytes"))
        == int(payload.get("peak_allocated_bytes"))
        - int(payload.get("memory_baseline_allocated_bytes")),
        "allocated peak increment mismatch",
    )
    _require(
        int(payload.get("peak_reserved_increment_bytes"))
        == int(payload.get("peak_reserved_bytes"))
        - int(payload.get("memory_baseline_reserved_bytes")),
        "reserved peak increment mismatch",
    )
    _require(payload.get("reporting_contract") == REPORTING_ONLY_CONTRACT, "resource metrics are not reporting-only")
    return dict(payload)


def measure_cuda_inference_callable(
    run_once: Callable[[], Any],
    *,
    cells_per_iteration: int,
    endpoint_predictions_per_iteration: int,
    warmup_iterations: int = 3,
    timed_iterations: int = 10,
    torch_module: Any,
    device: Any = None,
) -> dict[str, Any]:
    """Benchmark an already-materialised, model-compute-only CUDA workload."""

    cells = _nonnegative_int(cells_per_iteration, "cells_per_iteration")
    endpoints = _nonnegative_int(
        endpoint_predictions_per_iteration, "endpoint_predictions_per_iteration"
    )
    warmup = _nonnegative_int(warmup_iterations, "warmup_iterations")
    iterations = _nonnegative_int(timed_iterations, "timed_iterations")
    _require(cells > 0 and endpoints > 0 and iterations > 0, "inference workload must be non-empty")
    for _ in range(warmup):
        run_once()
    torch_module.cuda.synchronize(device)
    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        run_once()
    end.record()
    end.synchronize()
    seconds = _positive_finite(float(start.elapsed_time(end)) / 1000.0, "inference_cuda_seconds")
    payload = {
        "schema_version": THROUGHPUT_SCHEMA,
        "timing_backend": "cuda_event",
        "scope": "model_compute_only_preloaded_cuda_input",
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "cells_per_iteration": cells,
        "endpoint_predictions_per_iteration": endpoints,
        "timed_cells": cells * iterations,
        "timed_endpoint_predictions": endpoints * iterations,
        "inference_cuda_seconds": seconds,
        "cells_per_second": cells * iterations / seconds,
        "endpoint_predictions_per_second": endpoints * iterations / seconds,
        "cell_count_definition": "unique_phase_cell_observations_serviced_per_workload",
        "endpoint_prediction_count_definition": "scalar_endpoint_predictions_emitted",
        "query_cell_deduplication_policy": "count_one_shared_phase_cell_once_even_when_it_services_multiple_endpoint_queries",
        "host_to_device_transfer_included": False,
        "preprocessing_included": False,
        "reporting_contract": dict(REPORTING_ONLY_CONTRACT),
    }
    validate_throughput_metrics(payload)
    return payload


def validate_throughput_metrics(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == THROUGHPUT_SCHEMA, "throughput schema changed")
    iterations = _nonnegative_int(payload.get("timed_iterations"), "timed_iterations")
    cells = _nonnegative_int(payload.get("cells_per_iteration"), "cells_per_iteration")
    endpoints = _nonnegative_int(payload.get("endpoint_predictions_per_iteration"), "endpoint_predictions_per_iteration")
    _require(iterations > 0 and cells > 0 and endpoints > 0, "throughput workload is empty")
    seconds = _positive_finite(payload.get("inference_cuda_seconds"), "inference_cuda_seconds")
    _require(payload.get("timed_cells") == cells * iterations, "timed_cells mismatch")
    _require(payload.get("timed_endpoint_predictions") == endpoints * iterations, "timed_endpoint_predictions mismatch")
    _require(math.isclose(float(payload.get("cells_per_second")), cells * iterations / seconds, rel_tol=1e-12), "cells/s mismatch")
    _require(math.isclose(float(payload.get("endpoint_predictions_per_second")), endpoints * iterations / seconds, rel_tol=1e-12), "endpoint predictions/s mismatch")
    _require(
        payload.get("cell_count_definition")
        == "unique_phase_cell_observations_serviced_per_workload",
        "cell throughput definition changed",
    )
    _require(
        payload.get("endpoint_prediction_count_definition")
        == "scalar_endpoint_predictions_emitted",
        "endpoint throughput definition changed",
    )
    _require(
        payload.get("query_cell_deduplication_policy")
        == "count_one_shared_phase_cell_once_even_when_it_services_multiple_endpoint_queries",
        "query-model cell counting policy changed",
    )
    _require(
        payload.get("host_to_device_transfer_included") is False
        and payload.get("preprocessing_included") is False,
        "model-compute-only throughput included transfer or preprocessing",
    )
    _require(payload.get("reporting_contract") == REPORTING_ONLY_CONTRACT, "throughput metrics are not reporting-only")
    return dict(payload)


def summarize_inference_throughput(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pool sequential CUDA workloads by their measured time and work counts."""

    validated = [validate_throughput_metrics(row) for row in rows]
    _require(validated, "inference throughput rows are empty")
    seconds = math.fsum(float(row["inference_cuda_seconds"]) for row in validated)
    cells = sum(int(row["timed_cells"]) for row in validated)
    endpoints = sum(int(row["timed_endpoint_predictions"]) for row in validated)
    _require(seconds > 0.0 and cells > 0 and endpoints > 0, "pooled inference workload is empty")
    return {
        "n_workloads": len(validated),
        "pooled_inference_cuda_seconds": seconds,
        "pooled_timed_cells": cells,
        "pooled_timed_endpoint_predictions": endpoints,
        "pooled_cells_per_second": cells / seconds,
        "pooled_endpoint_predictions_per_second": endpoints / seconds,
        "aggregation": "sum_work_divided_by_sum_sequential_cuda_event_time",
        "reporting_contract": dict(REPORTING_ONLY_CONTRACT),
    }


def summarize_active_batches(
    rows: Iterable[Mapping[str, Any]],
    occurrence_by_signature: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    validated = [validate_active_parameter_snapshot(row) for row in rows]
    _require(validated, "active batch ledger is empty")
    signatures = Counter(str(row["active_signature_sha256"]) for row in validated)
    row_by_signature: dict[str, dict[str, Any]] = {}
    for row in validated:
        signature = str(row["active_signature_sha256"])
        existing = row_by_signature.get(signature)
        if existing is None:
            row_by_signature[signature] = row
        else:
            _require(
                int(existing["active_parameter_count"])
                == int(row["active_parameter_count"]),
                "duplicate active signature has variable active count; use CompactActiveSignatureLedger",
            )
    if occurrence_by_signature is not None:
        supplied = {str(key): int(value) for key, value in occurrence_by_signature.items()}
        _require(set(supplied) == set(signatures), "active signature occurrence keys differ")
        _require(all(value > 0 for value in supplied.values()), "active occurrences must be positive")
        signatures = Counter(supplied)
    count_by_signature = {
        signature: int(row["active_parameter_count"])
        for signature, row in row_by_signature.items()
    }
    total_batches = int(sum(signatures.values()))
    weighted_total = sum(
        count_by_signature[signature] * occurrences
        for signature, occurrences in signatures.items()
    )
    counts = list(count_by_signature.values())
    return {
        "definition": validated[0]["definition"],
        "n_observed_batches": total_batches,
        "n_active_signatures": len(signatures),
        "active_parameter_count_min": min(counts),
        "active_parameter_count_max": max(counts),
        "active_parameter_count_mean": weighted_total / total_batches,
        "batch_count_by_active_signature": dict(sorted(signatures.items())),
        "reporting_contract": dict(REPORTING_ONLY_CONTRACT),
    }


def build_runtime_metrics_manifest(
    *,
    parameter_inventory: Mapping[str, Any],
    active_batch_rows: Sequence[Mapping[str, Any]] = (),
    active_signature_occurrences: Mapping[str, int] | None = None,
    training_resources: Mapping[str, Any],
    inference_throughput: Sequence[Mapping[str, Any]],
    active_signature_ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inventory = validate_parameter_inventory(parameter_inventory)
    resource = validate_resource_metrics(training_resources)
    throughput = [validate_throughput_metrics(row) for row in inference_throughput]
    _require(throughput, "at least one inference throughput record is required")
    if active_signature_ledger is None:
        active_summary = summarize_active_batches(
            active_batch_rows, active_signature_occurrences
        )
        compact_ledger = None
    else:
        _require(
            not active_batch_rows and active_signature_occurrences is None,
            "provide either compact active ledger or legacy active rows, not both",
        )
        compact_ledger = validate_active_signature_ledger(active_signature_ledger)
        active_summary = summarize_active_signature_ledger(compact_ledger)
    core = {
        "schema_version": RUNTIME_METRICS_SCHEMA,
        "parameter_inventory": inventory,
        "active_batch_summary": active_summary,
        "active_signature_ledger": compact_ledger,
        "training_resources": resource,
        "inference_throughput": throughput,
        "inference_throughput_summary": summarize_inference_throughput(throughput),
        "selection_inputs": [],
        "reporting_contract": dict(REPORTING_ONLY_CONTRACT),
    }
    core["manifest_sha256"] = _sha256_json(core)
    validate_runtime_metrics_manifest(core)
    return core


def validate_runtime_metrics_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(payload.get("schema_version") == RUNTIME_METRICS_SCHEMA, "runtime metrics schema changed")
    _require(payload.get("selection_inputs") == [], "runtime metrics cannot be selection inputs")
    _require(payload.get("reporting_contract") == REPORTING_ONLY_CONTRACT, "runtime manifest is not reporting-only")
    unhashed = dict(payload)
    declared = unhashed.pop("manifest_sha256", None)
    _require(declared == _sha256_json(unhashed), "runtime metrics manifest hash mismatch")
    validate_parameter_inventory(payload.get("parameter_inventory"))
    validate_resource_metrics(payload.get("training_resources"))
    _require(isinstance(payload.get("inference_throughput"), list) and payload["inference_throughput"], "inference throughput is missing")
    for row in payload["inference_throughput"]:
        validate_throughput_metrics(row)
    _require(
        payload.get("inference_throughput_summary")
        == summarize_inference_throughput(payload["inference_throughput"]),
        "inference throughput summary mismatch",
    )
    summary = payload.get("active_batch_summary")
    _require(isinstance(summary, Mapping) and int(summary.get("n_observed_batches", 0)) > 0, "active batch summary is missing")
    _require(summary.get("reporting_contract") == REPORTING_ONLY_CONTRACT, "active summary is not reporting-only")
    compact_ledger = payload.get("active_signature_ledger")
    if compact_ledger is not None:
        validate_active_signature_ledger(compact_ledger)
        _require(
            summary == summarize_active_signature_ledger(compact_ledger),
            "active summary differs from compact ledger",
        )
    return dict(payload)


__all__ = [
    "ACTIVE_BATCH_SCHEMA",
    "ACTIVE_LEDGER_SCHEMA",
    "CompactActiveSignatureLedger",
    "CudaTrainingResourceMeter",
    "PARAMETER_INVENTORY_SCHEMA",
    "ParameterInventory",
    "REPORTING_ONLY_CONTRACT",
    "RUNTIME_METRICS_SCHEMA",
    "THROUGHPUT_SCHEMA",
    "active_parameter_snapshot_after_backward",
    "build_runtime_metrics_manifest",
    "measure_cuda_inference_callable",
    "summarize_active_batches",
    "summarize_active_signature_ledger",
    "summarize_inference_throughput",
    "validate_active_parameter_snapshot",
    "validate_active_signature_ledger",
    "validate_parameter_inventory",
    "validate_resource_metrics",
    "validate_runtime_metrics_manifest",
    "validate_throughput_metrics",
]
