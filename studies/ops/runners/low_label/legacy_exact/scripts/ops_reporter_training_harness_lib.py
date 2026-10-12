#!/usr/bin/env python3
"""Architecture-independent protocol harness for OPS reporter training.

The module contains no model or optimizer implementation.  It freezes data
cohorts, optimizer-update budgets, reporter exposure, validation selection,
development search budgets, and post-selection evaluation-state provenance.
All state exchanged with a trainer is JSON-serializable and fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


HARNESS_SCHEMA_VERSION = "ops-reporter-training-harness-v1"
SPLIT_SCHEMA_VERSION = "ops-reporter-frozen-split-contract-v1"
BUDGET_SCHEMA_VERSION = "ops-reporter-optimizer-update-budget-v1"
EXPOSURE_SCHEMA_VERSION = "ops-reporter-exposure-ledger-v1"
OBJECTIVE_SCHEMA_VERSION = "ops-reporter-normalized-checkpoint-objective-v1"
TRAJECTORY_SCHEMA_VERSION = "ops-reporter-validation-trajectory-v1"
SEARCH_SCHEMA_VERSION = "ops-reporter-method-search-budget-v1"
EVALUATION_SCHEMA_VERSION = "ops-reporter-read-only-evaluation-states-v1"

PARTITIONS = ("train", "validation", "test")
EVALUATION_STATE_KINDS = ("single", "ema", "checkpoint_average")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class HarnessValidationError(ValueError):
    """Raised when a frozen protocol or runtime record is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessValidationError(message)


def _require_nonempty_string(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{label} must be non-empty")
    return value.strip()


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{label} must be an integer",
    )
    _require(value >= minimum, f"{label} must be >= {minimum}")
    return int(value)


def _require_float(value: Any, label: str, *, minimum: float = 0.0) -> float:
    _require(not isinstance(value, bool), f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise HarnessValidationError(f"{label} must be numeric") from error
    _require(math.isfinite(result), f"{label} must be finite")
    _require(result >= minimum, f"{label} must be >= {minimum}")
    return result


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
        f"{label} must be a lowercase SHA256 digest",
    )
    return value


def _normalize_reporters(values: Iterable[Any], label: str = "reporters") -> tuple[str, ...]:
    reporters = tuple(_require_nonempty_string(value, label) for value in values)
    _require(reporters, f"{label} must be non-empty")
    _require(len(set(reporters)) == len(reporters), f"{label} must be unique")
    return reporters


def canonical_json(value: Any) -> str:
    """Return deterministic JSON and reject NaN, non-string keys, or custom objects."""

    def check_keys(item: Any, path: str = "root") -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                _require(isinstance(key, str), f"JSON key at {path} must be a string")
                check_keys(child, f"{path}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                check_keys(child, f"{path}[{index}]")

    check_keys(value)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise HarnessValidationError("Value is not strict JSON-serializable") from error
    return encoded


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(canonical_json(value))


@dataclass(frozen=True)
class StatisticalUnitContract:
    unit_name: str
    unit_id_field: str
    metric_unit: str
    resampling_unit: str
    repeated_unit_policy: str

    def __post_init__(self) -> None:
        for field_name in ("unit_name", "unit_id_field", "metric_unit", "resampling_unit"):
            _require_nonempty_string(getattr(self, field_name), field_name)
        _require(
            self.repeated_unit_policy
            in {
                "one_observation_per_unit",
                "aggregate_within_unit",
                "cluster_by_unit",
                "explicit_observation_level",
            },
            "Unsupported repeated_unit_policy",
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "unit_name": self.unit_name,
            "unit_id_field": self.unit_id_field,
            "metric_unit": self.metric_unit,
            "resampling_unit": self.resampling_unit,
            "repeated_unit_policy": self.repeated_unit_policy,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "StatisticalUnitContract":
        _require(isinstance(payload, Mapping), "statistical_unit must be a mapping")
        return cls(
            unit_name=payload.get("unit_name"),
            unit_id_field=payload.get("unit_id_field"),
            metric_unit=payload.get("metric_unit"),
            resampling_unit=payload.get("resampling_unit"),
            repeated_unit_policy=payload.get("repeated_unit_policy"),
        )


@dataclass(frozen=True)
class CohortFingerprint:
    n_statistical_units: int
    n_observations: int
    unit_ids_sha256: str
    observation_ids_sha256: str

    def __post_init__(self) -> None:
        _require_int(self.n_statistical_units, "n_statistical_units", minimum=1)
        _require_int(self.n_observations, "n_observations", minimum=1)
        _require(
            self.n_observations >= self.n_statistical_units,
            "n_observations cannot be smaller than n_statistical_units",
        )
        _require_sha256(self.unit_ids_sha256, "unit_ids_sha256")
        _require_sha256(self.observation_ids_sha256, "observation_ids_sha256")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "n_statistical_units": self.n_statistical_units,
            "n_observations": self.n_observations,
            "unit_ids_sha256": self.unit_ids_sha256,
            "observation_ids_sha256": self.observation_ids_sha256,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "CohortFingerprint":
        _require(isinstance(payload, Mapping), "cohort fingerprint must be a mapping")
        return cls(
            n_statistical_units=payload.get("n_statistical_units"),
            n_observations=payload.get("n_observations"),
            unit_ids_sha256=payload.get("unit_ids_sha256"),
            observation_ids_sha256=payload.get("observation_ids_sha256"),
        )


@dataclass(frozen=True)
class ReporterCohortContract:
    reporter_slug: str
    train: CohortFingerprint
    validation: CohortFingerprint
    test: CohortFingerprint

    def __post_init__(self) -> None:
        _require_nonempty_string(self.reporter_slug, "reporter_slug")
        for partition in PARTITIONS:
            _require(
                isinstance(getattr(self, partition), CohortFingerprint),
                f"{partition} must be a CohortFingerprint",
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "reporter_slug": self.reporter_slug,
            "partitions": {
                partition: getattr(self, partition).to_manifest()
                for partition in PARTITIONS
            },
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "ReporterCohortContract":
        _require(isinstance(payload, Mapping), "reporter cohort must be a mapping")
        partitions = payload.get("partitions")
        _require(isinstance(partitions, Mapping), "partitions must be a mapping")
        _require(set(partitions) == set(PARTITIONS), "partitions must be train/validation/test")
        return cls(
            reporter_slug=payload.get("reporter_slug"),
            train=CohortFingerprint.from_manifest(partitions["train"]),
            validation=CohortFingerprint.from_manifest(partitions["validation"]),
            test=CohortFingerprint.from_manifest(partitions["test"]),
        )


@dataclass(frozen=True)
class FrozenSplitContract:
    split_name: str
    fold: int
    split_manifest_sha256: str
    statistical_unit: StatisticalUnitContract
    reporter_cohorts: tuple[ReporterCohortContract, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.split_name, "split_name")
        _require_int(self.fold, "fold", minimum=0)
        _require_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        _require(
            isinstance(self.statistical_unit, StatisticalUnitContract),
            "statistical_unit must be a StatisticalUnitContract",
        )
        _require(self.reporter_cohorts, "reporter_cohorts must be non-empty")
        slugs = tuple(cohort.reporter_slug for cohort in self.reporter_cohorts)
        _normalize_reporters(slugs, "reporter cohort slugs")

    @property
    def reporters(self) -> tuple[str, ...]:
        return tuple(cohort.reporter_slug for cohort in self.reporter_cohorts)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "split_name": self.split_name,
            "fold": self.fold,
            "split_manifest_sha256": self.split_manifest_sha256,
            "statistical_unit": self.statistical_unit.to_manifest(),
            "selection_partition": "validation",
            "test_partition": "test",
            "test_cohort_frozen": True,
            "test_labels_available_during_training": False,
            "test_used_for_training_or_selection": False,
            "test_access": "final_evaluation_only",
            "reporters": list(self.reporters),
            "reporter_cohorts": [item.to_manifest() for item in self.reporter_cohorts],
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "FrozenSplitContract":
        _require(isinstance(payload, Mapping), "split contract must be a mapping")
        _require(payload.get("schema_version") == SPLIT_SCHEMA_VERSION, "wrong split schema")
        boundary = {
            "selection_partition": "validation",
            "test_partition": "test",
            "test_cohort_frozen": True,
            "test_labels_available_during_training": False,
            "test_used_for_training_or_selection": False,
            "test_access": "final_evaluation_only",
        }
        for key, expected in boundary.items():
            _require(payload.get(key) == expected, f"split contract violates {key}")
        raw_cohorts = payload.get("reporter_cohorts")
        _require(isinstance(raw_cohorts, Sequence), "reporter_cohorts must be a sequence")
        result = cls(
            split_name=payload.get("split_name"),
            fold=payload.get("fold"),
            split_manifest_sha256=payload.get("split_manifest_sha256"),
            statistical_unit=StatisticalUnitContract.from_manifest(
                payload.get("statistical_unit")
            ),
            reporter_cohorts=tuple(
                ReporterCohortContract.from_manifest(item) for item in raw_cohorts
            ),
        )
        _require(payload.get("reporters") == list(result.reporters), "reporter order mismatch")
        return result




def _normalize_exposure_map(
    values: Mapping[str, Any], reporters: tuple[str, ...], label: str
) -> tuple[tuple[str, int], ...]:
    _require(isinstance(values, Mapping), f"{label} must be a mapping")
    _require(set(values) == set(reporters), f"{label} reporter set differs")
    normalized = []
    for reporter in reporters:
        normalized.append(
            (reporter, _require_int(values[reporter], f"{label}.{reporter}", minimum=1))
        )
    return tuple(normalized)


@dataclass(frozen=True)
class OptimizerUpdateBudget:
    reporters: tuple[str, ...]
    total_optimizer_updates: int
    expected_example_exposures: tuple[tuple[str, int], ...]
    expected_update_exposures: tuple[tuple[str, int], ...]
    example_exposure_unit: str = "labeled_target_vectors"

    def __post_init__(self) -> None:
        normalized = _normalize_reporters(self.reporters)
        _require(normalized == self.reporters, "reporter order must be stable")
        _require_int(self.total_optimizer_updates, "total_optimizer_updates", minimum=1)
        _require_nonempty_string(self.example_exposure_unit, "example_exposure_unit")
        for label, pairs in (
            ("expected_example_exposures", self.expected_example_exposures),
            ("expected_update_exposures", self.expected_update_exposures),
        ):
            mapping = dict(pairs)
            _require(len(mapping) == len(pairs), f"{label} contains duplicates")
            normalized_pairs = _normalize_exposure_map(mapping, self.reporters, label)
            _require(normalized_pairs == pairs, f"{label} must follow reporter order")
        for reporter, value in self.expected_update_exposures:
            _require(
                value <= self.total_optimizer_updates,
                f"update exposure for {reporter} exceeds optimizer budget",
            )

    @classmethod
    def create(
        cls,
        reporters: Iterable[str],
        total_optimizer_updates: int,
        expected_example_exposures: Mapping[str, int],
        expected_update_exposures: Mapping[str, int],
        *,
        example_exposure_unit: str = "labeled_target_vectors",
    ) -> "OptimizerUpdateBudget":
        ordered = _normalize_reporters(reporters)
        return cls(
            reporters=ordered,
            total_optimizer_updates=total_optimizer_updates,
            expected_example_exposures=_normalize_exposure_map(
                expected_example_exposures, ordered, "expected_example_exposures"
            ),
            expected_update_exposures=_normalize_exposure_map(
                expected_update_exposures, ordered, "expected_update_exposures"
            ),
            example_exposure_unit=example_exposure_unit,
        )

    @property
    def expected_examples(self) -> dict[str, int]:
        return dict(self.expected_example_exposures)

    @property
    def expected_updates(self) -> dict[str, int]:
        return dict(self.expected_update_exposures)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": BUDGET_SCHEMA_VERSION,
            "budget_unit": "optimizer_update",
            "total_optimizer_updates": self.total_optimizer_updates,
            "reporters": list(self.reporters),
            "example_exposure_unit": self.example_exposure_unit,
            "expected_example_exposures": self.expected_examples,
            "expected_update_exposures": self.expected_updates,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "OptimizerUpdateBudget":
        _require(isinstance(payload, Mapping), "budget contract must be a mapping")
        _require(payload.get("schema_version") == BUDGET_SCHEMA_VERSION, "wrong budget schema")
        _require(payload.get("budget_unit") == "optimizer_update", "budget is not update-based")
        return cls.create(
            reporters=payload.get("reporters", []),
            total_optimizer_updates=payload.get("total_optimizer_updates"),
            expected_example_exposures=payload.get("expected_example_exposures"),
            expected_update_exposures=payload.get("expected_update_exposures"),
            example_exposure_unit=payload.get("example_exposure_unit"),
        )


class ExposureLedger:
    """Mutable runtime ledger governed by an immutable update budget."""

    def __init__(self, budget: OptimizerUpdateBudget) -> None:
        _require(isinstance(budget, OptimizerUpdateBudget), "budget has wrong type")
        self.budget = budget
        self.completed_optimizer_updates = 0
        self.example_exposures = {reporter: 0 for reporter in budget.reporters}
        self.update_exposures = {reporter: 0 for reporter in budget.reporters}
        self._finalized = False

    def record_update(self, example_exposures: Mapping[str, int]) -> None:
        _require(not self._finalized, "exposure ledger is finalized")
        _require(
            self.completed_optimizer_updates < self.budget.total_optimizer_updates,
            "optimizer update budget exceeded",
        )
        _require(isinstance(example_exposures, Mapping), "example_exposures must be a mapping")
        unknown = set(example_exposures) - set(self.budget.reporters)
        _require(not unknown, f"unknown reporters in exposure update: {sorted(unknown)}")
        normalized = {
            reporter: _require_int(value, f"exposure.{reporter}", minimum=0)
            for reporter, value in example_exposures.items()
        }
        _require(any(value > 0 for value in normalized.values()), "an update must expose a reporter")
        projected_examples = dict(self.example_exposures)
        projected_updates = dict(self.update_exposures)
        for reporter in self.budget.reporters:
            value = normalized.get(reporter, 0)
            projected_examples[reporter] += value
            if value > 0:
                projected_updates[reporter] += 1
            _require(
                projected_examples[reporter]
                <= self.budget.expected_examples[reporter],
                f"example exposure budget exceeded for {reporter}",
            )
            _require(
                projected_updates[reporter]
                <= self.budget.expected_updates[reporter],
                f"update exposure budget exceeded for {reporter}",
            )
        self.example_exposures = projected_examples
        self.update_exposures = projected_updates
        self.completed_optimizer_updates += 1

    def finalize(self) -> None:
        _require(
            self.completed_optimizer_updates == self.budget.total_optimizer_updates,
            "optimizer update trajectory is incomplete",
        )
        _require(
            self.example_exposures == self.budget.expected_examples,
            "final example exposures differ from frozen budget",
        )
        _require(
            self.update_exposures == self.budget.expected_updates,
            "final update exposures differ from frozen budget",
        )
        self._finalized = True

    @property
    def finalized(self) -> bool:
        return self._finalized

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": EXPOSURE_SCHEMA_VERSION,
            "budget_sha256": json_sha256(self.budget.to_manifest()),
            "completed_optimizer_updates": self.completed_optimizer_updates,
            "example_exposures": dict(self.example_exposures),
            "update_exposures": dict(self.update_exposures),
            "finalized": self.finalized,
        }

    @staticmethod
    def validate_manifest(
        payload: Mapping[str, Any], budget: OptimizerUpdateBudget
    ) -> None:
        _require(isinstance(payload, Mapping), "exposure ledger must be a mapping")
        _require(payload.get("schema_version") == EXPOSURE_SCHEMA_VERSION, "wrong exposure schema")
        _require(
            payload.get("budget_sha256") == json_sha256(budget.to_manifest()),
            "exposure ledger budget hash mismatch",
        )
        _require(payload.get("finalized") is True, "exposure ledger is not finalized")
        _require(
            payload.get("completed_optimizer_updates") == budget.total_optimizer_updates,
            "completed update count differs from budget",
        )
        _require(payload.get("example_exposures") == budget.expected_examples, "example exposure mismatch")
        _require(payload.get("update_exposures") == budget.expected_updates, "update exposure mismatch")


def _normalize_loss_map(
    values: Mapping[str, Any], reporters: tuple[str, ...], label: str
) -> tuple[tuple[str, float], ...]:
    _require(isinstance(values, Mapping), f"{label} must be a mapping")
    _require(set(values) == set(reporters), f"{label} reporter set differs")
    return tuple(
        (reporter, _require_float(values[reporter], f"{label}.{reporter}"))
        for reporter in reporters
    )


@dataclass(frozen=True)
class ReporterNormalizedCheckpointObjective:
    reporters: tuple[str, ...]
    null_validation_mse: tuple[tuple[str, float], ...]
    denominator_floor: float = 1.0e-3
    tie_relative_tolerance: float = 2.0e-3

    def __post_init__(self) -> None:
        normalized = _normalize_reporters(self.reporters)
        _require(normalized == self.reporters, "objective reporter order is unstable")
        mapping = dict(self.null_validation_mse)
        _require(len(mapping) == len(self.null_validation_mse), "duplicate null losses")
        normalized_losses = _normalize_loss_map(mapping, self.reporters, "null_validation_mse")
        _require(normalized_losses == self.null_validation_mse, "null losses must follow reporter order")
        _require_float(self.denominator_floor, "denominator_floor", minimum=0.0)
        _require(self.denominator_floor > 0.0, "denominator_floor must be positive")
        _require_float(self.tie_relative_tolerance, "tie_relative_tolerance", minimum=0.0)

    @classmethod
    def create(
        cls,
        reporters: Iterable[str],
        null_validation_mse: Mapping[str, float],
        *,
        denominator_floor: float = 1.0e-3,
        tie_relative_tolerance: float = 2.0e-3,
    ) -> "ReporterNormalizedCheckpointObjective":
        ordered = _normalize_reporters(reporters)
        return cls(
            reporters=ordered,
            null_validation_mse=_normalize_loss_map(
                null_validation_mse, ordered, "null_validation_mse"
            ),
            denominator_floor=denominator_floor,
            tie_relative_tolerance=tie_relative_tolerance,
        )

    @property
    def nulls(self) -> dict[str, float]:
        return dict(self.null_validation_mse)

    def score(self, validation_mse: Mapping[str, float]) -> float:
        observed = dict(
            _normalize_loss_map(validation_mse, self.reporters, "validation_mse")
        )
        ratios = [
            observed[reporter]
            / max(self.nulls[reporter], self.denominator_floor)
            for reporter in self.reporters
        ]
        score = math.fsum(ratios) / len(ratios)
        _require(math.isfinite(score), "checkpoint score is non-finite")
        return score

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": OBJECTIVE_SCHEMA_VERSION,
            "name": "reporter_equal_null_normalized_validation_mse",
            "direction": "minimize",
            "reporter_weighting": "equal",
            "endpoint_weighting_within_reporter": "equal",
            "denominator": f"max(null_validation_mse, {self.denominator_floor:g})",
            "denominator_source": "train_mean_predictor_evaluated_on_validation",
            "denominator_floor": self.denominator_floor,
            "tie_relative_tolerance": self.tie_relative_tolerance,
            "selection_input_provenance": ["complete_validation_trajectory"],
            "reporters": list(self.reporters),
            "null_validation_mse": self.nulls,
            "null_validation_mse_sha256": json_sha256(self.nulls),
        }

    @classmethod
    def from_manifest(
        cls, payload: Mapping[str, Any]
    ) -> "ReporterNormalizedCheckpointObjective":
        _require(isinstance(payload, Mapping), "checkpoint objective must be a mapping")
        expected = {
            "schema_version": OBJECTIVE_SCHEMA_VERSION,
            "name": "reporter_equal_null_normalized_validation_mse",
            "direction": "minimize",
            "reporter_weighting": "equal",
            "endpoint_weighting_within_reporter": "equal",
            "denominator_source": "train_mean_predictor_evaluated_on_validation",
            "selection_input_provenance": ["complete_validation_trajectory"],
        }
        for key, value in expected.items():
            _require(payload.get(key) == value, f"checkpoint objective violates {key}")
        objective = cls.create(
            reporters=payload.get("reporters", []),
            null_validation_mse=payload.get("null_validation_mse"),
            denominator_floor=payload.get("denominator_floor"),
            tie_relative_tolerance=payload.get("tie_relative_tolerance"),
        )
        _require(
            payload.get("denominator")
            == f"max(null_validation_mse, {objective.denominator_floor:g})",
            "checkpoint denominator expression differs",
        )
        _require(
            payload.get("null_validation_mse_sha256") == json_sha256(objective.nulls),
            "null validation MSE hash mismatch",
        )
        return objective


@dataclass(frozen=True)
class ValidationSchedule:
    total_optimizer_updates: int
    validation_updates: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_int(self.total_optimizer_updates, "total_optimizer_updates", minimum=1)
        _require(self.validation_updates, "validation_updates must be non-empty")
        checked = tuple(
            _require_int(value, "validation update", minimum=1)
            for value in self.validation_updates
        )
        _require(checked == tuple(sorted(set(checked))), "validation updates must be unique and increasing")
        _require(checked[-1] == self.total_optimizer_updates, "final optimizer update must be validated")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "budget_unit": "optimizer_update",
            "total_optimizer_updates": self.total_optimizer_updates,
            "required_validation_updates": list(self.validation_updates),
            "complete_trajectory_required": True,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "ValidationSchedule":
        _require(isinstance(payload, Mapping), "validation schedule must be a mapping")
        _require(payload.get("budget_unit") == "optimizer_update", "validation schedule is not update-based")
        _require(payload.get("complete_trajectory_required") is True, "trajectory completeness is not required")
        return cls(
            total_optimizer_updates=payload.get("total_optimizer_updates"),
            validation_updates=tuple(payload.get("required_validation_updates", [])),
        )


@dataclass(frozen=True)
class CheckpointSelection:
    optimizer_update: int
    checkpoint_sha256: str
    score: float
    raw_macro_mse: float
    strict_minimum_score: float
    eligibility_threshold: float
    trajectory_sha256: str

    def __post_init__(self) -> None:
        _require_int(self.optimizer_update, "selection optimizer_update", minimum=1)
        _require_sha256(self.checkpoint_sha256, "selection checkpoint_sha256")
        for name in (
            "score",
            "raw_macro_mse",
            "strict_minimum_score",
            "eligibility_threshold",
        ):
            _require_float(getattr(self, name), name)
        _require_sha256(self.trajectory_sha256, "trajectory_sha256")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "optimizer_update": self.optimizer_update,
            "checkpoint_sha256": self.checkpoint_sha256,
            "score": self.score,
            "raw_macro_mse": self.raw_macro_mse,
            "strict_minimum_score": self.strict_minimum_score,
            "eligibility_threshold": self.eligibility_threshold,
            "trajectory_sha256": self.trajectory_sha256,
            "selection_partition": "validation",
            "test_used_for_selection": False,
        }


class ValidationTrajectory:
    def __init__(
        self,
        objective: ReporterNormalizedCheckpointObjective,
        schedule: ValidationSchedule,
    ) -> None:
        self.objective = objective
        self.schedule = schedule
        self._records: list[dict[str, Any]] = []
        self._selection: CheckpointSelection | None = None

    @property
    def records(self) -> list[dict[str, Any]]:
        return _json_copy(self._records)

    @property
    def complete(self) -> bool:
        return [row["optimizer_update"] for row in self._records] == list(
            self.schedule.validation_updates
        )

    @property
    def selection(self) -> CheckpointSelection | None:
        return self._selection

    def append(
        self,
        optimizer_update: int,
        validation_mse: Mapping[str, float],
        checkpoint_sha256: str,
    ) -> None:
        _require(self._selection is None, "trajectory is frozen after checkpoint selection")
        index = len(self._records)
        _require(index < len(self.schedule.validation_updates), "too many validation records")
        expected_update = self.schedule.validation_updates[index]
        _require(optimizer_update == expected_update, f"expected validation at update {expected_update}")
        checkpoint_sha = _require_sha256(checkpoint_sha256, "checkpoint_sha256")
        losses = dict(
            _normalize_loss_map(
                validation_mse, self.objective.reporters, "validation_mse"
            )
        )
        raw_macro = math.fsum(losses.values()) / len(losses)
        self._records.append(
            {
                "optimizer_update": optimizer_update,
                "validation_mse_by_reporter": losses,
                "raw_reporter_macro_mse": raw_macro,
                "reporter_normalized_score": self.objective.score(losses),
                "checkpoint_sha256": checkpoint_sha,
            }
        )

    def select_checkpoint(self) -> CheckpointSelection:
        _require(self.complete, "validation trajectory is incomplete")
        if self._selection is not None:
            return self._selection
        minimum = min(row["reporter_normalized_score"] for row in self._records)
        threshold = minimum * (1.0 + self.objective.tie_relative_tolerance)
        eligible = [
            row
            for row in self._records
            if row["reporter_normalized_score"] == minimum
            or row["reporter_normalized_score"] < threshold
        ]
        selected = min(eligible, key=lambda row: row["optimizer_update"])
        trajectory_hash = json_sha256(self._records)
        self._selection = CheckpointSelection(
            optimizer_update=selected["optimizer_update"],
            checkpoint_sha256=selected["checkpoint_sha256"],
            score=selected["reporter_normalized_score"],
            raw_macro_mse=selected["raw_reporter_macro_mse"],
            strict_minimum_score=minimum,
            eligibility_threshold=threshold,
            trajectory_sha256=trajectory_hash,
        )
        return self._selection

    def to_manifest(self) -> dict[str, Any]:
        selection = self._selection.to_manifest() if self._selection is not None else None
        return {
            "schema_version": TRAJECTORY_SCHEMA_VERSION,
            "objective": self.objective.to_manifest(),
            "schedule": self.schedule.to_manifest(),
            "records": self.records,
            "records_sha256": json_sha256(self._records),
            "complete": self.complete,
            "selection": selection,
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "ValidationTrajectory":
        _require(isinstance(payload, Mapping), "validation trajectory must be a mapping")
        _require(payload.get("schema_version") == TRAJECTORY_SCHEMA_VERSION, "wrong trajectory schema")
        objective = ReporterNormalizedCheckpointObjective.from_manifest(payload.get("objective"))
        schedule = ValidationSchedule.from_manifest(payload.get("schedule"))
        trajectory = cls(objective, schedule)
        records = payload.get("records")
        _require(isinstance(records, Sequence), "trajectory records must be a sequence")
        for row in records:
            _require(isinstance(row, Mapping), "trajectory row must be a mapping")
            trajectory.append(
                optimizer_update=row.get("optimizer_update"),
                validation_mse=row.get("validation_mse_by_reporter"),
                checkpoint_sha256=row.get("checkpoint_sha256"),
            )
            rebuilt = trajectory._records[-1]
            _require(
                math.isclose(
                    rebuilt["raw_reporter_macro_mse"],
                    _require_float(row.get("raw_reporter_macro_mse"), "raw macro MSE"),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "raw validation macro was not reproducible",
            )
            _require(
                math.isclose(
                    rebuilt["reporter_normalized_score"],
                    _require_float(row.get("reporter_normalized_score"), "normalized score"),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "normalized validation score was not reproducible",
            )
        _require(payload.get("records_sha256") == json_sha256(trajectory._records), "trajectory hash mismatch")
        _require(payload.get("complete") is trajectory.complete, "trajectory completeness flag mismatch")
        raw_selection = payload.get("selection")
        if raw_selection is not None:
            selected = trajectory.select_checkpoint()
            _require(raw_selection == selected.to_manifest(), "checkpoint selection is not reproducible")
        return trajectory


@dataclass(frozen=True)
class DevelopmentFold:
    split_name: str
    fold: int

    def __post_init__(self) -> None:
        _require_nonempty_string(self.split_name, "development split_name")
        _require_int(self.fold, "development fold", minimum=0)

    def to_manifest(self) -> dict[str, Any]:
        return {"split_name": self.split_name, "fold": self.fold}


def _normalize_trial_configs(configs: Sequence[Mapping[str, Any]], label: str) -> tuple[dict[str, Any], ...]:
    _require(isinstance(configs, Sequence) and not isinstance(configs, (str, bytes)), f"{label} must be a sequence")
    normalized = tuple(_json_copy(config) for config in configs)
    _require(all(isinstance(config, dict) for config in normalized), f"{label} entries must be mappings")
    _require(len({canonical_json(config) for config in normalized}) == len(normalized), f"{label} contains duplicate trials")
    return normalized


@dataclass(frozen=True)
class MethodSearchSpace:
    method_id: str
    trial_configs: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.method_id, "method_id")
        normalized = _normalize_trial_configs(self.trial_configs, f"{self.method_id}.trial_configs")
        _require(normalized == self.trial_configs, "trial configs must already be normalized")

    @classmethod
    def create(
        cls, method_id: str, trial_configs: Sequence[Mapping[str, Any]]
    ) -> "MethodSearchSpace":
        return cls(
            method_id=_require_nonempty_string(method_id, "method_id"),
            trial_configs=_normalize_trial_configs(trial_configs, f"{method_id}.trial_configs"),
        )

    def to_manifest(self, seeds: tuple[int, ...]) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "trials": [
                {"trial_index": index, "seed": seeds[index], "config": config}
                for index, config in enumerate(self.trial_configs)
            ],
        }


@dataclass(frozen=True)
class SearchBudgetContract:
    trial_seeds: tuple[int, ...]
    development_folds: tuple[DevelopmentFold, ...]
    optimizer_updates_per_trial_fold: int
    method_spaces: tuple[MethodSearchSpace, ...]

    def __post_init__(self) -> None:
        _require(self.trial_seeds, "trial_seeds must be non-empty")
        checked_seeds = tuple(_require_int(seed, "trial seed", minimum=0) for seed in self.trial_seeds)
        _require(checked_seeds == self.trial_seeds, "trial seed order is unstable")
        _require(len(set(checked_seeds)) == len(checked_seeds), "trial seeds must be unique")
        _require(self.development_folds, "development_folds must be non-empty")
        fold_keys = tuple((fold.split_name, fold.fold) for fold in self.development_folds)
        _require(len(set(fold_keys)) == len(fold_keys), "development folds must be unique")
        _require_int(self.optimizer_updates_per_trial_fold, "optimizer_updates_per_trial_fold", minimum=1)
        _require(self.method_spaces, "method_spaces must be non-empty")
        method_ids = tuple(space.method_id for space in self.method_spaces)
        _require(len(set(method_ids)) == len(method_ids), "method IDs must be unique")
        for space in self.method_spaces:
            _require(
                len(space.trial_configs) == len(self.trial_seeds),
                f"{space.method_id} trial count differs from shared budget",
            )

    @property
    def trial_count_per_method(self) -> int:
        return len(self.trial_seeds)

    def method(self, method_id: str) -> MethodSearchSpace:
        matches = [space for space in self.method_spaces if space.method_id == method_id]
        _require(len(matches) == 1, f"unknown method_id: {method_id}")
        return matches[0]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SEARCH_SCHEMA_VERSION,
            "trial_count_per_method": self.trial_count_per_method,
            "trial_seeds": list(self.trial_seeds),
            "development_folds": [fold.to_manifest() for fold in self.development_folds],
            "optimizer_updates_per_trial_fold": self.optimizer_updates_per_trial_fold,
            "same_trial_seed_fold_budget_for_every_method": True,
            "method_specific_search_spaces": True,
            "test_cohort_available_to_search": False,
            "methods": [space.to_manifest(self.trial_seeds) for space in self.method_spaces],
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "SearchBudgetContract":
        _require(isinstance(payload, Mapping), "search contract must be a mapping")
        _require(payload.get("schema_version") == SEARCH_SCHEMA_VERSION, "wrong search schema")
        expected_flags = {
            "same_trial_seed_fold_budget_for_every_method": True,
            "method_specific_search_spaces": True,
            "test_cohort_available_to_search": False,
        }
        for key, value in expected_flags.items():
            _require(payload.get(key) == value, f"search contract violates {key}")
        seeds = tuple(payload.get("trial_seeds", []))
        method_spaces = []
        raw_methods = payload.get("methods")
        _require(isinstance(raw_methods, Sequence), "search methods must be a sequence")
        for method in raw_methods:
            _require(isinstance(method, Mapping), "search method must be a mapping")
            trials = method.get("trials")
            _require(isinstance(trials, Sequence), "method trials must be a sequence")
            for index, trial in enumerate(trials):
                _require(trial.get("trial_index") == index, "trial index mismatch")
                _require(trial.get("seed") == seeds[index], "trial seed mismatch")
            method_spaces.append(
                MethodSearchSpace.create(
                    method_id=method.get("method_id"),
                    trial_configs=[trial.get("config") for trial in trials],
                )
            )
        result = cls(
            trial_seeds=seeds,
            development_folds=tuple(
                DevelopmentFold(item.get("split_name"), item.get("fold"))
                for item in payload.get("development_folds", [])
            ),
            optimizer_updates_per_trial_fold=payload.get(
                "optimizer_updates_per_trial_fold"
            ),
            method_spaces=tuple(method_spaces),
        )
        _require(payload.get("trial_count_per_method") == result.trial_count_per_method, "trial count mismatch")
        return result


class EvaluationStateRegistry:
    """Provenance-only registry created after checkpoint selection is frozen."""

    def __init__(
        self,
        selection: CheckpointSelection,
        checkpoint_inventory: Mapping[int, str],
    ) -> None:
        _require(isinstance(selection, CheckpointSelection), "selection has wrong type")
        _require(isinstance(checkpoint_inventory, Mapping), "checkpoint inventory must be a mapping")
        inventory: dict[int, str] = {}
        for update, digest in checkpoint_inventory.items():
            # Update zero is the frozen initialization checkpoint.  It may be
            # used only as a source for a post-selection checkpoint average
            # when validation selects the first trained checkpoint.
            key = _require_int(update, "checkpoint inventory update", minimum=0)
            _require(key not in inventory, "duplicate checkpoint inventory update")
            inventory[key] = _require_sha256(digest, "checkpoint inventory digest")
        _require(
            inventory.get(selection.optimizer_update) == selection.checkpoint_sha256,
            "selected checkpoint is absent from inventory",
        )
        self.selection = selection
        self.checkpoint_inventory = dict(sorted(inventory.items()))
        self._selection_fingerprint = json_sha256(selection.to_manifest())
        self._states: list[dict[str, Any]] = []

    @property
    def selection_fingerprint(self) -> str:
        return self._selection_fingerprint

    @property
    def states(self) -> list[dict[str, Any]]:
        return _json_copy(self._states)

    def _add(self, state: Mapping[str, Any]) -> None:
        state_id = _require_nonempty_string(state.get("state_id"), "state_id")
        _require(state.get("kind") in EVALUATION_STATE_KINDS, "unknown evaluation-state kind")
        _require(all(item["state_id"] != state_id for item in self._states), "duplicate evaluation state_id")
        _require(state.get("read_only") is True, "evaluation state must be read-only")
        _require(state.get("used_for_training") is False, "evaluation state may affect training")
        _require(state.get("used_for_checkpoint_selection") is False, "evaluation state may affect selection")
        _require_sha256(state.get("artifact_sha256"), "evaluation artifact_sha256")
        self._states.append(_json_copy(state))
        _require(
            json_sha256(self.selection.to_manifest()) == self._selection_fingerprint,
            "evaluation-state registration mutated checkpoint selection",
        )

    def add_single(self, state_id: str = "single") -> None:
        update = self.selection.optimizer_update
        self._add(
            {
                "state_id": state_id,
                "kind": "single",
                "artifact_sha256": self.selection.checkpoint_sha256,
                "source_checkpoint_updates": [update],
                "source_checkpoint_sha256": [self.checkpoint_inventory[update]],
                "read_only": True,
                "used_for_training": False,
                "used_for_checkpoint_selection": False,
            }
        )

    def add_ema(self, state_id: str, artifact_sha256: str, decay: float) -> None:
        decay_value = _require_float(decay, "EMA decay", minimum=0.0)
        _require(0.0 < decay_value < 1.0, "EMA decay must be in (0, 1)")
        update = self.selection.optimizer_update
        self._add(
            {
                "state_id": state_id,
                "kind": "ema",
                "artifact_sha256": artifact_sha256,
                "ema_decay": decay_value,
                "ema_through_optimizer_update": update,
                "source_checkpoint_updates": [update],
                "source_checkpoint_sha256": [self.checkpoint_inventory[update]],
                "read_only": True,
                "used_for_training": False,
                "used_for_checkpoint_selection": False,
            }
        )

    def add_checkpoint_average(
        self,
        state_id: str,
        artifact_sha256: str,
        source_checkpoint_updates: Sequence[int],
        *,
        allow_post_selection_sources: bool = False,
    ) -> None:
        updates = tuple(
            _require_int(value, "checkpoint-average source update", minimum=0)
            for value in source_checkpoint_updates
        )
        _require(len(updates) >= 2, "checkpoint average requires at least two sources")
        _require(updates == tuple(sorted(set(updates))), "checkpoint-average sources must be unique and sorted")
        _require(self.selection.optimizer_update in updates, "checkpoint average must include selected checkpoint")
        includes_post_selection = max(updates) > self.selection.optimizer_update
        _require(
            not includes_post_selection or allow_post_selection_sources is True,
            "checkpoint average cannot use post-selection updates unless explicitly "
            "registered as a read-only validation diagnostic",
        )
        _require(all(update in self.checkpoint_inventory for update in updates), "checkpoint-average source is absent")
        state = {
            "state_id": state_id,
            "kind": "checkpoint_average",
            "artifact_sha256": artifact_sha256,
            "source_checkpoint_updates": list(updates),
            "source_checkpoint_sha256": [
                self.checkpoint_inventory[update] for update in updates
            ],
            "read_only": True,
            "used_for_training": False,
            "used_for_checkpoint_selection": False,
        }
        if includes_post_selection:
            state["includes_post_selection_trained_source"] = True
        self._add(state)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "selection_fingerprint_sha256": self.selection_fingerprint,
            "selected_optimizer_update": self.selection.optimizer_update,
            "selected_checkpoint_sha256": self.selection.checkpoint_sha256,
            "training_state_mutation_allowed": False,
            "checkpoint_selection_mutation_allowed": False,
            "test_metrics_may_select_evaluation_state": False,
            "checkpoint_inventory": {
                str(update): digest
                for update, digest in self.checkpoint_inventory.items()
            },
            "states": self.states,
        }

    @staticmethod
    def validate_manifest(
        payload: Mapping[str, Any], selection: CheckpointSelection
    ) -> None:
        _require(isinstance(payload, Mapping), "evaluation registry must be a mapping")
        _require(payload.get("schema_version") == EVALUATION_SCHEMA_VERSION, "wrong evaluation schema")
        expected = {
            "selection_fingerprint_sha256": json_sha256(selection.to_manifest()),
            "selected_optimizer_update": selection.optimizer_update,
            "selected_checkpoint_sha256": selection.checkpoint_sha256,
            "training_state_mutation_allowed": False,
            "checkpoint_selection_mutation_allowed": False,
            "test_metrics_may_select_evaluation_state": False,
        }
        for key, value in expected.items():
            _require(payload.get(key) == value, f"evaluation registry violates {key}")
        raw_inventory = payload.get("checkpoint_inventory")
        _require(isinstance(raw_inventory, Mapping), "checkpoint inventory must be a mapping")
        inventory = {
            _require_int(int(update), "checkpoint inventory update", minimum=0): _require_sha256(
                digest, "checkpoint inventory digest"
            )
            for update, digest in raw_inventory.items()
        }
        registry = EvaluationStateRegistry(selection, inventory)
        states = payload.get("states")
        _require(isinstance(states, Sequence), "evaluation states must be a sequence")
        for state in states:
            _require(isinstance(state, Mapping), "evaluation state must be a mapping")
            kind = state.get("kind")
            if kind == "single":
                registry.add_single(state.get("state_id"))
            elif kind == "ema":
                registry.add_ema(
                    state.get("state_id"),
                    state.get("artifact_sha256"),
                    state.get("ema_decay"),
                )
            elif kind == "checkpoint_average":
                includes_post_selection = state.get(
                    "includes_post_selection_trained_source", False
                )
                _require(
                    isinstance(includes_post_selection, bool),
                    "checkpoint-average post-selection disclosure must be boolean",
                )
                registry.add_checkpoint_average(
                    state.get("state_id"),
                    state.get("artifact_sha256"),
                    state.get("source_checkpoint_updates"),
                    allow_post_selection_sources=includes_post_selection,
                )
            else:
                raise HarnessValidationError("unknown evaluation-state kind")
        _require(registry.states == list(states), "evaluation-state provenance mismatch")


def build_training_harness_manifest(
    *,
    method_id: str,
    implementation_sha256: str,
    trial_index: int,
    seed: int,
    method_config: Mapping[str, Any],
    split_contract: FrozenSplitContract,
    update_budget: OptimizerUpdateBudget,
    exposure_ledger: ExposureLedger,
    validation_trajectory: ValidationTrajectory,
    search_contract: SearchBudgetContract,
    evaluation_states: EvaluationStateRegistry,
) -> dict[str, Any]:
    """Build and validate one complete trial/fold manifest."""

    method = _require_nonempty_string(method_id, "method_id")
    implementation = _require_sha256(implementation_sha256, "implementation_sha256")
    trial = _require_int(trial_index, "trial_index", minimum=0)
    trial_seed = _require_int(seed, "seed", minimum=0)
    _require(exposure_ledger.budget == update_budget, "ledger uses a different update budget")
    _require(exposure_ledger.finalized, "exposure ledger must be finalized")
    selection = validation_trajectory.select_checkpoint()
    _require(
        split_contract.reporters == update_budget.reporters
        == validation_trajectory.objective.reporters,
        "reporter order differs across data, budget, and validation contracts",
    )
    _require(
        update_budget.total_optimizer_updates
        == validation_trajectory.schedule.total_optimizer_updates
        == search_contract.optimizer_updates_per_trial_fold,
        "optimizer-update budgets differ across contracts",
    )
    space = search_contract.method(method)
    _require(trial < search_contract.trial_count_per_method, "trial_index is outside search budget")
    _require(search_contract.trial_seeds[trial] == trial_seed, "trial seed differs from shared search slot")
    normalized_config = _json_copy(method_config)
    _require(
        normalized_config == space.trial_configs[trial],
        "method config differs from frozen search trial",
    )
    fold_key = (split_contract.split_name, split_contract.fold)
    _require(
        fold_key
        in {(fold.split_name, fold.fold) for fold in search_contract.development_folds},
        "split/fold is outside development budget",
    )
    _require(
        evaluation_states.selection_fingerprint
        == json_sha256(selection.to_manifest()),
        "evaluation states are tied to a different selection",
    )

    data_manifest = split_contract.to_manifest()
    budget_manifest = update_budget.to_manifest()
    trajectory_manifest = validation_trajectory.to_manifest()
    search_manifest = search_contract.to_manifest()
    payload = {
        "schema_version": HARNESS_SCHEMA_VERSION,
        "method_id": method,
        "implementation_sha256": implementation,
        "trial_index": trial,
        "seed": trial_seed,
        "method_config": normalized_config,
        "split_contract": data_manifest,
        "split_contract_sha256": json_sha256(data_manifest),
        "update_budget": budget_manifest,
        "update_budget_sha256": json_sha256(budget_manifest),
        "exposure_ledger": exposure_ledger.to_manifest(),
        "validation_trajectory": trajectory_manifest,
        "validation_trajectory_sha256": json_sha256(trajectory_manifest),
        "checkpoint_selection": selection.to_manifest(),
        "checkpoint_selection_sha256": json_sha256(selection.to_manifest()),
        "search_contract": search_manifest,
        "search_contract_sha256": json_sha256(search_manifest),
        "evaluation_states": evaluation_states.to_manifest(),
        "development_selection_input": "complete_validation_trajectory",
        "outer_test_used_for_training_or_selection": False,
        "statistical_unit": split_contract.statistical_unit.to_manifest(),
    }
    payload["manifest_sha256"] = json_sha256(payload)
    validate_training_harness_manifest(payload)
    return payload


def validate_training_harness_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed validation of a complete serialized trial/fold record."""

    _require(isinstance(payload, Mapping), "harness manifest must be a mapping")
    _require(payload.get("schema_version") == HARNESS_SCHEMA_VERSION, "wrong harness schema")
    declared_hash = payload.get("manifest_sha256")
    unhashed = dict(payload)
    unhashed.pop("manifest_sha256", None)
    _require(declared_hash == json_sha256(unhashed), "harness manifest hash mismatch")
    _require(payload.get("outer_test_used_for_training_or_selection") is False, "outer test leaked into development")
    _require(
        payload.get("development_selection_input") == "complete_validation_trajectory",
        "checkpoint selection input is not the complete validation trajectory",
    )
    _require_sha256(payload.get("implementation_sha256"), "implementation_sha256")
    method_id = _require_nonempty_string(payload.get("method_id"), "method_id")
    split = FrozenSplitContract.from_manifest(payload.get("split_contract"))
    _require(
        payload.get("split_contract_sha256") == json_sha256(split.to_manifest()),
        "split contract hash mismatch",
    )
    _require(
        payload.get("statistical_unit") == split.statistical_unit.to_manifest(),
        "statistical-unit contract mismatch",
    )
    budget = OptimizerUpdateBudget.from_manifest(payload.get("update_budget"))
    _require(
        payload.get("update_budget_sha256") == json_sha256(budget.to_manifest()),
        "update budget hash mismatch",
    )
    ExposureLedger.validate_manifest(payload.get("exposure_ledger"), budget)
    trajectory = ValidationTrajectory.from_manifest(payload.get("validation_trajectory"))
    _require(trajectory.complete, "validation trajectory is incomplete")
    _require(
        payload.get("validation_trajectory_sha256")
        == json_sha256(trajectory.to_manifest()),
        "validation trajectory hash mismatch",
    )
    selection = trajectory.select_checkpoint()
    _require(payload.get("checkpoint_selection") == selection.to_manifest(), "checkpoint selection mismatch")
    _require(
        payload.get("checkpoint_selection_sha256") == json_sha256(selection.to_manifest()),
        "checkpoint selection hash mismatch",
    )
    search = SearchBudgetContract.from_manifest(payload.get("search_contract"))
    _require(
        payload.get("search_contract_sha256") == json_sha256(search.to_manifest()),
        "search contract hash mismatch",
    )
    trial = _require_int(payload.get("trial_index"), "trial_index", minimum=0)
    seed = _require_int(payload.get("seed"), "seed", minimum=0)
    _require(trial < search.trial_count_per_method, "trial index exceeds search budget")
    _require(search.trial_seeds[trial] == seed, "manifest seed differs from search slot")
    space = search.method(method_id)
    _require(payload.get("method_config") == space.trial_configs[trial], "method config differs from search contract")
    _require(
        (split.split_name, split.fold)
        in {(fold.split_name, fold.fold) for fold in search.development_folds},
        "manifest split/fold differs from search budget",
    )
    _require(
        split.reporters == budget.reporters == trajectory.objective.reporters,
        "reporter contracts differ",
    )
    _require(
        budget.total_optimizer_updates
        == trajectory.schedule.total_optimizer_updates
        == search.optimizer_updates_per_trial_fold,
        "optimizer update budget differs",
    )
    EvaluationStateRegistry.validate_manifest(payload.get("evaluation_states"), selection)
    return _json_copy(payload)


__all__ = [
    "CheckpointSelection",
    "CohortFingerprint",
    "DevelopmentFold",
    "EvaluationStateRegistry",
    "ExposureLedger",
    "FrozenSplitContract",
    "HarnessValidationError",
    "MethodSearchSpace",
    "OptimizerUpdateBudget",
    "ReporterCohortContract",
    "ReporterNormalizedCheckpointObjective",
    "SearchBudgetContract",
    "StatisticalUnitContract",
    "ValidationSchedule",
    "ValidationTrajectory",
    "build_training_harness_manifest",
    "canonical_json",
    "json_sha256",
    "validate_training_harness_manifest",
]
