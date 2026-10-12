"""Mask-aware, portable model primitives.

Optional third-party estimators are imported only when requested, so the core
schema and analysis package remains lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .schemas import SchemaError, coerce_boolean_column


class ModelDependencyError(ImportError):
    """Raised with an actionable message when an optional model extra is absent."""


class SparseTargetError(ValueError):
    """Raised for malformed sparse target matrices or masks."""


def _sklearn() -> tuple[object, object]:
    try:
        from sklearn.base import clone  # type: ignore
        from sklearn.ensemble import HistGradientBoostingRegressor  # type: ignore
        from sklearn.linear_model import Ridge  # type: ignore
        from sklearn.neural_network import MLPRegressor  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        raise ModelDependencyError(
            "Built-in model training requires optional dependency scikit-learn. "
            "Install measurement-sufficiency[sklearn]."
        ) from exc
    return clone, (Ridge, HistGradientBoostingRegressor, MLPRegressor)


def sklearn_estimator(model_id: str, *, seed: int = 0, **kwargs: object) -> object:
    """Build one supported single-output estimator without importing sklearn at import time."""
    _, classes = _sklearn()
    Ridge, HistGradientBoostingRegressor, MLPRegressor = classes
    if model_id == "ridge":
        return Ridge(**kwargs)
    if model_id == "gbdt":
        return HistGradientBoostingRegressor(random_state=seed, **kwargs)
    if model_id == "mlp":
        defaults: dict[str, object] = {"hidden_layer_sizes": (64, 32), "max_iter": 400, "random_state": seed}
        defaults.update(kwargs)
        return MLPRegressor(**defaults)
    raise ValueError(f"no built-in estimator named {model_id!r}")


def catboost_estimator(*, seed: int = 0, **kwargs: object) -> object:
    """Build a fail-clean CatBoost single-endpoint specialist."""
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional environment
        raise ModelDependencyError(
            "CatBoost training requires the optional CatBoost package; install "
            "measurement-sufficiency[catboost]."
        ) from exc
    defaults: dict[str, object] = {
        "loss_function": "RMSE",
        "random_seed": seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    defaults.update(kwargs)
    return CatBoostRegressor(**defaults)


@dataclass(frozen=True)
class SparseTargets:
    """A rectangular target array with a separate availability mask.

    ``values`` at unobserved locations may be arbitrary (including NaN): only
    ``observed`` determines membership in fitting and loss calculations.
    """

    observation_ids: tuple[object, ...]
    endpoint_ids: tuple[object, ...]
    values: np.ndarray
    observed: np.ndarray

    def __post_init__(self) -> None:
        values, observed = np.asarray(self.values, dtype=float), np.asarray(self.observed, dtype=bool)
        if values.ndim != 2 or observed.shape != values.shape:
            raise SparseTargetError("values and observed mask must be same-shape two-dimensional arrays")
        if values.shape != (len(self.observation_ids), len(self.endpoint_ids)):
            raise SparseTargetError("target dimensions do not match observation_ids and endpoint_ids")
        if len(set(self.observation_ids)) != len(self.observation_ids) or len(set(self.endpoint_ids)) != len(self.endpoint_ids):
            raise SparseTargetError("observation_ids and endpoint_ids must be unique")
        if np.any(~np.isfinite(values[observed])):
            raise SparseTargetError("observed targets must be finite")

    @classmethod
    def from_long(
        cls,
        targets: pd.DataFrame,
        *,
        observation_ids: list[object] | tuple[object, ...] | None = None,
        endpoint_ids: list[object] | tuple[object, ...] | None = None,
        value_column: str = "y_true",
        observed_column: str = "is_observed",
    ) -> "SparseTargets":
        required = {"observation_id", "endpoint_id", value_column}
        missing = required.difference(targets.columns)
        if missing:
            raise SparseTargetError(f"target table lacks columns: {', '.join(sorted(missing))}")
        if targets.duplicated(["observation_id", "endpoint_id"]).any():
            raise SparseTargetError("target table has duplicate observation_id/endpoint_id rows")
        obs = tuple(observation_ids) if observation_ids is not None else tuple(pd.unique(targets.observation_id))
        ends = tuple(endpoint_ids) if endpoint_ids is not None else tuple(pd.unique(targets.endpoint_id))
        if not obs or not ends:
            raise SparseTargetError("at least one observation and endpoint are required")
        unknown_obs, unknown_end = set(targets.observation_id).difference(obs), set(targets.endpoint_id).difference(ends)
        if unknown_obs or unknown_end:
            raise SparseTargetError("target table contains IDs outside the requested matrix")
        values = np.full((len(obs), len(ends)), np.nan, dtype=float)
        observed = np.zeros_like(values, dtype=bool)
        obs_i, end_i = {v: i for i, v in enumerate(obs)}, {v: i for i, v in enumerate(ends)}
        if observed_column in targets:
            # Strict parsing, never Python truthiness: a blank CSV cell reads as
            # NaN, and bool(nan) is True, which would admit an unmeasured
            # endpoint as a training label and as an evaluation y_true.
            try:
                flags = coerce_boolean_column(
                    targets[observed_column], column=observed_column, table="target table"
                ).to_numpy()
            except SchemaError as exc:
                raise SparseTargetError(str(exc)) from exc
        else:
            flags = pd.notna(targets[value_column]).to_numpy()
        numeric = pd.to_numeric(targets[value_column], errors="coerce").to_numpy(dtype=float)
        bad = flags & ~np.isfinite(numeric)
        if bad.any():
            sample = targets.loc[bad, ["observation_id", "endpoint_id"]].head(5)
            raise SparseTargetError(
                f"{observed_column}=True requires a finite target value; first offenders: "
                f"{sample.to_dict('records')}"
            )
        rows = targets.observation_id.map(obs_i).to_numpy()
        cols = targets.endpoint_id.map(end_i).to_numpy()
        values[rows[flags], cols[flags]] = numeric[flags]
        observed[rows[flags], cols[flags]] = True
        return cls(obs, ends, values, observed)

    def subset_observations(self, observation_ids: list[object] | tuple[object, ...]) -> "SparseTargets":
        lookup = {v: i for i, v in enumerate(self.observation_ids)}
        missing = set(observation_ids).difference(lookup)
        if missing:
            raise SparseTargetError("requested observations are absent from targets")
        idx = [lookup[v] for v in observation_ids]
        return SparseTargets(tuple(observation_ids), self.endpoint_ids, self.values[idx], self.observed[idx])


def observed_only_mse(y_true: np.ndarray, y_pred: np.ndarray, observed: np.ndarray) -> float:
    """Mean squared error whose numerator *and denominator* exclude unavailable endpoints."""
    true, pred, mask = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float), np.asarray(observed, dtype=bool)
    if true.shape != pred.shape or mask.shape != true.shape:
        raise SparseTargetError("y_true, y_pred, and observed mask must have identical shapes")
    if not mask.any():
        raise SparseTargetError("observed-only loss is undefined with no observed targets")
    if not np.isfinite(true[mask]).all() or not np.isfinite(pred[mask]).all():
        raise SparseTargetError("observed-only loss requires finite values at observed positions")
    return float(np.mean((true[mask] - pred[mask]) ** 2))


class SparsePredictor(Protocol):
    endpoint_ids: tuple[object, ...]

    def fit(self, features: np.ndarray, targets: SparseTargets) -> "SparsePredictor": ...

    def predict(self, features: np.ndarray) -> np.ndarray: ...


EstimatorFactory = Callable[[], object]


@dataclass
class SpecialistSparseRegressor:
    """One estimator per endpoint, each fit solely on its observed rows."""

    estimator_factory: EstimatorFactory
    endpoint_ids: tuple[object, ...] = ()
    estimators_: list[object | None] = field(default_factory=list)

    def fit(self, features: np.ndarray, targets: SparseTargets) -> "SpecialistSparseRegressor":
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[0] != targets.values.shape[0]:
            raise SparseTargetError("features must have one row per target observation")
        self.endpoint_ids = targets.endpoint_ids
        self.estimators_ = []
        for endpoint_idx in range(targets.values.shape[1]):
            mask = targets.observed[:, endpoint_idx]
            if not mask.any():
                self.estimators_.append(None)
                continue
            estimator = self.estimator_factory()
            estimator.fit(x[mask], targets.values[mask, endpoint_idx])
            self.estimators_.append(estimator)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("model must be fit before prediction")
        x = np.asarray(features, dtype=float)
        output = np.full((len(x), len(self.endpoint_ids)), np.nan, dtype=float)
        for idx, estimator in enumerate(self.estimators_):
            if estimator is not None:
                output[:, idx] = estimator.predict(x)
        return output


@dataclass
class SharedSparseRegressor:
    """A shared estimator over observed ``(cell, endpoint)`` examples.

    Endpoint identity is represented by a one-hot block, allowing heterogeneous
    endpoint availability without converting unobserved targets to zeros.
    """

    estimator_factory: EstimatorFactory
    endpoint_ids: tuple[object, ...] = ()
    estimator_: object | None = None

    def _design(self, features: np.ndarray, row_idx: np.ndarray, endpoint_idx: np.ndarray) -> np.ndarray:
        base = np.asarray(features, dtype=float)[row_idx]
        one_hot = np.zeros((len(row_idx), len(self.endpoint_ids)), dtype=float)
        one_hot[np.arange(len(row_idx)), endpoint_idx] = 1.0
        return np.hstack([base, one_hot])

    def fit(self, features: np.ndarray, targets: SparseTargets) -> "SharedSparseRegressor":
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[0] != targets.values.shape[0]:
            raise SparseTargetError("features must have one row per target observation")
        self.endpoint_ids = targets.endpoint_ids
        rows, endpoints = np.where(targets.observed)
        if not len(rows):
            raise SparseTargetError("shared model requires at least one observed target")
        self.estimator_ = self.estimator_factory()
        self.estimator_.fit(self._design(x, rows, endpoints), targets.values[rows, endpoints])
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("model must be fit before prediction")
        x = np.asarray(features, dtype=float)
        rows = np.repeat(np.arange(len(x)), len(self.endpoint_ids))
        endpoints = np.tile(np.arange(len(self.endpoint_ids)), len(x))
        return np.asarray(self.estimator_.predict(self._design(x, rows, endpoints)), dtype=float).reshape(len(x), len(self.endpoint_ids))
