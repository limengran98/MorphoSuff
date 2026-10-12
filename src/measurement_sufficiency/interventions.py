"""Same-cell linkage checks and falsification interventions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .adapters import AdapterCapabilities
from .schemas import PAIRING, SchemaError


def validate_exact_pairing(pairing: pd.DataFrame, capabilities: AdapterCapabilities | None = None) -> pd.DataFrame:
    if capabilities is not None:
        capabilities.require("exact_pairing", analysis="same-cell intervention")
    PAIRING.validate(pairing, "pairing")
    if pairing.duplicated(["assay_id", "input_cell_id"]).any() or pairing.duplicated(
        ["assay_id", "target_cell_id"]
    ).any():
        raise SchemaError(
            "same-cell analysis requires one-to-one input/target linkage within each assay"
        )
    if pairing.duplicated(["assay_id", "pairing_key"]).any():
        raise SchemaError("same-cell pairing_key must be unique within each assay")
    confidence = pd.to_numeric(pairing["pairing_confidence"], errors="coerce")
    if confidence.isna().any() or not confidence.between(0.0, 1.0).all():
        raise SchemaError("pairing_confidence must be numeric and within [0, 1]")
    return pairing


_MAX_DERANGEMENT_ATTEMPTS = 1000


def _random_derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation with no fixed point, by bounded rejection sampling."""
    if size < 2:
        raise ValueError("every derangement group must contain at least two rows")
    identity = np.arange(size)
    for _ in range(_MAX_DERANGEMENT_ATTEMPTS):
        permutation = rng.permutation(size)
        if not np.any(permutation == identity):
            return permutation
    raise RuntimeError(
        f"no derangement found for a group of {size} rows in "
        f"{_MAX_DERANGEMENT_ATTEMPTS} attempts"
    )


def derange_targets(frame: pd.DataFrame, *, target_columns: list[str], group_columns: list[str], seed: int = 0) -> pd.DataFrame:
    """Permute target values within each group, never leaving an item paired to itself."""
    if not group_columns:
        raise ValueError("derangement requires at least one grouping column")
    missing = set(target_columns).difference(frame.columns)
    if missing:
        raise KeyError(f"missing target columns: {', '.join(sorted(missing))}")
    if set(target_columns).intersection(group_columns):
        raise ValueError("target columns and grouping columns must be disjoint")
    result = frame.copy()
    values = frame.loc[:, target_columns].to_numpy()
    column_positions = [result.columns.get_loc(column) for column in target_columns]
    rng = np.random.default_rng(seed)
    # ``.indices`` is positional. The label-based ``.groups`` variant assigned
    # through ``.loc``, which broadcasts across a duplicated index and silently
    # destroys both the within-group permutation and the no-self-pair guarantee.
    for positions in frame.groupby(group_columns, sort=False).indices.values():
        positions = np.asarray(positions)
        permutation = _random_derangement(len(positions), rng)
        result.iloc[positions, column_positions] = values[positions[permutation]]
    return result


def covariate_matched_derangement(
    frame: pd.DataFrame,
    *,
    target_columns: list[str],
    group_columns: list[str],
    covariate_column: str,
    bins: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Derange targets only within covariate-matched strata.

    ``bins`` and ``seed`` are honoured. They were previously accepted and
    ignored, so every call returned the same fixed adjacent-rank swap: a
    falsification control with exactly one realisation, from which no null
    distribution can be drawn.

    Strata are contiguous rank blocks rather than quantile bins, because a
    continuous covariate with unique values makes every quantile bin a singleton
    and leaves nothing to permute.
    """
    if covariate_column not in frame:
        raise KeyError(f"missing covariate column: {covariate_column}")
    if bins < 1:
        raise ValueError("bins must be positive")
    missing = set(target_columns).difference(frame.columns)
    if missing:
        raise KeyError(f"missing target columns: {', '.join(sorted(missing))}")
    result = frame.copy()
    values = frame.loc[:, target_columns].to_numpy()
    column_positions = [result.columns.get_loc(column) for column in target_columns]
    positional = frame.reset_index(drop=True)
    for key, positions in positional.groupby(group_columns, sort=False).indices.items():
        positions = np.asarray(positions)
        if len(positions) < 2:
            raise ValueError("every group must contain at least two rows")
        order = positions[
            np.argsort(positional[covariate_column].to_numpy()[positions], kind="stable")
        ]
        n_blocks = max(1, min(int(bins), len(order) // 2))
        # A seed derived from the group key keeps each stratum independently
        # seeded while the whole call stays reproducible from ``seed`` alone.
        entropy = [int(seed), abs(hash(key)) % (2**31)]
        for block_index, block in enumerate(np.array_split(order, n_blocks)):
            rng = np.random.default_rng(np.random.SeedSequence(entropy + [block_index]))
            permutation = _random_derangement(len(block), rng)
            result.iloc[block, column_positions] = values[block[permutation]]
    return result


def select_features(frame: pd.DataFrame, feature_columns: list[str], *, mode: str) -> pd.DataFrame:
    if mode not in {"size_shape_only", "remove_size_shape"}:
        raise ValueError("mode must be size_shape_only or remove_size_shape")
    selected = [c for c in feature_columns if ("size" in c.lower() or "shape" in c.lower())]
    keep = selected if mode == "size_shape_only" else [c for c in feature_columns if c not in selected]
    if not keep:
        raise ValueError("feature intervention selected no features")
    return frame[keep].copy()


def cross_fitted_residuals(y: pd.Series | np.ndarray, covariates: pd.DataFrame, fold: pd.Series | np.ndarray) -> pd.Series:
    """Return residuals from ordinary least squares fit only on the opposite fold(s)."""
    y_array = np.asarray(y, dtype=float)
    x = np.asarray(covariates, dtype=float)
    folds = np.asarray(fold)
    if len(y_array) != len(x) or len(y_array) != len(folds):
        raise ValueError("y, covariates, and fold must have equal length")
    output = np.empty(len(y_array), dtype=float)
    for value in pd.unique(folds):
        test = folds == value
        train = ~test
        if train.sum() <= x.shape[1]:
            raise ValueError("insufficient training rows for cross-fitted residuals")
        design_train = np.column_stack([np.ones(train.sum()), x[train]])
        beta, *_ = np.linalg.lstsq(design_train, y_array[train], rcond=None)
        output[test] = y_array[test] - np.column_stack([np.ones(test.sum()), x[test]]) @ beta
    return pd.Series(output, index=getattr(y, "index", None), name="residual")
