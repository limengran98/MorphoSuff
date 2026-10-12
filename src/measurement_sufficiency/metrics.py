"""Cell- and perturbation-level metrics derived only from prediction long tables."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schemas import coerce_boolean_column, validate_predictions


def _pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2 or x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan")
    return float(x.corr(y))


def _metric_row(group: pd.DataFrame) -> pd.Series:
    err = group.y_pred - group.y_true
    ss_tot = ((group.y_true - group.y_true.mean()) ** 2).sum()
    r2 = float("nan") if ss_tot == 0 else float(1 - (err**2).sum() / ss_tot)
    return pd.Series({"n": len(group), "pearson_r": _pearson(group.y_true, group.y_pred), "mae": float(err.abs().mean()), "rmse": float(np.sqrt((err**2).mean())), "r2": r2})


def drop_baseline_rows(
    predictions: pd.DataFrame, *, control_column: str = "is_control"
) -> tuple[pd.DataFrame, int]:
    """Split off control rows, which are an assay reference rather than a score.

    In a gene-holdout design controls are present in every fold so that a
    control-relative response has a same-screen baseline. They are not held-out
    evaluation units, and their expected response is zero by construction, so
    scoring them would report accuracy on the reference itself. Returns the number
    dropped so the exclusion is always stated rather than silent.
    """
    if control_column not in predictions:
        return predictions, 0
    is_control = coerce_boolean_column(
        predictions[control_column], column=control_column, table="predictions"
    )
    return predictions.loc[~is_control], int(is_control.sum())


def cell_metrics(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    *,
    exclude_controls: bool = True,
) -> pd.DataFrame:
    """Compute cell-level accuracy without peeking into model-specific output files."""
    predictions = validate_predictions(predictions)
    n_excluded = 0
    if exclude_controls:
        predictions, n_excluded = drop_baseline_rows(predictions)
    if predictions.empty:
        raise ValueError("no scorable rows remain after excluding control observations")
    groups = group_columns or ["reporter_id", "endpoint_id", "split_name", "fold", "model_id"]
    result = predictions.groupby(groups, dropna=False, sort=False)[["y_true", "y_pred"]].apply(_metric_row).reset_index()
    result["n_excluded_controls"] = n_excluded
    return result


def ko_metrics(
    predictions: pd.DataFrame,
    group_columns: list[str] | None = None,
    *,
    exclude_controls: bool = True,
) -> pd.DataFrame:
    """Evaluate accuracy after mean aggregation at perturbation × screen level."""
    predictions = validate_predictions(predictions)
    n_excluded = 0
    if exclude_controls:
        predictions, n_excluded = drop_baseline_rows(predictions)
    if predictions.empty:
        raise ValueError("no scorable rows remain after excluding control observations")
    aggregate_keys = ["reporter_id", "endpoint_id", "split_name", "fold", "model_id", "perturbation_id", "screen_id"]
    aggregate = predictions.groupby(aggregate_keys, dropna=False, sort=False)[["y_true", "y_pred"]].mean().reset_index()
    groups = group_columns or ["reporter_id", "endpoint_id", "split_name", "fold", "model_id"]
    result = aggregate.groupby(groups, dropna=False, sort=False)[["y_true", "y_pred"]].apply(_metric_row).reset_index()
    result["n_excluded_controls"] = n_excluded
    return result


def screen_metrics(predictions: pd.DataFrame, *, exclude_controls: bool = True) -> pd.DataFrame:
    return cell_metrics(
        predictions,
        ["reporter_id", "endpoint_id", "split_name", "fold", "model_id", "screen_id"],
        exclude_controls=exclude_controls,
    )
