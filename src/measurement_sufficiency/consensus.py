"""Frozen model registry and unweighted median consensus."""

from __future__ import annotations

import pandas as pd

from .schemas import validate_predictions


FINAL_TEN_MODELS: tuple[str, ...] = (
    "ridge", "gbdt", "catboost", "mlp", "tabm", "scbutterfly", "resmlp", "multitab", "midas", "scpair",
)


def validate_final_registry(model_ids: pd.Series | list[str] | tuple[str, ...]) -> None:
    actual = set(model_ids)
    expected = set(FINAL_TEN_MODELS)
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError("consensus requires exactly the frozen ten-model registry (" + "; ".join(detail) + ")")


def consensus_median(predictions: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Collapse model predictions with the manuscript's unweighted median definition."""
    predictions = validate_predictions(predictions)
    keys = ["observation_id", "reporter_id", "endpoint_id", "split_name", "fold", "screen_id", "perturbation_id"]
    # is_control is a property of the observation, not of the model, so it groups
    # cleanly alongside the other keys. Without this the consensus output loses the
    # column and can no longer be fed to analysis.response_fidelity, which is the
    # documented next step for the ten-method median.
    if "is_control" in predictions:
        keys.append("is_control")
    model_counts = predictions.groupby(keys, dropna=False).model_id.agg(list)
    if strict:
        for model_ids in model_counts:
            validate_final_registry(model_ids)
        truth_counts = predictions.groupby(keys, dropna=False).y_true.nunique(dropna=False)
        if not truth_counts.eq(1).all():
            raise ValueError("consensus requires identical y_true across every aligned model row")
    result = predictions.groupby(keys, as_index=False, dropna=False).agg(y_true=("y_true", "first"), y_pred=("y_pred", "median"), n_models=("model_id", "nunique"))
    result["model_id"] = "ten_model_median"
    return result[keys + ["model_id", "y_true", "y_pred", "n_models"]]
