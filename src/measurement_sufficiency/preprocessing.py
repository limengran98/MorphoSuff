"""Train-only preprocessing and screen-matched control-relative responses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schemas import coerce_boolean_column


@dataclass
class FeaturePreprocessor:
    """Median imputation followed by train-fitted standardisation."""

    columns: list[str] | None = None
    median_: pd.Series | None = None
    mean_: pd.Series | None = None
    scale_: pd.Series | None = None

    def fit(self, frame: pd.DataFrame, columns: list[str] | None = None) -> "FeaturePreprocessor":
        self.columns = columns or frame.select_dtypes(include=[np.number]).columns.tolist()
        if not self.columns:
            raise ValueError("no numeric feature columns selected")
        values = frame[self.columns]
        self.median_ = values.median()
        filled = values.fillna(self.median_)
        self.mean_ = filled.mean()
        self.scale_ = filled.std(ddof=0).replace(0, 1.0)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None or self.median_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeaturePreprocessor must be fit before transform")
        missing = set(self.columns).difference(frame.columns)
        if missing:
            raise ValueError(f"transform frame lacks features: {', '.join(sorted(missing))}")
        result = frame.copy()
        result[self.columns] = (result[self.columns].fillna(self.median_) - self.mean_) / self.scale_
        return result

    def fit_transform(self, frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        return self.fit(frame, columns).transform(frame)


def control_relative_response(
    values: pd.DataFrame,
    *,
    value_columns: list[str],
    perturbation_column: str = "perturbation_id",
    screen_column: str = "screen_id",
    control_column: str = "is_control",
) -> pd.DataFrame:
    """Aggregate perturbation-by-screen values then subtract matching controls."""
    required = set(value_columns + [perturbation_column, screen_column, control_column])
    missing = required.difference(values.columns)
    if missing:
        raise ValueError(f"control-relative response lacks columns: {', '.join(sorted(missing))}")
    group_cols = [perturbation_column, screen_column]
    aggregate = values.groupby(group_cols, dropna=False)[value_columns].mean().reset_index()
    is_control = coerce_boolean_column(
        values[control_column], column=control_column, table="control-relative response input"
    )
    controls = values.loc[is_control].groupby(screen_column, dropna=False)[value_columns].mean()
    if controls.empty:
        raise ValueError("control-relative response requires at least one control")
    missing_screens = set(aggregate[screen_column]).difference(controls.index)
    if missing_screens:
        raise ValueError("missing screen-matched control for one or more screens")
    response = aggregate.copy()
    for column in value_columns:
        response[column] = aggregate[column] - response[screen_column].map(controls[column])
    return response
