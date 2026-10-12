"""Train-only preprocessing, sparse-target fitting, and canonical prediction export."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from .model_registry import create_model
from .models import SparsePredictor, SparseTargetError, SparseTargets
from .preprocessing import FeaturePreprocessor
from .schemas import coerce_boolean_column, validate_predictions


class TrainingProtocolError(ValueError):
    """Raised when IDs, partitions, or output metadata violate the training contract."""


#: Columns that name an entity, a partition or a label rather than a measurement.
#: Selecting features by numeric dtype alone silently trains on identifiers: a
#: numeric ``screen_id`` becomes a batch covariate and an integer-coded
#: ``is_control`` hands the model the very flag that defines the control-relative
#: baseline. ``studies/ops/runners/common.py`` keeps the same denylist for the OPS
#: runners and imports this tuple so the two layers cannot drift apart.
RESERVED_NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "observation_id", "input_cell_id", "target_cell_id", "reporter_id", "endpoint_id",
    "assay_id", "perturbation_id", "guide_id", "screen_id", "well_id", "field_id",
    "is_control", "is_observed", "scored", "fold", "role", "split_name", "model_id",
    "y_true", "y_pred", "pairing_key", "pairing_confidence",
)


def infer_feature_columns(
    inputs: pd.DataFrame,
    *,
    exclude: Sequence[str] = RESERVED_NON_FEATURE_COLUMNS,
    non_numeric_policy: Literal["error", "ignore"] = "error",
) -> tuple[str, ...]:
    """Choose feature columns explicitly, refusing to silently drop a broken one.

    ``select_dtypes`` alone is unsafe for two independent reasons. It keeps
    identifier columns that merely happen to be numeric, and pandas dtype is a
    property of the whole parsed file: a single non-numeric token anywhere,
    including in a held-out test row, turns a feature column to ``object`` and
    makes it disappear from the fitted feature set while every training value is
    unchanged. Both are silent today; here the first is excluded by name and the
    second raises.
    """
    reserved = set(exclude)
    candidates = [column for column in inputs.columns if column not in reserved]
    numeric, rejected = [], []
    for column in candidates:
        series = inputs[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        unusable = coerced.isna() & series.notna()
        n_bad = int(unusable.sum())
        if n_bad == 0 and not coerced.isna().all():
            # Object dtype holding only numbers. This is precisely the column the
            # old select_dtypes rule dropped, so keep it rather than losing a real
            # feature to a dtype artefact elsewhere in the file.
            numeric.append(column)
        elif 0 < n_bad < len(series):
            rejected.append(
                (column, n_bad, len(series), sorted({str(v) for v in series[unusable]})[:3])
            )
        # n_bad == len(series): a genuinely non-numeric column, skipped in silence.
    if rejected and non_numeric_policy == "error":
        column, n_bad, total, tokens = rejected[0]
        raise TrainingProtocolError(
            f"column {column!r} is object dtype but {total - n_bad}/{total} values are "
            f"numeric; the non-numeric tokens include {tokens}. pandas dtype is a "
            "whole-file property, so one such token in a held-out row silently removes "
            "the column from the fitted feature set and changes every prediction. Clean "
            "the column, or name the features explicitly with feature_columns= / --features."
        )
    if not numeric:
        raise TrainingProtocolError(
            "no usable feature column remains after excluding identifier and label "
            f"columns {sorted(reserved.intersection(inputs.columns))}; name the features "
            "explicitly with feature_columns= / --features"
        )
    return tuple(numeric)


@dataclass
class TrainedSparseModel:
    """A fitted sparse predictor plus the preprocessing learned from training cells only."""

    model_id: str
    task_mode: Literal["specialist", "shared_masked"]
    feature_columns: tuple[str, ...]
    endpoint_ids: tuple[object, ...]
    preprocessor: FeaturePreprocessor
    predictor: SparsePredictor

    def predict_matrix(self, inputs: pd.DataFrame) -> np.ndarray:
        _require_columns(inputs, ["observation_id", *self.feature_columns], "prediction inputs")
        unusable = {}
        for column in self.feature_columns:
            series = inputs[column]
            if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
                continue
            bad = series[pd.to_numeric(series, errors="coerce").isna() & series.notna()]
            if not bad.empty:
                unusable[column] = sorted({str(value) for value in bad})[:3]
        if unusable:
            raise TrainingProtocolError(
                f"prediction inputs carry non-numeric values in fitted feature columns: "
                f"{unusable}. The fitted feature set is recorded on the model, so these "
                "rows cannot be scored; clean the column rather than letting the value "
                "reach the estimator."
            )
        transformed = self.preprocessor.transform(inputs)
        matrix = self.predictor.predict(transformed.loc[:, self.feature_columns].to_numpy(dtype=float))
        if matrix.shape != (len(inputs), len(self.endpoint_ids)):
            raise TrainingProtocolError("predictor returned a matrix incompatible with fitted endpoint schema")
        return matrix


def _require_columns(frame: pd.DataFrame, columns: list[str] | tuple[str, ...], name: str) -> None:
    missing = set(columns).difference(frame.columns)
    if missing:
        raise TrainingProtocolError(f"{name} lacks columns: {', '.join(sorted(missing))}")


def _unique_inputs(inputs: pd.DataFrame, name: str) -> None:
    _require_columns(inputs, ["observation_id"], name)
    if inputs.observation_id.isna().any() or inputs.observation_id.duplicated().any():
        raise TrainingProtocolError(f"{name} must have one non-null row per observation_id")


def fit_sparse_model(
    inputs: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    model_id: str = "ridge",
    task_mode: Literal["specialist", "shared_masked"] = "specialist",
    feature_columns: list[str] | tuple[str, ...] | None = None,
    exclude_columns: Sequence[str] | None = None,
    endpoint_ids: list[object] | tuple[object, ...] | None = None,
    seed: int = 0,
    model_kwargs: dict[str, object] | None = None,
) -> TrainedSparseModel:
    """Fit on a set of training observations with endpoint-wise availability.

    The target long table may omit unavailable endpoint rows or represent them
    with ``is_observed=False``.  Neither form is imputed as zero.
    """
    _unique_inputs(inputs, "training inputs")
    if feature_columns is None:
        feature_columns = infer_feature_columns(
            inputs, exclude=exclude_columns if exclude_columns is not None else RESERVED_NON_FEATURE_COLUMNS
        )
    feature_columns = tuple(feature_columns)
    if not feature_columns:
        raise TrainingProtocolError("at least one feature column is required")
    _require_columns(inputs, list(feature_columns), "training inputs")
    target_batch = SparseTargets.from_long(
        targets,
        observation_ids=tuple(inputs.observation_id),
        endpoint_ids=endpoint_ids,
    )
    preprocessor = FeaturePreprocessor().fit(inputs, list(feature_columns))
    transformed = preprocessor.transform(inputs)
    predictor = create_model(model_id, task_mode=task_mode, seed=seed, **(model_kwargs or {}))
    predictor.fit(transformed.loc[:, feature_columns].to_numpy(dtype=float), target_batch)
    return TrainedSparseModel(model_id, task_mode, feature_columns, target_batch.endpoint_ids, preprocessor, predictor)


def prediction_long_table(
    fitted: TrainedSparseModel,
    inputs: pd.DataFrame,
    *,
    metadata: pd.DataFrame,
    split_name: str,
    fold: int,
    targets: pd.DataFrame | None = None,
    include_unobserved: bool = False,
    allow_unmodelled_endpoints: bool = False,
) -> pd.DataFrame:
    """Convert predictions to the package's canonical long prediction contract.

    With targets supplied, the default exports only observed endpoints so every
    emitted row has a valid ``y_true`` for evaluation.  ``include_unobserved``
    is reserved for inference exports and intentionally cannot be validated as
    an evaluation prediction table until labels are attached.
    """
    _unique_inputs(inputs, "prediction inputs")
    _unique_inputs(metadata, "prediction metadata")
    _require_columns(metadata, ["screen_id", "perturbation_id"], "prediction metadata")
    metadata_columns = ["observation_id", "screen_id", "perturbation_id"]
    if "reporter_id" in metadata:
        metadata_columns.append("reporter_id")
    # is_control is carried through when the caller supplies it, because
    # analysis.response_fidelity requires it on the prediction table and no other
    # producer in the repository emits it. Parsed once, up front, so a malformed
    # flag fails before the row loop rather than midway through it.
    carries_control = "is_control" in metadata
    if carries_control:
        metadata_columns.append("is_control")
    meta = inputs[["observation_id"]].merge(
        metadata[metadata_columns], on="observation_id", how="left", validate="one_to_one"
    )
    if meta[["screen_id", "perturbation_id"]].isna().any().any():
        raise TrainingProtocolError("prediction metadata must cover every input observation")
    control_flags = (
        coerce_boolean_column(meta["is_control"], column="is_control", table="prediction metadata")
        if carries_control
        else None
    )
    matrix = fitted.predict_matrix(inputs)
    target_batch: SparseTargets | None = None
    if targets is not None:
        known = {str(endpoint) for endpoint in fitted.endpoint_ids}
        unknown = sorted({str(endpoint) for endpoint in targets.endpoint_id} - known)
        if unknown:
            if not allow_unmodelled_endpoints:
                raise TrainingProtocolError(
                    f"evaluation targets contain {len(unknown)} endpoint(s) the fitted model "
                    f"never saw: {unknown[:5]}. The fitted endpoint schema is {sorted(known)}. "
                    "This happens when an endpoint is absent from the fold's train partition. "
                    "Re-fit with fit_sparse_model(..., endpoint_ids=<global endpoint list>) so "
                    "the schema is fold-independent, or pass allow_unmodelled_endpoints=True to "
                    "evaluate only the modelled subset."
                )
            targets = targets.loc[targets.endpoint_id.astype(str).isin(known)].copy()
        target_batch = SparseTargets.from_long(targets, observation_ids=tuple(inputs.observation_id), endpoint_ids=fitted.endpoint_ids)
    if targets is None and not include_unobserved:
        raise TrainingProtocolError("targets are required for an evaluable prediction export; set include_unobserved=True for inference")
    rows: list[dict[str, object]] = []
    for i, observation_id in enumerate(inputs.observation_id):
        metadata_row = meta.iloc[i]
        for j, endpoint_id in enumerate(fitted.endpoint_ids):
            observed = target_batch is not None and bool(target_batch.observed[i, j])
            if target_batch is not None and not observed and not include_unobserved:
                continue
            row: dict[str, object] = {
                "observation_id": observation_id,
                "reporter_id": str(metadata_row.get("reporter_id", "unspecified")),
                "endpoint_id": endpoint_id,
                "split_name": split_name,
                "fold": fold,
                "model_id": fitted.model_id,
                "y_true": target_batch.values[i, j] if observed else np.nan,
                "y_pred": matrix[i, j],
                "screen_id": metadata_row.screen_id,
                "perturbation_id": metadata_row.perturbation_id,
            }
            # reporter_id can be supplied per observation, otherwise callers
            # should use a stable explicit default above.
            if "reporter_id" in metadata:
                row["reporter_id"] = str(metadata_row.reporter_id)
            if control_flags is not None:
                row["is_control"] = bool(control_flags.iloc[i])
            rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise TrainingProtocolError("prediction export contains no observed endpoints")
    if not include_unobserved:
        validate_predictions(result)
    return result


def fit_from_manifest(
    inputs: pd.DataFrame,
    targets: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    fold: int,
    model_id: str = "ridge",
    task_mode: Literal["specialist", "shared_masked"] = "specialist",
    feature_columns: list[str] | tuple[str, ...] | None = None,
    exclude_columns: Sequence[str] | None = None,
    seed: int = 0,
) -> TrainedSparseModel:
    """Fit one fold after verifying that only its explicit train role is used."""
    _require_columns(assignments, ["observation_id", "fold", "role"], "split assignments")
    part = assignments.loc[(assignments.fold == fold) & (assignments.role == "train"), ["observation_id"]]
    if part.empty:
        raise TrainingProtocolError(f"fold {fold} has no train observations")
    if part.observation_id.duplicated().any():
        raise TrainingProtocolError("split assignments duplicate training observation IDs")
    train_ids = set(part.observation_id)
    train_inputs = inputs.loc[inputs.observation_id.isin(train_ids)].copy()
    if len(train_inputs) != len(train_ids):
        raise TrainingProtocolError("split assignment refers to inputs that are absent")
    train_targets = targets.loc[targets.observation_id.isin(train_ids)].copy()
    return fit_sparse_model(
        train_inputs,
        train_targets,
        model_id=model_id,
        task_mode=task_mode,
        feature_columns=feature_columns,
        exclude_columns=exclude_columns,
        seed=seed,
    )
