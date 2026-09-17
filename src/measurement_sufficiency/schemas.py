"""Canonical long-table schemas and strict, portable validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import pandas as pd


class SchemaError(ValueError):
    """Raised when a canonical table is incomplete or violates its key contract."""


TRUE_TOKENS = frozenset({"1", "true", "t", "yes", "y"})
FALSE_TOKENS = frozenset({"0", "false", "f", "no", "n"})


def coerce_boolean_column(
    series: pd.Series,
    *,
    column: str,
    table: str,
    extra_true: Iterable[str] = (),
    extra_false: Iterable[str] = (),
) -> pd.Series:
    """Parse a boolean flag column strictly, refusing anything ambiguous.

    Python truthiness is the wrong tool here.  ``bool(float("nan"))`` is ``True``
    and ``bool("no")`` is ``True``, so a blank cell or a string token silently
    becomes a positive flag.  For ``is_observed`` that turns an unmeasured
    endpoint into a training label, and for ``is_control`` it turns a perturbed
    cell into a baseline.  Every value that is not an explicit boolean token
    therefore raises, including nulls: a missing flag is a data defect, not a
    default.
    """
    truthy = TRUE_TOKENS.union(token.casefold() for token in extra_true)
    falsey = FALSE_TOKENS.union(token.casefold() for token in extra_false)
    overlap = truthy.intersection(falsey)
    if overlap:
        raise SchemaError(f"boolean token declared both true and false: {sorted(overlap)}")

    if pd.api.types.is_bool_dtype(series) and not series.isna().any():
        return series.astype(bool)

    def convert(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if value is None or (not isinstance(value, str) and pd.isna(value)):
            raise SchemaError(
                f"{table} column {column!r} has a null value; a missing boolean flag "
                "is a data defect, not a default"
            )
        if isinstance(value, (int, float)) and float(value) in (0.0, 1.0):
            return bool(value)
        token = str(value).strip().casefold()
        if token in truthy:
            return True
        if token in falsey:
            return False
        raise SchemaError(
            f"{table} column {column!r} contains an unrecognized boolean token: "
            f"{value!r}. Accepted: {sorted(truthy | falsey)} or a real boolean."
        )

    return series.map(convert).astype(bool)


@dataclass(frozen=True)
class FeatureSchema:
    """Declared input representation, with stable feature names and semantics."""

    schema_id: str
    feature_names: tuple[str, ...]
    representation: str = "tabular"

    def validate_columns(self, columns: list[str]) -> None:
        if tuple(columns) != self.feature_names:
            raise SchemaError("input feature order or names do not match FeatureSchema")


@dataclass(frozen=True)
class TargetSchema:
    """Declared reporter endpoint block; dense vectors are never assumed globally."""

    schema_id: str
    reporter_id: str
    endpoint_ids: tuple[str, ...]
    target_modality: str


@dataclass(frozen=True)
class TableSchema:
    required: tuple[str, ...]
    unique: tuple[str, ...] = ()
    nullable: tuple[str, ...] = ()
    #: Columns that a table may omit, but that must be complete when present.
    #: An optional flag carrying nulls is worse than an absent one, because a
    #: downstream consumer cannot tell "not recorded" from "recorded as false".
    optional_non_null: tuple[str, ...] = ()

    def validate(self, table: pd.DataFrame, name: str) -> pd.DataFrame:
        missing = sorted(set(self.required).difference(table.columns))
        if missing:
            raise SchemaError(f"{name} is missing required columns: {', '.join(missing)}")
        required_nonnull = set(self.required).difference(self.nullable)
        present_optional = set(self.optional_non_null).intersection(table.columns)
        nulls = [
            column
            for column in sorted(required_nonnull | present_optional)
            if table[column].isna().any()
        ]
        if nulls:
            raise SchemaError(f"{name} has null required values: {', '.join(sorted(nulls))}")
        if self.unique and table.duplicated(list(self.unique)).any():
            raise SchemaError(f"{name} violates unique key: {', '.join(self.unique)}")
        return table


OBSERVATIONS = TableSchema(
    (
        "observation_id",
        "input_cell_id",
        "target_cell_id",
        "assay_id",
        "perturbation_id",
        "guide_id",
        "screen_id",
        "well_id",
        "field_id",
        "is_control",
    ),
    unique=("observation_id",), nullable=("guide_id",),
)
REPORTERS = TableSchema(("reporter_id", "reporter_name", "biological_system", "target_modality", "endpoint_schema_id"), unique=("reporter_id",))
ASSAYS = TableSchema(("assay_id", "reporter_id", "screen_id", "endpoint_ids"), unique=("assay_id",))
PAIRING = TableSchema(
    (
        "observation_id",
        "input_cell_id",
        "target_cell_id",
        "assay_id",
        "pairing_key",
        "pairing_confidence",
    ),
    unique=("observation_id",),
)
PREDICTIONS = TableSchema(
    ("observation_id", "reporter_id", "endpoint_id", "split_name", "fold", "model_id", "y_true", "y_pred", "screen_id", "perturbation_id"),
    unique=("observation_id", "reporter_id", "endpoint_id", "split_name", "fold", "model_id"),
    # is_control is optional so that predictions produced before it was carried
    # through remain valid, but a control-relative analysis cannot run without it
    # and it must never be partially populated. See analysis.response_fidelity.
    optional_non_null=("is_control",),
)
#: Long target table consumed by SparseTargets.from_long and the CLI. ``y_true``
#: is nullable because an unobserved endpoint legitimately carries no value;
#: finiteness is enforced only where the availability mask is true.
TARGETS = TableSchema(
    ("observation_id", "endpoint_id", "y_true"),
    unique=("observation_id", "endpoint_id"),
    nullable=("y_true",),
)


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Validate the canonical model-output table used by every evaluator."""
    validated = PREDICTIONS.validate(predictions, "predictions")
    if not all(pd.api.types.is_numeric_dtype(validated[column]) for column in ("y_true", "y_pred")):
        raise SchemaError("predictions y_true and y_pred must be numeric")
    if "is_control" in validated:
        coerce_boolean_column(validated["is_control"], column="is_control", table="predictions")
    return validated


def validate_targets(targets: pd.DataFrame, *, observed_column: str = "is_observed") -> pd.DataFrame:
    """Validate a long target table before it reaches the sparse-target builder."""
    validated = TARGETS.validate(targets, "targets")
    if observed_column in validated:
        coerce_boolean_column(validated[observed_column], column=observed_column, table="targets")
    return validated


def validate_canonical_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    expected = {"observations": OBSERVATIONS, "reporters": REPORTERS, "assays": ASSAYS, "pairing": PAIRING}
    missing = sorted(set(expected).difference(tables))
    if missing:
        raise SchemaError(f"missing canonical tables: {', '.join(missing)}")
    for name, schema in expected.items():
        schema.validate(tables[name], name)

    observations, reporters, assays, pairing = (
        tables["observations"],
        tables["reporters"],
        tables["assays"],
        tables["pairing"],
    )
    unknown_reporters = set(assays.reporter_id).difference(reporters.reporter_id)
    if unknown_reporters:
        raise SchemaError(f"assays reference unknown reporter_id values: {', '.join(map(str, sorted(unknown_reporters)))}")

    assay_ids = set(assays.assay_id)
    observation_assays = set(observations.assay_id)
    unknown_assays = observation_assays.difference(assay_ids)
    if unknown_assays:
        raise SchemaError(
            "observations reference unknown assay_id values: "
            + ", ".join(map(str, sorted(unknown_assays)))
        )
    unused_assays = assay_ids.difference(observation_assays)
    if unused_assays:
        raise SchemaError(
            "assays have no canonical observations: "
            + ", ".join(map(str, sorted(unused_assays)))
        )
    assay_screens = assays.set_index("assay_id")["screen_id"]
    expected_screens = observations["assay_id"].map(assay_screens)
    if not observations["screen_id"].eq(expected_screens).all():
        raise SchemaError("observation screen_id must match the declared assay screen_id")

    pairing_assays = set(pairing.assay_id)
    unknown_pairing_assays = pairing_assays.difference(assay_ids)
    if unknown_pairing_assays:
        raise SchemaError(
            "pairing references unknown assay_id values: "
            + ", ".join(map(str, sorted(unknown_pairing_assays)))
        )
    linkage_columns = (
        "observation_id",
        "input_cell_id",
        "target_cell_id",
        "assay_id",
    )
    observed_linkages = set(
        observations.loc[:, list(linkage_columns)].itertuples(index=False, name=None)
    )
    declared_linkages = set(
        pairing.loc[:, list(linkage_columns)].itertuples(index=False, name=None)
    )
    if observed_linkages != declared_linkages or len(pairing) != len(observations):
        raise SchemaError(
            "pairing must provide exactly one assay-qualified linkage for every observation"
        )

    # A physical phase cell can legitimately participate in several reporter
    # assays. Exactness is therefore an assay-local bijection, not a global
    # input-cell/target-cell bijection.
    for table_name, table in (("observations", observations), ("pairing", pairing)):
        if table.duplicated(["assay_id", "input_cell_id"]).any():
            raise SchemaError(
                f"{table_name} maps an input_cell_id more than once within an assay"
            )
        if table.duplicated(["assay_id", "target_cell_id"]).any():
            raise SchemaError(
                f"{table_name} maps a target_cell_id more than once within an assay"
            )
    if pairing.duplicated(["assay_id", "pairing_key"]).any():
        raise SchemaError("pairing_key must be unique within each assay")
    confidence = pd.to_numeric(pairing["pairing_confidence"], errors="coerce")
    if confidence.isna().any() or not confidence.between(0.0, 1.0).all():
        raise SchemaError("pairing_confidence must be numeric and within [0, 1]")

    for assay in assays.itertuples(index=False):
        endpoints = assay.endpoint_ids
        if isinstance(endpoints, str):
            endpoints = [item.strip() for item in endpoints.split(",") if item.strip()]
        if not isinstance(endpoints, (list, tuple)) or not endpoints or len(set(endpoints)) != len(endpoints):
            raise SchemaError("assay endpoint_ids must be a nonempty, unique endpoint list")
    # Datasets that provide a concrete endpoint registry receive an additional
    # FK check; the four-table minimum stays portable for sparse modalities.
    if "endpoints" in tables:
        endpoint_table = tables["endpoints"]
        endpoint_schema = TableSchema(("endpoint_schema_id", "endpoint_id"), unique=("endpoint_schema_id", "endpoint_id"))
        endpoint_schema.validate(endpoint_table, "endpoints")
        reporter_schemas = reporters.set_index("reporter_id").endpoint_schema_id.to_dict()
        declared = set(zip(endpoint_table.endpoint_schema_id, endpoint_table.endpoint_id))
        for assay in assays.itertuples(index=False):
            endpoint_ids = assay.endpoint_ids.split(",") if isinstance(assay.endpoint_ids, str) else assay.endpoint_ids
            schema_id = reporter_schemas[assay.reporter_id]
            missing_endpoints = set(endpoint_ids).difference({endpoint for schema, endpoint in declared if schema == schema_id})
            if missing_endpoints:
                raise SchemaError(f"assay {assay.assay_id} has endpoints absent from target schema {schema_id}: {', '.join(map(str, sorted(missing_endpoints)))}")
