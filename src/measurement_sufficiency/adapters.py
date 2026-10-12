"""Dataset and model-facing protocols with explicit analytical eligibility."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import pandas as pd

from .schemas import (
    FeatureSchema,
    TargetSchema,
    coerce_boolean_column,
    validate_canonical_tables,
)


class EligibilityError(ValueError):
    """Raised when an analysis needs a capability that a dataset did not declare."""


@dataclass(frozen=True)
class AdapterCapabilities:
    exact_pairing: bool = False
    raw_images: bool = False
    guides: bool = False
    screen_matched_controls: bool = False
    repeated_screens: bool = False
    covariates: bool = False

    def require(self, *names: str, analysis: str = "analysis") -> None:
        missing = [name for name in names if not getattr(self, name, False)]
        if missing:
            joined = ", ".join(missing)
            raise EligibilityError(f"{analysis} requires declared capability: {joined}.")


@runtime_checkable
class MeasurementDatasetAdapter(Protocol):
    """Portable adapter contract; payload formats remain adapter-owned."""

    capabilities: AdapterCapabilities

    def observations(self) -> pd.DataFrame: ...
    def reporters(self) -> pd.DataFrame: ...
    def assays(self) -> pd.DataFrame: ...
    def pairing(self) -> pd.DataFrame: ...
    def input_schema(self) -> Any: ...
    def target_schema(self, reporter_id: str) -> Any: ...
    def load_inputs(self, ids: Sequence[str], representation: str) -> Any: ...
    def load_targets(self, reporter_id: str, ids: Sequence[str]) -> Any: ...
    def controls(self, screen_id: str) -> pd.DataFrame: ...


@dataclass(frozen=True)
class PredictorCapabilities:
    specialist: bool
    shared_multitask: bool
    sparse_heterogeneous_targets: bool
    tabular_input: bool
    raw_image_input: bool
    same_cell_interventions: bool
    residual_targets: bool
    deterministic: bool


def _read_local_table(path: Path) -> pd.DataFrame:
    # A directory is a partitioned parquet dataset, which is how a table too large to
    # hold whole is written: one file per screen or plate. Dispatching on the suffix
    # alone sent a directory to ``read_csv``, so a manifest naming a partitioned table
    # failed with a confusing error rather than reading it.
    if path.is_dir() or path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Parquet tables require an installed parquet engine; install "
                "measurement-sufficiency[parquet] or provide CSV files."
            ) from exc
    return pd.read_csv(path)


@dataclass
class LocalManifestAdapter:
    """Generic adapter for canonical tables named in a caller-local manifest.

    This implementation is intentionally dataset-neutral. It is suitable when
    a new resource can export canonical CSV or Parquet tables directly. More
    complex storage systems can implement :class:`MeasurementDatasetAdapter`
    without changing any downstream analysis.
    """

    manifest_path: Path

    def __post_init__(self) -> None:
        self.manifest_path = Path(self.manifest_path)
        payload = json.loads(self.manifest_path.read_text())
        self.dataset_id = str(payload.get("dataset_id", "unspecified_dataset"))
        self._specs = payload.get("tables", {})
        required = {"observations", "reporters", "assays", "pairing"}
        if missing := required.difference(self._specs):
            raise ValueError(f"local manifest lacks canonical tables: {', '.join(sorted(missing))}")
        self._tables = {name: _read_local_table(self._path(name)) for name in required}
        if "endpoints" in self._specs:
            self._tables["endpoints"] = _read_local_table(self._path("endpoints"))
        validate_canonical_tables(self._tables)
        caps = payload.get("capabilities", {})
        self.capabilities = AdapterCapabilities(
            exact_pairing=bool(caps.get("exact_pairing", False)),
            raw_images=bool(caps.get("raw_images", False)),
            guides=bool(caps.get("guides", False)),
            screen_matched_controls=bool(caps.get("screen_matched_controls", False)),
            repeated_screens=bool(caps.get("repeated_screens", False)),
            covariates=bool(caps.get("covariates", False)),
        )
        self._input_schema_spec = payload.get("input_schema", {})

    def _path(self, name: str) -> Path:
        value: Any = self._specs[name]
        value = value.get("local_path") if isinstance(value, dict) else value
        if not value:
            raise ValueError(f"manifest table {name!r} lacks local_path")
        path = Path(value)
        return path if path.is_absolute() else self.manifest_path.parent / path

    def observations(self) -> pd.DataFrame:
        return self._tables["observations"].copy()

    def reporters(self) -> pd.DataFrame:
        return self._tables["reporters"].copy()

    def assays(self) -> pd.DataFrame:
        return self._tables["assays"].copy()

    def pairing(self) -> pd.DataFrame:
        return self._tables["pairing"].copy()

    def input_schema(self) -> FeatureSchema:
        spec = self._input_schema_spec
        return FeatureSchema(
            schema_id=str(spec.get("schema_id", f"{self.dataset_id}_input_unspecified")),
            feature_names=tuple(spec.get("feature_names", ())),
            representation=str(spec.get("representation", "tabular")),
        )

    def target_schema(self, reporter_id: str) -> TargetSchema:
        reporter_rows = self.reporters().set_index("reporter_id")
        if reporter_id not in reporter_rows.index:
            raise KeyError(f"unknown reporter_id: {reporter_id}")
        reporter = reporter_rows.loc[reporter_id]
        endpoint_ids: set[str] = set()
        for value in self.assays().loc[lambda frame: frame.reporter_id.eq(reporter_id), "endpoint_ids"]:
            if isinstance(value, str):
                endpoint_ids.update(item.strip() for item in value.split(",") if item.strip())
            elif isinstance(value, (list, tuple)):
                endpoint_ids.update(map(str, value))
        return TargetSchema(
            schema_id=str(reporter.endpoint_schema_id),
            reporter_id=str(reporter_id),
            endpoint_ids=tuple(sorted(endpoint_ids)),
            target_modality=str(reporter.target_modality),
        )

    def load_inputs(self, ids: Sequence[str], representation: str) -> pd.DataFrame:
        if representation != self.input_schema().representation:
            raise ValueError(f"manifest exposes only {self.input_schema().representation!r} inputs")
        return self._load_optional("inputs", "input_cell_id", ids)

    def load_targets(self, reporter_id: str, ids: Sequence[str]) -> pd.DataFrame:
        table = self._load_optional("targets", "target_cell_id", ids)
        if "reporter_id" in table:
            table = table.loc[table.reporter_id.eq(reporter_id)]
        return table.copy()

    def controls(self, screen_id: str) -> pd.DataFrame:
        observations = self.observations()
        is_control = coerce_boolean_column(
            observations["is_control"], column="is_control", table="observations"
        )
        return observations.loc[observations.screen_id.eq(screen_id) & is_control].copy()

    def _load_optional(self, name: str, id_column: str, ids: Sequence[str]) -> pd.DataFrame:
        if name not in self._specs:
            raise ValueError(f"local manifest does not declare optional table {name!r}")
        table = _read_local_table(self._path(name))
        if id_column not in table:
            raise ValueError(f"{name} lacks {id_column}")
        return table.loc[table[id_column].isin(ids)].copy()
