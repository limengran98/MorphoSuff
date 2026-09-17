"""Portable local-file adapter for the public OPS canonical-table contract.

It never downloads data and never embeds a cohort path.  Callers provide a JSON
manifest mapping logical table names to local CSV/Parquet files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from measurement_sufficiency.adapters import AdapterCapabilities
from measurement_sufficiency.schemas import coerce_boolean_column
from measurement_sufficiency.schemas import FeatureSchema, TargetSchema, validate_canonical_tables


_TABLES = ("observations", "reporters", "assays", "pairing")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


@dataclass
class OPSAdapter:
    """Adapter backed by a caller-supplied local manifest.

    Optional ``inputs`` and ``targets`` entries follow the same path contract.
    They are loaded lazily and are intentionally not required for analysis-only
    figure reproduction.
    """

    manifest_path: Path

    def __post_init__(self) -> None:
        self.manifest_path = Path(self.manifest_path)
        supplied = json.loads(self.manifest_path.read_text())
        self._spec = supplied.get("tables", supplied)
        missing = set(_TABLES).difference(self._spec)
        if missing:
            raise ValueError(f"OPS manifest lacks tables: {', '.join(sorted(missing))}")
        self._tables = {name: _read_table(self._path(name)) for name in _TABLES}
        if "endpoints" in self._spec:
            self._tables["endpoints"] = _read_table(self._path("endpoints"))
        validate_canonical_tables(self._tables)
        caps = supplied.get("capabilities", {})
        self.capabilities = AdapterCapabilities(
            exact_pairing=bool(caps.get("exact_pairing", True)),
            raw_images=bool(caps.get("raw_images", False)),
            guides=bool(caps.get("guides", True)),
            screen_matched_controls=bool(caps.get("screen_matched_controls", True)),
            repeated_screens=bool(caps.get("repeated_screens", False)),
            covariates=bool(caps.get("covariates", False)),
        )

    def _path(self, name: str) -> Path:
        value: Any = self._spec[name]
        value = value.get("local_path") if isinstance(value, dict) else value
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
        spec = self._spec.get("inputs", {})
        names = tuple(spec.get("feature_names", ()))
        return FeatureSchema(str(spec.get("schema_id", "ops_phase_unspecified")), names)

    def target_schema(self, reporter_id: str) -> TargetSchema:
        reporter = self._tables["reporters"].set_index("reporter_id").loc[reporter_id]
        assays = self._tables["assays"].query("reporter_id == @reporter_id")
        endpoint_ids: set[str] = set()
        for values in assays.endpoint_ids:
            endpoint_ids.update(_endpoints(values))
        return TargetSchema(str(reporter.endpoint_schema_id), reporter_id, tuple(sorted(endpoint_ids)), str(reporter.target_modality))

    def load_inputs(self, ids: Sequence[str], representation: str) -> pd.DataFrame:
        if representation != "tabular":
            raise ValueError("OPS adapter only exposes declared tabular inputs")
        return self._load_optional("inputs", "input_cell_id", ids)

    def load_targets(self, reporter_id: str, ids: Sequence[str]) -> pd.DataFrame:
        table = self._load_optional("targets", "target_cell_id", ids)
        return table[table.reporter_id.eq(reporter_id)].copy() if "reporter_id" in table else table

    def controls(self, screen_id: str) -> pd.DataFrame:
        observations = self.observations()
        is_control = coerce_boolean_column(
            observations["is_control"], column="is_control", table="observations"
        )
        return observations[observations.screen_id.eq(screen_id) & is_control].copy()

    def _load_optional(self, logical: str, id_column: str, ids: Sequence[str]) -> pd.DataFrame:
        if logical not in self._spec:
            raise ValueError(f"manifest does not declare optional {logical} table")
        table = _read_table(self._path(logical))
        if id_column not in table:
            raise ValueError(f"{logical} lacks {id_column}")
        return table[table[id_column].isin(ids)].copy()


def _endpoints(value: object) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError("assay endpoint_ids must be a comma-delimited string or sequence")
