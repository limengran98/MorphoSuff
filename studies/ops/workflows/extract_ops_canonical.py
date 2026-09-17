#!/usr/bin/env python3
"""Normalize declared OPS release tables into the public canonical schema.

This entry point intentionally does not guess a private file layout. A local,
untracked JSON manifest declares each upstream table, canonical-to-source
column mapping and dataset capabilities. The command writes validated CSV
tables, a portable adapter manifest and a checksummed provenance record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from measurement_sufficiency.schemas import coerce_boolean_column  # noqa: E402
from measurement_sufficiency.schemas import validate_canonical_tables


REQUIRED = ("observations", "reporters", "assays", "pairing")
OPTIONAL = ("endpoints", "inputs", "targets")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported upstream table format: {path.name}")


def resolve_source(manifest_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def normalize_bool(series: pd.Series, column: str) -> pd.Series:
    """Strict boolean parsing with the OPS control vocabulary.

    Delegates to the package-level coercer so the toolkit and this normalizer
    cannot drift apart on what counts as a boolean; the OPS-specific tokens are
    supplied as extras rather than baked into the shared implementation.
    """
    return coerce_boolean_column(
        series,
        column=column,
        table="upstream table",
        extra_true=("control", "ntc"),
        extra_false=("perturbation", "ko"),
    )


def normalize_endpoint_ids(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value]
    else:
        text = str(value).strip()
        if text.startswith("["):
            parsed = json.loads(text)
            if not isinstance(parsed, list):
                raise ValueError("endpoint_ids JSON must encode a list")
            items = [str(item).strip() for item in parsed]
        else:
            separator = ";" if ";" in text and "," not in text else ","
            items = [item.strip() for item in text.split(separator)]
    items = [item for item in items if item]
    if not items or len(items) != len(set(items)):
        raise ValueError("endpoint_ids must be a nonempty list of unique identifiers")
    return ",".join(items)


def transform_table(name: str, source: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    mapping = spec.get("columns", {})
    if not isinstance(mapping, dict):
        raise ValueError(f"tables.{name}.columns must be an object")
    missing_sources = sorted(set(mapping.values()).difference(source.columns))
    if missing_sources:
        raise ValueError(f"{name} lacks mapped source columns: {', '.join(missing_sources)}")
    table = source.rename(columns={upstream: canonical for canonical, upstream in mapping.items()})
    keep = list(mapping)
    constants = spec.get("constants", {})
    for column, value in constants.items():
        table[column] = value
        if column not in keep:
            keep.append(column)
    table = table.loc[:, keep].copy()
    if name == "observations" and "is_control" in table:
        table["is_control"] = normalize_bool(table["is_control"], "is_control")
    if name == "assays" and "endpoint_ids" in table:
        table["endpoint_ids"] = table["endpoint_ids"].map(normalize_endpoint_ids)
    if name == "pairing" and "pairing_confidence" in table:
        table["pairing_confidence"] = pd.to_numeric(table["pairing_confidence"], errors="raise")
    for column in table.columns:
        if column.endswith("_id") and column != "guide_id":
            table[column] = table[column].astype(str)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace canonical outputs already present in output-dir.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.upstream_manifest.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = payload.get("tables", {})
    missing = sorted(set(REQUIRED).difference(specs))
    if missing:
        raise SystemExit(f"upstream manifest lacks required tables: {', '.join(missing)}")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    provenance_rows: list[dict[str, Any]] = []
    for name in (*REQUIRED, *OPTIONAL):
        if name not in specs:
            continue
        spec = specs[name]
        source_path = resolve_source(manifest_path, spec["path"])
        if not source_path.is_file():
            raise SystemExit(f"missing upstream table {name}: {source_path}")
        destination = output / f"{name}.csv"
        if destination.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {destination}; pass --overwrite")
        table = transform_table(name, read_table(source_path), spec)
        table.to_csv(destination, index=False)
        tables[name] = table
        provenance_rows.append(
            {
                "logical_table": name,
                "declared_source": str(spec["path"]),
                "source_sha256": sha256(source_path),
                "source_rows": int(len(table)),
                "canonical_file": destination.name,
                "canonical_sha256": sha256(destination),
            }
        )

    validate_canonical_tables(tables)
    adapter_manifest = {
        "schema_version": 1,
        "dataset_id": str(payload.get("dataset_id", "ops")),
        "capabilities": payload.get("capabilities", {}),
        "input_schema": payload.get("input_schema", {}),
        "tables": {name: f"{name}.csv" for name in tables},
    }
    adapter_path = output / "canonical_manifest.json"
    adapter_path.write_text(json.dumps(adapter_manifest, indent=2, sort_keys=True) + "\n")
    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream_manifest_sha256": sha256(manifest_path),
        "tables": provenance_rows,
        "validation": "measurement_sufficiency.schemas.validate_canonical_tables:PASS",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(tables)} validated canonical tables to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
