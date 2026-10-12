"""Small CLI helpers shared by the public OPS workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path)


def write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        table.to_parquet(path, index=False)
    else:
        table.to_csv(path, index=False)


def table_paths(manifest: Path) -> dict[str, Path]:
    payload = json.loads(manifest.read_text())
    specs = payload.get("tables", payload)
    paths: dict[str, Path] = {}
    for name, value in specs.items():
        value = value.get("local_path") if isinstance(value, dict) else value
        if value:
            path = Path(value)
            paths[name] = path if path.is_absolute() else manifest.parent / path
    return paths
