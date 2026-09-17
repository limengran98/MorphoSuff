#!/usr/bin/env python3
"""Create a portable OPS derived-data manifest from validated local assets."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_TABLES = {"observations", "reporters", "assays", "pairing"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True,
                        help="untracked JSON manifest with table local_path entries")
    parser.add_argument("--output", type=Path, required=True,
                        help="portable JSON manifest to keep outside source control")
    args = parser.parse_args()
    supplied = json.loads(args.input_manifest.read_text())
    tables = supplied.get("tables", {})
    missing = REQUIRED_TABLES - set(tables)
    if missing:
        raise SystemExit(f"missing required tables: {sorted(missing)}")
    portable = {"schema_version": 1, "dataset": "OPS", "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                "inventory": {"biological_systems": 10, "reporters": 52, "assays": 99, "physical_screens": 73},
                "tables": {}}
    for logical_name, spec in tables.items():
        local = Path(spec["local_path"])
        if not local.is_file():
            raise SystemExit(f"missing local asset for {logical_name}")
        portable["tables"][logical_name] = {
            "logical_name": logical_name,
            "sha256": file_sha256(local),
            "bytes": local.stat().st_size,
            "schema_id": spec.get("schema_id", "unspecified"),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(portable, indent=2, sort_keys=True) + "\n")
    print(f"wrote portable manifest with {len(portable['tables'])} tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
