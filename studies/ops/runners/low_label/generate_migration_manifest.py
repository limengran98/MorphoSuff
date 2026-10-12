#!/usr/bin/env python3
"""Regenerate checksums for the migrated low-label implementation closure."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-scripts", type=Path, required=True,
                        help="read-only MM/scripts tree used for provenance comparison")
    parser.add_argument("--output", type=Path, default=HERE / "MIGRATION_MANIFEST.tsv")
    args = parser.parse_args()
    released = HERE / "legacy_exact" / "scripts"
    rows = []
    for current in sorted(released.glob("*.py")):
        historical = args.historical_scripts / current.name
        rows.append({
            "logical_source": f"MM/scripts/{current.name}",
            "historical_sha256": sha256(historical) if historical.is_file() else "NOT_FOUND",
            "released_sha256": sha256(current),
            "byte_identical": str(historical.is_file() and historical.read_bytes() == current.read_bytes()).lower(),
            "role": "entrypoint_or_local_dependency",
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=tuple(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} migration records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
