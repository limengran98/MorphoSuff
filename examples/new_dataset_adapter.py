#!/usr/bin/env python3
"""Minimal preflight for a new dataset adapter using canonical CSV tables."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED = {
    "observations": {"observation_id", "input_cell_id", "target_cell_id", "assay_id", "perturbation_id", "guide_id", "screen_id", "well_id", "field_id", "is_control"},
    "reporters": {"reporter_id", "reporter_name", "biological_system", "target_modality", "endpoint_schema_id"},
    "assays": {"assay_id", "reporter_id", "screen_id", "endpoint_ids"},
    "pairing": {"observation_id", "input_cell_id", "target_cell_id", "assay_id", "pairing_key", "pairing_confidence"},
}


def headers(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return set(next(csv.reader(handle)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for table in REQUIRED:
        parser.add_argument(f"--{table}", type=Path, required=True)
    args = parser.parse_args()
    for table, required in REQUIRED.items():
        missing = required - headers(getattr(args, table))
        if missing:
            raise SystemExit(f"{table}: missing {sorted(missing)}")
    print("Canonical table headers are compatible. Next: validate assay FKs, assay-local pairing cardinality, and capabilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
