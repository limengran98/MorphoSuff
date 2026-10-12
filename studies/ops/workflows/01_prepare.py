#!/usr/bin/env python3
"""Validate and record a caller-provided OPS local canonical manifest."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import OPSAdapter
from _common import table_paths

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OPSAdapter(args.manifest)  # validates all canonical contracts before provenance is emitted
    paths = table_paths(args.manifest)
    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    payload = {"dataset": "OPS", "canonical_tables": {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in paths.items() if name in {"observations", "reporters", "assays", "pairing"}}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0
if __name__ == "__main__": raise SystemExit(main())
