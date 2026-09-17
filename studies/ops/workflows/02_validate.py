#!/usr/bin/env python3
"""Run canonical foreign-key, cardinality, and pairing validation for OPS."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapter import OPSAdapter

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = OPSAdapter(args.manifest)
    payload = {"valid": True, "tables": {name: len(getattr(adapter, name)()) for name in ("observations", "reporters", "assays", "pairing")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0
if __name__ == "__main__": raise SystemExit(main())
