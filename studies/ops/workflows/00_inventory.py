#!/usr/bin/env python3
"""Inventory a local OPS canonical manifest without downloading any asset."""
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
    data = OPSAdapter(args.manifest)
    observations, reporters, assays = data.observations(), data.reporters(), data.assays()
    inventory = {"dataset": "OPS", "observations": len(observations), "reporters": int(reporters.reporter_id.nunique()), "assays": int(assays.assay_id.nunique()), "physical_screens": int(observations.screen_id.nunique())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    return 0
if __name__ == "__main__": raise SystemExit(main())
