#!/usr/bin/env python3
"""Calculate declared replicate and guide-half stability tables."""
from __future__ import annotations
import argparse
from pathlib import Path
from _common import read_table, write_table
from measurement_sufficiency.analysis.stability import stability_summary

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=Path, required=True)
    parser.add_argument("--guide-halves", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    halves = read_table(args.guide_halves) if args.guide_halves else None
    write_table(stability_summary(read_table(args.replicates), halves), args.output)
    return 0
if __name__ == "__main__": raise SystemExit(main())
