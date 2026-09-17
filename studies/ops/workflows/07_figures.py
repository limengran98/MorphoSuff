#!/usr/bin/env python3
"""Render one or all OPS Figure 1–6 source-contract figure entry points."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=[str(i) for i in range(1, 7)], default=None)
    parser.add_argument("--tables-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    figures = [int(args.figure)] if args.figure else range(1, 7)
    root = Path(__file__).resolve().parents[1] / "figures"
    for number in figures:
        subprocess.run([sys.executable, str(root / f"figure{number}.py"), "--tables-manifest", str(args.tables_manifest), "--output-dir", str(args.output_dir)], check=True)
    return 0
if __name__ == "__main__": raise SystemExit(main())
