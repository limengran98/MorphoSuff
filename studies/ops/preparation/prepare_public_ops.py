#!/usr/bin/env python3
"""Run the public OPS H5AD-to-training preparation chain.

The command is resumable at each stage and prints every child command before
execution. It accepts any caller-selected data and output directories; no
machine-specific path is embedded in the released workflow.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad-dir", type=Path, required=True, help="directory containing all_cells_phase.h5ad and all_cells_fluor_*.h5ad")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reporter-registry", type=Path, required=True)
    parser.add_argument("--phase-name", default="all_cells_phase.h5ad")
    parser.add_argument("--stage", choices=("all", "exact", "phase172", "canonical"), default="all")
    parser.add_argument("--format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--reporter", action="append", default=[])
    parser.add_argument("--chunk-rows", type=int, default=8192)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool) -> None:
    print("$ " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    h5ad_dir = args.h5ad_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    phase = h5ad_dir / args.phase_name
    if not phase.is_file():
        raise FileNotFoundError(phase)
    reporter_paths = sorted(h5ad_dir.glob("all_cells_fluor_*.h5ad"))
    if args.reporter:
        selected = set(args.reporter)
        reporter_paths = [
            path
            for path in reporter_paths
            if path.stem.removeprefix("all_cells_fluor_") in selected
        ]
    if not reporter_paths:
        raise FileNotFoundError(f"no public reporter H5AD files under {h5ad_dir}")
    plan = {
        "phase": str(phase),
        "n_reporters": len(reporter_paths),
        "stages": [args.stage] if args.stage != "all" else ["exact", "phase172", "canonical"],
        "exact_cache_root": str(output / "exact" / "reporters"),
        "phase172_root": str(output / "phase172"),
        "canonical_root": str(output / "canonical"),
    }
    print(json.dumps(plan, indent=2), flush=True)
    stages = {args.stage} if args.stage != "all" else {"exact", "phase172", "canonical"}
    python = sys.executable
    if "exact" in stages:
        command = [
            python,
            str(root / "build_exact_caches.py"),
            "--phase",
            str(phase),
            "--output-root",
            str(output / "exact" / "reporters"),
        ]
        for path in reporter_paths:
            command.extend(["--reporter", str(path)])
        if args.overwrite:
            command.append("--overwrite")
        run(command, dry_run=args.dry_run)
    if "phase172" in stages:
        command = [
            python,
            str(root / "build_phase172.py"),
            "--phase",
            str(phase),
            "--exact-cache-root",
            str(output / "exact" / "reporters"),
            "--output-dir",
            str(output / "phase172"),
            "--chunk-rows",
            str(args.chunk_rows),
        ]
        if args.overwrite:
            command.append("--overwrite")
        run(command, dry_run=args.dry_run)
    if "canonical" in stages:
        command = [
            python,
            str(root / "export_canonical.py"),
            "--exact-cache-root",
            str(output / "exact" / "reporters"),
            "--phase-cache",
            str(output / "phase172"),
            "--reporter-registry",
            str(args.reporter_registry.expanduser().resolve()),
            "--output-dir",
            str(output / "canonical"),
            "--format",
            args.format,
            "--chunk-rows",
            str(args.chunk_rows),
        ]
        for reporter in args.reporter:
            command.extend(["--reporter", reporter])
        run(command, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
