#!/usr/bin/env python3
"""Unified command for every method in the frozen ten-method OPS benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


INTERNAL = {"ridge", "gbdt", "catboost", "mlp", "tabm", "resmlp", "multitab"}
EXTERNAL = {"scbutterfly", "midas", "scpair"}
METHODS = tuple(sorted(INTERNAL | EXTERNAL))
HERE = Path(__file__).resolve().parent
EXTERNAL_RUNNER = HERE.parent / "run_external_fold.py"
INTERNAL_RUNNER = HERE / "run_internal_fold.py"


def command(args: argparse.Namespace) -> list[str]:
    runner = INTERNAL_RUNNER if args.method in INTERNAL else EXTERNAL_RUNNER
    values = [
        sys.executable,
        str(runner),
        "--method",
        args.method,
        "--inputs",
        str(args.inputs),
        "--targets",
        str(args.targets),
        "--assignments",
        str(args.assignments),
        "--split-name",
        args.split_name,
        "--fold",
        str(args.fold),
        "--output-root",
        str(args.output_root),
        "--reporters",
        args.reporters,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.features:
        values += ["--features", args.features]
    if args.config:
        values += ["--config", str(args.config)]
    if args.smoke:
        values.append("--smoke")
    if args.contract_check_only:
        values.append("--contract-check-only")
    if args.method == "tabm":
        if args.tabm_source is None:
            raise ValueError("TabM requires --tabm-source")
        values += ["--tabm-source", str(args.tabm_source)]
        if args.allow_unpinned_tabm:
            values.append("--allow-unpinned-tabm")
    if args.method == "scpair":
        if args.scpair_source is None:
            raise ValueError("scPair requires --scpair-source")
        values += ["--scpair-source", str(args.scpair_source)]
        if args.allow_unpinned_scpair:
            values.append("--allow-unpinned-scpair")
    return values


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--method", required=True, choices=METHODS)
    output.add_argument("--inputs", required=True, type=Path)
    output.add_argument("--targets", required=True, type=Path)
    output.add_argument("--assignments", required=True, type=Path)
    output.add_argument("--split-name", required=True, choices=("field", "gene", "strict_whole_screen"))
    output.add_argument("--fold", required=True, type=int)
    output.add_argument("--output-root", required=True, type=Path)
    output.add_argument("--reporters", default="all")
    output.add_argument("--features")
    output.add_argument("--config", type=Path)
    output.add_argument("--device", default="cuda")
    output.add_argument("--seed", type=int, default=20260721)
    output.add_argument("--tabm-source", type=Path)
    output.add_argument("--scpair-source", type=Path)
    output.add_argument("--allow-unpinned-tabm", action="store_true")
    output.add_argument("--allow-unpinned-scpair", action="store_true")
    output.add_argument("--smoke", action="store_true")
    output.add_argument("--contract-check-only", action="store_true")
    output.add_argument("--print-command", action="store_true", help="print the exact child command without executing it")
    return output


def main() -> int:
    args = parser().parse_args()
    values = command(args)
    if args.print_command:
        print(json.dumps({"method": args.method, "command": values}, indent=2))
        return 0
    return subprocess.run(values, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

