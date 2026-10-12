#!/usr/bin/env python3
"""Static/import smoke test for the released low-label implementation closure."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "legacy_exact" / "scripts"
ENTRYPOINTS = (
    "build_ops_low_label_sampling_manifest.py",
    "run_ops_low_label_classical_task.py",
    "run_ops_low_label_transfer_fold0.py",
    "run_ops_low_label_generic_task.py",
    "run_ops_low_label_biological_task.py",
    "run_ops_low_label_scipenn_task.py",
    "run_ops_low_label_scpair_reporter_task.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true",
                        help="also invoke each entry point with --help")
    args = parser.parse_args()
    failures: list[str] = []
    for path in sorted(SCRIPTS.glob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as error:  # pragma: no cover - diagnostic path
            failures.append(f"AST {path.name}: {error}")
    for name in ENTRYPOINTS:
        if not (SCRIPTS / name).is_file():
            failures.append(f"missing entry point: {name}")
        elif args.imports:
            completed = subprocess.run(
                [sys.executable, str(SCRIPTS / name), "--help"],
                cwd=HERE / "legacy_exact", capture_output=True, text=True,
            )
            if completed.returncode:
                failures.append(f"import/help {name}: {completed.stderr[-500:]}")
    absolute_hits = []
    for path in SCRIPTS.glob("*.py"):
        if "L202500483" in path.read_text(encoding="utf-8"):
            absolute_hits.append(path.name)
    if absolute_hits:
        failures.append(f"machine-bound paths remain: {absolute_hits}")
    print(json.dumps({
        "status": "PASS" if not failures else "FAIL",
        "python_files": len(list(SCRIPTS.glob('*.py'))),
        "entrypoints": list(ENTRYPOINTS),
        "failures": failures,
    }, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
