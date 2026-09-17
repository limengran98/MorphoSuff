#!/usr/bin/env python3
"""Build the shared OPS Target12 low-label manifest from caller-supplied assets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
LEGACY_ROOT = HERE / "legacy_exact"
SCRIPT = LEGACY_ROOT / "scripts" / "build_ops_low_label_sampling_manifest.py"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fractions", default="0.001,0.01,0.2,1.0")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    supplied = json.loads(args.assets.expanduser().resolve().read_text())
    required = ("phase_cache", "exact_root", "target_table", "target_feature_dictionary")
    missing_keys = [name for name in required if name not in supplied]
    if missing_keys:
        raise SystemExit(f"assets manifest lacks keys: {missing_keys}")
    cmd = [
        sys.executable, str(SCRIPT),
        "--output-root", str(args.output_root.expanduser().resolve()),
        "--fractions", args.fractions,
        "--folds", args.folds,
        "--seed", str(args.seed),
    ]
    for key in required:
        cmd.extend((f"--{key.replace('_', '-')}", str(Path(supplied[key]).expanduser().resolve())))
    print(json.dumps({"status": "DRY_RUN" if args.dry_run else "STARTING", "command": cmd}, indent=2))
    if args.dry_run:
        return 0
    env = os.environ.copy()
    scripts = LEGACY_ROOT / "scripts"
    env["PYTHONPATH"] = str(scripts) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(cmd, cwd=LEGACY_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
