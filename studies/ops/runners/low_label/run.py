#!/usr/bin/env python3
"""Portable dispatcher for the frozen OPS low-label task implementations."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
LEGACY_ROOT = HERE / "legacy_exact"
SCRIPTS = LEGACY_ROOT / "scripts"
CONFIGS = LEGACY_ROOT / "configs"
REGISTRY = json.loads((HERE / "methods.json").read_text(encoding="utf-8"))["methods"]
COMMON_KEYS = (
    "phase_cache",
    "exact_root",
    "target_table",
    "target_feature_dictionary",
    "specialist_reference_root",
    "preprocessing_reference_root",
)
SCPAIR_MODEL_SOURCE = (
    LEGACY_ROOT / "external" / "original_methods" / "scpair" / "scpair" / "model.py"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--assets", type=Path, required=True,
                        help="untracked JSON using assets.example.json")
    result.add_argument("--method", choices=tuple(REGISTRY), required=True)
    result.add_argument("--fraction", type=float, required=True)
    result.add_argument("--fold", type=int, choices=range(5), required=True)
    result.add_argument("--output-root", type=Path, required=True)
    result.add_argument("--sampling-root", type=Path, required=True)
    result.add_argument("--reporter", help="required for reporter-specific scPair")
    result.add_argument("--seed", type=int, default=20260726)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--dry-run", action="store_true",
                        help="validate dispatch and print the exact command without training")
    result.add_argument("--smoke", action="store_true",
                        help="run the model-native end-to-end smoke where supported")
    result.add_argument("extra", nargs=argparse.REMAINDER,
                        help="additional runner arguments after --")
    return result


def load_assets(path: Path) -> dict[str, Path]:
    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit("--assets must contain one JSON object")
    result: dict[str, Path] = {}
    for key, raw in payload.items():
        if raw in {None, ""}:
            continue
        value = Path(str(raw)).expanduser()
        result[key] = value if value.is_absolute() else manifest_path.parent / value
    missing = [key for key in COMMON_KEYS if key not in result]
    if missing:
        raise SystemExit(f"assets manifest lacks required keys: {missing}")
    return result


def common_args(assets: dict[str, Path]) -> list[str]:
    result: list[str] = []
    for key in COMMON_KEYS:
        result.extend((f"--{key.replace('_', '-')}", str(assets[key].resolve())))
    return result


def method_requirements(
    args: argparse.Namespace, assets: dict[str, Path]
) -> tuple[list[str], list[Path]]:
    """Return only configuration keys and paths used by this invocation."""

    missing_keys: list[str] = []
    paths = [assets[key] for key in COMMON_KEYS]
    paths.append(args.sampling_root.expanduser().resolve())
    if args.method == "tabm":
        if "tabm_source" not in assets:
            missing_keys.append("tabm_source")
        else:
            paths.append(assets["tabm_source"])
        # The wheel is optional when the exact package version is installed.
        if "tabm_dependency_wheel" in assets:
            paths.append(assets["tabm_dependency_wheel"])
    if args.method == "scpair":
        paths.append(SCPAIR_MODEL_SOURCE)
    return missing_keys, paths


def missing_requirements(
    args: argparse.Namespace, assets: dict[str, Path]
) -> list[str]:
    missing_keys, paths = method_requirements(args, assets)
    missing = [f"assets:{key}" for key in missing_keys]
    missing.extend(str(path) for path in paths if not path.exists())
    return missing


def command(args: argparse.Namespace, assets: dict[str, Path]) -> list[str]:
    method = args.method
    common = common_args(assets)
    ordinary = [
        "--label-fraction", str(args.fraction),
        "--fold", str(args.fold),
        "--split", "gene_holdout_main",
        "--seed", str(args.seed),
        "--output-root", str(args.output_root.expanduser().resolve()),
    ]
    if method in {"ridge", "knn", "gbdt", "catboost"}:
        return [
            sys.executable, str(SCRIPTS / "run_ops_low_label_classical_task.py"),
            "--method", REGISTRY[method]["runner_method"],
            *ordinary,
            "--sampling-manifest-root", str(args.sampling_root.expanduser().resolve()),
            "--device", "cpu" if method == "ridge" else "cuda",
            "--workers", str(args.workers),
            *common,
        ]
    if method == "mlp":
        cmd = [
            sys.executable, str(SCRIPTS / "run_ops_low_label_transfer_fold0.py"),
            "--model", "mlp", *ordinary, "--workers", str(args.workers), *common,
        ]
        if args.smoke:
            cmd.append("--smoke")
        return cmd
    if method in {"resmlp", "mmoe", "multitab", "tabm"}:
        cmd = [
            sys.executable, str(SCRIPTS / "run_ops_low_label_generic_task.py"),
            "--method", REGISTRY[method]["runner_method"],
            *ordinary,
            "--sampling-manifest-root", str(args.sampling_root.expanduser().resolve()),
            "--workers", str(args.workers),
            *common,
        ]
        if args.smoke:
            cmd.append("--end-to-end-smoke")
        return cmd
    if method in {"scbutterfly", "midas"}:
        cmd = [
            sys.executable, str(SCRIPTS / "run_ops_low_label_biological_task.py"),
            "--method", REGISTRY[method]["runner_method"],
            *ordinary,
            "--sampling-manifest-root", str(args.sampling_root.expanduser().resolve()),
            "--campaign-config", str(CONFIGS / "ops_biological_baseline_full52_formal_v1.json"),
            "--workers", str(args.workers),
            *common,
        ]
        if args.smoke:
            cmd.append("--end-to-end-smoke")
        return cmd
    if method == "scipenn":
        cmd = [
            sys.executable, str(SCRIPTS / "run_ops_low_label_scipenn_task.py"),
            "--method", "scipenn_ops", *ordinary,
            "--sampling-manifest-root", str(args.sampling_root.expanduser().resolve()),
            "--campaign-config", str(CONFIGS / "ops_scipenn_full52_formal_v1.json"),
            "--workers", str(args.workers),
            *common,
        ]
        if args.smoke:
            cmd.append("--end-to-end-smoke")
        return cmd
    if method == "scpair":
        if not args.reporter:
            raise SystemExit("--reporter is required for scPair")
        cmd = [
            sys.executable, str(SCRIPTS / "run_ops_low_label_scpair_reporter_task.py"),
            "--method", "scpair_ops", *ordinary,
            "--reporter", args.reporter,
            "--sampling-manifest-root", str(args.sampling_root.expanduser().resolve()),
            "--workers", str(args.workers),
            *common,
        ]
        if args.smoke:
            cmd.append("--development-smoke")
        return cmd
    raise AssertionError(method)


def main() -> int:
    args = parser().parse_args()
    if not 0.0 < args.fraction <= 1.0:
        raise SystemExit("--fraction must be in (0,1]")
    assets = load_assets(args.assets)
    cmd = command(args, assets)
    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]
    cmd.extend(extra)
    missing = missing_requirements(args, assets)
    payload: dict[str, Any] = {
        "status": "DRY_RUN" if args.dry_run else "STARTING",
        "method": args.method,
        "fraction": args.fraction,
        "fold": args.fold,
        "command": shlex.join(cmd),
        "missing_assets": missing,
    }
    print(json.dumps(payload, indent=2), flush=True)
    if args.dry_run:
        return 0
    if missing:
        raise SystemExit("execution stopped because assets are missing")
    args.output_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPTS) + os.pathsep + env.get("PYTHONPATH", "")
    env["OPS_LOW_LABEL_TARGET_TABLE"] = str(assets["target_table"].resolve())
    if "tabm_source" in assets:
        env["OPS_TABM_SOURCE"] = str(assets["tabm_source"].resolve())
    if "tabm_dependency_wheel" in assets:
        env["OPS_TABM_DEPENDENCY_WHEEL"] = str(
            assets["tabm_dependency_wheel"].resolve()
        )
    return subprocess.run(cmd, cwd=LEGACY_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
