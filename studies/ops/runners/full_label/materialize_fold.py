#!/usr/bin/env python3
"""Materialize one runnable combined fold from per-reporter canonical bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def write(frame: pd.DataFrame, path: Path, format_: str) -> None:
    if format_ == "parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-root", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=("field", "gene", "strict_whole_screen"))
    parser.add_argument("--fold", required=True, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--strict-screen-plan", type=Path)
    parser.add_argument("--format", choices=("csv", "parquet"), default="parquet")
    parser.add_argument("--reporters", default="all")
    args = parser.parse_args()
    root = args.canonical_root.resolve()
    release = json.loads((root / "canonical_release.json").read_text(encoding="utf-8"))
    requested = None if args.reporters.casefold() == "all" else set(value.strip() for value in args.reporters.split(",") if value.strip())
    strict = None
    if args.split == "strict_whole_screen":
        if args.strict_screen_plan is None:
            raise ValueError("strict_whole_screen requires --strict-screen-plan")
        plan = json.loads(args.strict_screen_plan.read_text(encoding="utf-8"))
        rows = [row for row in plan["assignments"] if int(row["fold"]) == args.fold]
        if len(rows) != 1:
            raise ValueError(f"strict plan contains {len(rows)} assignment records for fold {args.fold}")
        strict = pd.read_csv(args.strict_screen_plan.parent / rows[0]["path"])
        strict["observation_id"] = strict.observation_id.astype(str)
    input_parts, target_parts, assignment_parts = [], [], []
    for relative in release["reporter_manifests"]:
        manifest_path = root / relative
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        reporter = str(manifest["reporter_id"])
        if requested is not None and reporter not in requested:
            continue
        bundle = manifest_path.parent
        inputs = read(bundle / manifest["tables"]["inputs"])
        targets = read(bundle / manifest["tables"]["targets"])
        inputs["observation_id"] = inputs.observation_id.astype(str)
        targets["observation_id"] = targets.observation_id.astype(str)
        if strict is None:
            assignments = read(bundle / manifest["splits"][args.split])
            assignments = assignments.loc[assignments.fold.astype(int).eq(args.fold)].copy()
        else:
            assignments = strict.loc[strict.reporter_id.astype(str).eq(reporter)].copy()
        assignments["observation_id"] = assignments.observation_id.astype(str)
        keep = set(assignments.observation_id)
        input_parts.append(inputs.loc[inputs.observation_id.isin(keep)].copy())
        target_parts.append(targets.loc[targets.observation_id.isin(keep)].copy())
        # Carry any split flags the producer emitted. Hard-coding three columns
        # here is what stopped is_control and scored from reaching the runners.
        assignment_columns = ["observation_id", "fold", "role"] + [
            column for column in ("is_control", "scored") if column in assignments
        ]
        assignment_parts.append(assignments[assignment_columns].copy())
    if not input_parts:
        raise RuntimeError("no reporter bundles matched")
    inputs = pd.concat(input_parts, ignore_index=True)
    targets = pd.concat(target_parts, ignore_index=True)
    assignments = pd.concat(assignment_parts, ignore_index=True)
    if inputs.observation_id.duplicated().any() or assignments.observation_id.duplicated().any():
        raise RuntimeError("combined fold has duplicate observation IDs")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    suffix = args.format
    paths = {name: output / f"{name}.{suffix}" for name in ("inputs", "targets", "assignments")}
    write(inputs, paths["inputs"], suffix)
    write(targets, paths["targets"], suffix)
    write(assignments, paths["assignments"], suffix)
    manifest = {
        "schema_version": "measurement-sufficiency-materialized-fold-v1",
        "split_name": args.split,
        "fold": args.fold,
        "reporters": sorted(inputs.reporter_id.astype(str).unique()),
        "n_observations": len(inputs),
        "n_target_rows": len(targets),
        "n_train": int(assignments.role.eq("train").sum()),
        "n_validation": int(assignments.role.eq("validation").sum()),
        "n_test": int(assignments.role.eq("test").sum()),
        "paths": {key: str(path.resolve()) for key, path in paths.items()},
    }
    (output / "fold_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
