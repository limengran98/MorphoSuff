#!/usr/bin/env python3
"""Build the portable 52-reporter, five-fold, six-arm OPS MLP plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import (
    ARM_TO_RUNNER,
    LEGACY_RUNNER,
    N_FOLDS,
    PRIMARY_MODEL,
    SCHEMA,
    atomic_json,
    load_reporters,
    parse_selection,
    validate_assets,
    validate_plan,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-cache", type=Path, required=True)
    parser.add_argument("--exact-cache-root", type=Path, required=True)
    parser.add_argument("--target-table", type=Path, required=True)
    parser.add_argument("--target-feature-dictionary", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--reporters", default="all52", help="all52 or comma-separated slugs")
    parser.add_argument("--folds", default="all", help="all or comma-separated values from 0 to 4")
    parser.add_argument("--arms", default="all", help="all or comma-separated public arm names")
    parser.add_argument("--model", choices=(PRIMARY_MODEL,), default=PRIMARY_MODEL)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-asset-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build(args: argparse.Namespace) -> dict:
    all_reporters = load_reporters(args.target_table)
    reporters = parse_selection(args.reporters, all_reporters, label="reporter")
    folds = [
        int(value)
        for value in parse_selection(args.folds, map(str, range(N_FOLDS)), label="fold")
    ]
    arms = parse_selection(args.arms, ARM_TO_RUNNER, label="arm")
    if not args.skip_asset_check:
        validate_assets(
            phase_cache=args.phase_cache,
            exact_cache_root=args.exact_cache_root,
            target_table=args.target_table,
            target_feature_dictionary=args.target_feature_dictionary,
            reporters=reporters,
        )
    jobs = []
    for reporter in reporters:
        for fold in folds:
            for arm in arms:
                runner_arm = ARM_TO_RUNNER[arm]
                output = args.result_root.resolve() / reporter / f"fold_{fold}" / arm
                command = [
                    args.python,
                    str(LEGACY_RUNNER),
                    "--reporter", reporter,
                    "--fold", str(fold),
                    "--model", args.model,
                    "--arm", runner_arm,
                    "--output-dir", str(output),
                    "--phase-cache", str(args.phase_cache.resolve()),
                    "--exact-cache-root", str(args.exact_cache_root.resolve()),
                    "--target-table", str(args.target_table.resolve()),
                    "--target-feature-dictionary", str(args.target_feature_dictionary.resolve()),
                ]
                jobs.append({
                    "job_id": f"{args.model}__{reporter}__fold{fold}__{arm}",
                    "model": args.model,
                    "reporter": reporter,
                    "fold": fold,
                    "arm": arm,
                    "runner_arm": runner_arm,
                    "output_dir": str(output),
                    "command": command,
                })
    plan = {
        "schema_version": SCHEMA,
        "scientific_contract": {
            "primary_estimator": PRIMARY_MODEL,
            "split": "gene_holdout_main",
            "reporter_statistical_unit": True,
            "reporters": len(reporters),
            "folds": folds,
            "arms": arms,
        },
        "assets": {
            "phase_cache": str(args.phase_cache.resolve()),
            "exact_cache_root": str(args.exact_cache_root.resolve()),
            "target_table": str(args.target_table.resolve()),
            "target_feature_dictionary": str(args.target_feature_dictionary.resolve()),
        },
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    validate_plan(
        plan,
        require_full_primary=(
            len(reporters) == 52 and len(folds) == 5 and len(arms) == 6
        ),
    )
    return plan


def main() -> None:
    args = arguments()
    plan = build(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    atomic_json(args.output_plan, plan)
    print(json.dumps({
        "status": "PLAN_WRITTEN",
        "plan": str(args.output_plan),
        "jobs": plan["n_jobs"],
    }, indent=2))


if __name__ == "__main__":
    main()

