#!/usr/bin/env python3
"""Download processed OPS data and run a Figure 2 benchmark fold.

The command is intentionally explicit: it validates the compact Hugging Face
release, materializes canonical per-reporter tables, freezes the selected
holdout and invokes the same ten-method facade used by the study.  `--dry-run`
prints every resolved command without downloading, preparing or fitting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
VALIDATOR = REPO / "studies/ops/preparation/validate_processed_release.py"
EXPORT = REPO / "studies/ops/preparation/export_canonical.py"
STRICT = REPO / "studies/ops/runners/full_label/build_strict_screen.py"
MATERIALIZE = REPO / "studies/ops/runners/full_label/materialize_fold.py"
RUNNER = REPO / "studies/ops/runners/full_label/run.py"
REGISTRY = REPO / "configs/ops/data/reporter_registry.csv"
METHODS = ("ridge", "gbdt", "catboost", "mlp", "tabm", "scbutterfly", "resmlp", "multitab", "midas", "scpair")


def execute(command: list[str], dry_run: bool) -> None:
    print(json.dumps({"command": command}, indent=2), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def acquire(args: argparse.Namespace) -> Path:
    if args.dataset_root is not None:
        return args.dataset_root.expanduser().resolve()
    if args.dry_run:
        return (args.download_root / "<huggingface-snapshot>").resolve()
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install measurement-sufficiency[release-data] or pass --dataset-root") from exc
    return Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.download_root,
            allow_patterns=["processed_ops/**", "MANIFEST.tsv", "README.md"],
        )
    ).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="Amanda1998/MorphoSuff")
    parser.add_argument("--dataset-root", type=Path, help="existing Hugging Face snapshot; otherwise download")
    parser.add_argument("--download-root", type=Path, default=Path("data/MorphoSuff"))
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--reporters", default="all", help="all or comma-separated reporter slugs")
    parser.add_argument("--split", choices=("field", "gene", "strict_whole_screen"), default="gene")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--method", choices=METHODS, default="ridge")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        help="CSV works in the minimal environment; Parquet is recommended for all-52 runs",
    )
    parser.add_argument("--tabm-source", type=Path)
    parser.add_argument("--scpair-source", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--contract-check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.fold not in range(5):
        raise ValueError("--fold must be in [0, 4]")
    if args.method == "tabm" and args.tabm_source is None and not args.prepare_only:
        raise ValueError("TabM requires --tabm-source")
    if args.method == "scpair" and args.scpair_source is None and not args.prepare_only:
        raise ValueError("scPair requires --scpair-source")
    dataset = acquire(args)
    workspace = args.workspace.expanduser().resolve()
    canonical = workspace / "canonical"
    reporter_args: list[str] = []
    if args.reporters.casefold() != "all":
        for reporter in (value.strip() for value in args.reporters.split(",")):
            if reporter:
                reporter_args += ["--reporter", reporter]
    execute([sys.executable, str(VALIDATOR), "--dataset-root", str(dataset)], args.dry_run)
    if not (canonical / "canonical_release.json").is_file():
        execute(
            [
                sys.executable,
                str(EXPORT),
                "--exact-cache-root",
                str(dataset / "processed_ops/exact_reporters"),
                "--phase-cache",
                str(dataset / "processed_ops/phase172"),
                "--reporter-registry",
                str(REGISTRY),
                "--output-dir",
                str(canonical),
                "--format",
                args.format,
                *reporter_args,
            ],
            args.dry_run,
        )
    if args.prepare_only:
        return 0
    strict_root = workspace / "strict_screen"
    strict_args: list[str] = []
    if args.split == "strict_whole_screen":
        if not (strict_root / "strict_screen_plan.json").is_file():
            execute(
                [sys.executable, str(STRICT), "--canonical-root", str(canonical), "--output-dir", str(strict_root)],
                args.dry_run,
            )
        strict_args = ["--strict-screen-plan", str(strict_root / "strict_screen_plan.json")]
    fold_root = workspace / "folds" / args.split / f"fold_{args.fold}"
    execute(
        [
            sys.executable,
            str(MATERIALIZE),
            "--canonical-root",
            str(canonical),
            "--split",
            args.split,
            "--fold",
            str(args.fold),
            "--output-dir",
            str(fold_root),
            "--format",
            args.format,
            "--reporters",
            args.reporters,
            *strict_args,
        ],
        args.dry_run,
    )
    suffix = args.format
    command = [
        sys.executable,
        str(RUNNER),
        "--method",
        args.method,
        "--inputs",
        str(fold_root / f"inputs.{suffix}"),
        "--targets",
        str(fold_root / f"targets.{suffix}"),
        "--assignments",
        str(fold_root / f"assignments.{suffix}"),
        "--split-name",
        args.split,
        "--fold",
        str(args.fold),
        "--output-root",
        str(workspace / "runs" / args.method / args.split / f"fold_{args.fold}"),
        "--device",
        args.device,
    ]
    if args.smoke:
        command.append("--smoke")
    if args.contract_check_only:
        command.append("--contract-check-only")
    if args.tabm_source is not None:
        command += ["--tabm-source", str(args.tabm_source)]
    if args.scpair_source is not None:
        command += ["--scpair-source", str(args.scpair_source)]
    execute(command, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
