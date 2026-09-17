#!/usr/bin/env python3
"""Build a portable reporter-by-fold raw-image training plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import atomic_json


HERE = Path(__file__).resolve().parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("dinov2-head", "cytoland-head", "cytoland-finetune"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output-plan", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def require(assets: dict, *names: str) -> None:
    missing = [name for name in names if not str(assets.get(name, "")).strip()]
    if missing:
        raise ValueError(f"Assets manifest lacks required entries: {missing}")


def main() -> None:
    args = arguments()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    asset_base = args.assets.resolve().parent

    def asset(name: str) -> str:
        value = Path(str(assets[name])).expanduser()
        return str(value.resolve() if value.is_absolute() else (asset_base / value).resolve())

    reporters = list(map(str, config["reporters"]))
    folds = list(map(int, config["evaluation"]["folds"]))
    require(assets, "phase_cache", "exact_cache_root", "target_feature_dictionary")
    jobs = []
    for reporter in reporters:
        for fold in folds:
            output = args.result_root.resolve() / reporter / f"fold_{fold}"
            if args.mode.endswith("head"):
                embedding_key = "dinov2_embedding_root" if args.mode.startswith("dinov2") else "cytoland_embedding_root"
                require(assets, embedding_key)
                command = [
                    args.python,
                    str(HERE / "train_frozen_head.py"),
                    "--reporter", reporter,
                    "--fold", str(fold),
                    "--embedding-root", asset(embedding_key),
                    "--output-dir", str(output),
                    "--phase-cache", asset("phase_cache"),
                    "--exact-cache-root", asset("exact_cache_root"),
                    "--target-feature-dictionary", asset("target_feature_dictionary"),
                ]
            else:
                require(assets, "stage_b_root", "stage_b_cells", "viscy_repository", "cytoland_checkpoint")
                command = [
                    args.python,
                    str(HERE / "finetune_cytoland.py"),
                    "--reporter", reporter,
                    "--fold", str(fold),
                    "--output-dir", str(output),
                    "--config", str(args.config.resolve()),
                    "--stage-b-root", asset("stage_b_root"),
                    "--stage-b-plan", asset("stage_b_cells"),
                    "--viscy-repo", asset("viscy_repository"),
                    "--checkpoint", asset("cytoland_checkpoint"),
                    "--phase-cache", asset("phase_cache"),
                    "--exact-cache-root", asset("exact_cache_root"),
                    "--target-feature-dictionary", asset("target_feature_dictionary"),
                ]
            jobs.append({
                "job_id": f"{args.mode}__{reporter}__fold{fold}",
                "reporter": reporter,
                "fold": fold,
                "output_dir": str(output),
                "command": command,
            })
    plan = {
        "schema_version": "measurement-sufficiency-raw-image-plan-v1",
        "mode": args.mode,
        "config": str(args.config.resolve()),
        "assets": str(args.assets.resolve()),
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    atomic_json(args.output_plan, plan)
    print(f"Wrote {len(jobs)} jobs to {args.output_plan}")


if __name__ == "__main__":
    main()
