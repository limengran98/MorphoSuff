#!/usr/bin/env python3
"""Run a raw-image plan with resumable outputs and optional GPU concurrency."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

from common import atomic_json


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="", help="comma-separated GPU IDs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def complete(job: dict) -> bool:
    path = Path(job["output_dir"]) / "training_result.json"
    if not path.is_file():
        return False
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return result.get("status") == "PASS" and result.get("completed") is True


def main() -> None:
    args = arguments()
    if args.workers <= 0 or args.max_jobs < 0:
        raise ValueError("workers must be positive and max-jobs non-negative")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    jobs = [job for job in plan["jobs"] if not complete(job)]
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if args.dry_run:
        print(json.dumps({"status": "DRY_RUN_PASS", "pending": len(jobs), "jobs": [job["job_id"] for job in jobs]}, indent=2))
        return
    args.run_root.mkdir(parents=True, exist_ok=True)
    status_path = args.run_root / "queue_status.json"
    started = time.time()
    failures: list[dict] = []

    def run(item: tuple[int, dict]) -> dict:
        index, job = item
        log = args.run_root / "logs" / f"{job['job_id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        Path(job["output_dir"]).mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        if gpu_ids:
            environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % len(gpu_ids)]
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(job["command"], env=environment, stdout=handle, stderr=subprocess.STDOUT, check=False)
        returncode = result.returncode
        if returncode == 0 and not complete(job):
            returncode = 2
        return {"job_id": job["job_id"], "returncode": returncode, "log": str(log)}

    completed = 0
    atomic_json(status_path, {"status": "RUNNING", "pending": len(jobs), "completed": 0, "failed": 0})
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, item): item[1] for item in enumerate(jobs)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += result["returncode"] == 0
            if result["returncode"] != 0:
                failures.append(result)
                if not args.continue_on_error:
                    for pending in futures:
                        pending.cancel()
                    break
            atomic_json(status_path, {
                "status": "RUNNING", "pending": len(jobs) - completed - len(failures),
                "completed": completed, "failed": len(failures), "failures": failures,
                "elapsed_seconds": time.time() - started,
            })
    final = {
        "status": "PASS" if not failures else "COMPLETE_WITH_FAILURES",
        "planned": len(jobs), "completed": completed, "failed": len(failures),
        "failures": failures, "elapsed_seconds": time.time() - started,
    }
    atomic_json(status_path, final)
    print(json.dumps(final, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
