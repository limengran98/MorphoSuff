#!/usr/bin/env python3
"""Execute a same-cell plan with resumable outputs and GPU assignment."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

from common import atomic_json, result_complete, validate_plan


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="0", help="comma-separated visible GPU IDs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--require-full-primary-contract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.workers <= 0 or args.max_jobs < 0:
        raise ValueError("workers must be positive and max-jobs non-negative")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    validate_plan(plan, require_full_primary=args.require_full_primary_contract)
    pending = [job for job in plan["jobs"] if not result_complete(job)]
    if args.max_jobs:
        pending = pending[: args.max_jobs]
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if not gpu_ids:
        raise ValueError("The formal MLP campaign requires at least one GPU ID")
    if args.dry_run:
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "plan_jobs": len(plan["jobs"]),
            "resume_complete": len(plan["jobs"]) - len(pending),
            "pending": len(pending),
            "workers": args.workers,
            "gpu_ids": gpu_ids,
            "first_jobs": [job["job_id"] for job in pending[:10]],
        }, indent=2))
        return

    args.run_root.mkdir(parents=True, exist_ok=True)
    logs = args.run_root / "logs"
    logs.mkdir(exist_ok=True)
    status_path = args.run_root / "queue_status.json"
    started = time.time()
    completed = 0
    failures: list[dict] = []
    atomic_json(status_path, {
        "status": "RUNNING", "planned": len(pending), "completed": 0, "failed": 0,
    })

    def execute(item: tuple[int, dict]) -> dict:
        index, job = item
        output = Path(job["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        log = logs / f"{job['job_id']}.log"
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % len(gpu_ids)]
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                job["command"],
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        returncode = process.returncode
        if returncode == 0 and not result_complete(job):
            returncode = 2
        return {"job_id": job["job_id"], "returncode": returncode, "log": str(log)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, item): item[1] for item in enumerate(pending)}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result["returncode"] == 0:
                completed += 1
            else:
                failures.append(result)
                if not args.continue_on_error:
                    for queued in futures:
                        queued.cancel()
                    break
            atomic_json(status_path, {
                "status": "RUNNING",
                "planned": len(pending),
                "completed": completed,
                "failed": len(failures),
                "remaining": len(pending) - completed - len(failures),
                "failures": failures,
                "elapsed_seconds": time.time() - started,
            })
    final = {
        "status": "PASS" if not failures else "COMPLETE_WITH_FAILURES",
        "planned": len(pending),
        "completed": completed,
        "failed": len(failures),
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(status_path, final)
    print(json.dumps(final, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

