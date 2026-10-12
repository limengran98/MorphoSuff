"""Shared contracts for the OPS all-reporter same-cell campaign."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


HERE = Path(__file__).resolve().parent
LEGACY_RUNNER = (
    HERE.parent
    / "low_label"
    / "legacy_exact"
    / "scripts"
    / "run_ops_same_cell_falsification.py"
)

SCHEMA = "measurement-sufficiency-same-cell-plan-v1"
PRIMARY_MODEL = "mlp"
N_REPORTERS = 52
N_FOLDS = 5

# Public protocol names map to the already validated study-runner identities.
ARM_TO_RUNNER = {
    "exact_same_cell": "exact_pair",
    "gene_screen_derangement": "gene_screen_deranged",
    "covariate_matched_derangement": "covariate_matched_deranged",
    "size_shape_only": "size_shape_only",
    "phase_without_size_shape": "remove_size_shape",
    "cross_fitted_within_gene_screen_residual": "crossfit_gene_screen_residual",
}


def atomic_json(path: Path, payload: Any) -> None:
    """Write JSON atomically without leaving a partially written plan/status."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_selection(value: str, allowed: Iterable[str], *, label: str) -> list[str]:
    allowed = list(allowed)
    if value.strip().casefold() in {"all", "all52"}:
        return allowed
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid {label} selection; unknown={unknown}")
    return list(dict.fromkeys(selected))


def load_reporters(target_table: Path) -> list[str]:
    table = pd.read_csv(target_table, usecols=["reporter_slug"])
    reporters = table["reporter_slug"].astype(str).tolist()
    if len(reporters) != N_REPORTERS or len(set(reporters)) != N_REPORTERS:
        raise ValueError(
            f"The frozen all-reporter campaign requires {N_REPORTERS} unique "
            f"reporter_slug values; found {len(reporters)} rows and "
            f"{len(set(reporters))} unique values"
        )
    return reporters


def validate_assets(
    *,
    phase_cache: Path,
    exact_cache_root: Path,
    target_table: Path,
    target_feature_dictionary: Path,
    reporters: Iterable[str],
) -> None:
    required = [
        phase_cache / "manifest.json",
        target_table,
        target_feature_dictionary,
        LEGACY_RUNNER,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if not exact_cache_root.is_dir():
        missing.append(str(exact_cache_root))
    missing.extend(
        str(exact_cache_root / f"all_cells_fluor_{reporter}.exact.h5")
        for reporter in reporters
        if not (exact_cache_root / f"all_cells_fluor_{reporter}.exact.h5").is_file()
    )
    if missing:
        preview = missing[:8]
        suffix = "" if len(missing) <= len(preview) else f" (+{len(missing) - len(preview)} more)"
        raise FileNotFoundError(f"Missing campaign assets: {preview}{suffix}")


def validate_plan(plan: dict[str, Any], *, require_full_primary: bool = False) -> None:
    if plan.get("schema_version") != SCHEMA:
        raise ValueError(f"Unexpected same-cell plan schema: {plan.get('schema_version')}")
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Same-cell plan contains no jobs")
    identities = {
        (job.get("reporter"), job.get("fold"), job.get("arm"), job.get("model"))
        for job in jobs
    }
    if len(identities) != len(jobs):
        raise ValueError("Same-cell plan contains duplicate job identities")
    output_dirs = [job.get("output_dir") for job in jobs]
    if None in output_dirs or len(set(output_dirs)) != len(output_dirs):
        raise ValueError("Every same-cell job must own a unique output directory")
    for job in jobs:
        if job.get("arm") not in ARM_TO_RUNNER:
            raise ValueError(f"Unknown canonical arm in plan: {job.get('arm')}")
        if job.get("runner_arm") != ARM_TO_RUNNER[job["arm"]]:
            raise ValueError(f"Runner-arm mapping mismatch for {job.get('job_id')}")
        if job.get("model") != PRIMARY_MODEL:
            raise ValueError("The primary same-cell campaign fixes the estimator to MLP")
        command = job.get("command")
        if not isinstance(command, list) or not command:
            raise ValueError(f"Job {job.get('job_id')} has no executable command")
    if require_full_primary:
        reporters = {identity[0] for identity in identities}
        folds = {identity[1] for identity in identities}
        arms = {identity[2] for identity in identities}
        models = {identity[3] for identity in identities}
        expected = N_REPORTERS * N_FOLDS * len(ARM_TO_RUNNER)
        if (
            len(reporters) != N_REPORTERS
            or folds != set(range(N_FOLDS))
            or arms != set(ARM_TO_RUNNER)
            or models != {PRIMARY_MODEL}
            or len(jobs) != expected
        ):
            raise ValueError(
                "Full primary contract requires 52 reporters x 5 folds x 6 arms "
                f"= {expected} MLP jobs"
            )


def result_complete(job: dict[str, Any]) -> bool:
    path = Path(job["output_dir"]) / "same_cell_result.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("completed") is True
        and payload.get("model") == job["model"]
        and payload.get("reporter_slug") == job["reporter"]
        and payload.get("fold") == job["fold"]
        and payload.get("arm") == job["runner_arm"]
    )

