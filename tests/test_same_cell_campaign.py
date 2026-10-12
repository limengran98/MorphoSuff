import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "studies" / "ops" / "runners" / "same_cell"


def _module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path):
    phase = tmp_path / "phase"; phase.mkdir()
    (phase / "manifest.json").write_text("{}\n", encoding="utf-8")
    exact = tmp_path / "exact"; exact.mkdir()
    reporters = [f"r{index:02d}" for index in range(52)]
    target = tmp_path / "target.csv"
    pd.DataFrame({"reporter_slug": reporters}).to_csv(target, index=False)
    dictionary = tmp_path / "dictionary.csv"
    pd.DataFrame({"reporter_slug": reporters, "target_feature_name": "mean"}).to_csv(
        dictionary, index=False
    )
    for reporter in reporters:
        (exact / f"all_cells_fluor_{reporter}.exact.h5").touch()
    return phase, exact, target, dictionary, reporters


def test_full_primary_plan_has_1560_unique_jobs(tmp_path):
    module = _module("same_cell_build_plan", RUNNER / "build_plan.py")
    phase, exact, target, dictionary, _ = _fixture(tmp_path)
    args = argparse.Namespace(
        phase_cache=phase,
        exact_cache_root=exact,
        target_table=target,
        target_feature_dictionary=dictionary,
        result_root=tmp_path / "results",
        output_plan=tmp_path / "plan.json",
        reporters="all52",
        folds="all",
        arms="all",
        model="mlp",
        python=sys.executable,
        skip_asset_check=False,
        dry_run=False,
    )
    plan = module.build(args)
    assert plan["n_jobs"] == 1560
    assert len({job["job_id"] for job in plan["jobs"]}) == 1560
    assert {job["model"] for job in plan["jobs"]} == {"mlp"}
    assert len({job["reporter"] for job in plan["jobs"]}) == 52
    assert {job["fold"] for job in plan["jobs"]} == set(range(5))
    assert len({job["arm"] for job in plan["jobs"]}) == 6


def test_subset_plan_is_explicit_and_still_uses_formal_runner(tmp_path):
    module = _module("same_cell_build_plan_subset", RUNNER / "build_plan.py")
    phase, exact, target, dictionary, reporters = _fixture(tmp_path)
    args = argparse.Namespace(
        phase_cache=phase,
        exact_cache_root=exact,
        target_table=target,
        target_feature_dictionary=dictionary,
        result_root=tmp_path / "results",
        output_plan=tmp_path / "plan.json",
        reporters=reporters[0],
        folds="0",
        arms="exact_same_cell,phase_without_size_shape",
        model="mlp",
        python=sys.executable,
        skip_asset_check=False,
        dry_run=False,
    )
    plan = module.build(args)
    assert plan["n_jobs"] == 2
    assert {job["runner_arm"] for job in plan["jobs"]} == {"exact_pair", "remove_size_shape"}
    assert all("run_ops_same_cell_falsification.py" in job["command"][1] for job in plan["jobs"])


def test_full_contract_rejects_incomplete_plan(tmp_path):
    common = _module("same_cell_common", RUNNER / "common.py")
    with pytest.raises(ValueError, match="Full primary contract"):
        common.validate_plan(
            {
                "schema_version": common.SCHEMA,
                "jobs": [{
                    "job_id": "x", "reporter": "r", "fold": 0,
                    "arm": "exact_same_cell", "runner_arm": "exact_pair",
                    "model": "mlp", "output_dir": str(tmp_path / "out"),
                    "command": [sys.executable, "runner.py"],
                }],
            },
            require_full_primary=True,
        )


def test_public_smoke_runs_without_gpu_or_ops_data():
    completed = subprocess.run(
        [sys.executable, str(RUNNER / "smoke.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["status"] == "CONTRACT_SMOKE_PASS"
    assert payload["jobs"] == 1560
    assert payload["gpu_training_invoked"] is False
