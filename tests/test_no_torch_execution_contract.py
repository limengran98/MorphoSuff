"""Routes that must run without PyTorch installed.

Ridge, GBDT and CatBoost are scikit-learn and CatBoost estimators; the external
contract check is pure bookkeeping. None of them has PyTorch as a dependency, and
the base contract smoke is dependency minimal.

Marking these tests ``requires_torch`` would not protect that: in the CI job where
PyTorch *is* installed they would pass regardless of whether a module-scope
``import torch`` had crept back in. These tests therefore block the import in a
subprocess, so they fail on a regression in either job.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
INTERNAL = ROOT / "studies" / "ops" / "runners" / "full_label" / "run_internal_fold.py"
EXTERNAL = ROOT / "studies" / "ops" / "runners" / "run_external_fold.py"

_BLOCKER = """
import sys


class _BlockTorch:
    def find_module(self, name, path=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError(
                "torch is blocked by the no-torch execution contract test"
            )
        return None

    def find_spec(self, name, path=None, target=None):
        self.find_module(name, path)
        return None


sys.meta_path.insert(0, _BlockTorch())
"""


@pytest.fixture(scope="module")
def blocked_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    shim = tmp_path_factory.mktemp("no_torch_shim")
    (shim / "sitecustomize.py").write_text(_BLOCKER, encoding="utf-8")
    existing = os.environ.get("PYTHONPATH")
    parts = [str(shim)] + ([existing] if existing else [])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(parts)}


def _assert_torch_is_blocked(env: dict[str, str]) -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert probe.returncode != 0, "the shim failed to block torch; the test is vacuous"
    assert "no-torch execution contract" in probe.stderr


def _fold_tables(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(0)
    n_cells, n_features = 48, 172
    observations = [f"cell_{i:03d}" for i in range(n_cells)]
    features = rng.normal(size=(n_cells, n_features))
    inputs = pd.DataFrame(features, columns=[f"phase_{j:03d}" for j in range(n_features)])
    inputs.insert(0, "observation_id", observations)
    inputs["reporter_id"] = "reporter_a"
    inputs["screen_id"] = ["screen_0" if i % 2 else "screen_1" for i in range(n_cells)]
    inputs["perturbation_id"] = [f"gene_{i % 6}" for i in range(n_cells)]
    inputs["is_control"] = [i % 6 == 0 for i in range(n_cells)]

    targets = pd.DataFrame(
        {
            "observation_id": observations,
            "reporter_id": "reporter_a",
            "endpoint_id": "endpoint_0",
            "y_true": features[:, 0] * 1.5 + rng.normal(scale=0.05, size=n_cells),
            "is_observed": True,
        }
    )
    roles = ["train"] * 32 + ["validation"] * 8 + ["test"] * 8
    assignments = pd.DataFrame(
        {"observation_id": observations, "fold": 0, "role": roles}
    )

    paths = tuple(tmp_path / name for name in ("inputs.csv", "targets.csv", "assignments.csv"))
    for frame, path in zip((inputs, targets, assignments), paths):
        frame.to_csv(path, index=False)
    return paths


@pytest.mark.parametrize("method", ["ridge", "gbdt"])
def test_sklearn_route_fits_a_fold_without_torch(
    tmp_path: Path, blocked_env: dict[str, str], method: str
) -> None:
    _assert_torch_is_blocked(blocked_env)
    inputs, targets, assignments = _fold_tables(tmp_path)
    completed = subprocess.run(
        [
            sys.executable, str(INTERNAL),
            "--method", method,
            "--inputs", str(inputs),
            "--targets", str(targets),
            "--assignments", str(assignments),
            "--split-name", "gene",
            "--fold", "0",
            "--output-root", str(tmp_path / method),
            "--device", "cpu",
            "--smoke",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=blocked_env,
        timeout=300,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (tmp_path / method / "predictions.csv").is_file()


def test_external_contract_check_runs_without_torch(
    tmp_path: Path, blocked_env: dict[str, str]
) -> None:
    _assert_torch_is_blocked(blocked_env)
    inputs, targets, assignments = _fold_tables(tmp_path)
    completed = subprocess.run(
        [
            sys.executable, str(EXTERNAL),
            "--method", "midas",
            "--inputs", str(inputs),
            "--targets", str(targets),
            "--assignments", str(assignments),
            "--split-name", "gene",
            "--fold", "0",
            "--output-root", str(tmp_path / "absent"),
            "--contract-check-only",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=blocked_env,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert not (tmp_path / "absent").exists(), "the contract check must not write output"


def test_toolkit_imports_without_torch(blocked_env: dict[str, str]) -> None:
    _assert_torch_is_blocked(blocked_env)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import measurement_sufficiency, measurement_sufficiency.cli, "
            "measurement_sufficiency.model_registry, measurement_sufficiency.training, "
            "measurement_sufficiency.ops_provenance; print('ok')",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=blocked_env,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert "ok" in completed.stdout
