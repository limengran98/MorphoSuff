from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from measurement_sufficiency.schemas import validate_predictions


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "studies/ops/runners/run_external_fold.py"


def _tables(root: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(913)
    count = 30
    inputs = pd.DataFrame(
        {
            "observation_id": [f"cell_{i:03d}" for i in range(count)],
            "reporter_id": "synthetic_reporter",
            "screen_id": [f"screen_{i % 3}" for i in range(count)],
            "perturbation_id": [f"gene_{i % 5}" for i in range(count)],
            "is_control": [i % 5 == 0 for i in range(count)],
        }
    )
    features = rng.normal(size=(count, 172)).astype(np.float32)
    feature_frame = pd.DataFrame(
        features, columns=[f"phase_{column:03d}" for column in range(172)]
    )
    inputs = pd.concat([inputs, feature_frame], axis=1)
    weights = rng.normal(size=(172, 4)).astype(np.float32)
    phenotype = features @ weights / np.sqrt(172) + rng.normal(scale=0.1, size=(count, 4))
    targets = pd.DataFrame(
        [
            {
                "observation_id": f"cell_{row:03d}",
                "reporter_id": "synthetic_reporter",
                "endpoint_id": f"endpoint_{column}",
                "y_true": float(phenotype[row, column]),
                "is_observed": True,
            }
            for row in range(count)
            for column in range(4)
        ]
    )
    role = ["train"] * 16 + ["validation"] * 6 + ["test"] * 8
    assignments = pd.DataFrame(
        {"observation_id": inputs.observation_id, "fold": 0, "role": role}
    )
    paths = root / "inputs.csv", root / "targets.csv", root / "assignments.csv"
    inputs.to_csv(paths[0], index=False)
    targets.to_csv(paths[1], index=False)
    assignments.to_csv(paths[2], index=False)
    return paths


def _fake_scpair(root: Path) -> Path:
    package = root / "scpair"
    package.mkdir(parents=True)
    (package / "model.py").write_text(
        """
import torch
from torch import nn

class Input_Module(nn.Module):
    def __init__(self, input_dim, _n_batches, hidden, **kwargs):
        super().__init__(); self.net = nn.Sequential(nn.Linear(input_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[-1]))
    def forward(self, x, *args): return self.net(x), None

class Output_Module_Gau(nn.Module):
    def __init__(self, output_dim, _n_batches, hidden, **kwargs):
        super().__init__(); self.net = nn.Sequential(nn.Linear(hidden[-1], hidden[0]), nn.ReLU(), nn.Linear(hidden[0], output_dim))
    def forward(self, x, *args): return self.net(x)

class Module_Module(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, **kwargs):
        super().__init__(); self.net = nn.Sequential(nn.Linear(input_dim, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], output_dim))
    def forward(self, x): return self.net(x)
""",
        encoding="utf-8",
    )
    return root


@pytest.mark.requires_torch
@pytest.mark.parametrize("method", ["scbutterfly", "midas", "scpair"])
def test_external_runner_cpu_smoke(tmp_path: Path, method: str) -> None:
    inputs, targets, assignments = _tables(tmp_path)
    output = tmp_path / f"out_{method}"
    command = [
        sys.executable,
        str(RUNNER),
        "--method",
        method,
        "--inputs",
        str(inputs),
        "--targets",
        str(targets),
        "--assignments",
        str(assignments),
        "--split-name",
        "gene",
        "--fold",
        "0",
        "--output-root",
        str(output),
        "--device",
        "cpu",
        "--smoke",
    ]
    if method == "scpair":
        command += [
            "--scpair-source",
            str(_fake_scpair(tmp_path / "fake_scpair")),
            "--allow-unpinned-scpair",
        ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True, timeout=180)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    marker = json.loads((output / "training_result.json").read_text())
    assert marker["status"] == "complete"
    assert marker["outer_test_used_for_selection"] is False
    predictions = pd.read_csv(output / "predictions.csv")
    validate_predictions(predictions)
    assert set(predictions.model_id) == {method}
    assert set(predictions.observation_id) == {f"cell_{i:03d}" for i in range(22, 30)}


# The contract check performs no tensor work and carries no PyTorch marker:
# run_external_fold.py defers torch and the three biological backends to
# _load_backends(), which runs only past the --contract-check-only exit.
def test_external_runner_contract_is_side_effect_free(tmp_path: Path) -> None:
    inputs, targets, assignments = _tables(tmp_path)
    output = tmp_path / "absent"
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--method",
            "midas",
            "--inputs",
            str(inputs),
            "--targets",
            str(targets),
            "--assignments",
            str(assignments),
            "--split-name",
            "gene",
            "--fold",
            "0",
            "--output-root",
            str(output),
            "--contract-check-only",
        ],
        check=False,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["status"] == "CONTRACT_PASS"
    assert payload["n_features"] == 172
    assert not output.exists()
