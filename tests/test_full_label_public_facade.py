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
FACADE = REPO / "studies/ops/runners/full_label/run.py"
STRICT = REPO / "studies/ops/runners/full_label/build_strict_screen.py"
MATERIALIZE = REPO / "studies/ops/runners/full_label/materialize_fold.py"


def flat_tables(
    root: Path,
    reporters: tuple[str, ...] = ("r1", "r2", "r3", "r4", "r5"),
) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(41)
    inputs, targets, assignments = [], [], []
    for reporter in reporters:
        x = rng.normal(size=(30, 172)).astype(np.float32)
        weight = rng.normal(size=(172, 3)).astype(np.float32)
        y = x @ weight / np.sqrt(172)
        metadata = pd.DataFrame(
            {
                "observation_id": [f"{reporter}:{i}" for i in range(30)],
                "reporter_id": reporter,
                "screen_id": [f"s{i % 3}" for i in range(30)],
                "perturbation_id": [f"g{i % 5}" for i in range(30)],
                # Canonical bundles carry is_control on the inputs table; the
                # runners propagate it into the prediction export.
                "is_control": [i % 5 == 0 for i in range(30)],
            }
        )
        inputs.append(pd.concat([metadata, pd.DataFrame(x, columns=[f"phase_{i:03d}" for i in range(172)])], axis=1))
        targets.append(
            pd.DataFrame(
                [
                    {
                        "observation_id": f"{reporter}:{row}",
                        "reporter_id": reporter,
                        "endpoint_id": f"{reporter}_e{column}",
                        "y_true": float(y[row, column]),
                        "is_observed": True,
                    }
                    for row in range(30)
                    for column in range(3)
                ]
            )
        )
        assignments.append(
            pd.DataFrame(
                {
                    "observation_id": metadata.observation_id,
                    "fold": 0,
                    "role": ["train"] * 18 + ["validation"] * 6 + ["test"] * 6,
                }
            )
        )
    paths = root / "inputs.csv", root / "targets.csv", root / "assignments.csv"
    pd.concat(inputs).to_csv(paths[0], index=False)
    pd.concat(targets).to_csv(paths[1], index=False)
    pd.concat(assignments).to_csv(paths[2], index=False)
    return paths


def fake_tabm_source(root: Path) -> Path:
    source = root / "tabm.py"
    source.write_text(
        """
import torch
from torch import nn

__version__ = "development-smoke"

class TabM(nn.Module):
    def __init__(self, n_num_features, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_num_features, 12), nn.ReLU(), nn.Linear(12, d_out))

    @classmethod
    def make(cls, n_num_features, cat_cardinalities, d_out, **kwargs):
        assert cat_cardinalities == []
        return cls(n_num_features, d_out)

    def forward(self, x):
        return self.net(x)
""",
        encoding="utf-8",
    )
    return source


# ridge, gbdt and catboost are pure scikit-learn/CatBoost routes and carry no
# PyTorch marker: run_internal_fold.py imports measurement_sufficiency.ops_training
# inside its neural branches only, so these three run in the dependency-minimal
# install. tests/test_no_torch_execution_contract.py pins that guarantee.
@pytest.mark.parametrize(
    "method",
    [
        pytest.param("ridge"),
        pytest.param("gbdt"),
        pytest.param("catboost", marks=pytest.mark.requires_catboost),
        pytest.param("mlp", marks=pytest.mark.requires_torch),
        pytest.param("tabm", marks=pytest.mark.requires_torch),
        pytest.param("resmlp", marks=pytest.mark.requires_torch),
        pytest.param("multitab", marks=pytest.mark.requires_torch),
    ],
)
def test_internal_methods_execute_cpu_smoke(tmp_path: Path, method: str) -> None:
    inputs, targets, assignments = flat_tables(tmp_path)
    output = tmp_path / method
    command = [
            sys.executable,
            str(FACADE),
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
    if method == "tabm":
        command += [
            "--tabm-source",
            str(fake_tabm_source(tmp_path)),
            "--allow-unpinned-tabm",
        ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    marker = json.loads((output / "training_result.json").read_text())
    assert marker["status"] == "complete"
    assert marker["outer_test_used_for_selection"] is False
    predictions = pd.read_csv(output / "predictions.csv")
    validate_predictions(predictions)
    assert set(predictions.model_id) == {method}


def test_facade_prints_all_ten_routes_without_side_effects(tmp_path: Path) -> None:
    inputs, targets, assignments = flat_tables(tmp_path)
    for method in ("ridge", "gbdt", "catboost", "mlp", "tabm", "scbutterfly", "resmlp", "multitab", "midas", "scpair"):
        command = [
            sys.executable,
            str(FACADE),
            "--method",
            method,
            "--inputs",
            str(inputs),
            "--targets",
            str(targets),
            "--assignments",
            str(assignments),
            "--split-name",
            "strict_whole_screen",
            "--fold",
            "0",
            "--output-root",
            str(tmp_path / "never"),
            "--print-command",
        ]
        if method == "tabm":
            command += ["--tabm-source", str(tmp_path / "tabm.py")]
        if method == "scpair":
            command += ["--scpair-source", str(tmp_path / "scpair")]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["method"] == method
    assert not (tmp_path / "never").exists()


def canonical_release(root: Path) -> Path:
    atlas = root / "atlas"
    atlas.mkdir(parents=True)
    reporters = ("r1", "r2")
    manifests = []
    task_rows = []
    rng = np.random.default_rng(912)
    for reporter in reporters:
        bundle = root / "reporters" / reporter
        bundle.mkdir(parents=True)
        count = 30
        metadata = pd.DataFrame(
            {
                "observation_id": [f"{reporter}:{index}" for index in range(count)],
                "reporter_id": reporter,
                "screen_id": [f"s{index % 3}" for index in range(count)],
                "perturbation_id": [f"g{index % 5}" for index in range(count)],
                "is_control": [index % 5 == 0 for index in range(count)],
            }
        )
        features = pd.DataFrame(
            rng.normal(size=(count, 172)),
            columns=[f"phase_{column:03d}" for column in range(172)],
        )
        pd.concat([metadata, features], axis=1).to_csv(bundle / "inputs.csv", index=False)
        pd.DataFrame(
            [
                {
                    "observation_id": observation,
                    "reporter_id": reporter,
                    "endpoint_id": f"{reporter}_e0",
                    "y_true": float(index),
                    "is_observed": True,
                }
                for index, observation in enumerate(metadata.observation_id)
            ]
        ).to_csv(bundle / "targets.csv", index=False)
        pd.DataFrame(
            {
                "observation_id": metadata.observation_id,
                "fold": 0,
                "role": ["train"] * 24 + ["test"] * 6,
            }
        ).to_csv(bundle / "split_field.csv", index=False)
        pd.DataFrame(
            {
                "observation_id": metadata.observation_id,
                "fold": 0,
                "role": ["train"] * 18 + ["validation"] * 6 + ["test"] * 6,
            }
        ).to_csv(bundle / "split_gene.csv", index=False)
        manifest = {
            "reporter_id": reporter,
            "tables": {"inputs": "inputs.csv", "targets": "targets.csv"},
            "splits": {"field": "split_field.csv", "gene": "split_gene.csv"},
        }
        (bundle / "canonical_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        manifests.append(f"reporters/{reporter}/canonical_manifest.json")
        task_rows.extend(
            {"reporter_id": reporter, "destination_screen": screen}
            for screen in ("s0", "s1", "s2")
        )
    pd.DataFrame(task_rows).to_csv(atlas / "strict_screen_tasks.csv", index=False)
    (root / "canonical_release.json").write_text(
        json.dumps({"reporter_manifests": manifests}), encoding="utf-8"
    )
    return root


def test_strict_screen_plan_and_materialized_fold_are_runnable(tmp_path: Path) -> None:
    canonical = canonical_release(tmp_path / "canonical")
    plan_root = tmp_path / "strict"
    built = subprocess.run(
        [
            sys.executable,
            str(STRICT),
            "--canonical-root",
            str(canonical),
            "--output-dir",
            str(plan_root),
            "--n-folds",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stdout + "\n" + built.stderr
    plan_path = plan_root / "strict_screen_plan.json"
    plan = json.loads(plan_path.read_text())
    assert plan["n_directions"] == 6
    assert plan["whole_screen_removed_from_every_reporter"] is True
    assert plan["target_values_accessed"] is False

    fold_root = tmp_path / "fold0"
    materialized = subprocess.run(
        [
            sys.executable,
            str(MATERIALIZE),
            "--canonical-root",
            str(canonical),
            "--split",
            "strict_whole_screen",
            "--fold",
            "0",
            "--strict-screen-plan",
            str(plan_path),
            "--output-dir",
            str(fold_root),
            "--format",
            "csv",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert materialized.returncode == 0, materialized.stdout + "\n" + materialized.stderr
    manifest = json.loads((fold_root / "fold_manifest.json").read_text())
    assert manifest["split_name"] == "strict_whole_screen"
    assert manifest["n_train"] > 0
    assert manifest["n_validation"] > 0
    assert manifest["n_test"] > 0
    assignments = pd.read_csv(fold_root / "assignments.csv")
    inputs = pd.read_csv(fold_root / "inputs.csv")
    joined = inputs[["observation_id", "screen_id"]].merge(assignments, on="observation_id")
    assert set(joined.loc[joined.role.eq("test"), "screen_id"]).isdisjoint(
        set(joined.loc[joined.role.ne("test"), "screen_id"])
    )
