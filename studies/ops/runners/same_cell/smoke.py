#!/usr/bin/env python3
"""CPU-only contract smoke for the public same-cell campaign facade."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[3]


def run(*arguments: str) -> str:
    completed = subprocess.run(
        [sys.executable, *map(str, arguments)],
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(REPOSITORY / "src")},
    )
    return completed.stdout


def synthetic_intervention_check() -> None:
    sys.path.insert(0, str(REPOSITORY / "src"))
    from measurement_sufficiency.interventions import (
        cross_fitted_residuals,
        derange_targets,
        select_features,
    )

    frame = pd.DataFrame({
        "gene": np.repeat(["g1", "g2"], 4),
        "screen": ["s1"] * 8,
        "target": np.arange(8),
        "cell_size": np.linspace(1, 2, 8),
        "texture": np.linspace(2, 3, 8),
        "cell_shape": np.linspace(3, 4, 8),
    })
    shuffled = derange_targets(
        frame,
        target_columns=["target"],
        group_columns=["gene", "screen"],
        seed=7,
    )
    if shuffled["target"].eq(frame["target"]).any():
        raise RuntimeError("Synthetic derangement retained an exact pair")
    if select_features(frame, list(frame.columns), mode="size_shape_only").shape[1] != 2:
        raise RuntimeError("Synthetic size/shape feature contract failed")
    residual = cross_fitted_residuals(
        pd.Series(np.arange(8, dtype=float)),
        pd.DataFrame({"x": np.tile([0.0, 1.0], 4)}),
        np.repeat([0, 1], 4),
    )
    if not np.isfinite(residual).all():
        raise RuntimeError("Synthetic cross-fitted residual contract failed")


def main() -> None:
    for entry in ("build_plan.py", "run_plan.py"):
        if "usage:" not in run(str(HERE / entry), "--help").casefold():
            raise RuntimeError(f"--help contract failed for {entry}")
    synthetic_intervention_check()
    with tempfile.TemporaryDirectory(prefix="same-cell-contract-") as temporary:
        root = Path(temporary)
        phase = root / "phase"; phase.mkdir()
        (phase / "manifest.json").write_text("{}\n", encoding="utf-8")
        exact = root / "exact"; exact.mkdir()
        reporters = [f"reporter_{index:02d}" for index in range(52)]
        target = root / "reporters.csv"
        pd.DataFrame({"reporter_slug": reporters}).to_csv(target, index=False)
        dictionary = root / "dictionary.csv"
        pd.DataFrame({"reporter_slug": reporters, "target_feature_name": "mean"}).to_csv(
            dictionary, index=False
        )
        for reporter in reporters:
            (exact / f"all_cells_fluor_{reporter}.exact.h5").touch()
        plan = root / "plan.json"
        run(
            str(HERE / "build_plan.py"),
            "--phase-cache", phase,
            "--exact-cache-root", exact,
            "--target-table", target,
            "--target-feature-dictionary", dictionary,
            "--result-root", root / "results",
            "--output-plan", plan,
        )
        payload = json.loads(plan.read_text(encoding="utf-8"))
        if payload["n_jobs"] != 52 * 5 * 6:
            raise RuntimeError("Full campaign plan does not contain 1,560 jobs")
        output = run(
            str(HERE / "run_plan.py"),
            "--plan", plan,
            "--run-root", root / "queue",
            "--require-full-primary-contract",
            "--dry-run",
        )
        if "DRY_RUN_PASS" not in output:
            raise RuntimeError("Queue dry-run contract failed")
    print(json.dumps({
        "status": "CONTRACT_SMOKE_PASS",
        "primary_estimator": "mlp",
        "reporters": 52,
        "folds": 5,
        "arms": 6,
        "jobs": 1560,
        "gpu_training_invoked": False,
    }, indent=2))


if __name__ == "__main__":
    main()

