#!/usr/bin/env python3
"""CPU-only synthetic smoke test for the frozen raw-image head pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent


def build_fixture(root: Path) -> dict[str, Path]:
    rng = np.random.default_rng(17)
    phase = root / "phase"
    exact = root / "exact"
    embedding = root / "embedding"
    for directory in (phase, exact, embedding):
        directory.mkdir(parents=True)
    n_folds, per_fold, dimension, endpoints = 5, 30, 12, 3
    n = n_folds * per_fold
    fold = np.repeat(np.arange(n_folds), per_fold).astype(np.int16)
    is_control = np.zeros(n, dtype=bool)
    gene = np.empty(n, dtype=np.int32)
    for split in range(n_folds):
        start = split * per_fold
        is_control[start : start + 6] = True
        gene[start : start + 6] = -1
        gene[start + 6 : start + per_fold] = split * 10 + np.repeat(np.arange(3), 8)
    x = rng.normal(size=(n, dimension)).astype(np.float32)
    weights = rng.normal(size=(dimension, endpoints)).astype(np.float32)
    y = x @ weights + 0.05 * rng.normal(size=(n, endpoints)).astype(np.float32)
    y += (~is_control)[:, None] * ((gene.clip(min=0) % 3)[:, None] - 1) * np.array([[0.2, -0.1, 0.3]], np.float32)
    (phase / "manifest.json").write_text(json.dumps({"n_folds": n_folds}), encoding="utf-8")
    np.save(phase / "screen_code.npy", np.zeros(n, dtype=np.int16))
    np.save(phase / "well_code.npy", np.arange(n, dtype=np.int32) // 10)
    np.save(phase / "tile_code.npy", np.arange(n, dtype=np.int32) // 5)
    np.save(phase / "gene_code.npy", gene)
    np.save(phase / "gene_holdout_main.fold.npy", fold)
    names = np.asarray(["endpoint_a", "endpoint_b", "endpoint_c"], dtype="S")
    with h5py.File(exact / "all_cells_fluor_smoke_reporter.exact.h5", "w") as handle:
        handle.create_dataset("phase_row_index", data=np.arange(n, dtype=np.int64))
        handle.create_dataset("fluorescence", data=y)
        handle.create_dataset("metadata/is_control", data=is_control)
        handle.create_dataset("features/target_feature_names", data=names)
    dictionary = root / "target_feature_dictionary.csv"
    pd.DataFrame({
        "reporter_slug": ["smoke_reporter"] * endpoints,
        "target_feature_name": names.astype(str),
        "selected_in_technical_core": [True] * endpoints,
    }).to_csv(dictionary, index=False)
    np.save(embedding / "embedding.float16.npy", x.astype(np.float16))
    np.save(embedding / "phase_row_index.npy", np.arange(n, dtype=np.int64))
    np.save(embedding / "phase_row_sort_order.npy", np.arange(n, dtype=np.int64))
    (embedding / "manifest.json").write_text(json.dumps({
        "status": "COMPLETE", "encoder": "synthetic", "fingerprint": "smoke"
    }), encoding="utf-8")
    return {"phase": phase, "exact": exact, "embedding": embedding, "dictionary": dictionary}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="measurement-sufficiency-raw-image-") as temporary:
        root = Path(temporary)
        assets = build_fixture(root)
        output = root / "result"
        command = [
            sys.executable, str(HERE / "train_frozen_head.py"),
            "--reporter", "smoke_reporter", "--fold", "0",
            "--embedding-root", str(assets["embedding"]),
            "--output-dir", str(output),
            "--phase-cache", str(assets["phase"]),
            "--exact-cache-root", str(assets["exact"]),
            "--target-feature-dictionary", str(assets["dictionary"]),
            "--hidden", "16,8", "--batch-size", "32", "--epochs", "6",
            "--patience", "3", "--bootstrap-draws", "5", "--device", "cpu",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        result = json.loads((output / "training_result.json").read_text(encoding="utf-8"))
        if result.get("status") != "PASS" or not result.get("completed"):
            raise RuntimeError(f"Synthetic frozen-head run failed: {result}")
        for name in ("cell_metrics", "gene_metrics"):
            if not np.isfinite(float(result[name]["gain_vs_train_mean"])):
                raise RuntimeError(f"Smoke metric is non-finite: {name}")
        print(json.dumps({
            "status": "SMOKE_PASS", "runner": "train_frozen_head.py",
            "n_train": result["n_train"], "n_test": result["n_test"],
            "cell_pearson": result["cell_metrics"]["macro_feature_pearson"],
            "gene_pearson": result["gene_metrics"]["macro_feature_pearson"],
        }, indent=2))


if __name__ == "__main__":
    main()
