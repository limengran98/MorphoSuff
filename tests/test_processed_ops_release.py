from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from studies.ops.preparation.validate_processed_release import validate


ROOT = Path(__file__).resolve().parents[1]
FIGURE2 = ROOT / "reproducibility/ops/figure2_from_huggingface.py"


def synthetic_processed_release(root: Path) -> Path:
    phase_root = root / "processed_ops/phase172"
    exact_root = root / "processed_ops/exact_reporters"
    metadata_root = root / "processed_ops/metadata"
    phase_root.mkdir(parents=True)
    exact_root.mkdir(parents=True)
    metadata_root.mkdir(parents=True)
    np.save(phase_root / "phase172.npy", np.arange(688, dtype=np.float32).reshape(4, 172))
    for name in ("screen_code", "well_code", "tile_code", "gene_code", "sgrna_code"):
        np.save(phase_root / f"{name}.npy", np.arange(4, dtype=np.int32))
    for name in ("is_target", "field_holdout_sanity.fold", "gene_holdout_main.fold"):
        np.save(phase_root / f"{name}.npy", np.arange(4, dtype=np.uint8))
    (phase_root / "categories.json").write_text("{}\n")
    (phase_root / "phase_features_172d.txt").write_text(
        "\n".join(f"phase_{index:03d}" for index in range(172)) + "\n"
    )
    path = exact_root / "all_cells_fluor_r1.exact.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("phase_row_index", data=np.arange(4))
        handle.create_dataset("fluorescence_row_index", data=np.arange(4))
        handle.create_dataset("fluorescence", data=np.ones((4, 2), dtype=np.float32))
        handle.create_dataset("features/target_feature_names", data=np.asarray([b"e1", b"e2"]))
        handle.create_dataset("metadata/screen_categories", data=np.asarray([b"s1"]))
        handle.create_dataset("metadata/gene_codes", data=np.arange(4))
        handle.create_dataset("metadata/sgRNA_codes", data=np.arange(4))
        handle.create_dataset("metadata/is_control", data=np.asarray([True, False, False, False]))
    pd.DataFrame({"reporter_slug": ["r1"]}).to_csv(metadata_root / "reporter_targets.csv", index=False)
    for name in ("target_feature_dictionary.csv", "target_screen_map.csv", "exact_pairing_integrity.csv"):
        pd.DataFrame({"fixture": [1]}).to_csv(metadata_root / name, index=False)
    return root


def test_processed_release_validator_accepts_schema_and_foreign_keys(tmp_path: Path) -> None:
    payload = validate(synthetic_processed_release(tmp_path), strict_census=False)
    assert payload["status"] == "PASS"
    assert payload["n_phase_cells"] == 4
    assert payload["n_reporters"] == 1
    assert payload["n_reporter_screen_assays"] == 1


def test_figure2_huggingface_entry_point_resolves_complete_pipeline(tmp_path: Path) -> None:
    data = tmp_path / "download"
    workspace = tmp_path / "workspace"
    command = [
        sys.executable,
        str(FIGURE2),
        "--dataset-root",
        str(data),
        "--workspace",
        str(workspace),
        "--reporters",
        "lysosome_lamp1",
        "--split",
        "gene",
        "--fold",
        "0",
        "--method",
        "ridge",
        "--dry-run",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "validate_processed_release.py" in result.stdout
    assert "export_canonical.py" in result.stdout
    assert "materialize_fold.py" in result.stdout
    assert "full_label/run.py" in result.stdout
    assert not workspace.exists()


def test_huggingface_spec_declares_training_assets_and_frozen_census() -> None:
    spec = json.loads((ROOT / "release/huggingface/release_spec.json").read_text())
    training = spec["processed_training"]
    assert "phase172.npy" in training["phase_files"]
    assert training["expected_census"]["n_reporters"] == 52
    assert training["expected_census"]["n_reporter_screen_assays"] == 99
    assert training["expected_census"]["n_exact_same_cell_observations"] == 9_996_286
