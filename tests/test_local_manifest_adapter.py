import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from measurement_sufficiency import LocalManifestAdapter


ROOT = Path(__file__).resolve().parents[1]


def test_local_manifest_adapter_validates_and_loads_canonical_csvs(tmp_path):
    tables = {
        "observations": pd.DataFrame(
            {
                "observation_id": ["o1", "o2"],
                "input_cell_id": ["i1", "i2"],
                "target_cell_id": ["t1", "t2"],
                "assay_id": ["a", "a"],
                "perturbation_id": ["control", "gene"],
                "guide_id": [None, "guide"],
                "screen_id": ["s", "s"],
                "well_id": ["w1", "w2"],
                "field_id": ["f1", "f2"],
                "is_control": [True, False],
            }
        ),
        "reporters": pd.DataFrame(
            {
                "reporter_id": ["r"],
                "reporter_name": ["Reporter"],
                "biological_system": ["system"],
                "target_modality": ["fluorescence"],
                "endpoint_schema_id": ["r_v1"],
            }
        ),
        "assays": pd.DataFrame(
            {"assay_id": ["a"], "reporter_id": ["r"], "screen_id": ["s"], "endpoint_ids": ["e1,e2"]}
        ),
        "pairing": pd.DataFrame(
            {
                "observation_id": ["o1", "o2"],
                "input_cell_id": ["i1", "i2"],
                "target_cell_id": ["t1", "t2"],
                "assay_id": ["a", "a"],
                "pairing_key": ["p1", "p2"],
                "pairing_confidence": [1.0, 1.0],
            }
        ),
        "inputs": pd.DataFrame({"input_cell_id": ["i1", "i2"], "feature_001": [0.0, 1.0]}),
        "targets": pd.DataFrame(
            {"target_cell_id": ["t1", "t2"], "reporter_id": ["r", "r"], "endpoint_id": ["e1", "e1"], "value": [2.0, 3.0]}
        ),
    }
    specs = {}
    for name, table in tables.items():
        path = tmp_path / f"{name}.csv"
        table.to_csv(path, index=False)
        specs[name] = path.name
    manifest = {
        "dataset_id": "example",
        "capabilities": {"exact_pairing": True, "screen_matched_controls": True},
        "input_schema": {"schema_id": "features", "representation": "tabular", "feature_names": ["feature_001"]},
        "tables": specs,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    adapter = LocalManifestAdapter(manifest_path)
    assert adapter.input_schema().feature_names == ("feature_001",)
    assert adapter.target_schema("r").endpoint_ids == ("e1", "e2")
    assert adapter.load_inputs(["i2"], "tabular").input_cell_id.tolist() == ["i2"]
    assert adapter.load_targets("r", ["t1"]).target_cell_id.tolist() == ["t1"]
    assert adapter.controls("s").observation_id.tolist() == ["o1"]

    result = subprocess.run(
        [sys.executable, "-m", "measurement_sufficiency.cli", "validate-dataset", str(manifest_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dataset_id"] == "example"
    assert payload["n_reporters"] == 1
    assert payload["capabilities"]["exact_pairing"] is True


def test_a_partitioned_table_directory_is_read_as_one_dataset(tmp_path) -> None:
    """A table too large to hold whole is written as one parquet file per partition.

    The manifest then names a directory. Dispatching on the file suffix alone sent it
    to ``read_csv``, so a partitioned table failed with a confusing error instead of
    being read. PERISCOPE writes its inputs and targets this way because the full arm
    is 11 million cells by 2,508 features.
    """
    import pandas as pd
    import pytest

    pytest.importorskip("pyarrow")
    from measurement_sufficiency.adapters import _read_local_table

    directory = tmp_path / "inputs"
    directory.mkdir()
    pd.DataFrame({"input_cell_id": ["a", "b"], "f": [1.0, 2.0]}).to_parquet(
        directory / "plate=P1.parquet", index=False
    )
    pd.DataFrame({"input_cell_id": ["c"], "f": [3.0]}).to_parquet(
        directory / "plate=P2.parquet", index=False
    )
    table = _read_local_table(directory)
    assert sorted(table["input_cell_id"]) == ["a", "b", "c"]
