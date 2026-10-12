import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from measurement_sufficiency import LocalManifestAdapter


def test_ops_extractor_maps_and_validates_declared_upstream_tables(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    source_tables = {
        "observations": pd.DataFrame(
            {
                "row": ["o1", "o2"],
                "phase": ["i1", "i2"],
                "target": ["t1", "t2"],
                "assay": ["a1", "a1"],
                "gene": ["NTC", "G1"],
                "guide": [None, "g1"],
                "screen": ["s1", "s1"],
                "well": ["w1", "w2"],
                "field": ["f1", "f2"],
                "condition": ["control", "ko"],
            }
        ),
        "reporters": pd.DataFrame(
            {
                "reporter": ["r1"],
                "name": ["Reporter"],
                "system": ["Nucleus"],
                "modality": ["fluorescence"],
                "schema": ["r1_v1"],
            }
        ),
        "assays": pd.DataFrame(
            {"assay": ["a1"], "reporter": ["r1"], "screen": ["s1"], "endpoints": ["e1;e2"]}
        ),
        "pairing": pd.DataFrame(
            {
                "row": ["o1", "o2"],
                "phase": ["i1", "i2"],
                "target": ["t1", "t2"],
                "assay": ["a1", "a1"],
                "key": ["p1", "p2"],
                "confidence": [1, 1],
            }
        ),
    }
    for name, table in source_tables.items():
        table.to_csv(upstream / f"{name}.csv", index=False)
    manifest = {
        "dataset_id": "test_ops",
        "capabilities": {"exact_pairing": True, "guides": True},
        "tables": {
            "observations": {
                "path": "upstream/observations.csv",
                "columns": {
                    "observation_id": "row", "input_cell_id": "phase", "target_cell_id": "target",
                    "assay_id": "assay",
                    "perturbation_id": "gene", "guide_id": "guide", "screen_id": "screen",
                    "well_id": "well", "field_id": "field", "is_control": "condition",
                },
            },
            "reporters": {
                "path": "upstream/reporters.csv",
                "columns": {
                    "reporter_id": "reporter", "reporter_name": "name",
                    "biological_system": "system", "target_modality": "modality",
                    "endpoint_schema_id": "schema",
                },
            },
            "assays": {
                "path": "upstream/assays.csv",
                "columns": {
                    "assay_id": "assay", "reporter_id": "reporter", "screen_id": "screen",
                    "endpoint_ids": "endpoints",
                },
            },
            "pairing": {
                "path": "upstream/pairing.csv",
                "columns": {
                    "observation_id": "row", "input_cell_id": "phase",
                    "target_cell_id": "target", "assay_id": "assay",
                    "pairing_key": "key", "pairing_confidence": "confidence",
                },
            },
        },
    }
    manifest_path = tmp_path / "upstream.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "canonical"
    repository = Path(__file__).resolve().parents[1]
    script = repository / "studies/ops/workflows/extract_ops_canonical.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--upstream-manifest", str(manifest_path), "--output-dir", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    adapter = LocalManifestAdapter(output / "canonical_manifest.json")
    assert adapter.observations().is_control.tolist() == [True, False]
    assert adapter.assays().endpoint_ids.tolist() == ["e1,e2"]
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["validation"].endswith(":PASS")
    assert len(provenance["tables"]) == 4
