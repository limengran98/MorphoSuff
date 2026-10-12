from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "studies"
    / "ops"
    / "acquisition"
    / "public_s3.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("ops_public_s3", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inventory_round_trip_and_selection(tmp_path):
    module = load_module()
    rows = [
        {
            "kind": "phase",
            "filename": "all_cells_phase.h5ad",
            "bucket": "public",
            "key": "release/all_cells_phase.h5ad",
            "s3_uri": "s3://public/release/all_cells_phase.h5ad",
            "https_url": "https://public.s3.amazonaws.com/release/all_cells_phase.h5ad",
            "bytes": 100,
            "last_modified": "2026-01-01T00:00:00+00:00",
        },
        {
            "kind": "reporter",
            "filename": "all_cells_fluor_lysosome_lamp1.h5ad",
            "bucket": "public",
            "key": "release/all_cells_fluor_lysosome_lamp1.h5ad",
            "s3_uri": "s3://public/release/all_cells_fluor_lysosome_lamp1.h5ad",
            "https_url": (
                "https://public.s3.amazonaws.com/"
                "release/all_cells_fluor_lysosome_lamp1.h5ad"
            ),
            "bytes": 200,
            "last_modified": "2026-01-01T00:00:00+00:00",
        },
    ]
    path = tmp_path / "inventory.csv"
    module.write_inventory(rows, path)
    recovered = module.read_inventory(path)
    assert len(recovered) == 2
    selected = module.select_rows(recovered, {"reporter"}, {"lysosome_lamp1"})
    assert [row["filename"] for row in selected] == [
        "all_cells_fluor_lysosome_lamp1.h5ad"
    ]


def test_sha256_file(tmp_path):
    module = load_module()
    path = tmp_path / "asset.bin"
    path.write_bytes(b"OPS")
    assert (
        module.sha256_file(path)
        == "bce6162200c91bcf5e7ba6dca8212c63a2aaab133f905c1b8ce2801e4e8b4618"
    )
