"""Export native phase crops as illustrative views for the field-holdout schematic.

The files are cell-centred crops, not full fields or benchmark split assignments.
No spatial transform, interpolation, resizing, or source-array change is applied.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
RENDER_SPEC = HERE / "render_spec.csv"
RAW = HERE / "raw_data"
OUT = HERE / "assets"
LOW = -0.5457885962724686
HIGH = 0.8108193516731264
LIMITATION = (
    "These are real cell-centred phase crops from three wells of Biohub_OPS0077, "
    "used only as illustrative views for a FIELD holdout schematic. They are not "
    "complete imaging fields, do not encode actual field identities, and do not "
    "represent the actual training, validation, or test assignments of this benchmark."
)


def sha256(file: Path) -> str:
    return hashlib.sha256(file.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with RENDER_SPEC.open(encoding="utf-8-sig", newline="") as handle:
        render_rows = list(csv.DictReader(handle))
    records = []
    for index in (1, 2, 3):
        asset_id = f"prb_sall3_ops0077_a{index}"
        raw_file = RAW / f"{asset_id}.npz"
        metadata_file = RAW / f"{asset_id}.json"
        metadata = json.loads(metadata_file.read_text(encoding="utf-8-sig"))
        assert sha256(raw_file) == metadata["file_sha256"]
        windows = [row for row in render_rows if row["asset_id"] == asset_id]
        assert windows, f"No frozen window for {asset_id}"
        for row in windows:
            assert float(row["phase_display_low"]) == LOW
            assert float(row["phase_display_high"]) == HIGH
            assert row["spatial_transform"] == "none"
        with np.load(raw_file, allow_pickle=False) as arrays:
            phase = arrays["phase"]
            assert phase.shape == (384, 384)
            assert phase.dtype == np.float32
            assert np.isfinite(phase).all(), "No pixel imputation is authorized"
            native_phase_hash = hashlib.sha256(phase.tobytes(order="C")).hexdigest()
            display = np.rint(
                np.clip((phase.astype(np.float64) - LOW) / (HIGH - LOW), 0.0, 1.0) * 255
            ).astype(np.uint8)
        output = OUT / f"field_view_a{index}.png"
        Image.fromarray(display).save(output)
        with Image.open(output) as check:
            assert check.size == (384, 384)
            assert check.mode == "L"
            assert np.array_equal(np.asarray(check), display)
        records.append({
            "asset_id": asset_id,
            "output": output.relative_to(HERE).as_posix(),
            "output_sha256": sha256(output),
            "raw_copy": raw_file.relative_to(HERE).as_posix(),
            "raw_copy_sha256": sha256(raw_file),
            "source_asset_reference": metadata["provenance"]["source_relative_path"],
            "metadata_copy": metadata_file.relative_to(HERE).as_posix(),
            "metadata_sha256": sha256(metadata_file),
            "phase_array_sha256": native_phase_hash,
            "raw_dimensions_pixels": [384, 384],
            "output_dimensions_pixels": [384, 384],
            "source_channel": "phase",
            "phase_display_low": LOW,
            "phase_display_high": HIGH,
            "display_mapping": "Fixed-window linear grayscale, clipped to [0, 1], rounded to nearest uint8; PNG lossless.",
            "spatial_transform": "none",
            "orientation": "native stored orientation",
            "interpolation": "none",
            "resampling": "none",
            "cropping": "none beyond the already stored cell-centred raw crop",
            "enhancement": "none",
            "scale_bar": "not burned into image; source pixel size is 0.325 micrometres",
            "intended_use_and_limitations": LIMITATION,
            "frozen_render_spec_rows": windows,
            "source_metadata": metadata,
        })
    manifest = {
        "title": "Real phase crop views for Figure 2a field-holdout schematic",
        "intended_use_and_limitations": LIMITATION,
        "path_base": "a_workflow/source_data",
        "render_script": Path(__file__).name,
        "source_render_spec": RENDER_SPEC.name,
        "source_render_spec_origin": "Frozen microscopy display windows from the adopted source-figure package.",
        "raw_source_files_changed": False,
        "assets": records,
    }
    provenance = HERE / "field_view_provenance.json"
    provenance.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [item["output"] for item in records], "provenance": provenance.name, "verified": True}))


if __name__ == "__main__":
    main()
