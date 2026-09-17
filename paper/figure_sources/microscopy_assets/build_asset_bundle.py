#!/usr/bin/env python3
"""Create the self-contained raw-microscopy asset bundle for manuscript figures.

The bundle intentionally stores native raw float32 arrays (NPZ/NPY), not rendered
PNG/JPEG derivatives.  It is a provenance and future-figure-editing asset only: it
does not rebuild or alter any manuscript figure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT.parent
FINAL = SOURCE_ROOT / "final"
RAW_DIR = ROOT / "raw_assets"
METADATA_DIR = ROOT / "metadata"


@dataclass(frozen=True)
class Asset:
    asset_id: str
    filename: str
    source_relative_path: str
    use_status: str
    figure_panel_usage: str
    reporter: str
    reporter_slug: str
    gene: str
    screen_id: str
    well: str
    perturbation_role: str
    phase_row_index: str
    segmentation_id: str
    pixel_size_um: float
    source_level: str
    channels: str
    selection_rule: str
    provenance_uri: str
    note: str = ""


# All current microscopy source arrays in Figures 1–6 and Supplementary Figure 2
# plus paired/reusable assets retained for future edits. Main-figure source paths are
# relative to final/; paths beginning supplementary_figure2/ are relative to
# figure_sources/.
ASSETS = [
    Asset(
        "lamp1_borcs7_ko", "lamp1_borcs7_ko.npz",
        "figure1/e_same_cell_response_maps/source_data/frozen_inputs/B_exact_same_cell_microscopy/lamp1_exact_same_cell.npz",
        "rendered; potential", "Fig.1c; Fig.1e; Fig.3d (retained potential)",
        "LAMP1", "lysosome_lamp1", "BORCS7", "Biohub_OPS0011", "A/1/0", "KO",
        "255214", "5095123", 0.325, "OME-Zarr level 0", "phase=0; fluorescence=2",
        "Frozen registered same-cell crop; fixed display windows are in render_spec.csv.",
        "s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v1.0.20260521/datasets/Biohub_OPS0011/Biohub_OPS0011.zarr",
    ),
    Asset(
        "ferhonox_alg12_ko", "ferhonox_alg12_ko.npz",
        "figure1/e_same_cell_response_maps/source_data/frozen_inputs/B_exact_same_cell_microscopy/ferhonox_exact_same_cell.npz",
        "rendered", "Fig.1c; Fig.1e",
        "FeRhoNox", "fe2+_ferhonox_live-cell_dye", "ALG12", "Biohub_OPS0047", "A/1/0", "KO",
        "4515", "3546304", 0.325, "OME-Zarr level 0", "phase=0; fluorescence=2",
        "Frozen registered same-cell crop; fixed display windows are in render_spec.csv.",
        "s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v1.0.20260521/datasets/Biohub_OPS0047/Biohub_OPS0047.zarr",
    ),
    Asset(
        "prb_fech_ko", "prb_fech_ko.npz",
        "figure1/e_same_cell_response_maps/source_data/frozen_inputs/B_exact_same_cell_microscopy/prb_exact_same_cell.npz",
        "rendered; potential", "Fig.1e; Fig.3d (retained potential)",
        "pRb", "prb", "FECH", "Biohub_OPS0077", "A/1/0", "KO",
        "751", "2246962", 0.325, "OME-Zarr level 0", "phase=0; fluorescence=9",
        "Frozen registered same-cell crop; phase and target channels remain paired.",
        "s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v1.0.20260521/datasets/Biohub_OPS0077/Biohub_OPS0077.zarr",
    ),
    Asset(
        "fastact_actr3_ko", "fastact_actr3_ko.npz",
        "figure3/source_data/microscopy/fastact_actr3_ko.npz",
        "rendered", "Fig.3a",
        "FastAct", "actin_filament_fastact_spy555_live_cell_dye", "ACTR3", "Biohub_OPS0033", "A/2/0", "KO",
        "5892936", "16150844", 0.325, "OME-Zarr level 0", "phase=0; fluorescence=2",
        "Frozen pre-specified microscopy roster; selection independent of canonical aggregation repair.",
        "s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v1.0.20260521/datasets/Biohub_OPS0033/Biohub_OPS0033.zarr",
    ),
    Asset(
        "fastact_actr3_ntc", "fastact_actr3_ntc.npz",
        "figure3/source_data/microscopy/fastact_actr3_ntc.npz",
        "potential", "Fig.3 paired future-edit counterpart",
        "FastAct", "actin_filament_fastact_spy555_live_cell_dye", "ACTR3", "Biohub_OPS0033", "A/3/0", "NTC",
        "8233075", "38969301", 0.325, "OME-Zarr level 0", "phase=0; fluorescence=2",
        "Frozen paired NTC counterpart for future same-reporter comparisons.",
        "s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v1.0.20260521/datasets/Biohub_OPS0033/Biohub_OPS0033.zarr",
    ),
    Asset(
        "lamp1_borcs7_ntc", "lamp1_borcs7_ntc.npz",
        "figure3/source_data/microscopy/lamp1_borcs7_ntc.npz",
        "rendered; potential", "Fig.3a; Fig.3 paired future-edit counterpart",
        "LAMP1", "lysosome_lamp1", "BORCS7", "Biohub_OPS0011", "A/2/0", "NTC",
        "8286394", "20443628", 0.325, "OME-Zarr level 0", "phase=0; fluorescence=2",
        "Frozen pre-specified microscopy roster; selection independent of canonical aggregation repair.",
        "s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v1.0.20260521/datasets/Biohub_OPS0011/Biohub_OPS0011.zarr",
    ),
    Asset(
        "5xupre_hspa5_ko", "5xupre_hspa5_ko.npz",
        "figure3/source_data/microscopy/5xupre_hspa5_ko.npz",
        "rendered", "Fig.3a",
        "5xUPRE", "5xupre", "HSPA5", "Biohub_OPS0001", "A/3/0", "KO",
        "6734514", "9973218", 0.325, "OME-Zarr level 0", "phase=0; fluorescence=2",
        "Frozen pre-specified microscopy roster; selection independent of canonical aggregation repair.",
        "s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v1.0.20260521/datasets/Biohub_OPS0001/Biohub_OPS0001.zarr",
    ),
    Asset(
        "5xupre_hspa5_ntc", "5xupre_hspa5_ntc.npz",
        "figure3/source_data/microscopy/5xupre_hspa5_ntc.npz",
        "potential", "Fig.3 paired future-edit counterpart",
        "5xUPRE", "5xupre", "HSPA5", "Biohub_OPS0001", "A/1/0", "NTC",
        "8353156", "21304555", 0.325, "OME-Zarr level 0", "phase=0; fluorescence=2",
        "Frozen paired NTC counterpart for future same-reporter comparisons.",
        "s3://ops-explorer-public/leonetti_ops/ops_data_portal_submission/v1.0.20260521/datasets/Biohub_OPS0001/Biohub_OPS0001.zarr",
    ),
    Asset(
        "prb_sall3_ops0077_a1", "prb_sall3_ops0077_a1.npz",
        "figure5/source_data/microscopy/cell_00.npz",
        "rendered", "Fig.5b",
        "pRb", "prb", "SALL3", "Biohub_OPS0077", "A/1/0", "KO",
        "703281", "34071508", 0.325, "Stage-B level-0 crop", "phase=0; fluorescence=9; masks",
        "Soft-QC non-edge targeting cell; phase/geometry medoid within well.",
        "Local staged raw crop; original OPS source documented in source figure manifest.",
    ),
    Asset(
        "prb_sall3_ops0077_a2", "prb_sall3_ops0077_a2.npz",
        "figure5/source_data/microscopy/cell_01.npz",
        "rendered", "Fig.4c; Fig.5b",
        "pRb", "prb", "SALL3", "Biohub_OPS0077", "A/2/0", "KO",
        "3159763", "8627706", 0.325, "Stage-B level-0 crop", "phase=0; fluorescence=9; masks",
        "Soft-QC non-edge targeting cell; phase/geometry medoid within well.",
        "Local staged raw crop; original OPS source documented in source figure manifest.",
    ),
    Asset(
        "prb_sall3_ops0077_a3", "prb_sall3_ops0077_a3.npz",
        "figure5/source_data/microscopy/cell_02.npz",
        "rendered", "Fig.5b",
        "pRb", "prb", "SALL3", "Biohub_OPS0077", "A/3/0", "KO",
        "742457", "38493100", 0.325, "Stage-B level-0 crop", "phase=0; fluorescence=9; masks",
        "Soft-QC non-edge targeting cell; phase/geometry medoid within well.",
        "Local staged raw crop; original OPS source documented in source figure manifest.",
    ),
    Asset(
        "ferhonox_ntc_ops0047", "ferhonox_ntc_ops0047.npz",
        "figure5/source_data/microscopy/cell_03.npz",
        "rendered", "Fig.5d",
        "FeRhoNox", "ferhonox", "NTC", "Biohub_OPS0047", "A/1/0", "NTC",
        "8229883", "28787683", 0.325, "downloaded level-0 crop", "phase=0; fluorescence=2; masks",
        "Soft-QC non-edge NTC; phase/geometry medoid within screen.",
        "Local downloaded crop; original OPS source documented in source figure manifest.",
    ),
    Asset(
        "ferhonox_ntc_ops0067", "ferhonox_ntc_ops0067.npz",
        "figure5/source_data/microscopy/cell_04.npz",
        "rendered", "Fig.5d",
        "FeRhoNox", "ferhonox", "NTC", "Biohub_OPS0067", "A/2/0", "NTC",
        "8266792", "30588988", 0.325, "downloaded level-0 crop", "phase=0; fluorescence=2; masks",
        "Soft-QC non-edge NTC; phase/geometry medoid within screen.",
        "Local downloaded crop; original OPS source documented in source figure manifest.",
    ),
    Asset(
        "eea1_phase_fold0", "eea1_phase_fold0.npy",
        "supplementary_figure2/b_representation_boundary/source_images/early_endosome_eea1_phase_crop.npy",
        "rendered", "Supplementary Fig.2b",
        "EEA1", "early_endosome_eea1", "", "Biohub_OPS0039", "", "KO",
        "2469794", "", 0.325, "downloaded level-0 crop", "phase only",
        "Closest to reporter-wise median of four phase-image statistics among 96 evenly spaced full-window targeting cells in held-out gene fold 0.",
        "Local downloaded crop; original OPS source documented in the Supplementary Figure 2 source table.",
    ),
    Asset(
        "tomm20_phase_fold0", "tomm20_phase_fold0.npy",
        "supplementary_figure2/b_representation_boundary/source_images/mitochondria_tomm20_phase_crop.npy",
        "rendered", "Supplementary Fig.2b",
        "TOMM20", "mitochondria_tomm20", "", "Biohub_OPS0043", "", "KO",
        "6999129", "", 0.325, "downloaded level-0 crop", "phase only",
        "Closest to reporter-wise median of four phase-image statistics among 96 evenly spaced full-window targeting cells in held-out gene fold 0.",
        "Local downloaded crop; original OPS source documented in the Supplementary Figure 2 source table.",
    ),
    Asset(
        "npm1_phase_fold0", "npm1_phase_fold0.npy",
        "supplementary_figure2/b_representation_boundary/source_images/nucleoli_npm1_phase_crop.npy",
        "rendered", "Supplementary Fig.2b",
        "NPM1", "nucleoli_npm1", "", "Biohub_OPS0043", "", "KO",
        "5976007", "", 0.325, "downloaded level-0 crop", "phase only",
        "Closest to reporter-wise median of four phase-image statistics among 96 evenly spaced full-window targeting cells in held-out gene fold 0.",
        "Local downloaded crop; original OPS source documented in the Supplementary Figure 2 source table.",
    ),
]


RENDER_SPECS = [
    # figure, panel, asset, phase low/high, fluorescence low/high, use
    ("Fig.1", "c", "lamp1_borcs7_ko", -0.3490651994943619, 0.6395067185163511, 106.61254043579102, 211.3301994323731, "same-cell inset"),
    ("Fig.1", "c", "ferhonox_alg12_ko", -0.775179436802864, 1.2993500113487255, 492.0, 4255.0, "same-cell inset"),
    ("Fig.1", "e", "lamp1_borcs7_ko", -0.3490651994943619, 0.6395067185163511, 106.61254043579102, 211.3301994323731, "same-cell response map"),
    ("Fig.1", "e", "ferhonox_alg12_ko", -0.775179436802864, 1.2993500113487255, 492.0, 4255.0, "same-cell response map"),
    ("Fig.1", "e", "prb_fech_ko", -0.5261821895837784, 0.71541805267334, 121.01383781433105, 1530.6487609863318, "same-cell response map"),
    ("Fig.3", "a", "fastact_actr3_ko", -0.7747886502742767, 1.0280601310729984, 173.0, 4551.0, "rank-atlas callout"),
    ("Fig.3", "a", "lamp1_borcs7_ntc", -0.4185225659608841, 0.6370304238796235, 105.15650020599364, 261.90295123291105, "rank-atlas callout"),
    ("Fig.3", "a", "5xupre_hspa5_ko", -0.556718013882637, 0.7848890328407298, 105.53495250701904, 452.9303023071303, "rank-atlas callout"),
    ("Fig.4", "c", "prb_sall3_ops0077_a2", -0.5457885962724686, 0.8108193516731264, 134.47124862670898, 1531.5057202148453, "pRb reconstruction case"),
    ("Fig.5", "b", "prb_sall3_ops0077_a1", -0.5457885962724686, 0.8108193516731264, 134.47124862670898, 1531.5057202148453, "three-well stability case"),
    ("Fig.5", "b", "prb_sall3_ops0077_a2", -0.5457885962724686, 0.8108193516731264, 134.47124862670898, 1531.5057202148453, "three-well stability case"),
    ("Fig.5", "b", "prb_sall3_ops0077_a3", -0.5457885962724686, 0.8108193516731264, 134.47124862670898, 1531.5057202148453, "three-well stability case"),
    ("Fig.5", "d", "ferhonox_ntc_ops0047", -0.3795731547474861, 0.8340505826473242, 112.0, 4465.0, "repeated-screen environment case"),
    ("Fig.5", "d", "ferhonox_ntc_ops0067", -0.3795731547474861, 0.8340505826473242, 112.0, 4465.0, "repeated-screen environment case"),
    ("Supplementary Fig.2", "b", "eea1_phase_fold0", -0.6203947246074677, 1.0548125565052042, "", "", "representation boundary example"),
    ("Supplementary Fig.2", "b", "tomm20_phase_fold0", -0.4085839435458183, 0.7731417328119281, "", "", "representation boundary example"),
    ("Supplementary Fig.2", "b", "npm1_phase_fold0", -0.3872894838452339, 0.5882009357214006, "", "", "representation boundary example"),
]


# Source-package aliases not copied a second time because their raw pixel arrays are
# already represented by the matching canonical asset above.  This makes deduplication
# explicit rather than silently dropping an image candidate from provenance.
DEDUPLICATION_ALIASES = [
    ("lamp1_borcs7_ko", "figure1/c_sparse_assay_topology/source_data/frozen_inputs/microscopy/lamp1_exact_same_cell.npz", "same raw paired channels as Fig.1e source"),
    ("ferhonox_alg12_ko", "figure1/c_sparse_assay_topology/source_data/frozen_inputs/microscopy/ferhonox_exact_same_cell.npz", "same raw paired channels as Fig.1e source"),
    ("lamp1_borcs7_ko", "figure3/source_data/microscopy/lamp1_borcs7.npz", "same raw pixels; retained Fig.3 descriptive future-edit alias"),
    ("lamp1_borcs7_ko", "figure3/source_data/microscopy/lamp1_borcs7_ko.npz", "same raw pixels; NPZ container differs but array-content checksum matches"),
    ("prb_fech_ko", "figure3/source_data/microscopy/prb_fech.npz", "same raw pixels; retained Fig.3 descriptive future-edit alias"),
    ("5xupre_hspa5_ko", "figure3/a_recoverability_rank_atlas/source_data/5xupre_hspa5_ko.npz", "same raw pixels as Fig.3 microscopy source"),
    ("fastact_actr3_ko", "figure3/a_recoverability_rank_atlas/source_data/fastact_actr3_ko.npz", "same raw pixels as Fig.3 microscopy source"),
    ("lamp1_borcs7_ntc", "figure3/a_recoverability_rank_atlas/source_data/lamp1_borcs7_ntc.npz", "same raw pixels as Fig.3 microscopy source"),
    ("prb_sall3_ops0077_a2", "figure4/source_data/panel_c/microscopy/cell_01.npz", "same raw pixels as Fig.5b A/2/0 crop"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(path: Path) -> tuple[str, str, str]:
    """Hash numeric arrays independently of NPZ container compression metadata."""
    digest = hashlib.sha256()
    if path.suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        keys = sorted(payload.files)
        array_spec = []
        for key in keys:
            value = np.ascontiguousarray(payload[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.tobytes(order="C"))
            array_spec.append(f"{key}:{'x'.join(map(str, value.shape))}:{value.dtype}")
    else:
        value = np.ascontiguousarray(np.load(path, allow_pickle=False))
        digest.update(b"array")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.tobytes(order="C"))
        array_spec = [f"array:{'x'.join(map(str, value.shape))}:{value.dtype}"]
    return digest.hexdigest(), "; ".join(array_spec), "float32" if "float32" in ";".join(array_spec) else "mixed"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for asset in ASSETS:
        source = (
            SOURCE_ROOT / asset.source_relative_path
            if asset.source_relative_path.startswith("supplementary_figure2/")
            else FINAL / asset.source_relative_path
        )
        destination = RAW_DIR / asset.filename
        if not source.exists():
            raise FileNotFoundError(f"Missing frozen microscopy source: {source}")
        shutil.copy2(source, destination)
        source_sha = sha256_file(source)
        destination_sha = sha256_file(destination)
        if source_sha != destination_sha:
            raise RuntimeError(f"Copy checksum mismatch for {asset.asset_id}")
        pixels_sha, array_spec, dtype_class = pixel_sha256(destination)
        metadata = {
            "asset_id": asset.asset_id,
            "canonical_relative_path": f"raw_assets/{asset.filename}",
            "file_sha256": destination_sha,
            "pixel_sha256": pixels_sha,
            "array_spec": array_spec,
            "spatial_transform": "none",
            "orientation": "native stored orientation",
            "display_policy": "Use the per-panel fixed windows in render_spec.csv. Do not alter pixels, register, denoise, enhance, or content-edit this asset.",
            "provenance": asdict(asset),
        }
        (METADATA_DIR / f"{asset.asset_id}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record = asdict(asset)
        record.update(
            canonical_relative_path=f"raw_assets/{asset.filename}",
            bytes=destination.stat().st_size,
            file_sha256=destination_sha,
            pixel_sha256=pixels_sha,
            array_spec=array_spec,
            dtype_class=dtype_class,
        )
        manifest_rows.append(record)

    manifest_fields = [
        "asset_id", "filename", "canonical_relative_path", "use_status", "figure_panel_usage",
        "reporter", "reporter_slug", "gene", "screen_id", "well", "perturbation_role",
        "phase_row_index", "segmentation_id", "pixel_size_um", "source_level", "channels",
        "selection_rule", "provenance_uri", "source_relative_path", "file_sha256",
        "pixel_sha256", "array_spec", "dtype_class", "bytes", "note",
    ]
    write_csv(ROOT / "asset_manifest.csv", manifest_rows, manifest_fields)

    render_rows = [
        {
            "figure": figure,
            "panel": panel,
            "asset_id": asset_id,
            "phase_display_low": phase_low,
            "phase_display_high": phase_high,
            "fluorescence_display_low": fluor_low,
            "fluorescence_display_high": fluor_high,
            "display_role": role,
            "spatial_transform": "none",
            "orientation": "native stored orientation",
            "source_pixel_size_um": 0.325,
            "scale_bar_um": 20 if figure in {"Fig.1", "Fig.3", "Fig.4", "Fig.5"} else "",
        }
        for figure, panel, asset_id, phase_low, phase_high, fluor_low, fluor_high, role in RENDER_SPECS
    ]
    write_csv(
        ROOT / "render_spec.csv", render_rows,
        ["figure", "panel", "asset_id", "phase_display_low", "phase_display_high",
         "fluorescence_display_low", "fluorescence_display_high", "display_role",
         "spatial_transform", "orientation", "source_pixel_size_um", "scale_bar_um"],
    )

    alias_rows = []
    by_id = {row["asset_id"]: row for row in manifest_rows}
    for asset_id, source_relative_path, reason in DEDUPLICATION_ALIASES:
        alias_source = (
            SOURCE_ROOT / source_relative_path
            if source_relative_path.startswith("supplementary_figure2/")
            else FINAL / source_relative_path
        )
        if not alias_source.exists():
            raise FileNotFoundError(f"Missing deduplication alias source: {alias_source}")
        alias_pixels_sha, _, _ = pixel_sha256(alias_source)
        if alias_pixels_sha != by_id[asset_id]["pixel_sha256"]:
            raise RuntimeError(f"Pixel checksum differs for alias {source_relative_path}")
        alias_rows.append({
            "canonical_asset_id": asset_id,
            "canonical_relative_path": by_id[asset_id]["canonical_relative_path"],
            "alias_source_relative_path": source_relative_path,
            "alias_pixel_sha256": alias_pixels_sha,
            "deduplication_reason": reason,
        })
    write_csv(
        ROOT / "deduplication_map.csv", alias_rows,
        ["canonical_asset_id", "canonical_relative_path", "alias_source_relative_path", "alias_pixel_sha256", "deduplication_reason"],
    )

    checksummed = [
        *sorted(RAW_DIR.glob("*")),
        *sorted(METADATA_DIR.glob("*.json")),
        ROOT / "asset_manifest.csv",
        ROOT / "render_spec.csv",
        ROOT / "deduplication_map.csv",
    ]
    with (ROOT / "asset_checksums.sha256").open("w", encoding="utf-8") as handle:
        for path in checksummed:
            handle.write(f"{sha256_file(path)}  {path.relative_to(ROOT)}\n")

    audit = {
        "status": "PASS",
        "bundle_scope": "All real microscopy source arrays rendered in final Figures 1–6 plus paired/reusable figure-edit counterparts.",
        "asset_count": len(ASSETS),
        "rendered_asset_count": sum("rendered" in a.use_status for a in ASSETS),
        "potential_only_asset_count": sum(a.use_status == "potential" for a in ASSETS),
        "stored_formats": {"npz": sum(a.filename.endswith(".npz") for a in ASSETS), "npy": sum(a.filename.endswith(".npy") for a in ASSETS)},
        "native_crop_geometry": "384 x 384 pixels; float32 arrays; 0.325 micrometres per pixel where recorded.",
        "deduplication": "One canonical copy per unique raw crop; shared figure uses are represented in figure_panel_usage rather than duplicate files.",
        "deduplicated_source_aliases": len(DEDUPLICATION_ALIASES),
        "derivative_policy": "No PNG/JPEG screenshot or enhancement is stored in this bundle.",
        "known_provenance_boundary": "A small number of legacy Fig.1/final-stage diagnostic crops are lossless paired arrays with frozen case metadata; their original local staging shard is not encoded in the final figure manifest. The public OPS Zarr location, screen, well, cell ID, crop geometry and channel mapping are recorded whenever available.",
    }
    (ROOT / "bundle_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
