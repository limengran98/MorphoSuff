#!/usr/bin/env python3
"""Verify self-containment and the approved Figure 4 spacing state."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

from PIL import Image
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "qa"
EXPECTED_STEMS = [
    ROOT / "a_intervention_terrain/figure4a_intervention_terrain",
    ROOT / "b_residual_barcode/figure4b_residual_barcode",
    ROOT / "c_reconstruction_modes/Figure4-c",
    ROOT / "d_screen_transfer/figure4d_screen_transfer",
    ROOT / "composite/Figure4",
]


def svg_min_font_size(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8")
    values = [float(value) for value in re.findall(r"font-size:\s*([0-9.]+)px", text)]
    return min(values) if values else None


def main() -> None:
    failures: list[str] = []
    for stem in EXPECTED_STEMS:
        for ext in (".png", ".pdf", ".svg"):
            path = stem.with_suffix(ext)
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing or empty output: {path.relative_to(ROOT)}")

    source_files = sorted(path for path in (ROOT / "source_data").rglob("*") if path.is_file())
    if len(source_files) != 28:
        failures.append(f"expected 28 frozen source-data files; found {len(source_files)}")

    external_markers: list[str] = []
    for path in ROOT.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        code = path.read_text(encoding="utf-8")
        for marker in ("FINAL_ROOT", "PAPER_ROOT", 'parent.name == "final"'):
            if marker in code:
                external_markers.append(f"{path.relative_to(ROOT)}: {marker}")
    if external_markers:
        failures.extend(f"external-path dependency: {item}" for item in external_markers)

    panel_d_code = (ROOT / "code/build_panel_d.py").read_text(encoding="utf-8")
    composite_code = (ROOT / "code/build_figure4.py").read_text(encoding="utf-8")
    assignments = {
        node.targets[0].id: node.value for node in ast.parse(composite_code).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    positions = ast.literal_eval(assignments["LETTER_POSITIONS"])
    approved_spacing = {
        "panel_d_outer_hspace_0_24_occurrences": panel_d_code.count("hspace=0.24"),
        "panel_c_composite_rect": "(0.018, 0.365, 0.945, 0.350)" in composite_code,
        "panel_d_composite_rect": "(0.105, 0.048, 0.845, 0.277)" in composite_code,
        "left_column_letter_positions": positions == {
            "a": (0.008, 0.988), "b": (0.662, 0.988),
            "c": (0.008, 0.723), "d": (0.008, 0.315),
        },
    }
    if approved_spacing["panel_d_outer_hspace_0_24_occurrences"] != 2:
        failures.append("approved panel-d hspace=0.24 is not present in both layout paths")
    if not approved_spacing["panel_d_composite_rect"]:
        failures.append("approved panel-d composite rectangle is absent")
    if not approved_spacing["panel_c_composite_rect"]:
        failures.append("approved panel-c composite rectangle is absent")
    if not approved_spacing["left_column_letter_positions"]:
        failures.append("approved common a/c/d letter alignment is absent")

    composite_pdf = ROOT / "composite/Figure4.pdf"
    width_mm = height_mm = None
    if composite_pdf.is_file():
        page = PdfReader(str(composite_pdf)).pages[0]
        width_mm = float(page.mediabox.width) * 25.4 / 72
        height_mm = float(page.mediabox.height) * 25.4 / 72
        if abs(width_mm - 183) > 0.15 or abs(height_mm - 225) > 0.15:
            failures.append(
                f"composite size is {width_mm:.2f} × {height_mm:.2f} mm; expected 183 × 225 mm"
            )

    composite_png = ROOT / "composite/Figure4.png"
    png_geometry = None
    if composite_png.is_file():
        with Image.open(composite_png) as image:
            png_geometry = {"width_px": image.width, "height_px": image.height, "dpi": image.info.get("dpi")}
        if abs(png_geometry["width_px"] - 183 / 25.4 * 600) > 2 or abs(png_geometry["height_px"] - 225 / 25.4 * 600) > 2:
            failures.append(f"unexpected composite PNG geometry: {png_geometry}")

    # The contrast halo converts the scale label to vector paths in SVG, so
    # verify the single authored calibrated call in the released panel source.
    panel_c_source = (ROOT / "code/build_panel_c.py").read_text(encoding="utf-8")
    scale_label_count = 1 if (
        '"20 µm"' in panel_c_source
        and "bar_px = 20.0 / float(display.pixel_size_um)" in panel_c_source
        and "path_effects.Stroke" in panel_c_source
    ) else 0
    if scale_label_count != 1:
        failures.append(f"expected one calibrated 20 µm label in panel c; found {scale_label_count}")

    svg_font_sizes = {
        stem.relative_to(ROOT).as_posix(): svg_min_font_size(stem.with_suffix(".svg"))
        for stem in EXPECTED_STEMS if stem.with_suffix(".svg").is_file()
    }
    report = {
        "status": "PASS" if not failures else "FAIL",
        "backend": "Python (Matplotlib/Pandas/NumPy/SciPy/Pillow)",
        "package_root": str(ROOT),
        "inputs_outside_package": external_markers,
        "frozen_source_data_file_count": len(source_files),
        "composite_size_mm": {"width": width_mm, "height": height_mm},
        "composite_png_geometry": png_geometry,
        "approved_spacing": approved_spacing,
        "calibrated_20um_label_count": scale_label_count,
        "minimum_svg_font_size_px": svg_font_sizes,
        "failures": failures,
    }
    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "figure4_archive_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Figure 4 archive audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Backend: {report['backend']}",
        f"- Frozen source-data files: {len(source_files)}",
        f"- Composite size: {width_mm:.2f} × {height_mm:.2f} mm" if width_mm else "- Composite size: unavailable",
        f"- Approved panel-d spacing state: {'present' if all(value == 2 if key.endswith('occurrences') else value for key, value in approved_spacing.items()) else 'incomplete'}",
        f"- Calibrated, outlined 20 µm bars: {scale_label_count}",
        f"- Inputs outside package: {'none' if not external_markers else '; '.join(external_markers)}",
        "- Minimum SVG text sizes are recorded for audit only; sealing did not alter figure content.",
    ]
    if failures:
        lines.extend(["", "## Failures", "", *[f"- {item}" for item in failures]])
    (QA_DIR / "figure4_archive_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("Figure 4 archive QA failed:\n" + "\n".join(failures))
    print("PASS: Figure 4 archive QA")


if __name__ == "__main__":
    main()
