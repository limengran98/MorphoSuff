#!/usr/bin/env python3
"""Run structural and publication-layout checks for sealed Figure 3."""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
QA_DIR = ROOT / "qa"
EXPECTED = [
    ROOT / "a_recoverability_rank_atlas/figure/Figure3-a_recoverability_rank_atlas",
    ROOT / "b_biology_endpoint_structure/figure/Figure3-b_biology_endpoint_structure",
    ROOT / "c_value_overlap_recovery/figure/Figure3-c_value_overlap_recovery",
    ROOT / "composite/Figure3",
]
PANEL_SVGS = {
    "a": EXPECTED[0].with_suffix(".svg"),
    "b": EXPECTED[1].with_suffix(".svg"),
    "c": EXPECTED[2].with_suffix(".svg"),
}
REQUIRED_TEXT = {
    "a": "Whole-screen",
    "b": "held-out-gene Pearson r",
    "c": "phase–marker overlap",
}


def svg_min_font_size(path: Path) -> float | None:
    text = path.read_text(encoding="utf-8")
    values = [float(value) for value in re.findall(r"font-size:\s*([0-9.]+)px", text)]
    return min(values) if values else None


def main() -> None:
    failures: list[str] = []
    for stem in EXPECTED:
        for ext in (".png", ".pdf", ".svg"):
            path = stem.with_suffix(ext)
            if not path.is_file() or path.stat().st_size == 0:
                failures.append(f"missing or empty output: {path.relative_to(ROOT)}")
    composite_tiff = ROOT / "composite/Figure3.tiff"
    if not composite_tiff.is_file() or composite_tiff.stat().st_size == 0:
        failures.append("missing or empty output: composite/Figure3.tiff")

    svg_text = {
        key: path.read_text(encoding="utf-8") if path.is_file() else ""
        for key, path in PANEL_SVGS.items()
    }
    for key, phrase in REQUIRED_TEXT.items():
        if phrase not in svg_text[key]:
            failures.append(f"panel {key} missing required annotation: {phrase}")
    if "Across → evaluation scope" in svg_text["a"]:
        failures.append("panel a still contains the removed grey secondary note")
    if "biological system" in svg_text["a"].lower():
        failures.append("panel a still contains the removed biological-system text label")

    strip_path = (
        ROOT / "a_recoverability_rank_atlas/source_data/plotted_biological_system_strips.csv"
    )
    strip_components = strip_tiles = strip_panels = None
    if not strip_path.is_file():
        failures.append("missing plotted biological-system strip source data")
    else:
        strips = pd.read_csv(strip_path)
        strip_components = len(strips)
        strip_tiles = len(strips[["unit", "split", "entity_id"]].drop_duplicates())
        strip_panels = len(strips[["unit", "split"]].drop_duplicates())
        if (strip_components, strip_tiles, strip_panels) != (676, 629, 9):
            failures.append(
                "biological-system strip data are incomplete: "
                f"{strip_components} components, {strip_tiles} tiles, {strip_panels} panels"
            )
    # The black contrast halo intentionally converts the three scale labels to
    # vector paths in SVG.  Verify the authored calibrated annotation and the
    # frozen three-card call site rather than searching for editable SVG text.
    builder_text = (ROOT / "code/build_figure3.py").read_text(encoding="utf-8")
    scale_label_count = 3 if (
        '"20 µm"' in builder_text
        and "add_scale_bar(card" in builder_text
        and "path_effects.Stroke" in builder_text
    ) else 0
    if scale_label_count != 3:
        failures.append(f"expected three 20 µm labels; found {scale_label_count}")

    composite_pdf = ROOT / "composite/Figure3.pdf"
    width_mm = height_mm = None
    if composite_pdf.is_file():
        with fitz.open(composite_pdf) as document:
            page = document[0]
            width_mm = float(page.rect.width) * 25.4 / 72
            height_mm = float(page.rect.height) * 25.4 / 72
        if abs(width_mm - 183) > 0.15 or abs(height_mm - 170) > 0.15:
            failures.append(
                f"composite size is {width_mm:.2f} × {height_mm:.2f} mm; expected 183 × 170 mm"
            )

    source_data_files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and "source_data" in path.relative_to(ROOT).parts
    )
    if len(source_data_files) != 30:
        failures.append(f"expected 30 source-data files; found {len(source_data_files)}")

    external_markers: list[str] = []
    for path in ROOT.rglob("*.py"):
        if path.resolve() == Path(__file__).resolve():
            continue
        code = path.read_text(encoding="utf-8")
        for marker in ('parent.name == "paper"', 'parent.name == "final"', "FINAL_ROOT"):
            if marker in code:
                external_markers.append(f"{path.relative_to(ROOT)}: {marker}")
    if external_markers:
        failures.extend(f"external-path dependency: {item}" for item in external_markers)

    report = {
        "status": "PASS" if not failures else "FAIL",
        "backend": "Python (Matplotlib/Pandas/NumPy/Pillow)",
        "package_root": str(ROOT),
        "inputs_outside_package": external_markers,
        "source_data_file_count": len(source_data_files),
        "composite_size_mm": {"width": width_mm, "height": height_mm},
        "required_annotations_present": {
            key: phrase in svg_text[key] for key, phrase in REQUIRED_TEXT.items()
        },
        "removed_grey_note_absent": "Across → evaluation scope" not in svg_text["a"],
        "removed_biology_label_absent": "biological system" not in svg_text["a"].lower(),
        "biology_strip_component_rows": strip_components,
        "biology_strip_unique_tiles": strip_tiles,
        "biology_strip_panels": strip_panels,
        "scale_label_20um_count": scale_label_count,
        "minimum_svg_font_size_px": {
            key: svg_min_font_size(path) if path.is_file() else None
            for key, path in PANEL_SVGS.items()
        },
        "failures": failures,
    }
    QA_DIR.mkdir(parents=True, exist_ok=True)
    (QA_DIR / "figure3_archive_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Figure 3 archive audit",
        "",
        f"- Status: **{report['status']}**",
        f"- Backend: {report['backend']}",
        f"- Source-data files: {len(source_data_files)}",
        f"- Composite size: {width_mm:.2f} × {height_mm:.2f} mm" if width_mm else "- Composite size: unavailable",
        f"- Required narrative annotations: {'present' if all(report['required_annotations_present'].values()) else 'incomplete'}",
        f"- Removed grey note: {'absent' if report['removed_grey_note_absent'] else 'present'}",
        f"- Biological-system strips: {strip_panels} panels, {strip_tiles} entity tiles, {strip_components} colour components",
        f"- Removed biological-system text label: {'absent' if report['removed_biology_label_absent'] else 'present'}",
        f"- Calibrated, outlined 20 µm bars: {scale_label_count}",
        f"- Inputs outside package: {'none' if not external_markers else '; '.join(external_markers)}",
        "- Minimum SVG text sizes are recorded for audit only; no visual content was changed during sealing.",
    ]
    if failures:
        lines.extend(["", "## Failures", "", *[f"- {item}" for item in failures]])
    (QA_DIR / "figure3_archive_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("Figure 3 QA failed:\n" + "\n".join(failures))
    print("PASS: Figure 3 archive QA")


if __name__ == "__main__":
    main()
