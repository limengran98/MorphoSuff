#!/usr/bin/env python3
"""Audit transformed Figure 2 text and accepted PPT source, not raster-free artwork."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import fitz
from PIL import Image

MIN_FONT_PT = 5.0
MIN_QUANTITATIVE_FONT_PT = 6.5
MIN_PLACED_FONT_PT = 5.5
MM_PER_PT = 25.4 / 72.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_manuscript(root: Path, supplied: Path | None) -> Path | None:
    if supplied is not None:
        if not supplied.is_file():
            raise SystemExit(f"missing manuscript source: {supplied}")
        return supplied
    return next((parent / "manuscript.tex" for parent in root.parents
                 if (parent / "manuscript.tex").is_file()), None)


def placement_from_tex(path: Path | None, name: str, width: float, height: float) -> dict:
    """Read the active TeX contract; never infer an obsolete figure height cap."""
    if path is None:
        return {"status": "not_checked", "reason": "No manuscript.tex found; pass --manuscript for print-scale QA."}
    tex = re.sub(r"(?<!\\)%.*", "", path.read_text(encoding="utf-8"))
    if "letterpaper" in tex:
        page_w, page_h = 215.9, 279.4
    elif "a4paper" in tex:
        page_w, page_h = 210.0, 297.0
    else:
        raise ValueError("unsupported paper size; declare letterpaper/a4paper explicitly")
    geometry = re.search(r"\\usepackage\[([^]]+)\]\{geometry\}", tex)
    if not geometry:
        raise ValueError("missing explicit geometry package options")
    options = geometry.group(1)
    def margin(name: str) -> float:
        match = re.search(rf"(?:^|,)\s*{name}\s*=\s*([0-9.]+)\s*(cm|mm|in|pt)", options)
        if not match:
            raise ValueError(f"missing explicit geometry {name} margin")
        return float(match.group(1)) * {"cm": 10, "mm": 1, "in": 25.4, "pt": MM_PER_PT}[match.group(2)]
    text_w = (page_w - margin("left") - margin("right")) / MM_PER_PT
    text_h = (page_h - margin("top") - margin("bottom")) / MM_PER_PT
    call = re.search(r"\\includegraphics\[([^]]+)\]\{" + re.escape(name) + r"\}", tex)
    if not call:
        raise ValueError(f"cannot locate includegraphics for {name}")
    opts = call.group(1)
    def fraction(axis: str, macro: str) -> float | None:
        match = re.search(rf"{axis}\s*=\s*([0-9.]*)\\{macro}\b", opts)
        return (float(match.group(1)) if match.group(1) else 1.0) if match else None
    width_fraction = fraction("width", "textwidth")
    height_fraction = fraction("height", "textheight")
    limits = []
    if width_fraction is not None:
        limits.append(width_fraction * text_w / width)
    if height_fraction is not None:
        limits.append(height_fraction * text_h / height)
    for axis, authored in (("width", width), ("height", height)):
        literal = re.search(rf"{axis}\s*=\s*([0-9.]+)\s*(mm|cm|in|pt)\b", opts)
        if literal:
            points = float(literal.group(1)) * {"mm": 1 / MM_PER_PT,
                "cm": 10 / MM_PER_PT, "in": 72, "pt": 1}[literal.group(2)]
            limits.append(points / authored)
    if not limits or "keepaspectratio" not in opts:
        raise ValueError("figure placement needs an explicit size and keepaspectratio")
    scale = min(limits)
    return {
        "status": "checked", "manuscript_source": str(path),
        "includegraphics_options": opts,
        "text_width_points": text_w, "text_height_points": text_h,
        "placement_scale": scale, "placed_size_mm": [width * scale * MM_PER_PT, height * scale * MM_PER_PT],
    }


def pdf_evidence(path: Path) -> dict:
    """MuPDF resolves nested Form/text matrices, including imported PPT artwork."""
    with fitz.open(path) as document:
        if len(document) != 1:
            raise ValueError(f"expected one page: {path}")
        page = document[0]
        spans = [span for block in page.get_text("dict")["blocks"] if block["type"] == 0
                 for line in block["lines"] for span in line["spans"] if span["text"].strip()]
        sizes = sorted({round(span["size"], 6) for span in spans})
        font_info = []
        for font in page.get_fonts(full=True):
            xref, extension, kind, basefont, resource = font[:5]
            payload = document.extract_font(xref)[3] if xref else b""
            font_info.append({"xref": xref, "type": kind, "basefont": basefont,
                              "resource": resource, "embedded": bool(payload)})
        images = []
        for item in page.get_image_info(xrefs=True):
            bbox = fitz.Rect(item["bbox"])
            if bbox.width > 0 and bbox.height > 0:
                images.append({
                    "width_pixels": item["width"], "height_pixels": item["height"],
                    "bbox_points": list(bbox),
                    "authored_dpi": [item["width"] * 72 / bbox.width, item["height"] * 72 / bbox.height],
                })
        return {"page_points": [page.rect.width, page.rect.height],
                "page_mm": [page.rect.width * MM_PER_PT, page.rect.height * MM_PER_PT],
                "text": page.get_text(), "font_sizes_points": sizes, "fonts": font_info, "images": images}


def binding_check(root: Path, figure: int, panel_pdf: Path) -> dict:
    source = root / "a_workflow" / "source"
    pptx = source / f"Figure{figure}a.pptx"
    manifest = source / "export_manifest.json"
    checks = {"pptx_exists": pptx.is_file(), "manifest_exists": manifest.is_file()}
    details = {"source": pptx.relative_to(root).as_posix(),
               "manifest": manifest.relative_to(root).as_posix(), "checks": checks}
    if not all(checks.values()):
        return details
    record = json.loads(manifest.read_text(encoding="utf-8-sig"))
    pptx_hash, pdf_hash = sha256(pptx), sha256(panel_pdf)
    frozen_pdf = root / "a_workflow" / record["exported_pdf"]
    checks.update({
        "current_pptx_sha256_bound": pptx_hash == record.get("source_pptx_sha256"),
        "current_panel_pdf_sha256_bound": pdf_hash == record.get("exported_pdf_sha256"),
        "versioned_frozen_pdf_exists": frozen_pdf.is_file(),
        "generated_panel_matches_frozen_export": frozen_pdf.is_file() and sha256(frozen_pdf) == pdf_hash,
    })
    details.update({"pptx_sha256": pptx_hash, "panel_pdf_sha256": pdf_hash})
    return details


def normalize(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").replace("\u00d7", "×").split())


def report_markdown(title: str, report: dict) -> str:
    geometry = report["pdf"]
    lines = [f"# {title}", "", f"- Status: **{report['status']}**",
             f"- Composite: {geometry['page_mm'][0]:.3f} × {geometry['page_mm'][1]:.3f} mm.",
             f"- Minimum resolved PDF text: {min(geometry['font_sizes_points']):.3f} pt.",
             "- PDF text sizes include all composition and nested-form transformations.",
             f"- Embedded image placements: {len(geometry['images'])}; microscopy and new schematic assets are intentional.",
             "- Editable panel-a source: accepted PowerPoint, SHA-256 bound to its frozen PDF export."]
    placement = report["placement"]
    if placement["status"] == "checked":
        lines += [f"- Manuscript scale: {placement['placement_scale']:.5f}.",
                  f"- Minimum placed text: {report['minimum_placed_font_pt']:.3f} pt."]
    else:
        lines += ["- Manuscript print scale: **not checked**; pass --manuscript PATH before publication."]
    lines += ["", "## Checks", ""]
    lines += [f"- {'PASS' if passed else 'FAIL'}: {key.replace('_', ' ')}"
              for key, passed in report["checks"].items()]
    lines += ["", "This is mechanical QA, not proof of absence of visual overlaps. Inspect the full-size",
              "PDF/PNG and the compiled manuscript figure at its actual placement before release.", ""]
    return "\n".join(lines)

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "composite/Figure2.pdf"
SVG = ROOT / "composite/Figure2.svg"
PNG = ROOT / "composite/Figure2.png"
PANELS = ROOT / "panels"
QA = ROOT / "qa"

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, help="Active/staged manuscript.tex for placement QA.")
    args = parser.parse_args()
    for path in (PDF, SVG, PNG):
        if not path.is_file():
            raise SystemExit(f"missing required output: {path}")
    panel_files = [PANELS / letter / f"Figure2{letter}.{suffix}" for letter in "abcd"
                   for suffix in ("pdf", "svg", "png")]
    evidence = pdf_evidence(PDF)
    if not evidence["font_sizes_points"]:
        raise SystemExit("no live PDF text found")
    placement = placement_from_tex(locate_manuscript(ROOT, args.manuscript), "Figure2.pdf",
                                   *evidence["page_points"])
    binding = binding_check(ROOT, 2, PANELS / "a/Figure2a.pdf")
    text = normalize(evidence["text"])
    required = ["Predictor", "Example cell crops",
                "172 features", "20–72 endpoints", "Field", "Gene", "Whole-screen",
                "KO Pearson r", "Cell Pearson r", "screens ranked by consensus", "99 assays",
                "LysoTracker", "LAMP1", "FeRhoNox", "pRb"]
    text_checks = {label: normalize(label) in text for label in required}
    a_evidence = pdf_evidence(PANELS / "a/Figure2a.pdf")
    quantitative_minima = {
        letter: min(pdf_evidence(PANELS / letter / f"Figure2{letter}.pdf")["font_sizes_points"])
        for letter in "bcd"
    }
    svg_text = SVG.read_text(encoding="utf-8")
    quantitative_svgs = [PANELS / letter / f"Figure2{letter}.svg" for letter in "bcd"]
    png = Image.open(PNG)
    minimum_placed = None
    checks = {
        "resolved_composite_text_at_least_5pt": min(evidence["font_sizes_points"]) >= MIN_FONT_PT - 0.01,
        "composite_width_is_183mm": abs(evidence["page_mm"][0] - 183) < 0.1,
        "required_pdf_labels_extractable": all(text_checks.values()),
        "intentional_panel_a_images_present": bool(a_evidence["images"]),
        "panel_a_images_at_least_300dpi": bool(a_evidence["images"]) and all(
            min(im["authored_dpi"]) >= 300 for im in a_evidence["images"]),
        "panel_a_source_and_pdf_binding_verified": all(binding["checks"].values()),
        "composite_svg_present": "<svg" in svg_text,
        "quantitative_panel_svgs_have_live_text": all(p.is_file() and "<text" in p.read_text(encoding="utf-8")
                                                    for p in quantitative_svgs),
        "quantitative_panel_text_at_least_6_5pt": all(
            size >= MIN_QUANTITATIVE_FONT_PT - 0.01 for size in quantitative_minima.values()),
        "standalone_panels_complete": all(p.is_file() and p.stat().st_size > 0 for p in panel_files),
        "all_pdf_fonts_embedded": bool(evidence["fonts"]) and all(f["embedded"] for f in evidence["fonts"]),
        "no_type3_fonts": all("type3" not in f["type"].lower() for f in evidence["fonts"]),
        "png_is_600dpi": png.info.get("dpi", (0, 0))[0] >= 599,
    }
    if placement["status"] == "checked":
        minimum_placed = min(evidence["font_sizes_points"]) * placement["placement_scale"]
        checks["placed_text_at_least_5pt"] = minimum_placed >= MIN_FONT_PT - 0.01
        checks["placed_text_at_least_5_5pt"] = minimum_placed >= MIN_PLACED_FONT_PT - 0.01
    report = {"status": ("PASS" if placement["status"] == "checked" else "PASS_AUTHORED_ONLY")
              if all(checks.values()) else "FAIL", "pdf": evidence, "placement": placement,
              "minimum_placed_font_pt": minimum_placed, "checks": checks,
              "required_pdf_text": text_checks, "panel_a_binding": binding,
              "standalone_panel_files": len(panel_files), "png_pixels": list(png.size),
              "quantitative_panel_minimum_authored_font_pt": quantitative_minima,
              "png_dpi": list(png.info.get("dpi", (0, 0))),
              "panel_a_minimum_pdf_font_pt": min(a_evidence["font_sizes_points"]),
              "panel_a_minimum_authored_image_dpi": min(min(im["authored_dpi"]) for im in a_evidence["images"])
              if a_evidence["images"] else None}
    QA.mkdir(exist_ok=True)
    scale = placement.get("placement_scale", 1.0)
    proof_size = tuple(round(value / 72 * scale * 300) for value in evidence["page_points"])
    png.convert("RGB").resize(proof_size, Image.Resampling.LANCZOS).save(
        QA / "Figure2_printproof_300dpi.png", dpi=(300, 300))
    (QA / "figure2_printscale_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = report_markdown("Figure 2 source and print-scale audit", report)
    (QA / "figure2_printscale_audit.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if all(checks.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
