#!/usr/bin/env python3
"""Audit hybrid PowerPoint/Python Figure 1 and its actual manuscript print scale."""
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
MIN_QUANTITATIVE_FONT_PT = 6.0
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
QA = ROOT / "qa"
PANELS = {
    "a": ("a_workflow", "Figure1-a_workflow"),
    "b": ("b_exact_linkage_scale", "Figure1-b_exact_linkage_scale"),
    "c": ("c_sparse_assay_topology", "Figure1-c_assay_topology"),
    "d": ("d_ko_response_atlas", "Figure1-d_ko_response_atlas"),
    "e": ("e_same_cell_response_maps", "Figure1-e_same_cell_response_maps"),
}

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, help="Active/staged manuscript.tex for placement QA.")
    args = parser.parse_args()
    missing = []
    for letter, (folder, stem) in PANELS.items():
        for suffix in ("pdf", "svg", "png"):
            path = ROOT / folder / "figure" / f"{stem}.{suffix}"
            if not path.is_file() or not path.stat().st_size:
                missing.append(path.relative_to(ROOT).as_posix())
    pdf = ROOT / "composite/Figure1.pdf"
    png = ROOT / "composite/Figure1.png"
    for path in (pdf, png):
        if not path.is_file() or not path.stat().st_size:
            missing.append(path.relative_to(ROOT).as_posix())
    if missing:
        raise SystemExit("missing or empty outputs:\n" + "\n".join(missing))
    panels = {k: pdf_evidence(ROOT / folder / "figure" / f"{stem}.pdf")
              for k, (folder, stem) in PANELS.items()}
    evidence = pdf_evidence(pdf)
    if not evidence["font_sizes_points"]:
        raise SystemExit("no live PDF text found")
    text = normalize(evidence["text"])
    required = ["distinct phase cells", "Reporter–screen assays (rank)", "Endpoints per reporter",
                "Screens per reporter", "172D phase", "24D target", "30D target",
                "Pooled CRISPR", "Cell segmentation", "In situ sequencing", "Same cell ID",
                "52 reporters", "across screens", "Reporter-specific"]
    text_checks = {label: normalize(label) in text for label in required}
    binding = binding_check(ROOT, 1, ROOT / "a_workflow/figure/Figure1-a_workflow.pdf")
    placement = placement_from_tex(locate_manuscript(ROOT, args.manuscript), "Figure1.pdf",
                                   *evidence["page_points"])
    source_files = [p for p in ROOT.rglob("*") if p.is_file() and "source_data" in p.parts]
    quantitative_svgs = [ROOT / PANELS[k][0] / "figure" / f"{PANELS[k][1]}.svg" for k in "bcde"]
    checks = {
        "composite_width_is_183mm": abs(evidence["page_mm"][0] - 183) < 0.1,
        "resolved_composite_text_at_least_5pt": min(evidence["font_sizes_points"]) >= MIN_FONT_PT - 0.01,
        "quantitative_panel_text_at_least_6pt": all(
            min(panels[k]["font_sizes_points"]) >= MIN_QUANTITATIVE_FONT_PT - 0.01
            for k in "bcde"),
        "required_pdf_labels_extractable": all(text_checks.values()),
        "all_panel_outputs_complete": len(panels) == 5,
        "quantitative_panel_svgs_have_live_text": all("<text" in p.read_text(encoding="utf-8") for p in quantitative_svgs),
        "panel_a_source_and_pdf_binding_verified": all(binding["checks"].values()),
        "intentional_panel_a_images_present": bool(panels["a"]["images"]),
        "panel_a_images_at_least_300dpi": bool(panels["a"]["images"]) and all(
            min(im["authored_dpi"]) >= 300 for im in panels["a"]["images"]),
        "all_pdf_fonts_embedded": bool(evidence["fonts"]) and all(f["embedded"] for f in evidence["fonts"]),
        "no_type3_fonts": all("type3" not in f["type"].lower() for f in evidence["fonts"]),
        "frozen_source_data_complete": len(source_files) >= 73,
        "composite_preview_is_600dpi": Image.open(png).info.get("dpi", (0, 0))[0] >= 599,
    }
    minimum_placed = None
    if placement["status"] == "checked":
        minimum_placed = min(evidence["font_sizes_points"]) * placement["placement_scale"]
        checks["placed_text_at_least_5pt"] = minimum_placed >= MIN_FONT_PT - 0.01
        checks["placed_text_at_least_5_5pt"] = minimum_placed >= MIN_PLACED_FONT_PT - 0.01
    report = {"status": ("PASS" if placement["status"] == "checked" else "PASS_AUTHORED_ONLY")
              if all(checks.values()) else "FAIL", "pdf": evidence, "placement": placement,
              "minimum_placed_font_pt": minimum_placed, "checks": checks,
              "required_pdf_text": text_checks, "panel_a_binding": binding,
              "source_data_files": len(source_files),
              "panel_a_image_placements": panels["a"]["images"],
              "panel_minimum_resolved_pdf_font_pt": {k: min(v["font_sizes_points"]) for k, v in panels.items()}}
    QA.mkdir(exist_ok=True)
    (QA / "figure1_archive_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = report_markdown("Figure 1 source and print-scale audit", report)
    (QA / "figure1_archive_audit.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0 if all(checks.values()) else 1

if __name__ == "__main__":
    raise SystemExit(main())
