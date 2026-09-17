#!/usr/bin/env python3
"""Verify final-size vector/raster exports and numerical key provenance.

Use --visual-reviewed only after inspecting the composite and standalone PNGs
and a rendered PDF. Automated checks cannot replace visual inspection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import fitz
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1]
DATA = PAPER / "supplementary_data/SupplementaryData1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-reviewed", action="store_true")
    args = parser.parse_args()
    evidence = pd.read_csv(DATA / "evidence.csv").set_index(["run_state", "reporter_slug", "method_id"])
    keys = pd.read_csv(ROOT / "source_data/plotted_point_keys.csv")
    assert len(keys) == 52*11 + 3*4
    for row in keys.itertuples(index=False):
        evidence.loc[(row.run_state, row.reporter_slug, row.method_id)]
    exports = [
        (ROOT/"a_predictor_tiers/SupplementaryFigure3-a", 112, []),
        (ROOT/"b_lamp1_gates/SupplementaryFigure3-b", 48, []),
        (ROOT/"composite/SupplementaryFigure3", 162, ["a", "b"]),
    ]
    checks = []
    for stem, height, letters in exports:
        document = fitz.open(stem.with_suffix(".pdf"))
        assert len(document) == 1
        page = document[0]
        assert abs(page.rect.width*25.4/72-183) < .02
        assert abs(page.rect.height*25.4/72-height) < .02
        spans = [s for block in page.get_text("dict")["blocks"] for line in block.get("lines", []) for s in line["spans"] if s["text"].strip()]
        assert min(s["size"] for s in spans) >= 5.99
        actual_letters = [s for s in spans if s["text"] in {"a", "b"}]
        assert sorted(s["text"] for s in actual_letters) == letters
        assert all(abs(s["size"]-9) < .01 for s in actual_letters)
        assert all("Type3" not in str(font) for font in page.get_fonts(full=True))
        assert page.get_images(full=True) == []
        svg = ET.parse(stem.with_suffix(".svg"))
        texts = svg.findall(".//{http://www.w3.org/2000/svg}text")
        assert len(texts) > 20
        im = Image.open(stem.with_suffix(".png"))
        assert min(im.info.get("dpi", (0,0))) >= 599
        layout = json.loads((ROOT/"qa"/f"{stem.name}_layout.json").read_text())
        assert not layout["text_collisions"] and not layout["clipped_text"]
        checks.append(dict(export=stem.relative_to(ROOT).as_posix(), width_mm=183, height_mm=height,
                           minimum_pdf_font_pt=min(s["size"] for s in spans), live_svg_text_count=len(texts),
                           png_dpi=im.info["dpi"], panel_letters=letters, whole_page_rasterized=False))
        if letters:
            page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False).save(ROOT/"qa/pdf_render_preview.png")
    rendered = ROOT/"composite/SupplementaryFigure3.pdf"
    paper = PAPER/"figures/SupplementaryFigure3.pdf"
    assert hashlib.sha256(rendered.read_bytes()).digest() == hashlib.sha256(paper.read_bytes()).digest()
    report = {"status": "PASS" if args.visual_reviewed else "REVIEW", "scientific_contract": {
        "analysis_mode": "descriptive_enumeration", "one_row": "reporter", "n_reporters": 52,
        "primary_stability_denominator": 43, "fixed_reliability_failures": 9,
        "n_matrix_cells": 572, "n_case_points": 12, "same_state": "checkpoint_average",
        "reference": "checkpoint_average/coherent_median_ensemble", "data_authority": "SupplementaryData1/evidence.csv"},
        "automated_checks": checks, "manuscript_pdf_copy_identical": True,
        "visual_review": {"reviewer": "Codex agent", "final_size_checked": args.visual_reviewed,
                          "vector_pdf_render_checked": args.visual_reviewed, "issues": []},
        "claim_boundary": {"can_say": ["Tier assignments depend on predictor under fixed measurement/evaluation conditions"],
                           "cannot_say": ["Methods are biological replicates", "N-tier invariance establishes model robustness", "LAMP1 was prespecified"]}}
    (ROOT/"qa/figure_qa_report.json").write_text(json.dumps(report, indent=2)+"\n")
    skill_path = ROOT/"qa/skill_bundle_qa.json"
    if skill_path.exists():
        skill = json.loads(skill_path.read_text())
        skill["bundle_root"] = "."
        if args.visual_reviewed:
            assert skill["automated_status"] == "PASS"
            skill["status"] = "PASS"
            skill["visual_review"].update(final_size_checked=True, reviewer="Codex agent", issues=[])
        skill_path.write_text(json.dumps(skill, indent=2)+"\n")
    # The archived manifest contains sources only. Rendered panels, the local
    # composition and QA products can be absent from a clean source checkout.
    sources = [ROOT/name for name in ["README.md", "figure_legend.md", "figure_methods.md",
                                      "figure_plot_spec.yaml", "requirements.txt"]]
    sources += sorted((ROOT/"code").glob("*.py"))
    sources += sorted((ROOT/"source_data").glob("*.csv"))
    pd.DataFrame([dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                       sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(sources)]).to_csv(ROOT/"manifest_sha256.csv", index=False)
    artifact_dirs = ["a_predictor_tiers", "b_lamp1_gates", "composite", "qa"]
    artifacts = [p for name in artifact_dirs for p in (ROOT/name).rglob("*") if p.is_file()]
    records = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(artifacts)]
    records.append(dict(path="../../figures/SupplementaryFigure3.pdf", bytes=paper.stat().st_size,
                        sha256=hashlib.sha256(paper.read_bytes()).hexdigest()))
    (ROOT/"build").mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(ROOT/"build/artifact_manifest_sha256.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
