#!/usr/bin/env python3
"""Assemble Figure 1 from the five fixed-size panel exports.

The PDF assembly preserves each panel as vector content.  The PNG is a
publication-size raster preview of the final labelled PDF. Source panels remain
unlabelled; no panel content is redrawn or rescaled non-uniformly.
"""

from __future__ import annotations

from pathlib import Path
import io
import sys

import pymupdf as fitz
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject


MM_TO_PT = 72.0 / 25.4
CANVAS_MM = (183.0, 239.46820809248555)
# The manuscript PDF is assembled from panel PDFs and therefore preserves live
# vector text.  Keep the companion PNG at the same 600-dpi proof resolution as
# the panel exports so native 384-pixel microscopy crops are never additionally
# degraded by a low-resolution composite preview.
PREVIEW_DPI = 600

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from _portable_fonts import pdf_panel_letters
OUT = ROOT / "composite" / "Figure1"

# x, y and panel dimensions use a bottom-left origin in millimetres.
PANELS = [
    # Full-width accepted workflow and three-chart linkage row preserve type.
    ("a", ROOT / "a_workflow/figure/Figure1-a_workflow", 0.0, 176.0, 183.0, 63.46820809248555),
    ("b", ROOT / "b_exact_linkage_scale/figure/Figure1-b_exact_linkage_scale", 0.0, 143.0, 183.0, 30.0),
    # Native quantitative canvases preserve >=6-pt type and 3-mm row gutters.
    ("c", ROOT / "c_sparse_assay_topology/figure/Figure1-c_assay_topology", 0.0, 55.0, 183.0, 85.0),
    ("d", ROOT / "d_ko_response_atlas/figure/Figure1-d_ko_response_atlas", 0.8, 0.0, 58.0, 52.0),
    ("e", ROOT / "e_same_cell_response_maps/figure/Figure1-e_same_cell_response_maps", 59.4, 0.0, 123.0, 52.0),
]


def assemble_pdf() -> None:
    width_pt, height_pt = (value * MM_TO_PT for value in CANVAS_MM)
    canvas = PageObject.create_blank_page(width=width_pt, height=height_pt)
    for _, stem, x_mm, y_mm, width_mm, height_mm in PANELS:
        page = PdfReader(stem.with_suffix(".pdf")).pages[0]
        source_width = float(page.mediabox.width)
        source_height = float(page.mediabox.height)
        target_width = width_mm * MM_TO_PT
        target_height = height_mm * MM_TO_PT
        assert abs(target_width / source_width - target_height / source_height) < 1e-5, (
            f"Non-uniform panel scaling is not permitted: {stem.name}"
        )
        transform = (
            Transformation()
            .scale(target_width / source_width, target_height / source_height)
            .translate(x_mm * MM_TO_PT, y_mm * MM_TO_PT)
        )
        canvas.merge_transformed_page(page, transform, over=True)

    writer = PdfWriter()
    writer.add_page(canvas)
    writer.add_metadata({"/Title": "Figure 1"})
    buffer = io.BytesIO()
    writer.write(buffer)
    with fitz.open(stream=buffer.getvalue(), filetype="pdf") as document:
        pdf_panel_letters(document[0], {
            letter: ((x + 1) * MM_TO_PT, (CANVAS_MM[1] - y - h + 1) * MM_TO_PT)
            for letter, _, x, y, _, h in PANELS
        })
        document.save(OUT.with_suffix(".pdf"), garbage=4, deflate=True)


def assemble_png() -> None:
    with fitz.open(OUT.with_suffix(".pdf")) as document:
        document[0].get_pixmap(dpi=PREVIEW_DPI, alpha=False).save(OUT.with_suffix(".png"))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    assemble_pdf()
    assemble_png()
    with fitz.open(OUT.with_suffix(".pdf")) as document:
        OUT.with_suffix(".svg").write_text(
            document[0].get_svg_image(text_as_path=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
