"""Shared, presentation-only typography for the six main manuscript figures.

The canonical composites are 183 mm wide and are placed at one common TeX width.
Panel letters are 9 pt bold; ordinary text is at least 6 pt. Explanatory panel
headings belong in the caption. Condition, channel, axis and key labels remain.
This module never reads or modifies scientific data.
"""
from __future__ import annotations

import json
from pathlib import Path

from matplotlib.text import Text

FONT = "DejaVu Sans"
LETTER_PT = 9.0
MIN_TEXT_PT = 6.0
AXIS_PT = 6.5
PLACED_WIDTH_MM = 168.0

# Exact strings make removal reviewable; this is not a heuristic that can
# accidentally delete an experimental condition or a scientific annotation.
HEADINGS = {
    1: {
        "73 screens · 7.344M phase cells · 9.996M reporter observations",
        "Reporter–screen assays ordered by coverage · n = 99",
        "Reporter-specific target dimensions · n = 52",
        "56 single-reporter screens · compressed",
        "17 multi-reporter screens · expanded",
        "Mean across 52 reporters",
    },
    2: {
        "KO-response · reporter-level distributions",
        "Cell phenotype · reporter-level distributions",
        "KO-response · screen-level consensus",
        "Cell phenotype · screen-level consensus",
        "KO-response · reporter × screen assays",
        "Cell phenotype · reporter × screen assays",
    },
    3: {
        "Recovery varies by evaluation scope and analysis unit",
        "Highest means: Endosome/Lysosome and level/range endpoints",
        "Marker value associates with recovery; phase-overlap CI crosses 0",
        "Biological systems", "Endpoint families", "Association with recovery",
        "Spearman ρ (reporter-bootstrap 95% CI; n=52)", "Marker value",
    },
    4: {
        "Absolute recoverability", "Information-intervention effect",
        "quantitative recovery", "pattern retained", "amplitude compressed",
        "Response reconstruction", "Strong-response ranking",
        "81 reporter→screen directions per method",
    },
    5: {
        "Association with recoverability", "Constraint overlap", "Joint association",
        "n=3, 3, 100 (top→bottom)",
        "10/11: ρ > 0.5", "11/52 below 0.5",
    },
    6: {
        "Response ordering", "Biological programmes", "Utility associations",
        "Utility retained after response replacement", "Margin to the frozen decision gate",
        "Threshold sensitivity",
    },
}


def visible_texts(fig):
    suppressed = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            all_ticks = [*axis.get_major_ticks(), *axis.get_minor_ticks()]
            drawn_ticks = axis._update_ticks() if ax.axison and axis.get_visible() else []
            drawn = {id(t) for tick in drawn_ticks for t in (tick.label1, tick.label2)}
            suppressed.update(id(t) for tick in all_ticks for t in (tick.label1, tick.label2)
                              if id(t) not in drawn)
    return [t for t in fig.findobj(Text)
            if t.get_visible() and t.get_text().strip()
            and id(t) not in suppressed
            and (t.axes is None or t.axes.get_visible())]


def format_figure(fig, number: int) -> None:
    """Apply the manuscript type hierarchy before layout checks and export."""
    removed = set(getattr(fig, "_removed_panel_headings", ()))
    # Axis labels are retained even when they name the same quantity as an
    # explanatory heading elsewhere (e.g. the Figure 6a colourbar).
    axis_labels = {id(t) for ax in fig.axes for t in (ax.xaxis.label, ax.yaxis.label)}
    composite = bool(getattr(fig, '_composite_panel_letters', False)
                     or getattr(fig.get_figure(), '_composite_panel_letters', False))
    for t in visible_texts(fig):
        value = " ".join(t.get_text().split())
        if value in HEADINGS[number] and id(t) not in axis_labels:
            t.set_visible(False)
            removed.add(value)
            continue
        t.set_fontfamily(FONT)
        if value in tuple("abcdefg") and len(value) == 1:
            if not composite:
                t.set_visible(False)
                continue
            t.set_fontsize(LETTER_PT)
            t.set_fontweight("bold")
            t.set_fontstyle("normal")
            t.set_color("#202A33")
        else:
            t.set_fontsize(max(AXIS_PT if id(t) in axis_labels else MIN_TEXT_PT,
                               t.get_fontsize()))
    fig._removed_panel_headings = sorted(removed)


def pdf_panel_letter_collisions(page, letters):
    """Check letters against live text after all PDF composition transforms."""
    import fitz
    spans = [s for b in page.get_text('dict')['blocks']
             for line in b.get('lines', []) for s in line['spans']
             if s['text'].strip()]
    failures = []
    for label in spans:
        if label['text'] not in letters:
            continue
        for other in spans:
            if other is label:
                continue
            overlap = fitz.Rect(label['bbox']) & fitz.Rect(other['bbox'])
            if overlap.width > 0.15 and overlap.height > 0.15:
                failures.append({'letter': label['text'], 'text': other['text'],
                                 'overlap_pt': [overlap.width, overlap.height]})
    return failures


def pdf_panel_letters(page, positions_pt):
    """Stamp labels only on an assembled PDF; clean source panels stay unlabelled.

    Positions are top-left coordinates in PDF points, independent of panel size.
    """
    import fitz
    from matplotlib import font_manager
    font_path = font_manager.findfont(font_manager.FontProperties(family=FONT, weight='bold'))
    font = fitz.Font(fontfile=font_path)
    page.insert_font(fontname='CompositePanelBold', fontfile=font_path)
    for letter, (x, y) in positions_pt.items():
        page.insert_text((x, y + LETTER_PT * font.ascender), letter,
                         fontsize=LETTER_PT, fontname='CompositePanelBold',
                         color=(32 / 255, 42 / 255, 51 / 255), overlay=True)
    collisions = pdf_panel_letter_collisions(page, positions_pt)
    if collisions:
        raise ValueError(f'Composite panel-letter/text collision: {collisions}')


def svg_panel_letters(root, positions_mm):
    """Live SVG labels for the Figure 3 compositor's millimetre viewBox."""
    import xml.etree.ElementTree as ET
    size_mm = LETTER_PT * 25.4 / 72
    for letter, (x, y) in positions_mm.items():
        t = ET.SubElement(root, '{http://www.w3.org/2000/svg}text', {
            'x': str(x), 'y': str(y + size_mm * 0.92822265625),
            'font-family': FONT, 'font-size': str(size_mm),
            'font-weight': '700', 'fill': '#202A33'})
        t.text = letter


def layout_report(fig, path: Path, *, boundary=None) -> dict:
    """Record actual text boxes; visual inspection must resolve every warning."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = visible_texts(fig)
    canvas = fig.bbox if boundary is None else boundary
    px_to_pt = 72 / fig.dpi
    boxes = [(t, Text.get_window_extent(t, renderer=renderer)) for t in texts]
    collisions, clipped = [], []
    tolerance = 0.15 / px_to_pt
    for i, (a, ba) in enumerate(boxes):
        if (ba.x0 < canvas.x0 - tolerance or ba.x1 > canvas.x1 + tolerance
                or ba.y0 < canvas.y0 - tolerance or ba.y1 > canvas.y1 + tolerance):
            clipped.append(a.get_text())
        for b, bb in boxes[i + 1:]:
            dx = min(ba.x1, bb.x1) - max(ba.x0, bb.x0)
            dy = min(ba.y1, bb.y1) - max(ba.y0, bb.y0)
            if dx > tolerance and dy > tolerance:
                collisions.append({"a": a.get_text(), "b": b.get_text(),
                                   "overlap_pt": [round(dx * px_to_pt, 3),
                                                  round(dy * px_to_pt, 3)]})
    report = {
        "status": "REVIEW" if collisions or clipped else "AUTOMATED_PASS",
        "removed_panel_headings": getattr(fig, "_removed_panel_headings", []),
        "text_count": len(boxes), "font_family": FONT,
        "minimum_authored_pt": min(t.get_fontsize() for t in texts),
        "panel_letter_pt": LETTER_PT,
        "text_collisions": collisions, "clipped_text": clipped,
        "visual_review_required": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report
