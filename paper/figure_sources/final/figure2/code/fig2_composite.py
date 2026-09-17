#!/usr/bin/env python3
"""Build Figure 2a-d from accepted workflow a and six frozen quantitative views.

The quantitative coordinates below retain the former 183 x 190 mm body system.
Panel a is now an independent live-text PDF bound by hash to the author-edited
PPTX. Its full-width 63.47 mm row is composed above the original quantitative
body without distortion. Quantitative labels are sized for the actual manuscript
reduction. Each quantitative row is one panel: b compares reporters, c screens
and d assays, with KO-response on the left and cell phenotype on the right.
The input identifiers b-g are unchanged. Reporter fold/direction arithmetic means
are rebuilt separately by rebuild_benchmark_sources.py before rendering.
"""

from __future__ import annotations

import sys
import shutil
import runpy
import tempfile
import re
import fitz
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import Bbox
from matplotlib.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig2_data as D  # noqa: E402
from portable_fonts import configure_sans, format_figure, layout_report, pdf_panel_letters  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "composite"
PANEL_OUT = HERE.parent / "panels"

PAGE_W, PAGE_H = 518.7402, 538.5827                 # 183 x 190 mm
A_RECT = (38.0, 436.0, 474.0, 94.0)                # x, y_bottom, w, h in pt
LETTER_X = 29.0
ROW_TOP = [112.0, 251.0, 390.0]                    # from the page top
COL_PITCH = 249.81
ROW_PITCH = 139.0
LETTER_DY = 0.0

FS_LETTER, FS_TITLE, FS_AXLAB, FS_TICK, FS_SMALL = 9.0, 7.8, 6.8, 6.5, 6.5
TEST_COLOR = {"Field": "#3E6F9C", "Gene": "#DDA06A", "Whole-screen": "#D56C75"}
# The frozen palette separates the three tests by hue only. That is safe where the
# original put them in three separate sub-columns, but d,e overlay them in one axes,
# and Field and Gene differ by 0.042 in relative luminance, so they merge in
# greyscale. Line style carries the distinction independently of colour.
TEST_DASH = {"Field": (0, ()), "Gene": (0, (4.5, 1.6)), "Whole-screen": (0, (1.1, 1.3))}
# Ink and family-band colours taken from the frozen panels rather than chosen, so the
# rebuilt page keeps the palette the rest of the figure set already uses.
INK, MUTED = "#202A33", "#6F7E88"
FAMILY_BAND = {
    "Classical tabular": "#E5EDF3",
    "Neural specialists": "#EEEAF2",
    "Generic multi-task": "#E7F0EC",
    "Biological completion": "#F5E9DE",
}

# Reporters the manuscript names in text or legends, so the reader has anchors that
# connect panels f,g to the rest of the paper.
ANCHORS = {
    "lysosome_lysotracker_live-cell_dye": "LysoTracker",
    "prb": "pRb",
    "fe2+_ferhonox_live-cell_dye": "FeRhoNox",
    "lysosome_lamp1": "LAMP1",
}

# Complete row crops, in PDF points from the lower-left corner. Standalone
# exports contain both evaluation resolutions but no panel letters or titles.
PANEL_BBOX_PT = {
    "a": (25.0, 430.0, PAGE_W - 25.0, PAGE_H - 430.0),
    "b": (0.0, 282.0, PAGE_W, 148.0),
    "c": (0.0, 145.0, PAGE_W, 143.0),
    "d": (0.0, 0.0, PAGE_W, 154.0),
}
GROUPED_SOURCE_VIEWS = {"b": ("b", "c"), "c": ("d", "e"), "d": ("f", "g")}


def fx(pt: float) -> float:
    return pt / PAGE_W


def fy_from_top(pt: float) -> float:
    return 1.0 - pt / PAGE_H


def accepted_panel_a() -> Path:
    """Use the accepted, manually edited PowerPoint export, never the old diagram."""
    builder = HERE.parent / "a_workflow/code/build_figure2a.py"
    namespace = runpy.run_path(str(builder))
    pdf = namespace["build"]()
    target = PANEL_OUT / "a"
    target.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png", ".svg"):
        shutil.copy2(pdf.with_suffix(ext), target / ("Figure2a" + ext))
    return pdf


def assemble_accepted_a(lower_pdf: Path, panel_a: Path) -> None:
    """Preserve b-g at their native size and append accepted a above the body."""
    lower = fitz.open(lower_pdf)
    a_doc = fitz.open(panel_a)
    width = PAGE_W
    body_height = 430.0
    gap = 4.0
    a_height = width * a_doc[0].rect.height / a_doc[0].rect.width
    doc = fitz.open()
    page = doc.new_page(width=width, height=a_height + gap + body_height)
    page.show_pdf_page(fitz.Rect(0, 0, width, a_height), a_doc, 0)
    page.show_pdf_page(fitz.Rect(0, a_height+gap, width, a_height+gap+body_height),
                       lower, 0, clip=fitz.Rect(0, PAGE_H-body_height, PAGE_W, PAGE_H))
    pdf_panel_letters(page, {"a": (3.0, 2.5)})
    doc.set_metadata({"title":"Figure 2", "creator":"Accepted PowerPoint panel a and original quantitative source composition"})
    doc.save(OUT/"Figure2.pdf", garbage=4, deflate=True)
    page.get_pixmap(dpi=600,alpha=False).save(OUT/"Figure2.png")
    svg = page.get_svg_image(text_as_path=False)
    # MuPDF emits unitless width/height (CSS pixels) for point-based coordinates.
    # Explicit pt units keep the SVG's physical canvas/font size equal to the PDF.
    for dimension in ("width", "height"):
        svg = re.sub(rf'(<svg\b[^>]*?\b{dimension}=")([0-9.]+)(")',
                     r'\g<1>\g<2>pt\g<3>', svg, count=1)
    (OUT/"Figure2.svg").write_text(svg, encoding="utf-8")


def style() -> None:
    configure_sans(plt, font_manager, HERE.parent)
    plt.rcParams.update({
        "font.size": FS_TICK,
        "axes.linewidth": 0.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",      # Nature requires editable vector text
        "svg.hashsalt": "figure2-cplus-readability-v2",
        # Without this matplotlib emits Type 3 fonts, which publishers reject; the
        # current figure and its five siblings all ship CID TrueType.
        "pdf.fonttype": 42,
        "text.color": INK,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 1.6,
        "ytick.major.size": 1.6,
        "xtick.major.pad": 1.4,
        "ytick.major.pad": 1.4,
    })


def slot(fig, row: int, col: int, left_pad: float, bottom_pad: float,
         width: float, height: float):
    """Axes inside the measured panel slot, positioned in points from the slot corner."""
    x0 = LETTER_X + col * COL_PITCH + left_pad
    y0 = PAGE_H - ROW_TOP[row] - ROW_PITCH + bottom_pad
    return fig.add_axes([fx(x0), y0 / PAGE_H, fx(width), height / PAGE_H])


def stamp(fig, row: int, col: int, letter: str) -> None:
    x = 3.0 + col * COL_PITCH
    y = fy_from_top(ROW_TOP[row] + LETTER_DY)
    fig.text(fx(x), y, letter, fontsize=FS_LETTER, weight="bold", va="top", ha="left")


def panel_subtitle(fig, row: int, col: int, title: str, top_offset: float = 1.0) -> None:
    """Panel interpretation is now carried by the manuscript legend."""
    return


def draw_box_panel(fig, row, col, panel, letter, xlabel) -> None:
    """b,c keep their encoding: ten methods as rows, three tests as sub-columns.

    Redrawn rather than reused, because the current panels exist only inside the
    flattened page and cannot be extracted as vectors. The four pale family bands and
    the family key are reproduced from the frozen panels, so this is visually
    equivalent to the original, though not byte-identical.
    """
    frame = D.load(panel)
    rost = D.roster()
    order = list(rost["method"])
    colors = dict(zip(rost["method"], rost["method_color"]))
    families = dict(zip(rost["method"], rost["method_family"]))
    rows_of = {}
    for k, method in enumerate(order):
        rows_of.setdefault(families[method], []).append(len(order) - 1 - k)

    sub_w = 64.0
    for i, test in enumerate(D.TESTS):
        ax = slot(fig, row, col, 26.0 + i * (sub_w + 5.0), 16.0, sub_w, 96.0)
        # Pale family bands, as in the frozen panels: the four scientific model
        # families stay visible without competing with method identity.
        for family, ys in rows_of.items():
            ax.axhspan(min(ys) - 0.5, max(ys) + 0.5, color=FAMILY_BAND[family],
                       zorder=0, lw=0)
        part = frame[frame["split_label"] == test]
        for k, method in enumerate(order):
            v = part[part["method"] == method]["score"].to_numpy()
            y = len(order) - 1 - k
            jitter = np.random.default_rng(k).uniform(-0.16, 0.16, v.size)
            ax.scatter(v, np.full_like(v, y) + jitter, s=0.65, color=colors[method],
                       alpha=0.50, lw=0, zorder=2)
            # whis=(5, 95) is not optional: the legend states 5th-95th percentile
            # whiskers, and matplotlib's default would silently draw Tukey 1.5 x IQR.
            bp = ax.boxplot(v, positions=[y], vert=False, widths=0.52, showfliers=False,
                            whis=(5, 95), patch_artist=True, zorder=3)
            for box in bp["boxes"]:
                box.set(facecolor="white", edgecolor=colors[method], linewidth=0.6, alpha=0.9)
            for group in ("whiskers", "caps", "medians"):
                for artist in bp[group]:
                    artist.set(color=colors[method], linewidth=0.75 if group == "medians" else 0.55)
        # Keep the previously approved shared negative margin after the scientific
        # aggregation correction; never clip a genuine negative value to zero.
        ax.set_xlim(-0.025, 1)
        ax.set_ylim(-0.7, len(order) - 0.3)
        ax.set_xticks([0, 0.5, 1.0])
        # The neighbouring sub-column's "1.0" sits against this one's "0.0".
        ax.set_xticklabels(["0.0" if i == 0 else "", "0.5", "1.0"], fontsize=FS_TICK)
        ax.tick_params(labelsize=FS_TICK)
        if i == 0:
            ax.set_yticks(range(len(order)))
            ax.set_yticklabels(order[::-1], fontsize=FS_TICK)
            for tick, method in zip(ax.get_yticklabels(), order[::-1]):
                tick.set_color(colors[method])
                tick.set_fontweight("bold")
        else:
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
        ax.text(0.5, 1.02, test, transform=ax.transAxes, color=TEST_COLOR[test],
                fontsize=FS_SMALL, weight="bold", ha="center")
        if i == 1:
            ax.set_xlabel(xlabel, fontsize=FS_AXLAB)
    panel_subtitle(
        fig, row, col,
        "KO-response · reporter-level distributions" if panel == "b"
        else "Cell phenotype · reporter-level distributions",
    )


def draw_ranked_panel(fig, row, col, panel, letter, summary: bool) -> None:
    """d,e re-encoded: screens ranked by consensus.

    The curve spans the observed distribution across screens, combining reporter
    composition and acquisition context. Bands show between-method spread at each
    screen; this is descriptive, not a decomposition of environmental and model
    effects. Ranks are percentiles because the tests hold 73, 73 and 66 screens.
    """
    con = D.consensus(panel)
    ax = slot(fig, row, col, 30.0, 28.0, 136.0 if summary else 196.0, 81.0)
    for test in D.TESTS:
        s = con[con["split_label"] == test].sort_values("consensus").reset_index(drop=True)
        x = np.linspace(0, 100, len(s))
        c = TEST_COLOR[test]
        ax.fill_between(x, s["lo"], s["hi"], color=c, alpha=0.10, lw=0, zorder=2)
        ax.fill_between(x, s["q25"], s["q75"], color=c, alpha=0.32, lw=0, zorder=3)
        ax.plot(x, s["consensus"], color=c, lw=1.05, zorder=4,
                linestyle=TEST_DASH[test], label=test)
    # A dedicated strip keeps every legend word and sample off all curves/bands.
    # The order matches the Field/Gene/Whole-screen columns in b,c,f,g.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.035), ncol=3,
              borderaxespad=0, borderpad=0, columnspacing=0.7,
              handletextpad=0.3, handlelength=1.25,
              prop={"size": FS_SMALL, "weight": "bold"},
              labelcolor="linecolor")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.15, 1)
    ax.axhline(0, color=MUTED, lw=0.4, zorder=1)
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100%"], fontsize=FS_TICK)
    ax.set_yticks([0, 0.5, 1.0])
    ax.tick_params(labelsize=FS_TICK)
    ax.set_xlabel("screens ranked by consensus", fontsize=FS_AXLAB)
    ax.set_ylabel("Pearson r", fontsize=FS_AXLAB, labelpad=1.5)

    if summary:
        # Both bars are interquartile ranges. An earlier draft compared a 5th-95th
        # interval across screens against an interquartile range across methods, which
        # is a 90 per cent interval against a 50 per cent one and roughly doubled the
        # apparent ratio. Matched intervals are the only honest comparison.
        #
        # The bars are labelled for what they measure and nothing more. 56 of the 73
        # screens hold exactly one reporter assay, so spread across screens cannot be
        # separated from spread across targets, and calling it an environment effect
        # would claim a decomposition this design does not support.
        ax2 = slot(fig, row, col, 182.0, 28.0, 32.0, 81.0)
        for i, test in enumerate(D.TESTS):
            s = con[con["split_label"] == test]
            across = s["consensus"].quantile(0.75) - s["consensus"].quantile(0.25)
            within = float((s["q75"] - s["q25"]).median())
            c = TEST_COLOR[test]
            ax2.barh(i + 0.18, across, height=0.32, color=c, alpha=0.85, lw=0)
            ax2.barh(i - 0.18, within, height=0.32, facecolor=c, alpha=0.28,
                     edgecolor=c, linewidth=0.4)
        ax2.set_yticks([])
        # The unchanged KO Whole-screen IQR is 0.255567; the historical 0.25
        # limit clipped its bar. Both resolution views retain one common scale.
        ax2.set_xlim(0, 0.28)
        assert all(patch.get_width() <= 0.28 for patch in ax2.patches), "IQR bar clipped"
        ax2.set_xticks([0, 0.2])
        ax2.set_xticklabels(["0", "0.2"], fontsize=FS_TICK)
        ax2.tick_params(labelsize=FS_TICK)
        ax2.set_xlabel("IQR in r", fontsize=FS_AXLAB)
        ax2.add_patch(Rectangle((0.02, 1.085), 0.09, 0.035,
                                transform=ax2.transAxes, clip_on=False,
                                facecolor=MUTED, edgecolor="none", alpha=0.85))
        ax2.add_patch(Rectangle((0.02, 0.985), 0.09, 0.035,
                                transform=ax2.transAxes, clip_on=False,
                                facecolor=MUTED, edgecolor=MUTED, linewidth=0.4,
                                alpha=0.28))
        ax2.text(0.15, 1.102, "across screens", transform=ax2.transAxes,
                 fontsize=FS_TICK, color=MUTED, va="center", clip_on=False)
        ax2.text(0.15, 1.002, "across methods", transform=ax2.transAxes,
                 fontsize=FS_TICK, color=MUTED, va="center", clip_on=False)
    panel_subtitle(
        fig, row, col,
        "KO-response · screen-level consensus" if panel == "d"
        else "Cell phenotype · screen-level consensus",
        top_offset=6.0,
    )


def draw_rows_panel(fig, row, col, panel, letter) -> None:
    """f,g re-encoded: one row per reporter, one dot per assay.

    Rows distinguish reporters and connectors span their observed assay scores;
    these summaries do not identify causal target or environmental effects. Panel g
    field/gene views repeat reporter-level cell scores by assay membership and cannot
    resolve within-reporter cell-level screen variation. The common row order retains
    reporter identity across both panels and all three tests.
    """
    ax_df = D.assay_axes(panel)
    src = D.assay_axes("f")
    order = (src[src["split_label"] == "Gene"]
             .groupby("reporter_slug")["consensus"].mean().sort_values())
    ry = {r: k for k, r in enumerate(order.index)}
    sub_w = 56.0
    for i, test in enumerate(D.TESTS):
        ax = slot(fig, row, col, 40.0 + i * (sub_w + 5.0), 16.0, sub_w, 96.0)
        part = ax_df[ax_df["split_label"] == test]
        for slug in ANCHORS:
            ax.axhline(ry[slug], color="#D4DCE2", lw=0.45,
                       linestyle=(0, (1.4, 2.0)), zorder=1)
        for reporter, grp in part.groupby("reporter_slug"):
            if len(grp) > 1:
                ax.plot([grp["consensus"].min(), grp["consensus"].max()],
                        [ry[reporter]] * 2, color="#B7C1C8", lw=0.55, zorder=2)
        ax.scatter(part["consensus"], [ry[r] for r in part["reporter_slug"]],
                   color=TEST_COLOR[test], s=2.1, lw=0, zorder=3)
        anchor_part = part[part["reporter_slug"].isin(ANCHORS)]
        ax.scatter(
            anchor_part["consensus"],
            [ry[r] for r in anchor_part["reporter_slug"]],
            facecolor=TEST_COLOR[test], edgecolor=INK,
            s=11.0, linewidth=0.6, zorder=4,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.5, len(ry) + 0.5)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0.0" if i == 0 else "", "0.5", "1.0"], fontsize=FS_TICK)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=FS_TICK)
        ax.text(0.5, 1.10, test, transform=ax.transAxes, color=TEST_COLOR[test],
                fontsize=FS_SMALL, weight="bold", ha="center")
        # Panel g's field and gene columns resolve 99 assays to 52 reporter-level
        # values; the annotation states what the column actually resolves.
        ax.text(0.5, 1.02, D.assay_unit_note(panel, test), transform=ax.transAxes,
                color=MUTED, fontsize=FS_TICK, ha="center")
        if i == 0:
            for slug, label in ANCHORS.items():
                ax.text(-0.04, ry[slug], label, fontsize=FS_AXLAB, color=INK,
                        weight="bold", va="center", ha="right",
                        transform=ax.get_yaxis_transform())
                ax.plot([-0.025, 0.025], [ry[slug], ry[slug]],
                        transform=ax.get_yaxis_transform(), color=INK,
                        lw=0.5, clip_on=False, zorder=5)
        if i == 1:
            ax.set_xlabel("consensus Pearson r", fontsize=FS_AXLAB)
    panel_subtitle(
        fig, row, col,
        "KO-response · reporter × screen assays" if panel == "f"
        else "Cell phenotype · reporter × screen assays",
    )


def family_key(fig) -> None:
    """One key for the four model families, under the b,c titles.

    The frozen figure repeated this key inside all six lower panels. Only b and c
    still resolve individual methods, so the key is drawn once for that row instead
    of six times.
    """
    x = 47.0
    y = fy_from_top(127.5)
    for family, colour in FAMILY_BAND.items():
        fig.patches.append(plt.Rectangle(
            (fx(x), y - 0.0042), fx(6.0), 0.0060, transform=fig.transFigure,
            facecolor=colour, edgecolor=MUTED, linewidth=0.3, clip_on=False))
        fig.text(fx(x + 8.5), y, family, fontsize=FS_TICK, color=MUTED,
                 va="center", ha="left")
        x += 16.0 + 4.05 * len(family)


def compact_family_key(fig, col: int) -> None:
    """Complete compact family key for standalone b/c panels."""
    labels = {
        "Classical tabular": "Classical",
        "Neural specialists": "Neural",
        "Generic multi-task": "Multi-task",
        "Biological completion": "Biological",
    }
    x = LETTER_X + col * COL_PITCH + 18.0
    y = fy_from_top(127.5)
    for family, colour in FAMILY_BAND.items():
        short = labels[family]
        fig.patches.append(plt.Rectangle(
            (fx(x), y - 0.0040), fx(5.0), 0.0058, transform=fig.transFigure,
            facecolor=colour, edgecolor=MUTED, linewidth=0.3, clip_on=False))
        fig.text(fx(x + 7.0), y, short, fontsize=FS_TICK, color=MUTED,
                 va="center", ha="left")
        x += 13.0 + 3.1 * len(short)


def draw_one_panel(fig, letter: str) -> None:
    """One unlabelled grouped panel; legacy view IDs map only to frozen inputs."""
    if letter == "b":
        draw_box_panel(fig, 0, 0, "b", "b", "KO Pearson r")
        draw_box_panel(fig, 0, 1, "c", "c", "Cell Pearson r")
        family_key(fig)
    elif letter == "c":
        draw_ranked_panel(fig, 1, 0, "d", "d", summary=True)
        draw_ranked_panel(fig, 1, 1, "e", "e", summary=True)
    elif letter == "d":
        draw_rows_panel(fig, 2, 0, "f", "f")
        draw_rows_panel(fig, 2, 1, "g", "g")
    else:
        raise ValueError(f"unknown panel: {letter}")


def verify_text_layout(fig, name: str, crop: Bbox | None = None) -> None:
    """Fail on text collisions/clipping before export; visual QA is still required.

    Use renderer-measured artist boxes rather than guessed character widths. Tick
    labels repeated at different locations remain independent artists. Deliberate
    data overplotting and the light family bands are not text collisions.
    """
    format_figure(fig, 2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [artist for artist in fig.findobj(Text)
             if artist.get_visible() and artist.get_text().strip()]
    boxes = [(artist, artist.get_window_extent(renderer)) for artist in texts]
    # 0.15 pt accommodates renderer rounding, not genuinely colliding labels.
    tolerance = fig.dpi / 72 * 0.15
    failures = []
    for i, (first, a) in enumerate(boxes):
        for second, b in boxes[i + 1:]:
            if min(a.x1, b.x1) - max(a.x0, b.x0) > tolerance and \
                    min(a.y1, b.y1) - max(a.y0, b.y0) > tolerance:
                failures.append(f"overlap: {first.get_text()!r} / {second.get_text()!r}")
    boundary = crop.transformed(fig.dpi_scale_trans) if crop is not None else fig.bbox
    for artist, box in boxes:
        if box.x0 < boundary.x0 - tolerance or box.x1 > boundary.x1 + tolerance or \
                box.y0 < boundary.y0 - tolerance or box.y1 > boundary.y1 + tolerance:
            failures.append(f"clipped: {artist.get_text()!r}")
    if failures:
        raise RuntimeError(f"{name} text-layout check failed: " + "; ".join(failures))
    print(f"PASS {name}: {len(boxes)} visible text artists, no text overlaps or clipping")


def export_standalone_panels() -> None:
    PANEL_OUT.mkdir(parents=True, exist_ok=True)
    for letter, (x, y, width, height) in PANEL_BBOX_PT.items():
        if letter == "a":
            continue  # copied directly from the authoritative PPT/PDF pair
        panel_fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
        draw_one_panel(panel_fig, letter)
        panel_dir = PANEL_OUT / letter
        panel_dir.mkdir(parents=True, exist_ok=True)
        crop = Bbox.from_bounds(x / 72.0, y / 72.0, width / 72.0, height / 72.0)
        verify_text_layout(panel_fig, f"panel {letter}", crop)
        panel_stem = panel_dir / f"Figure2{letter}"
        panel_fig.savefig(
            panel_stem.with_suffix(".pdf"), bbox_inches=crop, pad_inches=0,
            metadata={"Creator": "Figure 2 reproducibility package", "CreationDate": None},
        )
        panel_fig.savefig(
            panel_stem.with_suffix(".svg"), bbox_inches=crop, pad_inches=0,
            metadata={"Date": None},
        )
        panel_fig.savefig(panel_stem.with_suffix(".png"), bbox_inches=crop,
                          pad_inches=0, dpi=600)
        plt.close(panel_fig)


def main() -> int:
    style()
    print("Verifying frozen source data against the reported medians:")
    for line in D.verify():
        print(line)

    panel_a = accepted_panel_a()
    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
    fig._composite_panel_letters = True

    for row, letter in enumerate(GROUPED_SOURCE_VIEWS):
        draw_one_panel(fig, letter)
        stamp(fig, row, 0, letter)
    verify_text_layout(fig, "quantitative body")

    OUT.mkdir(parents=True, exist_ok=True)
    # Render the frozen quantitative data with print-size labels and clear keys.
    with tempfile.TemporaryDirectory(prefix="figure2_quantitative_body_") as work:
        lower_pdf = Path(work) / "body.pdf"
        fig.savefig(lower_pdf, metadata={"Creator":"Figure 2 quantitative body","CreationDate":None})
        assemble_accepted_a(lower_pdf, panel_a)
    stem = "Figure2"
    plt.close(fig)
    export_standalone_panels()
    (HERE.parent / "published").mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / "Figure2.pdf", HERE.parent / "published/Figure2.pdf")
    print(f"\nwrote {OUT / f'{stem}.pdf'} (+ .svg, .png at 600 dpi)")
    print(f"wrote unlabelled standalone panels a-d under {PANEL_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
