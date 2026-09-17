#!/usr/bin/env python3
"""Assemble the final Figure 4 from native Matplotlib artists.

The four panels are imported from their standalone builders so that the PDF
and SVG retain editable text and vector axes.  No panel is pasted as a raster
image and no scientific quantity is recomputed here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from build_panel_ab import draw_panel_a, draw_panel_b, load_and_validate
from build_panel_c import plot_panel_c
from build_panel_d import draw_panel_d
from figure4_style import COLORS, FIGURE4_ROOT, apply_style, save_figure


OUT = FIGURE4_ROOT / "composite"
LETTER_POSITIONS = {
    "a": (0.008, 0.988),
    "b": (0.662, 0.988),
    "c": (0.008, 0.723),
    "d": (0.008, 0.315),
}


def _letter(fig: plt.Figure, x: float, y: float, label: str) -> None:
    fig.text(
        x,
        y,
        label,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
    )


def build() -> dict[str, object]:
    apply_style()
    means, advantage, residual = load_and_validate()

    # 183 mm is the Nature-family two-column width.  The asymmetric vertical
    # allocation preserves publication-scale labels in the dense c/d panels.
    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 225 * mm), facecolor="white")
    fig._composite_panel_letters = True

    # Top evidence row: fixed-model intervention and residual falsification.
    draw_panel_a(
        fig,
        (0.018, 0.745, 0.642, 0.235),
        means,
        advantage,
        add_letter=False,
    )
    draw_panel_b(
        fig,
        (0.675, 0.745, 0.315, 0.225),
        residual,
        add_letter=False,
    )

    # Shift the two lower rows left by 0.025 of the canvas (4.575 mm),
    # preserving their sizes and the space needed by long scientific labels.
    plot_panel_c(
        fig,
        (0.018, 0.365, 0.945, 0.350),
        add_letter=False,
    )
    draw_panel_d(
        fig,
        (0.105, 0.048, 0.845, 0.277),
        letter=None,
    )

    for label, (x, y) in LETTER_POSITIONS.items():
        _letter(fig, x, y, label)

    stem = OUT / "Figure4"
    save_figure(fig, stem, dpi=600)
    width_in, height_in = fig.get_size_inches()
    plt.close(fig)
    return {
        "status": "built",
        "width_mm": width_in * 25.4,
        "height_mm": height_in * 25.4,
        "panels": ["a", "b", "c", "d"],
        "assembly": "native_matplotlib_artists",
        "primary_model_panel_a": "reporter-specific MLP",
        "panel_a_reporters": int(means.reporter_slug.nunique()),
        "panel_b_reporters": int(residual.reporter_slug.nunique()),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = build()
    (FIGURE4_ROOT / "qa" / "composite_build_report.json").parent.mkdir(
        parents=True, exist_ok=True
    )
    (FIGURE4_ROOT / "qa" / "composite_build_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
