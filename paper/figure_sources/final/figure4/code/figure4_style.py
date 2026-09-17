"""Shared Nature-ready style helpers for Figure 4.

This module contains presentation-only settings.  It must not aggregate or alter
the frozen Figure 4 source tables.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
from matplotlib import font_manager


FIGURE4_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = FIGURE4_ROOT / "source_data"
if str(FIGURE4_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE4_ROOT))
from _portable_fonts import configure_sans, format_figure, layout_report  # noqa: E402


COLORS = {
    "ink": "#202A33",
    "muted": "#6F7E88",
    "grid": "#DCE3E7",
    "exact": "#3E6F9C",
    "shuffle": "#D56C75",
    "covariate": "#E8943A",
    "size": "#7E8D96",
    "minus_size": "#79A8C9",
    "residual": "#A783AD",
    "residual_dark": "#66527F",
    "observed": "#202A33",
}


def apply_style() -> None:
    """Apply the frozen Figure 4 publication style."""

    configure_sans(mpl, font_manager, FIGURE4_ROOT)
    mpl.rcParams.update(
        {
            "font.size": 6.2,
            "axes.titlesize": 7.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 6.2,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "axes.linewidth": 0.65,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def clean_axis(ax, *, left: bool = True, bottom: bool = True) -> None:
    """Use the manuscript's sparse-axis convention."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(width=0.6, length=2.5, pad=1.5)


def add_panel_letter(fig, rect, letter: str) -> None:
    """Place a lowercase panel letter on the shared outer grid."""

    x0, _y0, _w, _h = rect
    fig.text(x0 - 0.014, _y0 + _h + 0.002, letter, fontsize=8.0,
             fontweight="bold", ha="left", va="bottom", color=COLORS["ink"])


def save_figure(fig, stem: Path, *, dpi: int = 600) -> None:
    """Save editable vector outputs and a final-size raster preview."""

    stem.parent.mkdir(parents=True, exist_ok=True)
    format_figure(fig, 4)
    layout_report(fig, FIGURE4_ROOT / 'qa' / (stem.name + '_typography.json'))
    for ext in ("pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
