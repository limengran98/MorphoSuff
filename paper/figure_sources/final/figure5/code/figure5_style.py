"""Shared presentation-only style for Figure 5.

The module fixes typography, colour semantics and export settings.  It does not
aggregate or transform any frozen scientific source table.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
from matplotlib import font_manager


FIGURE5_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = FIGURE5_ROOT / "source_data"
PAPER_ROOT = FIGURE5_ROOT
if str(FIGURE5_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE5_ROOT))
from _portable_fonts import configure_sans, format_figure, layout_report  # noqa: E402


COLORS = {
    "ink": "#202A33",
    "muted": "#6F7E88",
    "mid": "#91A0A9",
    "grid": "#DCE3E7",
    "pale": "#F4F6F7",
    "stable": "#3E6F9C",
    "stable_light": "#D7E5EF",
    "constraint": "#D56C75",
    "constraint_light": "#F1D9DD",
    "orange": "#DDA06A",
    "orange_light": "#F3E5D5",
    "blue": "#3E6F9C",
    "violet": "#7B6597",
    "white": "#FFFFFF",
}

BIOLOGY_COLORS = {
    "Endosome": "#C6A16A",
    "Lysosome": "#E8943A",
    "ER/Golgi": "#4F9A70",
    "Cytoskeleton": "#4FA7B8",
    "Peroxisome": "#D56C75",
    "Stress/PQC": "#CC79A7",
    "Mitochondria": "#C85A3C",
    "Plasma Membrane": "#9C8C72",
    "Nucleus": "#3E6F9C",
    "Signaling": "#7B6597",
}


def apply_style() -> None:
    configure_sans(mpl, font_manager, PAPER_ROOT)
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
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(width=0.6, length=2.3, pad=1.4)


def save_figure(fig, stem: Path, *, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    format_figure(fig, 5)
    layout_report(fig, FIGURE5_ROOT / 'qa' / (stem.name + '_typography.json'))
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), dpi=dpi, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, facecolor="white")
