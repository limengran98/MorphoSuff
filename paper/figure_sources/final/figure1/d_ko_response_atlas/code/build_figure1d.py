#!/usr/bin/env python3
"""Build Figure 1d: the reporter-specific KO-response atlas.

The frozen KO ordering and within-reporter response percentiles are unchanged.
The blue-white-red palette matches Figure 1e, with 0.5 as the explicit midpoint
of the percentile scale; it does not re-interpret the values as signed z-scores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


WIDTH_MM = 58
HEIGHT_MM = 52
MM_TO_IN = 1 / 25.4
FIGURE_ROOT = Path(__file__).resolve().parents[2]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))
from _portable_fonts import configure_sans  # noqa: E402
sys.path.insert(0, str(FIGURE_ROOT / "code"))
from layout_qa import assert_text_layout  # noqa: E402

RED = "#D56C75"
BIOLOGY_COLORS = {
    "Endosome": "#C6A16A", "Lysosome": "#E8943A", "ER/Golgi": "#4F9A70",
    "Cytoskeleton": "#4FA7B8", "Peroxisome": "#D56C75", "Stress/PQC": "#CC79A7",
    "Mitochondria": "#C85A3C", "Plasma Membrane": "#9C8C72", "Nucleus": "#3E6F9C",
    "Signaling": "#7B6597",
}
KO_CMAP = LinearSegmentedColormap.from_list(
    "ops_ko_response", ["#3E6F9C", "#79A8C9", "#F7F7F4", "#E5A0A8", "#A44758"], N=256
)


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def frozen_input_root() -> Path:
    """Return the compact, released Figure 1d input snapshot."""
    return bundle_root() / "source_data" / "frozen_inputs" / "F_ko_response_landscape"


def configure() -> None:
    configure_sans(mpl, font_manager, FIGURE_ROOT)
    mpl.rcParams.update({
        "font.size": 6.0, "axes.labelsize": 6.0,
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def clean(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def build() -> None:
    configure()
    root = frozen_input_root()
    if not root.is_dir():
        raise FileNotFoundError(f"Released Figure 1d inputs are missing: {root}")
    response_root = root / "response"
    order = pd.read_csv(root / "reporter_recoverability_order.csv")
    annotations = (
        pd.read_csv(response_root / "reporter_annotations.csv")
        .set_index("reporter_slug").loc[order.reporter_slug].reset_index()
    )
    frame = pd.read_csv(root / "ordered_response_percentile.csv.gz", index_col=0)
    frame = frame.loc[order.reporter_slug]
    genes = frame.columns.astype(str).tolist()
    values = frame.to_numpy(float)
    mean = (
        pd.read_csv(root / "mean_response_percentile.csv").set_index("gene")
        .reindex(genes).mean_response_percentile.to_numpy(float)
    )
    assert values.shape == (52, 1000)
    assert np.nanmin(values) >= 0 and np.nanmax(values) <= 1

    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    fig.text(0.006, 0.992, "d", ha="left", va="top", fontsize=8.0, fontweight="bold")
    grid = fig.add_gridspec(
        4, 3, left=0.052, right=0.985, bottom=0.145, top=0.925,
        height_ratios=[0.22, 1.0, 0.055, 0.10],
        width_ratios=[0.026, 1.0, 0.370],
        hspace=0.045, wspace=0.035,
    )
    trace = fig.add_subplot(grid[0, 1])
    strip = fig.add_subplot(grid[1, 0])
    heat = fig.add_subplot(grid[1, 1])
    labels = fig.add_subplot(grid[1, 2])
    ko_label = fig.add_subplot(grid[2, 1])
    cax = fig.add_subplot(grid[3, 1])

    trace.fill_between(np.arange(len(mean)), 0, mean,
                       color=mpl.colors.to_rgba(RED, 0.16), lw=0)
    trace.plot(mean, color=RED, lw=0.85)
    trace.set_xlim(0, len(mean) - 1); trace.set_ylim(0.30, 1.0); clean(trace)
    trace.text(
        0.012, 1.035, "Mean across 52 reporters",
        transform=trace.transAxes, ha="left", va="bottom", clip_on=False,
        fontsize=6.0, color="#526573", fontweight="bold",
    )

    row_colours = np.asarray([
        mpl.colors.to_rgba(BIOLOGY_COLORS[value])
        for value in annotations.biological_category
    ])
    strip.imshow(row_colours[:, None, :], aspect="auto", interpolation="nearest")
    clean(strip)

    image = heat.imshow(
        values, aspect="auto", interpolation="nearest", cmap=KO_CMAP,
        vmin=0, vmax=1, rasterized=True,
    )
    clean(heat)
    fig.text(0.026, 0.515, "52 reporters", ha="center", va="center",
             rotation=90, fontsize=6.0, color="#526573")
    clean(ko_label)
    ko_label.text(0.5, 0.50, "1,000 KOs · fixed response order",
                  ha="center", va="center", fontsize=6.0,
                  color="#526573")

    labels.set_xlim(0, 1); labels.set_ylim(len(annotations) - 0.5, -0.5); clean(labels)
    selected = {"FeRhoNox live-cell dye": "FeRhoNox", "LAMP1": "LAMP1", "pRb*": "pRb"}
    for index, name in enumerate(annotations.short_name.astype(str)):
        if name in selected:
            colour = BIOLOGY_COLORS[annotations.iloc[index].biological_category]
            # A coloured leader begins at the matrix edge and terminates at the
            # exact reporter row, so the direct label cannot be read as an
            # approximate vertical grouping.
            labels.plot([-0.10, 0.14], [index, index], color=colour, lw=0.85,
                        solid_capstyle="round", clip_on=False, zorder=2)
            labels.scatter([0.02], [index], s=5.5, color=colour,
                           edgecolor="white", linewidth=0.25, clip_on=False, zorder=3)
            labels.text(0.20, index, selected[name], fontsize=6.0, va="center",
                        color=colour, fontweight="bold")

    colourbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[0, 0.5, 1])
    colourbar.outline.set_linewidth(0.45)
    cax.tick_params(labelsize=6.0, length=1.2, pad=1)
    cax.set_xlabel("within-reporter response percentile", fontsize=6.0, labelpad=1)

    assert_text_layout(fig)
    out = bundle_root() / "figure" / "Figure1-d_ko_response_atlas"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(out.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    build()
