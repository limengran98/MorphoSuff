#!/usr/bin/env python3
"""Build Supplementary Figure 2b from the frozen six-reporter representation pilot.

The three phase crops are the registered examples deposited with this supplementary
source package. The matrix contains every reporter x representation mean, and the forest
uses the deposited reporter-hierarchical bootstrap intervals.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
PAPER = PACKAGE_ROOT
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from _portable_fonts import configure_sans  # noqa: E402
MATRIX = HERE / "figure_source_representation_matrix.csv"
CONTRASTS = HERE / "figure_source_paired_contrasts.csv"
SELECTION = HERE / "figure_source_phase_crops.csv"
CROP_ROOT = HERE / "source_images"

MM = 1 / 25.4
INK = "#202A33"
MUTED = "#6F7E88"
MID = "#91A0A9"
GRID = "#DCE3E7"
WHITE = "#FFFFFF"
TEAL = "#3E6F9C"
GOLD = "#DDA06A"
RED = "#D56C75"

REPRESENTATIONS = [
    "DINOv2 frozen",
    "Cytoland frozen",
    "Cytoland partial fine-tune",
    "Official 172D + MLP",
]
REP_LABELS = ["DINOv2", "Cytoland", "Cytoland FT", "Engineered 172D"]
REPORTERS = [
    "stress_granule_g3bp1",
    "early_endosome_eea1",
    "lysosome_lamp1",
    "mitochondria_tomm20",
    "ps6",
    "nucleoli_npm1",
]
REPORTER_LABELS = {
    "early_endosome_eea1": "EEA1",
    "lysosome_lamp1": "LAMP1",
    "mitochondria_tomm20": "TOMM20",
    "nucleoli_npm1": "NPM1",
    "ps6": "pS6",
    "stress_granule_g3bp1": "G3BP1",
}
CROP_REPORTERS = ["early_endosome_eea1", "mitochondria_tomm20", "nucleoli_npm1"]
CROP_COLORS = {
    "early_endosome_eea1": "#C6A16A",
    "mitochondria_tomm20": "#C85A3C",
    "nucleoli_npm1": "#3E6F9C",
}
# All three deposited crops are native OPS level-0 images (0.325 µm px−1).
# The historical source table contains display windows only, so retain the
# physical calibration as an explicit panel constant and mirror it in the
# central microscopy asset manifest.
PIXEL_SIZE_UM = 0.325


def configure() -> None:
    configure_sans(mpl, fm, PAPER)
    mpl.rcParams.update({
        "font.size": 6.2,
        "axes.labelsize": 6.2,
        "axes.titlesize": 6.7,
        "axes.linewidth": 0.58,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    })


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = pd.read_csv(MATRIX)
    contrasts = pd.read_csv(CONTRASTS)
    selection = pd.read_csv(SELECTION)
    if len(matrix) != 24 or matrix.reporter_slug.nunique() != 6 or matrix.representation.nunique() != 4:
        raise SystemExit("panel-local representation matrix is not the frozen 6 x 4 summary")
    if set(REPRESENTATIONS) != set(matrix.representation.unique()):
        raise SystemExit("representation roster differs from the frozen panel")
    if len(contrasts) != 3 or len(selection) != 3:
        raise SystemExit("contrast or crop-selection source is incomplete")
    required = {"reporter_slug", "representation", "gene_pearson_mean", "gene_pearson_sd", "n_folds"}
    missing = sorted(required.difference(matrix.columns))
    if missing:
        raise SystemExit(f"representation matrix is missing columns: {missing}")
    if not matrix["n_folds"].eq(5).all():
        raise SystemExit("representation matrix does not retain the frozen five-fold summaries")
    if matrix.isna().any().any():
        raise SystemExit("representation matrix contains missing values")
    return matrix, contrasts, selection


def _draw_panel_e_compact(container, matrix: pd.DataFrame, contrasts: pd.DataFrame,
                          selection: pd.DataFrame, add_letter: bool):
    """Compact 73 × 105 mm vertical grammar for asymmetric assembly."""
    outer = container.add_gridspec(
        3, 1, height_ratios=[0.27, 0.42, 0.31],
        left=0.205, right=0.975, top=0.92, bottom=0.105, hspace=0.58,
    )
    selection_i = selection.set_index("reporter_slug")
    image_grid = outer[0, 0].subgridspec(1, 3, wspace=0.08)
    image_axes = []
    for i, reporter in enumerate(CROP_REPORTERS):
        ax = container.add_subplot(image_grid[0, i]); image_axes.append(ax)
        arr = np.load(CROP_ROOT / f"{reporter}_phase_crop.npy")
        rec = selection_i.loc[reporter]
        ax.imshow(arr, cmap="gray", vmin=float(rec.display_low), vmax=float(rec.display_high),
                  interpolation="none", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color(CROP_COLORS[reporter]); spine.set_linewidth(0.75)
        ax.text(0.025, 0.975, REPORTER_LABELS[reporter], transform=ax.transAxes,
                ha="left", va="top", fontsize=5.0, fontweight="bold",
                color=CROP_COLORS[reporter],
                bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.78, "pad": 0.25})
        # 20 μm at 0.325 μm px−1 is 61.54 native image pixels.  Use data
        # coordinates so the visual scale bar remains physically calibrated
        # after any Axes resize.
        bar_px = 20.0 / PIXEL_SIZE_UM
        x_mid = (arr.shape[1] - 1) / 2.0
        x0, x1 = x_mid - bar_px / 2.0, x_mid + bar_px / 2.0
        y_bar = arr.shape[0] - 25.0
        effects = [path_effects.Stroke(linewidth=2.4, foreground="black"), path_effects.Normal()]
        line, = ax.plot([x0, x1], [y_bar, y_bar], color=WHITE, lw=1.35,
                        solid_capstyle="butt")
        line.set_path_effects(effects)
        label = ax.text((x0 + x1) / 2.0, y_bar - 8.0, "20 µm", ha="center",
                        va="bottom", color=WHITE, fontsize=5.2, fontweight="bold")
        label.set_path_effects(effects)
    container.text(0.20, 0.982, "Registered phase inputs", ha="left", va="top",
                   fontsize=6.5, fontweight="bold", color=INK)

    axm = container.add_subplot(outer[1, 0])
    cmap_mean = mpl.colormaps["GnBu"]; cmap_sd = mpl.colormaps["YlOrBr"]
    norm_mean = Normalize(0.35, 0.95)
    norm_sd = Normalize(0.0, max(0.06, float(matrix.gene_pearson_sd.max())))
    ymap = {r: len(REPORTERS) - 1 - i for i, r in enumerate(REPORTERS)}
    for row in matrix.itertuples(index=False):
        x = REPRESENTATIONS.index(row.representation); y = ymap[row.reporter_slug]
        axm.add_patch(Rectangle((x - 0.45, y - 0.40), 0.70, 0.80,
                                facecolor=cmap_mean(norm_mean(row.gene_pearson_mean)),
                                edgecolor=WHITE, lw=0.45))
        axm.add_patch(Rectangle((x + 0.25, y - 0.40), 0.20, 0.80,
                                facecolor=cmap_sd(norm_sd(row.gene_pearson_sd)),
                                edgecolor=WHITE, lw=0.35))
        axm.text(x - 0.10, y, f"{row.gene_pearson_mean:.2f}", ha="center", va="center",
                 fontsize=5.0, fontweight="bold",
                 color=WHITE if row.gene_pearson_mean > 0.72 else INK)
    axm.set_xlim(-0.55, 3.55); axm.set_ylim(-0.55, 5.55)
    axm.set_xticks(range(4), REP_LABELS, rotation=23, ha="right")
    axm.xaxis.tick_top(); axm.tick_params(axis="x", top=True, labeltop=True,
                                         bottom=False, labelbottom=False, pad=1)
    axm.set_yticks([ymap[r] for r in REPORTERS], [REPORTER_LABELS[r] for r in REPORTERS])
    axm.tick_params(length=0)
    for spine in axm.spines.values(): spine.set_visible(False)
    axm.set_title("Reporter × representation", loc="left", pad=4,
                  fontsize=6.5, fontweight="bold", color=INK)
    cax1 = axm.inset_axes([0.10, -0.15, 0.34, 0.025])
    cax2 = axm.inset_axes([0.62, -0.15, 0.23, 0.025])
    cb1 = container.colorbar(mpl.cm.ScalarMappable(norm=norm_mean, cmap=cmap_mean),
                             cax=cax1, orientation="horizontal")
    cb2 = container.colorbar(mpl.cm.ScalarMappable(norm=norm_sd, cmap=cmap_sd),
                             cax=cax2, orientation="horizontal")
    cb1.set_ticks([0.4, 0.9]); cb2.set_ticks([0.0, round(norm_sd.vmax, 2)])
    cb1.ax.tick_params(labelsize=5.0, length=1.1, pad=0.5)
    cb2.ax.tick_params(labelsize=5.0, length=1.1, pad=0.5)
    cb1.set_label("mean r", fontsize=5.0, labelpad=0.5)
    cb2.set_label("fold SD", fontsize=5.0, labelpad=0.5)

    axf = container.add_subplot(outer[2, 0])
    label_map = {
        "fine-tune minus frozen Cytoland": "FT − Cytoland",
        "fine-tune minus frozen DINOv2": "FT − DINOv2",
        "fine-tune minus official 172D": "FT − 172D",
    }
    draw = contrasts.copy(); draw["label"] = draw.contrast.map(label_map)
    draw = draw.set_index("contrast").loc[[
        "fine-tune minus frozen DINOv2", "fine-tune minus frozen Cytoland",
        "fine-tune minus official 172D",
    ]].reset_index()
    yy = np.arange(len(draw))[::-1]
    axf.axvline(0, color=INK, lw=0.6); axf.axvspan(-0.02, 0.02, color="#EEF2F4")
    for y, row in zip(yy, draw.itertuples(index=False)):
        low = row.reporter_hierarchical_bootstrap_ci_low
        high = row.reporter_hierarchical_bootstrap_ci_high
        effect = row.mean_paired_delta
        colour = TEAL if low > 0 else RED if high < 0 else GOLD
        axf.hlines(y, low, high, color=colour, lw=1.45)
        axf.scatter(effect, y, s=22, color=colour, edgecolor=WHITE, lw=0.45, zorder=3)
        axf.text(0.205, y, f"{int(row.reporters_with_positive_mean_delta)}/6",
                 ha="right", va="center", fontsize=5.0, color=INK)
    axf.set_yticks(yy, draw.label)
    axf.set_xlim(-0.21, 0.21); axf.set_ylim(-0.65, 2.65)
    axf.set_xticks([-0.2, 0, 0.2]); axf.set_xlabel("Paired Δ held-out-gene r")
    axf.set_title("Paired contrasts", loc="left", pad=1,
                  fontsize=6.5, fontweight="bold", color=INK)
    axf.text(0.98, 1.03, "reporters with Δ>0", transform=axf.transAxes,
             ha="right", va="bottom", fontsize=5.0, color=INK)
    axf.grid(axis="x", color=GRID, lw=0.32)
    axf.spines[["top", "right"]].set_visible(False)
    axf.spines[["left", "bottom"]].set_color("#6F7E88")
    if add_letter:
        container.text(0.006, 0.985, "e", ha="left", va="top",
                       fontsize=8.5, fontweight="bold", color=INK)
    return {"image_axes": image_axes, "matrix_axis": axm, "contrast_axis": axf,
            "matrix_table": matrix, "contrast_table": contrasts}


def draw_panel_e(container, add_letter: bool = False, compact: bool = False):
    """Draw panel e natively into a Matplotlib ``SubFigure``."""
    configure()
    matrix, contrasts, selection = load_and_validate()

    if compact:
        return _draw_panel_e_compact(container, matrix, contrasts, selection, add_letter)

    outer = container.add_gridspec(
        1, 3, width_ratios=[0.27, 0.43, 0.30],
        left=0.035, right=0.985, top=0.855, bottom=0.15, wspace=0.36,
    )

    # Registered raw phase crops. Each crop uses its deposited deterministic window.
    image_grid = outer[0, 0].subgridspec(3, 1, hspace=0.19)
    selection_i = selection.set_index("reporter_slug")
    image_axes = []
    for i, reporter in enumerate(CROP_REPORTERS):
        ax = container.add_subplot(image_grid[i, 0])
        image_axes.append(ax)
        arr = np.load(CROP_ROOT / f"{reporter}_phase_crop.npy")
        rec = selection_i.loc[reporter]
        ax.imshow(arr, cmap="gray", vmin=float(rec.display_low), vmax=float(rec.display_high),
                  interpolation="none", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color(CROP_COLORS[reporter]); spine.set_linewidth(0.9)
        ax.text(0.01, 1.02, REPORTER_LABELS[reporter], transform=ax.transAxes,
                ha="left", va="bottom", fontsize=5.2, fontweight="bold",
                color=CROP_COLORS[reporter], clip_on=False)
        bar_px = 20.0 / PIXEL_SIZE_UM
        x_mid = (arr.shape[1] - 1) / 2.0
        x0, x1 = x_mid - bar_px / 2.0, x_mid + bar_px / 2.0
        y_bar = arr.shape[0] - 25.0
        effects = [path_effects.Stroke(linewidth=2.4, foreground="black"), path_effects.Normal()]
        line, = ax.plot([x0, x1], [y_bar, y_bar], color=WHITE, lw=1.35,
                        solid_capstyle="butt")
        line.set_path_effects(effects)
        label = ax.text((x0 + x1) / 2.0, y_bar - 8.0, "20 µm", ha="center",
                        va="bottom", color=WHITE, fontsize=5.2, fontweight="bold")
        label.set_path_effects(effects)
    container.text(0.035, 0.942, "Registered phase inputs", ha="left", va="bottom",
                   fontsize=6.7, fontweight="bold", color=INK)

    # Complex 6 x 4 matrix: the main cell encodes mean recovery, the narrow right
    # strip encodes fold SD. Values are printed because this matrix is intentionally
    # small enough to remain auditable at 183-mm width.
    axm = container.add_subplot(outer[0, 1])
    cmap_mean = mpl.colormaps["GnBu"]
    cmap_sd = mpl.colormaps["YlOrBr"]
    norm_mean = Normalize(0.35, 0.95)
    norm_sd = Normalize(0.0, max(0.06, float(matrix.gene_pearson_sd.max())))
    ymap = {r: len(REPORTERS) - 1 - i for i, r in enumerate(REPORTERS)}
    for row in matrix.itertuples(index=False):
        x = REPRESENTATIONS.index(row.representation)
        y = ymap[row.reporter_slug]
        axm.add_patch(Rectangle((x - 0.45, y - 0.40), 0.70, 0.80,
                                facecolor=cmap_mean(norm_mean(row.gene_pearson_mean)),
                                edgecolor=WHITE, lw=0.55))
        axm.add_patch(Rectangle((x + 0.25, y - 0.40), 0.20, 0.80,
                                facecolor=cmap_sd(norm_sd(row.gene_pearson_sd)),
                                edgecolor=WHITE, lw=0.45))
        text_colour = WHITE if row.gene_pearson_mean > 0.72 else INK
        axm.text(x - 0.10, y, f"{row.gene_pearson_mean:.2f}", ha="center", va="center",
                 fontsize=4.7, fontweight="bold", color=text_colour)
    axm.set_xlim(-0.55, 3.55); axm.set_ylim(-0.55, 5.55)
    axm.set_xticks(range(4), REP_LABELS, rotation=24, ha="right")
    axm.xaxis.tick_top(); axm.tick_params(axis="x", top=True, labeltop=True,
                                         bottom=False, labelbottom=False, pad=2)
    axm.set_yticks([ymap[r] for r in REPORTERS], [REPORTER_LABELS[r] for r in REPORTERS])
    axm.tick_params(length=0)
    for spine in axm.spines.values(): spine.set_visible(False)
    container.text(0.345, 0.942, "Reporter × representation", ha="left", va="bottom",
                   fontsize=6.7, fontweight="bold", color=INK)
    cax1 = axm.inset_axes([0.10, -0.13, 0.35, 0.026])
    cax2 = axm.inset_axes([0.60, -0.13, 0.24, 0.026])
    cb1 = container.colorbar(mpl.cm.ScalarMappable(norm=norm_mean, cmap=cmap_mean), cax=cax1,
                       orientation="horizontal")
    cb2 = container.colorbar(mpl.cm.ScalarMappable(norm=norm_sd, cmap=cmap_sd), cax=cax2,
                       orientation="horizontal")
    cb1.set_ticks([0.4, 0.7, 0.9]); cb2.set_ticks([0.0, round(norm_sd.vmax, 2)])
    cb1.ax.tick_params(labelsize=4.1, length=1.4, pad=1)
    cb2.ax.tick_params(labelsize=4.1, length=1.4, pad=1)
    cb1.set_label("mean held-out-gene r", fontsize=4.5, labelpad=1)
    cb2.set_label("fold SD", fontsize=4.5, labelpad=1)

    # Deposited reporter-hierarchical bootstrap contrasts.
    axf = container.add_subplot(outer[0, 2])
    label_map = {
        "fine-tune minus frozen Cytoland": "FT − Cytoland",
        "fine-tune minus frozen DINOv2": "FT − DINOv2",
        "fine-tune minus official 172D": "FT − 172D",
    }
    draw = contrasts.copy()
    draw["label"] = draw.contrast.map(label_map)
    draw = draw.set_index("contrast").loc[[
        "fine-tune minus frozen DINOv2",
        "fine-tune minus frozen Cytoland",
        "fine-tune minus official 172D",
    ]].reset_index()
    yy = np.arange(len(draw))[::-1]
    axf.axvline(0, color=INK, lw=0.65)
    axf.axvspan(-0.02, 0.02, color="#EEF2F4", zorder=0)
    for y, row in zip(yy, draw.itertuples(index=False)):
        low = row.reporter_hierarchical_bootstrap_ci_low
        high = row.reporter_hierarchical_bootstrap_ci_high
        effect = row.mean_paired_delta
        if low > 0:
            colour = TEAL
        elif high < 0:
            colour = RED
        else:
            colour = GOLD
        axf.hlines(y, low, high, color=colour, lw=1.7, zorder=2)
        axf.scatter(effect, y, s=27, color=colour, edgecolor=WHITE, lw=0.55, zorder=3)
        axf.text(0.205, y, f"{int(row.reporters_with_positive_mean_delta)}/6",
                 ha="right", va="center", fontsize=4.7, color=INK)
    axf.set_yticks(yy, draw.label)
    axf.set_xlim(-0.21, 0.21); axf.set_ylim(-0.65, 2.65)
    axf.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    axf.set_xlabel("Paired Δ held-out-gene r")
    axf.set_title("Paired contrasts", loc="left", pad=7,
                  fontsize=6.7, fontweight="bold", color=INK)
    axf.text(0.98, 1.035, "reporters with Δ>0", transform=axf.transAxes,
             ha="right", va="bottom", fontsize=4.6, color=INK)
    axf.grid(axis="x", color=GRID, lw=0.38)
    axf.spines[["top", "right"]].set_visible(False)
    axf.spines[["left", "bottom"]].set_color("#6F7E88")

    if add_letter:
        container.text(0.006, 0.985, "e", ha="left", va="top",
                       fontsize=8.5, fontweight="bold", color=INK)

    return {
        "image_axes": image_axes,
        "matrix_axis": axm,
        "contrast_axis": axf,
        "matrix_table": matrix,
        "contrast_table": contrasts,
    }


def build() -> None:
    configure()
    matrix, contrasts, selection = load_and_validate()
    matrix.to_csv(HERE / "figure_source_representation_matrix.csv", index=False)
    contrasts.to_csv(HERE / "figure_source_paired_contrasts.csv", index=False)
    selection.to_csv(HERE / "figure_source_phase_crops.csv", index=False)

    fig = plt.figure(figsize=(183 * MM, 76 * MM), facecolor=WHITE)
    subfig = fig.subfigures(1, 1)
    draw_panel_e(subfig, add_letter=False)

    fig.savefig(HERE / "SupplementaryFigure2-b.png", dpi=450)
    fig.savefig(HERE / "SupplementaryFigure2-b.pdf")
    fig.savefig(HERE / "SupplementaryFigure2-b.svg")
    plt.close(fig)

    compact_fig = plt.figure(figsize=(73 * MM, 105 * MM), facecolor=WHITE)
    compact_subfig = compact_fig.subfigures(1, 1)
    draw_panel_e(compact_subfig, add_letter=False, compact=True)
    compact_fig.savefig(HERE / "SupplementaryFigure2-b-compact.png", dpi=450)
    compact_fig.savefig(HERE / "SupplementaryFigure2-b-compact.pdf")
    compact_fig.savefig(HERE / "SupplementaryFigure2-b-compact.svg")
    plt.close(compact_fig)


if __name__ == "__main__":
    build()
