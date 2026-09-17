#!/usr/bin/env python3
"""Build Supplementary Figure 2a from the frozen low-label tables.

The panel deliberately keeps the 100% result visually separate: it is a matched,
independently frozen full-label campaign rather than a fourth point in the same
low-label run.  No model is fitted and no statistical test is run here.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Wedge
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PANEL_DATA = HERE / "figure_source_data.csv"
PANEL_EFFICIENCY = HERE / "figure_source_label_efficiency.csv"
PACKAGE_ROOT = HERE.parent
PAPER = PACKAGE_ROOT
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
from _portable_fonts import configure_sans  # noqa: E402

MM = 1 / 25.4
INK = "#202A33"
MID = "#7E8D96"
GRID = "#DCE3E7"
PALE = "#F4F6F7"
WHITE = "#FFFFFF"

FRACTIONS = [0.001, 0.01, 0.2, 1.0]
FRACTION_LABELS = ["0.1%", "1%", "20%", "100%"]
FAMILIES = [
    "Classical tabular",
    "Neural specialists",
    "Generic multi-task",
    "Biological completion",
]


def configure() -> None:
    configure_sans(mpl, fm, PAPER)
    mpl.rcParams.update({
        "font.size": 6.2,
        "axes.labelsize": 6.2,
        "axes.titlesize": 6.6,
        "axes.linewidth": 0.58,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "legend.fontsize": 5.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    })


def despine(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#6F7E88")


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame]:
    long = pd.read_csv(PANEL_DATA)
    score = pd.read_csv(PANEL_EFFICIENCY)
    required_long = {
        "method", "fraction", "gene_macro_feature_pearson_mean",
        "gene_macro_feature_pearson_sd", "n_folds", "n_reporters", "source_regime",
        "method_color", "method_family", "method_order", "full_label_anchor",
        "relative_to_full_anchor", "fraction_label",
    }
    required_score = {
        "method", "method_order", "method_family", "method_color",
        "relative_log_auc_0p1_to_20pct",
    }
    if len(long) != 40 or long.method.nunique() != 10 or set(long.fraction) != set(FRACTIONS):
        raise SystemExit("panel-local low-label source is not the frozen 10-method x 4-budget table")
    if len(score) != 10 or score.method.nunique() != 10:
        raise SystemExit("panel-local label-efficiency source is incomplete")
    missing_long = sorted(required_long.difference(long.columns))
    missing_score = sorted(required_score.difference(score.columns))
    if missing_long or missing_score:
        raise SystemExit(f"panel-local source missing columns: {missing_long + missing_score}")
    long["fraction_label"] = pd.Categorical(long["fraction_label"], FRACTION_LABELS, ordered=True)
    long = long.sort_values(["method_order", "fraction"], kind="stable")
    score = score.sort_values(["relative_log_auc_0p1_to_20pct", "method_order"], ascending=[False, True])
    if long.isna().any().any() or score.isna().any().any():
        raise SystemExit("panel-local display table contains missing values")
    return long, score


def draw_split_glyph(
    ax: plt.Axes, x: float, y: float, absolute: float, relative: float,
    cmap_abs, cmap_rel, norm_abs: Normalize, norm_rel: Normalize,
) -> None:
    radius = 0.29
    ax.add_patch(Wedge((x, y), radius, 90, 270,
                       facecolor=cmap_abs(norm_abs(absolute)), edgecolor="none"))
    ax.add_patch(Wedge((x, y), radius, -90, 90,
                       facecolor=cmap_rel(norm_rel(relative)), edgecolor="none"))
    ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor=WHITE, lw=0.65))
    ax.plot([x, x], [y - radius, y + radius], color=WHITE, lw=0.48, solid_capstyle="butt")


def _draw_panel_d_compact(container, long: pd.DataFrame, score: pd.DataFrame,
                          add_letter: bool):
    """Compact 105 × 65 mm grammar for asymmetric manuscript assembly."""
    outer = container.add_gridspec(
        1, 2, width_ratios=[0.53, 0.47],
        left=0.095, right=0.985, top=0.925, bottom=0.17, wspace=0.34,
    )
    curve_grid = outer[0, 0].subgridspec(4, 1, hspace=0.17)
    curve_axes = [container.add_subplot(curve_grid[i, 0]) for i in range(4)]
    x = np.array([0.0, 1.0, 2.0, 3.12])
    for idx, (family, ax) in enumerate(zip(FAMILIES, curve_axes)):
        part = long.loc[long.method_family.eq(family)]
        ax.axvspan(2.70, 3.35, color=PALE, zorder=0)
        ax.axvline(2.70, color="#AEB9BE", lw=0.5, ls=(0, (2.0, 1.6)))
        for method_index, (method, frame) in enumerate(part.groupby("method", sort=False)):
            frame = frame.sort_values("fraction")
            colour = frame.method_color.iloc[0]
            y = frame.gene_macro_feature_pearson_mean.to_numpy(float)
            sd = frame.gene_macro_feature_pearson_sd.to_numpy(float)
            ax.fill_between(x[:3], y[:3] - sd[:3], y[:3] + sd[:3],
                            color=colour, alpha=0.10, linewidth=0)
            ax.plot(x[:3], y[:3], color=colour, lw=0.95, marker="o", ms=2.25,
                    markeredgecolor=WHITE, markeredgewidth=0.30)
            ax.plot(x[2:], y[2:], color=colour, lw=0.70, ls=(0, (2.2, 1.5)))
            ax.errorbar(x[3], y[3], yerr=sd[3], color=colour, marker="D", ms=2.45,
                        markeredgecolor=WHITE, markeredgewidth=0.30,
                        elinewidth=0.45, capsize=0.8)
            # Do not repeat individual method names at 3.9 pt in the curve facet.
            # The adjacent, colour-matched 10-method roster provides the complete
            # identity key at publication scale; the curves retain every trajectory.
        # A quiet in-panel header band avoids the title/facet-label collision while
        # preserving an unobscured region above every frozen trajectory.
        ax.axhspan(0.790, 0.830, color=PALE, zorder=0.4)
        ax.text(0.025, 0.808, family, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=5.0, color=INK,
                fontweight="bold", clip_on=False, zorder=4)
        ax.set_xlim(-0.12, 3.35); ax.set_ylim(0.39, 0.83)
        ax.set_yticks([0.5, 0.7]); ax.grid(axis="y", color=GRID, lw=0.32)
        despine(ax)
        if idx < 3:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            ax.set_xticks(x, FRACTION_LABELS)
            ax.set_xlabel("Target labels available", labelpad=2)
    container.text(0.020, 0.53, "Held-out-gene Pearson r", rotation=90,
                   ha="center", va="center", fontsize=5.5, color=INK)
    curve_axes[0].set_title("Five-fold calibration", loc="left", pad=5,
                            fontsize=6.4, fontweight="bold", color=INK)

    right = outer[0, 1].subgridspec(1, 2, width_ratios=[0.74, 0.26], wspace=0.18)
    axm = container.add_subplot(right[0, 0])
    axe = container.add_subplot(right[0, 1])
    roster = long[["method", "method_order", "method_color", "method_family"]].drop_duplicates()
    roster = roster.sort_values("method_order")
    ymap = {m: len(roster) - 1 - i for i, m in enumerate(roster.method)}
    cmap_abs = mpl.colormaps["GnBu"]; cmap_rel = mpl.colormaps["YlOrBr"]
    norm_abs = Normalize(0.40, 0.82); norm_rel = Normalize(0.55, 1.05)
    axm.axvspan(2.55, 3.45, color=PALE, zorder=0)
    for row in long.itertuples(index=False):
        draw_split_glyph(
            axm, FRACTIONS.index(float(row.fraction)), ymap[row.method],
            float(row.gene_macro_feature_pearson_mean), float(row.relative_to_full_anchor),
            cmap_abs, cmap_rel, norm_abs, norm_rel,
        )
    axm.set_xlim(-0.55, 3.45); axm.set_ylim(-0.65, len(roster) - 0.35)
    axm.set_xticks(range(4), FRACTION_LABELS)
    axm.xaxis.tick_top(); axm.tick_params(axis="x", top=True, labeltop=True,
                                         bottom=False, labelbottom=False, pad=1)
    axm.set_yticks([ymap[m] for m in roster.method], roster.method)
    for tick, colour in zip(axm.get_yticklabels(), roster.method_color):
        tick.set_color(colour); tick.set_fontweight("bold")
    for boundary in [6.5, 3.5, 1.5]:
        axm.axhline(boundary, color=GRID, lw=0.45)
    for spine in axm.spines.values(): spine.set_visible(False)
    axm.tick_params(length=0)
    axm.set_title("absolute | relative", loc="left", pad=8,
                  fontsize=6.0, fontweight="bold", color=INK)

    rank = score.sort_values("relative_log_auc_0p1_to_20pct")
    for i, row in enumerate(rank.itertuples(index=False)):
        axe.hlines(i, 0.78, row.relative_log_auc_0p1_to_20pct,
                   color=GRID, lw=0.75)
        axe.scatter(row.relative_log_auc_0p1_to_20pct, i, s=12,
                    color=row.method_color, edgecolor=WHITE, lw=0.35, zorder=3)
    axe.axvline(1.0, color="#AEB9BE", lw=0.5, ls=(0, (2.0, 1.6)))
    axe.set_xlim(0.78, 1.055); axe.set_ylim(-0.65, len(rank) - 0.35)
    axe.set_xticks([0.8, 1.0]); axe.set_yticks([])
    axe.set_xlabel("rel. log-AULC", labelpad=2)
    axe.set_title("log-AULC", loc="left", pad=8, fontsize=6.0,
                  fontweight="bold", color=INK)
    axe.grid(axis="x", color=GRID, lw=0.32); despine(axe)

    if add_letter:
        container.text(0.006, 0.985, "d", ha="left", va="top",
                       fontsize=8.5, fontweight="bold", color=INK)
    return {
        "curve_axes": curve_axes, "matrix_axis": axm, "efficiency_axis": axe,
        "long_table": long, "efficiency_table": score,
    }


def draw_panel_d(container, add_letter: bool = False, compact: bool = False):
    """Draw panel d into a Matplotlib ``SubFigure``.

    The function intentionally owns only the panel-local layout so the final
    manuscript composite can call it without rasterising or re-importing a PNG.
    """
    configure()
    long, score = load_and_validate()

    if compact:
        return _draw_panel_d_compact(container, long, score, add_letter)

    outer = container.add_gridspec(
        1, 3, width_ratios=[0.39, 0.38, 0.23],
        left=0.060, right=0.985, top=0.915, bottom=0.178, wspace=0.29,
    )

    # Four family facets avoid a ten-line spaghetti plot while retaining every method.
    curve_grid = outer[0, 0].subgridspec(4, 1, hspace=0.16)
    curve_axes = [container.add_subplot(curve_grid[i, 0]) for i in range(4)]
    x = np.array([0.0, 1.0, 2.0, 3.15])
    for idx, (family, ax) in enumerate(zip(FAMILIES, curve_axes)):
        part = long.loc[long.method_family.eq(family)]
        ax.axvspan(2.72, 3.42, color=PALE, zorder=0)
        ax.axvline(2.72, color="#AEB9BE", lw=0.55, ls=(0, (2.0, 1.6)), zorder=1)
        for method_index, (method, frame) in enumerate(part.groupby("method", sort=False)):
            frame = frame.sort_values("fraction")
            colour = frame.method_color.iloc[0]
            y = frame.gene_macro_feature_pearson_mean.to_numpy(float)
            sd = frame.gene_macro_feature_pearson_sd.to_numpy(float)
            ax.fill_between(x[:3], y[:3] - sd[:3], y[:3] + sd[:3],
                            color=colour, alpha=0.10, linewidth=0)
            ax.plot(x[:3], y[:3], color=colour, lw=1.10, marker="o", ms=2.8,
                    markeredgecolor=WHITE, markeredgewidth=0.35)
            ax.plot(x[2:], y[2:], color=colour, lw=0.78, ls=(0, (2.2, 1.5)), alpha=0.82)
            ax.errorbar(x[3], y[3], yerr=sd[3], color=colour, marker="D", ms=3.1,
                        markeredgecolor=WHITE, markeredgewidth=0.35,
                        elinewidth=0.55, capsize=1.0, zorder=4)
            # Tether labels to distinct trajectory positions.  The two crowded
            # endpoints are offset below the solid segment; the others sit above.
            anchor_index = min(method_index, 2)
            xoff, yoff = (7, -9) if method in {"CatBoost", "scButterfly"} else (4, 7)
            ax.annotate(
                method, xy=(x[anchor_index], y[anchor_index]),
                xytext=(xoff, yoff), textcoords="offset points",
                ha="left", va="center", fontsize=4.65, color=colour,
                fontweight="bold",
                bbox={"facecolor": WHITE, "edgecolor": "none", "alpha": 0.82, "pad": 0.15},
                arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.45,
                            "shrinkA": 1.5, "shrinkB": 2.0},
                zorder=6,
            )
        # Keep the family label inside the facet.  This remains legible after
        # full-width composite assembly and cannot be clipped at the page edge.
        ax.text(0.005, 1.015, family, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=4.55, color=INK, fontweight="bold", clip_on=False)
        ax.set_xlim(-0.12, 3.42)
        ax.set_ylim(0.39, 0.83)
        ax.set_yticks([0.5, 0.7])
        ax.grid(axis="y", color=GRID, lw=0.38)
        despine(ax)
        if idx < 3:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(axis="x", bottom=False, labelbottom=False)
        else:
            ax.set_xticks(x, FRACTION_LABELS)
            ax.set_xlabel("Target labels available")
        if idx == 1:
            container.text(0.014, 0.53, "Held-out-gene Pearson r", rotation=90,
                           ha="center", va="center", fontsize=6.2, color=INK)
    curve_axes[0].text(3.06, 0.845, "independent\n100% anchor", ha="center", va="bottom",
                       fontsize=4.5, color=INK, clip_on=False)
    curve_axes[0].set_title("Five-fold calibration trajectories", loc="left", pad=6,
                            fontsize=6.7, fontweight="bold", color=INK)

    # Matrix of absolute performance (left semicircle) and fraction of each method's
    # full-label anchor (right semicircle).
    axm = container.add_subplot(outer[0, 1])
    roster = long[["method", "method_order", "method_color", "method_family"]].drop_duplicates()
    roster = roster.sort_values("method_order")
    ymap = {m: len(roster) - 1 - i for i, m in enumerate(roster.method)}
    cmap_abs = mpl.colormaps["GnBu"]
    cmap_rel = mpl.colormaps["YlOrBr"]
    norm_abs = Normalize(0.40, 0.82)
    norm_rel = Normalize(0.55, 1.05)
    axm.axvspan(2.58, 3.42, color=PALE, zorder=0)
    axm.axvline(2.58, color="#AEB9BE", lw=0.55, ls=(0, (2.0, 1.6)))
    for row in long.itertuples(index=False):
        xi = FRACTIONS.index(float(row.fraction))
        yi = ymap[row.method]
        draw_split_glyph(
            axm, xi, yi, float(row.gene_macro_feature_pearson_mean),
            float(row.relative_to_full_anchor), cmap_abs, cmap_rel, norm_abs, norm_rel,
        )
    axm.set_xlim(-0.55, 3.48)
    axm.set_ylim(-0.7, len(roster) - 0.3)
    axm.set_xticks(range(4), FRACTION_LABELS)
    axm.xaxis.tick_top()
    axm.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=1.5)
    axm.set_yticks([ymap[m] for m in roster.method], roster.method)
    for tick, colour in zip(axm.get_yticklabels(), roster.method_color):
        tick.set_color(colour)
        tick.set_fontweight("bold")
    for boundary in [6.5, 3.5, 1.5]:
        axm.axhline(boundary, color=GRID, lw=0.55)
    axm.set_title("Absolute recovery | relative to full label", loc="left", pad=11,
                  fontsize=6.35, fontweight="bold", color=INK)
    for spine in axm.spines.values():
        spine.set_visible(False)
    axm.tick_params(length=0)
    cax_abs = axm.inset_axes([0.09, -0.135, 0.31, 0.025])
    cax_rel = axm.inset_axes([0.57, -0.135, 0.31, 0.025])
    cb1 = container.colorbar(mpl.cm.ScalarMappable(norm=norm_abs, cmap=cmap_abs), cax=cax_abs,
                       orientation="horizontal")
    cb2 = container.colorbar(mpl.cm.ScalarMappable(norm=norm_rel, cmap=cmap_rel), cax=cax_rel,
                       orientation="horizontal")
    cb1.set_ticks([0.4, 0.8]); cb2.set_ticks([0.6, 1.0])
    cb1.ax.tick_params(labelsize=4.2, length=1.5, pad=1)
    cb2.ax.tick_params(labelsize=4.2, length=1.5, pad=1)
    cb1.set_label("absolute r", fontsize=4.5, labelpad=1)
    cb2.set_label("relative", fontsize=4.5, labelpad=1)

    # Descriptive label-efficiency rank. This is not an inferential model comparison.
    axe = container.add_subplot(outer[0, 2])
    rank = score.sort_values("relative_log_auc_0p1_to_20pct")
    yy = np.arange(len(rank))
    for i, row in enumerate(rank.itertuples(index=False)):
        axe.hlines(i, 0.78, row.relative_log_auc_0p1_to_20pct,
                   color=GRID, lw=1.0, zorder=1)
        axe.scatter(row.relative_log_auc_0p1_to_20pct, i, s=20,
                    color=row.method_color, edgecolor=WHITE, lw=0.45, zorder=3)
    axe.axvline(1.0, color="#AEB9BE", lw=0.55, ls=(0, (2.0, 1.6)))
    axe.set_yticks(yy, rank.method)
    for tick, colour in zip(axe.get_yticklabels(), rank.method_color):
        tick.set_color(colour); tick.set_fontweight("bold")
    axe.set_xlim(0.78, 1.055)
    axe.set_xticks([0.8, 0.9, 1.0])
    axe.set_xlabel("Relative log-AULC")
    axe.set_title("Label efficiency", loc="left", pad=6,
                  fontsize=6.7, fontweight="bold", color=INK)
    axe.grid(axis="x", color=GRID, lw=0.38)
    despine(axe)
    axe.text(0.46, -0.18, "0.1–20%; each method's 100% anchor",
             transform=axe.transAxes, ha="center", va="top", fontsize=4.4,
             color=INK, clip_on=False)

    if add_letter:
        container.text(0.006, 0.985, "d", ha="left", va="top",
                       fontsize=8.5, fontweight="bold", color=INK)

    return {
        "curve_axes": curve_axes,
        "matrix_axis": axm,
        "efficiency_axis": axe,
        "long_table": long,
        "efficiency_table": score,
    }


def build() -> None:
    configure()
    long, score = load_and_validate()
    long.to_csv(HERE / "figure_source_data.csv", index=False)
    score.to_csv(HERE / "figure_source_label_efficiency.csv", index=False)

    fig = plt.figure(figsize=(183 * MM, 78 * MM), facecolor=WHITE)
    subfig = fig.subfigures(1, 1)
    draw_panel_d(subfig, add_letter=False)

    fig.savefig(HERE / "SupplementaryFigure2-a.png", dpi=450)
    fig.savefig(HERE / "SupplementaryFigure2-a.pdf")
    fig.savefig(HERE / "SupplementaryFigure2-a.svg")
    plt.close(fig)

    compact_fig = plt.figure(figsize=(105 * MM, 65 * MM), facecolor=WHITE)
    compact_subfig = compact_fig.subfigures(1, 1)
    draw_panel_d(compact_subfig, add_letter=False, compact=True)
    compact_fig.savefig(HERE / "SupplementaryFigure2-a-compact.png", dpi=450)
    compact_fig.savefig(HERE / "SupplementaryFigure2-a-compact.pdf")
    compact_fig.savefig(HERE / "SupplementaryFigure2-a-compact.svg")
    plt.close(compact_fig)


if __name__ == "__main__":
    build()
