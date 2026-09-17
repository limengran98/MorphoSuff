#!/usr/bin/env python3
"""Build Figure 6a: reporter-level counterfactual response-replacement utility atlas.

The builder is presentation-only.  It consumes the frozen Figure 6 display tables,
does not refit a model or change the reporter order, and exports the exact plotted
source table beside the figure.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIGURE6_ROOT = HERE.parent
PAPER_ROOT = FIGURE6_ROOT
if str(FIGURE6_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE6_ROOT))
from _portable_fonts import configure_sans, format_figure, layout_report  # noqa: E402

MM = 1 / 25.4
INK = "#202A33"
MUTED = "#6F7E88"
GRID = "#DCE3E7"
PALE = "#F4F6F7"
TEAL = "#3E6F9C"
TEAL_DARK = "#294A66"
WHITE = "#FFFFFF"
TIER_ORDER = [
    "quantitative_proxy",
    "ranking_proxy",
    "measurement_required",
    "not_identifiable",
    "unresolved",
]
TIER_COLORS = {
    "quantitative_proxy": "#3E6F9C",
    "ranking_proxy": "#DDA06A",
    "measurement_required": "#D56C75",
    "not_identifiable": "#7E8D96",
    "unresolved": "#C8D0D4",
}
TIER_LABELS = {
    "quantitative_proxy": "Quantitative",
    "ranking_proxy": "Ranking",
    "measurement_required": "Required",
    "not_identifiable": "Not ID",
    "unresolved": "Unresolved",
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
METRICS = [
    ("magnitude_spearman", "Rank"),
    ("top5pct_recall", "Top-5%"),
    ("go_bp_top10_jaccard", "GO BP"),
    ("go_cc_top10_jaccard", "GO CC"),
    ("ebi_complex_top10_jaccard", "Complex"),
]
SELECTED = {
    "nuclei_nucleolive_live_cell_dye",
    "nuclear_speckles_srrm2",
    "stress_granule_g3bp1",
    "prb",
    "p53",
}


def configure() -> None:
    configure_sans(mpl, font_manager, PAPER_ROOT)
    mpl.rcParams.update(
        {
            "font.size": 6.2,
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.3,
            "ytick.labelsize": 5.3,
            "axes.linewidth": 0.62,
            "axes.edgecolor": INK,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
        }
    )


def load_source() -> pd.DataFrame:
    """Load the released, self-contained reporter-level source table.

    This table is intentionally stored next to the renderer so a public clone
    does not need an unpublished display-table hierarchy.
    """
    merged = pd.read_csv(HERE / "figure_source_data.csv")
    required = {
        "display_order", "reporter_slug", "scientific_utility_index", "short_name",
        "biological_category", "substitutability_tier", "selected_label",
        *(column for column, _ in METRICS),
    }
    missing = sorted(required.difference(merged.columns))
    if missing:
        raise RuntimeError(f"Figure 6a source table is missing columns: {missing}")
    if len(merged) != 52 or merged["reporter_slug"].duplicated().any():
        raise RuntimeError("Figure 6a requires exactly 52 unique reporters")
    if merged[[c for c, _ in METRICS] + ["scientific_utility_index"]].isna().any().any():
        raise RuntimeError("Figure 6a canonical utility table contains missing plotted values")
    if merged[["biological_category", "substitutability_tier"]].isna().any().any():
        raise RuntimeError("Figure 6a annotation merge is incomplete")
    merged = merged.sort_values("display_order", kind="stable").reset_index(drop=True)
    merged["short_name"] = merged["short_name"].astype(str).str.replace("*", "", regex=False)
    merged.loc[
        merged["short_name"].eq("NucleoLIVE Live Cell dye"), "short_name"
    ] = "NucleoLIVE"
    # The set of visible labels is frozen with the released source table.  The
    # assertion protects against a silent change of its pre-specified roster.
    if not set(merged.loc[merged["selected_label"].astype(bool), "reporter_slug"]).issubset(SELECTED):
        raise RuntimeError("Figure 6a selected labels disagree with the released roster")
    return merged


def fixed_bandwidth_density(values: np.ndarray, grid: np.ndarray, bandwidth: float = 0.055) -> np.ndarray:
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))
    return density


def colored_strip(ax: plt.Axes, colors: list[str]) -> None:
    rgba = np.asarray([mpl.colors.to_rgba(c) for c in colors], dtype=float).reshape(-1, 1, 4)
    ax.imshow(rgba, aspect="auto", interpolation="nearest", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_panel_a(container, add_letter: bool = False, data: pd.DataFrame | None = None):
    """Draw panel a into a Matplotlib Figure or SubFigure.

    Parameters
    ----------
    container
        A :class:`matplotlib.figure.Figure` or ``SubFigure``.  All text remains
        native vector text, so this function can be used directly by the final
        manuscript compositor without rasterising a panel image.
    add_letter
        Add the manuscript panel letter in the upper-left corner.
    data
        Optional already-loaded canonical source table.  When omitted, the
        frozen Figure 6 table is loaded and exported beside this script.
    """

    if data is None:
        data = load_source()
    values = data[[c for c, _ in METRICS]].to_numpy(float)
    cmap = LinearSegmentedColormap.from_list("utility", [PALE, "#D7E5EF", "#79A8C9", TEAL_DARK])

    gs = container.add_gridspec(
        3,
        5,
        height_ratios=[1.12, 4.55, 0.34],
        width_ratios=[0.12, 0.12, 2.85, 1.35, 0.05],
        left=0.055,
        right=0.985,
        top=0.805,
        bottom=0.075,
        hspace=0.18,
        wspace=0.08,
    )
    ax_dist = container.add_subplot(gs[0, 2])
    ax_bio = container.add_subplot(gs[1, 0])
    ax_tier = container.add_subplot(gs[1, 1])
    ax_heat = container.add_subplot(gs[1, 2])
    ax_rank = container.add_subplot(gs[1, 3], sharey=ax_heat)
    ax_cbar = container.add_subplot(gs[2, 2])
    ax_key = container.add_subplot(gs[0, 3])

    # Shared-bandwidth half violins retain all 52 reporter-level observations.
    density_grid = np.linspace(0, 1, 220)
    rng = np.random.default_rng(20260831)
    for x, (column, label) in enumerate(METRICS):
        y = data[column].to_numpy(float)
        dens = fixed_bandwidth_density(y, density_grid)
        width = 0.34 * dens / max(dens.max(), 1e-12)
        ax_dist.fill_betweenx(density_grid, x, x + width, color="#BDD5E4", lw=0, alpha=0.92)
        ax_dist.plot(x + width, density_grid, color=TEAL, lw=0.65)
        jitter = rng.uniform(-0.055, 0.015, len(y))
        ax_dist.scatter(np.full_like(y, x) + jitter, y, s=2.6, c=TEAL_DARK, alpha=0.42, linewidth=0)
        med = float(np.median(y))
        q1, q3 = np.quantile(y, [0.25, 0.75])
        ax_dist.plot([x - 0.055, x + 0.27], [med, med], color=INK, lw=0.75, zorder=4)
        ax_dist.plot([x - 0.02, x - 0.02], [q1, q3], color=INK, lw=1.05, zorder=4)
    ax_dist.set_xlim(-0.3, 4.55)
    ax_dist.set_ylim(0, 1.02)
    ax_dist.set_xticks(range(5))
    ax_dist.set_xticklabels([label for _, label in METRICS], fontweight="bold")
    ax_dist.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=2)
    ax_dist.set_yticks([0, 0.5, 1])
    ax_dist.set_yticklabels(["0", ".5", "1"])
    # Values are explicitly ticked on the shared 0--1 utility scale; omitting a
    # long y label protects the two annotation-strip headers at print size.
    ax_dist.set_ylabel("")
    ax_dist.spines[["top", "right", "bottom"]].set_visible(False)
    ax_dist.spines["left"].set_color(GRID)
    # Group headings are deliberately separated from the metric labels.  They
    # live in the reserved top margin, not on the density axes.
    ax_dist.text(0.205, 1.82, "Response ordering", transform=ax_dist.transAxes,
                 ha="center", va="bottom", fontsize=5.25, fontweight="bold", clip_on=False)
    ax_dist.text(0.765, 1.82, "Biological programmes", transform=ax_dist.transAxes,
                 ha="center", va="bottom", fontsize=5.25, fontweight="bold", clip_on=False)

    colored_strip(ax_bio, [BIOLOGY_COLORS[x] for x in data["biological_category"]])
    colored_strip(ax_tier, [TIER_COLORS[x] for x in data["substitutability_tier"]])
    # Put the short strip identifiers below the vertical strips.  This removes
    # their competition with the density axis's 0 tick above.
    ax_bio.text(0.5, -0.035, "Bio.", transform=ax_bio.transAxes,
                ha="center", va="top", fontsize=5.0, fontweight="bold",
                color=INK, clip_on=False, rotation=90)
    ax_tier.text(0.5, -0.035, "Tier", transform=ax_tier.transAxes,
                 ha="center", va="top", fontsize=5.0, fontweight="bold",
                 color=INK, clip_on=False, rotation=90)

    image = ax_heat.imshow(values, vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="nearest", origin="upper")
    ax_heat.set_xlim(-0.5, 4.5)
    ax_heat.set_ylim(51.5, -0.5)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.axvline(1.5, color=WHITE, lw=2.0)
    ax_heat.axvline(1.5, color=GRID, lw=0.55)

    y = np.arange(len(data))
    util = data["scientific_utility_index"].to_numpy(float)
    ax_rank.hlines(y, 0, util, color=GRID, lw=0.45, zorder=1)
    point_colors = [TIER_COLORS[t] for t in data["substitutability_tier"]]
    ax_rank.scatter(util, y, s=8, c=point_colors, edgecolor=WHITE, linewidth=0.25, alpha=0.82, zorder=3)
    selected = data["selected_label"].to_numpy(bool)
    ax_rank.scatter(util[selected], y[selected], s=17, c=np.asarray(point_colors)[selected], edgecolor=INK, linewidth=0.45, zorder=4)
    selected_rows = data.loc[selected].sort_index()
    # A separate label rail retains every selected reporter and its leader.
    label_ys = np.linspace(22, 47, len(selected_rows))
    for label_y, (_, row) in zip(label_ys, selected_rows.iterrows()):
        ax_rank.annotate(
            row["short_name"],
            xy=(row["scientific_utility_index"], row.name),
            xytext=(1.06, label_y),
            textcoords="data",
            ha="left",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=INK,
            arrowprops={"arrowstyle": "-", "lw": 0.45, "color": TIER_COLORS[row["substitutability_tier"]]},
            clip_on=False,
        )
    ax_rank.set_xlim(0, 2.25)
    ax_rank.set_ylim(51.5, -0.5)
    ax_rank.set_xlabel("Scientific utility index", labelpad=2)
    ax_rank.set_xticks([0, 0.5, 1])
    ax_rank.set_xticklabels(["0", "0.5", "1"])
    ax_rank.get_xticklabels()[0].set_ha("left")
    ax_rank.tick_params(axis="y", left=False, labelleft=False)
    ax_rank.spines[["top", "right", "left"]].set_visible(False)
    ax_rank.spines["bottom"].set_color(INK)

    cbar = container.colorbar(image, cax=ax_cbar, orientation="horizontal")
    cbar.set_ticks([0, 0.5, 1])
    cbar.ax.tick_params(labelsize=5.0, length=1.5, pad=1)
    cbar.outline.set_linewidth(0.45)
    cbar.set_label("Utility retained after response replacement", fontsize=5.1, labelpad=1)
    # Keep the complete label inside the short composite row at its original size.
    cbar_pos = ax_cbar.get_position()
    cbar_dy = (3.5 / 72) / (container.bbox.height / container.figure.dpi)
    ax_cbar.set_position([cbar_pos.x0, cbar_pos.y0 + cbar_dy,
                          cbar_pos.width, cbar_pos.height])
    cbar.ax.get_xticklabels()[-1].set_ha("right")

    handles = [
        Line2D([0], [0], marker="s", linestyle="", markersize=4, markerfacecolor=TIER_COLORS[t], markeredgewidth=0, label=TIER_LABELS[t])
        for t in TIER_ORDER
    ]
    ax_key.axis("off")
    ax_key.legend(handles=handles, loc="center", bbox_to_anchor=(0.50, 0.48),
                  ncol=2, columnspacing=0.72, handletextpad=0.28,
                  borderaxespad=0, frameon=False, fontsize=5.0)
    if add_letter:
        container.text(0.012, 0.972, "a", ha="left", va="top",
                       fontsize=8.5, fontweight="bold", color=INK)

    return {
        "density": ax_dist,
        "heatmap": ax_heat,
        "rank": ax_rank,
        "colorbar": ax_cbar,
        "key": ax_key,
    }


def build() -> None:
    configure()
    data = load_source()
    fig = plt.figure(figsize=(112 * MM, 65 * MM), constrained_layout=False)
    draw_panel_a(fig, add_letter=True, data=data)

    stem = HERE / "figure6a_counterfactual_utility_atlas"
    format_figure(fig, 6)
    qa_root = next(p for p in Path(__file__).resolve().parents if p.name == "figure6") / "qa"
    layout_report(fig, qa_root / (stem.name + "_typography.json"))
    for ext, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(stem.with_suffix(f".{ext}"), facecolor=WHITE, bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    build()
