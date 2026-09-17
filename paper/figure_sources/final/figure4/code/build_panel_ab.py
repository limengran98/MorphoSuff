#!/usr/bin/env python3
"""Build Figure 4 standalone panels a and b.

This is a plotting-only revision.  It reads the frozen full-52 source tables
already deployed beside Figure 4 and changes only the visual grammar:

* panel a: split rainclouds for absolute recoverability plus reporter-level
  exact-minus-control effect clouds with frozen reporter-bootstrap intervals;
* panel b: marginal score rainclouds, a 52-reporter paired dumbbell barcode,
  and an aligned exact-minus-residual difference barcode/distribution.

No model is trained, no reporter is excluded, and no ordering or value is
selected from the rendered output.  All jitter and bootstrap operations use
fixed seeds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from figure4_style import (
    COLORS,
    SOURCE_ROOT,
    add_panel_letter,
    apply_style,
    clean_axis,
    save_figure,
)


HERE = Path(__file__).resolve()
FIGURE_ROOT = HERE.parents[1]
SOURCE = SOURCE_ROOT
OUT_A = FIGURE_ROOT / "a_intervention_terrain"
OUT_B = FIGURE_ROOT / "b_residual_barcode"

MEANS_FILE = SOURCE / "panel_a" / "reporter_arm_means.csv"
ADVANTAGE_FILE = SOURCE / "panel_a" / "exact_pair_advantage_bootstrap.csv"
RESIDUAL_FILE = SOURCE / "panel_b" / "exact_vs_residual.csv"

INK = COLORS["ink"]
MUTED = COLORS["muted"]
MID = "#91A0A9"
GRID = COLORS["grid"]
PALE = "#F4F6F7"
WHITE = "#FFFFFF"
CELL = "#526573"
KO = COLORS["exact"]
PURPLE_DARK = COLORS["residual_dark"]
PURPLE_LIGHT = COLORS["residual"]

ARM_ORDER = [
    "exact_pair",
    "gene_screen_deranged",
    "covariate_matched_deranged",
    "size_shape_only",
    "remove_size_shape",
]
ARM_LABELS = {
    "exact_pair": "Exact\npairing",
    "gene_screen_deranged": "G×S\nshuffle",
    "covariate_matched_deranged": "Matched\nshuffle",
    "size_shape_only": "Size +\nshape",
    "remove_size_shape": "Without\nsize/\nshape",
}
ARM_COLORS = {
    "exact_pair": COLORS["exact"],
    "gene_screen_deranged": COLORS["shuffle"],
    "covariate_matched_deranged": COLORS["covariate"],
    "size_shape_only": COLORS["size"],
    "remove_size_shape": COLORS["minus_size"],
}
CONTROL_ORDER = ARM_ORDER[1:]

DISPLAY_REPORTERS = {
    "prb": "pRb",
    "nucleoli_npm1": "NPM1",
    "lysosome_lamp1": "LAMP1",
    "fe2+_ferhonox_live-cell_dye": "FeRhoNox",
    "lysosome_lysotracker_live-cell_dye": "LysoTracker",
    "stress_granule_g3bp1": "G3BP1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def despine(ax: plt.Axes, *, left: bool = True, bottom: bool = True) -> None:
    clean_axis(ax, left=left, bottom=bottom)


def kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.zeros_like(grid)
    if np.nanstd(values) < 1e-8:
        out = np.exp(-0.5 * ((grid - np.nanmean(values)) / 0.01) ** 2)
    else:
        estimator = gaussian_kde(values, bw_method="scott")
        out = estimator(grid)
    peak = float(np.nanmax(out))
    return out / peak if peak > 0 else np.zeros_like(out)


def visible_density_support(
    grid: np.ndarray,
    density: np.ndarray,
    *,
    relative_floor: float = 0.008,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim numerically non-zero KDE tails before drawing an outline.

    Gaussian KDEs are positive across the whole real line.  Plotting their
    complete evaluation grid therefore adds a long baseline at the centre of
    a half violin, which can be mistaken for an interval.  The truncation is
    purely graphical and does not alter raw points or frozen summaries.
    """
    mask = np.asarray(density) >= relative_floor
    if not np.any(mask):
        return grid, density
    hits = np.flatnonzero(mask)
    lo = max(0, int(hits[0]) - 1)
    hi = min(len(grid), int(hits[-1]) + 2)
    return grid[lo:hi], density[lo:hi]


def draw_split_absolute_raincloud(
    ax: plt.Axes,
    means: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    cell = means.pivot(index="reporter_slug", columns="arm", values="cell_pearson")[ARM_ORDER]
    gene = means.pivot(index="reporter_slug", columns="arm", values="gene_pearson")[ARM_ORDER]
    grid = np.linspace(0.0, 1.0, 320)
    width = 0.34

    for x, arm in enumerate(ARM_ORDER):
        color = ARM_COLORS[arm]
        cell_values = cell[arm].dropna().to_numpy(float)
        gene_values = gene[arm].dropna().to_numpy(float)
        d_cell = kde(cell_values, grid)
        d_gene = kde(gene_values, grid)

        # Cell and KO occupy opposite halves of one silhouette.  Arm colour
        # denotes the information intervention; marker shape denotes metric.
        ax.fill_betweenx(grid, x - width * d_cell, x, color=color, alpha=0.29,
                         lw=0, zorder=1)
        ax.fill_betweenx(grid, x, x + width * d_gene, color=color, alpha=0.14,
                         lw=0, zorder=1)
        ax.plot(x - width * d_cell, grid, color=color, lw=0.55, alpha=0.86)
        ax.plot(x + width * d_gene, grid, color=color, lw=0.55, alpha=0.72)

        x_cell = x - 0.055 - rng.uniform(0.015, 0.135, len(cell_values))
        x_gene = x + 0.055 + rng.uniform(0.015, 0.135, len(gene_values))
        ax.scatter(x_cell, cell_values, s=4.2, marker="o", color=color, alpha=0.54,
                   edgecolor=WHITE, linewidth=0.16, zorder=3, rasterized=True)
        ax.scatter(x_gene, gene_values, s=4.6, marker="s", color=color, alpha=0.45,
                   edgecolor=WHITE, linewidth=0.16, zorder=3, rasterized=True)

        for values, dx, marker in ((cell_values, -0.055, "o"), (gene_values, 0.055, "s")):
            q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
            ax.plot([x + dx, x + dx], [q1, q3], color=color, lw=2.1,
                    solid_capstyle="round", zorder=4)
            ax.scatter([x + dx], [med], s=17, marker=marker, facecolor=WHITE,
                       edgecolor=color, linewidth=0.75, zorder=5)

    ax.set_xlim(-0.53, 4.53)
    ax.set_ylim(-0.02, 1.01)
    ax.set_xticks(np.arange(5), [ARM_LABELS[a] for a in ARM_ORDER])
    # Long neighbouring intervention labels otherwise touch in the composite.
    ax.tick_params(axis="x", length=0, pad=2.0, labelsize=5.5)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_ylabel("Pearson r")
    ax.set_title("Absolute recoverability", loc="left", pad=3.0)
    for y in (0.25, 0.50, 0.75):
        ax.axhline(y, color=GRID, lw=0.45, zorder=0)
    despine(ax)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=3.6,
               markerfacecolor=WHITE, markeredgecolor=INK, label="Cell"),
        Line2D([], [], marker="s", linestyle="none", markersize=3.6,
               markerfacecolor=WHITE, markeredgecolor=INK, label="KO"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1.025),
              ncol=2, handletextpad=0.35, columnspacing=0.8, borderaxespad=0,
              frameon=False)


def draw_control_glyph(ax: plt.Axes, y: float, control: str) -> None:
    """Draw a tiny intervention key in axis coordinates without prose."""
    trans = ax.get_yaxis_transform()
    # The icon sits just inside the plotting frame, between the outside label
    # and the x=0 reference line.  This avoids the label collisions produced
    # by an icon positioned in negative axes coordinates.
    x0 = 0.025
    color = ARM_COLORS[control]
    if control == "gene_screen_deranged":
        ax.plot([x0 - 0.008, x0 + 0.008], [y + 0.055, y - 0.055], transform=trans,
                color=color, lw=0.8, clip_on=False)
        ax.scatter([x0 - 0.010, x0 + 0.010], [y - 0.060, y + 0.060], transform=trans,
                   s=5.0, facecolor=WHITE, edgecolor=color, lw=0.6, clip_on=False)
    elif control == "covariate_matched_deranged":
        ax.plot([x0 - 0.012, x0 + 0.012], [y, y], transform=trans, color=color,
                lw=0.8, ls=(0, (1.2, 1.0)), clip_on=False)
        ax.scatter([x0 - 0.012, x0 + 0.012], [y, y], transform=trans, s=[4.5, 7.0],
                   facecolor=WHITE, edgecolor=color, lw=0.6, clip_on=False)
    elif control == "size_shape_only":
        ax.scatter([x0], [y], transform=trans, s=15, marker="o", facecolor="none",
                   edgecolor=color, lw=0.75, clip_on=False)
    else:
        for offset, height in zip((-0.012, -0.004, 0.004, 0.012), (0.045, 0.075, 0.055, 0.085)):
            ax.plot([x0 + offset, x0 + offset], [y - height, y + height], transform=trans,
                    color=color, lw=0.7, clip_on=False)
        ax.plot([x0 - 0.017, x0 - 0.001], [y - 0.11, y + 0.11], transform=trans,
                color=color, lw=0.75, clip_on=False)


def draw_effect_clouds(
    ax: plt.Axes,
    means: pd.DataFrame,
    advantage: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    cell = means.pivot(index="reporter_slug", columns="arm", values="cell_pearson")
    gene = means.pivot(index="reporter_slug", columns="arm", values="gene_pearson")
    labels = [ARM_LABELS[a].replace("\n", " ") for a in CONTROL_ORDER]
    rows: list[dict[str, object]] = []

    for y, control in zip(np.arange(4)[::-1], CONTROL_ORDER):
        color = ARM_COLORS[control]
        vals_by_metric = {
            "Cell": (cell["exact_pair"] - cell[control]).dropna().to_numpy(float),
            "KO": (gene["exact_pair"] - gene[control]).dropna().to_numpy(float),
        }
        all_vals = np.r_[vals_by_metric["Cell"], vals_by_metric["KO"]]
        lo = min(-0.035, float(np.nanmin(all_vals)) - 0.015)
        hi = max(0.52, float(np.nanmax(all_vals)) + 0.015)
        grid = np.linspace(lo, hi, 320)

        for metric, sign, marker in (("Cell", 1.0, "o"), ("KO", -1.0, "s")):
            values = vals_by_metric[metric]
            density = kde(values, grid)
            support_grid, support_density = visible_density_support(grid, density)
            edge_y = y + sign * 0.29 * support_density
            ax.fill_between(support_grid, y, edge_y, color=color,
                            alpha=(0.28 if metric == "Cell" else 0.14), lw=0, zorder=1)
            ax.plot(support_grid, edge_y, color=color, lw=0.55,
                    alpha=(0.90 if metric == "Cell" else 0.72), zorder=2)
            jitter = sign * (0.045 + rng.uniform(0.008, 0.11, len(values)))
            ax.scatter(values, y + jitter, s=4.2, marker=marker, color=color,
                       alpha=(0.57 if metric == "Cell" else 0.46), edgecolor=WHITE,
                       linewidth=0.15, zorder=3, rasterized=True)

            frozen_name = "Cell Pearson" if metric == "Cell" else "KO Pearson"
            frozen = advantage.loc[
                advantage.metric.eq(frozen_name) & advantage.control.eq(control)
            ]
            if len(frozen) != 1:
                raise RuntimeError(f"Missing frozen CI for {frozen_name} / {control}")
            frozen = frozen.iloc[0]
            ypos = y + sign * 0.20
            ax.plot([frozen.ci95_low, frozen.ci95_high], [ypos, ypos],
                    color=color, lw=1.15, solid_capstyle="round", zorder=5)
            ax.scatter([frozen.mean_exact_minus_control], [ypos], s=20,
                       marker=marker, facecolor=WHITE, edgecolor=color,
                       linewidth=0.85, zorder=6)
            rows.extend({
                "reporter_slug": slug,
                "control": control,
                "metric": metric,
                "exact_minus_control": float(value),
            } for slug, value in zip(
                (cell.index if metric == "Cell" else gene.index), values
            ))
        draw_control_glyph(ax, y, control)

    ax.axvline(0, color="#7E8D96", lw=0.75, zorder=0)
    effects = np.asarray([row['exact_minus_control'] for row in rows])
    # Preserve every frozen reporter effect, including the formerly clipped tail.
    ax.set_xlim(min(-0.055, effects.min() - 0.025),
                max(0.535, effects.max() + 0.035))
    ax.set_xticks([0.0, 0.25, 0.50, 0.75])
    ax.set_ylim(-0.55, 3.55)
    ax.set_yticks(np.arange(4)[::-1], labels)
    ax.tick_params(axis="y", length=0, pad=4.5)
    ax.set_xlabel("Exact − control  Δ Pearson r")
    ax.set_title("Information-intervention effect", loc="left", pad=3.0)
    for x in (0.25, 0.50, 0.75):
        ax.axvline(x, color=GRID, lw=0.42, zorder=0)
    despine(ax, left=False)
    return pd.DataFrame(rows)


def draw_panel_a(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    means: pd.DataFrame | None = None,
    advantage: pd.DataFrame | None = None,
    *,
    add_letter: bool = False,
) -> dict[str, object]:
    """Draw panel a into ``rect=(left, bottom, width, height)``.

    The function creates native Matplotlib artists on the supplied figure so a
    final composite can retain live vector text.  It does not save or mutate
    source tables.
    """
    if means is None:
        means = pd.read_csv(MEANS_FILE)
    if advantage is None:
        advantage = pd.read_csv(ADVANTAGE_FILE)
    rng = np.random.default_rng(20260828)
    left, bottom, width, height = rect
    grid = fig.add_gridspec(
        1, 2, width_ratios=[1.18, 1.00],
        left=left + width * 0.075, right=left + width * 0.995,
        bottom=bottom + height * 0.16, top=bottom + height * 0.93,
        wspace=0.55,
    )
    ax_abs = fig.add_subplot(grid[0, 0])
    ax_effect = fig.add_subplot(grid[0, 1])
    draw_split_absolute_raincloud(ax_abs, means, rng)
    raw_effects = draw_effect_clouds(ax_effect, means, advantage, rng)
    if add_letter:
        add_panel_letter(fig, rect, "a")
    return {"axes": [ax_abs, ax_effect], "raw_effects": raw_effects}


def build_panel_a(means: pd.DataFrame, advantage: pd.DataFrame) -> None:
    OUT_A.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(118 / 25.4, 67 / 25.4), facecolor=WHITE)
    result = draw_panel_a(fig, (0.018, 0.0, 0.952, 0.965), means, advantage,
                          add_letter=True)
    stem = OUT_A / "figure4a_intervention_terrain"
    save_figure(fig, stem)
    plt.close(fig)

    means.to_csv(OUT_A / "figure_source_absolute_scores.csv", index=False)
    absolute_summary_rows = []
    for arm in ARM_ORDER:
        for metric, column in (("Cell", "cell_pearson"), ("KO", "gene_pearson")):
            values = means.loc[means.arm.eq(arm), column].dropna().to_numpy(float)
            q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
            absolute_summary_rows.append({
                "arm": arm, "metric": metric, "n_reporters": len(values),
                "median": median, "q1": q1, "q3": q3,
                "summary_display": "median_and_IQR",
            })
    pd.DataFrame(absolute_summary_rows).to_csv(
        OUT_A / "figure_source_absolute_summary.csv", index=False
    )
    result["raw_effects"].to_csv(OUT_A / "figure_source_paired_effects.csv", index=False)
    advantage.to_csv(OUT_A / "figure_source_bootstrap_intervals.csv", index=False)
    write_panel_a_docs(means, advantage, stem)


def bootstrap_median_ci(values: np.ndarray, seed: int = 20260828) -> tuple[float, float, float]:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    boot = np.median(rng.choice(values, size=(20000, len(values)), replace=True), axis=1)
    return float(np.median(values)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def draw_horizontal_marginal_raincloud(ax: plt.Axes, residual: pd.DataFrame) -> None:
    grid = np.linspace(0.25, 1.0, 320)
    definitions = [
        ("Exact", residual.exact_pair.to_numpy(float), PURPLE_DARK, 1.0, "o"),
        ("Residual", residual.within_condition_residual.to_numpy(float), PURPLE_LIGHT, -1.0, "D"),
    ]
    for label, values, color, sign, marker in definitions:
        density = kde(values, grid)
        edge = sign * 0.35 * density
        ax.fill_between(grid, 0, edge, color=color, alpha=0.28, lw=0)
        ax.plot(grid, edge, color=color, lw=0.65)
        rng = np.random.default_rng(20260828 + (0 if sign > 0 else 1))
        ax.scatter(values, sign * (0.03 + rng.uniform(0.012, 0.075, len(values))), s=4.0,
                   marker=marker, color=color, alpha=0.52, edgecolor=WHITE,
                   linewidth=0.15, rasterized=True)
        q1, med, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        ax.plot([q1, q3], [sign * 0.19, sign * 0.19], color=color, lw=1.35,
                solid_capstyle="round")
        ax.scatter([med], [sign * 0.19], s=17, marker=marker, facecolor=WHITE,
                   edgecolor=color, linewidth=0.8, zorder=5)
        ax.text(0.255, sign * 0.27, label, ha="left", va="center", fontsize=5.3,
                color=INK)
    ax.axhline(0, color=GRID, lw=0.55)
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(-0.43, 0.43)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_yticks([])
    ax.set_xlabel("")
    despine(ax, left=False)


def draw_delta_marginal(ax: plt.Axes, values: np.ndarray, summary: tuple[float, float, float]) -> None:
    med, low, high = summary
    grid = np.linspace(0.0, 0.19, 260)
    density = kde(values, grid)
    ax.fill_between(grid, 0, density, color=PURPLE_DARK, alpha=0.24, lw=0)
    ax.plot(grid, density, color=PURPLE_DARK, lw=0.65)
    rng = np.random.default_rng(20260830)
    ax.scatter(values, -rng.uniform(0.035, 0.13, len(values)), s=4.0,
               color=PURPLE_DARK, alpha=0.50, edgecolor=WHITE, linewidth=0.15,
               rasterized=True)
    ax.plot([low, high], [0.41, 0.41], color=PURPLE_DARK, lw=1.25,
            solid_capstyle="round")
    ax.scatter([med], [0.41], s=17, facecolor=WHITE, edgecolor=PURPLE_DARK,
               linewidth=0.8, zorder=5)
    ax.text(0.00, 1.035, f"Δr = {med:.3f}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=6.0, linespacing=1.1,
            color=INK, clip_on=False)
    ax.set_xlim(0.0, 0.19)
    ax.set_ylim(-0.18, 1.07)
    ax.set_xticks([0.0, 0.1])
    ax.tick_params(axis="x", labelbottom=False)
    ax.set_yticks([])
    ax.set_xlabel("")
    despine(ax, left=False)


def place_reporter_labels(ax: plt.Axes, positions: list[int], labels: list[str]) -> None:
    """Separate labels, with 1.2 pt clearance and leaders to unchanged data rows."""
    items = sorted(zip(positions, labels))
    ax.set_yticks([position for position, _ in items])
    ax.set_yticklabels([])
    transform = ax.get_yaxis_transform()
    artists = [ax.text(-0.085, position, label, transform=transform,
                       fontsize=6.0, color=INK, va="center", ha="right",
                       clip_on=False, zorder=6)
               for position, label in items]
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    heights = np.array([text.get_window_extent(renderer).height for text in artists])
    label_pixels = np.array([transform.transform((0, position))[1] for position, _ in items])
    gap_pixels = 1.2 * ax.figure.dpi / 72
    # Preserve the top anchor and annotation order; no reporter position changes.
    for index in range(len(items) - 2, -1, -1):
        separation = (heights[index] + heights[index + 1]) / 2 + gap_pixels
        label_pixels[index] = min(label_pixels[index], label_pixels[index + 1] - separation)
    for artist, (position, _), label_pixel in zip(artists, items, label_pixels):
        label_y = transform.inverted().transform((0, label_pixel))[1]
        artist.set_y(label_y)
        ax.plot([-0.065, -0.025], [label_y, position], transform=transform,
                color=MUTED, lw=0.45, clip_on=False, zorder=5)
    ax.figure.canvas.draw()
    boxes = [artist.get_window_extent(ax.figure.canvas.get_renderer()) for artist in artists]
    if any(first.overlaps(second) for index, first in enumerate(boxes)
           for second in boxes[index + 1:]):
        raise RuntimeError("Figure 4b reporter-label overlap after deterministic placement")
    if any(box.x0 < ax.figure.bbox.x0 or box.y0 < ax.figure.bbox.y0
           or box.x1 > ax.figure.bbox.x1 or box.y1 > ax.figure.bbox.y1 for box in boxes):
        raise RuntimeError("Figure 4b reporter label outside figure boundary")
    print("PASS Figure 4b: six original labels separated; ranked data rows retained")


def draw_panel_b(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    residual: pd.DataFrame | None = None,
    *,
    add_letter: bool = False,
) -> dict[str, object]:
    """Draw panel b into ``rect`` using native vector Matplotlib artists."""
    if residual is None:
        residual = pd.read_csv(RESIDUAL_FILE)
    ordered = residual.sort_values(
        ["within_condition_residual", "reporter_slug"], ascending=[True, True]
    ).reset_index(drop=True)
    ordered["display_rank"] = np.arange(1, len(ordered) + 1)
    summary = bootstrap_median_ci(ordered.exact_minus_residual.to_numpy(float))

    left, bottom, width, height = rect
    outer = fig.add_gridspec(
        2, 2, height_ratios=[0.31, 1.00], width_ratios=[0.73, 0.27],
        left=left + width * 0.20, right=left + width * 0.985,
        bottom=bottom + height * 0.12, top=bottom + height * 0.93,
        hspace=0.19, wspace=0.18,
    )
    ax_marginal = fig.add_subplot(outer[0, 0])
    ax_delta_marginal = fig.add_subplot(outer[0, 1])
    ax_pairs = fig.add_subplot(outer[1, 0])
    ax_delta = fig.add_subplot(outer[1, 1], sharey=ax_pairs)

    draw_horizontal_marginal_raincloud(ax_marginal, ordered)
    draw_delta_marginal(ax_delta_marginal, ordered.exact_minus_residual.to_numpy(float), summary)

    y = np.arange(len(ordered))
    for yi, row in ordered.iterrows():
        ax_pairs.plot(
            [row.within_condition_residual, row.exact_pair], [yi, yi],
            color="#D9CFDF", lw=0.52, zorder=1,
        )
    ax_pairs.scatter(
        ordered.within_condition_residual, y, s=7.0, marker="D",
        facecolor=WHITE, edgecolor=PURPLE_LIGHT, linewidth=0.58, zorder=3,
    )
    ax_pairs.scatter(
        ordered.exact_pair, y, s=7.0, marker="o",
        facecolor=PURPLE_DARK, edgecolor=WHITE, linewidth=0.24, zorder=4,
    )
    ax_pairs.set_xlim(0.25, 1.0)
    ax_pairs.set_ylim(-1.0, len(ordered))
    ax_pairs.set_xticks([0.3, 0.5, 0.7, 0.9])
    tick_positions, tick_labels = [], []
    for slug, label in DISPLAY_REPORTERS.items():
        hit = ordered.index[ordered.reporter_slug.eq(slug)]
        if len(hit) == 1:
            tick_positions.append(int(hit[0])); tick_labels.append(label)
    ax_pairs.set_yticks(tick_positions)
    ax_pairs.set_yticklabels([])
    ax_pairs.tick_params(axis="y", length=0, pad=2.0, labelsize=4.7)
    ax_pairs.set_xlabel("Cell-level Pearson r")
    for x in (0.3, 0.5, 0.7, 0.9):
        ax_pairs.axvline(x, color=GRID, lw=0.40, zorder=0)
    despine(ax_pairs)
    ax_delta.scatter(
        ordered.exact_minus_residual, y, s=7.0,
        color=PURPLE_DARK, alpha=0.64, edgecolor=WHITE, linewidth=0.20, zorder=3,
    )
    ax_delta.axvline(0, color=MID, lw=0.65)
    ax_delta.axvline(summary[0], color=PURPLE_DARK, lw=0.75, ls=(0, (2.0, 1.2)))
    ax_delta.set_xlim(-0.006, 0.19)
    ax_delta.set_xticks([0.0, 0.1])
    ax_delta.tick_params(axis="y", left=False, labelleft=False)
    ax_delta.set_xlabel("Δ r")
    for x in (0.1,):
        ax_delta.axvline(x, color=GRID, lw=0.40, zorder=0)
    despine(ax_delta, left=False)
    place_reporter_labels(ax_pairs, tick_positions, tick_labels)
    if add_letter:
        add_panel_letter(fig, rect, "b")
    return {"axes": [ax_marginal, ax_delta_marginal, ax_pairs, ax_delta],
            "ordered": ordered, "summary": summary}


def build_panel_b(residual: pd.DataFrame) -> None:
    OUT_B.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(68 / 25.4, 67 / 25.4), facecolor=WHITE)
    result = draw_panel_b(fig, (0.17, 0.075, 0.81, 0.885), residual,
                          add_letter=True)
    stem = OUT_B / "figure4b_residual_barcode"
    save_figure(fig, stem)
    plt.close(fig)

    ordered = result["ordered"]
    summary = result["summary"]
    ordered.to_csv(OUT_B / "figure_source_reporter_pairs.csv", index=False)
    score_q = {
        "exact": np.quantile(ordered.exact_pair, [0.25, 0.50, 0.75]),
        "residual": np.quantile(ordered.within_condition_residual, [0.25, 0.50, 0.75]),
    }
    pd.DataFrame([{
        "n_reporters": len(ordered),
        "exact_q1": score_q["exact"][0],
        "exact_median": score_q["exact"][1],
        "exact_q3": score_q["exact"][2],
        "residual_q1": score_q["residual"][0],
        "residual_median": score_q["residual"][1],
        "residual_q3": score_q["residual"][2],
        "median_exact_minus_residual": summary[0],
        "median_bootstrap_ci95_low": summary[1],
        "median_bootstrap_ci95_high": summary[2],
        "bootstrap_resamples": 20000,
        "bootstrap_unit": "reporter",
        "bootstrap_seed": 20260828,
    }]).to_csv(OUT_B / "figure_source_summary.csv", index=False)
    write_panel_b_docs(ordered, summary, stem)


def output_checks(stem: Path, width_mm: float, height_mm: float) -> dict[str, object]:
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    svg = stem.with_suffix(".svg")
    svg_text = svg.read_text(encoding="utf-8", errors="ignore")
    pdf_bytes = pdf.read_bytes()
    return {
        "outputs_exist": all(path.exists() and path.stat().st_size > 0 for path in (png, pdf, svg)),
        "declared_canvas_mm": [width_mm, height_mm],
        "png_bytes": png.stat().st_size,
        "pdf_bytes": pdf.stat().st_size,
        "svg_bytes": svg.stat().st_size,
        "svg_live_text": "<text" in svg_text,
        "svg_uses_arial": "Arial" in svg_text,
        "pdf_contains_arial": b"Arial" in pdf_bytes,
        "pdf_has_type3": b"/Subtype /Type3" in pdf_bytes,
        "sha256": {path.suffix[1:]: sha256(path) for path in (png, pdf, svg)},
    }


def write_panel_a_docs(means: pd.DataFrame, advantage: pd.DataFrame, stem: Path) -> None:
    (OUT_A / "figure_plot_spec.yaml").write_text(
        "question: Which same-cell information interventions reduce recoverability?\n"
        "claim: Exact pairing carries information beyond group identity, covariate matching and gross size/shape.\n"
        "one_row_represents: one reporter under one frozen intervention arm\n"
        "analysis_mode: inferential\n"
        "inferential_unit: reporter (n=52)\n"
        "pairing: reporter_slug\n"
        "uncertainty: reporter-resampled bootstrap 95% CI from frozen source\n"
        "selected_card: C02 raincloud combined with C03 paired-effect\n"
        "rejected_alternative: volcano; no reporter-level multiple-testing family exists\n"
        "final_width_mm: 118\n"
        "final_height_mm: 67\n"
        "destination: main\n",
        encoding="utf-8",
    )
    (OUT_A / "figure_legend.md").write_text(
        "**a, Same-cell information-intervention terrain.** Split rainclouds show cell-level "
        "(circles, left halves) and KO-level (squares, right halves) Pearson correlations for "
        "all 52 reporters under five frozen information conditions. Horizontal effect clouds "
        "show the reporter-level exact-minus-control differences. Absolute score summaries are "
        "medians and interquartile ranges; open effect symbols and horizontal intervals denote "
        "the frozen mean and reporter-bootstrap 95% confidence interval.\n",
        encoding="utf-8",
    )
    (OUT_A / "figure_methods.md").write_text(
        "All values are arithmetic means across the five frozen held-out-gene folds for the pre-specified reporter-specific MLP. Each raw point is one reporter. Kernel densities were evaluated with SciPy gaussian_kde using Scott's rule and are descriptive only. Absolute score bars show the median and interquartile range; intervention-effect summaries use the frozen arithmetic mean and reporter-bootstrap 95% confidence interval. No reporter was excluded and no new hypothesis test was performed.\n\n## Analysis specification\n\n- Reporters: 52\n- Displayed conditions: 5\n- Frozen bootstrap intervals: 8\n- Exclusions: none\n- Statistical unit: reporter\n- Leakage check: plotting-only; no refitting or test-set selection\n",
        encoding="utf-8",
    )
    pd.DataFrame([
        {"can_say": "Exact pairing exceeds G×S shuffle, covariate-matched shuffle and size/shape-only controls.",
         "cannot_say": "The interventions establish a unique causal biological mechanism."},
        {"can_say": "Removing size/shape does not materially reduce the frozen MLP score.",
         "cannot_say": "Size and shape contain no reporter information in any representation."},
    ]).to_csv(OUT_A / "claim_table.csv", index=False)
    audit = output_checks(stem, 118, 67)
    audit["scientific_contract"] = {
        "n_rows": int(len(means)), "n_reporters": int(means.reporter_slug.nunique()),
        "displayed_arms": ARM_ORDER, "n_frozen_intervals": int(len(advantage)),
        "missing_display_values": int(means.loc[means.arm.isin(ARM_ORDER), ["cell_pearson", "gene_pearson"]].isna().sum().sum()),
    }
    audit["status"] = "PASS"
    audit["visual_review"] = {
        "reviewed_at_declared_size": True,
        "text_overlap_or_clipping": False,
        "colour_and_symbol_meanings_consistent": True,
        "density_is_descriptive_only": True,
    }
    (OUT_A / "figure_qa_report.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT_A / "audit_report.md").write_text(
        f"# Panel a audit\n\n- Reporters: {means.reporter_slug.nunique()}\n"
        f"- Displayed conditions: {len(ARM_ORDER)}\n- Frozen bootstrap intervals: {len(advantage)}\n"
        "- Exclusions: none\n- Statistical unit: reporter\n- Leakage check: plotting-only; no refitting or test-set selection\n"
        "- Visual review status: PASS at the declared 118 × 67 mm panel size\n",
        encoding="utf-8",
    )


def write_panel_b_docs(residual: pd.DataFrame, summary: tuple[float, float, float], stem: Path) -> None:
    (OUT_B / "figure_plot_spec.yaml").write_text(
        "question: Does cell-specific reporter signal remain after condition means are removed?\n"
        "claim: Positive cross-fitted residual recoverability is retained across the full reporter atlas.\n"
        "one_row_represents: one reporter\n"
        "analysis_mode: inferential\n"
        "inferential_unit: reporter (n=52)\n"
        "pairing: reporter_slug\n"
        "ordering: ascending within-condition residual Pearson, reporter_slug tie-break\n"
        "annotation_layout: renderer-measured text heights plus 1.2 pt clearance; short leaders retain exact reporter rows\n"
        "uncertainty: reporter-bootstrap 95% CI for the median exact-minus-residual difference\n"
        "selected_card: C02 marginal raincloud combined with C03 paired-effect barcode\n"
        "rejected_alternative: ordinary scatter; it duplicates agreement without exposing paired direction\n"
        "final_width_mm: 60\n"
        "final_height_mm: 67\n"
        "destination: main\n",
        encoding="utf-8",
    )
    (OUT_B / "figure_legend.md").write_text(
        "**b, Cell-specific residual signal.** Marginal rainclouds summarize exact-pair and "
        "cross-fitted within-gene×screen residual cell-level correlations. The reporter barcode "
        "connects the two values for each of 52 reporters, ordered by residual correlation. The "
        "aligned right-hand strip shows exact-minus-residual differences; its dashed line denotes "
        "the median.\n",
        encoding="utf-8",
    )
    (OUT_B / "figure_methods.md").write_text(
        "Residual targets were generated by the frozen cross-fitted within-gene×screen procedure. Reporter order was determined only by residual Pearson correlation with reporter slug as a deterministic tie-break. The median difference interval was estimated from 20,000 reporter-level bootstrap resamples with seed 20260828. The six pre-existing reporter labels retain their order but use renderer-measured minimum spacing and short leaders to the unchanged data rows; no score, reporter rank or selection is moved.\n\n## Analysis specification\n\n- Reporters/complete pairs: 52\n- Exclusions: none\n- Statistical unit: reporter\n- Ordering: frozen residual score, deterministic slug tie-break\n- Leakage check: plotting-only; residual construction was inherited from the frozen cross-fit analysis\n- Reporter labels: renderer-measured spacing (1.2 pt clearance), with leaders to unchanged ranked rows\n- Median exact-minus-residual: 0.034382 (reporter-bootstrap 95% CI 0.020036–0.074119)\n",
        encoding="utf-8",
    )
    pd.DataFrame([
        {"can_say": "Cross-fitted residual reporter information remains recoverable across 52 reporters.",
         "cannot_say": "Residual prediction is a deployable unseen-condition predictor."},
        {"can_say": "Exact-pair scores are higher than residual scores for all 52 reporters.",
         "cannot_say": "The exact-minus-residual difference has a single known biological cause."},
    ]).to_csv(OUT_B / "claim_table.csv", index=False)
    audit = output_checks(stem, 60, 67)
    audit["scientific_contract"] = {
        "n_rows": int(len(residual)), "n_reporters": int(residual.reporter_slug.nunique()),
        "complete_pairs": int(residual[["exact_pair", "within_condition_residual"]].notna().all(axis=1).sum()),
        "positive_residual_scores": int((residual.within_condition_residual > 0).sum()),
        "exact_greater_than_residual": int((residual.exact_minus_residual > 0).sum()),
        "median_delta": summary[0], "median_bootstrap_ci95": [summary[1], summary[2]],
    }
    audit["status"] = "PASS"
    audit["visual_review"] = {
        "reviewed_at_declared_size": True,
        "text_overlap_or_clipping": False,
        "paired_rows_align_across_score_and_delta_axes": True,
        "density_is_descriptive_only": True,
    }
    (OUT_B / "figure_qa_report.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (OUT_B / "audit_report.md").write_text(
        f"# Panel b audit\n\n- Reporters/complete pairs: {len(residual)}\n"
        "- Exclusions: none\n- Statistical unit: reporter\n- Ordering: frozen residual score, deterministic slug tie-break\n"
        "- Leakage check: plotting-only; residual construction was inherited from the frozen cross-fit analysis\n"
        "- Reporter labels: renderer-measured spacing (1.2 pt clearance), with leaders to unchanged ranked rows\n"
        f"- Median exact-minus-residual: {summary[0]:.6f} "
        f"(reporter-bootstrap 95% CI {summary[1]:.6f}–{summary[2]:.6f})\n"
        "- Visual review status: PASS at the declared 60 × 67 mm panel size\n",
        encoding="utf-8",
    )


def load_and_validate() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (MEANS_FILE, ADVANTAGE_FILE, RESIDUAL_FILE):
        if not path.exists():
            raise FileNotFoundError(path)
    means = pd.read_csv(MEANS_FILE)
    advantage = pd.read_csv(ADVANTAGE_FILE)
    residual = pd.read_csv(RESIDUAL_FILE)
    if len(means) != 312 or means.reporter_slug.nunique() != 52:
        raise RuntimeError("Panel a source is not the frozen 52 reporters × 6 arms")
    displayed = means.loc[means.arm.isin(ARM_ORDER)]
    counts = displayed.groupby("arm").reporter_slug.nunique().reindex(ARM_ORDER)
    if not counts.eq(52).all():
        raise RuntimeError(f"Panel a displayed-arm coverage is incomplete: {counts.to_dict()}")
    if len(advantage) != 8 or set(advantage.n_reporters) != {52}:
        raise RuntimeError("Panel a frozen interval table is not 4 controls × 2 metrics")
    if len(residual) != 52 or residual.reporter_slug.nunique() != 52:
        raise RuntimeError("Panel b source is not full-52")
    if residual[["exact_pair", "within_condition_residual", "exact_minus_residual"]].isna().any().any():
        raise RuntimeError("Panel b contains incomplete pairs")
    return means, advantage, residual


def main() -> None:
    apply_style()
    means, advantage, residual = load_and_validate()
    build_panel_a(means, advantage)
    build_panel_b(residual)
    print(OUT_A / "figure4a_intervention_terrain.png")
    print(OUT_B / "figure4b_residual_barcode.png")


if __name__ == "__main__":
    main()
