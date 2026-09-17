#!/usr/bin/env python3
"""Render Figure 5b from frozen low-stability and microscopy assets.

The public ``draw_panel_b`` function accepts a Matplotlib Figure or SubFigure
for vector-preserving composition into the manuscript figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from figure5_style import COLORS, SOURCE_ROOT, apply_style, clean_axis, save_figure


HERE = Path(__file__).resolve().parent
PANEL_ROOT = HERE.parent / "b_low_stability_cases"
FIGURE_ROOT = PANEL_ROOT / "figure"
MICRO_ROOT = SOURCE_ROOT / "microscopy"

TARGETS = ["pRb", "NPM1"]
TARGET_COLORS = {"pRb": COLORS["constraint"], "NPM1": COLORS["violet"]}
TESTS = ["Well pairs", "Independent guide halves", "Random cell halves"]
TEST_LABELS = ["Well pairs", "Guide halves", "Random cell halves"]


def _read_case_values(target: str) -> dict[str, np.ndarray]:
    slug = "prb" if target == "pRb" else "npm1"
    random = pd.read_csv(SOURCE_ROOT / f"panel_b/{slug}_random_cell_halves.csv")["pearson_r"].to_numpy(float)
    guide = pd.read_csv(SOURCE_ROOT / f"panel_b/{slug}_guide_halves.csv")["pearson_r"].to_numpy(float)
    well = pd.read_csv(SOURCE_ROOT / f"panel_b/{slug}_well_pairs.csv")["gene_response_magnitude_pearson"].to_numpy(float)
    return {"Well pairs": well, "Independent guide halves": guide, "Random cell halves": random}


def prepare_panel_b() -> dict[str, Any]:
    summary = pd.read_csv(SOURCE_ROOT / "panel_b/low_stability_case_summary.csv")
    value_rows: list[dict[str, Any]] = []
    values: dict[str, dict[str, np.ndarray]] = {}
    for target in TARGETS:
        values[target] = _read_case_values(target)
        for test in TESTS:
            for replicate, value in enumerate(values[target][test]):
                value_rows.append(
                    {
                        "target": target,
                        "test": test,
                        "replicate_or_draw": replicate,
                        "pearson_r": float(value),
                        "statistical_role": "resampling draw" if test == "Random cell halves" else "observed partition",
                    }
                )
    value_table = pd.DataFrame(value_rows)

    selection = pd.read_csv(MICRO_ROOT / "selection_manifest.csv")
    selection = selection.loc[selection.target_key.eq("prb")].copy().reset_index(drop=True)
    display = pd.read_csv(MICRO_ROOT / "display_manifest.csv").set_index("target_key").loc["prb"]
    if list(selection.asset_key) != ["cell_00", "cell_01", "cell_02"]:
        raise AssertionError("Frozen pRb microscopy selection changed")
    return {"summary": summary, "values": values, "value_table": value_table, "selection": selection, "display": display}


def _half_raincloud(ax: plt.Axes, values: np.ndarray, y: float, color: str, seed: int) -> None:
    xmin, xmax = -0.06, 0.25
    grid = np.linspace(max(xmin, values.min() - 0.03), min(xmax, values.max() + 0.03), 180)
    density = gaussian_kde(values)(grid)
    density = density / density.max() * 0.26
    ax.fill_between(grid, y, y + density, color=color, alpha=0.23, linewidth=0)
    ax.plot(grid, y + density, color=color, lw=0.75, alpha=0.9)
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.19, -0.055, size=len(values))
    ax.scatter(values, y + jitter, s=4.5, color=color, alpha=0.28, edgecolor="none", rasterized=True)
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    ax.plot([q25, q75], [y + 0.03, y + 0.03], color=COLORS["ink"], lw=1.2, zorder=4)
    ax.scatter([median], [y + 0.03], marker="|", s=58, linewidths=1.4, color=COLORS["ink"], zorder=5)


def _small_n_summary(ax: plt.Axes, values: np.ndarray, y: float, color: str, marker: str) -> None:
    offsets = np.array([-0.095, 0.0, 0.095])[: len(values)]
    ax.scatter(values, y + offsets, s=20, marker=marker, color=color, edgecolor="white", lw=0.55, zorder=4)
    ax.plot([values.min(), values.max()], [y, y], color=COLORS["ink"], lw=0.85, zorder=2)
    # Use a transparent, slightly enlarged ring for the median so the observed
    # coloured point at the same value remains visible instead of being erased
    # by an opaque white marker.
    ax.scatter(
        [np.median(values)],
        [y],
        s=40,
        facecolor="none",
        edgecolor=COLORS["ink"],
        lw=0.8,
        zorder=5,
    )


def _draw_case_axis(ax: plt.Axes, target: str, values: dict[str, np.ndarray], seed: int) -> None:
    color = TARGET_COLORS[target]
    y_positions = np.array([2.0, 1.0, 0.0])
    _small_n_summary(ax, values[TESTS[0]], y_positions[0], color, "o")
    _small_n_summary(ax, values[TESTS[1]], y_positions[1], color, "D")
    _half_raincloud(ax, values[TESTS[2]], y_positions[2], color, seed)
    ax.axvline(0, color=COLORS["mid"], lw=0.65, zorder=0)
    ax.set_xlim(-0.06, 0.25)
    ax.set_ylim(-0.28, 2.32)
    ax.set_yticks(y_positions, [label.replace('Random cell halves', 'Cell halves') for label in TEST_LABELS])
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5)
    ax.text(-0.058, 2.28, target, ha="left", va="bottom", fontsize=7.0, fontweight="bold", color=color)
    ax.text(
        0.248,
        2.28,
        "n=3, 3, 100 (top→bottom)",
        ha="right",
        va="bottom",
        fontsize=5.0,
    )
    clean_axis(ax)


def _add_scale_bar(ax: plt.Axes, pixel_size_um: float, bar_um: float = 20.0) -> None:
    length = bar_um / pixel_size_um
    height, width = np.asarray(ax.images[-1].get_array()).shape[:2]
    x_mid = (width - 1) / 2.0
    x_start, x_end = x_mid - length / 2.0, x_mid + length / 2.0
    y = height - 25.0
    effects = [path_effects.Stroke(linewidth=2.4, foreground="black"), path_effects.Normal()]
    line, = ax.plot([x_start, x_end], [y, y], color="white", lw=1.35,
                    solid_capstyle="butt")
    line.set_path_effects(effects)
    label = ax.text((x_start + x_end) / 2, y - 8, "20 µm", ha="center",
                    va="bottom", color="white", fontsize=5.2, fontweight="bold")
    label.set_path_effects(effects)


def _draw_microscopy(
    grid_spec,
    fig: Any,
    data: dict[str, Any],
    *,
    compact: bool = False,
) -> list[plt.Axes]:
    sub = grid_spec.subgridspec(
        2,
        3,
        wspace=0.025 if compact else 0.035,
        hspace=0.012 if compact else 0.055,
    )
    phase_limits = (float(data["display"].phase_low), float(data["display"].phase_high))
    fluorescence_limits = (float(data["display"].fluorescence_low), float(data["display"].fluorescence_high))
    cyan = LinearSegmentedColormap.from_list("prb_cyan", ["#07151B", "#173947", "#79C1D0", "#DDF1F3"])
    axes: list[plt.Axes] = []
    for col, row in data["selection"].iterrows():
        asset = np.load(MICRO_ROOT / f"{row.asset_key}.npz")
        for channel_row, channel in enumerate(["phase", "fluorescence"]):
            ax = fig.add_subplot(sub[channel_row, col])
            axes.append(ax)
            if channel == "phase":
                ax.imshow(asset[channel], cmap="gray", vmin=phase_limits[0], vmax=phase_limits[1], interpolation="none")
            else:
                ax.imshow(asset[channel], cmap=cyan, vmin=fluorescence_limits[0], vmax=fluorescence_limits[1], interpolation="none")
                _add_scale_bar(ax, float(data["display"].pixel_size_um))
            if compact:
                # Image axes are square.  Anchoring the two rows towards their
                # shared boundary removes the artificial blank band that would
                # otherwise appear in a half-width manuscript panel.
                ax.set_anchor("S" if channel_row == 0 else "N")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if channel_row == 0:
                well = str(row.well_canonical).split("/")[1]
                ax.set_title(f"A{well}", pad=2.0, fontsize=6.0, fontweight="bold")
            if col == 0:
                ax.text(
                    -0.055, 0.5, "phase" if channel == "phase" else "pRb",
                    transform=ax.transAxes, rotation=90, ha="right", va="center",
                    fontsize=5.4, color=COLORS["ink"] if channel == "phase" else COLORS["stable"],
                )
    return axes


def draw_panel_b(
    container: Any,
    data: dict[str, Any] | None = None,
    *,
    add_letter: bool = True,
    compact: bool = False,
):
    """Draw Figure 5b into a Figure or SubFigure and return its Axes."""
    if data is None:
        data = prepare_panel_b()
    outer = container.add_gridspec(
        2,
        2,
        width_ratios=[0.80, 1.20] if compact else [1.05, 1.0],
        wspace=0.22 if compact else 0.34,
        hspace=0.18 if compact else 0.34,
    )
    ax_top = container.add_subplot(outer[0, 0])
    ax_bottom = container.add_subplot(outer[1, 0], sharex=ax_top)
    _draw_case_axis(ax_top, "pRb", data["values"]["pRb"], 20260851)
    _draw_case_axis(ax_bottom, "NPM1", data["values"]["NPM1"], 20260852)
    if compact:
        for ax in (ax_top, ax_bottom):
            ax.tick_params(axis="y", labelsize=5.0, pad=0.7)
    ax_top.tick_params(labelbottom=False)
    ax_bottom.set_xlabel("KO-response agreement (Pearson r)")
    image_axes = _draw_microscopy(outer[:, 1], container, data, compact=compact)
    container.text(
        0.73 if compact else 0.745,
        0.982,
        "pRb · SALL3",
        ha="center",
        va="top",
        fontsize=5.8 if compact else 6.2,
        fontweight="bold",
    )
    if add_letter:
        container.text(0.002, 0.985, "b", ha="left", va="top", fontsize=8, fontweight="bold")
    return [ax_top, ax_bottom, *image_axes]


def write_plotted_sources(data: dict[str, Any]) -> None:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    data["value_table"].to_csv(PANEL_ROOT / "figure_source_repeatability_values.csv", index=False)
    data["summary"].to_csv(PANEL_ROOT / "figure_source_repeatability_summary.csv", index=False)
    data["selection"].to_csv(PANEL_ROOT / "figure_source_microscopy_selection.csv", index=False)
    display_record = {"target_key": "prb", **data["display"].to_dict()}
    pd.DataFrame([display_record]).to_csv(PANEL_ROOT / "figure_source_microscopy_display.csv", index=False)
    asset_rows = []
    for row in data["selection"].itertuples(index=False):
        path = MICRO_ROOT / f"{row.asset_key}.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        asset_rows.append(
            {
                "asset_key": row.asset_key,
                "diagnostic_cell_id": row.diagnostic_cell_id,
                "well_canonical": row.well_canonical,
                "relative_path": f"source_data/microscopy/{path.name}",
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(asset_rows).to_csv(PANEL_ROOT / "figure_source_microscopy_asset_manifest.csv", index=False)


def main() -> None:
    apply_style()
    data = prepare_panel_b()
    write_plotted_sources(data)
    fig = plt.figure(figsize=(183 / 25.4, 62 / 25.4), layout="constrained")
    draw_panel_b(fig, data)
    save_figure(fig, FIGURE_ROOT / "Figure5-b")
    plt.close(fig)


if __name__ == "__main__":
    main()
