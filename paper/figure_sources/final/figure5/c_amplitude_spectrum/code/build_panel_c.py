#!/usr/bin/env python3
"""Build Figure 5c: aligned reporter rank/amplitude fidelity spectrum.

This is a plotting-only transformation of the frozen reporter-level source table.
Rows are deterministically ordered by predicted/observed KO-response variance ratio,
with reporter slug as the tie breaker. No inferential quantities are recomputed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.path import Path as DrawingPath
from matplotlib.text import Text
from matplotlib.transforms import Bbox


PANEL_ROOT = Path(__file__).resolve().parents[1]
FIGURE5_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FIGURE5_ROOT / "code"))

from figure5_style import BIOLOGY_COLORS, COLORS, apply_style, clean_axis, save_figure, format_figure  # noqa: E402


INPUT = FIGURE5_ROOT / "source_data" / "panel_c" / "amplitude_fidelity_spectrum.csv"
PLOTTED = PANEL_ROOT / "source_data" / "figure5c_plotted_source.csv"
OUT = PANEL_ROOT / "figure" / "Figure5-c_amplitude_spectrum"
QA = PANEL_ROOT / "qa_report.json"

WIDTH_MM = 90.0
HEIGHT_MM = 60.0
MM_TO_IN = 1 / 25.4

LABELS = {
    "prb": "pRb",
    "p53": "p53",
    "mitochondria_tomm20": "TOMM20",
    "nucleoli_npm1": "NPM1",
    "lysosome_lysotracker_live-cell_dye": "LysoTracker",
}


def load_source() -> pd.DataFrame:
    df = pd.read_csv(INPUT)
    required = {
        "reporter_slug",
        "short_name",
        "biological_category",
        "magnitude_spearman",
        "magnitude_variance_ratio",
        "amplitude_compressed",
        "consensus_aggregation",
        "statistical_unit",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required source fields: {sorted(missing)}")
    if len(df) != 52 or df["reporter_slug"].nunique() != 52:
        raise ValueError("Expected exactly 52 unique reporter rows")
    if df[["magnitude_spearman", "magnitude_variance_ratio"]].isna().any().any():
        raise ValueError("Fidelity tracks contain missing values")
    if set(df["consensus_aggregation"]) != {"unweighted_median_10_models"}:
        raise ValueError("Consensus definition is not the frozen ten-model median")

    df = df.sort_values(
        ["magnitude_variance_ratio", "reporter_slug"], kind="mergesort"
    ).reset_index(drop=True)
    df["reporter_order"] = np.arange(1, len(df) + 1)
    df["is_below_half_variance"] = df["magnitude_variance_ratio"] < 0.5
    df["retains_rank_above_half"] = df["magnitude_spearman"] > 0.5
    df["display_colour"] = np.where(
        df["is_below_half_variance"], COLORS["constraint"], COLORS["stable"]
    )
    df["label_flag"] = df["reporter_slug"].isin(LABELS)
    df["display_label"] = df["reporter_slug"].map(LABELS).fillna("")
    return df


def annotate_selected(ax_rank, ax_var, df: pd.DataFrame) -> None:
    """Label five pre-specified biological anchors without creating a label cloud.

    Labels are deliberately split across the two aligned tracks.  pRb, p53 and
    NPM1 are attached to ranking fidelity; TOMM20 and LysoTracker are attached to
    amplitude fidelity.  Thus every callout points to the quantity for which the
    reporter is discussed, while the shared x order preserves cross-track identity.
    """
    placements = {
        # reporter: (axis, dx, dy)
        "prb": (ax_rank, 1.35, 0.215),
        # Put the two low-rank callouts in the existing lower margin, away
        # from both the connected rank curve and the reporter stems.
        "p53": (ax_rank, 0.0, 0.0),
        "nucleoli_npm1": (ax_rank, -1.60, 0.0),
        "mitochondria_tomm20": (ax_var, 6.80, -0.200),
        # Lift LysoTracker above the local high-value data cloud; the previous
        # below-left placement crossed several stems and neighbouring points.
        "lysosome_lysotracker_live-cell-dye": (ax_var, -0.80, 0.160),
    }
    # The published slug contains "live-cell-dye"; retain a tolerant alias in case
    # an older source table used a slightly different separator.
    placements["lysosome_lysotracker_live-cell_dye"] = placements[
        "lysosome_lysotracker_live-cell-dye"
    ]
    for row in df.loc[df["label_flag"]].itertuples(index=False):
        ax, dx, dy = placements[row.reporter_slug]
        x = row.reporter_order - 1
        y = (
            float(row.magnitude_spearman)
            if ax is ax_rank
            else float(row.magnitude_variance_ratio)
        )
        lower_margin = row.reporter_slug in {"p53", "nucleoli_npm1"}
        ha = "center" if lower_margin else ("right" if dx < 0 else "left")
        ax.annotate(
            row.display_label,
            xy=(x, y),
            xytext=(x + dx, 0.365 if lower_margin else y + dy),
            ha=ha,
            va="center",
            fontsize=6.0,
            color=COLORS["ink"],
            fontweight="bold" if row.reporter_slug in {"prb", "lysosome_lysotracker_live-cell_dye"} else "normal",
            arrowprops={
                "arrowstyle": "-",
                "color": COLORS["mid"],
                "lw": 0.55,
                "shrinkA": 1.5,
                "shrinkB": 2.2,
            },
            zorder=6,
        )


def assert_low_callouts_clear(fig, ax_rank, ax_var, df: pd.DataFrame) -> None:
    """Check all five direct labels against text, points, stems and reference lines."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 0.25 * fig.dpi / 72.0
    labels = [t for ax in (ax_rank, ax_var) for t in ax.texts
              if t.get_text() in {"pRb", "p53", "NPM1", "TOMM20", "LysoTracker"}]
    assert len(labels) == 5
    point_radius = (np.sqrt(14.0) / 2.0 + 0.45 / 2.0) * fig.dpi / 72.0
    for label in labels:
        axis = label.axes
        metric = "magnitude_spearman" if axis is ax_rank else "magnitude_variance_ratio"
        points = axis.transData.transform(np.column_stack([np.arange(len(df)), df[metric].to_numpy(float)]))
        box = Text.get_window_extent(label, renderer=renderer)
        guarded = Bbox.from_extents(box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad)
        if label.get_text() in {"p53", "NPM1"}:
            assert guarded.y0 > ax_var.bbox.y1, f"{label.get_text()} enters the variance track"
        for line in axis.lines:
            path = line.get_path().transformed(line.get_transform())
            assert not path.intersects_bbox(guarded, filled=False), (
                f"{label.get_text()} intersects a rank-curve/reference line"
            )
        for collection in axis.collections:
            if not hasattr(collection, "get_segments"):
                continue
            for segment in collection.get_segments():
                path = DrawingPath(segment).transformed(collection.get_transform())
                assert not path.intersects_bbox(guarded, filled=False), (
                    f"{label.get_text()} intersects a reporter stem"
                )
        for x, y in points:
            mark = Bbox.from_extents(x - point_radius, y - point_radius,
                                    x + point_radius, y + point_radius)
            assert not guarded.overlaps(mark), f"{label.get_text()} touches a reporter marker"
        for other in axis.texts:
            if other is label or not other.get_visible() or not other.get_text().strip():
                continue
            other_box = Text.get_window_extent(other, renderer=renderer)
            assert not guarded.overlaps(other_box), f"{label.get_text()} overlaps another label"


def draw_panel_c(fig, spec, df: pd.DataFrame | None = None):
    """Draw panel c into an existing Matplotlib ``SubplotSpec``.

    The function is the canonical rendering implementation used by both the
    standalone panel and the final manuscript composite.
    """
    if df is None:
        df = load_source()
    gs = spec.subgridspec(
        nrows=3,
        ncols=1,
        height_ratios=[0.76, 1.0, 0.095],
        hspace=0.22,
    )
    ax_rank = fig.add_subplot(gs[0])
    ax_var = fig.add_subplot(gs[1], sharex=ax_rank)
    ax_bio = fig.add_subplot(gs[2], sharex=ax_rank)

    x = np.arange(len(df), dtype=float)
    compressed = df["is_below_half_variance"].to_numpy(bool)
    colours = df["display_colour"].tolist()

    # A shared pale region makes the frozen 11-reporter amplitude boundary readable
    # in both tracks without encoding it twice in point geometry.
    for ax in (ax_rank, ax_var):
        ax.axvspan(-0.5, compressed.sum() - 0.5, color=COLORS["constraint_light"], alpha=0.34, lw=0)
        ax.axvline(compressed.sum() - 0.5, color=COLORS["constraint"], lw=0.65, alpha=0.85)

    # Upper track: reporter ranking fidelity, kept on its natural Spearman scale.
    rank = df["magnitude_spearman"].to_numpy(float)
    ax_rank.plot(x, rank, color=COLORS["mid"], lw=0.7, alpha=0.72, zorder=1)
    ax_rank.vlines(x, 0.5, rank, color=colours, lw=0.62, alpha=0.38, zorder=2)
    ax_rank.scatter(x, rank, s=14, c=colours, edgecolors="white", linewidths=0.45, zorder=3)
    ax_rank.axhline(0.5, color=COLORS["constraint"], lw=0.65, ls=(0, (3, 2)), alpha=0.8)
    ax_rank.set_ylim(0.36, 1.025)
    ax_rank.set_yticks([0.5, 0.75, 1.0])
    ax_rank.set_ylabel("Magnitude rank (ρ)", labelpad=2.5)
    ax_rank.grid(axis="y", color=COLORS["grid"], lw=0.55)
    ax_rank.tick_params(axis="x", bottom=False, labelbottom=False)
    clean_axis(ax_rank, left=True, bottom=False)
    ax_rank.text(
        0.018,
        0.965,
        "10/11: ρ > 0.5",
        transform=ax_rank.transAxes,
        ha="left",
        va="top",
        fontsize=5.2,
        linespacing=0.95,
        color=COLORS["constraint"],
        fontweight="bold",
    )

    # Lower track: deviation from the no-compression reference (variance ratio=1).
    ratio = df["magnitude_variance_ratio"].to_numpy(float)
    ax_var.vlines(x, 0.5, ratio, color=colours, lw=0.95, alpha=0.72, zorder=2)
    ax_var.scatter(x, ratio, s=16, c=colours, edgecolors="white", linewidths=0.45, zorder=3)
    ax_var.axhline(0.5, color=COLORS["constraint"], lw=0.75, ls=(0, (3, 2)), alpha=0.9)
    ax_var.axhline(1.0, color=COLORS["mid"], lw=0.6, ls=(0, (2, 2)), alpha=0.9)
    ax_var.set_ylim(-0.015, 1.20)
    ax_var.set_yticks([0.0, 0.5, 1.0])
    ax_var.set_ylabel("Variance ratio", labelpad=2.5)
    ax_var.grid(axis="y", color=COLORS["grid"], lw=0.55)
    clean_axis(ax_var, left=True, bottom=False)
    ax_var.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_var.text(
        0.018,
        0.79,
        "11/52 below 0.5",
        transform=ax_var.transAxes,
        ha="left",
        va="top",
        fontsize=5.4,
        color=COLORS["constraint"],
        fontweight="bold",
    )
    annotate_selected(ax_rank, ax_var, df)

    # Biology is an annotation track, not a second quantitative scale.
    ax_bio.set_xlim(-0.5, len(df) - 0.5)
    for i, category in enumerate(df["biological_category"]):
        colour = BIOLOGY_COLORS.get(category, COLORS["mid"])
        ax_bio.add_patch(Rectangle((i - 0.48, 0.0), 0.96, 1.0, facecolor=colour, edgecolor="none"))
    ax_bio.set_ylim(0, 1)
    ax_bio.set_yticks([0.5], ["Biology"])
    ax_bio.set_xticks([0, 10, 25, 51], labels=["1", "11", "26", "52"])
    ax_bio.set_xlabel("Reporter order by variance ratio", labelpad=1.8)
    for spine in ax_bio.spines.values():
        spine.set_visible(False)
    ax_bio.tick_params(axis="x", width=0.5, length=1.8, pad=1.0)
    ax_bio.tick_params(axis="y", length=0, pad=3.0, labelsize=5.0)

    format_figure(fig, 5)
    assert_low_callouts_clear(fig, ax_rank, ax_var, df)

    return {"rank": ax_rank, "variance": ax_var, "biology": ax_bio}


def draw(df: pd.DataFrame):
    """Standalone 90-mm rendering using the same reusable panel function."""
    apply_style()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN))
    outer = fig.add_gridspec(
        1,
        1,
        left=0.115,
        right=0.992,
        top=0.94,
        bottom=0.155,
    )
    draw_panel_c(fig, outer[0, 0], df)
    return fig


def run_qa(df: pd.DataFrame) -> dict:
    below = int((df["magnitude_variance_ratio"] < 0.5).sum())
    retained = int(
        ((df["magnitude_variance_ratio"] < 0.5) & (df["magnitude_spearman"] > 0.5)).sum()
    )
    expected_order = df["magnitude_variance_ratio"].to_numpy(float)
    checks = {
        "n_rows": int(len(df)),
        "n_unique_reporters": int(df["reporter_slug"].nunique()),
        "n_variance_ratio_below_0_5": below,
        "n_below_0_5_retaining_rank_above_0_5": retained,
        "sorted_non_decreasing": bool(np.all(np.diff(expected_order) >= 0)),
        "consensus_aggregation": sorted(df["consensus_aggregation"].unique().tolist()),
        "labelled_reporters": df.loc[df["label_flag"], "reporter_slug"].tolist(),
        "figure_size_mm": [WIDTH_MM, HEIGHT_MM],
    }
    checks["status"] = "PASS" if (
        checks["n_rows"] == 52
        and checks["n_unique_reporters"] == 52
        and below == 11
        and retained == 10
        and checks["sorted_non_decreasing"]
        and checks["consensus_aggregation"] == ["unweighted_median_10_models"]
        and set(checks["labelled_reporters"]) == set(LABELS)
    ) else "FAIL"
    return checks


def main() -> None:
    df = load_source()
    PLOTTED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PLOTTED, index=False)
    fig = draw(df)
    save_figure(fig, OUT, dpi=600)
    plt.close(fig)
    qa = run_qa(df)
    QA.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    if qa["status"] != "PASS":
        raise RuntimeError(f"Figure 5c QA failed: {qa}")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
