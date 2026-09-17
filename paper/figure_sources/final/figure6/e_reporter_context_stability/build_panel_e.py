#!/usr/bin/env python3
"""Show within-reporter stability across directed destination-screen evaluations."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIG6 = HERE.parent
if str(FIG6) not in sys.path:
    sys.path.insert(0, str(FIG6))
from _portable_fonts import configure_sans, format_figure, layout_report  # noqa: E402

MM = 1 / 25.4
WHITE = "#FFFFFF"
INK = "#202A33"
MUTED = "#6F7E88"
GRID = "#DCE3E7"
BLUE = "#3E6F9C"
GOLD = "#DDA06A"
MID = "#7E8D96"
GATE = 0.70
METRICS = [
    ("consensus10_strict_gene_pearson", "Recovery across screens"),
    ("consensus10_strict_magnitude_spearman", "Rank across screens"),
]
STATUS_COLORS = {
    "stable_above": BLUE,
    "crosses_gate": GOLD,
    "stable_below": MID,
}


def configure() -> None:
    configure_sans(mpl, fm, FIG6)
    mpl.rcParams.update({
        "font.size": 6.1,
        "axes.labelsize": 5.9,
        "axes.titlesize": 6.6,
        "axes.linewidth": .55,
        "xtick.labelsize": 5.3,
        "ytick.labelsize": 4.7,
        "legend.fontsize": 5.1,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    })


def load_and_validate() -> pd.DataFrame:
    data = pd.read_csv(HERE / "figure_source_directed_screen_evaluations.csv")
    required = {
        "reporter_slug", "short_name", "destination_screen", "display_screen",
        "biological_category_y", *[metric for metric, _ in METRICS],
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise RuntimeError(f"Context source is missing columns: {missing}")
    if len(data) != 81 or data.reporter_slug.nunique() != 34:
        raise RuntimeError("Context source must contain 81 directions and 34 repeated reporters")
    if data[[m for m, _ in METRICS]].isna().any().any():
        raise RuntimeError("Context metrics contain missing values")
    if data.duplicated(["reporter_slug", "destination_screen"]).any():
        raise RuntimeError("Reporter-destination directions are not unique")
    return data


def shorten(name: str) -> str:
    replacements = {
        "NucleoLIVE Live Cell dye": "NucleoLIVE",
        "Peroxi_SPY650 live cell dye": "Peroxi-SPY650",
        "pHrodo-dextran Live Cell Dye": "pHrodo-dextran",
        "CellEvent-Caspase live-cell dye": "CellEvent-Caspase",
        "CellROX live-cell dye": "CellROX",
        "FeRhoNox live-cell dye": "FeRhoNox",
        "LysoTracker live-cell dye": "LysoTracker",
        "FastAct_SPY555 Live Cell Dye": "FastAct-SPY555",
        "ChromaLIVE 488 excitation": "ChromaLIVE 488",
        "ChromaLIVE 561 excitation": "ChromaLIVE 561",
        "BODIPY live cell dye": "BODIPY",
    }
    return replacements.get(name, name)


def gate_status(values: pd.Series) -> str:
    low, high = float(values.min()), float(values.max())
    if low >= GATE:
        return "stable_above"
    if high < GATE:
        return "stable_below"
    return "crosses_gate"


def reporter_eta_squared(data: pd.DataFrame, metric: str) -> float:
    values = data[metric].to_numpy(float)
    grand = float(values.mean())
    grouped = data.groupby("reporter_slug", sort=False)[metric]
    between = sum(len(v) * (float(v.mean()) - grand) ** 2 for _, v in grouped)
    total = float(np.sum((values - grand) ** 2))
    return float(between / total) if total > 0 else float("nan")


def prepare_plot_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_parts = []
    for metric, label in METRICS:
        part = (
            data.groupby(["reporter_slug", "short_name", "biological_category_y"], as_index=False)
            .agg(n_directions=(metric, "size"), minimum=(metric, "min"),
                 median=(metric, "median"), maximum=(metric, "max"))
        )
        part["metric"] = metric
        part["metric_label"] = label
        part["range"] = part.maximum - part.minimum
        part["status"] = np.select(
            [part.minimum >= GATE, part.maximum < GATE],
            ["stable_above", "stable_below"], default="crosses_gate",
        )
        summary_parts.append(part)
    reporter_metric = pd.concat(summary_parts, ignore_index=True)

    order = (
        reporter_metric.pivot(index="reporter_slug", columns="metric", values="median")
        .mean(axis=1).sort_values(ascending=False, kind="stable")
    )
    order_map = {reporter: i + 1 for i, reporter in enumerate(order.index)}
    reporter_metric["reporter_order"] = reporter_metric.reporter_slug.map(order_map)
    reporter_metric["display_label"] = reporter_metric.short_name.map(shorten)
    reporter_metric = reporter_metric.sort_values(
        ["reporter_order", "metric"], kind="stable"
    ).reset_index(drop=True)

    long = data.copy()
    long["reporter_order"] = long.reporter_slug.map(order_map)
    long["display_label"] = long.short_name.map(shorten)
    status_lookup = reporter_metric.set_index(["reporter_slug", "metric"])["status"]
    records = []
    for row in long.itertuples(index=False):
        for metric, label in METRICS:
            records.append({
                "reporter_slug": row.reporter_slug,
                "short_name": row.short_name,
                "display_label": shorten(row.short_name),
                "biological_category": row.biological_category_y,
                "destination_screen": row.destination_screen,
                "display_screen": row.display_screen,
                "metric": metric,
                "metric_label": label,
                "value": float(getattr(row, metric)),
                "reporter_order": order_map[row.reporter_slug],
                "status": status_lookup.loc[(row.reporter_slug, metric)],
            })
    plotted = pd.DataFrame(records).sort_values(
        ["reporter_order", "metric", "destination_screen"], kind="stable"
    )

    overall_rows = []
    for metric, label in METRICS:
        q = reporter_metric.loc[reporter_metric.metric.eq(metric)]
        counts = q.status.value_counts().to_dict()
        overall_rows.append({
            "metric": metric,
            "metric_label": label,
            "n_directions": int(len(data)),
            "n_reporters": int(data.reporter_slug.nunique()),
            "stable_above_gate": int(counts.get("stable_above", 0)),
            "crosses_gate": int(counts.get("crosses_gate", 0)),
            "stable_below_gate": int(counts.get("stable_below", 0)),
            "median_within_reporter_range": float(q.range.median()),
            "iqr25_within_reporter_range": float(q.range.quantile(.25)),
            "iqr75_within_reporter_range": float(q.range.quantile(.75)),
            "maximum_within_reporter_range": float(q.range.max()),
            "reporter_group_eta_squared_descriptive": reporter_eta_squared(data, metric),
            "gate": GATE,
        })
    overall = pd.DataFrame(overall_rows)
    return plotted, reporter_metric, overall


def _marker_safe_xlim(ax, lower=0.0, upper=1.0):
    """Reserve physical room for the entire largest diamond at either endpoint."""
    # The normalized D path has horizontal radius 1/sqrt(2); scatter's size is
    # in points squared. Include half the outline and an antialiasing gutter.
    radius_pt = np.sqrt(17.0 / 2.0) + 0.45 / 2.0 + 0.65
    width_pt = ax.bbox.width * 72.0 / ax.get_figure().dpi
    fraction = radius_pt / width_pt
    if fraction >= 0.5:
        raise RuntimeError("Context-stability axis is too narrow for its markers")
    padding = (upper - lower) * fraction / (1.0 - 2.0 * fraction)
    ax.set_xlim(lower - padding, upper + padding)


def assert_marker_clearance(axes):
    """Check full marker extents, not just centres, in native or composite axes."""
    if not axes:
        raise RuntimeError("No context-stability axes were supplied")
    axes[0].get_figure().canvas.draw()
    for ax in axes:
        for collection in ax.collections:
            if not getattr(collection, "_context_marker", False):
                continue
            centres = collection.get_offset_transform().transform(collection.get_offsets())
            vertices = collection.get_paths()[0].vertices
            half_path = np.max(np.abs(vertices), axis=0)
            radius = (half_path * np.sqrt(collection.get_sizes().max())
                      + collection.get_linewidths().max() / 2 + 0.25)
            radius *= ax.get_figure().dpi / 72.0
            box = ax.bbox
            if np.any(centres - radius < [box.x0, box.y0]) or np.any(
                    centres + radius > [box.x1, box.y1]):
                raise RuntimeError("A context-stability marker would be clipped")


def draw_panel_e(container, *, add_letter: bool = False):
    configure()
    raw = load_and_validate()
    plotted, reporter_metric, overall = prepare_plot_data(raw)
    ordered = (
        reporter_metric[["reporter_slug", "display_label", "reporter_order"]]
        .drop_duplicates().sort_values("reporter_order").reset_index(drop=True)
    )
    union_cross = set(
        reporter_metric.loc[reporter_metric.status.eq("crosses_gate"), "reporter_slug"]
    )

    # Continue the same frozen ordering across two blocks of 17 reporters.
    # This shortens the panel without shrinking its six-point labels.
    axes = []
    split = (len(ordered) + 1) // 2
    specifications = []
    for block in range(2):
        subset = ordered.iloc[block * split:(block + 1) * split]
        for metric_index, metric_title in enumerate(METRICS):
            ax = container.add_axes([.195 + .5 * block + .15 * metric_index,
                                     .17, .12, .70])
            axes.append(ax)
            specifications.append((ax, metric_title, subset, metric_index))

    for ax, (metric, title), subset, metric_index in specifications:
        metric_summary = reporter_metric.loc[
            reporter_metric.metric.eq(metric)
        ].set_index("reporter_slug")
        for y, reporter in enumerate(subset.reporter_slug):
            row = metric_summary.loc[reporter]
            colour = STATUS_COLORS[row.status]
            ax.hlines(y, row.minimum, row.maximum, color=colour, lw=.8, alpha=.75, zorder=1)
            points = plotted.loc[
                plotted.reporter_slug.eq(reporter) & plotted.metric.eq(metric)
            ].sort_values("destination_screen", kind="stable")
            jitter = np.linspace(-.105, .105, len(points)) if len(points) > 1 else np.array([0.0])
            screen_marks = ax.scatter(points.value, y + jitter, s=8.5,
                                      facecolor=mpl.colors.to_rgba(colour, .42),
                                      edgecolor=WHITE, linewidth=.22, zorder=2)
            median_mark = ax.scatter(row["median"], y, s=17, marker="D",
                                     facecolor=colour, edgecolor=WHITE,
                                     linewidth=.45, zorder=3)
            screen_marks._context_marker = True
            median_mark._context_marker = True

        ax.axvline(GATE, color=GOLD, lw=.75, ls=(0, (3, 2)), zorder=0)
        _marker_safe_xlim(ax)
        ax.set_ylim(len(subset) - .5, -.5)
        # At 13 mm per axis, three numeric labels collide after adding the
        # marker gutter. The shared category key already states the 0.70 gate.
        ax.set_xticks([0, 1.0])
        ax.set_xticklabels(["0", "1"])
        ax.set_xlabel("Recovery\n(r)" if metric_index == 0 else "Rank\n(ρ)", labelpad=1.2)
        short_title = "Recovery" if metric == METRICS[0][0] else "Rank"
        row = overall.loc[overall.metric.eq(metric)].iloc[0]
        # Category totals and gate definitions are reported in the source
        # summary and caption; reserve the plot header for its shared key.
        ax.grid(axis="x", color=GRID, lw=.35, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.set_yticks(range(len(subset)), subset.display_label if metric_index == 0 else [])
        for tick, reporter in zip(ax.get_yticklabels(), subset.reporter_slug):
            tick.set_color(GOLD if reporter in union_cross else INK)
            tick.set_fontweight("bold" if reporter in union_cross else "normal")

    handles = [
        Line2D([], [], marker="D", ls="-", lw=.8, ms=3.2,
               color=STATUS_COLORS[status], markerfacecolor=STATUS_COLORS[status],
               markeredgecolor=WHITE, markeredgewidth=.35, label=label)
        for status, label in [
            ("stable_above", "≥0.70"),
            ("crosses_gate", "crosses"),
            ("stable_below", "<0.70"),
        ]
    ]
    handles.append(Line2D([], [], marker="o", ls="", ms=3,
                          markerfacecolor="#C5CDD2", markeredgecolor=WHITE,
                          markeredgewidth=.3, label="screen"))
    container.legend(handles=handles, ncol=4, loc="upper center",
                     bbox_to_anchor=(.55, .980), frameon=False,
                     columnspacing=.85, handletextpad=.28, borderaxespad=0)
    if add_letter:
        container.text(.004, .995, "e", ha="left", va="top",
                       fontsize=8.5, fontweight="bold", color=INK)
    return {"axes": axes, "plotted": plotted,
            "reporter_summary": reporter_metric, "overall": overall}


def build() -> None:
    configure()
    plotted, reporter_metric, overall = prepare_plot_data(load_and_validate())
    plotted.to_csv(HERE / "figure_source_context_stability_long.csv", index=False)
    reporter_metric.to_csv(HERE / "figure_source_context_stability_reporter_summary.csv", index=False)
    summary_path = HERE / "figure_source_context_stability_summary.csv"
    if summary_path.is_file():
        # A renderer must not rewrite frozen summary bytes merely because a
        # library serializes an equivalent floating-point result differently.
        pd.testing.assert_frame_equal(pd.read_csv(summary_path), overall,
                                      check_exact=False, rtol=1e-12, atol=1e-12)
    else:
        overall.to_csv(summary_path, index=False)
    fig = plt.figure(figsize=(111 * MM, 65 * MM), facecolor=WHITE)
    result = draw_panel_e(fig, add_letter=False)
    stem = HERE / "Figure6-e_reporter_context_stability"
    format_figure(fig, 6)
    assert_marker_clearance(result["axes"])
    qa_root = next(p for p in Path(__file__).resolve().parents if p.name == "figure6") / "qa"
    layout_report(fig, qa_root / (stem.name + "_typography.json"))
    fig.savefig(stem.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    build()
