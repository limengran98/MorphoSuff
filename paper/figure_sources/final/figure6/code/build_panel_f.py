#!/usr/bin/env python3
"""Build Figure 6f: cross-context frozen-gate decision paths.

This panel is a C11 decision-gate matrix.  It deliberately draws the external
measurement readouts as individual rows rather than making two observations look
like a population.  The OPS row is an atlas summary: the pale ticks are the 52
reporters and the point is their median.  All thresholds and verdicts are frozen by
the deposited Figure 6 analysis.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
OUTROOT = HERE.parents[1] / "f_cross_context_decisions"
FIG6 = HERE.parents[1]
PAPER_ROOT = FIG6
if str(FIG6) not in sys.path:
    sys.path.insert(0, str(FIG6))
from _portable_fonts import configure_sans, format_figure, layout_report  # noqa: E402
LOCAL_LEDGER = FIG6 / "c_decision_ledger" / "source_data" / "figure6c_decision_ledger_source.csv"
LOCAL_GATE_ROWS = OUTROOT / "figure_source_gate_rows.csv"
LOCAL_REPLICATES = OUTROOT / "figure_source_replicate_detail.csv"
configure_sans(mpl, font_manager, PAPER_ROOT)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 6.2,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)

INK = "#202A33"
MUTED = "#6F7E88"
GRID = "#DCE3E7"
TEAL = "#3E6F9C"
CARMINE = "#D56C75"
ORANGE = "#DDA06A"
TIER = {
    "quantitative_proxy": "#3E6F9C",
    "ranking_proxy": "#DDA06A",
    "measurement_required": "#D56C75",
    "not_identifiable": "#7E8D96",
    "unresolved": "#C8D0D4",
}

METRICS = [
    ("recoverability_r", "Recovery", (0.0, 1.0), 0.70, None),
    ("magnitude_spearman", "Rank", (0.0, 1.0), 0.70, None),
    ("variance_ratio", "Amplitude", (0.0, 1.75), 0.50, 1.50),
    ("top5pct_recall", "Top-5%", (0.0, 1.0), 0.60, None),
    ("reliability", "Reliability", (0.0, 1.0), 0.30, None),
]

ROW_ORDER = [
    ("ops", "OPS reporter atlas", "A549 · 52 reporters"),
    ("oasis:mtt_viability", "Metabolic activity", "OASIS · primary hepatocytes"),
    ("oasis:ldh_release", "LDH release", "OASIS · primary hepatocytes"),
    ("periscope_cells:tomm20_mito", "anti-TOMM20", "PERISCOPE · per cell"),
    ("periscope_guide:tomm20_mito", "anti-TOMM20", "PERISCOPE · per guide"),
]


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tiers = pd.read_csv(LOCAL_LEDGER)
    ext = pd.read_csv(LOCAL_GATE_ROWS)
    reps = pd.read_csv(LOCAL_REPLICATES)
    if len(tiers) != 52 or tiers.reporter_slug.nunique() != 52:
        raise RuntimeError("panel-local OPS decision ledger is incomplete")
    if ext.loc[ext.band.eq("admitted")].groupby("row_key").quantity.nunique().lt(5).any():
        raise RuntimeError("panel-local external gate source is incomplete")
    return tiers, ext, reps


def ops_metric_values(tiers: pd.DataFrame, metric: str) -> np.ndarray:
    column = {
        "recoverability_r": "ensemble_recoverability_r",
        "magnitude_spearman": "magnitude_spearman",
        "variance_ratio": "magnitude_variance_ratio",
        "top5pct_recall": "top5pct_recall",
        "reliability": "reliability",
    }[metric]
    return tiers[column].astype(float).to_numpy()


def scale_x(value: float, bounds: tuple[float, float], x0: float, x1: float) -> float:
    lo, hi = bounds
    return x0 + (np.clip(value, lo, hi) - lo) / (hi - lo) * (x1 - x0)


def draw_gate(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    y: float,
    bounds: tuple[float, float],
    threshold_low: float,
    threshold_high: float | None,
    value: float,
    replicate_values: np.ndarray | None,
    passed: bool,
    deciding: bool,
    atlas_values: np.ndarray | None = None,
) -> None:
    """Draw one scale-aware threshold rail in axes coordinates."""
    ax.plot([x0, x1], [y, y], color=GRID, lw=1.0, solid_capstyle="round", zorder=1)
    tx0 = scale_x(threshold_low, bounds, x0, x1)
    if threshold_high is None:
        ax.add_patch(
            Rectangle(
                (tx0, y - 0.022), x1 - tx0, 0.044,
                transform=ax.transAxes, facecolor="#E7EEF3", edgecolor="none", zorder=0
            )
        )
        ax.plot([tx0, tx0], [y - 0.034, y + 0.034], color=INK, lw=1.0, zorder=3)
    else:
        tx1 = scale_x(threshold_high, bounds, x0, x1)
        ax.add_patch(
            Rectangle(
                (tx0, y - 0.022), tx1 - tx0, 0.044,
                transform=ax.transAxes, facecolor="#E7EEF3", edgecolor="none", zorder=0
            )
        )
        ax.plot([tx0, tx0], [y - 0.034, y + 0.034], color=INK, lw=0.9, zorder=3)
        ax.plot([tx1, tx1], [y - 0.034, y + 0.034], color=INK, lw=0.9, zorder=3)

    if atlas_values is not None:
        finite = atlas_values[np.isfinite(atlas_values)]
        for v in finite:
            xv = scale_x(float(v), bounds, x0, x1)
            ax.plot([xv, xv], [y - 0.026, y + 0.026], color="#9EB7C8", lw=0.38, alpha=0.65)
    elif replicate_values is not None:
        for v in replicate_values[np.isfinite(replicate_values)]:
            xv = scale_x(float(v), bounds, x0, x1)
            ax.plot([xv, xv], [y - 0.022, y + 0.022], color="#9EB7C8", lw=0.55, alpha=0.85)

    xv = scale_x(float(value), bounds, x0, x1)
    edge = CARMINE if not passed else TEAL
    face = "white" if not passed else TEAL
    ax.scatter(
        [xv], [y], transform=ax.transAxes, s=21 if deciding else 16,
        facecolor=face, edgecolor=edge, linewidth=1.0 if deciding else 0.7,
        zorder=5,
    )
    if deciding:
        ax.scatter(
            [xv], [y], transform=ax.transAxes, s=43, facecolor="none",
            edgecolor=CARMINE, linewidth=0.7, zorder=4,
        )


def build_display_table(tiers: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tier_counts = tiers.substitutability_tier.value_counts().to_dict()
    for metric, label, bounds, threshold_low, threshold_high in METRICS:
        vals = ops_metric_values(tiers, metric)
        passed = (
            (vals >= threshold_low)
            if threshold_high is None
            else ((vals >= threshold_low) & (vals <= threshold_high))
        )
        rows.append(
            {
                "row_key": "ops",
                "label": "OPS reporter atlas",
                "metric": metric,
                "metric_label": label,
                "value": float(np.nanmedian(vals)),
                "threshold_min": threshold_low,
                "threshold_max": threshold_high,
                "passes": bool(np.nanmedian(vals) >= threshold_low) if threshold_high is None else bool(threshold_low <= np.nanmedian(vals) <= threshold_high),
                "passed_units": int(passed.sum()),
                "total_units": int(np.isfinite(vals).sum()),
                "verdict": "30 quantitative / 7 ranking / 3 required / 9 not ID / 3 unresolved",
                "tier_counts": ";".join(f"{k}:{tier_counts.get(k,0)}" for k in TIER),
            }
        )
    admitted = ext.loc[ext.band.eq("admitted")].copy()
    verdict = {
        "oasis:mtt_viability": "unresolved",
        "oasis:ldh_release": "unresolved",
        "periscope_cells:tomm20_mito": "not identifiable",
        "periscope_guide:tomm20_mito": "not identifiable",
    }
    for key, label, _ in ROW_ORDER[1:]:
        part = admitted.loc[admitted.row_key.eq(key)]
        for metric, metric_label, *_ in METRICS:
            row = part.loc[part.quantity.eq(metric)].iloc[0]
            rows.append(
                {
                    "row_key": key,
                    "label": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "value": float(row.value),
                    "threshold_min": float(row.threshold_min),
                    "threshold_max": float(row.threshold_max) if pd.notna(row.threshold_max) else np.nan,
                    "passes": bool(row.passes),
                    "passed_units": np.nan,
                    "total_units": np.nan,
                    "verdict": verdict[key],
                    "tier_counts": "",
                }
            )
    return pd.DataFrame(rows)


def draw_panel_f(
    container, *, add_letter: bool = False, compact: bool = False
) -> tuple[plt.Axes, pd.DataFrame]:
    """Draw the panel into a Figure or SubFigure for native-vector assembly.

    ``compact=True`` preserves all three datasets, five gates and five readout
    rows while tightening labels for a narrow manuscript slot.
    """
    tiers, ext, reps = load_tables()
    display = build_display_table(tiers, ext)
    wide = display.pivot(index="row_key", columns="metric", values="value")
    passed = display.pivot(index="row_key", columns="metric", values="passes")
    ext_admitted = ext.loc[ext.band.eq("admitted")].copy()

    ax = container.add_axes([0.012, 0.035 if compact else 0.05, 0.978, 0.94 if compact else 0.91])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if add_letter:
        container.text(0.001, 0.995, "f", ha="left", va="top",
                       fontsize=8.2, fontweight="bold", color=INK)

    x_label = 0.012
    # With the redundant decision column removed, use the released width for
    # the five frozen gates and their reporter/replicate distributions.
    # Bring the five gate rails closer to their row labels and use the recovered
    # horizontal span to make each quantitative scale easier to read.
    x_metric_start, x_metric_end = (0.225, 0.990) if compact else (0.225, 0.992)
    metric_width = 0.124 if compact else 0.132
    metric_centres = np.linspace(x_metric_start + metric_width / 2, x_metric_end - metric_width / 2, 5)
    y_header = 0.900 if compact else 0.875
    row_y = [0.705, 0.545, 0.405, 0.245, 0.105] if compact else [0.72, 0.55, 0.40, 0.25, 0.10]
    fs_head = 4.7 if compact else 6.3
    fs_metric = 4.35 if compact else 6.0
    fs_criterion = 3.65 if compact else 5.0
    fs_row = 4.45 if compact else 6.2
    fs_sub = 3.55 if compact else 5.0
    fs_value = 3.65 if compact else 5.0

    ax.text(x_label, y_header, "Measurement context", transform=ax.transAxes,
            ha="left", va="bottom", color=INK, fontsize=fs_head, fontweight="bold")
    for centre, (_, label, bounds, t0, t1) in zip(metric_centres, METRICS):
        ax.text(centre, y_header, label, transform=ax.transAxes, ha="center", va="bottom",
                color=INK, fontsize=fs_metric, fontweight="bold")
        criterion = f"{t0:.2g}–{t1:.2g}" if t1 is not None else f"≥{t0:.2g}"
        ax.text(centre, y_header - 0.035, criterion, transform=ax.transAxes,
                ha="center", va="top", color=MUTED, fontsize=fs_criterion)
    for idx, ((row_key, label, sublabel), y) in enumerate(zip(ROW_ORDER, row_y)):
        if compact:
            compact_label = {
                "ops": "OPS atlas",
                "oasis:mtt_viability": "OASIS · MTT",
                "oasis:ldh_release": "OASIS · LDH",
                "periscope_cells:tomm20_mito": "PERISCOPE · cell",
                "periscope_guide:tomm20_mito": "PERISCOPE · guide",
            }[row_key]
            compact_sub = {
                "ops": "A549 · 52 reporters",
                "oasis:mtt_viability": "primary hepatocytes",
                "oasis:ldh_release": "primary hepatocytes",
                "periscope_cells:tomm20_mito": "anti-TOMM20",
                "periscope_guide:tomm20_mito": "anti-TOMM20",
            }[row_key]
            label, sublabel = compact_label, compact_sub
        ax.text(x_label, y + 0.034, label, transform=ax.transAxes, color=INK,
                fontsize=fs_row, fontweight="bold", va="center")
        ax.text(x_label, y - 0.036, sublabel, transform=ax.transAxes, color=MUTED,
                fontsize=fs_sub, va="center")
        first_fail = None
        if row_key != "ops":
            for metric, *_ in METRICS:
                if not bool(passed.loc[row_key, metric]):
                    first_fail = metric
                    break

        for centre, (metric, _, bounds, threshold_low, threshold_high) in zip(metric_centres, METRICS):
            x0, x1 = centre - metric_width * 0.47, centre + metric_width * 0.47
            if row_key == "ops":
                vals = ops_metric_values(tiers, metric)
                value = float(np.nanmedian(vals))
                pass_value = bool(
                    value >= threshold_low if threshold_high is None
                    else threshold_low <= value <= threshold_high
                )
                draw_gate(
                    ax, x0=x0, x1=x1, y=y, bounds=bounds,
                    threshold_low=threshold_low, threshold_high=threshold_high,
                    value=value, replicate_values=None, passed=pass_value,
                    deciding=False, atlas_values=vals,
                )
                pass_count = int(
                    ((vals >= threshold_low) if threshold_high is None else
                     ((vals >= threshold_low) & (vals <= threshold_high))).sum()
                )
                ax.text(centre, y - (0.049 if compact else 0.057), f"{pass_count}/52", transform=ax.transAxes,
                        ha="center", va="top", color=MUTED, fontsize=fs_value)
            else:
                value = float(wide.loc[row_key, metric])
                repvals = reps.loc[
                    reps.row_key.eq(row_key) & reps.quantity.eq(metric), "value"
                ].astype(float).to_numpy()
                draw_gate(
                    ax, x0=x0, x1=x1, y=y, bounds=bounds,
                    threshold_low=threshold_low, threshold_high=threshold_high,
                    value=value, replicate_values=repvals,
                    passed=bool(passed.loc[row_key, metric]),
                    deciding=metric == first_fail, atlas_values=None,
                )
                ax.text(centre, y - (0.049 if compact else 0.057), f"{value:.2f}", transform=ax.transAxes,
                        ha="center", va="top", color=MUTED,
                        fontsize=fs_value, fontweight="normal")

    # The repeated gate glyph is self-evident when shown five times per row;
    # its detailed definition and the resulting tier belong in the legend and
    # source table rather than a redundant sixth display column.
    return ax, display


def draw_panel() -> tuple[plt.Figure, pd.DataFrame]:
    """Return the standalone panel figure and its display table."""
    fig = plt.figure(figsize=(7.20, 2.23))
    _, display = draw_panel_f(fig, add_letter=False)
    return fig, display


def main() -> int:
    OUTROOT.mkdir(parents=True, exist_ok=True)
    fig, display = draw_panel()
    base = OUTROOT / "Figure6-f_cross_context_decisions"
    format_figure(fig, 6)
    qa_root = next(p for p in Path(__file__).resolve().parents if p.name == "figure6") / "qa"
    layout_report(fig, qa_root / (base.name + "_typography.json"))
    for ext in ("pdf", "svg", "png"):
        kwargs = {"dpi": 600} if ext == "png" else {}
        fig.savefig(base.with_suffix(f".{ext}"), bbox_inches=None, **kwargs)
    display.to_csv(OUTROOT / "figure_source_data.csv", index=False)
    plt.close(fig)
    print(base.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
