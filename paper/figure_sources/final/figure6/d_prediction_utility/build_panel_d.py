#!/usr/bin/env python3
"""Directly compare predictive recovery with retained scientific utility."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
FIG6 = HERE.parent
if str(FIG6) not in sys.path:
    sys.path.insert(0, str(FIG6))
from _portable_fonts import configure_sans, format_figure, layout_report  # noqa: E402
from b_nested_loo_audit.build_panel_b import utility_spearman  # noqa: E402

MM = 1 / 25.4
WHITE = "#FFFFFF"
INK = "#202A33"
MUTED = "#6F7E88"
GRID = "#DCE3E7"
BLUE = "#3E6F9C"
GOLD = "#DDA06A"
RED = "#D56C75"
MID = "#7E8D96"
LIGHT = "#C8D0D4"
TIER_COLORS = {
    "quantitative_proxy": BLUE,
    "ranking_proxy": GOLD,
    "measurement_required": RED,
    "not_identifiable": MID,
    "unresolved": LIGHT,
}
TIER_LABELS = {
    "quantitative_proxy": "Quantitative",
    "ranking_proxy": "Ranking",
    "measurement_required": "Required",
    "not_identifiable": "Not ID",
    "unresolved": "Unresolved",
}


def configure() -> None:
    configure_sans(mpl, fm, FIG6)
    mpl.rcParams.update({
        "font.size": 6.5,
        "axes.labelsize": 6.6,
        "axes.titlesize": 6.8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 5.8,
        "ytick.labelsize": 5.8,
        "legend.fontsize": 4.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    })


def load_and_validate() -> pd.DataFrame:
    utility = pd.read_csv(HERE / "figure_source_reporter_utility.csv")
    ledger = pd.read_csv(HERE / "figure_source_decision_ledger.csv")
    needed = {
        "reporter_slug", "short_name", "ensemble_recoverability_r",
        "scientific_utility_index", "substitutability_tier",
    }
    missing = sorted(needed.difference(utility.columns))
    if missing:
        raise RuntimeError(f"Prediction-utility source is missing columns: {missing}")
    if len(utility) != 52 or utility.reporter_slug.nunique() != 52:
        raise RuntimeError("Prediction-utility source must contain 52 unique reporters")
    flags = ledger[["reporter_slug", "domain_sensitive"]].drop_duplicates()
    data = utility.merge(flags, on="reporter_slug", how="left", validate="one_to_one")
    data["domain_sensitive"] = data.domain_sensitive.fillna(False).astype(bool)
    if data[["ensemble_recoverability_r", "scientific_utility_index"]].isna().any().any():
        raise RuntimeError("Prediction or utility contains missing values")
    unknown = sorted(set(data.substitutability_tier) - set(TIER_COLORS))
    if unknown:
        raise RuntimeError(f"Unknown decision tiers: {unknown}")
    return data


def prepare_plot_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x_gate = 0.70
    y_split = float(data.scientific_utility_index.median())
    plot = data.copy()
    plot["prediction_region"] = np.where(
        plot.ensemble_recoverability_r >= x_gate, "high", "low"
    )
    plot["utility_region"] = np.where(
        plot.scientific_utility_index >= y_split, "high", "low"
    )
    plot["region"] = plot.prediction_region + "_prediction__" + plot.utility_region + "_utility"
    plot = plot.sort_values(
        ["scientific_utility_index", "ensemble_recoverability_r"], kind="stable"
    ).reset_index(drop=True)
    plot["plot_order"] = np.arange(1, len(plot) + 1)
    plot["prediction_gate"] = x_gate
    plot["descriptive_utility_split"] = y_split

    rho, pvalue = utility_spearman(
        plot.ensemble_recoverability_r, plot.scientific_utility_index
    )
    counts = plot.region.value_counts().to_dict()
    association = pd.read_csv(FIG6 / "b_nested_loo_audit/figure_source_association_bootstrap.csv")
    association = association.loc[association.predictor.eq("ensemble_recoverability_r")].iloc[0]
    np.testing.assert_allclose(float(rho), association.rho, rtol=0, atol=1e-12)
    summary = pd.DataFrame([{
        "n_reporters": len(plot),
        "prediction_gate": x_gate,
        "utility_split_definition": "cohort median; descriptive, not a decision gate",
        "descriptive_utility_split": y_split,
        "spearman_rho": float(rho),
        "spearman_p_two_sided": float(pvalue),
        "spearman_ci_low": float(association.ci_low),
        "spearman_ci_high": float(association.ci_high),
        "bootstrap_reps": int(association.bootstrap_reps),
        "bootstrap_seed": int(association.bootstrap_seed),
        "high_prediction_high_utility": counts.get("high_prediction__high_utility", 0),
        "high_prediction_low_utility": counts.get("high_prediction__low_utility", 0),
        "low_prediction_high_utility": counts.get("low_prediction__high_utility", 0),
        "low_prediction_low_utility": counts.get("low_prediction__low_utility", 0),
    }])
    return plot, summary


def draw_panel_d(container, *, add_letter: bool = False):
    configure()
    data, summary = prepare_plot_data(load_and_validate())
    y_split = float(summary.descriptive_utility_split.iloc[0])
    ax = container.add_axes([0.160, 0.125, 0.805, 0.825])

    # Very light region fields make agreement/disagreement immediately legible
    # without turning the scientific scatter into a four-colour infographic.
    ax.axvspan(0.0, 0.70, ymin=0.0, ymax=y_split, color="#F5F6F7", zorder=0)
    ax.axvspan(0.70, 1.0, ymin=0.0, ymax=y_split, color="#FBF2F0", zorder=0)
    ax.axvspan(0.0, 0.70, ymin=y_split, ymax=1.0, color="#F6F2E9", zorder=0)
    ax.axvspan(0.70, 1.0, ymin=y_split, ymax=1.0, color="#EEF4F8", zorder=0)
    ax.axvline(0.70, color=GOLD, lw=0.85, ls=(0, (3, 2)), zorder=1)
    ax.axhline(y_split, color=MUTED, lw=0.65, ls=(0, (2.2, 2.2)), zorder=1)

    regular = data.loc[~data.domain_sensitive]
    sensitive = data.loc[data.domain_sensitive]
    for tier, part in regular.groupby("substitutability_tier", sort=False):
        ax.scatter(
            part.ensemble_recoverability_r, part.scientific_utility_index,
            s=25, color=TIER_COLORS[tier], edgecolor=WHITE, linewidth=.55,
            alpha=.92, zorder=3,
        )
    for tier, part in sensitive.groupby("substitutability_tier", sort=False):
        ax.scatter(
            part.ensemble_recoverability_r, part.scientific_utility_index,
            s=34, marker="D", facecolor=TIER_COLORS[tier], edgecolor=INK,
            linewidth=.75, alpha=.96, zorder=4,
        )

    # Frozen examples, with labels in empty regions rather than over points
    # or the recovery gate after the d/e row was shortened.
    label_positions = {
        "VAPA": (.735, .895), "VPS35": (.84, .185), "5xUPRE": (.445, .71),
        "Hoechst": (.80, .075), "CellROX": (.42, .87), "NCLN": (.345, .22),
    }
    label_rows = []
    for label, (xx, yy) in label_positions.items():
        row = data.loc[data.short_name.eq(label)]
        if len(row) != 1:
            raise RuntimeError(f"Expected one reporter labelled {label}; found {len(row)}")
        rec = row.iloc[0]
        ax.annotate(
            label,
            (rec.ensemble_recoverability_r, rec.scientific_utility_index),
            xytext=(xx, yy), textcoords="data",
            fontsize=5.2, color=INK, fontweight="bold",
            ha="left", va="center",
            arrowprops={"arrowstyle": "-", "color": MID, "lw": .45,
                        "connectionstyle": "arc3,rad=0.2" if label == "NCLN" else "arc3,rad=0",
                        "shrinkA": 1.5, "shrinkB": 2.5},
            zorder=5,
        )
        label_rows.append({
            "short_name": label,
            "reporter_slug": rec.reporter_slug,
            "label_x_data": xx,
            "label_y_data": yy,
            "label_reason": "discordant prediction-utility example",
        })

    q = summary.iloc[0]
    region_labels = [
        (.020, .980, f"n={int(q.low_prediction_high_utility)}", "left", "top"),
        (.700, .980, f"n={int(q.high_prediction_high_utility)}", "left", "top"),
        (.020, .020, f"n={int(q.low_prediction_low_utility)}", "left", "bottom"),
        (.980, .020, f"n={int(q.high_prediction_low_utility)}", "right", "bottom"),
    ]
    for xx, yy, text, ha, va in region_labels:
        ax.text(xx, yy, text, transform=ax.transAxes, ha=ha, va=va,
                fontsize=5.2, color=MUTED)

    ax.set_xlim(0.10, 1.0); ax.set_ylim(0.0, 1.0)
    ax.set_xticks([.2, .4, .6, .8, 1.0]); ax.set_yticks([0, .25, .5, .75, 1.0])
    ax.set_xlabel("Ensemble recoverability (Pearson r)", labelpad=2)
    ax.set_ylabel("Scientific utility retained", labelpad=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.grid(False)
    ax.text(0.70, 1.012, "recovery gate 0.70", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=4.8, color=GOLD)
    ax.text(.105, y_split + .018, f"cohort median utility {y_split:.2f}",
            ha="left", va="bottom", fontsize=5.1, color=MUTED)

    if add_letter:
        container.text(.004, .995, "d", ha="left", va="top",
                       fontsize=8.5, fontweight="bold", color=INK)

    return {"axis": ax, "plot_data": data, "summary": summary,
            "label_rows": pd.DataFrame(label_rows)}


def assert_label_clearance(ax) -> None:
    """Check the final six-point text against all markers and both split lines."""
    from matplotlib.text import Text
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    markers = []
    for collection in ax.collections:
        offsets = collection.get_offsets()
        if not len(offsets):
            continue
        centers = collection.get_offset_transform().transform(offsets)
        sizes = collection.get_sizes()
        if not len(sizes):
            continue
        extent = max(np.linalg.norm(p.vertices, axis=1).max() for p in collection.get_paths())
        radius = np.sqrt(float(max(sizes))) * extent * fig.dpi / 72 + .6 * fig.dpi / 72
        markers.extend((x, y, radius) for x, y in centers)
    recovery_x = ax.transData.transform((.70, 0))[0]
    utility_y = ax.transData.transform((0, float(prepare_plot_data(load_and_validate())[1]
                                             .descriptive_utility_split.iloc[0])))[1]
    checked = [t for t in ax.texts if t.get_visible() and t.get_text().strip()]
    for t in checked:
        box = Text.get_window_extent(t, renderer=renderer)
        for x, y, radius in markers:
            dx = max(box.x0 - x, 0, x - box.x1)
            dy = max(box.y0 - y, 0, y - box.y1)
            assert dx * dx + dy * dy > radius * radius, f'Figure 6d text/marker: {t.get_text()}'
        inside = box.overlaps(ax.bbox)
        assert not (inside and box.x0 <= recovery_x <= box.x1), f'Figure 6d text/gate: {t.get_text()}'
        assert not (inside and box.y0 <= utility_y <= box.y1), f'Figure 6d text/median: {t.get_text()}'


def build() -> None:
    configure()
    data, summary = prepare_plot_data(load_and_validate())
    data.to_csv(HERE / "figure_source_prediction_utility_plot.csv", index=False)
    summary.to_csv(HERE / "figure_source_prediction_utility_summary.csv", index=False)
    fig = plt.figure(figsize=(72 * MM, 65 * MM), facecolor=WHITE)
    result = draw_panel_d(fig, add_letter=False)
    result["label_rows"].to_csv(HERE / "figure_source_direct_labels.csv", index=False)
    stem = HERE / "Figure6-d_prediction_utility"
    format_figure(fig, 6)
    assert_label_clearance(result['axis'])
    qa_root = next(p for p in Path(__file__).resolve().parents if p.name == "figure6") / "qa"
    layout_report(fig, qa_root / (stem.name + "_typography.json"))
    fig.savefig(stem.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    build()
