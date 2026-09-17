#!/usr/bin/env python3
"""Build Figure 6b: nested leave-one-reporter-out calibration audit.

All points are reporters.  Association confidence intervals are deterministic
percentile intervals obtained by resampling the 52 reporter rows; no cell-, gene-
or fold-level observations are treated as independent replicates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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
RED = "#D56C75"
WHITE = "#FFFFFF"
TIER_COLORS = {
    "quantitative_proxy": "#3E6F9C",
    "ranking_proxy": "#DDA06A",
    "measurement_required": "#D56C75",
    "not_identifiable": "#7E8D96",
    "unresolved": "#C8D0D4",
}
PREDICTORS = [
    ("ensemble_recoverability_r", "Recovery"),
    ("magnitude_spearman", "Rank"),
    ("amplitude_fidelity", "Amplitude"),
    ("within_screen_split_half_pearson_median", "Reliability"),
]
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_REPS = 10000
UTILITY_RANK_DECIMALS = 12


def configure() -> None:
    configure_sans(mpl, font_manager, PAPER_ROOT)
    mpl.rcParams.update(
        {
            "font.size": 6.1,
            "axes.titlesize": 6.4,
            "axes.labelsize": 5.9,
            "xtick.labelsize": 5.2,
            "ytick.labelsize": 5.2,
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


def utility_spearman(x, utility):
    """Spearman with mathematical utility ties stabilized before average ranking.

    The four-component training-reference index has half-rank steps of 1/408.
    Twelve decimal places retain every distinct index value while preventing
    floating-point summation or CSV parsing from splitting mathematical ties.
    The rounded copy is used only for rank statistics, never for saved values,
    prediction residuals, R2, plot coordinates or reporter classification.
    """
    return spearmanr(x, np.round(np.asarray(utility, dtype=float), UTILITY_RANK_DECIMALS))


def bootstrap_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, int]:
    keep = np.isfinite(x) & np.isfinite(y)
    x = x[keep]
    y = np.round(y[keep], UTILITY_RANK_DECIMALS)
    n = len(x)
    estimate = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_REPS, dtype=float)
    draws.fill(np.nan)
    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        draws[i] = spearmanr(x[idx], y[idx]).statistic
    draws = draws[np.isfinite(draws)]
    if len(draws) < 0.99 * BOOTSTRAP_REPS:
        raise RuntimeError("Too many undefined reporter-bootstrap Spearman replicates")
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high), n


def fixed_density(values: np.ndarray, grid: np.ndarray, bandwidth: float = 0.055) -> np.ndarray:
    z = (grid[:, None] - values[None, :]) / bandwidth
    return np.exp(-0.5 * z * z).sum(axis=1) / (len(values) * bandwidth * np.sqrt(2 * np.pi))


def load_source() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load released strict outer-LOO inputs stored beside this panel.

    The panel no longer depends on the private common-scale analysis hierarchy;
    these tables are its frozen, public source-of-truth snapshot.
    """
    data = pd.read_csv(HERE / "figure_source_nested_loo.csv")
    associations = pd.read_csv(HERE / "figure_source_association_bootstrap.csv")
    summary = pd.read_csv(HERE / "figure_source_nested_loo_summary.csv").iloc[0]
    required = {
        "reporter_slug", "scientific_utility_index", "outer_loo_prediction",
        "short_name", "substitutability_tier", "label_flag", "label_reason",
        *(column for column, _ in PREDICTORS),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise RuntimeError(f"Figure 6b source table is missing columns: {missing}")
    if len(data) != 52 or data["reporter_slug"].duplicated().any():
        raise RuntimeError("Figure 6b requires exactly 52 unique reporter rows")
    if data[["scientific_utility_index", "outer_loo_prediction"]].isna().any().any():
        raise RuntimeError("Nested-LOO observed or predicted utility is missing")
    data["short_name"] = data["short_name"].astype(str).str.replace("*", "", regex=False)
    data.loc[data["short_name"].eq("NucleoLIVE Live Cell dye"), "short_name"] = "NucleoLIVE"
    data.loc[data["short_name"].eq("CellROX live-cell dye"), "short_name"] = "CellROX"
    y = data["scientific_utility_index"].to_numpy(float)
    for column, label in PREDICTORS:
        estimate = float(utility_spearman(data[column].to_numpy(float), y).statistic)
        row = associations.loc[associations["predictor"].eq(column)]
        if len(row) != 1:
            raise RuntimeError(f"Figure 6b association table lacks {column}")
        frozen = float(row.iloc[0]["rho"])
        if not np.isclose(estimate, frozen, rtol=0, atol=1e-12):
            raise RuntimeError(
                f"Strict association mismatch for {column}: recomputed={estimate:.6f}, frozen={frozen:.6f}"
            )
        if str(row.iloc[0]["label"]) != label:
            raise RuntimeError(f"Figure 6b association label mismatch for {column}")
    if int(summary["n_reporters"]) != 52:
        raise RuntimeError("Figure 6b nested-LOO summary must describe 52 reporters")
    predicted = data["outer_loo_prediction"].to_numpy(float)
    expected_summary = {
        "outer_loo_spearman": float(utility_spearman(predicted, y).statistic),
        "outer_loo_r2": float(1 - np.sum((predicted - y) ** 2) / np.sum((y - y.mean()) ** 2)),
        "median_selected_alpha": float(data["selected_alpha"].median()),
    }
    for column, expected in expected_summary.items():
        if not np.isclose(expected, float(summary[column]), rtol=0, atol=1e-12):
            raise RuntimeError(f"Figure 6b summary mismatch for {column}: {expected:.12g}")
    return data, associations, summary


def verify_statistics() -> None:
    """Read-only checks of mathematical ties, headlines, all intervals and panel d."""
    # Known answer: ranks [1, 2, 3, 4] against [1.5, 1.5, 3, 4]. A one-ULP
    # perturbation must not turn this into a perfect correlation.
    x = np.arange(1.0, 5.0)
    y = np.array([0.1, np.nextafter(0.1, np.inf), 0.2, 0.3])
    np.testing.assert_allclose(utility_spearman(x, y).statistic, np.sqrt(0.9), rtol=0, atol=1e-14)
    data, associations, summary = load_source()
    utility = data["scientific_utility_index"].to_numpy(float)
    for column, _ in PREDICTORS:
        estimate, low, high, n = bootstrap_spearman(data[column].to_numpy(float), utility)
        row = associations.loc[associations["predictor"].eq(column)].iloc[0]
        np.testing.assert_allclose(
            [estimate, low, high], row[["rho", "ci_low", "ci_high"]].to_numpy(float),
            rtol=0, atol=1e-12, err_msg=f"Reporter-bootstrap mismatch for {column}",
        )
        assert int(row.n_reporters) == n == 52
        assert int(row.bootstrap_reps) == BOOTSTRAP_REPS
        assert int(row.bootstrap_seed) == BOOTSTRAP_SEED
        print(f"{column}: rho={estimate:.12f}, 95% CI [{low:.12f}, {high:.12f}]")

    from d_prediction_utility.build_panel_d import load_and_validate, prepare_plot_data

    panel_d = FIGURE6_ROOT / "d_prediction_utility"
    _, computed_d = prepare_plot_data(load_and_validate())
    archived_d = pd.read_csv(panel_d / "figure_source_prediction_utility_summary.csv")
    numeric = computed_d.select_dtypes(include="number").columns
    np.testing.assert_allclose(computed_d[numeric], archived_d[numeric], rtol=0, atol=1e-12)
    print(f"outer-LOO: rho={summary.outer_loo_spearman:.12f}, R2={summary.outer_loo_r2:.12f}")
    print("PASS: known-answer average ties, primary statistics, four bootstrap intervals and panel d")


def draw_panel_b(
    container,
    add_letter: bool = False,
    data: pd.DataFrame | None = None,
    associations: pd.DataFrame | None = None,
    summary: pd.Series | None = None,
    left_margin: float = 0.125,
):
    """Draw panel b into a Matplotlib Figure or SubFigure using live text."""

    if data is None or associations is None or summary is None:
        data, associations, summary = load_source()
    assoc = associations
    observed = data["scientific_utility_index"].to_numpy(float)
    predicted = data["outer_loo_prediction"].to_numpy(float)
    rho = float(summary["outer_loo_spearman"])
    r2 = float(summary["outer_loo_r2"])

    outer = container.add_gridspec(
        1,
        2,
        width_ratios=[1.82, 1.12],
        left=left_margin,
        right=0.925,
        top=0.920,
        bottom=0.100,
        wspace=0.43,
    )
    left = outer[0].subgridspec(
        3,
        2,
        height_ratios=[0.28, 1.45, 0.88],
        width_ratios=[1.52, 0.23],
        hspace=0.55,
        wspace=0.05,
    )
    ax_top = container.add_subplot(left[0, 0])
    ax_scatter = container.add_subplot(left[1, 0])
    ax_right = container.add_subplot(left[1, 1])
    ax_ba = container.add_subplot(left[2, 0])
    ax_forest = container.add_subplot(outer[1])

    # Raise only the scatter and its matched right marginal.  The top density,
    # outer-LOO statistic and lower residual plot remain fixed.
    scatter_shift = 0.055
    for axis in (ax_scatter, ax_right):
        pos = axis.get_position()
        axis.set_position([pos.x0, pos.y0 + scatter_shift, pos.width, pos.height])

    # Marginal distributions use the same bandwidth on the common 0-1 scale.
    grid = np.linspace(0, 1, 240)
    dobs = fixed_density(observed, grid)
    dpred = fixed_density(predicted, grid)
    ax_top.fill_between(grid, 0, dobs / dobs.max(), color="#D4E3ED", lw=0)
    ax_top.plot(grid, dobs / dobs.max(), color=TEAL, lw=0.7)
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(0, 1.05)
    ax_top.axis("off")
    ax_right.fill_betweenx(grid, 0, dpred / dpred.max(), color="#D7E5EF", lw=0)
    ax_right.plot(dpred / dpred.max(), grid, color=MUTED, lw=0.7)
    ax_right.set_ylim(0, 1)
    ax_right.set_xlim(0, 1.05)
    ax_right.axis("off")

    point_colors = [TIER_COLORS[x] for x in data["substitutability_tier"]]
    ax_scatter.plot([0, 1], [0, 1], color=MUTED, lw=0.75, ls=(0, (3, 2)), zorder=1)
    ax_scatter.scatter(observed, predicted, s=15, c=point_colors, edgecolor=WHITE, linewidth=0.35, alpha=0.90, zorder=3)
    # Four pre-specified largest absolute residuals are labelled in compact
    # rails, preventing data-dependent text collisions inside the scatter.
    labelled = data.loc[data["label_flag"]]
    for _, row in labelled.iterrows():
        name = row["short_name"]
        if name == "PSMB7":
            text_x, text_y = 0.05, 0.88
        elif name == "pRb":
            text_x, text_y = 0.05, 0.075
        elif name == "VAMP3":
            text_x, text_y = 0.96, 0.075
        elif name == "CellROX":
            text_x, text_y = 0.72, 0.18
        else:
            text_x = 0.96 if row["scientific_utility_index"] >= 0.55 else 0.05
            text_y = 0.90 if row["prediction_residual"] >= 0 else 0.08
        ax_scatter.annotate(
            row["short_name"],
            (row["scientific_utility_index"], row["outer_loo_prediction"]),
            xytext=(text_x, text_y),
            textcoords="axes fraction",
            ha="right" if text_x > 0.5 else "left",
            va="center",
            fontsize=5.0,
            fontweight="bold",
            color=INK,
            arrowprops={"arrowstyle": "-", "lw": 0.38, "color": MUTED,
                        "connectionstyle": "arc3,rad=0.3" if name == "CellROX" else "arc3,rad=0"},
        )
    container.text(
        0.43,
        0.972,
        f"outer-LOO  ρ={rho:.2f}  R²={r2:.2f}",
        ha="center",
        va="top",
        fontsize=5.2,
        fontweight="bold",
    )
    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0, 1)
    ax_scatter.set_xlabel("Observed utility", labelpad=2.0)
    ax_scatter.set_ylabel("Held-out prediction", labelpad=1.5)
    ax_scatter.set_xticks([0, 0.5, 1])
    ax_scatter.set_yticks([0, 0.5, 1])
    ax_scatter.tick_params(axis="x", pad=1.5)
    ax_scatter.spines[["top", "right"]].set_visible(False)
    ax_scatter.grid(color=GRID, lw=0.42, axis="both")
    ax_scatter.set_axisbelow(True)

    means = data["mean_utility"].to_numpy(float)
    residuals = data["prediction_residual"].to_numpy(float)
    bias = residuals.mean()
    sd = residuals.std(ddof=1)
    ax_ba.axhline(0, color=INK, lw=0.55)
    ax_ba.axhline(bias, color=TEAL, lw=0.75)
    ax_ba.axhline(bias + 1.96 * sd, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax_ba.axhline(bias - 1.96 * sd, color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax_ba.scatter(means, residuals, s=10, c=point_colors, edgecolor=WHITE, linewidth=0.25, alpha=0.88)
    ax_ba.set_xlim(0, 1)
    lim = max(0.22, float(np.nanmax(np.abs(residuals))) * 1.12)
    ax_ba.set_ylim(-lim, lim)
    ax_ba.set_xlabel("Mean utility", labelpad=0.5)
    ax_ba.set_ylabel("Pred. − obs.", labelpad=1.0)
    ax_ba.set_xticks([0, 0.5, 1])
    # Keep the residual panel's upper tick clear of the scatter panel's lower
    # x tick at their shared boundary.
    ax_ba.set_yticks([-0.25, 0.0, 0.25])
    ax_ba.tick_params(axis="y", pad=1.5)
    # Keep the bottom label inside the composite row without moving the axes.
    ax_ba.tick_params(axis="x", pad=0.5, length=1.5)
    ax_ba.spines[["top", "right"]].set_visible(False)
    ax_ba.grid(color=GRID, lw=0.38, axis="x")
    ax_ba.set_axisbelow(True)

    y = np.arange(len(assoc))[::-1]
    for yi, row in zip(y, assoc.itertuples(index=False)):
        ax_forest.plot([row.ci_low, row.ci_high], [yi, yi], color=MUTED, lw=1.15, solid_capstyle="round")
        ax_forest.scatter(row.rho, yi, s=23, color=TEAL, edgecolor=WHITE, linewidth=0.5, zorder=3)
        ax_forest.text(0.96, yi + .18, f"{row.rho:.2f}",
                       transform=ax_forest.get_yaxis_transform(),
                       ha="right", va="bottom", fontsize=6.0,
                       fontweight="bold", clip_on=False)
    ax_forest.axvline(0, color=GRID, lw=0.7)
    ax_forest.set_xlim(-0.05, 1.05)
    ax_forest.set_ylim(-0.7, len(assoc) - 0.3)
    ax_forest.set_yticks(y)
    ax_forest.set_yticklabels(assoc["label"])
    ax_forest.set_xticks([0, 0.5, 1])
    ax_forest.set_xlabel("Spearman ρ", labelpad=0.5)
    ax_forest.spines[["top", "right", "left"]].set_visible(False)
    ax_forest.tick_params(axis="y", length=0, pad=0.3)
    ax_forest.tick_params(axis="x", pad=0.5, length=1.5)
    ax_forest.grid(axis="x", color=GRID, lw=0.42)
    ax_forest.set_axisbelow(True)
    ax_forest.text(0.0, 1.04, "Utility associations", transform=ax_forest.transAxes,
                   ha="left", va="bottom", fontsize=5.4, fontweight="bold")

    if add_letter:
        container.text(0.014, 0.972, "b", ha="left", va="top",
                       fontsize=8.5, fontweight="bold", color=INK)

    return {
        "scatter": ax_scatter,
        "observed_density": ax_top,
        "predicted_density": ax_right,
        "residual": ax_ba,
        "associations": ax_forest,
    }


def build() -> None:
    configure()
    data, assoc, summary = load_source()
    fig = plt.figure(figsize=(82 * MM, 65 * MM), constrained_layout=False)
    draw_panel_b(fig, add_letter=False, data=data, associations=assoc, summary=summary,
                 left_margin=0.155)
    stem = HERE / "figure6b_nested_loo_audit"
    format_figure(fig, 6)
    qa_root = next(p for p in Path(__file__).resolve().parents if p.name == "figure6") / "qa"
    layout_report(fig, qa_root / (stem.name + "_typography.json"))
    for ext, kwargs in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(stem.with_suffix(f".{ext}"), facecolor=WHITE, bbox_inches=None, pad_inches=0, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-statistics", action="store_true", help="check statistics without rendering or writing files")
    args = parser.parse_args()
    verify_statistics() if args.verify_statistics else build()
