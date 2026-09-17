#!/usr/bin/env python3
"""Render Figure 5a from the frozen reporter-stability source tables.

The public ``draw_panel_a`` function accepts either a Matplotlib Figure or
SubFigure, so the panel can be placed into the final composite without
rasterising its vector layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from figure5_style import COLORS, SOURCE_ROOT, apply_style, clean_axis, save_figure


HERE = Path(__file__).resolve().parent
PANEL_ROOT = HERE.parent / "a_stability_constraints"
FIGURE_ROOT = PANEL_ROOT / "figure"

TERM_ORDER = [
    "within_screen_split_half_pearson_median",
    "guide_half_consistency",
    "cross_screen_pearson_median",
    "log10_exact_links",
]
SHORT_LABELS = {
    "within_screen_split_half_pearson_median": "Within-screen",
    "guide_half_consistency": "Guide halves",
    "cross_screen_pearson_median": "Cross-screen",
    "log10_exact_links": "Coverage",
}
FOREST_LABELS = {
    "within_screen_split_half_pearson_median": "Within-screen reliability",
    "guide_half_consistency": "Guide-half consistency",
    "cross_screen_pearson_median": "Cross-screen consistency",
    "log10_exact_links": "Exact-link coverage",
}
JOINT_TERMS = [
    "within_screen_split_half_pearson_median",
    "guide_half_consistency",
    "log10_exact_links",
]


def prepare_panel_a(source_root: Path = SOURCE_ROOT) -> dict[str, Any]:
    """Rebuild the declared summaries without altering their definitions."""
    canonical = pd.read_csv(source_root / "panel_a/canonical_reporter_stability_median10.csv")
    associations = pd.read_csv(source_root / "panel_a/stability_associations_median10.csv")
    joint = pd.read_csv(source_root / "panel_a/joint_constraint_model_median10.csv").iloc[0]

    # Pairwise-complete Spearman matrix.  Reporter is the inferential unit.
    pair_rows: list[dict[str, Any]] = []
    for row_term in TERM_ORDER:
        for col_term in TERM_ORDER:
            use = canonical[[row_term, col_term]].dropna()
            rho = 1.0 if row_term == col_term else float(spearmanr(use[row_term], use[col_term]).statistic)
            pair_rows.append(
                {
                    "row_term": row_term,
                    "column_term": col_term,
                    "spearman_rho": rho,
                    "n_reporters_pairwise_complete": int(len(use)),
                }
            )
    pairs = pd.DataFrame(pair_rows)

    # Deterministically reconstruct the frozen 3,000-label permutation test.
    complete = canonical.dropna(subset=JOINT_TERMS + ["consensus10_gene_pearson"]).copy()
    x = StandardScaler().fit_transform(complete[JOINT_TERMS].to_numpy(float))
    y = complete["consensus10_gene_pearson"].to_numpy(float)
    observed = float(LinearRegression().fit(x, y).score(x, y))
    seed = int(joint["permutation_seed"])
    n_draws = int(joint["permutation_draws"])
    rng = np.random.default_rng(seed)
    null = np.empty(n_draws, dtype=float)
    for index in range(n_draws):
        permuted = rng.permutation(y)
        null[index] = LinearRegression().fit(x, permuted).score(x, permuted)
    p_value = float((1 + np.sum(null >= observed)) / (n_draws + 1))

    if not np.isclose(observed, float(joint["r2_in_sample"]), atol=1e-12):
        raise AssertionError("Joint-model R2 does not reproduce the frozen source table")
    if not np.isclose(p_value, float(joint["permutation_p"]), atol=1e-15):
        raise AssertionError("Permutation P value does not reproduce the frozen source table")

    null_table = pd.DataFrame(
        {
            "draw": np.arange(n_draws, dtype=int),
            "permuted_r2": null,
            "permutation_seed": seed,
            "observed_r2": observed,
            "n_reporters_complete_case": int(len(complete)),
        }
    )
    return {
        "canonical": canonical,
        "associations": associations,
        "pairs": pairs,
        "joint": joint,
        "null": null_table,
        "p_value": p_value,
    }


def _draw_forest(ax: plt.Axes, associations: pd.DataFrame) -> None:
    lookup = associations.set_index("predictor")
    ordered = lookup.loc[TERM_ORDER].reset_index()
    y = np.arange(len(ordered))[::-1]
    ax.axvline(0, color=COLORS["mid"], lw=0.65, zorder=0)
    for ypos, row in zip(y, ordered.itertuples(index=False)):
        ax.plot([row.ci95_low, row.ci95_high], [ypos, ypos], color=COLORS["stable"], lw=1.25, zorder=2)
        ax.scatter(row.rho, ypos, s=26, color=COLORS["stable"], edgecolor="white", lw=0.55, zorder=3)
        ax.text(0.84, ypos + .12, f"n={int(row.n_reporters)}", ha="right", va="bottom", fontsize=6.0)
    ax.set_ylim(-.25, 3.45)
    ax.set_yticks(y, [FOREST_LABELS[t] for t in TERM_ORDER])
    ax.set_xlim(-0.16, 0.87)
    ax.set_xticks([0.0, 0.4, 0.8])
    ax.set_xlabel("Spearman ρ with recoverability")
    ax.set_title("Association with recoverability", loc="left", pad=3.5)
    ax.grid(axis="x", color=COLORS["grid"], lw=0.55)
    clean_axis(ax)


def _draw_triangle(ax: plt.Axes, pairs: pd.DataFrame) -> None:
    n = len(TERM_ORDER)
    cmap = LinearSegmentedColormap.from_list(
        "constraint_diverging", [COLORS["orange"], COLORS["white"], COLORS["stable"]]
    )
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    pair_lookup = pairs.set_index(["row_term", "column_term"])
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    for i, row_term in enumerate(TERM_ORDER):
        for j, col_term in enumerate(TERM_ORDER):
            value = pair_lookup.loc[(row_term, col_term)]
            if i >= j:
                rho = float(value.spearman_rho)
                rect = plt.Rectangle(
                    (j - 0.46, i - 0.46), 0.92, 0.92,
                    facecolor=cmap(norm(rho)), edgecolor="white", lw=0.8,
                )
                ax.add_patch(rect)
                color = "white" if rho > 0.62 else COLORS["ink"]
                ax.text(j, i, f"{rho:.2f}", ha="center", va="center", color=color, fontsize=5.15)
            else:
                ax.text(
                    j, i, f"n={int(value.n_reporters_pairwise_complete)}",
                    ha="center", va="center", fontsize=4.9, color=COLORS["ink"],
                )
    ax.set_xticks(np.arange(n), [SHORT_LABELS[t] for t in TERM_ORDER], rotation=38, ha="right")
    ax.set_yticks(np.arange(n), [SHORT_LABELS[t] for t in TERM_ORDER])
    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Constraint overlap", loc="left", pad=3.5)
    ax.text(
        0.99, 1.01, "lower: ρ   upper: n", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=5.0, color=COLORS["ink"],
    )


def _draw_null(ax: plt.Axes, null: pd.DataFrame, joint: pd.Series, p_value: float) -> None:
    values = null["permuted_r2"].to_numpy(float)
    observed = float(joint["r2_in_sample"])
    ax.hist(
        values, bins=np.linspace(0, 0.52, 32), density=True,
        color=COLORS["stable_light"], edgecolor="white", linewidth=0.35,
    )
    ax.axvline(observed, color=COLORS["constraint"], lw=1.5, zorder=3)
    ax.scatter([observed], [0.42], s=25, color=COLORS["constraint"], edgecolor="white", lw=0.5, zorder=4)
    ax.text(
        observed - 0.012, 6.85,
        f"Observed R²={observed:.3f}\nP={p_value:.4f}\nn={int(joint['n_reporters'])}",
        ha="right", va="top", fontsize=5.35,
    )
    ax.set_xlim(0, 0.52)
    ax.set_ylim(0, 7.25)
    ax.set_yticks([])
    ax.set_xlabel("Permuted-label R²")
    ax.set_title("Joint association", loc="left", pad=3.5)
    clean_axis(ax, left=False)


def draw_panel_a(container: Any, data: dict[str, Any] | None = None, *, add_letter: bool = True):
    """Draw Figure 5a into a Figure or SubFigure and return its Axes."""
    if data is None:
        data = prepare_panel_a()
    axes = container.subplots(
        1, 3,
        gridspec_kw={"width_ratios": [1.14, 1.03, 0.88], "wspace": 0.46},
    )
    _draw_forest(axes[0], data["associations"])
    _draw_triangle(axes[1], data["pairs"])
    _draw_null(axes[2], data["null"], data["joint"], data["p_value"])
    if add_letter:
        container.text(0.002, 0.985, "a", ha="left", va="top", fontsize=8, fontweight="bold")
    return axes


def write_plotted_sources(data: dict[str, Any]) -> None:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    data["associations"].to_csv(PANEL_ROOT / "figure_source_associations.csv", index=False)
    data["pairs"].to_csv(PANEL_ROOT / "figure_source_constraint_correlations.csv", index=False)
    data["null"].to_csv(PANEL_ROOT / "figure_source_joint_permutation_null.csv", index=False)
    pd.DataFrame([data["joint"]]).to_csv(PANEL_ROOT / "figure_source_joint_model.csv", index=False)


def render_standalone(data: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(183 / 25.4, 60 / 25.4))
    # Match the composite's wide matrix columns instead of allowing the
    # constrained solver to squeeze long rotated labels between empty gutters.
    fig.subplots_adjust(left=0.178, right=0.985, top=0.94, bottom=0.255)
    draw_panel_a(fig, data, add_letter=False)
    save_figure(fig, FIGURE_ROOT / "Figure5-a")
    plt.close(fig)


def main() -> None:
    apply_style()
    data = prepare_panel_a()
    write_plotted_sources(data)
    render_standalone(data)


if __name__ == "__main__":
    main()
