#!/usr/bin/env python3
"""Render Figure 5e from the frozen conditional-ambiguity table.

The panel shows all available gene-level discrepancy ratios for three frozen
reporter--screen cases.  For each case, the upper half-raincloud uses nearest
neighbours in the engineered 172D phase representation; the lower half uses
nearest neighbours in raw phase thumbnails.  A ratio of one means that the
target-state discrepancy is indistinguishable from a random cell match.

The public :func:`draw_panel_e` entry point accepts either a Figure/SubFigure
container (the manuscript-composite interface) or an explicit ``fig, spec``
pair (the reusable panel interface).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from figure5_style import COLORS, SOURCE_ROOT, apply_style, clean_axis, save_figure


HERE = Path(__file__).resolve().parent
FIGURE5_ROOT = HERE.parent
PANEL_ROOT = FIGURE5_ROOT / "e_conditional_ambiguity"
FIGURE_ROOT = PANEL_ROOT / "figure"
SOURCE = SOURCE_ROOT / "panel_e/conditional_ambiguity_per_gene.csv"
SUMMARY = SOURCE_ROOT / "panel_e/conditional_ambiguity_summary.csv"
OUTPUT_STEM = FIGURE_ROOT / "Figure5-e_conditional_ambiguity"

CASE_ORDER = [
    ("FeRhoNox", "Biohub_OPS0047"),
    ("FeRhoNox", "Biohub_OPS0067"),
    ("pRb", "Biohub_OPS0077"),
]
CASE_LABELS = {
    ("FeRhoNox", "Biohub_OPS0047"): "FeRhoNox\nOPS0047",
    ("FeRhoNox", "Biohub_OPS0067"): "FeRhoNox\nOPS0067",
    ("pRb", "Biohub_OPS0077"): "pRb\nOPS0077",
}
REP_ORDER = ["phase172", "raw_thumbnail"]
REP_LABELS = {
    "phase172": "Engineered 172D",
    "raw_thumbnail": "Raw-thumbnail PCA",
}
REP_COLORS = {
    "phase172": COLORS["stable"],
    "raw_thumbnail": COLORS["orange"],
}


def prepare_panel_e() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the frozen full-gene source tables."""
    df = pd.read_csv(SOURCE)
    summary = pd.read_csv(SUMMARY)
    required = {
        "target",
        "screen",
        "gene",
        "representation",
        "endpoint_discrepancy_ratio",
        "n_cells",
        "n_guides",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Figure 5e source fields: {sorted(missing)}")
    if len(df) != 5990:
        raise ValueError(f"Expected 5,990 gene-level rows, found {len(df):,}")
    if df["endpoint_discrepancy_ratio"].isna().any():
        raise ValueError("Conditional-ambiguity ratio contains missing values")
    if (df["endpoint_discrepancy_ratio"] <= 0).any():
        raise ValueError("Conditional-ambiguity ratios must be positive")
    observed_cases = set(zip(df["target"], df["screen"]))
    if observed_cases != set(CASE_ORDER):
        raise ValueError(f"Frozen case set changed: {sorted(observed_cases)}")
    if set(df["representation"]) != set(REP_ORDER):
        raise ValueError("Frozen representation set changed")

    order_map = {case: i for i, case in enumerate(CASE_ORDER)}
    rep_map = {rep: i for i, rep in enumerate(REP_ORDER)}
    df = df.copy()
    df["case_order"] = [order_map[(t, s)] for t, s in zip(df.target, df.screen)]
    df["representation_order"] = df["representation"].map(rep_map)
    df["case_label"] = [CASE_LABELS[(t, s)] for t, s in zip(df.target, df.screen)]
    df["representation_label"] = df["representation"].map(REP_LABELS)
    df["log2_discrepancy_ratio"] = np.log2(df["endpoint_discrepancy_ratio"].to_numpy(float))
    df = df.sort_values(
        ["case_order", "representation_order", "gene"], kind="mergesort"
    ).reset_index(drop=True)

    expected_n = {
        ("FeRhoNox", "Biohub_OPS0047", "phase172"): 992,
        ("FeRhoNox", "Biohub_OPS0047", "raw_thumbnail"): 1000,
        ("FeRhoNox", "Biohub_OPS0067", "phase172"): 998,
        ("FeRhoNox", "Biohub_OPS0067", "raw_thumbnail"): 1000,
        ("pRb", "Biohub_OPS0077", "phase172"): 1000,
        ("pRb", "Biohub_OPS0077", "raw_thumbnail"): 1000,
    }
    observed_n = df.groupby(["target", "screen", "representation"]).size().to_dict()
    if observed_n != expected_n:
        raise ValueError(f"Frozen per-case counts changed: {observed_n}")

    # Validate the provided summary against direct quantiles, without changing
    # the frozen scientific statistic used in the figure.
    direct = (
        df.groupby(["target", "screen", "representation"], sort=False)
        ["endpoint_discrepancy_ratio"]
        .agg(
            n_genes="size",
            median_endpoint_discrepancy_ratio="median",
            q25_endpoint_discrepancy_ratio=lambda x: x.quantile(0.25),
            q75_endpoint_discrepancy_ratio=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    merged = summary.merge(
        direct,
        on=["target", "screen", "representation"],
        suffixes=("_frozen", "_direct"),
        validate="one_to_one",
    )
    for field in [
        "n_genes",
        "median_endpoint_discrepancy_ratio",
        "q25_endpoint_discrepancy_ratio",
        "q75_endpoint_discrepancy_ratio",
    ]:
        left = merged[f"{field}_frozen"].to_numpy(float)
        right = merged[f"{field}_direct"].to_numpy(float)
        if not np.allclose(left, right, rtol=0, atol=1e-12):
            raise ValueError(f"Frozen summary mismatch for {field}")
    return df, summary


def _density(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a KDE in log2-ratio space for a log-scaled display axis."""
    log_values = np.log2(values)
    low = max(np.log2(0.30), float(np.quantile(log_values, 0.001)) - 0.08)
    high = min(np.log2(4.00), float(np.quantile(log_values, 0.999)) + 0.08)
    log_grid = np.linspace(low, high, 320)
    density = gaussian_kde(log_values, bw_method="scott")(log_grid)
    density = density / max(float(density.max()), np.finfo(float).eps)
    return 2.0**log_grid, density


def _draw_half_raincloud(
    ax: plt.Axes,
    values: np.ndarray,
    y: float,
    direction: float,
    color: str,
    seed: int,
) -> None:
    """Draw one complete distribution without pairing thousands of genes."""
    xgrid, density = _density(values)
    height = 0.265 * density
    ax.fill_between(
        xgrid,
        y,
        y + direction * height,
        color=color,
        alpha=0.23,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        xgrid,
        y + direction * height,
        color=color,
        lw=0.75,
        alpha=0.95,
        zorder=2,
    )

    # Full gene-level rain: each point is one gene.  Deterministic vertical
    # jitter avoids a solid rug while preserving the inferential unit.
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(0.018, 0.070, len(values))
    ax.scatter(
        values,
        y + direction * jitter,
        s=2.0,
        color=color,
        alpha=0.115,
        edgecolors="none",
        rasterized=True,
        zorder=3,
    )

    q10, q25, median, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    summary_y = y + direction * 0.125
    ax.plot([q10, q90], [summary_y, summary_y], color=color, lw=0.65, alpha=0.82, zorder=4)
    ax.plot([q25, q75], [summary_y, summary_y], color=color, lw=2.0, solid_capstyle="round", zorder=5)
    ax.scatter(
        [median],
        [summary_y],
        s=21,
        facecolor="white",
        edgecolor=color,
        lw=0.9,
        zorder=6,
    )


def _draw(container: Any, spec: Any, df: pd.DataFrame) -> plt.Axes:
    ax = container.add_subplot(spec)
    y_positions = [2.0, 1.0, 0.0]
    for case_i, case in enumerate(CASE_ORDER):
        y = y_positions[case_i]
        for rep_i, representation in enumerate(REP_ORDER):
            values = df.loc[
                df["target"].eq(case[0])
                & df["screen"].eq(case[1])
                & df["representation"].eq(representation),
                "endpoint_discrepancy_ratio",
            ].to_numpy(float)
            _draw_half_raincloud(
                ax,
                values,
                y,
                1.0 if representation == "phase172" else -1.0,
                REP_COLORS[representation],
                seed=20260860 + 10 * case_i + rep_i,
            )
        ax.axhline(y, color=COLORS["grid"], lw=0.55, zorder=0)

    ax.axvline(1.0, color=COLORS["ink"], lw=0.75, ls=(0, (3, 2)), zorder=0)
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.30, 4.00)
    ax.set_xticks([0.5, 1.0, 2.0, 4.0], labels=["0.5", "1", "2", "4"])
    ax.set_ylim(-0.39, 2.39)
    ax.set_yticks(y_positions, [CASE_LABELS[c] for c in CASE_ORDER])
    ax.set_xlabel("Nearest / random target discrepancy")
    ax.grid(axis="x", color=COLORS["grid"], lw=0.5, zorder=0)
    clean_axis(ax)
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.text(
        1.04,
        1.50,
        "Random match",
        ha="left",
        va="center",
        fontsize=5.0,
        color=COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 0.45},
        zorder=8,
    )

    legend_handles = [
        Line2D([0], [0], color=REP_COLORS[rep], lw=2.1, label=REP_LABELS[rep])
        for rep in REP_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.55, 1.14),
        frameon=False,
        ncol=1,
        columnspacing=1.2,
        handlelength=1.4,
        handletextpad=0.45,
        borderaxespad=0,
    )
    return ax


def draw_panel_e(
    container: Any,
    spec: Any | None = None,
    data: pd.DataFrame | None = None,
    *,
    add_letter: bool = False,
) -> list[plt.Axes]:
    """Draw panel e into a container or an explicit Figure/SubplotSpec pair."""
    if data is None:
        data, _ = prepare_panel_e()
    if spec is None:
        spec = container.add_gridspec(
            1,
            1,
            left=0.19,
            right=0.985,
            top=0.85,
            bottom=0.18,
        )[0, 0]
    ax = _draw(container, spec, data)
    if add_letter:
        container.text(
            0.002,
            0.985,
            "e",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=COLORS["ink"],
        )
    return [ax]


def _write_bundle(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    (PANEL_ROOT / "source_data").mkdir(parents=True, exist_ok=True)
    plotted = PANEL_ROOT / "source_data/figure5e_plotted_source.csv"
    plotted_summary = PANEL_ROOT / "source_data/figure5e_summary.csv"
    df.to_csv(plotted, index=False)
    summary.to_csv(plotted_summary, index=False)

    (PANEL_ROOT / "audit_report.md").write_text(
        """# Figure 5e audit report

- Statistical unit: gene within a frozen reporter--screen case and neighbour representation.
- Coverage: 5,990 rows across three cases and two representations; per-group n is 992--1,000.
- Exclusions: rows absent from the upstream cross-guide constrained diagnostic are not imputed (8 and 2 missing phase172 gene rows in OPS0047 and OPS0067, respectively).
- Display statistic: endpoint discrepancy for the nearest morphology match divided by the median discrepancy for a random match; 1 is the no-advantage reference.
- Pairing: the panel shows marginal gene-level distributions and does not connect genes across representations.
- Transform: KDEs are evaluated in log2-ratio space and displayed on a base-2 logarithmic axis. Medians and quantiles are computed on the original ratio scale.
- Leakage check: source rows are restricted to the frozen cross-guide diagnostic; no model fitting or label-informed re-selection occurs in this plotting step.
- Images: none; microscopy evidence is reserved for panels b and d.
""",
        encoding="utf-8",
    )
    (PANEL_ROOT / "figure_methods.md").write_text(
        "# Figure 5e methods\n\nFor each gene, cells were matched to the nearest morphology neighbour either in the engineered 172-dimensional phase representation or in a 32-dimensional principal-component representation of raw phase thumbnails under the frozen cross-guide diagnostic. Target-state discrepancy for the nearest match was divided by the median discrepancy obtained from random matches. Ratios below one indicate that morphologically similar cells are more similar in the targeted state than random cells; ratios at or above one indicate unresolved conditional ambiguity. All 5,990 available gene-level ratios are represented. Half-density envelopes were estimated in log2-ratio space; points show individual genes, thick intervals show the interquartile range, thin intervals show the 10th--90th percentiles, and open circles show medians.\n\n## Analysis specification\n\n- Statistical unit: gene within a frozen reporter--screen case and neighbour representation.\n- Coverage: 5,990 rows across three cases and two representations; per-group n is 992--1,000.\n- Exclusions: rows absent from the upstream cross-guide constrained diagnostic are not imputed (8 and 2 missing phase172 gene rows in OPS0047 and OPS0067, respectively).\n- Display statistic: endpoint discrepancy for the nearest morphology match divided by the median discrepancy for a random match; 1 is the no-advantage reference.\n- Pairing: the panel shows marginal gene-level distributions and does not connect genes across representations.\n- Transform: KDEs are evaluated in log2-ratio space and displayed on a base-2 logarithmic axis. Medians and quantiles are computed on the original ratio scale.\n- Leakage check: source rows are restricted to the frozen cross-guide diagnostic; no model fitting or label-informed re-selection occurs in this plotting step.\n- Images: none; microscopy evidence is reserved for panels b and d.\n",
        encoding="utf-8",
    )
    (PANEL_ROOT / "figure_legend.md").write_text(
        """**e, Conditional ambiguity among morphologically matched cells.** Distributions show the target-state discrepancy of the nearest morphology match relative to random matches for two FeRhoNox screens and one pRb screen. Teal upper half-rainclouds use the engineered 172-dimensional phase representation; orange lower half-rainclouds use a 32-dimensional principal-component representation of raw phase thumbnails. Points denote genes, open circles denote medians, thick bars denote interquartile ranges, thin bars denote 10th--90th percentiles and the vertical dashed line marks a ratio of one. Values near or above one show that similar morphology does not uniquely determine the targeted state.
""",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "claim": "Morphologically matched cells can retain substantial targeted-state ambiguity.",
                "can_say": "In the frozen cross-guide diagnostic, phase-nearest discrepancy ratios vary by reporter and screen and can approach or exceed the random-match reference.",
                "cannot_say": "The panel does not establish that phase contains no targeted-state information, nor does it provide a raw-image upper bound.",
                "positive_interpretation": "Ratios below one indicate conditional agreement beyond random matching.",
                "negative_interpretation": "Ratios near or above one identify conditional ambiguity at the available representation and sampling depth.",
            }
        ]
    ).to_csv(PANEL_ROOT / "claim_table.csv", index=False)
    (PANEL_ROOT / "plot_spec.yaml").write_text(
        """panel: e
title: conditional ambiguity
statistical_unit: gene within reporter-screen and representation
source_rows: 5990
cases:
  - FeRhoNox / OPS0047
  - FeRhoNox / OPS0067
  - pRb / OPS0077
representations:
  phase172: teal upper half-raincloud
  raw_thumbnail: orange lower half-raincloud
x_scale: base-2 logarithmic ratio
reference: 1
summary_marks:
  open_circle: median
  thick_bar: interquartile range
  thin_bar: 10th-90th percentiles
font: Arial
exports: [png_600dpi, pdf_vector, svg_live_text]
""",
        encoding="utf-8",
    )

    medians = (
        df.groupby(["target", "screen", "representation"], sort=False)
        ["endpoint_discrepancy_ratio"]
        .median()
        .to_dict()
    )
    qa = {
        "status": "PASS",
        "n_rows": int(len(df)),
        "n_cases": int(df[["target", "screen"]].drop_duplicates().shape[0]),
        "n_representations": int(df["representation"].nunique()),
        "per_group_n": {
            " | ".join(k): int(v)
            for k, v in df.groupby(["target", "screen", "representation"]).size().to_dict().items()
        },
        "medians": {" | ".join(k): float(v) for k, v in medians.items()},
        "reference_ratio": 1.0,
        "font": "Arial",
        "png_dpi": 600,
        "svg_text_mode": "live",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    }
    (PANEL_ROOT / "figure_qa_report.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    apply_style()
    df, summary = prepare_panel_e()
    _write_bundle(df, summary)
    fig = plt.figure(figsize=(62 / 25.4, 62 / 25.4))
    draw_panel_e(fig, data=df, add_letter=True)
    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)


if __name__ == "__main__":
    main()
