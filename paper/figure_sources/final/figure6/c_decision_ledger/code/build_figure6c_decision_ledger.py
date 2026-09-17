#!/usr/bin/env python3
"""Build the Figure 6c reporter decision ledger from frozen canonical tables.

The panel is descriptive at the reporter level (n=52). It combines two distinct
quantities without mixing their colour scales: retained scientific utility after
counterfactual response replacement, and signed distance to the five frozen decision gates.
The right inset summarizes already-frozen threshold-sensitivity scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
PANEL = HERE.parents[1]
FIGURE6_ROOT = HERE.parents[2]
PAPER = FIGURE6_ROOT
if str(FIGURE6_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE6_ROOT))
from _portable_fonts import configure_sans, format_figure, layout_report  # noqa: E402

OUT_FIG = PANEL / "figure"
OUT_DATA = PANEL / "source_data"
OUT_QA = PANEL / "qa"
LEDGER_SOURCE = OUT_DATA / "figure6c_decision_ledger_source.csv"
TRANSITION_SOURCE = OUT_DATA / "figure6c_threshold_transition_matrix.csv"

INK = "#202A33"
MUTED = "#6F7E88"
GRID = "#DCE3E7"
PALE = "#F4F6F7"
WHITE = "#FFFFFF"
TEAL = "#3E6F9C"
TEAL_DARK = "#294A66"
RED = "#D56C75"
GOLD = "#DDA06A"

TIER_ORDER = [
    "quantitative_proxy",
    "ranking_proxy",
    "measurement_required",
    "not_identifiable",
    "unresolved",
]
TIER_LABELS = {
    "quantitative_proxy": "Quantitative",
    "ranking_proxy": "Ranking",
    "measurement_required": "Required",
    "not_identifiable": "Not identifiable",
    "unresolved": "Unresolved",
}
TIER_COLORS = {
    "quantitative_proxy": "#3E6F9C",
    "ranking_proxy": "#DDA06A",
    "measurement_required": "#D56C75",
    "not_identifiable": "#7E8D96",
    "unresolved": "#C8D0D4",
}

BIOLOGY_ORDER = [
    "Endosome", "Lysosome", "ER/Golgi", "Cytoskeleton", "Peroxisome",
    "Stress/PQC", "Mitochondria", "Plasma Membrane", "Nucleus", "Signaling",
]
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

UTILITY = [
    ("magnitude_spearman", "KO rank"),
    ("top5pct_recall", "Top-5% hits"),
    ("go_bp_top10_jaccard", "GO BP"),
    ("go_cc_top10_jaccard", "GO CC"),
    ("ebi_complex_top10_jaccard", "Protein complexes"),
]
GATES = ["Recovery", "Rank", "Amplitude", "Top-5%", "Reliability"]

CASE_LABELS = {
    "stress_granule_g3bp1": "G3BP1",
    "nuclei_nucleolive_live_cell_dye": "NucleoLIVE",
    "nuclear_speckles_srrm2": "SRRM2",
    "p53": "p53",
    "prb": "pRb",
}

DOMAIN_ONLY_LABELS = [
    ("autophagosome_map1lc3b", "MAP1LC3B", -0.2, 0.58, "right"),
    ("er_golgi_cope", "COPE", 0.0, 0.58, "center"),
]


def configure() -> None:
    configure_sans(mpl, font_manager, PAPER)
    mpl.rcParams.update({
        "font.size": 6.2,
        "axes.labelsize": 6.0,
        "xtick.labelsize": 5.2,
        "ytick.labelsize": 5.4,
        "axes.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    })


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the deposited, panel-local canonical ledger and transition table.

    The ledger is the frozen merge of utility, gate-margin and threshold-stability
    inputs.  Keeping that merge beside this builder makes Figure 6c independently
    reproducible without resolving historical Figure 6 paths.
    """
    order = pd.read_csv(LEDGER_SOURCE).sort_values("display_order", kind="stable").reset_index(drop=True)
    transitions = pd.read_csv(TRANSITION_SOURCE)
    required = {
        "reporter_slug", "display_order", "substitutability_tier", "baseline",
        "scientific_utility_index", "fraction_of_all_scenarios_unchanged",
        "domain_sensitive", *[c for c, _ in UTILITY],
        *[f"gate_margin__{gate}" for gate in GATES],
        *[f"gate_value__{gate}" for gate in GATES],
        *[f"gate_pass__{gate}" for gate in GATES],
    }
    missing = sorted(required.difference(order.columns))
    if missing:
        raise RuntimeError(f"Panel-local decision ledger is missing columns: {missing}")
    if len(order) != 52 or order.reporter_slug.nunique() != 52:
        raise RuntimeError("Panel-local Figure 6 reporter coverage is incomplete")
    if transitions.shape != (5, 6) or set(transitions.columns) != {"baseline", *TIER_ORDER}:
        raise RuntimeError("Panel-local threshold-transition table is incomplete")

    expected_counts = {"quantitative_proxy": 30, "ranking_proxy": 7,
                       "measurement_required": 3, "not_identifiable": 9,
                       "unresolved": 3}
    observed = order.substitutability_tier.value_counts().to_dict()
    if observed != expected_counts:
        raise RuntimeError(f"Tier counts changed: {observed}")
    if not order["baseline"].eq(order["substitutability_tier"]).all():
        raise RuntimeError("Threshold-sensitivity baseline disagrees with final tier")

    return order, pd.DataFrame(), transitions


def _matrix_axis(fig, rect, matrix, *, cmap, norm, ylabels, separators):
    ax = fig.add_axes(rect)
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm,
              rasterized=True)
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_xticks([])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, color=INK)
    ax.tick_params(axis="y", length=0, pad=4)
    for pos in separators:
        ax.axvline(pos - 0.5, color=WHITE, lw=1.3, zorder=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def draw_panel_c(container, add_letter: bool = False) -> pd.DataFrame:
    """Draw panel c natively into a Figure or SubFigure.

    The function deliberately owns no canvas size and adds no panel letter by
    default.  This lets the manuscript compositor preserve live text and vector
    marks instead of pasting a pre-rendered panel image.
    """
    d, margins_long, transitions = load_tables()
    n = len(d)
    x = np.arange(n)
    counts = d.substitutability_tier.value_counts().reindex(TIER_ORDER)
    bounds = counts.cumsum().tolist()[:-1]

    utility_matrix = d[[c for c, _ in UTILITY]].to_numpy(float).T
    gate_matrix = d[[f"gate_margin__{c}" for c in GATES]].to_numpy(float).T

    util_cmap = LinearSegmentedColormap.from_list(
        "utility", ["#F4F6F7", "#D7E5EF", "#79A8C9", "#3E6F9C", "#294A66"]
    )
    gate_cmap = LinearSegmentedColormap.from_list(
        "gate", ["#D56C75", "#E5A0A8", "#F7F8F8", "#A9C6DA", "#3E6F9C"]
    )
    stability_cmap = LinearSegmentedColormap.from_list(
        "stability", ["#F3E5D5", "#DDA06A", "#9B6E45"]
    )

    fig = container
    if add_letter:
        fig.text(0.012, 0.975, "c", ha="left", va="top", fontsize=9.0,
                 fontweight="bold", color=INK)

    left, width = 0.140, 0.600
    # Keep the threshold inset visually distinct from the two matrix colourbars;
    # their labels otherwise compete at final manuscript size.
    right_x, right_w = 0.850, 0.140

    # Tier and biological annotation strips.
    ax_tier = fig.add_axes([left, 0.865, width, 0.042])
    start = 0
    for tier in TIER_ORDER:
        count = int(counts[tier])
        ax_tier.add_patch(Rectangle((start - 0.5, 0), count, 1,
                                    facecolor=TIER_COLORS[tier], edgecolor="none"))
        label = TIER_LABELS[tier]
        if tier == 'not_identifiable':
            label = 'Not ID'
        if count <= 3:
            label = {"measurement_required": "Req.", "unresolved": "Unres."}.get(tier, label)
        # Stack the count below each tier name so narrow neighbouring classes
        # remain distinct at the 183-mm composite width.
        ax_tier.text(start + (count - 1) / 2, 1.15, f"{label}\n{count}",
                     ha="center", va="bottom", fontsize=5.4, fontweight="bold",
                     linespacing=0.88, color=INK, clip_on=False)
        start += count
    ax_tier.set_xlim(-0.5, n - 0.5); ax_tier.set_ylim(0, 1)
    ax_tier.set_xticks([]); ax_tier.set_yticks([])
    for s in ax_tier.spines.values(): s.set_visible(False)

    ax_bio = fig.add_axes([left, 0.820, width, 0.022])
    bio_rgba = np.array([[mpl.colors.to_rgba(BIOLOGY_COLORS[v]) for v in d.biological_category]])
    ax_bio.imshow(bio_rgba, aspect="auto", interpolation="nearest")
    ax_bio.set_xticks([]); ax_bio.set_yticks([])
    for pos in bounds: ax_bio.axvline(pos - 0.5, color=WHITE, lw=1.3)
    for s in ax_bio.spines.values(): s.set_visible(False)
    fig.text(left - 0.010, 0.831, "Biological system", ha="right", va="center",
             fontsize=5.5, color=INK)

    # Two analytically distinct blocks with their own scales.
    util_ax = _matrix_axis(
        fig, [left, 0.532, width, 0.245], utility_matrix,
        cmap=util_cmap, norm=Normalize(0, 1),
        ylabels=[lab for _, lab in UTILITY], separators=bounds,
    )
    gate_ax = _matrix_axis(
        fig, [left, 0.245, width, 0.245], gate_matrix,
        cmap=gate_cmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        ylabels=GATES, separators=bounds,
    )
    fig.text(left, 0.777, "Utility retained after response replacement", ha="left", va="bottom",
             fontsize=6.6, fontweight="bold", color=INK)
    fig.text(left, 0.491, "Margin to the frozen decision gate", ha="left", va="bottom",
             fontsize=6.6, fontweight="bold", color=INK)

    # Threshold-stability and domain-sensitivity tracks.
    ax_stab = fig.add_axes([left, 0.170, width, 0.036])
    stability = d.fraction_of_all_scenarios_unchanged.to_numpy(float)[None, :]
    ax_stab.imshow(stability, aspect="auto", interpolation="nearest",
                   cmap=stability_cmap, norm=Normalize(0.75, 1.0), rasterized=True)
    for pos in bounds: ax_stab.axvline(pos - 0.5, color=WHITE, lw=1.3)
    domain = np.flatnonzero(d.domain_sensitive.to_numpy(bool))
    ax_stab.scatter(domain, np.zeros_like(domain), marker="D", s=12, facecolor=WHITE,
                    edgecolor=INK, linewidth=0.65, zorder=5, clip_on=False)
    ax_stab.set_xlim(-0.5, n - 0.5); ax_stab.set_ylim(-0.5, 0.5)
    ax_stab.set_xticks([]); ax_stab.set_yticks([])
    for s in ax_stab.spines.values(): s.set_visible(False)
    fig.text(left - 0.010, 0.188, "Tier stability", ha="right", va="center",
             fontsize=5.5, color=INK)

    # The five pre-registered cases and the two additional domain-sensitive
    # reporters share one label rail.  NucleoLIVE belongs to both sets and is
    # therefore labelled only once.
    ax_names = fig.add_axes([left, 0.090, width, 0.060])
    ax_names.set_xlim(-0.5, n - 0.5); ax_names.set_ylim(0, 1); ax_names.axis("off")
    for slug, label, dx, text_y, ha in DOMAIN_ONLY_LABELS:
        row = d.loc[d.reporter_slug.eq(slug)]
        if row.empty:
            raise RuntimeError(f"Missing domain-sensitive reporter {slug}")
        xi = int(row.index[0])
        color = TIER_COLORS[row.iloc[0].substitutability_tier]
        ax_names.annotate(
            label, xy=(xi, 1.00), xytext=(xi + dx, text_y),
            ha=ha, va="top", fontsize=5.2, fontweight="bold", color=color,
            arrowprops={"arrowstyle": "-", "lw": 0.45, "color": INK,
                        "shrinkA": 1.0, "shrinkB": 3.5},
            clip_on=False,
        )
    for slug, label in CASE_LABELS.items():
        row = d.loc[d.reporter_slug.eq(slug)]
        if row.empty: raise RuntimeError(f"Missing registered case {slug}")
        xi = int(row.index[0])
        color = TIER_COLORS[row.iloc[0].substitutability_tier]
        if slug == "nuclei_nucleolive_live_cell_dye":
            text_x, text_y, ha = xi + 0.2, 0.58, "left"
        else:
            text_x, text_y, ha = xi, 0.58, "center"
        ax_names.annotate(label, xy=(xi, 1.00), xytext=(text_x, text_y),
                          ha=ha, va="top", fontsize=5.2, fontweight="bold",
                          color=color, arrowprops=dict(arrowstyle="-", color=INK,
                          lw=0.45, shrinkA=1, shrinkB=3.5), clip_on=False)

    # Colourbars and small legend are kept adjacent to the encoded blocks.
    # Leave a clean gutter between the matrix edge (x=0.760) and both colour
    # scales so their short titles do not intrude into the final matrix column.
    colourbar_x = 0.756
    cax_u = fig.add_axes([colourbar_x, 0.557, 0.009, 0.190])
    cb_u = fig.colorbar(mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=util_cmap), cax=cax_u)
    cb_u.set_ticks([0, 0.5, 1]); cb_u.ax.tick_params(length=2, labelsize=5.0)
    cb_u.outline.set_visible(False)
    cax_u.set_title("utility", x=0.50, ha="center", fontsize=5.0,
                    color=INK, pad=9.0)
    cax_g = fig.add_axes([colourbar_x, 0.270, 0.009, 0.145])
    cb_g = fig.colorbar(
        mpl.cm.ScalarMappable(
            norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), cmap=gate_cmap
        ),
        cax=cax_g,
    )
    cb_g.set_ticks([-1, 0, 1]); cb_g.set_ticklabels(["below", "gate", "above"])
    cb_g.ax.tick_params(length=2, labelsize=5.0)
    cb_g.outline.set_visible(False)
    cax_g.set_title("margin", x=0.50, ha="center", fontsize=5.0,
                    color=INK, pad=9.0)

    # Right inset: row-normalized tier transitions across 36 frozen sensitivity scenarios.
    trans = transitions.set_index("baseline").reindex(TIER_ORDER)[TIER_ORDER].astype(float)
    trans_prop = trans.div(trans.sum(axis=1), axis=0)
    ax_tr = fig.add_axes([right_x, 0.505, right_w, 0.300])
    # Only 25 cells: use the same exact vector grid for fills and outlines.
    # A rasterized image in a translated SubFigure can be offset from vector
    # rectangles after mixed-mode PDF export, despite matching data coordinates.
    edges = np.arange(6, dtype=float) - 0.5
    ax_tr.pcolormesh(edges, edges, trans_prop.to_numpy(), cmap=util_cmap,
                     norm=Normalize(0, 1), shading="flat", edgecolors="none",
                     antialiased=False, rasterized=False)
    ax_tr.set_xlim(edges[0], edges[-1])
    ax_tr.set_ylim(edges[-1], edges[0])
    ax_tr.set_aspect("equal")
    ax_tr.set_gid("threshold_transition_matrix")
    ax_tr.set_xticks(range(5)); ax_tr.set_yticks(range(5))
    short = ["Q", "R", "M", "N", "U"]
    ax_tr.set_xticklabels(short, fontsize=6.0)
    ax_tr.set_yticklabels(short, fontsize=5.0)
    ax_tr.tick_params(length=0, pad=2)
    for i in range(5):
        ax_tr.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                  edgecolor=INK, linewidth=0.75, clip_on=False))
    for s in ax_tr.spines.values(): s.set_visible(False)
    fig.text(right_x, 0.842, "Threshold sensitivity", ha="left", va="bottom",
             fontsize=6.6, fontweight="bold", color=INK)
    fig.text(right_x, 0.810, "Baseline → perturbed", ha="left", va="bottom",
             fontsize=5.0, color=INK)

    ax_summary = fig.add_axes([right_x, 0.175, right_w, 0.190])
    ax_summary.set_xlim(0, 1); ax_summary.set_ylim(-0.5, 2.5); ax_summary.axis("off")
    summary = [
        ("Exact tier stable", int(d.exact_tier_unchanged_in_every_scenario.sum()), 52, TEAL),
        ("Proxy status stable", int(d.proxy_status_unchanged_in_every_scenario.sum()), 52, TEAL_DARK),
        ("Domain-sensitive", int(d.domain_sensitive.sum()), 52, GOLD),
    ]
    for yi, (label, passed, total, color) in zip([2, 1, 0], summary):
        frac = passed / total
        ax_summary.add_patch(Rectangle((0, yi - 0.14), 1, 0.24, facecolor=PALE, edgecolor="none"))
        ax_summary.add_patch(Rectangle((0, yi - 0.14), frac, 0.24, facecolor=color, edgecolor="none"))
        label_x = 0.10 if label == "Domain-sensitive" else 0
        if label == "Domain-sensitive":
            ax_summary.scatter([0.035], [yi + 0.26], marker="D", s=11,
                               facecolor=WHITE, edgecolor=INK, linewidth=.6,
                               zorder=3, clip_on=False)
        short_label = {'Exact tier stable': 'Exact tier', 'Proxy status stable': 'Proxy',
                       'Domain-sensitive': 'Domain'}[label]
        ax_summary.text(label_x, yi + 0.18, short_label, ha="left", va="bottom", fontsize=6.0, color=INK)
        ax_summary.text(1, yi + 0.18, f"{passed}/{total}", ha="right", va="bottom",
                        fontsize=5.2, color=INK, fontweight="bold")

    # Exportable canonical table: one row per reporter, every displayed value.
    source = d.copy()
    if "panel" not in source.columns:
        source.insert(0, "panel", "c")
    else:
        source["panel"] = "c"
        source = source.loc[:, ["panel", *[c for c in source.columns if c != "panel"]]]
    return source


def build() -> tuple[plt.Figure, pd.DataFrame]:
    """Build the validated standalone 183 x 96 mm panel."""
    configure()
    fig = plt.figure(figsize=(183 / 25.4, 96 / 25.4), facecolor=WHITE)
    source = draw_panel_c(fig, add_letter=True)
    return fig, source


def write_support(source: pd.DataFrame) -> None:
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_QA.mkdir(parents=True, exist_ok=True)
    # The ledger is a frozen input, not a rendering output. Do not rewrite its
    # floating-point serialization after loading it for this display.
    (PANEL / "figure_methods.md").write_text(
        """# Figure 6c methods

The statistical unit is the reporter (`n=52`). Reporter order and final tiers are
frozen. The upper matrix displays five counterfactual response-replacement utility metrics
on their native 0–1 scale. The lower matrix displays signed scaled margins to five
frozen decision criteria; negative values are below and positive values are above the
criterion. The threshold-sensitivity inset is the row-normalized 5×5 transition table
from 36 precomputed threshold perturbation scenarios. Its cells and diagonal
outlines share exact vector coordinates; no raster layer can shift relative to
the outlines. No model fitting, threshold
estimation, clustering or reporter selection occurs in the figure builder.
""",
        encoding="utf-8",
    )
    (PANEL / "figure_legend.md").write_text(
        """**c,** Reporter-level measurement-decision ledger. The upper fingerprint
shows KO-ranking, top-hit and functional-program utility retained after
counterfactual response replacement. The lower fingerprint shows each reporter's signed
margin to the frozen recovery, rank, amplitude, top-hit and reliability criteria.
Reporters are ordered by the five final decision tiers; the narrow annotation strip
denotes biological system. The lower track reports the fraction of 36
threshold-sensitivity scenarios that preserve the exact tier, with diamonds
identifying domain-sensitive reporters. The inset summarizes tier transitions under
those scenarios.
""",
        encoding="utf-8",
    )
    plot_spec = {
        "question": "Which reporters support quantitative or ranking replacement, and which frozen criterion limits the remaining reporters?",
        "claim": "Utility retention varies across reporters; fidelity and reliability criteria define response-level tiers, with threshold stability shown separately.",
        "one_row_represents": "one OPS reporter",
        "analysis_mode": "descriptive_enumeration",
        "inferential_unit": "reporter (n=52)",
        "pairing": "shared reporter order across all tracks",
        "selected_card": "C11 decision-gate matrix + C05 complex heatmap",
        "transformations": ["frozen scaled gate margins", "row-normalized frozen transition counts"],
        "uncertainty": "none; threshold sensitivity is displayed descriptively",
        "destination": "main",
        "final_width_mm": 183,
        "final_height_mm": 96,
        "transition_rendering": "25 vector cells and diagonal outlines share the same half-integer boundaries",
    }
    (PANEL / "figure_plot_spec.json").write_text(json.dumps(plot_spec, indent=2), encoding="utf-8")


def main() -> int:
    fig, source = build()
    write_support(source)
    stem = OUT_FIG / "figure6c_decision_ledger"
    format_figure(fig, 6)
    qa_root = next(p for p in Path(__file__).resolve().parents if p.name == "figure6") / "qa"
    layout_report(fig, qa_root / (stem.name + "_typography.json"))
    for suffix, kwargs in [("png", {"dpi": 600}), ("pdf", {}), ("svg", {})]:
        fig.savefig(stem.with_suffix(f".{suffix}"), facecolor=WHITE,
                    bbox_inches=None, pad_inches=0, **kwargs)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    wpx, hpx = fig.canvas.get_width_height()
    overflow = []
    for text in fig.findobj(mpl.text.Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > wpx + 1 or box.y1 > hpx + 1:
            overflow.append(text.get_text())
    qa = {
        "status": "PASS" if not overflow else "FAIL",
        "canvas_mm": [183, 96],
        "reporters": int(len(source)),
        "tier_counts": source.substitutability_tier.value_counts().to_dict(),
        "texts_outside_canvas": overflow,
        "font": "Arial TrueType",
        "vector_text": True,
        "source_rows": int(len(source)),
    }
    (OUT_QA / "figure6c_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    plt.close(fig)
    if overflow:
        raise RuntimeError(f"Text outside canvas: {overflow}")
    print(stem.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
