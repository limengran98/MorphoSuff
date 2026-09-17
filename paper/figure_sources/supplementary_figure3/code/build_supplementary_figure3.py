#!/usr/bin/env python3
"""Render predictor-conditional tiers from the one canonical Supplementary Data 1.

No fitting, statistical inference, threshold selection or metric recomputation is
performed here. Clean standalone panels and the labelled composite share geometry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parents[1]
DATA = PAPER / "supplementary_data/SupplementaryData1"
sys.path.insert(0, str(ROOT.parent))
from publication_style import FONT, LETTER_PT, format_figure, layout_report

WIDTH = 183.0
HEIGHT = 162.0
INK = "#202A33"
TIERS = ["quantitative_proxy", "ranking_proxy", "measurement_required", "not_identifiable", "unresolved"]
LETTERS = dict(zip(TIERS, "QRMNU"))
COLORS = dict(zip(TIERS, ["#3E6F9C", "#DDA06A", "#D56C75", "#7E8D96", "#C8D0D4"]))
STATE = "checkpoint_average"
REFERENCE = "coherent_median_ensemble"
SHORT = {
    "lysosome_lysotracker_live-cell_dye": "LysoTracker",
    "lipid_droplet_bodipy_live_cell_dye": "BODIPY",
    "actin_filament_fastact_spy555_live_cell_dye": "FastAct",
    "fe2+_ferhonox_live-cell_dye": "FeRhoNox",
    "nuclei_nucleolive_live_cell_dye": "NucleoLIVE",
    "mitochondria_chromalive_561_excitation": "ChromaLIVE 561",
    "peroxisome_peroxi_spy650_live_cell_dye": "Peroxi",
    "chromalive_488_excitation": "ChromaLIVE 488",
    "endocytic_vesicle_ph_phrodo-dextran_live_cell_dye": "pHrodo-dextran",
    "oxidative_stress_cellrox_live-cell_dye": "CellROX",
    "caspase_activity_cellevent-caspase_live-cell_dye": "CellEvent-Caspase",
    "b-catenin": "β-catenin",
}


def configure():
    mpl.rcParams.update({
        "font.family": FONT, "font.size": 6.3, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": .65, "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.major.size": 2.3, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "svg.hashsalt": "SupplementaryFigure3",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "legend.frameon": False,
    })


def text_mm(fig, x, y, value, **kwargs):
    w, h = fig.get_size_inches() * 25.4
    return fig.text(x/w, y/h, value, **kwargs)


def ax_mm(fig, x, y, w, h):
    fw, fh = fig.get_size_inches() * 25.4
    return fig.add_axes([x/fw, y/fh, w/fw, h/fh])


def load():
    evidence = pd.read_csv(DATA / "evidence.csv")
    assert len(evidence) == 1144
    assert not evidence.duplicated(["run_state", "reporter_slug", "method_id"]).any()
    evidence = evidence[evidence.run_state.eq(STATE)].copy()
    order = pd.read_csv(DATA / "reporter_order.csv").sort_values("display_order")
    roster = pd.read_csv(DATA / "method_roster.csv").sort_values("method_order")
    assert len(order) == 52 and len(roster) == 10
    assert order.reliability.lt(.3).sum() == 9
    return evidence, order, roster


def draw_a(fig, y_base, evidence, order, roster):
    labels = ["Q  Quantitative", "R  Ranking", "M  Measurement required", "N  Not identifiable", "U  Unresolved"]
    # One legend, shared by both reporter blocks and the case-study points.
    fig.legend([Patch(facecolor=COLORS[t], edgecolor="none") for t in TIERS], labels,
               loc="center", bbox_to_anchor=(.52, (y_base + 106)/(fig.get_figheight()*25.4)),
               ncol=5, handlelength=1.0, handleheight=.9, handletextpad=.4,
               columnspacing=1.1, borderaxespad=0, fontsize=6.3)
    methods = roster.method_id.tolist() + [REFERENCE]
    method_names = roster.method.tolist() + ["Ensemble"]
    names = dict(zip(order.reporter_slug, order.short_name))
    keys = []
    indexed = evidence.set_index(["reporter_slug", "method_id"])
    for block_index in range(2):
        reporters = order.iloc[block_index*26:(block_index+1)*26]
        ax = ax_mm(fig, 30.0 + 92.0*block_index, y_base+10.0, 58.6, 78.0)
        xs = np.r_[np.arange(10), 10.7]
        ax.set_xlim(-.5, 11.2)
        ax.set_ylim(25.5, -.5)
        for row_no, item in enumerate(reporters.itertuples(index=False)):
            for x, method in zip(xs, methods):
                row = indexed.loc[(item.reporter_slug, method)]
                category = row.tier
                ax.add_patch(Rectangle((x-.46, row_no-.44), .92, .88,
                                       facecolor=COLORS[category], edgecolor="none"))
                ax.text(x, row_no, LETTERS[category], va="center", ha="center", fontsize=6.0,
                        color="white" if category == "quantitative_proxy" else INK)
                keys.append(dict(panel="a", block=block_index+1, display_row=row_no+1,
                                 run_state=STATE, reporter_slug=item.reporter_slug, method_id=method))
        ticklabels = []
        for row in reporters.itertuples(index=False):
            name = SHORT.get(row.reporter_slug, str(names[row.reporter_slug]).strip().rstrip("*"))
            ticklabels.append(name + (" †" if row.reliability < .3 else ""))
        ax.set_yticks(np.arange(26), ticklabels, fontsize=6.3)
        ax.set_xticks(xs, method_names, rotation=90, ha="center", va="bottom", fontsize=6.1)
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, pad=3.2, length=0)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.axvline(10.02, color=INK, linewidth=.65, ymin=0, ymax=1)
        for spine in ax.spines.values():
            spine.set_visible(False)
    text_mm(fig, 30, y_base+3, "† Reliability <0.30 (9 reporters; excluded from the stability denominator)", fontsize=6.3, va="center")
    return keys


def draw_b(fig, y_base, evidence):
    methods = ["mlp", "midas_ops", "scbutterfly_ops_b"]
    case = evidence[evidence.reporter_slug.eq("lysosome_lamp1")].set_index("method_id").loc[methods]
    assert case.tier.tolist() == ["quantitative_proxy", "ranking_proxy", "measurement_required"]
    text_mm(fig, 10, y_base+43, "LAMP1", fontsize=6.5, fontweight="bold", va="center")
    specs = [
        ("recoverability_r", "Recovery (Pearson r)", (.4, .9), [.4, .6, .8], (.7, .9), [.6]),
        ("magnitude_spearman", "Magnitude rank (ρ)", (.5, 1), [.5, .7, .9], (.7, 1), []),
        ("variance_ratio", "Magnitude variance ratio", (0, 1.6), [0, .5, 1, 1.5], (.5, 1.5), []),
        ("top5pct_recall", "Top-5% hit recall", (.4, .9), [.4, .6, .8], (.6, .9), [.5]),
    ]
    keys = []
    for i, (metric, xlabel, limits, ticks, good, additional) in enumerate(specs):
        ax = ax_mm(fig, 43 + 34*i, y_base+13, 29.0, 25.0)
        ax.axvspan(*good, color="#F2F4F6", zorder=0)
        # Solid bounds delimit the quantitative gate; dotted bounds are the
        # distinct measurement-recovery or ranking-hit cut-offs.
        for bound in good:
            if limits[0] < bound < limits[1]:
                ax.axvline(bound, color="#63727B", lw=.7, zorder=1)
        for bound in additional:
            ax.axvline(bound, color="#63727B", lw=.7, ls=(0, (1.3, 2.0)), zorder=1)
        for j, row in enumerate(case.itertuples()):
            value = float(getattr(row, metric))
            ax.plot(value, j, "o", markersize=5.0, markerfacecolor=COLORS[row.tier],
                    markeredgecolor="white", markeredgewidth=.5, zorder=3)
            keys.append(dict(panel="b", block=1, display_row=j+1, run_state=STATE,
                             reporter_slug="lysosome_lamp1", method_id=row.Index, metric=metric))
        ax.set_ylim(2.55, -.55)
        ax.set_xlim(*limits)
        ax.set_xticks(ticks, [f"{v:g}" for v in ticks], fontsize=6.1)
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=6.5, labelpad=3.5)
        ax.spines[["top", "right", "left"]].set_visible(False)
        if i == 0:
            for j, row in enumerate(case.itertuples()):
                ax.annotate(f"{row.method}  [{LETTERS[row.tier]}]", xy=(0, j),
                            xycoords=("axes fraction", "data"), xytext=(-6, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=6.4)
    return keys


def export(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    format_figure(fig, 6)
    report = layout_report(fig, ROOT / "qa" / f"{stem}_layout.json")
    if report["text_collisions"] or report["clipped_text"]:
        raise ValueError(f"Unresolved layout issues in {stem}: {report}")
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(directory / f"{stem}.{suffix}", dpi=600, metadata={"Creator": "Supplementary Figure 3 deterministic renderer"} if suffix == "pdf" else None)
    plt.close(fig)


def main():
    configure()
    evidence, order, roster = load()
    fig = plt.figure(figsize=(WIDTH/25.4, 112/25.4))
    draw_a(fig, 0, evidence, order, roster)
    export(fig, ROOT / "a_predictor_tiers", "SupplementaryFigure3-a")
    fig = plt.figure(figsize=(WIDTH/25.4, 48/25.4))
    draw_b(fig, 0, evidence)
    export(fig, ROOT / "b_lamp1_gates", "SupplementaryFigure3-b")
    fig = plt.figure(figsize=(WIDTH/25.4, HEIGHT/25.4))
    fig._composite_panel_letters = True
    keys = draw_a(fig, 50, evidence, order, roster)
    keys += draw_b(fig, 0, evidence)
    text_mm(fig, 2, 159, "a", fontsize=LETTER_PT, fontweight="bold", va="top")
    text_mm(fig, 2, 47, "b", fontsize=LETTER_PT, fontweight="bold", va="top")
    source = ROOT / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(keys).to_csv(source / "plotted_point_keys.csv", index=False)
    export(fig, ROOT / "composite", "SupplementaryFigure3")
    shutil.copy2(ROOT / "composite/SupplementaryFigure3.pdf", PAPER / "figures/SupplementaryFigure3.pdf")
    files = [DATA / "evidence.csv", DATA / "reporter_order.csv", DATA / "method_roster.csv", ROOT.parent / "publication_style.py"]
    rows = [dict(path=Path("../..").joinpath(p.relative_to(PAPER)).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    pd.DataFrame(rows).to_csv(source / "input_manifest_sha256.csv", index=False)
    print("Rendered clean panels and 183 × 162 mm composite; visual inspection still required.")


if __name__ == "__main__":
    main()
