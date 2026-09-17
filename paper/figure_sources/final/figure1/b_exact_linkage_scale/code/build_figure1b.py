#!/usr/bin/env python3
"""Build Figure 1b as three aligned, unit-explicit descriptive plots.

The left spectrum enumerates 73 physical screens and separates distinct
phase cells from reporter observations aggregated within the same screen. The
middle spectrum resolves the same reporter-observation total into 99 acquired
reporter-screen assays. The right frequency chart enumerates the 52 reporter
blocks by endpoint dimensionality. No prediction outcome enters any ordering.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D


FIGURE_ROOT = Path(__file__).resolve().parents[2]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))
from _portable_fonts import configure_sans  # noqa: E402
sys.path.insert(0, str(FIGURE_ROOT / "code"))
from layout_qa import assert_text_layout  # noqa: E402


WIDTH_MM = 183
HEIGHT_MM = 30
MM_TO_IN = 1 / 25.4

INK = "#202A33"
SLATE = "#526573"
BLUE = "#3E6F9C"
BLUE_MID = "#5F9BC3"
BLUE_PALE = "#BFD4E1"
ORANGE = "#E8943A"
PURPLE = "#7B6597"
GRID = "#DCE3E7"
WHITE = "#FFFFFF"

ANCHOR_STYLE = {
    "lysosome_lamp1": ("LAMP1", ORANGE),
    "fe2+_ferhonox_live-cell_dye": ("FeRhoNox", PURPLE),
    "prb": ("pRb", PURPLE),
}


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure() -> None:
    configure_sans(mpl, font_manager, FIGURE_ROOT)
    mpl.rcParams.update({
        "font.size": 6.0,
        "axes.titlesize": 6.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 6.0,
        "axes.edgecolor": "#6F7E88",
        "axes.linewidth": 0.50,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "xtick.major.width": 0.45,
        "ytick.major.width": 0.45,
        "xtick.major.size": 1.7,
        "ytick.major.size": 1.7,
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
    })


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inp = bundle_root() / "source_data" / "input"
    screens = pd.read_csv(inp / "screen_order.csv")
    endpoints = pd.read_csv(inp / "endpoint_block_summary.csv")
    assays = pd.read_csv(inp / "assay_coverage_rank.tsv", sep="\t")

    assert len(screens) == screens["screen_id"].nunique() == 73
    assert int(screens["unique_phase_cells_with_exact_reporter"].sum()) == 7_344_374
    assert int(screens["exact_assay_cell_links"].sum()) == 9_996_286
    assert int(screens["n_reporter_targets"].sum()) == 99

    dims_col = next(c for c in endpoints.columns if c in {
        "output_dimensions", "dimension", "n_endpoints"
    })
    count_col = next(c for c in endpoints.columns if c in {
        "n_reporters", "count", "reporter_count"
    })
    endpoints = endpoints.rename(
        columns={dims_col: "output_dimensions", count_col: "n_reporters"}
    )[["output_dimensions", "n_reporters"]].copy()
    endpoints = endpoints.astype(int).sort_values(
        "output_dimensions", kind="stable"
    ).reset_index(drop=True)
    assert endpoints.set_index("output_dimensions")["n_reporters"].to_dict() == {
        20: 1, 24: 38, 30: 6, 60: 1, 72: 6,
    }
    assert int((endpoints.output_dimensions * endpoints.n_reporters).sum()) == 1_604

    screens = screens.sort_values(
        ["unique_phase_cells_with_exact_reporter", "screen_id"], kind="stable"
    ).reset_index(drop=True)
    screens["coverage_rank"] = np.arange(1, len(screens) + 1)
    screens["screen_class"] = np.where(
        screens["n_reporter_targets"].eq(1), "one reporter", "2–7 reporters"
    )

    required_assay_columns = {
        "assay_key", "reporter_slug", "display_name", "biological_category",
        "screen_id", "gene_guide_concordant_exact_links",
    }
    assert required_assay_columns <= set(assays.columns)
    assert len(assays) == assays["assay_key"].nunique() == 99
    assert assays["screen_id"].nunique() == 73
    assert assays["reporter_slug"].nunique() == 52
    assert int(assays["gene_guide_concordant_exact_links"].sum()) == 9_996_286
    assays = assays.sort_values(
        ["gene_guide_concordant_exact_links", "assay_key"], kind="stable"
    ).reset_index(drop=True)
    assays["coverage_rank"] = np.arange(1, len(assays) + 1)
    return screens, assays, endpoints


def write_sources(
    screens: pd.DataFrame,
    assays: pd.DataFrame,
    endpoints: pd.DataFrame,
) -> None:
    out = bundle_root() / "source_data"
    screens[[
        "coverage_rank", "screen_id", "n_reporter_targets", "screen_class",
        "unique_phase_cells_with_exact_reporter", "exact_assay_cell_links",
    ]].to_csv(out / "figure_source_data_screen.tsv", sep="\t", index=False,
              lineterminator="\n")
    assays[[
        "coverage_rank", "assay_key", "reporter_slug", "display_name",
        "biological_category", "screen_id",
        "gene_guide_concordant_exact_links",
    ]].to_csv(out / "figure_source_data_assay.tsv", sep="\t", index=False,
              lineterminator="\n")
    endpoints.to_csv(out / "figure_source_data_endpoint_schema.tsv", sep="\t",
                     index=False, lineterminator="\n")

    contract = {
        "panel": "Figure 1b",
        "claim": (
            "Exact same-cell linkage yields 7.344M distinct phase cells across "
            "73 screens and 9.996M reporter observations across 99 acquired "
            "reporter-screen assays; 52 reporter-specific targets define 1,604 endpoints."
        ),
        "statistical_units": {
            "left_spectrum": "physical screen (n=73)",
            "middle_spectrum": "acquired reporter-screen assay (n=99)",
            "right_frequency_chart": "reporter-specific output block (n=52)",
        },
        "totals": {
            "physical_screens": 73,
            "reporter_screen_assays": 99,
            "reporters": 52,
            "distinct_exact_linked_phase_cells": 7_344_374,
            "exact_linked_reporter_observations": 9_996_286,
            "phenotype_endpoints": 1_604,
        },
        "ordering": {
            "screens": "ascending distinct exact-linked phase-cell count; stable screen_id tie-break",
            "assays": "ascending exact-linked reporter observations; stable assay_key tie-break",
        },
        "exclusions": "none",
        "analysis_mode": "descriptive_enumeration",
        "inferential_unit": "none; complete enumeration, not population inference",
        "uncertainty": "none",
        "layout": "three plots side by side at 183 x 30 mm; no native panel letter",
        "endpoint_encoding": "categorical endpoint count on x; reporter count on y",
        "endpoint_marks": "blue filled bars with darker outlines from zero; exact counts above bars",
        "outcome_based_ordering": False,
        "font": "DejaVu Sans; editable embedded TrueType in PDF",
        "input_sha256": {
            p.name: sha256(p) for p in sorted((out / "input").glob("*")) if p.is_file()
        },
    }
    (out / "source_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )


def assert_text_clear(fig: plt.Figure, name: str, artists: list[mpl.text.Text]) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer=renderer) for artist in artists]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert not boxes[i].overlaps(boxes[j]), (
                f"Text collision in {name}: {artists[i].get_text()!r} vs "
                f"{artists[j].get_text()!r}"
            )


def build() -> None:
    configure()
    screens, assays, endpoints = load_data()
    write_sources(screens, assays, endpoints)

    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN))
    # Separate axis gutters and a reserved top-left letter zone remain clear
    # even after assembly. The compositing layer alone owns the b label.
    ax_screen = fig.add_axes([0.064, 0.250, 0.288, 0.500])
    ax_assay = fig.add_axes([0.425, 0.250, 0.288, 0.500])
    ax_schema = fig.add_axes([0.799, 0.250, 0.187, 0.500])

    # Physical-screen spectrum: the two marks share the same screen rank.
    x_screen = screens["coverage_rank"].to_numpy(float)
    phase = screens["unique_phase_cells_with_exact_reporter"].to_numpy(float) / 1000
    paired = screens["exact_assay_cell_links"].to_numpy(float) / 1000
    multi = screens["n_reporter_targets"].gt(1).to_numpy()
    ax_screen.plot(
        x_screen, phase, color=SLATE, lw=0.56, alpha=0.76, zorder=1.5,
    )
    for x, y0, y1 in zip(x_screen[multi], phase[multi], paired[multi]):
        ax_screen.plot(
            [x, x], [y0, y1], color=ORANGE, lw=0.62,
            alpha=0.86, zorder=1,
        )
    ax_screen.scatter(
        x_screen, phase, s=6.8, color=SLATE, edgecolor=WHITE,
        linewidth=0.22, zorder=3,
    )
    ax_screen.scatter(
        x_screen[multi], paired[multi], s=8.5, marker="s", color=BLUE,
        edgecolor=WHITE, linewidth=0.22, zorder=4,
    )
    ax_screen.set_xlim(0, 74)
    ax_screen.set_ylim(0, 840)
    ax_screen.set_xticks([1, 37, 73])
    ax_screen.set_yticks([0, 400, 800])
    ax_screen.set_ylabel("Count (×10³)", labelpad=2.0)
    ax_screen.set_xlabel("Physical screens (rank)", labelpad=1.5)
    ax_screen.tick_params(axis="x", pad=1.1)
    ax_screen.yaxis.grid(True, color=GRID, lw=0.38)
    ax_screen.set_axisbelow(True)
    ax_screen.spines[["top", "right"]].set_visible(False)
    # The key is outside the data rectangle: no unusually large multi-reporter
    # total can land on a legend label when fonts differ across platforms.
    screen_key = fig.legend(
        [Line2D([], [], marker="o", color=SLATE, linestyle="none", markersize=2.8),
         Line2D([], [], marker="s", color=BLUE, linestyle="none", markersize=2.8)],
        ["distinct phase cells", "reporter observations"],
        loc="upper left", bbox_to_anchor=(0.064, 0.960), ncol=1,
        frameon=False, fontsize=6.0, handlelength=0.6, handletextpad=0.35,
        columnspacing=0.9, borderaxespad=0.0, borderpad=0.0,
    )
    phase_key, multi_key = screen_key.get_texts()
    # Reporter-screen assay spectrum moved out of panel c. One point is one
    # acquired assay; selected recurring manuscript examples are direct-labelled.
    x_assay = assays["coverage_rank"].to_numpy(float)
    y_assay = assays["gene_guide_concordant_exact_links"].to_numpy(float) / 1000
    ax_assay.vlines(x_assay, 28, y_assay, color=BLUE_PALE, lw=0.30, alpha=0.78, zorder=1)
    ax_assay.plot(x_assay, y_assay, color="#6E9CB8", lw=0.62, zorder=1.5)
    ax_assay.scatter(
        x_assay, y_assay, s=6.8, color=BLUE_MID, alpha=0.86,
        edgecolor=WHITE, linewidth=0.20, zorder=2,
    )
    assay_labels = []
    label_specs = {
        "lysosome_lamp1": (13.0, 111.0),
        "prb": (43.0, 67.0),
        "fe2+_ferhonox_live-cell_dye": (71.0, 137.0),
    }
    for slug, text_xy in label_specs.items():
        selected = assays.loc[assays["reporter_slug"].eq(slug)].iloc[-1]
        label, colour = ANCHOR_STYLE[slug]
        ax_assay.scatter(
            [selected.coverage_rank],
            [selected.gene_guide_concordant_exact_links / 1000],
            s=18.0, color=colour, edgecolor=INK, linewidth=0.55, zorder=4,
        )
        assay_labels.append(ax_assay.annotate(
            label,
            xy=(float(selected.coverage_rank),
                float(selected.gene_guide_concordant_exact_links) / 1000),
            xytext=text_xy,
            ha="center", va="center", fontsize=6.0, fontweight="bold",
            color=INK,
            arrowprops=dict(arrowstyle="-", color=INK, lw=0.38,
                            shrinkA=1.0, shrinkB=1.5),
            zorder=5,
        ))
    ax_assay.set_xlim(0, 100)
    ax_assay.set_ylim(28, 181)
    ax_assay.set_xticks([1, 50, 99])
    ax_assay.set_yticks([40, 100, 160])
    ax_assay.set_ylabel("Count (×10³)", labelpad=2.0)
    ax_assay.set_xlabel("Reporter–screen assays (rank)", labelpad=1.5)
    ax_assay.tick_params(axis="both", pad=1.0)
    ax_assay.yaxis.grid(True, color=GRID, lw=0.38)
    ax_assay.set_axisbelow(True)
    ax_assay.spines[["top", "right"]].set_visible(False)
    # Complete frequency enumeration: no binning, density estimate or error
    # bars. Equal categorical spacing avoids merging the adjacent 20/24/30
    # dimensions; the labelled x values and heights retain the exact schema.
    positions = np.arange(len(endpoints))
    counts = endpoints["n_reporters"].to_numpy(int)
    # A complete five-category frequency table benefits from a stronger area
    # mark than the adjacent dense assay spectrum. Keep the same blue identity
    # and exact zero-baseline heights; no gradients, broken axis or trend line.
    ax_schema.bar(positions, counts, width=0.52, color=BLUE_MID,
                  edgecolor=BLUE, linewidth=0.70, zorder=3)
    schema_artists = [ax_schema.annotate(str(count), xy=(x, count),
                                        xytext=(0, 2.8), textcoords="offset points",
                                        ha="center", va="bottom", fontsize=6.0)
                      for x, count in zip(positions, counts)]
    ax_schema.set_xlim(-0.6, len(endpoints) - 0.4)
    ax_schema.set_ylim(0, 46)
    ax_schema.set_xticks(positions, endpoints["output_dimensions"].astype(str))
    ax_schema.set_yticks([0, 20, 40])
    ax_schema.set_xlabel("Endpoints per reporter", labelpad=1.5)
    ax_schema.set_ylabel("Reporters", labelpad=2.0)
    ax_schema.yaxis.grid(True, color=GRID, lw=0.38)
    ax_schema.set_axisbelow(True)
    ax_schema.spines[["top", "right"]].set_visible(False)

    assert int(counts.sum()) == 52
    assert_text_clear(fig, "screen key", [phase_key, multi_key])
    assert_text_clear(fig, "assay labels", assay_labels)
    assert_text_clear(fig, "schema accounting", schema_artists)
    assert_text_layout(fig)

    out = bundle_root() / "figure" / "Figure1-b_exact_linkage_scale"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(out.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    build()
