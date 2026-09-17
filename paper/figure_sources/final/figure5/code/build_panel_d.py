#!/usr/bin/env python3
"""Render Figure 5d: environment shift and whole-screen transfer loss.

The panel is a plotting-only view of the frozen 81-direction consensus table.
It preserves the directed reporter-to-destination-screen statistical unit and
the unweighted ten-model median.  The public ``draw_panel_d`` function accepts
a Matplotlib Figure or SubFigure so the panel remains vector-native in the
manuscript composite.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from figure5_style import (
    BIOLOGY_COLORS,
    COLORS,
    SOURCE_ROOT,
    apply_style,
    clean_axis,
    save_figure,
)


HERE = Path(__file__).resolve().parent
PANEL_ROOT = HERE.parent / "d_environment_landscape"
FIGURE_ROOT = PANEL_ROOT / "figure"
MICRO_ROOT = SOURCE_ROOT / "microscopy"

SCREEN_COLORS = {
    "Biohub_OPS0047": COLORS["constraint"],
    "Biohub_OPS0067": COLORS["stable"],
}
SCREEN_LABELS = {
    "Biohub_OPS0047": "OPS0047",
    "Biohub_OPS0067": "OPS0067",
}
SHIFT_CMAP = LinearSegmentedColormap.from_list(
    "environment_shift",
    ["#F4F8F8", "#D4E3ED", "#79A8C9", COLORS["stable"], "#294A66"],
    N=256,
)
FLUOR_CMAP = LinearSegmentedColormap.from_list(
    "ferhonox_cyan",
    ["#020609", "#173947", "#4FA7B8", "#DDF1F3"],
    N=256,
)


def prepare_panel_d() -> dict[str, Any]:
    transfer = pd.read_csv(SOURCE_ROOT / "panel_d/canonical_environment_transfer_display.csv")
    assoc = pd.read_csv(SOURCE_ROOT / "panel_d/association_summary.csv")
    mapping = pd.read_csv(SOURCE_ROOT / "panel_d/ferhonox_point_to_image_mapping.csv")
    selection = pd.read_csv(MICRO_ROOT / "selection_manifest.csv")
    selection = selection.loc[selection.target_key.eq("ferhonox")].copy()
    display = pd.read_csv(MICRO_ROOT / "display_manifest.csv").set_index("target_key").loc["ferhonox"]

    # The order is determined solely by the frozen positive transfer-loss
    # column.  A stable secondary key makes the plotting order deterministic.
    transfer = transfer.sort_values(
        ["gene_minus_strict_screen_pearson", "reporter_slug", "destination_screen"],
        kind="mergesort",
    ).reset_index(drop=True)
    transfer["ordered_direction_index"] = np.arange(1, len(transfer) + 1)
    transfer["display_screen"] = transfer.destination_screen.str.replace("Biohub_", "", regex=False)

    if len(transfer) != 81:
        raise AssertionError(f"Expected 81 directed transfers, found {len(transfer)}")
    if transfer.reporter_slug.nunique() != 34:
        raise AssertionError("Expected 34 reporter clusters")
    if transfer.destination_screen.nunique() != 66:
        raise AssertionError("Expected 66 destination screens")
    fe = transfer.loc[transfer.reporter_short_name.str.contains("FeRhoNox", case=False, na=False)]
    if set(fe.destination_screen) != set(SCREEN_COLORS):
        raise AssertionError("Frozen FeRhoNox two-direction anchor changed")
    if set(selection.screen_id) != set(SCREEN_COLORS):
        raise AssertionError("Microscopy screens do not match highlighted transfer directions")

    forest_rows = []
    for predictor, label, color in [
        ("phase_ntc_shift_rms_smd", "Phase NTC", COLORS["blue"]),
        ("phenotype_ntc_shift_rms_smd", "Target NTC", COLORS["stable"]),
    ]:
        row = assoc.loc[assoc.predictor.eq(predictor)].iloc[0]
        forest_rows.append(
            {
                "predictor": predictor,
                "label": label,
                "rho_transfer_loss": -float(row.spearman_rho),
                "ci95_low": -float(row.ci95_high),
                "ci95_high": -float(row.ci95_low),
                "n_directions": int(row.n_rows),
                "n_reporters": int(row.n_clusters),
                "cluster_permutation_p": float(row.reporter_cluster_permutation_p),
                "color": color,
            }
        )
    forest = pd.DataFrame(forest_rows)
    return {
        "transfer": transfer,
        "forest": forest,
        "mapping": mapping,
        "selection": selection,
        "display": display,
    }


def _draw_shift_strip(ax: plt.Axes, values: np.ndarray, label: str, norm: LogNorm) -> None:
    image = ax.imshow(
        values[np.newaxis, :],
        aspect="auto",
        interpolation="nearest",
        cmap=SHIFT_CMAP,
        norm=norm,
        extent=(-0.5, len(values) - 0.5, -0.5, 0.5),
        rasterized=True,
    )
    ax.set_xlim(-0.7, len(values) - 0.3)
    ax.set_yticks([0], [label])
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _add_scale_bar(ax: plt.Axes, pixel_size_um: float, bar_um: float = 20.0) -> None:
    length_px = bar_um / pixel_size_um
    height, width = np.asarray(ax.images[-1].get_array()).shape[:2]
    x_mid = (width - 1) / 2.0
    x_start, x_end = x_mid - length_px / 2.0, x_mid + length_px / 2.0
    y = height - 25.0
    effects = [path_effects.Stroke(linewidth=2.4, foreground="black"), path_effects.Normal()]
    line, = ax.plot([x_start, x_end], [y, y], color="white", lw=1.35,
                    solid_capstyle="butt")
    line.set_path_effects(effects)
    label = ax.text(
        (x_start + x_end) / 2,
        y - 8,
        "20 µm",
        ha="center",
        va="bottom",
        color="white",
        fontsize=5.2,
        fontweight="bold",
    )
    label.set_path_effects(effects)


def _draw_microscopy_pair(
    container: Any,
    spec: Any,
    screen: str,
    row: pd.Series,
    display: pd.Series,
) -> tuple[list[plt.Axes], plt.Axes]:
    group = spec.subgridspec(1, 2, wspace=0.018)
    asset = np.load(MICRO_ROOT / f"{row.asset_key}.npz")
    axes: list[plt.Axes] = []
    for col, channel in enumerate(["phase", "fluorescence"]):
        ax = container.add_subplot(group[0, col])
        axes.append(ax)
        if channel == "phase":
            ax.imshow(
                asset[channel], cmap="gray", vmin=float(display.phase_low),
                vmax=float(display.phase_high), interpolation="none", rasterized=True,
            )
        else:
            ax.imshow(
                asset[channel], cmap=FLUOR_CMAP, vmin=float(display.fluorescence_low),
                vmax=float(display.fluorescence_high), interpolation="none", rasterized=True,
            )
            _add_scale_bar(ax, float(display.pixel_size_um))
        ax.set_xticks([]); ax.set_yticks([])
        # ``imshow`` keeps a square data box.  Anchoring the phase and reporter
        # axes towards their shared boundary removes the otherwise large blank
        # band between the two registered views.
        ax.set_anchor("E" if col == 0 else "W")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.65)
            spine.set_color(SCREEN_COLORS[screen])
        ax.text(
            0.50, -0.045, "phase" if channel == "phase" else "FeRhoNox",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.0,
            color=COLORS["ink"] if channel == "phase" else COLORS["stable"],
            clip_on=False,
        )
    # The coloured header and matching pair borders identify the registered
    # screen.  A spanning black keyline is redundant and visually reads as an
    # unexplained rule, so the pair is left open between title and images.
    axes[0].text(
        1.0, 1.045, SCREEN_LABELS[screen], transform=axes[0].transAxes,
        ha="center", va="bottom", fontsize=5.8, fontweight="bold",
        color=SCREEN_COLORS[screen], clip_on=False,
    )
    return axes, axes[0]


def draw_panel_d(container: Any, data: dict[str, Any] | None = None, *, add_letter: bool = True):
    """Draw Figure 5d into a Figure or SubFigure and return its artists."""
    if data is None:
        data = prepare_panel_d()
    transfer: pd.DataFrame = data["transfer"]
    forest: pd.DataFrame = data["forest"]
    selection: pd.DataFrame = data["selection"]
    display: pd.Series = data["display"]

    outer = container.add_gridspec(2, 1, height_ratios=[0.68, 0.32], hspace=0.30)
    top = outer[0].subgridspec(1, 2, width_ratios=[4.15, 1.45], wspace=0.28)
    landscape = top[0].subgridspec(
        5, 1, height_ratios=[2.65, 0.42, 0.42, 0.22, 0.38], hspace=0.16
    )
    ax_loss = container.add_subplot(landscape[0, 0])
    ax_phase = container.add_subplot(landscape[1, 0])
    ax_target = container.add_subplot(landscape[2, 0])
    ax_biology = container.add_subplot(landscape[3, 0])
    ax_rank = container.add_subplot(landscape[4, 0])

    x = np.arange(len(transfer))
    loss = transfer.gene_minus_strict_screen_pearson.to_numpy(float)
    point_colors = [BIOLOGY_COLORS.get(str(v), COLORS["mid"]) for v in transfer.biological_category_x]
    ax_loss.axhspan(-0.16, 0.0, color=COLORS["pale"], zorder=0)
    ax_loss.axhline(0, color=COLORS["mid"], lw=0.65, ls="--", zorder=1)
    ax_loss.plot(x, loss, color="#91A0A9", lw=0.65, zorder=2)
    ax_loss.scatter(
        x, loss, s=10, c=point_colors, edgecolor="white", linewidth=0.3,
        zorder=3, rasterized=True,
    )
    highlighted_positions: dict[str, tuple[int, float]] = {}
    for screen, color in SCREEN_COLORS.items():
        idx = transfer.index[transfer.destination_screen.eq(screen)].tolist()
        if len(idx) != 1:
            raise AssertionError(f"Expected one ordered row for {screen}")
        i = idx[0]
        y = float(transfer.loc[i, "gene_minus_strict_screen_pearson"])
        ax_loss.scatter(i, y, s=32, marker="D", color=color, edgecolor="white", lw=0.75, zorder=6)
        ax_loss.annotate(
            SCREEN_LABELS[screen], xy=(i, y), xytext=(-5, -10 if screen.endswith("0067") else 5),
            textcoords="offset points", ha="right", va="center", fontsize=5.0,
            fontweight="bold", color=color, clip_on=False,
        )
        highlighted_positions[screen] = (i, y)
    # Preserve a full marker-width of air around the two rightmost diamonds.
    # The same x-limit is used by all aligned tracks below.
    aligned_xlim = (-0.7, len(transfer) - 0.05)
    ax_loss.set_xlim(*aligned_xlim)
    ax_loss.set_ylim(-0.16, 0.84)
    ax_loss.set_ylabel("Transfer loss (Δr)")
    ax_loss.set_yticks([0, .25, .5, .75])
    ax_loss.set_xticks([])
    ax_loss.grid(axis="y", color=COLORS["grid"], lw=0.45)
    clean_axis(ax_loss, bottom=False)

    norm = LogNorm(vmin=0.04, vmax=140)
    im = _draw_shift_strip(
        ax_phase, transfer.phase_ntc_shift_rms_smd.to_numpy(float), "Phase NTC", norm
    )
    _draw_shift_strip(
        ax_target, transfer.phenotype_ntc_shift_rms_smd.to_numpy(float), "Target NTC", norm
    )
    bio_rgb = np.array([
        plt.matplotlib.colors.to_rgb(BIOLOGY_COLORS.get(str(v), COLORS["mid"]))
        for v in transfer.biological_category_x
    ])[np.newaxis, :, :]
    ax_biology.imshow(bio_rgb, aspect="auto", interpolation="nearest")
    ax_biology.set_xlim(*aligned_xlim)
    ax_biology.set_yticks([0], ["Biology"])
    ax_biology.tick_params(axis="y", length=0, pad=3.0)
    ax_biology.set_xticks([])
    for spine in ax_biology.spines.values():
        spine.set_visible(False)

    ax_rank.set_xlim(*aligned_xlim)
    ax_rank.set_ylim(0, 1)
    ax_rank.set_yticks([])
    ax_rank.set_xticks([0, 20, 40, 60, 80], ["1", "21", "41", "61", "81"])
    ax_rank.set_xlabel("")
    ax_rank.text(
        0.5, 0.72, "ordered transfer-loss rank", transform=ax_rank.transAxes,
        ha="center", va="center", fontsize=4.8, color=COLORS["ink"],
    )
    for spine in ax_rank.spines.values():
        spine.set_visible(False)
    ax_rank.tick_params(axis="x", width=0.5, length=2.0, pad=1.0)

    # A compact logarithmic legend for the two aligned NTC-shift strips sits
    # in unused upper-left space of the ordered landscape.
    cax = container.add_axes([0.26, 0.965, 0.22, 0.018])
    cb = ax_loss.figure.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0.05, 1, 100])
    cb.set_ticklabels(["0.05", "1", "100"])
    cb.set_label("")
    container.text(.25, .974, "NTC shift (log)", ha='right', va='center', fontsize=6.0)
    cb.ax.tick_params(labelsize=5.0, length=1.2, pad=0.4)
    cb.outline.set_linewidth(0.4)

    # Compact cluster-aware association forest.  The sign is intentionally
    # flipped from the archived screen-penalty source to positive transfer loss.
    ax_forest = container.add_subplot(top[1])
    y_positions = np.array([1.0, 0.0])
    for y, row in zip(y_positions, forest.itertuples(index=False)):
        ax_forest.plot([row.ci95_low, row.ci95_high], [y, y], color=row.color, lw=1.35)
        ax_forest.scatter(row.rho_transfer_loss, y, s=30, color=row.color,
                          edgecolor="white", lw=0.65, zorder=4)
        ax_forest.text(1.02, y + .12, f"{row.rho_transfer_loss:.2f}", ha="right", va="bottom",
                       fontsize=5.0, fontweight="bold")
    ax_forest.axvline(0, color=COLORS["mid"], lw=0.65)
    ax_forest.set_xlim(0, 1.08)
    ax_forest.set_ylim(-0.55, 1.55)
    ax_forest.set_yticks(y_positions, forest.label.tolist())
    ax_forest.set_xlabel("")
    ax_forest.grid(axis="x", color=COLORS["grid"], lw=0.45)
    # The legend defines the reporter-median permutation and direction-level
    # bootstrap units. A single P annotation avoids calling it a cluster test.
    ax_forest.text(0.98, 0.97, "P = 0.0003", transform=ax_forest.transAxes,
                   ha="right", va="top", fontsize=6.0)
    ax_forest.text(0.50, 0.025, "Spearman ρ", transform=ax_forest.transAxes,
                   ha="center", va="bottom", fontsize=4.8)
    clean_axis(ax_forest)

    # Registered, outcome-agnostic NTC medoids for the two frozen FeRhoNox
    # destination directions.  The images are evidence anchors, not decoration.
    bottom = outer[1].subgridspec(1, 2, wspace=0.10)
    image_first_axes: dict[str, plt.Axes] = {}
    image_axes: list[plt.Axes] = []
    for col, screen in enumerate(["Biohub_OPS0067", "Biohub_OPS0047"]):
        row = selection.loc[selection.screen_id.eq(screen)].iloc[0]
        pair_axes, first_ax = _draw_microscopy_pair(container, bottom[0, col], screen, row, display)
        image_axes.extend(pair_axes)
        image_first_axes[screen] = first_ax

    if add_letter:
        container.text(0.002, 0.985, "d", ha="left", va="top", fontsize=8, fontweight="bold")
    return [ax_loss, ax_phase, ax_target, ax_biology, ax_rank, cax, ax_forest, *image_axes]


def _write_bundle(data: dict[str, Any]) -> None:
    PANEL_ROOT.mkdir(parents=True, exist_ok=True)
    transfer = data["transfer"].copy()
    transfer.to_csv(PANEL_ROOT / "figure_source_ordered_environment_landscape.csv", index=False)
    data["forest"].drop(columns=["color"]).to_csv(
        PANEL_ROOT / "figure_source_environment_associations.csv", index=False
    )
    data["mapping"].to_csv(PANEL_ROOT / "figure_source_ferhonox_point_to_image_mapping.csv", index=False)
    data["selection"].to_csv(PANEL_ROOT / "figure_source_microscopy_selection.csv", index=False)
    asset_rows = []
    for row in data["selection"].itertuples(index=False):
        path = MICRO_ROOT / f"{row.asset_key}.npz"
        asset_rows.append(
            {
                "asset_key": row.asset_key,
                "screen_id": row.screen_id,
                "relative_path": f"source_data/microscopy/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        )
    pd.DataFrame(asset_rows).to_csv(PANEL_ROOT / "figure_source_microscopy_asset_manifest.csv", index=False)

    (PANEL_ROOT / "audit_report.md").write_text(
        "# Figure 5d audit\n\n"
        "- Statistical unit: directed reporter-to-destination-screen evaluation.\n"
        "- Displayed units: 81 directions nested in 34 reporters and spanning 66 destination screens.\n"
        "- Consensus: unweighted median of the frozen ten predictors.\n"
        "- Ordering: ascending `gene_minus_strict_screen_pearson`, with stable reporter and screen tie-breaks.\n"
        "- Association inference: direction-level Spearman correlations and reporter-cluster bootstrap confidence intervals; two-sided P values permute reporter-median shift summaries 3,000 times, with plus-one correction, as in the frozen source.\n"
        "- Sign convention: archived correlations with screen penalty were multiplied by −1 to display positive transfer loss.\n"
        "- Microscopy: all two directions of the prespecified FeRhoNox repeated-screen case; each image is the outcome-agnostic phase/geometry NTC medoid for that screen.\n"
        "- Missingness: no plotted direction is missing the primary transfer-loss or NTC-shift variables.\n"
        "- Leakage: microscopy selection did not use model performance or transfer loss.\n",
        encoding="utf-8",
    )
    (PANEL_ROOT / "figure_methods.md").write_text(
        "# Figure 5d methods\n\nDirected reporter-to-destination-screen evaluations were ordered by the difference between held-out-gene and strict whole-screen gene-level Pearson correlation. Phase and targeted-phenotype NTC shifts were displayed on an aligned logarithmic colour scale. Spearman correlations were calculated across 81 directions, with 95% confidence intervals obtained by resampling complete reporter clusters. For two-sided permutation tests, each shift and transfer-loss measure was first summarized by its median within each of the 34 reporters; reporter-level shift summaries were then permuted 3,000 times, with plus-one correction. The two FeRhoNox directions were linked to registered screen-specific NTC medoid images selected without reference to transfer performance.\n\n## Analysis specification\n\n- Statistical unit: directed reporter-to-destination-screen evaluation.\n- Displayed units: 81 directions nested in 34 reporters and spanning 66 destination screens.\n- Consensus: unweighted median of the frozen ten predictors.\n- Ordering: ascending `gene_minus_strict_screen_pearson`, with stable reporter and screen tie-breaks.\n- Association inference: direction-level Spearman correlations and reporter-cluster bootstrap confidence intervals; two-sided P values permute reporter-median shift summaries 3,000 times, with plus-one correction, as in the frozen source.\n- Sign convention: archived correlations with screen penalty were multiplied by −1 to display positive transfer loss.\n- Microscopy: all two directions of the prespecified FeRhoNox repeated-screen case; each image is the outcome-agnostic phase/geometry NTC medoid for that screen.\n- Missingness: no plotted direction is missing the primary transfer-loss or NTC-shift variables.\n- Leakage: microscopy selection did not use model performance or transfer loss.\n",
        encoding="utf-8",
    )
    (PANEL_ROOT / "figure_legend.md").write_text(
        "# Figure 5d legend\n\n"
        "Whole-screen transfer loss across 81 directed reporter-to-destination-screen evaluations, ordered from the smallest to the largest loss. Aligned strips show phase and targeted-phenotype NTC shifts, and the lower strip identifies the reporter biological system. The forest plot shows direction-level Spearman correlations with reporter-cluster bootstrap 95% confidence intervals; P values use 3,000 permutations of reporter-median shift summaries, with plus-one correction. Diamonds identify the two FeRhoNox directions; registered phase and FeRhoNox images show the prespecified screen-specific NTC medoid cells. Scale bars, 20 µm.\n",
        encoding="utf-8",
    )
    (PANEL_ROOT / "plot_spec.yaml").write_text(
        "panel: d\n"
        "statistical_unit: directed reporter-to-destination-screen evaluation\n"
        "n_directions: 81\n"
        "n_reporters: 34\n"
        "n_destination_screens: 66\n"
        "consensus: unweighted median of ten frozen predictors\n"
        "primary_outcome: gene_holdout_pearson_minus_strict_whole_screen_pearson\n"
        "ordering: ascending primary outcome\n"
        "association: direction-level Spearman with reporter-cluster bootstrap CI; two-sided reporter-median permutation P (3000 draws, plus-one correction)\n"
        "microscopy_case: all two directions of frozen FeRhoNox repeated-screen case\n"
        "font: Arial\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "claim": "NTC environment shift is associated with whole-screen transfer loss.",
                "supported": True,
                "boundary": "Association does not establish that the NTC shift caused the transfer failure.",
            },
            {
                "claim": "The FeRhoNox images identify a general mechanism for all difficult screens.",
                "supported": False,
                "boundary": "The registered case is an illustrative two-direction anchor, not a prevalence sample.",
            },
        ]
    ).to_csv(PANEL_ROOT / "claim_table.csv", index=False)


def main() -> None:
    apply_style()
    data = prepare_panel_d()
    _write_bundle(data)
    fig = plt.figure(figsize=(122 / 25.4, 64 / 25.4))
    fig.subplots_adjust(left=0.125, right=0.985, bottom=0.065, top=0.925)
    draw_panel_d(fig, data)
    save_figure(fig, FIGURE_ROOT / "Figure5-d_environment_landscape")
    plt.close(fig)
    qa = {
        "status": "PASS",
        "n_directions": int(len(data["transfer"])),
        "n_reporters": int(data["transfer"].reporter_slug.nunique()),
        "n_destination_screens": int(data["transfer"].destination_screen.nunique()),
        "highlighted_screens": sorted(SCREEN_COLORS),
        "font": "Arial",
        "exports": ["png", "pdf", "svg"],
    }
    (PANEL_ROOT / "figure_qa_report.json").write_text(
        json.dumps(qa, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
