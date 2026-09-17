#!/usr/bin/env python3
"""Build Figure 4c: representative modes of KO-response reconstruction.

The panel is plotting-only.  It reads the frozen, self-contained Figure 4
source tables and never refits, reselects, or reaggregates a predictor.  The
three reporters and the displayed gene-screen profiles are preselected in the
source contract.  Dense matrices and 1,000-gene point clouds are rasterized;
all typography, axes and annotations remain editable in PDF/SVG output.

``plot_panel_c`` is intentionally importable by the full Figure 4 builder.  It
accepts either a Matplotlib ``SubplotSpec`` or a figure-coordinate rectangle
``(left, bottom, width, height)`` so the standalone and composite use exactly
the same drawing implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import ConnectionPatch
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

from figure4_style import COLORS, SOURCE_ROOT, apply_style, clean_axis, save_figure


OUT_ROOT = Path(__file__).resolve().parents[1] / "c_reconstruction_modes"
PANEL_SOURCE = SOURCE_ROOT / "panel_c"

REPORTERS = (
    {
        "slug": "lysosome_lysotracker_live-cell_dye",
        "name": "LysoTracker",
        "mode": "quantitative recovery",
        "color": "#E8943A",
    },
    {
        "slug": "chromalive_488_excitation",
        "name": "ChromaLIVE 488",
        "mode": "pattern retained",
        "color": "#D56C75",
    },
    {
        "slug": "prb",
        "name": "pRb",
        "mode": "amplitude compressed",
        "color": "#7B6597",
    },
)

KO_CMAP = LinearSegmentedColormap.from_list(
    "ko_response",
    ["#3E6F9C", "#79A8C9", "#F7F7F4", "#E5A0A8", "#A44758"],
    N=256,
)
REPORTER_CMAP = LinearSegmentedColormap.from_list(
    "reporter_image", ["#020609", "#173947", "#4FA7B8", "#DDF1F3"], N=256
)


def _host_spec(fig: plt.Figure, host):
    """Return a SubplotSpec and its figure-coordinate bounding box."""

    if hasattr(host, "subgridspec"):
        return host, host.get_position(fig)
    if isinstance(host, (tuple, list)) and len(host) == 4:
        left, bottom, width, height = (float(x) for x in host)
        spec = fig.add_gridspec(
            1,
            1,
            left=left,
            right=left + width,
            bottom=bottom,
            top=bottom + height,
        )[0, 0]
        return spec, Bbox.from_bounds(left, bottom, width, height)
    raise TypeError("host must be a SubplotSpec or (left, bottom, width, height)")


def _load_profile(slug: str, source_root: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.read_csv(source_root / slug / "response_profiles.csv.gz")
    observed_cols = [c for c in frame.columns if c.startswith("observed__")]
    predicted_cols = [c for c in frame.columns if c.startswith("median10__")]
    observed_suffix = [c.removeprefix("observed__") for c in observed_cols]
    predicted_map = {c.removeprefix("median10__"): c for c in predicted_cols}
    if not observed_cols or set(observed_suffix) != set(predicted_map):
        raise RuntimeError(f"{slug}: observed and predicted endpoint blocks are not aligned")
    predicted_cols = [predicted_map[s] for s in observed_suffix]
    truth = frame[observed_cols].to_numpy(float)
    prediction = frame[predicted_cols].to_numpy(float)
    if truth.shape[0] != 12 or truth.shape != prediction.shape:
        raise RuntimeError(f"{slug}: expected aligned 12-profile matrices, got {truth.shape}")
    return frame, truth, prediction


def _profile_labels(frame: pd.DataFrame) -> list[str]:
    """Expose the real gene-screen row unit without long internal identifiers."""

    labels: list[str] = []
    for row in frame.itertuples(index=False):
        screen = str(row.screen_code)
        if screen.endswith(".0"):
            screen = screen[:-2]
        labels.append(f"{row.gene_symbol} · S{screen}")
    return labels


def _format_scale(value: float) -> str:
    if value >= 10:
        return f"{value:.0f}"
    if value >= 3:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _draw_response_pair(
    fig: plt.Figure,
    spec,
    frame: pd.DataFrame,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> tuple[list[plt.Axes], float]:
    """Draw aligned observed/predicted heatmaps and a reporter-specific scale."""

    grid = spec.subgridspec(
        2,
        2,
        height_ratios=[1.0, 0.105],
        width_ratios=[1.0, 1.0],
        wspace=0.055,
        hspace=0.18,
    )
    scale = max(
        float(np.nanpercentile(np.abs(np.concatenate([truth.ravel(), prediction.ravel()])), 97.5)),
        1e-8,
    )
    labels = _profile_labels(frame)
    axes: list[plt.Axes] = []
    for col, (matrix, title) in enumerate(((truth, "Observed"), (prediction, "Predicted"))):
        ax = fig.add_subplot(grid[0, col])
        axes.append(ax)
        ax.imshow(
            matrix,
            cmap=KO_CMAP,
            norm=Normalize(-scale, scale),
            interpolation="nearest",
            aspect="auto",
            rasterized=True,
        )
        ax.set_title(title, fontsize=5.9, fontweight="bold", pad=1.7)
        ax.set_xticks([])
        if col == 0:
            ax.set_yticks(np.arange(frame.shape[0]), labels, fontsize=4.45)
            ax.tick_params(axis="y", length=0, pad=1.2)
        else:
            ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cax = fig.add_subplot(grid[1, :])
    axes.append(cax)
    cbar = mpl.colorbar.ColorbarBase(
        cax,
        cmap=KO_CMAP,
        norm=Normalize(-scale, scale),
        orientation="horizontal",
        ticks=[-scale, 0, scale],
    )
    s = _format_scale(scale)
    cbar.ax.set_xticklabels([f"−{s}", "0", f"+{s}"])
    cbar.ax.tick_params(labelsize=4.6, length=1.7, width=0.45, pad=0.7)
    cbar.outline.set_linewidth(0.45)
    cbar.set_label("Response (z)", fontsize=6.5, labelpad=0.7)
    return axes, scale


def _density_scatter(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str, limit: float) -> None:
    """Draw a density-aware cloud while retaining all 1,000 reporter-gene points."""

    light_cmap = LinearSegmentedColormap.from_list(
        "density_" + color.replace("#", ""), ["#FFFFFF", color], N=128
    )
    ax.hexbin(
        x,
        y,
        gridsize=31,
        extent=(0, limit, 0, limit),
        mincnt=3,
        cmap=light_cmap,
        bins="log",
        linewidths=0,
        alpha=0.56,
        rasterized=True,
        zorder=1,
    )
    points = ax.scatter(x, y, s=6.0, color=color, alpha=0.58, linewidths=0,
                        rasterized=True, zorder=2)
    points.set_gid("complete_gene_magnitude_cloud")


def _add_prb_zoom(ax: plt.Axes, genes: pd.DataFrame, color: str) -> plt.Axes:
    """Enlarge the crowded origin without removing outliers from the main axes."""
    zoom = ax.inset_axes([0.105, 0.405, 0.285, 0.37], zorder=9)
    # Feed the complete table to both views. Axes limits alone define the inset;
    # the frozen 1,000-gene statistics are neither recomputed nor filtered.
    zoom.scatter(genes.true_magnitude, genes.predicted_magnitude, s=3.0,
                 color=color, alpha=0.62, linewidths=0, rasterized=True)
    zoom.plot([0, 5], [0, 5], color="#91A0A9", lw=0.55,
              ls=(0, (3, 2)), zorder=0)
    zoom.set_xlim(0, 5)
    zoom.set_ylim(0, 5)
    zoom.set_xticks([0, 5])
    zoom.set_yticks([0, 5])
    zoom.tick_params(length=1.5, width=0.5, pad=0.7, labelsize=6.0)
    zoom.set_title("0–5 zoom", fontsize=6.0, fontweight="normal", pad=1.4)
    for spine in zoom.spines.values():
        spine.set_linewidth(0.5)
        spine.set_color("#91A0A9")
    zoom.set_gid("prb_origin_zoom")
    return zoom


def _label_fixed_genes(ax: plt.Axes, genes: pd.DataFrame, slug: str, limit: float) -> None:
    """Use the deterministic labels carried by the prior frozen display contract."""

    labels = list(genes.nlargest(2, "true_magnitude").gene_name.astype(str))
    # SALL3 is identified by the registered microscopy keyline rather than a
    # second, competing text label at the compressed origin.
    offsets = {
        "RPL9": (-3, 4),
        "RPL34": (-3, -7),
        "SNRPG": (-3, 4),
        "RRM1": (-3, -7),
        "TIPARP": (-4, 4),
        "ELP2": (3, 3),
        "SALL3": (3, -8),
    }
    for gene in labels:
        part = genes.loc[genes.gene_name.astype(str).str.upper().eq(gene.upper())]
        if len(part) != 1:
            continue
        row = part.iloc[0]
        dx, dy = offsets.get(gene, (3, 3))
        x = float(row.true_magnitude)
        y = float(row.predicted_magnitude)
        ha = "right" if x > 0.75 * limit else "left"
        if ha == "right" and dx > 0:
            dx = -dx
        ax.annotate(
            gene,
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="bottom",
            fontsize=4.8,
            color=COLORS["ink"],
            zorder=6,
        )


def _add_prb_microscopy(
    ax: plt.Axes,
    genes: pd.DataFrame,
    source_root: Path,
) -> list[plt.Axes]:
    """Add the registered pRb SALL3 same-cell evidence and its exact keyline."""

    selection = pd.read_csv(source_root / "microscopy_selection_manifest.csv").iloc[0]
    display = pd.read_csv(source_root / "microscopy_display_manifest.csv").iloc[0]
    sall3 = genes.loc[genes.gene_name.astype(str).str.upper().eq("SALL3")]
    if len(sall3) != 1 or str(selection.gene).upper() != "SALL3":
        raise RuntimeError("The registered pRb microscopy cell does not map uniquely to SALL3")
    sall3 = sall3.iloc[0]

    with np.load(source_root / "microscopy" / "cell_01.npz", allow_pickle=False) as payload:
        phase = np.asarray(payload["phase"], float)
        fluorescence = np.asarray(payload["fluorescence"], float)
        cell_mask = np.asarray(payload["cell_mask"])
        reporter_mask = np.asarray(payload["reporter_mask"])

    # The images occupy a deliberately large, data-empty part of the pRb plot.
    # Lower the image pair so its titles clear the two-line magnitude summary.
    phase_ax = ax.inset_axes([0.50, 0.35, 0.22, 0.30], zorder=8)
    reporter_ax = ax.inset_axes([0.745, 0.35, 0.22, 0.30], zorder=8)
    phase_ax.imshow(
        phase,
        cmap="gray",
        vmin=float(display.phase_low),
        vmax=float(display.phase_high),
        interpolation="none",
        rasterized=True,
    )
    reporter_ax.imshow(
        fluorescence,
        cmap=REPORTER_CMAP,
        vmin=float(display.fluorescence_low),
        vmax=float(display.fluorescence_high),
        interpolation="none",
        rasterized=True,
    )
    selected = cell_mask == int(selection.segmentation_id)
    if selected.any():
        phase_ax.contour(selected.astype(float), levels=[0.5], colors=["white"], linewidths=0.55)
        reporter_ax.contour(selected.astype(float), levels=[0.5], colors=["white"], linewidths=0.55)
    if np.any(reporter_mask > 0):
        reporter_ax.contour(
            (reporter_mask > 0).astype(float),
            levels=[0.5],
            colors=["#F2C06C"],
            linewidths=0.42,
        )

    for image_ax, label in ((phase_ax, "phase"), (reporter_ax, "reporter")):
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        image_ax.set_title(
            label, fontsize=4.7, pad=1.1, fontweight="normal",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
        )
        for spine in image_ax.spines.values():
            spine.set_color("white")
            spine.set_linewidth(0.5)

    bar_px = 20.0 / float(display.pixel_size_um)
    y_bar = fluorescence.shape[0] - 25
    x_mid = (fluorescence.shape[1] - 1) / 2.0
    x0, x1 = x_mid - bar_px / 2.0, x_mid + bar_px / 2.0
    effects = [path_effects.Stroke(linewidth=2.4, foreground="black"), path_effects.Normal()]
    scale_line, = reporter_ax.plot(
        [x0, x1], [y_bar, y_bar], color="white", lw=1.35,
        solid_capstyle="butt",
    )
    scale_line.set_path_effects(effects)
    scale_label = reporter_ax.text(
        (x0 + x1) / 2,
        y_bar - 8,
        "20 µm",
        color="white",
        fontsize=5.2,
        fontweight="bold",
        ha="center",
        va="bottom",
    )
    scale_label.set_path_effects(effects)

    # Label belongs to the evidence pair, not to the cell image pixels.
    ax.text(
        0.735,
        0.295,
        "SALL3 · S77",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.3},
        zorder=10,
    )

    x_s = float(sall3.true_magnitude)
    y_s = float(sall3.predicted_magnitude)
    ax.scatter(
        [x_s], [y_s], s=20, facecolor="white", edgecolor="#B77B3E", linewidth=0.9, zorder=11
    )
    # Route below the zoom's tick labels, then through the gutter between the
    # zoom and microscopy. Keep the leader clear of both data and SALL3 caption.
    source_xy = ax.transAxes.inverted().transform(ax.transData.transform((x_s, y_s)))
    ax.plot([source_xy[0], 0.045, 0.455, 0.455],
            [source_xy[1], 0.25, 0.25, 0.43], transform=ax.transAxes,
            color="#B77B3E", lw=0.75, zorder=7, solid_joinstyle="round")
    connection = ConnectionPatch(
        xyA=(0.455, 0.43),
        coordsA=ax.transAxes,
        xyB=(0.0, 0.25),
        coordsB=phase_ax.transAxes,
        color="#B77B3E",
        linewidth=0.75,
        connectionstyle="arc3,rad=0",
        arrowstyle="-|>",
        mutation_scale=5.0,
        zorder=7,
        clip_on=False,
    )
    # Keep the connector in this axes' z-order so inset labels remain above it.
    ax.add_artist(connection)
    return [phase_ax, reporter_ax]


def _draw_magnitude_axis(
    ax: plt.Axes,
    genes: pd.DataFrame,
    summary: pd.Series,
    slug: str,
    color: str,
    source_root: Path,
) -> list[plt.Axes]:
    x = genes.true_magnitude.to_numpy(float)
    y = genes.predicted_magnitude.to_numpy(float)
    limit = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.035

    _density_scatter(ax, x, y, color, limit)
    identity_end = 0.78 * limit if slug == "prb" else limit
    ax.plot([0, identity_end], [0, identity_end], color="#91A0A9", lw=0.65,
            ls=(0, (3, 2)), zorder=0)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Observed KO magnitude", labelpad=1.6)
    ax.set_ylabel("Predicted KO magnitude", labelpad=1.8)
    ax.grid(False)
    clean_axis(ax)

    rho = float(summary["magnitude_spearman"])
    ratio = float(summary["magnitude_variance_ratio"])
    ratio_text = f"{ratio:.3f}" if ratio < 0.1 else f"{ratio:.2f}"
    ax.text(
        0.985 if slug == "prb" else 0.015,
        0.985,
        f"ρ = {rho:.2f}\nVar. ratio = {ratio_text}",
        transform=ax.transAxes,
        ha="right" if slug == "prb" else "left",
        va="top",
        fontsize=4.8,
        fontweight="bold",
        linespacing=1.16,
        color=COLORS["ink"],
        zorder=12,
    )

    _label_fixed_genes(ax, genes, slug, limit)

    image_axes: list[plt.Axes] = []
    if slug == "prb":
        image_axes = _add_prb_microscopy(ax, genes, source_root)
        image_axes.append(_add_prb_zoom(ax, genes, color))
    return image_axes


def plot_panel_c(
    fig: plt.Figure,
    host,
    *,
    source_root: Path = PANEL_SOURCE,
    add_letter: bool = True,
    letter: str = "c",
) -> list[plt.Axes]:
    """Draw panel c into ``host`` and return every created Axes."""

    spec, bbox = _host_spec(fig, host)
    summary = pd.read_csv(source_root / "reconstruction_summary.csv").set_index("reporter_slug")
    manifest = pd.read_csv(source_root / "display_manifest.csv").set_index("reporter_slug")
    if list(summary.index) != [x["slug"] for x in REPORTERS]:
        raise RuntimeError("Reporter ordering in the frozen reconstruction summary has changed")

    # The reserved left gutter keeps the first column's gene-screen labels
    # inside the panel rectangle; inter-column spacing carries the labels of
    # the second and third reporter without shrinking their heatmaps.
    padded = spec.subgridspec(1, 2, width_ratios=[0.105, 0.895], wspace=0.0)
    outer = padded[0, 1].subgridspec(1, 3, wspace=0.47)
    all_axes: list[plt.Axes] = []
    for col, reporter in enumerate(REPORTERS):
        slug = reporter["slug"]
        if int(manifest.loc[slug, "n_genes"]) != 1000:
            raise RuntimeError(f"{slug}: expected 1,000 frozen gene rows")
        frame, truth, prediction = _load_profile(slug, source_root)
        genes = pd.read_csv(source_root / slug / "gene_magnitude_scatter.csv.gz")
        if len(genes) != 1000:
            raise RuntimeError(f"{slug}: expected 1,000 gene-magnitude rows")

        col_grid = outer[0, col].subgridspec(
            3,
            1,
            height_ratios=[0.040, 0.560, 0.400],
            hspace=0.28,
        )
        header_ax = fig.add_subplot(col_grid[0, 0])
        all_axes.append(header_ax)
        header_ax.axis("off")
        header_ax.text(
            0.5,
            0.70,
            reporter["name"],
            transform=header_ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color=reporter["color"],
        )
        header_ax.text(
            0.5,
            0.10,
            reporter["mode"],
            transform=header_ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.55,
            color=COLORS["ink"],
        )

        heat_axes, _scale = _draw_response_pair(
            fig, col_grid[1, 0], frame, truth, prediction
        )
        all_axes.extend(heat_axes)

        scatter_ax = fig.add_subplot(col_grid[2, 0])
        all_axes.append(scatter_ax)
        all_axes.extend(
            _draw_magnitude_axis(
                scatter_ax,
                genes,
                summary.loc[slug],
                slug,
                reporter["color"],
                source_root,
            )
        )

    if add_letter:
        fig.text(
            bbox.x0 - 0.016,
            bbox.y1 + 0.002,
            letter,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=COLORS["ink"],
        )
    return all_axes


def main() -> None:
    apply_style()
    mm = 1 / 25.4
    fig = plt.figure(figsize=(183 * mm, 94 * mm))
    plot_panel_c(fig, (0.045, 0.105, 0.94, 0.84), add_letter=True)
    save_figure(fig, OUT_ROOT / "Figure4-c", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()
