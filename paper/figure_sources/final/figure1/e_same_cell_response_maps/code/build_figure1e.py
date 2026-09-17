#!/usr/bin/env python3
"""Build Figure 1e: exact same-cell images and matched KO-response maps."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as path_effects
from scipy.cluster.hierarchy import leaves_list


WIDTH_MM = 123
HEIGHT_MM = 52
MM_TO_IN = 1 / 25.4


FIGURE_ROOT = Path(__file__).resolve().parents[2]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))
from _portable_fonts import configure_sans
sys.path.insert(0, str(FIGURE_ROOT / "code"))
from layout_qa import assert_text_layout  # noqa: E402


PHASE = "#526573"
REPORTER = "#3E6F9C"
GOLD = "#DDA06A"
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
PHASE_CLASS_COLORS = {
    "cell": "#7E8D96",
    "phase2d_vesicular": "#79A8C9",
    "phase2d_vesicular_dark": "#3E6F9C",
    "phase2d_tubular": "#DDA06A",
}
ENDPOINT_CLASS_COLORS = {
    "Level": "#3E6F9C",
    "Extrema/range": "#79A8C9",
    "Dispersion": "#7B6597",
    "Integrated": "#A783AD",
    "Spatial": "#DDA06A",
    "Other": "#91A0A9",
}
FLUOR_CMAP = LinearSegmentedColormap.from_list(
    "ops_fluorescence", ["#020609", "#173947", "#4FA7B8", "#DDF1F3"], N=256
)
KO_CMAP = LinearSegmentedColormap.from_list(
    "ops_ko_response", ["#3E6F9C", "#79A8C9", "#F7F7F4", "#E5A0A8", "#A44758"], N=256
)


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def frozen_input_root() -> Path:
    return bundle_root() / "source_data" / "frozen_inputs"


def configure() -> None:
    configure_sans(mpl, font_manager, FIGURE_ROOT)
    mpl.rcParams.update({
        "font.size": 6.0, "axes.labelsize": 6.0,
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def read_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame = pd.read_csv(path)
    genes = frame.pop("gene").astype(str).to_numpy()
    return genes, frame.to_numpy(float), frame.columns.astype(str).tolist()


def load_exact(key: str) -> tuple[np.ndarray, np.ndarray]:
    path = frozen_input_root() / "B_exact_same_cell_microscopy" / f"{key}_exact_same_cell.npz"
    with np.load(path) as values:
        return np.asarray(values["phase"]), np.asarray(values["fluorescence"])


def endpoint_family(value: object) -> str:
    text = str(value).lower()
    if "integrated" in text or text.endswith("_sum") or "intensity_sum" in text:
        return "Integrated"
    if "std" in text or "mad" in text or "variance" in text:
        return "Dispersion"
    if any(key in text for key in ["max", "min", "quartile", "range"]):
        return "Extrema/range"
    if any(key in text for key in ["displacement", "radial", "edge", "distance"]):
        return "Spatial"
    if "mean" in text or "median" in text or "intensity" in text:
        return "Level"
    return "Other"


def strip_rgba(values: np.ndarray, palette: dict[str, str]) -> np.ndarray:
    return np.asarray([[mpl.colors.to_rgba(palette.get(str(value), "#B8C2C7")) for value in values]])


def case_data(slug: str, anchor_gene: str) -> dict[str, object]:
    root = frozen_input_root() / "G_representative_phase_fluorescence" / slug
    genes, phase, phase_names = read_matrix(root / "phase_ko_response_display_z.csv.gz")
    target_genes, target, target_names = read_matrix(root / "fluorescence_ko_response_display_z.csv.gz")
    if not np.array_equal(genes, target_genes):
        raise RuntimeError(f"Phase and target KO rows do not match for {slug}")
    selected = leaves_list(np.load(root / "ko_linkage.npy"))
    phase_columns = leaves_list(np.load(root / "phase_feature_linkage.npy"))
    target_columns = leaves_list(np.load(root / "fluorescence_feature_linkage.npy"))
    phase_annotations = pd.read_csv(root / "phase_feature_annotations.csv")
    target_annotations = pd.read_csv(root / "fluorescence_feature_annotations.csv")
    descriptor = target_annotations.get("numerical_descriptor", target_annotations.iloc[:, 0]).astype(str)
    target_classes = descriptor.map(endpoint_family).to_numpy()
    rms = np.sqrt(np.nanmean(np.square(phase), axis=1))
    return {
        "genes": genes[selected],
        "phase": phase[selected][:, phase_columns],
        "target": target[selected][:, target_columns],
        "phase_names": np.asarray(phase_names)[phase_columns],
        "target_names": np.asarray(target_names)[target_columns],
        "phase_classes": phase_annotations.iloc[phase_columns].organelle.astype(str).to_numpy(),
        "target_classes": target_classes[target_columns],
        "anchor_gene": anchor_gene,
        "rms": rms[selected],
    }


def clean(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_centered_scale_bar(ax, pixel_size_um: float, crop_px: int) -> None:
    """Draw a legible 20-µm bar at the lower centre of a microscopy crop."""
    fraction = min(0.42, 20.0 / (pixel_size_um * crop_px))
    x0, x1 = 0.5 - fraction / 2.0, 0.5 + fraction / 2.0
    effects = [path_effects.Stroke(linewidth=2.4, foreground="black"), path_effects.Normal()]
    line, = ax.plot([x0, x1], [0.070, 0.070], transform=ax.transAxes,
                    color="white", lw=1.35, solid_capstyle="butt", clip_on=True, zorder=8)
    line.set_path_effects(effects)
    label = ax.text(0.5, 0.105, "20 µm", transform=ax.transAxes,
                    ha="center", va="bottom", color="white", fontsize=6.0,
                    fontweight="bold", clip_on=True, zorder=9)
    label.set_path_effects(effects)


def build() -> None:
    configure()
    manifest = pd.read_csv(
        frozen_input_root() / "B_exact_same_cell_microscopy/display_manifest.csv"
    ).set_index("key")
    cases = [
        ("ferhonox", "fe2+_ferhonox_live-cell_dye", "FeRhoNox", "ALG12", "OPS0047", "Signaling"),
        ("lamp1", "lysosome_lamp1", "LAMP1", "BORCS7", "OPS0011", "Lysosome"),
        ("prb", "prb", "pRb", "FECH", "OPS0077", "Signaling"),
    ]

    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    fig.text(0.004, 0.992, "e", ha="left", va="top", fontsize=8.0, fontweight="bold")
    outer = fig.add_gridspec(
        2, 1, left=0.034, right=0.988, bottom=0.120, top=0.975,
        height_ratios=[0.935, 0.065], hspace=0.115,
    )
    case_grid = outer[0].subgridspec(1, 3, wspace=0.090)
    selection_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []

    for index, (key, slug, reporter_name, gene, screen_id, system) in enumerate(cases):
        row = manifest.loc[key]
        phase_image, reporter_image = load_exact(key)
        case = case_data(slug, gene)
        inner = case_grid[index].subgridspec(
            4, 1, height_ratios=[0.22, 0.57, 0.14, 0.70], hspace=0.090
        )

        title_ax = fig.add_subplot(inner[0]); clean(title_ax)
        title_ax.text(
            0.5, 0.70, f"{reporter_name} · {gene}", ha="center", va="center",
            fontsize=6.0, color=BIOLOGY_COLORS[system], fontweight="bold",
        )

        image_grid = inner[1].subgridspec(1, 2, wspace=0.035)
        for column, (array, cmap, low, high, label) in enumerate([
            (phase_image, "gray", float(row.phase_low), float(row.phase_high), "phase"),
            (reporter_image, FLUOR_CMAP, float(row.fluorescence_low),
             float(row.fluorescence_high), "reporter"),
        ]):
            ax = fig.add_subplot(image_grid[column])
            ax.imshow(array, cmap=cmap, vmin=low, vmax=high,
                      # Direct native-pixel rendering; no synthetic image
                      # interpolation or image enhancement is applied.
                      interpolation="none", rasterized=True)
            ax.set_title(
                label, fontsize=6.0,
                color=PHASE if column == 0 else REPORTER,
                pad=1.4, fontweight="bold",
            )
            clean(ax)
            if column == 1:
                add_centered_scale_bar(
                    ax, pixel_size_um=float(row.pixel_size_um),
                    crop_px=int(row.crop_size_pixels),
                )

        target_dim = int(np.asarray(case["target"]).shape[1])
        widths = [172, max(target_dim, 20)]
        strip_grid = inner[2].subgridspec(
            2, 2, height_ratios=[0.40, 0.60], width_ratios=widths,
            hspace=0.045, wspace=0.035,
        )
        phase_strip = fig.add_subplot(strip_grid[0, 0])
        target_strip = fig.add_subplot(strip_grid[0, 1])
        phase_strip.imshow(
            strip_rgba(np.asarray(case["phase_classes"]), PHASE_CLASS_COLORS),
            aspect="auto", interpolation="nearest",
        )
        target_strip.imshow(
            strip_rgba(np.asarray(case["target_classes"]), ENDPOINT_CLASS_COLORS),
            aspect="auto", interpolation="nearest",
        )
        clean(phase_strip); clean(target_strip)
        phase_label = fig.add_subplot(strip_grid[1, 0]); target_label = fig.add_subplot(strip_grid[1, 1])
        clean(phase_label); clean(target_label)
        phase_label.text(0.45, 0.30, "172D phase", ha="center", va="center",
                         fontsize=6.0, color=PHASE, fontweight="bold")
        target_label.text(0.96, 0.30, f"{target_dim}D target", ha="right", va="center",
                          fontsize=6.0, color=REPORTER, fontweight="bold")

        heat_grid = inner[3].subgridspec(1, 2, width_ratios=widths, wspace=0.035)
        phase_ax = fig.add_subplot(heat_grid[0]); target_ax = fig.add_subplot(heat_grid[1])
        phase_ax.imshow(np.asarray(case["phase"]), aspect="auto", cmap=KO_CMAP,
                        vmin=-3, vmax=3, interpolation="nearest", rasterized=True)
        target_ax.imshow(np.asarray(case["target"]), aspect="auto", cmap=KO_CMAP,
                         vmin=-3, vmax=3, interpolation="nearest", rasterized=True)
        clean(phase_ax); clean(target_ax)
        genes = np.asarray(case["genes"]).astype(str)
        hits = np.flatnonzero(genes == gene)
        if len(hits):
            row_index = int(hits[0])
            for ax in (phase_ax, target_ax):
                ax.axhline(row_index, color=GOLD, lw=0.8)
            phase_ax.scatter([-2.2], [row_index], marker=">", s=9,
                             color=GOLD, edgecolor="none", clip_on=False, zorder=5)
        # Keep the out-of-bounds KO pointer from changing the phase-image data
        # limits.  These exact limits make the feature strips and heatmaps
        # column-aligned in every case.
        phase_ax.set_xlim(-0.5, np.asarray(case["phase"]).shape[1] - 0.5)
        target_ax.set_xlim(-0.5, np.asarray(case["target"]).shape[1] - 0.5)

        for display_row, (gene_name, rms_value) in enumerate(zip(genes, np.asarray(case["rms"]))):
            selection_rows.append({
                "reporter_slug": slug, "reporter": reporter_name, "image_gene": gene,
                "selected_gene": gene_name, "display_row": display_row,
                "phase_response_rms": float(rms_value),
                "selection_rule": "all KO rows in frozen KO-cluster order",
            })
        for position, value in enumerate(np.asarray(case["phase_classes"]).astype(str)):
            feature_rows.append({"reporter_slug": slug, "modality": "phase",
                                 "display_column": position, "feature_class": value})
        for position, value in enumerate(np.asarray(case["target_classes"]).astype(str)):
            feature_rows.append({"reporter_slug": slug, "modality": "target",
                                 "display_column": position, "feature_class": value})
        image_rows.append({
            "case_key": key, "reporter_slug": slug, "reporter": reporter_name,
            "gene": gene, "screen_id": screen_id, "pixel_size_um": float(row.pixel_size_um),
            "crop_size_pixels": int(row.crop_size_pixels), "phase_low": float(row.phase_low),
            "phase_high": float(row.phase_high), "fluorescence_low": float(row.fluorescence_low),
            "fluorescence_high": float(row.fluorescence_high),
        })

    source = bundle_root() / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(source / "ko_display_order.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(source / "feature_display_order.csv", index=False)
    pd.DataFrame(image_rows).to_csv(source / "exact_image_display_manifest.csv", index=False)

    colour_ax = fig.add_subplot(outer[1])
    colourbar = mpl.colorbar.ColorbarBase(
        colour_ax, cmap=KO_CMAP, norm=Normalize(-3, 3),
        orientation="horizontal", ticks=[-3, 0, 3],
    )
    colourbar.outline.set_linewidth(0.45)
    colour_ax.tick_params(labelsize=6.0, length=1.2, pad=1)
    colour_ax.set_xlabel("control-relative KO response (z)", fontsize=6.0, labelpad=1)

    assert_text_layout(fig)
    out = bundle_root() / "figure" / "Figure1-e_same_cell_response_maps"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches=None, pad_inches=0)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches=None, pad_inches=0)
    fig.savefig(out.with_suffix(".svg"), bbox_inches=None, pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    build()
