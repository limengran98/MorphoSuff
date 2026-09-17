#!/usr/bin/env python3
"""Build Figure 3 from frozen ten-model OPS source tables.

The script changes visual grammar only. It does not re-fit models or alter the
frozen aggregation contract (arithmetic fold/direction means within model;
unweighted median across ten models). Microscopy crops are descriptive assets
selected in the frozen v1 manifest independently of model scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from _portable_fonts import configure_sans, format_figure, layout_report, pdf_panel_letters, svg_panel_letters  # noqa: E402


A_DIR = ROOT / "a_recoverability_rank_atlas"
B_DIR = ROOT / "b_biology_endpoint_structure"
C_DIR = ROOT / "c_value_overlap_recovery"
COMPOSITE = ROOT / "composite"

MM_TO_IN = 1 / 25.4
INK = "#202A33"
MUTED = "#6F7E88"
MID = "#7E8D96"
GRID = "#DCE3E7"
WHITE = "#FFFFFF"
TEAL = "#3E6F9C"
TEAL_LIGHT = "#D7E5EF"
RED = "#D56C75"
GOLD = "#DDA06A"
BLUE = "#3E6F9C"

SPLIT_ORDER = ["Field", "Gene", "Whole-screen"]
SPLIT_COLORS = {"Field": BLUE, "Gene": "#DDA06A", "Whole-screen": RED}
DIFFICULTY = {"Low": RED, "Intermediate": GOLD, "High": "#3E6F9C"}
BIOLOGY_COLORS = {
    "Endosome": "#C6A16A", "Lysosome": "#E8943A", "ER/Golgi": "#4F9A70",
    "Cytoskeleton": "#4FA7B8", "Peroxisome": "#D56C75",
    "Stress/PQC": "#CC79A7", "Mitochondria": "#C85A3C",
    "Plasma Membrane": "#9C8C72", "Nucleus": "#3E6F9C",
    "Signaling": "#7B6597",
}
FLUOR_CMAP = LinearSegmentedColormap.from_list(
    "ops_fluorescence", ["#020609", "#173947", "#4FA7B8", "#DDF1F3"], N=256
)
RECOVERY_CMAP = LinearSegmentedColormap.from_list(
    "ops_recovery", ["#E7EEF3", "#79A8C9", "#3E6F9C", "#294A66"], N=256
)


def configure() -> None:
    configure_sans(mpl, fm, ROOT)
    mpl.rcParams.update({
        "font.size": 6.2,
        "axes.titlesize": 7.0, "axes.titleweight": "bold",
        "axes.labelsize": 6.3, "axes.edgecolor": "#6F7E88",
        "axes.linewidth": 0.55, "xtick.labelsize": 5.6,
        "ytick.labelsize": 5.6, "xtick.major.width": 0.5,
        "ytick.major.width": 0.5, "xtick.major.size": 2.0,
        "ytick.major.size": 2.0, "legend.fontsize": 5.3,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def clean(ax: plt.Axes) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MID)
    ax.spines["bottom"].set_color(MID)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(fig: plt.Figure, stem: Path, dpi: int = 600) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    format_figure(fig, 3)
    layout_report(fig, ROOT / 'qa' / (stem.name + '_typography.json'))
    for ext, kwargs in (("png", {"dpi": dpi}), ("pdf", {}), ("svg", {})):
        fig.savefig(stem.with_suffix(f".{ext}"), bbox_inches=None, pad_inches=0,
                    facecolor=WHITE, **kwargs)
    plt.close(fig)


def _prefix_svg_ids(root: ET.Element, prefix: str) -> None:
    """Prevent clip-path and marker-id collisions in a nested SVG composite."""
    mapping = {}
    for element in root.iter():
        old = element.get("id")
        if old:
            mapping[old] = f"{prefix}_{old}"
            element.set("id", mapping[old])
    reference = re.compile(r"#([A-Za-z_][A-Za-z0-9_.:-]*)")

    def rewrite(value: str) -> str:
        return reference.sub(lambda match: "#" + mapping.get(match.group(1), match.group(1)), value)

    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if "#" in value:
                element.set(key, rewrite(value))
        if element.text and "#" in element.text:
            element.text = rewrite(element.text)


def write_vector_svg_composite(placements: list[tuple[Path, float, float, float, float]]) -> None:
    svg_ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", svg_ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.Element(
        f"{{{svg_ns}}}svg",
        {"width": "183mm", "height": "170mm", "viewBox": "0 0 183 170", "version": "1.1"},
    )
    for i, (path, x, y, width, height) in enumerate(placements):
        source = ET.parse(path).getroot()
        _prefix_svg_ids(source, f"panel{i}")
        nested = ET.SubElement(
            root, f"{{{svg_ns}}}svg",
            {"x": str(x), "y": str(y), "width": str(width), "height": str(height),
             "viewBox": source.get("viewBox", "0 0 1 1"), "preserveAspectRatio": "none"},
        )
        for child in list(source):
            nested.append(child)
    svg_panel_letters(root, {"a": (1, 1), "b": (1, 99), "c": (86, 99)})
    ET.ElementTree(root).write(
        COMPOSITE / "Figure3.svg", encoding="utf-8", xml_declaration=True
    )


def write_vector_pdf_composite(placements: list[tuple[Path, float, float, float, float]]) -> None:
    points_per_mm = 72 / 25.4
    document = fitz.open()
    page = document.new_page(width=183 * points_per_mm, height=170 * points_per_mm)
    sources = []
    try:
        for path, x, y, width, height in placements:
            source = fitz.open(path)
            sources.append(source)
            rect = fitz.Rect(x * points_per_mm, y * points_per_mm,
                             (x + width) * points_per_mm, (y + height) * points_per_mm)
            page.show_pdf_page(rect, source, pno=0, keep_proportion=False, overlay=True)
        pdf_panel_letters(page, {k: (x * points_per_mm, y * points_per_mm)
                                for k, (x, y) in {"a": (1, 1), "b": (1, 99), "c": (86, 99)}.items()})
        document.save(COMPOSITE / "Figure3.pdf", garbage=4, deflate=True)
    finally:
        for source in sources:
            source.close()
        document.close()


def difficulty_key(value: object) -> str:
    text = str(value).lower()
    if text.startswith(("hard", "low")):
        return "Low"
    if text.startswith("intermediate"):
        return "Intermediate"
    return "High"


def rgb_image(image: np.ndarray, low: float, high: float, cmap: object) -> np.ndarray:
    values = Normalize(vmin=float(low), vmax=float(high), clip=True)(image)
    mapping = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    return np.asarray(mapping(values)[..., :3], dtype=float)


def image_montage(npz_path: Path, record: pd.Series) -> np.ndarray:
    with np.load(npz_path) as data:
        phase = rgb_image(data["phase"], record.phase_low, record.phase_high, "gray")
        reporter = rgb_image(data["fluorescence"], record.fluorescence_low,
                             record.fluorescence_high, FLUOR_CMAP)
    gap = np.ones((phase.shape[0], 7, 3), dtype=float)
    return np.concatenate([phase, gap, reporter], axis=1)


def add_scale_bar(ax: plt.Axes, *, pixel_size_um: float) -> None:
    # The montage is 775 pixels wide; the reporter crop occupies x=391..775.
    length = 20.0 / pixel_size_um
    # Match Figure 1e: bottom-centred in the reporter crop with a black halo
    # that remains legible over either bright or dark image structures.
    reporter_center = (391.0 + 774.0) / 2.0
    x0, x1 = reporter_center - length / 2.0, reporter_center + length / 2.0
    y = 358.0
    effects = [path_effects.Stroke(linewidth=2.4, foreground="black"), path_effects.Normal()]
    line, = ax.plot([x0, x1], [y, y], color=WHITE, lw=1.35,
                    solid_capstyle="butt", zorder=5)
    line.set_path_effects(effects)
    label = ax.text((x0 + x1) / 2, y - 8, "20 µm", color=WHITE, fontsize=5.2,
                    fontweight="bold", ha="center", va="bottom", zorder=6)
    label.set_path_effects(effects)


def selected_image_rows() -> pd.DataFrame:
    manifest = pd.read_csv(A_DIR / "source_data/image_selection_manifest.csv")
    selections = [
        ("Reporter", "Whole-screen", "fastact_actr3_ko"),
        ("Physical screen", "Field", "lamp1_borcs7_ntc"),
        ("Reporter × screen", "Gene", "5xupre_hspa5_ko"),
    ]
    rows = []
    for unit, split, asset in selections:
        q = manifest.loc[
            manifest.unit.eq(unit) & manifest.split.eq(split) & manifest.asset_key.eq(asset)
        ]
        if len(q) != 1:
            raise RuntimeError(f"Expected one frozen image for {unit}/{split}/{asset}; found {len(q)}")
        rows.append(q.iloc[0])
    out = pd.DataFrame(rows)
    out.to_csv(A_DIR / "source_data/selected_image_manifest.csv", index=False)
    return out


def plot_difficulty_strip(
    strip: plt.Axes,
    counts: pd.DataFrame,
    unit: str,
    split: str,
    *,
    show_rank_axis: bool = False,
) -> None:
    strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)
    q = counts.loc[counts.unit.eq(unit) & counts.split.eq(split)].copy()
    q["key"] = q.difficulty.map(difficulty_key)
    total = int(q.total.iloc[0])
    left = 0.0
    for key in ("Low", "Intermediate", "High"):
        n = int(q.loc[q.key.eq(key), "n"].iloc[0])
        frac = n / total
        strip.add_patch(Rectangle((left, 0), frac, 1, facecolor=DIFFICULTY[key],
                                  edgecolor=WHITE, linewidth=.35))
        if frac >= .13:
            strip.text(left + frac / 2, .5, str(n), ha="center", va="center",
                       fontsize=4.35, color=WHITE, fontweight="bold")
        left += frac
    strip.set_xlim(0, 1); strip.set_ylim(0, 1)
    if show_rank_axis:
        strip.set_xticks([0, .5, 1], ["0", "50", "100"])
        strip.tick_params(axis="x", length=1.8, width=.45, pad=1.0, labelsize=5.5)
        strip.set_xlabel("rank percentile", labelpad=1.2, fontsize=5.9)
    else:
        strip.set_xticks([])


def plot_biology_strip(
    strip: plt.Axes,
    ranked: pd.DataFrame,
    x: np.ndarray,
    *,
    unit: str,
    split: str,
    identifier: str,
) -> pd.DataFrame:
    """Draw system tiles in the exact left-to-right order of one rank spectrum.

    Reporter and reporter-by-screen assay tiles contain one system. A physical
    screen can contain reporters from several systems, so its tile is
    subdivided rather than assigned an unsupported dominant category.
    """
    required = {identifier, "system_components"}
    missing = sorted(required.difference(ranked.columns))
    if missing:
        raise RuntimeError(f"Biological-system strip is missing columns: {missing}")
    if len(ranked) != len(x):
        raise RuntimeError(
            f"Rank/strip length mismatch for {unit}/{split}: {len(ranked)} vs {len(x)}"
        )

    if len(x) > 1:
        step = 100.0 / (len(x) - 1)
        edges = np.r_[x[0] - step / 2, (x[:-1] + x[1:]) / 2, x[-1] + step / 2]
    else:
        edges = np.array([0.0, 100.0])
    rows = []
    for i, row in ranked.reset_index(drop=True).iterrows():
        components = tuple(row.system_components)
        if not components:
            raise RuntimeError(f"No biological system for {unit}/{split}/{row[identifier]}")
        unknown = sorted(set(components).difference(BIOLOGY_COLORS))
        if unknown:
            raise RuntimeError(f"Unknown biological systems for {unit}/{split}: {unknown}")
        tile_left = edges[i]
        tile_width = edges[i + 1] - edges[i]
        component_width = tile_width / len(components)
        for component_i, category in enumerate(components):
            colour = BIOLOGY_COLORS[category]
            strip.add_patch(Rectangle(
                (tile_left + component_i * component_width, 0), component_width, 1,
                facecolor=colour, edgecolor="none",
            ))
            rows.append({
                "unit": unit,
                "split": split,
                "entity_id": row[identifier],
                "rank_low_to_high": i + 1,
                "rank_percentile": float(x[i]),
                "component_index": component_i + 1,
                "n_components": len(components),
                "biological_category": category,
                "category_color": colour,
            })
        strip.add_patch(Rectangle(
            (tile_left, 0), tile_width, 1,
            facecolor="none", edgecolor=WHITE, linewidth=.13,
        ))
    strip.set_xlim(-2, 102); strip.set_ylim(0, 1)
    strip.set_xticks([]); strip.set_yticks([])
    for spine in strip.spines.values():
        spine.set_visible(False)
    return pd.DataFrame(rows)


def build_panel_a() -> None:
    src = A_DIR / "source_data"
    reporter = pd.read_csv(src / "reporter_consensus_10method.csv")
    reporter_meta = reporter[[
        "reporter_slug", "short_name", "biological_category", "category_color"
    ]].drop_duplicates()
    conflicts = reporter_meta.groupby("reporter_slug").biological_category.nunique()
    if int(conflicts.max()) != 1:
        raise RuntimeError("A reporter maps to more than one biological system")
    reporter_meta = reporter_meta.drop_duplicates("reporter_slug")
    reporter["system_components"] = reporter.biological_category.map(lambda value: (str(value),))

    assay = pd.read_csv(src / "assay_consensus_10method.csv")
    assay["reporter_slug"] = assay.assay_id.str.split("::").str[0]
    assay["screen_id"] = assay.assay_id.str.split("::").str[-1]
    assay = assay.merge(reporter_meta, on="reporter_slug", how="left", validate="many_to_one")
    if assay.biological_category.isna().any():
        missing_reporters = sorted(assay.loc[assay.biological_category.isna(), "reporter_slug"].unique())
        raise RuntimeError(f"Assays lack reporter system metadata: {missing_reporters}")
    assay["system_components"] = assay.biological_category.map(lambda value: (str(value),))

    system_order = {name: i for i, name in enumerate(BIOLOGY_COLORS)}
    screen_systems = (
        assay.groupby(["split_label", "screen_id"], sort=False).biological_category
        .agg(lambda values: tuple(sorted(set(map(str, values)), key=system_order.__getitem__)))
        .rename("system_components")
        .reset_index()
    )
    screen = pd.read_csv(src / "screen_consensus_10method.csv").merge(
        screen_systems, on=["split_label", "screen_id"], how="left", validate="one_to_one"
    )
    if screen.system_components.isna().any():
        missing_screens = sorted(screen.loc[screen.system_components.isna(), "screen_id"].unique())
        raise RuntimeError(f"Screens lack assay-derived system metadata: {missing_screens}")

    tables = {
        "Reporter": (reporter, "reporter_slug"),
        "Physical screen": (screen, "screen_id"),
        "Reporter × screen": (assay, "assay_id"),
    }
    counts = pd.read_csv(src / "difficulty_stratum_counts.csv")
    images = selected_image_rows()

    fig = plt.figure(figsize=(183 * MM_TO_IN, 98 * MM_TO_IN))
    outer = fig.add_gridspec(
        4, 4, left=.055, right=.992, bottom=.090, top=.972,
        height_ratios=[.068, 1, 1, 1], width_ratios=[1, 1, 1, .70],
        hspace=.30, wspace=.15,
    )
    title_ax = fig.add_subplot(outer[0, :]); clean(title_ax)
    fig.text(.006, .988, "a", fontsize=9.0, fontweight="bold", color=INK,
                  ha="left", va="top")
    title_ax.text(.065, .90,
                  "Recovery varies by evaluation scope and analysis unit",
                  fontsize=5.4, fontweight="bold", color=INK,
                  ha="left", va="top")
    x0 = .700
    for offset, key in ((0.0, "Low"), (.096, "Intermediate"), (.225, "High")):
        title_ax.add_patch(Rectangle((x0 + offset, .64), .014, .18,
                                     color=DIFFICULTY[key], transform=title_ax.transAxes))
        title_ax.text(x0 + offset + .019, .89, key.lower(), color=INK,
                      fontsize=5.15, va="top")

    unit_labels = {"Reporter": "Reporter", "Physical screen": "Physical screen",
                   "Reporter × screen": "Reporter × screen assay"}
    highlight_rows = []
    rank_rows = []
    biology_strip_rows = []
    for row_i, (unit, (data, identifier)) in enumerate(tables.items(), start=1):
        image_record = images.loc[images.unit.eq(unit)].iloc[0]
        for col_i, split in enumerate(SPLIT_ORDER):
            bottom_row = row_i == 3
            cell = outer[row_i, col_i].subgridspec(
                3, 1,
                height_ratios=[.805, .055, .14] if bottom_row else [.84, .055, .105],
                hspace=.030,
            )
            ax = fig.add_subplot(cell[0])
            biology_ax = fig.add_subplot(cell[1])
            strip_ax = fig.add_subplot(cell[2])
            q = data.loc[data.split_label.eq(split)].sort_values(
                "consensus_ko_pearson", kind="stable"
            ).reset_index(drop=True)
            x = np.linspace(0, 100, len(q)) if len(q) > 1 else np.array([50.0])
            keys = q.difficulty.map(difficulty_key)
            colors = [DIFFICULTY[k] for k in keys]
            ax.axhspan(.20, .60, color=mpl.colors.to_rgba(RED, .045), zorder=0)
            ax.axhspan(.60, .75, color=mpl.colors.to_rgba(GOLD, .045), zorder=0)
            ax.axhspan(.75, 1.00, color=mpl.colors.to_rgba(TEAL, .045), zorder=0)
            ax.vlines(x, q.q25, q.q75, color="#AEB9BE", linewidth=.6, alpha=.82, zorder=1)
            ax.plot(x, q.consensus_ko_pearson, color="#BCC6CA", lw=.52, zorder=1)
            ax.scatter(x, q.consensus_ko_pearson, s=10.5, c=colors,
                       edgecolor=WHITE, linewidth=.32, zorder=3)
            ax.axhline(.60, color=RED, lw=.5, ls=(0, (2.4, 2.0)), alpha=.75)
            ax.axhline(.75, color=GOLD, lw=.5, ls=(0, (2.4, 2.0)), alpha=.75)
            ax.set_xlim(-2, 102); ax.set_ylim(.20, 1.005)
            ax.grid(axis="y", color=GRID, lw=.42, zorder=0)
            if col_i == 0:
                ax.set_ylabel(unit_labels[unit], labelpad=4.2, fontsize=5.9)
                ax.set_yticks([.4, .6, .8, 1.0])
            else:
                ax.set_yticks([.4, .6, .8, 1.0]); ax.set_yticklabels([])
            if row_i == 1:
                ax.set_title(split, color=SPLIT_COLORS[split], fontsize=6.9,
                             fontweight="bold", pad=2.0)
            ax.set_xticks([])
            ax.text(.015, .965, f"n={len(q)}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=4.75, color=INK)
            despine(ax)
            count_unit = "Reporter×screen assay" if unit == "Reporter × screen" else unit
            plot_difficulty_strip(
                strip_ax, counts, count_unit, split, show_rank_axis=bottom_row
            )
            biology_strip_rows.append(plot_biology_strip(
                biology_ax, q, x, unit=unit, split=split, identifier=identifier,
            ))

            for rank, row in q.iterrows():
                rank_rows.append({"unit": unit, "split": split,
                                  "entity_id": row[identifier], "rank_low_to_high": rank + 1,
                                  "rank_percentile": x[rank],
                                  "consensus_ko_pearson": row.consensus_ko_pearson,
                                  "q25": row.q25, "q75": row.q75,
                                  "difficulty": row.difficulty})
            if split == str(image_record.split):
                match = q[identifier].astype(str).eq(str(image_record.entity_key))
                if int(match.sum()) != 1:
                    raise RuntimeError(f"Image point not found for {unit}/{split}/{image_record.entity_key}")
                idx = int(np.flatnonzero(match.to_numpy())[0])
                ax.scatter([x[idx]], [q.loc[idx, "consensus_ko_pearson"]], s=28,
                           facecolor=WHITE, edgecolor=INK, linewidth=.9, zorder=5)
                highlight_rows.append({"unit": unit, "split": split,
                                       "entity_id": image_record.entity_key,
                                       "asset_key": image_record.asset_key,
                                       "display_label": image_record.display_label,
                                       "rank_percentile": x[idx],
                                       "score": q.loc[idx, "consensus_ko_pearson"]})

        card = fig.add_subplot(outer[row_i, 3]); clean(card)
        montage = image_montage(src / f"{image_record.asset_key}.npz", image_record)
        # ``montage`` is constructed directly from deposited raw float32
        # phase/reporter arrays; do not synthesize pixels during rendering.
        card.imshow(montage, interpolation="none", rasterized=True)
        card.add_patch(Rectangle((.5, .5), montage.shape[1]-1, montage.shape[0]-1,
                                 fill=False, edgecolor=INK, linewidth=.7))
        add_scale_bar(card, pixel_size_um=float(image_record.pixel_size_um))
        card.text(.02, 1.045, str(image_record.display_label), transform=card.transAxes,
                  ha="left", va="bottom", fontsize=5.25, fontweight="bold", color=INK)
        # The longest first-row case label needs its split tag on a second
        # baseline; the other two short title pairs retain their alignment.
        split_y = 1.27
        card.text(.98, split_y, f"{image_record.split}", transform=card.transAxes,
                  ha="right", va="bottom", fontsize=4.65,
                  fontweight="bold", color=SPLIT_COLORS[str(image_record.split)])
        card.text(.25, -.018, "phase", transform=card.transAxes, ha="center",
                  va="top", fontsize=4.5, color=INK)
        card.text(.75, -.018, "reporter", transform=card.transAxes, ha="center",
                  va="top", fontsize=4.5, color=TEAL, fontweight="bold")

    pd.DataFrame(rank_rows).to_csv(src / "plotted_rank_order.csv", index=False)
    pd.DataFrame(highlight_rows).to_csv(src / "plotted_image_callouts.csv", index=False)
    if len(biology_strip_rows) != 9:
        raise RuntimeError(f"Expected nine biological-system strips; created {len(biology_strip_rows)}")
    biology_strip_table = pd.concat(biology_strip_rows, ignore_index=True)
    biology_strip_table.to_csv(src / "plotted_biological_system_strips.csv", index=False)
    legacy_strip = src / "plotted_reporter_gene_biology_strip.csv"
    if legacy_strip.exists():
        legacy_strip.unlink()
    save(fig, A_DIR / "figure/Figure3-a_recoverability_rank_atlas")


def build_panel_b() -> None:
    src = B_DIR / "source_data"
    reporter = pd.read_csv(src / "reporter_consensus_10method.csv")
    reporter = reporter.loc[reporter.split_label.eq("Gene")].copy()
    bio = pd.read_csv(src / "biology_system_summary.csv").sort_values("mean", ascending=False)
    ep_raw = pd.read_csv(src / "reporter_endpoint_family_means.csv")
    endpoint = pd.read_csv(src / "endpoint_family_mean_ci.csv").sort_values("mean", ascending=False)

    fig = plt.figure(figsize=(82 * MM_TO_IN, 72 * MM_TO_IN))
    gs = fig.add_gridspec(2, 1, left=.335, right=.875, bottom=.13, top=.935,
                          height_ratios=[10, 6], hspace=.19)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    rng = np.random.default_rng(20260827)
    for ax, summary, raw_frame, group_col, value_col, group_title in [
        (axes[0], bio, reporter, "biological_category", "consensus_ko_pearson", "Biological systems"),
        (axes[1], endpoint, ep_raw, "endpoint_superfamily", "pearson", "Endpoint families"),
    ]:
        if group_col == "biological_category":
            label_col, low_col, high_col = "biological_category", "ci95_low", "ci95_high"
        else:
            label_col, low_col, high_col = "endpoint_superfamily", "bootstrap_ci_low", "bootstrap_ci_high"
        y = np.arange(len(summary))[::-1]
        for yi, (_, row) in zip(y, summary.iterrows()):
            label = str(row[label_col])
            color = BIOLOGY_COLORS.get(label, TEAL)
            raw = raw_frame.loc[raw_frame[group_col].eq(label), value_col].dropna().to_numpy(float)
            jitter = rng.uniform(-.16, .16, len(raw))
            ax.scatter(raw, yi + jitter, s=9.5,
                       facecolor=mpl.colors.to_rgba(color, .27), edgecolor="none", zorder=2)
            ax.plot([row[low_col], row[high_col]], [yi, yi], color=color, lw=1.45,
                    solid_capstyle="round", zorder=3)
            ax.scatter([row["mean"]], [yi], s=28, color=color, edgecolor=WHITE,
                       linewidth=.7, zorder=4)
            n = int(row.n_reporters)
            ax.text(1.025, yi, f"n={n}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=4.65, color=INK,
                    clip_on=False)
        labels = list(summary[label_col].astype(str))
        labels = [x.replace("Plasma Membrane", "Plasma membrane")
                    .replace("Intensity ", "") for x in labels]
        ax.set_yticks(y, labels)
        ax.set_xlim(.30, 1.0); ax.set_xticks([.4, .6, .8, 1.0])
        ax.axvspan(.30, .60, color=mpl.colors.to_rgba(RED, .035), zorder=0)
        ax.axvspan(.60, .75, color=mpl.colors.to_rgba(GOLD, .035), zorder=0)
        ax.axvspan(.75, 1.00, color=mpl.colors.to_rgba(TEAL, .035), zorder=0)
        ax.axvline(.60, color=RED, lw=.45, ls=(0, (2.4, 2.0)), alpha=.65)
        ax.axvline(.75, color=GOLD, lw=.45, ls=(0, (2.4, 2.0)), alpha=.65)
        ax.grid(axis="x", color=GRID, lw=.42, zorder=0)
        ax.set_title(group_title, loc="left", fontsize=6.6, pad=3.0,
                     fontweight="bold", color=INK)
        despine(ax)
    axes[0].set_xticklabels([])
    axes[1].set_xlabel("held-out-gene Pearson r")
    fig.text(.018, .965, "b", ha="left", va="top", fontsize=8.2,
             fontweight="bold", color=INK)
    fig.text(.30, .972,
             "Highest means: Endosome/Lysosome and level/range endpoints",
             ha="left", va="top", fontsize=5.2, fontweight="bold", color=INK)
    save(fig, B_DIR / "figure/Figure3-b_biology_endpoint_structure")


def build_panel_c() -> None:
    src = C_DIR / "source_data"
    data = pd.read_csv(src / "canonical_reporter_value_overlap_recoverability.csv")
    assoc = pd.read_csv(src / "association_summary.csv")
    predictors = [
        ("original_marker_map_full", "Full depth", "#DDA06A", "o"),
        ("original_marker_map_300_cells_per_guide", "Matched depth", GOLD, "D"),
        ("original_phase_marker_tfidf_pearson", "Phase–marker overlap", TEAL, "^"),
    ]

    fig = plt.figure(figsize=(98 * MM_TO_IN, 72 * MM_TO_IN))
    gs = fig.add_gridspec(1, 2, left=.145, right=.975, bottom=.275, top=.91,
                          width_ratios=[1.0, 1.28], wspace=.20)
    ax = fig.add_subplot(gs[0])
    x = data.original_marker_map_full.to_numpy(float)
    y = data.original_phase_marker_tfidf_pearson.to_numpy(float)
    score = data.consensus10_gene_pearson.to_numpy(float)
    norm = Normalize(vmin=.45, vmax=.94)
    scatter = ax.scatter(x, y, c=score, cmap=RECOVERY_CMAP, norm=norm, s=24,
                         alpha=.88, edgecolor=WHITE, linewidth=.45, zorder=3)
    ax.grid(color=GRID, lw=.42, zorder=0)
    ax.axhline(0, color=MID, lw=.45, zorder=1)
    ax.set_xlabel("marker value")
    ax.set_ylabel("phase–marker overlap")
    ax.set_xlim(0, .135); ax.set_ylim(-.025, .295)
    ax.set_xticks([0, .05, .10], ["0", "0.05", "0.10"])
    ax.set_yticks([0, .1, .2])
    despine(ax)
    label_specs = {
        "pRb*": (28, -9, "left"), "NPM1*": (20, 3, "left"),
        "LAMP1": (25, 3, "left"), "LysoTracker live-cell dye": (-6, -9, "right"),
        "FastAct_SPY555 Live Cell Dye": (-6, 7, "right"),
    }
    label_rows = []
    for name, (dx, dy, ha) in label_specs.items():
        q = data.loc[data.short_name.eq(name)]
        if q.empty:
            continue
        row = q.iloc[0]
        label = name.replace("*", "").replace(" live-cell dye", "").replace("_SPY555 Live Cell Dye", "")
        ax.annotate(label, (row.original_marker_map_full, row.original_phase_marker_tfidf_pearson),
                    xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=4.65, color=INK, zorder=5,
                    bbox=dict(boxstyle="square,pad=.10", facecolor=WHITE,
                              edgecolor="none", alpha=.86),
                    arrowprops=dict(arrowstyle="-", color=MID, lw=.38,
                                    shrinkA=1.5, shrinkB=2.5))
        label_rows.append({"short_name": name, "display_label": label,
                           "x": row.original_marker_map_full,
                           "y": row.original_phase_marker_tfidf_pearson,
                           "recoverability": row.consensus10_gene_pearson})
    pd.DataFrame(label_rows).to_csv(src / "plotted_direct_labels.csv", index=False)
    pos = ax.get_position()
    cax = fig.add_axes([pos.x0, .105, pos.width, .024])
    cbar = fig.colorbar(scatter, cax=cax, orientation="horizontal")
    cbar.set_ticks([.5, .7, .9])
    cbar.set_label("directed recovery (Pearson r)", labelpad=.4, fontsize=5.2)
    cbar.outline.set_linewidth(.45)

    right = gs[1].subgridspec(2, 1, height_ratios=[.18, .82], hspace=.02)
    header = fig.add_subplot(right[0]); clean(header)
    body = right[1].subgridspec(1, 3, width_ratios=[1.05, 1.20, 1.85], wspace=.04)
    label_ax = fig.add_subplot(body[0]); clean(label_ax)
    forest = fig.add_subplot(body[1])
    value_ax = fig.add_subplot(body[2]); clean(value_ax)
    label_map = {p: label for p, label, _, _ in predictors}
    color_map = {p: color for p, _, color, _ in predictors}
    marker_map = {p: marker for p, _, _, marker in predictors}
    q = assoc.loc[assoc.predictor.isin(label_map)].copy()
    q["order"] = q.predictor.map({p: i for i, (p, _, _, _) in enumerate(predictors)})
    q = q.sort_values("order")
    yy = np.array([2.55, 1.65, .35])
    for yi, (_, row) in zip(yy, q.iterrows()):
        color = color_map[row.predictor]
        forest.plot([row.ci95_low, row.ci95_high], [yi, yi], color=color, lw=1.6,
                    solid_capstyle="round", zorder=2)
        filled = row.predictor != "original_marker_map_300_cells_per_guide"
        forest.scatter([row.spearman_rho], [yi], s=34, marker=marker_map[row.predictor],
                       facecolor=color if filled else WHITE, edgecolor=color,
                       linewidth=.85, zorder=3)
        value_ax.text(.02, yi,
                      f"{row.spearman_rho:.2f}\n({row.ci95_low:.2f}–{row.ci95_high:.2f})",
                      ha="left", va="center", fontsize=4.15, color=INK)
    forest.axvline(0, color=MID, lw=.65, zorder=1)
    forest.set_xlim(-.25, .65)
    forest.set_ylim(-.05, 3.05)
    forest.set_xticks([0, .5], ["0", "0.5"])
    forest.tick_params(axis="x", labelsize=4.4, pad=1.0)
    forest.set_yticks([])
    label_ax.set_ylim(-.05, 3.05)
    value_ax.set_ylim(-.05, 3.05)
    for yi, predictor in zip(yy, q.predictor):
        label = label_map[predictor].replace('Phase–marker overlap', 'Phase–marker\noverlap')
        label_ax.text(.98, yi, label, ha="right", va="center", fontsize=4.75,
                      color=INK, clip_on=False)
    label_ax.text(.98, 2.94, "Marker value", ha="right", va="bottom",
                  fontsize=4.45, fontweight="bold", color=INK)
    for small_ax in (label_ax, forest, value_ax):
        small_ax.axhline(1.02, color=GRID, lw=.48, zorder=0)
    forest.grid(axis="x", color=GRID, lw=.42, zorder=0)
    forest.set_xlabel("Spearman ρ")
    header.text(0, .98, "Association with recovery",
                ha="left", va="top", fontsize=5.65, fontweight="bold", color=INK)
    header.text(0, .48, "Spearman ρ (reporter-bootstrap 95% CI; n=52)",
                ha="left", va="top", fontsize=4.45, color=INK)
    despine(forest)

    q.assign(
        display_label=q.predictor.map(label_map),
        glyph=q.predictor.map(marker_map),
    ).to_csv(src / "plotted_association_forest.csv", index=False)

    fig.text(.018, .965, "c", ha="left", va="top", fontsize=8.2,
             fontweight="bold", color=INK)
    fig.text(.10, .965,
             "Marker value associates with recovery; phase-overlap CI crosses 0",
             ha="left", va="top", fontsize=5.2, fontweight="bold", color=INK)
    save(fig, C_DIR / "figure/Figure3-c_value_overlap_recovery")


def build_composite() -> None:
    COMPOSITE.mkdir(parents=True, exist_ok=True)
    # Preserve live vectors, add compositor-owned labels, then render proofs.
    svg_placements = [
        (A_DIR / "figure/Figure3-a_recoverability_rank_atlas.svg", 0, 0, 183, 98),
        (B_DIR / "figure/Figure3-b_biology_endpoint_structure.svg", 0, 98, 82, 72),
        (C_DIR / "figure/Figure3-c_value_overlap_recovery.svg", 85, 98, 98, 72),
    ]
    pdf_placements = [
        (A_DIR / "figure/Figure3-a_recoverability_rank_atlas.pdf", 0, 0, 183, 98),
        (B_DIR / "figure/Figure3-b_biology_endpoint_structure.pdf", 0, 98, 82, 72),
        (C_DIR / "figure/Figure3-c_value_overlap_recovery.pdf", 85, 98, 98, 72),
    ]
    write_vector_svg_composite(svg_placements)
    write_vector_pdf_composite(pdf_placements)
    # Raster proofs must include the compositor-owned letters too.
    with fitz.open(COMPOSITE / "Figure3.pdf") as document:
        pix = document[0].get_pixmap(dpi=600, alpha=False)
        pix.save(COMPOSITE / "Figure3.png")
        Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(
            COMPOSITE / "Figure3.tiff", dpi=(600, 600), compression="raw")


def write_integrity_report() -> None:
    paths = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".npz"}
        and "source_data" in path.relative_to(ROOT).parts
    )
    records = []
    for path in paths:
        records.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                        "sha256": sha256(path)})
    pd.DataFrame(records).to_csv(ROOT / "source_manifest_sha256.csv", index=False)
    payload = {
        "aggregation": "arithmetic fold/direction mean within method; unweighted median across ten frozen models",
        "model_spread": "Q25-Q75 across ten models; descriptive, not biological confidence interval",
        "difficulty_thresholds": {"low": "r < 0.60", "intermediate": "0.60 <= r < 0.75", "high": "r >= 0.75"},
        "panel_a_units": {"Field": [52, 73, 99], "Gene": [52, 73, 99], "Whole-screen": [34, 66, 81]},
        "panel_a_biology_strips": "nine entity-aligned strips; multi-system physical-screen tiles are subdivided",
        "microscopy_role": "descriptive frozen callouts; no image contributes to the plotted statistic",
        "font": "DejaVu Sans; shared 9-pt letters, 6-pt text floor; live SVG text",
    }
    (ROOT / "source_contract.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=["a", "b", "c", "all"], default="all")
    args = parser.parse_args()
    configure()
    if args.panel in {"a", "all"}:
        build_panel_a()
    if args.panel in {"b", "all"}:
        build_panel_b()
    if args.panel in {"c", "all"}:
        build_panel_c()
    if args.panel == "all":
        build_composite()
        write_integrity_report()


if __name__ == "__main__":
    main()
