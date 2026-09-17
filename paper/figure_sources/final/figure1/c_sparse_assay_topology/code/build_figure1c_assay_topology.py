#!/usr/bin/env python3
"""Build Figure 1c: the frozen OPS reporter-by-screen acquisition topology.

This is a descriptive enumeration.  One glyph is one observed reporter-screen
assay; structurally unassayed combinations remain blank.  The code deliberately
does not use recoverability or any downstream prediction outcome for ordering.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.transforms import Bbox


FIGURE_ROOT = Path(__file__).resolve().parents[2]
if str(FIGURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIGURE_ROOT))
from _portable_fonts import configure_sans  # noqa: E402
sys.path.insert(0, str(FIGURE_ROOT / "code"))
from layout_qa import assert_text_layout  # noqa: E402


WIDTH_MM = 183
HEIGHT_MM = 85
MM_TO_IN = 1 / 25.4

INK = "#202A33"
MUTED = "#6F7E88"
MID_GREY = "#91A0A9"
GRID = "#DCE3E7"
PALE = "#EFF2F4"
SCREEN_THREAD = "#AEBAC1"
WHITE = "#FFFFFF"
PHASE = "#526573"

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
BIOLOGY_ORDER = list(BIOLOGY_COLORS)

ANCHORS = {
    "lysosome_lamp1": "LAMP1",
    "fe2+_ferhonox_live-cell_dye": "FeRhoNox",
    "prb": "pRb",
}

# The x-axis is categorical, so v3 assigns different visual pitch to the two
# acquisition regimes. All 73 screens remain explicit, but the 56 one-assay
# columns are compacted and the 17 multi-assay columns are expanded.
N_SINGLE_SCREENS = 56
SINGLE_STEP = 0.68
MULTI_START = 40.20
MULTI_STEP = 1.85
SINGLE_LAST = (N_SINGLE_SCREENS - 1) * SINGLE_STEP
MULTI_LAST = MULTI_START + (73 - N_SINGLE_SCREENS - 1) * MULTI_STEP
SINGLE_EDGE = SINGLE_LAST + SINGLE_STEP / 2
MULTI_EDGE = MULTI_START - MULTI_STEP / 2
DISPLAY_LEFT = -0.55
DISPLAY_RIGHT = MULTI_LAST + MULTI_STEP / 2


def remap_screen_columns(values: object) -> np.ndarray:
    columns = np.asarray(values, dtype=float)
    return np.where(
        columns < N_SINGLE_SCREENS,
        columns * SINGLE_STEP,
        MULTI_START + (columns - N_SINGLE_SCREENS) * MULTI_STEP,
    )

# Reporter names that recur as predeclared biological examples in Figures 2--6.
# This list is frozen independently of the appearance of this panel and contains
# no model-performance values.
MANUSCRIPT_REPORTERS = {
    "late_endosome_rab7a",
    "lysosome_lysotracker_live-cell_dye",
    "lysosome_lamp1",
    "autophagosome_map1lc3b",
    "er_sec61b",
    "er_golgi_cop-ii_sec23a",
    "actin_filament_fastact_spy555_live_cell_dye",
    "peroxisome_peroxi_spy650_live_cell_dye",
    "5xupre",
    "chaperones_hspa1b",
    "mitochondria_tomm20",
    "plasma_membrane_wga",
    "nuclei_hoechst",
    "nucleoli_npm1",
    "nucleolus-gc_npm3",
    "fe2+_ferhonox_live-cell_dye",
    "c-myc",
    "prb",
    "ps6",
}

# A deliberately suppressed long row label. The reporter remains in the
# matrix and its system colour, assay glyphs and recurrence track are intact.
# TOMM20 already provides a labelled mitochondrial reference in this panel.
SUPPRESSED_REPORTER_LABELS = {"chromalive_488_excitation"}

CYAN_CMAP = LinearSegmentedColormap.from_list(
    "ops_reporter_cyan", ["#020609", "#173947", "#4FA7B8", "#DDF1F3"]
)

DISPLAY_NAMES = {
    "CellEvent-Caspase live-cell dye": "Caspase",
    "pHrodo-dextran Live Cell Dye": "pHrodo-dextran",
    "FastAct_SPY555 Live Cell Dye": "FastAct",
    "Peroxi_SPY650 live cell dye": "Peroxi",
    "ChromaLIVE 561 excitation": "ChromaLIVE 561",
    "LysoTracker live-cell dye": "LysoTracker",
    "ChromaLIVE 488 excitation": "ChromaLIVE 488",
    "NucleoLIVE Live Cell dye": "NucleoLIVE",
    "FeRhoNox live-cell dye": "FeRhoNox",
    "CellROX live-cell dye": "CellROX",
    "BODIPY live cell dye": "BODIPY",
}


def configure() -> None:
    configure_sans(mpl, font_manager, FIGURE_ROOT)
    mpl.rcParams.update({
        "font.size": 6.4,
        "axes.titlesize": 7.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 6.3,
        "axes.edgecolor": "#6F7E88",
        "axes.linewidth": 0.55,
        "xtick.labelsize": 6.0,
        "ytick.labelsize": 6.0,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.1,
        "ytick.major.size": 2.1,
        "legend.fontsize": 6.0,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
    })


def bundle_root() -> Path:
    return Path(__file__).resolve().parents[1]


def upstream_root() -> Path:
    """Frozen topology inputs retained inside this public panel bundle."""
    return bundle_root() / "source_data" / "frozen_inputs" / "topology"


def microscopy_root() -> Path:
    """Registered microscopy insets retained inside this public panel bundle."""
    return bundle_root() / "source_data" / "frozen_inputs" / "microscopy"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_and_order() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = upstream_root()
    assays = pd.read_csv(root / "coverage_assays.tsv", sep="\t")
    reporters = pd.read_csv(root / "reporter_order.tsv", sep="\t")
    screens = pd.read_csv(root / "screen_order.tsv", sep="\t")

    assay_key = assays["reporter_slug"].astype(str) + "::" + assays["screen_id"].astype(str)
    assert len(assays) == 99 and assay_key.nunique() == 99
    assert assays["reporter_slug"].nunique() == 52
    assert assays["screen_id"].nunique() == 73
    assert set(assays["reporter_slug"]) == set(reporters["reporter_slug"])
    assert set(assays["screen_id"]) == set(screens["screen_id"])

    # The public bundle may contain the already-frozen ordered tables exported
    # by this builder.  Reuse those coordinates directly instead of merging the
    # same annotation columns a second time (which creates suffixed duplicates).
    frozen_assay_columns = {
        "assay_key",
        "display_name",
        "screen_group",
        "figure_row",
        "figure_column",
        "marker_area_pt2",
        "is_anchor",
    }
    if frozen_assay_columns.issubset(assays.columns):
        reporters = reporters.sort_values("figure_row", kind="stable").reset_index(drop=True)
        screens = screens.sort_values("figure_column", kind="stable").reset_index(drop=True)
        assays = assays.sort_values(["figure_row", "figure_column"], kind="stable").reset_index(drop=True)
        assert screens["screen_group"].value_counts().to_dict() == {
            "Single-reporter": 56,
            "Multi-reporter": 17,
        }
        assert reporters["n_screens"].eq(1).sum() == 18
        assert reporters["n_screens"].gt(1).sum() == 34
        assert int(assays["gene_guide_concordant_exact_links"].sum()) == 9_996_286
        screens["display_column"] = remap_screen_columns(screens["figure_column"])
        assays["display_column"] = remap_screen_columns(assays["figure_column"])
        assays["marker_area_pt2"] = (
            28.0
            * assays["gene_guide_concordant_exact_links"]
            / assays["gene_guide_concordant_exact_links"].max()
        )
        return assays, reporters, screens

    # Frozen, outcome-independent ordering: manuscript biological-system order,
    # repeated reporters first within a system, then stable short name.
    reporters = reporters.copy()
    reporters["system_order"] = reporters["biological_category"].map(
        {name: idx for idx, name in enumerate(BIOLOGY_ORDER)}
    )
    assert reporters["system_order"].notna().all()
    reporters["display_name"] = reporters["display_name"].str.replace(
        "*", "", regex=False
    ).map(lambda name: DISPLAY_NAMES.get(name, name))
    reporters = reporters.sort_values(
        ["system_order", "n_screens", "display_name", "reporter_slug"],
        ascending=[True, False, True, True],
        kind="stable",
    ).reset_index(drop=True)
    reporters["figure_row"] = np.arange(len(reporters), dtype=int)

    # Screens are split by assay occupancy.  Within each block, barycentric
    # ordering exposes the acquisition topology without using prediction results.
    row_map = reporters.set_index("reporter_slug")["figure_row"]
    assay_rows = assays.assign(figure_row=assays["reporter_slug"].map(row_map))
    barycentre = assay_rows.groupby("screen_id", sort=False)["figure_row"].mean()
    screens = screens.copy()
    screens["screen_group"] = np.where(
        screens["n_reporter_targets"].eq(1), "Single-reporter", "Multi-reporter"
    )
    screens["group_order"] = screens["screen_group"].map(
        {"Single-reporter": 0, "Multi-reporter": 1}
    )
    screens["barycentre"] = screens["screen_id"].map(barycentre)
    screens = screens.sort_values(
        ["group_order", "barycentre", "n_reporter_targets", "screen_number"],
        ascending=[True, True, False, True],
        kind="stable",
    ).reset_index(drop=True)
    screens["figure_column"] = np.arange(len(screens), dtype=int)

    assays = assays.merge(
        reporters[
            [
                "reporter_slug",
                "figure_row",
                "display_name",
                "biological_category",
                "n_screens",
            ]
        ],
        on=["reporter_slug", "biological_category"],
        validate="many_to_one",
    ).merge(
        screens[
            [
                "screen_id",
                "figure_column",
                "screen_group",
                "n_reporter_targets",
                "screen_number",
            ]
        ],
        on=["screen_id", "screen_number"],
        validate="many_to_one",
    )
    assays["assay_key"] = assays["reporter_slug"] + "::" + assays["screen_id"]
    assays["is_anchor"] = assays["reporter_slug"].isin(ANCHORS)
    assays["marker_area_pt2"] = (
        28.0
        * assays["gene_guide_concordant_exact_links"]
        / assays["gene_guide_concordant_exact_links"].max()
    )
    screens["display_column"] = remap_screen_columns(screens["figure_column"])
    assays["display_column"] = remap_screen_columns(assays["figure_column"])

    # Frozen numerical checks.
    assert screens["screen_group"].value_counts().to_dict() == {
        "Single-reporter": 56,
        "Multi-reporter": 17,
    }
    assert reporters["n_screens"].eq(1).sum() == 18
    assert reporters["n_screens"].gt(1).sum() == 34
    assert int(assays["gene_guide_concordant_exact_links"].sum()) == 9_996_286
    return assays, reporters, screens


def freeze_reporter_labels(reporters: pd.DataFrame, target_n: int = 26) -> pd.DataFrame:
    """Select readable row labels without using recoverability outcomes.

    Priority is: recurring manuscript cases, reporters repeated in at least
    three screens, one high-coverage representative per biological system,
    then alternating rows in the frozen biological order until ``target_n``.
    Every reporter remains in the matrix regardless of label status.
    """
    out = reporters.copy()
    reasons: dict[str, list[str]] = {slug: [] for slug in out["reporter_slug"]}
    for slug in out.loc[out["reporter_slug"].isin(MANUSCRIPT_REPORTERS), "reporter_slug"]:
        reasons[slug].append("manuscript_case")
    for slug in out.loc[out["n_screens"].ge(3), "reporter_slug"]:
        reasons[slug].append("repeated_ge3_screens")
    for _, block in out.groupby("biological_category", sort=False):
        slug = block.sort_values(
            ["n_screens", "figure_row"], ascending=[False, True], kind="stable"
        ).iloc[0]["reporter_slug"]
        reasons[slug].append("system_representative")

    selected = {
        slug for slug, why in reasons.items()
        if why and slug not in SUPPRESSED_REPORTER_LABELS
    }
    for parity in (0, 1):
        for row in out.loc[out["figure_row"].mod(2).eq(parity)].itertuples(index=False):
            if len(selected) >= target_n:
                break
            if row.reporter_slug in SUPPRESSED_REPORTER_LABELS:
                continue
            selected.add(row.reporter_slug)
            reasons[row.reporter_slug].append("alternating_fill")
        if len(selected) >= target_n:
            break

    out["label_flag"] = out["reporter_slug"].isin(selected)
    out["label_reason"] = out["reporter_slug"].map(
        lambda slug: ";".join(dict.fromkeys(reasons[slug]))
    )
    assert int(out["label_flag"].sum()) == target_n
    assert set(MANUSCRIPT_REPORTERS).intersection(out["reporter_slug"]) <= selected
    return out


def load_microscopy() -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    root = microscopy_root()
    manifest = pd.read_csv(root / "display_manifest.csv")
    manifest = manifest.loc[manifest["key"].isin(["lamp1", "ferhonox"])].copy()
    assert set(manifest["key"]) == {"lamp1", "ferhonox"}
    images: dict[str, dict[str, object]] = {}
    for row in manifest.itertuples(index=False):
        npz_path = root / f"{row.key}_exact_same_cell.npz"
        with np.load(npz_path) as payload:
            phase = np.asarray(payload["phase"], dtype=float)
            fluorescence = np.asarray(payload["fluorescence"], dtype=float)
        assert phase.shape == fluorescence.shape == (int(row.crop_size_pixels),) * 2
        images[row.key] = {
            "phase": phase,
            "fluorescence": fluorescence,
            "row": row,
            "npz_path": npz_path,
        }
    return images, manifest


def relaxed_label_positions(midpoints: list[float], n_rows: int, gap: float = 2.35) -> list[float]:
    """Deterministically separate biological-system labels in row units."""
    out = np.asarray(midpoints, dtype=float).copy()
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    overflow = out[-1] - (n_rows - 0.75)
    if overflow > 0:
        out -= overflow
    for i in range(len(out) - 2, -1, -1):
        out[i] = min(out[i], out[i + 1] - gap)
    underflow = -0.25 - out[0]
    if underflow > 0:
        out += underflow
    return out.tolist()


def export_source_tables(
    assays: pd.DataFrame,
    reporters: pd.DataFrame,
    screens: pd.DataFrame,
    microscopy_manifest: pd.DataFrame,
) -> None:
    root = bundle_root()
    source = root / "source_data"
    source.mkdir(parents=True, exist_ok=True)
    assays[
        [
            "assay_key",
            "reporter_slug",
            "display_name",
            "biological_category",
            "screen_id",
            "screen_number",
            "screen_group",
            "figure_row",
            "figure_column",
            "display_column",
            "gene_guide_concordant_exact_links",
            "targeting_exact_links",
            "control_exact_links",
            "marker_area_pt2",
            "is_anchor",
        ]
    ].sort_values(["figure_row", "figure_column"]).to_csv(
        source / "figure_source_data.tsv", sep="\t", index=False
    )
    reporters[
        [
            "reporter_slug",
            "display_name",
            "biological_category",
            "n_screens",
            "screen_ids",
            "n_exact_assay_cell_links",
            "figure_row",
            "label_flag",
            "label_reason",
        ]
    ].to_csv(source / "reporter_annotations.tsv", sep="\t", index=False)
    reporters[
        [
            "reporter_slug",
            "display_name",
            "biological_category",
            "figure_row",
            "label_flag",
            "label_reason",
        ]
    ].to_csv(source / "reporter_label_manifest.tsv", sep="\t", index=False)
    screens[
        [
            "screen_id",
            "screen_number",
            "screen_group",
            "n_reporter_targets",
            "reporter_targets",
            "figure_column",
            "display_column",
        ]
    ].to_csv(source / "screen_annotations.tsv", sep="\t", index=False)
    microscopy_manifest.assign(
        npz_file=microscopy_manifest["key"].map(
            lambda key: f"source_data/frozen_inputs/microscopy/{key}_exact_same_cell.npz"
        ),
        display_transform="fixed manifest min/max; no content modification",
        role="registered assay-definition inset",
    ).to_csv(source / "microscopy_display_manifest.tsv", sep="\t", index=False)

def square_card_extent(
    ax: plt.Axes, *, x0: float, y0: float, y1: float
) -> tuple[float, float, float, float]:
    """Return a data-coordinate card extent with square image channels on paper."""
    p00 = ax.transData.transform((0.0, 0.0))
    p10 = ax.transData.transform((1.0, 0.0))
    p01 = ax.transData.transform((0.0, 1.0))
    px_per_x = abs(float(p10[0] - p00[0]))
    px_per_y = abs(float(p01[1] - p00[1]))
    header_h = 4.90
    footer_h = 0.30
    gap = 0.48
    image_h = y1 - y0 - header_h - footer_h
    channel_w = image_h * px_per_y / px_per_x
    x1 = x0 + 3 * gap + 2 * channel_w
    return (x0, x1, y0, y1)


def draw_microscopy_card(
    ax: plt.Axes,
    item: dict[str, object],
    *,
    extent: tuple[float, float, float, float],
    title: str,
    target_xy: tuple[float, float],
) -> dict[str, object]:
    """Draw two registered real channels in a verified empty matrix rectangle."""
    x0, x1, y0, y1 = extent
    header_h = 4.90
    footer_h = 0.30
    gap = 0.48
    image_top = y0 + header_h
    image_bottom = y1 - footer_h
    channel_w = (x1 - x0 - 3 * gap) / 2
    phase_x0, phase_x1 = x0 + gap, x0 + gap + channel_w
    fluor_x0, fluor_x1 = phase_x1 + gap, phase_x1 + gap + channel_w
    row = item["row"]

    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=WHITE,
            edgecolor="#B5C0C6",
            linewidth=0.45,
            zorder=5.0,
        )
    )
    ax.imshow(
        item["phase"],
        cmap="gray",
        vmin=float(row.phase_low),
        vmax=float(row.phase_high),
        origin="upper",
        extent=(phase_x0, phase_x1, image_bottom, image_top),
        # Render the deposited level-0 float32 crop without synthetic image
        # interpolation.  The raw asset remains available in microscopy_assets.
        interpolation="none",
        aspect="auto",
        zorder=5.2,
        rasterized=True,
    )
    ax.imshow(
        item["fluorescence"],
        cmap=CYAN_CMAP,
        vmin=float(row.fluorescence_low),
        vmax=float(row.fluorescence_high),
        origin="upper",
        extent=(fluor_x0, fluor_x1, image_bottom, image_top),
        interpolation="none",
        aspect="auto",
        zorder=5.2,
        rasterized=True,
    )
    title_artist = ax.text(
        (x0 + x1) / 2,
        y0 + 1.15,
        title,
        ha="center",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    phase_artist = ax.text(
        (phase_x0 + phase_x1) / 2,
        y0 + 3.50,
        "phase",
        ha="center",
        va="center",
        fontsize=6.0,
        color=PHASE,
        zorder=6,
    )
    reporter_artist = ax.text(
        (fluor_x0 + fluor_x1) / 2,
        y0 + 3.50,
        "reporter",
        ha="center",
        va="center",
        fontsize=6.0,
        color="#3E6F9C",
        zorder=6,
    )

    bar_width = (fluor_x1 - fluor_x0) * 20.0 / (
        float(row.crop_size_pixels) * float(row.pixel_size_um)
    )
    bar_y = image_bottom - 1.02
    # Match Figure 1e exactly: centred white bar, black halo and bold label.
    bar_x0 = (fluor_x0 + fluor_x1 - bar_width) / 2.0
    bar_x1 = bar_x0 + bar_width
    effects = [path_effects.Stroke(linewidth=2.4, foreground="black"), path_effects.Normal()]
    scale_line, = ax.plot(
        [bar_x0, bar_x1], [bar_y, bar_y], color=WHITE,
        lw=1.35, solid_capstyle="butt", zorder=6,
    )
    scale_line.set_path_effects(effects)
    scale_artist = ax.text(
        bar_x1 - bar_width / 2,
        bar_y - 0.42,
        "20 µm",
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color=WHITE,
        zorder=6,
    )
    scale_artist.set_path_effects(effects)

    # The leader registers the illustrative crop to the exact assay glyph.
    source_xy = (x0, min(max(target_xy[1], y0 + 1.4), y1 - 0.8)) if target_xy[0] < x0 \
        else (x1, min(max(target_xy[1], y0 + 1.4), y1 - 0.8))
    ax.annotate(
        "",
        xy=target_xy,
        xytext=source_xy,
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.55,
                        shrinkA=0.8, shrinkB=2.2),
        zorder=6.2,
    )
    return {
        "title": title_artist,
        "phase": phase_artist,
        "reporter": reporter_artist,
        "scale": scale_artist,
        "card_extent": (x0, x1, y0, y1),
        "fluor_extent": (fluor_x0, fluor_x1, image_bottom, image_top),
    }


def assert_text_group_clear(
    fig: plt.Figure, name: str, artists: list[mpl.text.Text]
) -> None:
    """Fail the build if authored text boxes overlap at final canvas size."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [artist.get_window_extent(renderer=renderer) for artist in artists]
    for left in range(len(boxes)):
        for right in range(left + 1, len(boxes)):
            assert not boxes[left].overlaps(boxes[right]), (
                f"Text collision in {name}: "
                f"{artists[left].get_text()!r} {tuple(round(v, 1) for v in boxes[left].bounds)} "
                f"vs {artists[right].get_text()!r} "
                f"{tuple(round(v, 1) for v in boxes[right].bounds)}"
            )


def assert_text_inside_data_rect(
    fig: plt.Figure,
    ax: plt.Axes,
    artist: mpl.text.Text,
    extent: tuple[float, float, float, float],
    name: str,
) -> None:
    """Fail if a text label is clipped outside its authored data rectangle."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_box = artist.get_window_extent(renderer=renderer)
    x0, x1, y0, y1 = extent
    corners = ax.transData.transform([[x0, y0], [x1, y1]])
    rect = Bbox.from_extents(
        float(np.min(corners[:, 0])),
        float(np.min(corners[:, 1])),
        float(np.max(corners[:, 0])),
        float(np.max(corners[:, 1])),
    )
    assert (
        text_box.x0 >= rect.x0
        and text_box.x1 <= rect.x1
        and text_box.y0 >= rect.y0
        and text_box.y1 <= rect.y1
    ), (
        f"Text escapes {name}: text={tuple(round(v, 1) for v in text_box.bounds)} "
        f"rect={tuple(round(v, 1) for v in rect.bounds)}"
    )


def draw() -> None:
    configure()
    assays, reporters, screens = load_and_order()
    reporters = freeze_reporter_labels(reporters)
    images, microscopy_manifest = load_microscopy()
    export_source_tables(assays, reporters, screens, microscopy_manifest)

    fig = plt.figure(
        figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN),
        facecolor=WHITE,
    )
    grid = fig.add_gridspec(
        2,
        5,
        left=0.035,
        right=0.985,
        bottom=0.112,
        top=0.972,
        height_ratios=[0.16, 0.84],
        # Keep both biological-system colour encodings, but shorten the
        # reporter-label corridor so the complete left annotation block sits
        # closer to the acquisition matrix.
        width_ratios=[0.075, 0.009, 0.075, 0.735, 0.106],
        hspace=0.025,
        wspace=0.008,
    )
    ax_system = fig.add_subplot(grid[1, 0])
    ax_strip = fig.add_subplot(grid[1, 1], sharey=ax_system)
    ax_labels = fig.add_subplot(grid[1, 2], sharey=ax_system)
    ax_top = fig.add_subplot(grid[0, 3])
    ax_matrix = fig.add_subplot(grid[1, 3], sharey=ax_system)
    ax_right = fig.add_subplot(grid[1, 4], sharey=ax_system)

    n_rows = len(reporters)
    n_cols = len(screens)
    ylimits = (n_rows - 0.5, -0.5)

    fig.text(0.006, 0.992, "c", ha="left", va="top", fontsize=8.0,
             fontweight="bold", color=INK)

    # Biological-system labels, leader lines and annotation strip.
    group_rows = []
    for system in BIOLOGY_ORDER:
        rows = reporters.loc[
            reporters["biological_category"].eq(system), "figure_row"
        ].to_numpy()
        group_rows.append((system, int(rows.min()), int(rows.max()), float(rows.mean())))
    text_y = relaxed_label_positions([item[3] for item in group_rows], n_rows)
    ax_system.set_xlim(0, 1)
    ax_system.set_ylim(*ylimits)
    ax_system.axis("off")
    for (system, lo, hi, midpoint), label_y in zip(group_rows, text_y):
        colour = BIOLOGY_COLORS[system]
        label = "Plasma memb." if system == "Plasma Membrane" else system
        ax_system.plot([0.88, 0.96], [midpoint, midpoint], color=colour, lw=1.15,
                       solid_capstyle="round", clip_on=False)
        ax_system.plot([0.88, 0.88], [lo - 0.38, hi + 0.38], color=colour, lw=1.15,
                       solid_capstyle="round", clip_on=False)
        ax_system.annotate(
            label,
            xy=(0.86, midpoint),
            xytext=(0.76, label_y),
            ha="right",
            va="center",
            fontsize=6.0,
            color=INK,
            arrowprops=dict(arrowstyle="-", color=GRID, lw=0.45,
                            shrinkA=1, shrinkB=1),
        )

    strip_rgba = np.asarray(
        [mpl.colors.to_rgba(BIOLOGY_COLORS[value]) for value in reporters["biological_category"]]
    )
    ax_strip.imshow(strip_rgba[:, None, :], aspect="auto", interpolation="nearest")
    ax_strip.set_ylim(*ylimits)
    ax_strip.axis("off")

    # Reporter labels: all 52 rows remain in the matrix, while a frozen subset
    # is labelled at publication size and connected back to its exact row.
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(*ylimits)
    ax_labels.axis("off")
    labelled = reporters.loc[reporters["label_flag"]].copy()
    label_y = relaxed_label_positions(labelled["figure_row"].tolist(), n_rows, gap=2.05)
    for row, y_text in zip(labelled.itertuples(index=False), label_y):
        is_anchor = row.reporter_slug in ANCHORS
        ax_labels.plot([0.955, 0.998], [y_text, row.figure_row], color=GRID,
                       lw=0.38, clip_on=False, zorder=1)
        ax_labels.text(
            0.940,
            y_text,
            row.display_name,
            ha="right",
            va="center",
            fontsize=6.0,
            fontweight="bold" if is_anchor else "normal",
            color=INK,
            zorder=2,
        )

    # Main topology field: blanks are structural absence, never numerical zero.
    ax_matrix.set_xlim(DISPLAY_LEFT, DISPLAY_RIGHT)
    ax_matrix.set_ylim(*ylimits)
    ax_matrix.set_facecolor("#FBFCFC")
    # BioFabric-inspired reporter threads: every row is a stable biological
    # entity, while assay membership remains encoded only by filled glyphs.
    for row in reporters.itertuples(index=False):
        lane_colour = mpl.colors.to_rgba(BIOLOGY_COLORS[row.biological_category], 0.17)
        ax_matrix.plot(
            [DISPLAY_LEFT, SINGLE_EDGE], [row.figure_row, row.figure_row],
            color=lane_colour, lw=0.34, zorder=0.12,
        )
        ax_matrix.plot(
            [MULTI_EDGE, DISPLAY_RIGHT], [row.figure_row, row.figure_row],
            color=lane_colour, lw=0.34, zorder=0.12,
        )
    for _, lo, hi, _ in group_rows:
        ax_matrix.axhspan(
            lo - 0.5,
            hi + 0.5,
            color=mpl.colors.to_rgba(BIOLOGY_COLORS[reporters.iloc[lo]["biological_category"]], 0.012),
            zorder=0.1,
        )
        ax_matrix.axhline(hi + 0.5, color=GRID, lw=0.40, zorder=0.2)
    ax_matrix.axvline(MULTI_EDGE, color=INK, lw=0.72, zorder=1)
    gap_mid = (SINGLE_EDGE + MULTI_EDGE) / 2
    for offset in (-0.23, 0.23):
        ax_matrix.plot(
            [gap_mid + offset - 0.13, gap_mid + offset + 0.13],
            [-0.010, 0.010],
            transform=ax_matrix.get_xaxis_transform(),
            color="#6F7E88", lw=0.52, clip_on=False, zorder=6,
        )

    # UpSet-inspired membership connectors. Single-reporter screens are tied
    # directly to their occupancy tile; multi-reporter screens connect only the
    # acquired reporters in that screen. The dots remain the assay marks.
    for screen in screens.itertuples(index=False):
        members = assays.loc[assays["screen_id"].eq(screen.screen_id)]
        x = float(screen.display_column)
        if len(members) == 1:
            y = float(members.iloc[0]["figure_row"])
            ax_matrix.plot(
                [x, x], [-0.43, y],
                color=SCREEN_THREAD, lw=0.27, alpha=0.62, zorder=0.55,
            )
        else:
            ymin = float(members["figure_row"].min())
            ymax = float(members["figure_row"].max())
            ax_matrix.plot(
                [x, x], [ymin, ymax],
                color="#667681", lw=0.48, alpha=0.78, zorder=1.2,
            )

    colours = assays["biological_category"].map(BIOLOGY_COLORS)
    ax_matrix.scatter(
        assays["display_column"],
        assays["figure_row"],
        s=assays["marker_area_pt2"],
        c=colours,
        edgecolors=WHITE,
        linewidths=0.28,
        zorder=3,
    )
    anchor = assays[assays["is_anchor"]]
    ax_matrix.scatter(
        anchor["display_column"],
        anchor["figure_row"],
        s=anchor["marker_area_pt2"] + 9.0,
        facecolors="none",
        edgecolors=INK,
        linewidths=0.65,
        zorder=4,
    )
    ax_matrix.set_yticks([])
    ax_matrix.set_xticks([])
    ax_matrix.spines["top"].set_visible(False)
    ax_matrix.spines["right"].set_visible(False)
    ax_matrix.spines["left"].set_color(GRID)
    ax_matrix.spines["bottom"].set_color("#6F7E88")
    ax_matrix.set_xlabel("")

    # Resolve the physical data-unit aspect before sizing square microscopy.
    fig.canvas.draw()
    lamp_extent = square_card_extent(ax_matrix, x0=23.0, y0=-0.05, y1=16.40)
    fer_extent = square_card_extent(ax_matrix, x0=6.5, y0=35.00, y1=51.45)

    # Registered examples occupy empty structural regions of the single-screen
    # matrix and point to the corresponding LAMP1 and FeRhoNox assay glyphs.
    for x0, x1, y0, y1 in [lamp_extent, fer_extent]:
        hit = assays.loc[
            assays["display_column"].between(x0, x1)
            & assays["figure_row"].between(y0, y1)
        ]
        assert hit.empty, f"Microscopy card would cover assay glyphs: {hit['assay_key'].tolist()}"

    lamp_target = anchor.loc[
        anchor["reporter_slug"].eq("lysosome_lamp1")
    ].sort_values("figure_column").iloc[-1]
    lamp_text = draw_microscopy_card(
        ax_matrix,
        images["lamp1"],
        extent=lamp_extent,
        title="LAMP1 · BORCS7 KO",
        target_xy=(float(lamp_target.display_column), float(lamp_target.figure_row)),
    )
    fer_target = anchor.loc[
        anchor["reporter_slug"].eq("fe2+_ferhonox_live-cell_dye")
    ].sort_values("figure_column").iloc[-1]
    fer_text = draw_microscopy_card(
        ax_matrix,
        images["ferhonox"],
        extent=fer_extent,
        title="FeRhoNox · ALG12 KO",
        target_xy=(float(fer_target.display_column), float(fer_target.figure_row)),
    )

    prb_row = anchor.loc[anchor["reporter_slug"].eq("prb")].iloc[0]
    ax_matrix.annotate(
        "pRb · OPS0077",
        xy=(float(prb_row.display_column), float(prb_row.figure_row)),
        xytext=(50.0, 49.2),
        ha="left", va="center", fontsize=6.0, fontweight="bold", color=INK,
        bbox=dict(boxstyle="square,pad=0.08", fc=WHITE, ec="none", alpha=0.93),
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.48,
                        shrinkA=1.0, shrinkB=1.5),
        zorder=7,
    )

    # Screen occupancy and biological composition track aligned to the matrix.
    # Each unit-height segment is one assayed reporter; the stack height remains
    # the exact reporter count and the colour reuses the frozen system palette.
    for screen in screens.itertuples(index=False):
        members = assays.loc[
            assays["screen_id"].eq(screen.screen_id)
        ].sort_values("figure_row", kind="stable")
        assert len(members) == int(screen.n_reporter_targets), (
            f"Occupancy mismatch for {screen.screen_id}: "
            f"{len(members)} assay reporters vs {screen.n_reporter_targets} declared"
        )
        for bottom, assay in enumerate(members.itertuples(index=False)):
            bar_width = 0.54 if screen.screen_group == "Single-reporter" else 1.46
            ax_top.bar(
                screen.display_column,
                1.0,
                bottom=float(bottom),
                width=bar_width,
                color=BIOLOGY_COLORS[assay.biological_category],
                edgecolor=WHITE,
                linewidth=0.22,
                zorder=2,
            )
    ax_top.set_xlim(DISPLAY_LEFT, DISPLAY_RIGHT)
    ax_top.set_ylim(0, 8.8)
    ax_top.set_yticks([1, 4, 7])
    ax_top.set_ylabel("Reporters\nper screen", labelpad=3.2)
    ax_top.yaxis.set_label_coords(-0.075, 0.56)
    ax_top.set_xticks([])
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.spines["bottom"].set_visible(False)
    ax_top.spines["left"].set_color("#6F7E88")
    single_heading = ax_top.text(
        (DISPLAY_LEFT + SINGLE_EDGE) / 2,
        8.30,
        "56 single-reporter screens · compressed",
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color=INK,
    )
    multi_heading = ax_top.text(
        (MULTI_EDGE + DISPLAY_RIGHT) / 2,
        8.30,
        "17 multi-reporter screens · expanded",
        ha="center",
        va="bottom",
        fontsize=6.0,
        fontweight="bold",
        color=INK,
    )

    # The two occupancy-extreme labels occupy the empty upper part of the
    # multi-reporter matrix. Cross-axes connectors terminate at the tops of
    # the corresponding occupancy bars, never at individual assay glyphs.
    occupancy_callouts = [
        ("Biohub_OPS0043", "OPS0043", (42.2, 2.7), (50.2, 2.7)),
        ("Biohub_OPS0077", "OPS0077", (42.2, 6.0), (50.2, 6.0)),
    ]
    occupancy_annotation_text = []
    for screen_id, label, text_xy, connector_xy in occupancy_callouts:
        screen = screens.loc[screens["screen_id"].eq(screen_id)].iloc[0]
        artist = ax_matrix.text(
            *text_xy,
            label,
            ha="left",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color=INK,
            bbox=dict(boxstyle="square,pad=0.08", fc=WHITE, ec="none", alpha=0.96),
            zorder=8,
        )
        connector = ConnectionPatch(
            xyA=connector_xy,
            coordsA=ax_matrix.transData,
            xyB=(
                float(screen.display_column) - 0.66,
                float(screen.n_reporter_targets) - 0.35,
            ),
            coordsB=ax_top.transData,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.08",
            mutation_scale=3.4,
            color=INK,
            lw=0.36,
            shrinkA=1.2,
            shrinkB=1.5,
            clip_on=False,
            zorder=7.5,
        )
        fig.add_artist(connector)
        occupancy_annotation_text.append(artist)
    # Reporter recurrence track.
    ax_right.set_xlim(0.75, 6.35)
    ax_right.set_ylim(*ylimits)
    for row in reporters.itertuples(index=False):
        ax_right.plot(
            [0.95, row.n_screens], [row.figure_row, row.figure_row],
            color=BIOLOGY_COLORS[row.biological_category],
            lw=0.95, alpha=0.78, solid_capstyle="round", zorder=1,
        )
    ax_right.scatter(
        reporters["n_screens"],
        reporters["figure_row"],
        s=8.8,
        color=reporters["biological_category"].map(BIOLOGY_COLORS),
        edgecolor=WHITE,
        linewidth=0.25,
        zorder=3,
    )
    anchor_reporters = reporters.loc[reporters["reporter_slug"].isin(ANCHORS)]
    ax_right.scatter(
        anchor_reporters["n_screens"],
        anchor_reporters["figure_row"],
        s=15.5,
        facecolors="none",
        edgecolors=INK,
        linewidths=0.55,
        zorder=4,
    )
    ax_right.set_xticks([1, 2, 3, 6])
    ax_right.set_xlabel("")
    ax_right.set_yticks([])
    ax_right.spines["top"].set_visible(False)
    ax_right.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_color("#6F7E88")
    recurrence_heading = ax_right.text(3.55, -1.25, "1 screen: 18\n≥2 screens: 34",
                  ha="center", va="bottom", fontsize=6.0, fontweight="bold",
                  color=INK, clip_on=False)
    fig.canvas.draw()
    recurrence_box = recurrence_heading.get_window_extent(fig.canvas.get_renderer())
    assert recurrence_box.x0 >= ax_right.bbox.x0, "Recurrence key crosses into occupancy bars"
    assert recurrence_box.x1 <= ax_right.bbox.x1, "Recurrence key escapes its track"

    # Shared x labels and an explicit area legend within the fixed canvas.
    fig.text(0.613, 0.061, "Physical screens ordered by assay occupancy",
             ha="center", va="center", fontsize=6.3, color=INK)
    fig.text(0.936, 0.039, "Screens per reporter",
             ha="center", va="center", fontsize=6.3, color=INK)
    legend_counts = [40_000, 100_000, 170_000]
    max_count = assays["gene_guide_concordant_exact_links"].max()
    ax_legend = fig.add_axes([0.315, 0.006, 0.295, 0.032])
    ax_legend.set_xlim(0, 1); ax_legend.set_ylim(0, 1); ax_legend.axis("off")
    ax_legend.text(0.0, 0.50, "Exact-linked observations", ha="left", va="center",
                   fontsize=6.0, color=INK)
    for x, value, label in zip([0.58, 0.75, 0.92], legend_counts,
                               ["40k", "100k", "170k"]):
        ax_legend.scatter([x], [0.51], s=28.0 * value / max_count,
                          facecolor=WHITE, edgecolor=INK, linewidth=0.55,
                          clip_on=False)
        ax_legend.text(x + 0.035, 0.50, label, ha="left", va="center",
                       fontsize=6.0, color=INK)

    assert_text_group_clear(
        fig,
        "LAMP1 microscopy header",
        [lamp_text["title"], lamp_text["phase"], lamp_text["reporter"]],
    )
    assert_text_group_clear(
        fig,
        "FeRhoNox microscopy header",
        [fer_text["title"], fer_text["phase"], fer_text["reporter"]],
    )
    assert_text_group_clear(
        fig,
        "occupancy annotations",
        [single_heading, multi_heading, *occupancy_annotation_text],
    )
    for name, payload in (("LAMP1", lamp_text), ("FeRhoNox", fer_text)):
        assert_text_inside_data_rect(
            fig, ax_matrix, payload["title"], payload["card_extent"],
            f"{name} microscopy card",
        )
        assert_text_inside_data_rect(
            fig, ax_matrix, payload["scale"], payload["fluor_extent"],
            f"{name} fluorescence crop",
        )

    assert_text_layout(fig)
    out = bundle_root() / "figure" / "Figure1-c_assay_topology"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches=None, pad_inches=0,
                facecolor=WHITE)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches=None, pad_inches=0,
                facecolor=WHITE)
    fig.savefig(out.with_suffix(".svg"), bbox_inches=None, pad_inches=0,
                facecolor=WHITE)
    plt.close(fig)

    manifest = {
        "canvas_mm": [WIDTH_MM, HEIGHT_MM],
        "n_assays": int(len(assays)),
        "n_reporters": int(len(reporters)),
        "n_screens": int(len(screens)),
        "single_reporter_screens": int(screens["n_reporter_targets"].eq(1).sum()),
        "multi_reporter_screens": int(screens["n_reporter_targets"].gt(1).sum()),
        "single_screen_reporters": int(reporters["n_screens"].eq(1).sum()),
        "cross_screen_reporters": int(reporters["n_screens"].gt(1).sum()),
        "exact_linked_observations": int(assays["gene_guide_concordant_exact_links"].sum()),
        "ordering": {
            "reporters": "manuscript biological-system order; n_screens descending; display_name",
            "screens": "single/multi occupancy; reporter-row barycentre; occupancy; screen_number",
        },
        "display_mapping": {
            "axis_type": "categorical; display pitch is not a quantitative scale",
            "single_reporter_screen_pitch": SINGLE_STEP,
            "multi_reporter_screen_pitch": MULTI_STEP,
            "single_reporter_screens": "compressed",
            "multi_reporter_screens": "expanded",
            "source_coordinate": "figure_column",
            "visual_coordinate": "display_column",
        },
        "upstream_sha256": {
            name: sha256(upstream_root() / name)
            for name in ("coverage_assays.tsv", "reporter_order.tsv", "screen_order.tsv")
        },
        "microscopy_sha256": {
            name: sha256(microscopy_root() / name)
            for name in (
                "display_manifest.csv",
                "lamp1_exact_same_cell.npz",
                "ferhonox_exact_same_cell.npz",
            )
        },
        "reporter_labels": {
            "displayed": int(reporters["label_flag"].sum()),
            "selection": "manuscript cases; >=3 screens; system representative; alternating frozen rows",
        },
        "font": {
            "policy": "Use local Arial when available; otherwise use a portable sans-serif fallback.",
            "family": str(mpl.rcParams["font.sans-serif"][0]),
        },
    }
    (bundle_root() / "source_data" / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    draw()
