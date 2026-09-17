#!/usr/bin/env python3
"""Build Supplementary Figure 1 from panel-local frozen source tables.

The supplementary audit is deliberately self-contained.  It does not import the
historical Figure 6 analysis tree: all OPS margins and all cross-context gate and
replicate summaries used for drawing live in ``../source_data``.  The builder
only visualizes registered summaries; it never fits a model, changes a threshold,
or reassigns a measurement decision.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.colors as mcolors
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]
PAPER = PACKAGE.parents[1]
OUTPUT = PACKAGE / "published"
SOURCE = PACKAGE / "source_data"
EXTERNAL_SOURCE = SOURCE / "external_context"
sys.path.insert(0, str(PACKAGE.parent))
from publication_style import format_figure, layout_report

OPS_MARGINS = SOURCE / "ops_reporter_threshold_margins.csv"
EXTERNAL_SUMMARY = EXTERNAL_SOURCE / "cross_context_gate_summary.csv"
EXTERNAL_GATES = EXTERNAL_SOURCE / "cross_context_gate_rows.csv"
EXTERNAL_REPLICATES = EXTERNAL_SOURCE / "cross_context_replicate_detail.csv"

WIDTH_MM, HEIGHT_MM = 183.0, 190.0
MM_TO_IN = 1 / 25.4

INK = "#17232D"
MUTED = "#60717E"
GRID = "#D9E2E7"
TEAL = "#3E6F9C"
CARMINE = "#D56C75"
ORANGE = "#DDA06A"
TIER_COLORS = {
    "quantitative_proxy": "#3E6F9C",
    "ranking_proxy": "#DDA06A",
    "measurement_required": "#D56C75",
    "not_identifiable": "#7E8D96",
    "unresolved": "#C8D0D4",
}

METRICS = [
    ("recoverability_r", "Recovery", (0.0, 1.0), 0.70, None),
    ("magnitude_spearman", "Rank", (0.0, 1.0), 0.70, None),
    ("variance_ratio", "Amplitude", (0.0, 1.75), 0.50, 1.50),
    ("top5pct_recall", "Top-5%", (0.0, 1.0), 0.60, None),
    ("reliability", "Reliability", (0.0, 1.0), 0.30, None),
]
METRIC_BY_KEY = {item[0]: item for item in METRICS}
EXTERNAL_ORDER = [
    "oasis:mtt_viability",
    "oasis:ldh_release",
    "periscope_cells:tomm20_mito",
    "periscope_guide:tomm20_mito",
    "oasis_control:mtt_viability",
    "control:tomm20_mito",
    "dropout_dapi:stain_dropout_dapi",
    "dropout_cona:stain_dropout_cona",
    "dropout_phalloidin:stain_dropout_phalloidin",
    "dropout_wga:stain_dropout_wga",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def configure() -> str:
    """Use local Arial when supplied, otherwise a portable sans-serif fallback."""
    local_font_dir = PAPER / "assets" / "local_fonts" / "arial" / "extracted"
    for filename in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
        candidate = local_font_dir / filename
        if candidate.is_file():
            font_manager.fontManager.addfont(str(candidate))
    selected = "DejaVu Sans"
    for family in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except Exception:
            continue
        selected = family
        break
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [selected, "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 6.2,
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return selected


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read and validate the four frozen, panel-local tables."""
    required = (OPS_MARGINS, EXTERNAL_SUMMARY, EXTERNAL_GATES, EXTERNAL_REPLICATES)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing panel-local source table(s): " + ", ".join(missing))

    ops = pd.read_csv(OPS_MARGINS)
    summary = pd.read_csv(EXTERNAL_SUMMARY)
    gates = pd.read_csv(EXTERNAL_GATES)
    replicates = pd.read_csv(EXTERNAL_REPLICATES)
    if len(ops) != 260 or ops.reporter_slug.nunique() != 52 or ops.metric.nunique() != 5:
        raise RuntimeError("OPS threshold-margin table must contain 52 reporters by five gates")
    if set(ops.metric) != {item[1] for item in METRICS}:
        raise RuntimeError("OPS threshold-margin metrics do not match the frozen gate contract")
    if len(summary) != 25 or set(summary.row_key) != {"ops", *EXTERNAL_ORDER[:4]}:
        raise RuntimeError("cross-context summary must contain OPS plus four admitted readouts")
    if gates.row_key.nunique() != len(EXTERNAL_ORDER) or gates.groupby("row_key").quantity.nunique().ne(5).any():
        raise RuntimeError("cross-context gate rows must contain five gates for every registered row")
    if not set(EXTERNAL_ORDER).issubset(set(replicates.row_key)):
        raise RuntimeError("cross-context replicate table is incomplete")
    return ops, summary, gates, replicates


def _scale(value: float, bounds: tuple[float, float], x0: float, x1: float) -> float:
    lower, upper = bounds
    return x0 + (np.clip(value, lower, upper) - lower) / (upper - lower) * (x1 - x0)


def draw_ops_audit(fig: plt.Figure, rect: tuple[float, float, float, float], ops: pd.DataFrame) -> None:
    """Draw the all-reporter, five-gate threshold-margin matrix."""
    x0, y0, width, height = rect
    matrix = (
        ops.pivot(index="metric_index", columns="reporter_index", values="scaled_margin")
        .reindex(index=range(5), columns=range(52))
    )
    ax = fig.add_axes((x0 + 0.070, y0 + 0.105 * height, width - 0.085, 0.635 * height))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "decision_margin", [CARMINE, "#F4F7F8", TEAL]
    )
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap=cmap, vmin=-1, vmax=1,
                      interpolation="nearest", rasterized=True)
    ax.set_yticks(range(5), [item[1] for item in METRICS])
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    tier_by_reporter = (
        ops.loc[ops.metric_index.eq(0), ["reporter_index", "tier"]]
        .drop_duplicates()
        .sort_values("reporter_index")
    )
    tier_strip = fig.add_axes((x0 + 0.070, y0 + 0.760 * height, width - 0.085, 0.043 * height))
    colours = np.asarray([
        mcolors.to_rgb(TIER_COLORS[tier]) for tier in tier_by_reporter.tier
    ])[None, :, :]
    tier_strip.imshow(colours, aspect="auto", interpolation="nearest")
    tier_strip.set_axis_off()

    fig.text(x0, y0 + 0.965 * height, "a", fontsize=9.2, fontweight="bold",
             ha="left", va="top", color=INK)
    fig.text(x0 + 0.070, y0 + 0.825 * height, "52 reporters ordered by final tier",
             fontsize=5.2, ha="left", va="bottom", color=MUTED)

    cax = fig.add_axes((x0 + 0.62 * width, y0 + 0.018 * height, 0.28 * width, 0.032 * height))
    colourbar = fig.colorbar(image, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
    colourbar.ax.set_xticklabels(["below", "threshold", "above"])
    colourbar.ax.tick_params(labelsize=4.5, length=1.8, pad=1)
    colourbar.outline.set_visible(False)
    fig.text(x0 + 0.070, y0 + 0.040 * height,
             "Scaled decision margin", fontsize=6, color=MUTED, ha="left", va="center")


def _draw_gate_rail(
    ax: plt.Axes,
    *,
    x0: float,
    x1: float,
    y: float,
    metric: str,
    value: float,
    passed: bool,
    replicate_values: np.ndarray,
) -> None:
    """Draw one threshold-aware registered estimate in axes coordinates."""
    _, _, bounds, threshold_low, threshold_high = METRIC_BY_KEY[metric]
    ax.plot([x0, x1], [y, y], transform=ax.transAxes, color=GRID, lw=0.7, zorder=1)
    threshold_x0 = _scale(threshold_low, bounds, x0, x1)
    if threshold_high is None:
        ax.add_patch(Rectangle((threshold_x0, y - 0.012), x1 - threshold_x0, 0.024,
                               transform=ax.transAxes, facecolor="#E0F0EF", edgecolor="none", zorder=0))
        ax.plot([threshold_x0, threshold_x0], [y - 0.018, y + 0.018], transform=ax.transAxes,
                color=INK, lw=0.55, zorder=2)
    else:
        threshold_x1 = _scale(threshold_high, bounds, x0, x1)
        ax.add_patch(Rectangle((threshold_x0, y - 0.012), threshold_x1 - threshold_x0, 0.024,
                               transform=ax.transAxes, facecolor="#E0F0EF", edgecolor="none", zorder=0))
        for threshold_x in (threshold_x0, threshold_x1):
            ax.plot([threshold_x, threshold_x], [y - 0.018, y + 0.018], transform=ax.transAxes,
                    color=INK, lw=0.55, zorder=2)

    for replicate in replicate_values[np.isfinite(replicate_values)]:
        x = _scale(float(replicate), bounds, x0, x1)
        ax.plot([x, x], [y - 0.013, y + 0.013], transform=ax.transAxes,
                color="#9AABB5", lw=0.45, alpha=0.85, zorder=3)
    x = _scale(value, bounds, x0, x1)
    colour = TEAL if passed else CARMINE
    ax.scatter([x], [y], transform=ax.transAxes, s=13, facecolor=colour,
               edgecolor="white", linewidth=0.45, zorder=4)


def draw_external_audit(
    fig: plt.Figure,
    rect: tuple[float, float, float, float],
    gates: pd.DataFrame,
    replicates: pd.DataFrame,
) -> None:
    """Draw all admitted, control and sensitivity readouts without hidden inputs."""
    x0, y0, width, height = rect
    ax = fig.add_axes((x0, y0, width, height))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.000, 0.990, "b", fontsize=9.2, fontweight="bold", ha="left", va="top", color=INK)

    x_label = 0.012
    x_start, x_end = 0.280, 0.775
    rail_width = 0.078
    centres = np.linspace(x_start + rail_width / 2, x_end - rail_width / 2, 5)
    for centre, (_, label, _, threshold_low, threshold_high) in zip(centres, METRICS):
        ax.text(centre, 0.888, label, transform=ax.transAxes, fontsize=5.1,
                fontweight="bold", ha="center", va="bottom", color=INK)
        criterion = f"{threshold_low:.2g}–{threshold_high:.2g}" if threshold_high is not None else f"≥{threshold_low:.2g}"
        ax.text(centre, 0.860, criterion, transform=ax.transAxes, fontsize=4.0,
                ha="center", va="top", color=MUTED)
    ax.text(0.808, 0.888, "Evidence role", transform=ax.transAxes, fontsize=6,
            fontweight="bold", ha="left", va="bottom", color=INK)

    rows = []
    compact_context = {
        "oasis:mtt_viability": "OASIS · brightfield · compound",
        "oasis:ldh_release": "OASIS · brightfield · compound",
        "periscope_cells:tomm20_mito": "PERISCOPE · four dyes · cell",
        "periscope_guide:tomm20_mito": "PERISCOPE · four dyes · guide",
        "oasis_control:mtt_viability": "OASIS · reliability unchanged",
        "control:tomm20_mito": "PERISCOPE · reliability stop",
        "dropout_dapi:stain_dropout_dapi": "From ConA / phalloidin / WGA",
        "dropout_cona:stain_dropout_cona": "From DAPI / phalloidin / WGA",
        "dropout_phalloidin:stain_dropout_phalloidin": "From DAPI / ConA / WGA",
        "dropout_wga:stain_dropout_wga": "From DAPI / ConA / phalloidin",
    }
    for row_key in EXTERNAL_ORDER:
        part = gates.loc[gates.row_key.eq(row_key)]
        label = str(part.label.iloc[0])
        sublabel = compact_context[row_key]
        band = str(part.band.iloc[0])
        rows.append((row_key, label, sublabel, band))
    # Keep two explicit section gaps so group headers do not collide with the
    # preceding row's descriptive sublabel at publication size.
    y_positions = np.array([0.758, 0.697, 0.636, 0.575, 0.470, 0.409, 0.304, 0.243, 0.182, 0.121])
    group_starts = {"admitted": 0, "control": 4, "second_question": 6}
    group_labels = {
        "admitted": "Capability-matched applications",
        "control": "Input-permutation controls",
        "second_question": "Four-dye substitution sensitivity",
    }
    for band, index in group_starts.items():
        y = y_positions[index] + 0.046
        ax.text(x_label, y, group_labels[band], transform=ax.transAxes, color=MUTED,
                fontsize=4.8, fontweight="bold", va="bottom")
        if index:
            ax.plot([0.01, 0.99], [y + 0.028, y + 0.028], transform=ax.transAxes,
                    color=GRID, lw=0.55)

    for (row_key, label, sublabel, band), y in zip(rows, y_positions):
        ax.text(x_label, y + 0.012, label, transform=ax.transAxes, fontsize=5.15,
                fontweight="bold", ha="left", va="center", color=INK)
        ax.text(x_label, y - 0.016, sublabel, transform=ax.transAxes, fontsize=3.65,
                ha="left", va="center", color=MUTED)
        part = gates.loc[gates.row_key.eq(row_key)].set_index("quantity")
        for centre, (metric, _, _, _, _) in zip(centres, METRICS):
            entry = part.loc[metric]
            rail_x0, rail_x1 = centre - rail_width / 2, centre + rail_width / 2
            rep = replicates.loc[
                replicates.row_key.eq(row_key) & replicates.quantity.eq(metric), "value"
            ].to_numpy(float)
            _draw_gate_rail(
                ax, x0=rail_x0, x1=rail_x1, y=y, metric=metric,
                value=float(entry.value), passed=bool(entry.passes), replicate_values=rep,
            )
        verdict = {
            "admitted": "admitted",
            "control": "control",
            "second_question": "sensitivity",
        }[band]
        colour = TEAL if band == "admitted" else ORANGE if band == "second_question" else MUTED
        ax.text(0.808, y, verdict, transform=ax.transAxes, fontsize=4.5,
                fontweight="bold", ha="left", va="center", color=colour)

    ax.text(0.280, 0.030,
            "Ticks: replicate estimates; point: summary; rails: gates.",
            transform=ax.transAxes, fontsize=6, ha="left", va="center", color=MUTED)


def build(
    ops: pd.DataFrame,
    gates: pd.DataFrame,
    replicates: pd.DataFrame,
) -> plt.Figure:
    configure()
    fig = plt.figure(figsize=(WIDTH_MM * MM_TO_IN, HEIGHT_MM * MM_TO_IN), facecolor="white")
    fig._composite_panel_letters = True
    draw_ops_audit(fig, (0.018, 0.655, 0.965, 0.325), ops)
    draw_external_audit(fig, (0.018, 0.030, 0.965, 0.600), gates, replicates)
    format_figure(fig, 6)
    return fig


def main() -> int:
    ops, summary, gates, replicates = load_tables()
    fig = build(ops, gates, replicates)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    written = []
    for extension, dpi in (("pdf", None), ("svg", None), ("png", 600)):
        target = OUTPUT / f"SupplementaryFigure1.{extension}"
        kwargs = {"bbox_inches": None, "pad_inches": 0}
        if dpi is not None:
            kwargs["dpi"] = dpi
        fig.savefig(target, **kwargs)
        written.append(target)
    layout_report(fig, OUTPUT / "typography_qa.json")
    plt.close(fig)
    _atomic_copy(OUTPUT / "SupplementaryFigure1.pdf", PAPER / "figures" / "SupplementaryFigure1.pdf")

    hashes = {path.suffix.lstrip("."): _sha256(path) for path in written}
    contract = {
        "figure": "Supplementary Figure 1",
        "claim": "The common decision framework is fully auditable across OPS, OASIS and PERISCOPE.",
        "panel_a": "all 52 OPS reporters by five frozen evidence gates",
        "ops_estimator": "coherent checkpoint-average median ensemble, before equal-screen gene averaging",
        "ops_recovery": "arithmetic fivefold mean of ensemble endpoint-macro Pearson",
        "derived_ops_source": "../final/figure6/code/rebuild_coherent_statistics.py",
        "physical_size_mm": [WIDTH_MM, HEIGHT_MM],
        "panel_b": "all admitted external rows, permutation controls and four-dye substitution sensitivity rows",
        "no_model_fitting": True,
        "source_tables": [
            "source_data/ops_reporter_threshold_margins.csv",
            "source_data/external_context/cross_context_gate_summary.csv",
            "source_data/external_context/cross_context_gate_rows.csv",
            "source_data/external_context/cross_context_replicate_detail.csv",
        ],
        "supplementary_table_sources": [
            "source_data/external_context/admission_funnel.csv",
            "source_data/external_context/admission_evidence_tiers.csv",
            "source_data/external_context/batch_verdicts.csv",
        ],
        "sha256": hashes,
    }
    (PACKAGE / "source_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    with (PACKAGE / "artifact_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_file() and path.name != "artifact_manifest.csv" and "__pycache__" not in path.parts:
                writer.writerow([path.relative_to(PACKAGE).as_posix(), path.stat().st_size, _sha256(path)])
    print(json.dumps({
        "status": "PASS",
        "outputs": [str(path) for path in written],
        "ops_rows": len(ops),
        "external_summary_rows": len(summary),
        "external_gate_rows": len(gates),
        "external_replicate_rows": len(replicates),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
