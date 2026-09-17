#!/usr/bin/env python3
"""Build Figure 4d: screen-transfer fidelity.

The scientific contract is frozen:

* the two transfer traces are the two preselected physical screen directions;
* the lower summaries contain all 81 reporter-to-destination-screen directions
  for each of the ten Figure 2 methods;
* method identity, method order and method colours come from the frozen roster;
* no model is averaged and no direction is subsampled.

``draw_panel_d`` is intentionally importable.  It accepts either a Matplotlib
``SubplotSpec`` or a figure-coordinate rectangle ``(left, bottom, width,
height)`` so the same vector artists can be reused in the full Figure 4 build.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from figure4_style import (  # noqa: E402
    COLORS,
    SOURCE_ROOT,
    apply_style,
    clean_axis,
    save_figure,
)


PANEL_ROOT = CODE_ROOT.parent / "d_screen_transfer"


@dataclass(frozen=True)
class PanelDData:
    roster: pd.DataFrame
    directions: pd.DataFrame
    cases: pd.DataFrame
    profiles: pd.DataFrame


def _read_sources(source_root: Path = SOURCE_ROOT) -> PanelDData:
    roster = pd.read_csv(source_root / "method_roster.csv").sort_values("method_order")
    directions = pd.read_csv(source_root / "panel_d" / "all10_strict_direction_metrics.csv")
    cases = pd.read_csv(source_root / "panel_d" / "transfer_case_selection.csv").sort_values("case_order")
    profiles = pd.read_csv(source_root / "panel_d" / "transfer_case_profiles.csv")

    method_ids = roster["method_id"].astype(str).tolist()
    if len(roster) != 10 or roster["method_id"].nunique() != 10:
        raise RuntimeError("Panel d requires the frozen ten-method Figure 2 roster")
    if len(directions) != 810 or directions["method_id"].nunique() != 10:
        raise RuntimeError("Panel d requires 10 methods x 81 strict-screen directions")
    counts = directions.groupby("method_id", observed=True).size().reindex(method_ids)
    if counts.isna().any() or not np.all(counts.to_numpy() == 81):
        raise RuntimeError(f"Expected 81 directions per method, obtained {counts.to_dict()}")

    direction_keys = (
        directions.assign(
            direction_key=lambda x: x["reporter_slug"].astype(str)
            + "::" + x["destination_screen"].astype(str)
        )
        .groupby("method_id", observed=True)["direction_key"]
        .apply(lambda x: tuple(sorted(x)))
        .reindex(method_ids)
    )
    if direction_keys.isna().any() or len(set(direction_keys.tolist())) != 1:
        raise RuntimeError("The 81 reporter-screen directions are not aligned across methods")
    if len(cases) != 2 or cases["case_order"].nunique() != 2:
        raise RuntimeError("Panel d requires exactly two frozen transfer cases")
    if len(profiles) != 2000:
        raise RuntimeError("Panel d requires 1,000 ranked KO profiles for each frozen case")
    for case_order in cases["case_order"]:
        part = profiles.loc[profiles["case_order"].eq(case_order)]
        if len(part) != 1000 or set(part["observed_response_rank"].astype(int)) != set(range(1, 1001)):
            raise RuntimeError(f"Transfer case {case_order} does not contain ranks 1..1000")

    return PanelDData(roster=roster, directions=directions, cases=cases, profiles=profiles)


def _subgrid(fig: plt.Figure, container):
    """Return the outer two-row GridSpec and figure-coordinate bounds."""

    if hasattr(container, "subgridspec") and hasattr(container, "get_position"):
        bounds = container.get_position(fig).bounds
        grid = container.subgridspec(
            2,
            1,
            height_ratios=[0.92, 1.18],
            hspace=0.24,
        )
        return grid, bounds

    if not isinstance(container, Sequence) or len(container) != 4:
        raise TypeError("container must be a SubplotSpec or (left, bottom, width, height)")
    left, bottom, width, height = (float(v) for v in container)
    grid = fig.add_gridspec(
        2,
        1,
        left=left,
        right=left + width,
        bottom=bottom,
        top=bottom + height,
        height_ratios=[0.92, 1.18],
        hspace=0.24,
    )
    return grid, (left, bottom, width, height)


def _draw_transfer_case(
    ax: plt.Axes,
    case: pd.Series,
    profiles: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    show_yticks: bool,
    ymax: float,
) -> list[mpl.lines.Line2D]:
    part = (
        profiles.loc[profiles["case_order"].eq(int(case["case_order"]))]
        .sort_values("observed_response_rank")
    )
    method_colors = dict(zip(roster["method"], roster["method_color"]))
    curves = [
        ("truth_response_norm", "Observed", COLORS["observed"], "-", 1.45, 5),
        ("mlp_prediction_response_norm", "MLP", method_colors["MLP"], "--", 0.90, 4),
        ("catboost_prediction_response_norm", "CatBoost", method_colors["CatBoost"], ":", 1.05, 3),
        ("scpair_prediction_response_norm", "scPair", method_colors["scPair"], "-.", 0.90, 2),
    ]
    x = part["observed_response_rank"].to_numpy(float)
    handles: list[mpl.lines.Line2D] = []
    for column, label, color, linestyle, linewidth, zorder in curves:
        line, = ax.plot(
            x,
            np.log1p(part[column].to_numpy(float)),
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            solid_capstyle="round",
            zorder=zorder,
        )
        handles.append(line)

    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0, ymax)
    ax.set_xticks([1, 10, 100, 1000], labels=["1", "10", "100", "1,000"])
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")
    ax.set_xlabel("Observed KO rank")
    ax.set_ylabel("")
    if not show_yticks:
        ax.tick_params(labelleft=False)
    ax.axhline(0, color=COLORS["ink"], linewidth=0.55, zorder=0)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.42, zorder=0)
    clean_axis(ax)

    title = f"{case['reporter_name']} → {case['screen_label']}"
    ax.text(
        0.0,
        0.955,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.6,
        fontweight="bold",
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.88},
    )
    ax.text(
        1.0,
        0.955,
        f"MLP r = {float(case['mlp_direction_pearson']):.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.5,
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7, "alpha": 0.88},
    )
    return handles


def _fixed_bandwidth_density(values: np.ndarray, grid: np.ndarray, bandwidth: float = 0.040) -> np.ndarray:
    """Gaussian KDE with a fixed absolute bandwidth for comparable row shapes."""

    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z * z).mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    maximum = float(np.nanmax(density))
    return density / maximum if maximum > 0 else density


def _draw_direction_rainclouds(
    ax: plt.Axes,
    directions: pd.DataFrame,
    roster: pd.DataFrame,
    *,
    metric: str,
    xlabel: str,
    show_method_labels: bool,
    seed: int,
) -> None:
    method_ids = roster["method_id"].astype(str).tolist()
    methods = roster["method"].astype(str).tolist()
    colors = roster["method_color"].astype(str).tolist()
    y_positions = np.arange(len(method_ids), dtype=float)
    x_grid = np.linspace(-0.15, 1.0, 360)
    rng = np.random.default_rng(seed)

    ax.axvspan(-0.15, 0, color="#F4F6F7", zorder=-5)
    ax.axvline(0, color="#91A0A9", linewidth=0.55, zorder=-1)
    for separator in (2.5, 5.5, 7.5):
        ax.axhline(separator, color=COLORS["grid"], linewidth=0.45, zorder=-2)

    for y, method_id, color in zip(y_positions, method_ids, colors):
        values = directions.loc[directions["method_id"].eq(method_id), metric].to_numpy(float)
        if len(values) != 81:
            raise RuntimeError(f"{method_id}/{metric}: expected all 81 directions")

        density = _fixed_bandwidth_density(values, x_grid)
        lower = np.full_like(x_grid, y)
        upper = y + 0.27 * density
        ax.fill_between(
            x_grid,
            lower,
            upper,
            facecolor=mpl.colors.to_rgba(color, 0.20),
            edgecolor="none",
            zorder=1,
        )
        ax.plot(x_grid, upper, color=mpl.colors.to_rgba(color, 0.82), linewidth=0.55, zorder=2)

        jitter = rng.uniform(-0.27, -0.08, len(values))
        ax.scatter(
            values,
            y + jitter,
            s=4.1,
            color=mpl.colors.to_rgba(color, 0.28),
            edgecolor="none",
            rasterized=True,
            zorder=3,
        )

        q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        ax.plot([q05, q95], [y, y], color=mpl.colors.to_rgba(color, 0.85), linewidth=0.65, zorder=4)
        ax.plot([q25, q75], [y, y], color=color, linewidth=2.0, solid_capstyle="round", zorder=5)
        ax.scatter(
            [q50],
            [y],
            s=11,
            facecolor="white",
            edgecolor=color,
            linewidth=0.85,
            zorder=6,
        )

    ax.set_xlim(-0.15, 1.0)
    ax.set_ylim(len(method_ids) - 0.48, -0.52)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel(xlabel)
    ax.set_yticks(y_positions)
    if show_method_labels:
        ax.set_yticklabels([])
        for y, method, color in zip(y_positions, methods, colors):
            ax.text(
                -0.060,
                y,
                method,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=5.5,
                color=COLORS["ink"],
                clip_on=False,
            )
            ax.scatter(
                [-0.030],
                [y],
                s=10,
                color=color,
                edgecolor="none",
                transform=ax.get_yaxis_transform(),
                clip_on=False,
                zorder=8,
            )
        ax.tick_params(axis="y", length=0)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.42, zorder=-3)
    clean_axis(ax)


def draw_panel_d(
    fig: plt.Figure,
    container,
    *,
    source_root: Path = SOURCE_ROOT,
    letter: str | None = None,
) -> dict[str, object]:
    """Draw panel d into ``container`` and return its axes and source counts."""

    apply_style()
    data = _read_sources(Path(source_root))
    outer_grid, bounds = _subgrid(fig, container)

    top_grid = outer_grid[0, 0].subgridspec(
        2,
        2,
        height_ratios=[0.20, 0.80],
        width_ratios=[1, 1],
        hspace=0.08,
        wspace=0.20,
    )
    legend_ax = fig.add_subplot(top_grid[0, :])
    legend_ax.axis("off")
    bottom_grid = outer_grid[1, 0].subgridspec(
        2,
        2,
        height_ratios=[0.16, 0.84],
        width_ratios=[1, 1],
        hspace=0.06,
        wspace=0.20,
    )
    bottom_header_ax = fig.add_subplot(bottom_grid[0, :])
    bottom_header_ax.axis("off")

    # Use one y scale for both frozen cases so their amplitude separation is real.
    profile_columns = [
        "truth_response_norm",
        "mlp_prediction_response_norm",
        "catboost_prediction_response_norm",
        "scpair_prediction_response_norm",
    ]
    ymax = 1.18 * float(np.log1p(data.profiles[profile_columns].to_numpy(float)).max())

    case_axes: list[plt.Axes] = []
    legend_handles: list[mpl.lines.Line2D] | None = None
    for col, (_, case) in enumerate(data.cases.iterrows()):
        ax = fig.add_subplot(top_grid[1, col])
        handles = _draw_transfer_case(
            ax,
            case,
            data.profiles,
            data.roster,
            show_yticks=(col == 0),
            ymax=ymax,
        )
        if legend_handles is None:
            legend_handles = handles
        case_axes.append(ax)

    reconstruction_ax = fig.add_subplot(bottom_grid[1, 0])
    ranking_ax = fig.add_subplot(bottom_grid[1, 1])
    _draw_direction_rainclouds(
        reconstruction_ax,
        data.directions,
        data.roster,
        metric="gene_macro_feature_pearson",
        xlabel="KO-response Pearson r",
        show_method_labels=True,
        seed=20260828,
    )
    _draw_direction_rainclouds(
        ranking_ax,
        data.directions,
        data.roster,
        metric="gene_response_magnitude_spearman",
        xlabel="Magnitude Spearman ρ",
        show_method_labels=False,
        seed=20260829,
    )
    bottom_header_ax.text(
        0.0,
        0.06,
        "Response reconstruction",
        transform=bottom_header_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        fontweight="bold",
        color=COLORS["ink"],
    )
    bottom_header_ax.text(
        0.552,
        0.06,
        "Strong-response ranking",
        transform=bottom_header_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.7,
        fontweight="bold",
        color=COLORS["ink"],
    )
    bottom_header_ax.text(
        1.0,
        0.76,
        "81 reporter→screen directions per method",
        transform=bottom_header_ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        color=COLORS["ink"],
    )
    left, bottom, width, height = bounds
    if legend_handles:
        legend = legend_ax.legend(
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="center",
            ncol=4,
            frameon=False,
            borderaxespad=0,
            handlelength=2.2,
            handletextpad=0.45,
            columnspacing=1.15,
            fontsize=5.5,
        )
    else:
        legend = None

    # One shared y label avoids extending into the top legend band.
    first_case_position = case_axes[0].get_position(fig)
    shared_ylabel = fig.text(
        left - 0.031,
        first_case_position.y0 + first_case_position.height / 2,
        "log(1 + response magnitude)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6.2,
        color=COLORS["ink"],
    )

    # Panel letters are owned by the composite builder.  ``letter`` remains an
    # opt-in convenience for callers that explicitly need it.
    letter_artist = None
    if letter:
        letter_artist = fig.text(
            left - 0.030,
            bottom + height + 0.002,
            letter,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=COLORS["ink"],
        )

    return {
        "axes": [legend_ax] + case_axes + [bottom_header_ax, reconstruction_ax, ranking_ax],
        "legend": legend,
        "letter": letter_artist,
        "shared_ylabel": shared_ylabel,
        "n_methods": int(data.roster["method_id"].nunique()),
        "n_directions_per_method": 81,
        "n_transfer_cases": int(len(data.cases)),
    }


def build_standalone() -> dict[str, object]:
    apply_style()
    fig = plt.figure(figsize=(183 / 25.4, 82 / 25.4), facecolor="white")
    result = draw_panel_d(fig, (0.130, 0.100, 0.855, 0.835), letter=None)
    fig.text(0.018, 0.940, "d", ha="left", va="top", fontsize=8.0,
             fontweight="bold", color=COLORS["ink"])
    save_figure(fig, PANEL_ROOT / "figure4d_screen_transfer", dpi=600)
    plt.close(fig)
    return result


def main() -> None:
    result = build_standalone()
    print(
        "Figure 4d complete: "
        f"{result['n_methods']} methods x {result['n_directions_per_method']} directions; "
        f"{result['n_transfer_cases']} frozen transfer cases"
    )


if __name__ == "__main__":
    main()
