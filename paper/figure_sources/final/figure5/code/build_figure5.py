#!/usr/bin/env python3
"""Assemble Figure 5 as a native-vector Matplotlib composite.

This file intentionally contains layout only.  Scientific preparation and
panel drawing remain in ``build_panel_[a-e].py``.  The builder refuses to
write a composite until all five public ``draw_panel_*`` functions exist, so
an incomplete figure cannot be mistaken for the adopted manuscript asset.

Each panel drawer must accept a Matplotlib Figure/SubFigure-like container and
an ``add_letter`` keyword.  The composite calls every drawer with
``add_letter=False`` and owns the lowercase panel letters itself.
"""

from __future__ import annotations

import argparse
from importlib import import_module
import inspect
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt

from figure5_style import COLORS, apply_style, save_figure


HERE = Path(__file__).resolve().parent
FIGURE5_ROOT = HERE.parent
OUTPUT_STEM = FIGURE5_ROOT / "composite" / "Figure5"

CANVAS_MM = (183.0, 185.0)

# Row 1: a spans the full width.
# Row 2: b is slightly wider because it contains repeatability distributions
#        and a 2 x 3 registered microscopy grid; c occupies the remaining area.
# Row 3: d receives approximately two thirds of the width, as prespecified.
ROW_HEIGHT_RATIOS = (0.28, 0.34, 0.38)
ROW2_WIDTH_RATIOS = (1.16, 0.84)
ROW3_WIDTH_RATIOS = (2.0, 1.0)

PANEL_MODULES = {
    "a": ("build_panel_a", "draw_panel_a"),
    "b": ("build_panel_b", "draw_panel_b"),
    "c": ("build_panel_c", "draw_panel_c"),
    "d": ("build_panel_d", "draw_panel_d"),
    "e": ("build_panel_e", "draw_panel_e"),
}


def _load_drawers() -> tuple[dict[str, Callable[..., Any]], list[str]]:
    """Load panel interfaces without rendering or creating output files."""
    drawers: dict[str, Callable[..., Any]] = {}
    missing: list[str] = []
    for letter, (module_name, function_name) in PANEL_MODULES.items():
        try:
            module = import_module(module_name)
        except ModuleNotFoundError as exc:
            # Only convert absence of the requested panel module into a clean
            # preflight status.  Missing dependencies inside an existing module
            # remain real errors and are re-raised.
            if exc.name == module_name:
                missing.append(f"{letter}: {module_name}.py")
                continue
            raise
        drawer = getattr(module, function_name, None)
        if drawer is None or not callable(drawer):
            missing.append(f"{letter}: {module_name}.{function_name}")
            continue
        signature = inspect.signature(drawer)
        if "add_letter" not in signature.parameters:
            raise TypeError(
                f"{module_name}.{function_name} must expose add_letter= for composite ownership"
            )
        drawers[letter] = drawer
    return drawers, missing


def _outer_letter(subfigure: Any, letter: str) -> None:
    """Place one manuscript-level panel letter on a SubFigure."""
    subfigure.text(
        0.001,
        0.982 if letter == "a" else 0.992,
        letter,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
    )


def build_composite(drawers: dict[str, Callable[..., Any]]):
    """Return the complete Figure 5 without saving it."""
    apply_style()
    # A manual layout is deliberate here.  Panel d contains nested analytical
    # microscopy and constrained_layout can collapse those nested axes during
    # vector export.  The shared subplot parameters reserve enough room for
    # the longest left labels and x-axis labels at the final printed size.
    fig = plt.figure(figsize=(CANVAS_MM[0] / 25.4, CANVAS_MM[1] / 25.4))
    fig._composite_panel_letters = True
    fig.subplots_adjust(left=0.178, right=0.985, top=0.925, bottom=0.155)
    row1, row2, row3 = fig.subfigures(
        3,
        1,
        height_ratios=ROW_HEIGHT_RATIOS,
        hspace=0.075,
    )
    row2_left, row2_right = row2.subfigures(
        1,
        2,
        width_ratios=ROW2_WIDTH_RATIOS,
        wspace=0.055,
    )
    row3_left, row3_right = row3.subfigures(
        1,
        2,
        width_ratios=ROW3_WIDTH_RATIOS,
        wspace=0.055,
    )

    panel_containers = {
        "a": row1,
        "b": row2_left,
        "c": row2_right,
        "d": row3_left,
        "e": row3_right,
    }
    axes_by_panel: dict[str, Any] = {}
    for letter in "abcde":
        _outer_letter(panel_containers[letter], letter)
        kwargs = {"add_letter": False}
        if "compact" in inspect.signature(drawers[letter]).parameters:
            kwargs["compact"] = True
        axes_by_panel[letter] = drawers[letter](panel_containers[letter], **kwargs)
    return fig, axes_by_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report panel-interface readiness without creating a figure.",
    )
    args = parser.parse_args()

    drawers, missing = _load_drawers()
    if args.check:
        for letter in "abcde":
            state = "READY" if letter in drawers else "PENDING"
            print(f"panel {letter}: {state}")
        if missing:
            print("missing interfaces: " + "; ".join(missing))
        return

    if missing:
        raise SystemExit(
            "Figure 5 composite not written: required panel interfaces are pending: "
            + "; ".join(missing)
        )

    fig, _ = build_composite(drawers)
    save_figure(fig, OUTPUT_STEM)
    plt.close(fig)


if __name__ == "__main__":
    main()
