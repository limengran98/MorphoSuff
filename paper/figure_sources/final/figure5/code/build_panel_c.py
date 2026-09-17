#!/usr/bin/env python3
"""Composite adapter for the canonical Figure 5c renderer.

The standalone panel keeps its plotting implementation beside its own source
bundle (``c_amplitude_spectrum/code/build_panel_c.py``).  This small adapter
exposes the same ``draw_panel_<letter>(container, add_letter=...)`` interface
as panels a, b, d and e, without copying or changing any scientific logic.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE.parent / "c_amplitude_spectrum" / "code" / "build_panel_c.py"


def _load_implementation():
    spec = spec_from_file_location("figure5_panel_c_implementation", IMPLEMENTATION)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Figure 5c implementation from {IMPLEMENTATION}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_IMPL = _load_implementation()


def draw_panel_c(container: Any, data=None, *, add_letter: bool = True):
    """Draw panel c in a Figure/SubFigure using its canonical renderer."""
    if data is None:
        data = _IMPL.load_source()
    spec = container.add_gridspec(1, 1)[0, 0]
    axes = _IMPL.draw_panel_c(container, spec, data)
    if add_letter:
        container.text(
            0.002,
            0.985,
            "c",
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
        )
    return axes


__all__ = ["draw_panel_c"]
