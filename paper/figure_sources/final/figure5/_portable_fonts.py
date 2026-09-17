"""Portable typography helper for the released manuscript figure sources.

The submitted artwork uses Arial when it is installed locally.  Arial binaries
are not redistributed with the repository, so a clean checkout must be able to
render with an open system sans-serif fallback while preserving editable
TrueType text in PDF/SVG outputs.
"""

from __future__ import annotations

from pathlib import Path
import sys

_STYLE_ROOT = next(p for p in Path(__file__).resolve().parents
                   if (p / "publication_style.py").is_file())
sys.path.insert(0, str(_STYLE_ROOT))
from publication_style import format_figure, layout_report  # noqa: E402


FONT_STACK = ["DejaVu Sans", "Arial", "Liberation Sans"]


def configure_sans(mpl, font_manager, paper_root: Path | None = None) -> str:
    """Register local Arial when available; otherwise select a safe fallback.

    Returns the first usable font family.  The caller remains responsible for
    its ordinary Matplotlib size and export settings.
    """
    if paper_root is not None:
        local = paper_root / "assets" / "local_fonts" / "arial" / "extracted"
        for filename in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf"):
            path = local / filename
            if path.is_file():
                font_manager.fontManager.addfont(str(path))

    selected = None
    for family in FONT_STACK:
        try:
            font_manager.findfont(family, fallback_to_default=False)
        except Exception:
            continue
        selected = family
        break
    if selected is None:
        # Matplotlib always has a default; this branch keeps rendering robust
        # even in minimal CI images.
        selected = "DejaVu Sans"

    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [selected, *FONT_STACK]})
    return selected
