"""Portable sans-serif font configuration for the sealed Figure 1 package."""

from __future__ import annotations

from pathlib import Path
import sys

_STYLE_ROOT = next(p for p in Path(__file__).resolve().parents
                   if (p / "publication_style.py").is_file())
sys.path.insert(0, str(_STYLE_ROOT))
from publication_style import format_figure, layout_report, pdf_panel_letters  # noqa: E402


FONT_STACK = ["DejaVu Sans", "Arial", "Liberation Sans"]


def configure_sans(mpl, font_manager, package_root: Path | None = None) -> str:
    """Use the shared DejaVu Sans manuscript font before layout measurement."""
    if package_root is not None:
        local = package_root / "assets" / "local_fonts" / "arial" / "extracted"
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
        selected = "DejaVu Sans"

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [selected, *FONT_STACK],
    })
    return selected
