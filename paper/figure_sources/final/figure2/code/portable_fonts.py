"""Self-contained sans-serif font selection for the Figure 2 release."""

from __future__ import annotations

from pathlib import Path
import sys

_STYLE_ROOT = next(p for p in Path(__file__).resolve().parents
                   if (p / "publication_style.py").is_file())
sys.path.insert(0, str(_STYLE_ROOT))
from publication_style import format_figure, layout_report, pdf_panel_letters  # noqa: E402


FONT_STACK = ["DejaVu Sans", "Arial", "Liberation Sans"]


def configure_sans(mpl, font_manager, package_root: Path | None = None) -> str:
    """Use Arial when available, with open sans-serif fallbacks for clean systems."""
    if package_root is not None:
        local = package_root / "assets" / "fonts"
        if local.is_dir():
            for path in sorted(local.glob("*.ttf")):
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
