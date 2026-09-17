"""Fail quantitative-panel export on text collisions or cropped live text.

Intentional overlays (scale bars on microscopy and annotations on data) are
checked visually; this gate checks every visible text artist against every other
text artist and the physical figure canvas. It never alters plotted values.
"""

from __future__ import annotations

from itertools import combinations

from matplotlib.text import Text
from _portable_fonts import format_figure


def assert_text_layout(fig, *, tolerance_pt: float = 0.15) -> None:
    format_figure(fig, 1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tolerance = tolerance_pt * fig.dpi / 72.0
    boxes = []
    for artist in fig.findobj(match=Text):
        if not artist.get_visible() or not artist.get_text().strip():
            continue
        # Annotation.get_window_extent also includes the leader line. Use the
        # base Text implementation to screen the text itself, not its leader.
        box = Text.get_window_extent(artist, renderer=renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        assert (
            box.x0 >= -tolerance and box.y0 >= -tolerance
            and box.x1 <= fig.bbox.width + tolerance
            and box.y1 <= fig.bbox.height + tolerance
        ), f"Text clipped by figure canvas: {artist.get_text()!r}, {box.bounds}"
        boxes.append((artist, box))
    for (left, a), (right, b) in combinations(boxes, 2):
        overlap_x = min(a.x1, b.x1) - max(a.x0, b.x0)
        overlap_y = min(a.y1, b.y1) - max(a.y0, b.y0)
        assert not (overlap_x > tolerance and overlap_y > tolerance), (
            f"Text overlap: {left.get_text()!r} vs {right.get_text()!r}; "
            f"boxes={a.bounds}, {b.bounds}"
        )
    print(
        f"Text-layout QA PASS: {len(boxes)} visible text objects; "
        f"minimum {min(artist.get_fontsize() for artist, _ in boxes):g} pt; "
        "no pairwise text collision or canvas clipping. "
        "Artwork interactions require separate visual review."
    )
