"""Export the five v2 figures with editable text and publication-size audits.

Run directly or as a module. Set REPROFLY_PROJECT_ROOT for worktree source data.
The text audit compares rendered text against text and the canvas, never against
its background artist: intentionally cell-contained heatmap text is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.text import Text

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from figure_v2.figure1 import build_figure1, STYLE
from figure_v2.figure2 import build_figure2
from figure_v2.figure3 import build_figure3
from figure_v2.figure4 import build_figure4
from figure_v2.figure5 import build_figure5
from figure_v2.data_contract import validate_all_sources


@dataclass(frozen=True)
class Collision:
    """Two colliding text strings, or a text string crossing the canvas edge."""

    first: str
    second: str
    overlap_width: float
    overlap_height: float


def build_all() -> list[Figure]:
    """Return the five figures in manuscript order, owned by the caller."""
    return [builder() for builder in (build_figure1, build_figure2, build_figure3,
                                      build_figure4, build_figure5)]


def all_text_artists(fig: Figure) -> list[Text]:
    """Return rendered text, including ticks, offset labels and legend titles.

    Axes-off decorations and ticks outside view limits are not rendered, so they
    are omitted. Heatmap text remains subject to font and text/text checks.
    """
    fig.canvas.draw()
    candidates = list(fig.texts)
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        candidates.extend(ax.texts)
        candidates.extend([ax.title, ax._left_title, ax._right_title])
        if ax.axison:
            for axis in (ax.xaxis, ax.yaxis):
                if not axis.get_visible():
                    continue
                candidates.extend([axis.label, axis.get_offset_text()])
                low, high = sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                    if tick.get_visible() and low <= tick.get_loc() <= high:
                        candidates.extend([tick.label1, tick.label2])
    for legend in fig.findobj(Legend):
        if legend.get_visible() and (legend.axes is None or legend.axes.get_visible()):
            candidates.extend(legend.get_texts())
            candidates.append(legend.get_title())
    return list(dict.fromkeys(t for t in candidates
                              if t.get_visible() and t.get_alpha() != 0 and t.get_text().strip()))


def collision_report(fig: Figure) -> list[Collision]:
    """Audit text bounding boxes on the rendered canvas (0.5-pixel tolerance).

    Text.get_window_extent deliberately excludes an Annotation's arrow. All
    visible text pairs are checked, including different axes and heatmap text.
    Intentional cell or legend background containment is not an intersection.
    """
    texts = all_text_artists(fig)
    renderer = fig.canvas.get_renderer()
    boxes = [(text, Text.get_window_extent(text, renderer)) for text in texts]
    problems = []
    tolerance = 0.5
    for text, box in boxes:
        if (box.x0 < -tolerance or box.y0 < -tolerance or
                box.x1 > fig.bbox.width + tolerance or box.y1 > fig.bbox.height + tolerance):
            problems.append(Collision(text.get_text(), "<canvas edge>", box.width, box.height))
    for (first, a), (second, b) in combinations(boxes, 2):
        width = min(a.x1, b.x1) - max(a.x0, b.x0)
        height = min(a.y1, b.y1) - max(a.y0, b.y0)
        if width > tolerance and height > tolerance:
            problems.append(Collision(first.get_text(), second.get_text(), width, height))
    return problems


def render_all(output_dir: Path) -> list[Path]:
    """Write SVG, PDF, 600-dpi PNG and 150-dpi 183-mm preview for every figure."""
    validate_all_sources()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = build_all()
    paths = []
    try:
        # rc_context must also surround savefig: PDF/SVG font settings are read
        # by the backend at export, after builder-local rc contexts have ended.
        with mpl.rc_context(STYLE):
            for number, fig in enumerate(figures, 1):
                small = [t.get_text() for t in all_text_artists(fig) if t.get_fontsize() < 7]
                problems = collision_report(fig)
                if small or problems:
                    raise ValueError(f"Figure {number}: small text={small}; collisions={problems}")
                for suffix, dpi in ((".svg", 600), (".pdf", 600), (".png", 600),
                                    ("_183mm_preview.png", 150)):
                    path = output_dir / f"Figure{number}_v2{suffix}"
                    # No bbox_inches='tight': preserve the exact 183-mm page.
                    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches=None)
                    paths.append(path)
    finally:
        for fig in figures:
            plt.close(fig)
    return paths


if __name__ == "__main__":
    for output in render_all(Path(__file__).resolve().parent / "output"):
        print(output)
