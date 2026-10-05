"""Benchmark world model: explicitly schematic geometry, frozen design counts."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Wedge
import numpy as np

from .data_contract import load_design_constants
from .visual_primitives import candidate_landscape, draw_search_path


WINE, BLUE, ORANGE, GREEN = "#9E1B3D", "#3D78A8", "#D47A16", "#4F7262"
INK, MID, LIGHT = "#202428", "#68717A", "#D8DEE3"
STYLE = {
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "text.color": INK,
    "axes.labelcolor": INK, "axes.edgecolor": LIGHT,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white",
}


def _panel(ax, letter, title, subtitle=False):
    ax.set_title(title, loc="center", pad=25 if subtitle else 14)
    ax.text(-0.065, 1.12 if subtitle else 1.07, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom")


def build_figure1():
    """Compose four schematic panels; all printed counts load at render time."""
    d = load_design_constants()
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(183 / 25.4, 7.5))
        a = fig.add_axes((0.08, 0.61, 0.88, 0.30))
        b = fig.add_axes((0.08, 0.28, 0.40, 0.235))
        c = fig.add_axes((0.57, 0.28, 0.39, 0.235))
        e = fig.add_axes((0.08, 0.022, 0.88, 0.18))
        for ax in (a, b, c, e):
            ax.set_axis_off()
        _panel(a, "A", "A locally revealed search universe")
        candidate_landscape(a, seed=42, regions=7)
        for artist in a.lines + list(a.collections):
            artist.set_alpha(0.30)
        a.set_xlim(0, 1.7)
        a.set_ylim(-0.15, 1.16)
        a.add_patch(Ellipse((0.54, 0.46), 0.77, 0.83, facecolor="#F4E8ED",
                            edgecolor=WINE, linewidth=0.7, alpha=0.5, zorder=0))
        path = np.array([[0.27, 0.35], [0.40, 0.57], [0.52, 0.40], [0.67, 0.57], [0.78, 0.41]])
        draw_search_path(a, path, WINE)
        a.scatter(path[:, 0], path[:, 1], s=27, color=WINE, edgecolor="white", zorder=6)
        for x, y, label in ((0.23, 0.25, "inspect"), (0.39, 0.69, "predict"),
                            (0.58, 0.27, "perturb"), (0.78, 0.66, "observe")):
            a.text(x, y, label, ha="center", fontsize=8, color=WINE)
        a.text(0.03, 1.03, f'{d["candidate_universe_n"]:,} candidates', fontsize=10, fontweight="bold")
        a.text(0.03, 0.90, "Local records revealed during search", fontsize=8, color=MID)
        a.text(0.03, -0.07, "Illustrative search geometry; schematic", fontsize=7, color=MID)
        a.annotate("", (1.18, 0.47), (0.95, 0.47), arrowprops=dict(arrowstyle="->", color=WINE, lw=1.3))
        a.axvline(1.17, ymin=0.23, ymax=0.84, color=MID, ls=(0, (3, 3)), lw=0.8)
        a.text(1.42, 0.94, "Sealed evaluation", ha="center", fontsize=9, fontweight="bold")
        targets = [("PCD", "gt_pcd_m", BLUE), ("LC10a", "gt_lc10a_m", ORANGE), ("CRZ", "gt_crz_m", MID)]
        for y, (name, key, color) in zip((0.72, 0.58, 0.44), targets):
            a.scatter([1.28], [y], s=35, color=color, zorder=5)
            a.text(1.36, y, f'{name} · {d[key]}', va="center", fontsize=8)
        a.text(1.43, 0.24, "AUTSRC", ha="center", color=WINE, fontweight="bold", fontsize=11)
        a.text(1.43, 0.09, f'{d["random_null_permutations"]:,} permutations', ha="center", fontsize=8)
        a.text(0.85, 1.11,
               f'{d["glm_runs"]} runs × {d["charged_attempts_per_run"]} charged attempts',
               ha="center", fontsize=8, color=MID)

        _panel(b, "B", "Information boundary")
        candidate_landscape(b, seed=42, regions=7)
        for artist in b.lines + list(b.collections):
            artist.set_alpha(0.22)
        b.add_patch(Ellipse((0.30, 0.51), 0.55, 0.78, facecolor="#EDF3F7",
                            edgecolor=BLUE, lw=0.8, zorder=0))
        b.text(0.27, 0.70, "Anonymous\nlocal records", ha="center", va="center", fontsize=8, color=BLUE)
        b.text(0.27, 0.31, "delivered", ha="center", fontsize=8, color=BLUE)
        b.plot((0.62, 0.62), (0.12, 0.88), color=MID, ls=(0, (3, 3)), lw=0.8)
        b.text(0.81, 0.66, "Names\nTarget mapping\nFull graph", ha="center", va="center",
               color=MID, linespacing=1.7, fontsize=8)
        b.text(0.81, 0.27, "withheld", ha="center", color=MID, fontsize=8)
        b.text(0.02, -0.04, "Schematic visibility surface", color=MID, fontsize=7)

        _panel(c, "C", "Structural endpoint oracle")
        c.set_xlim(0, 1)
        c.set_ylim(0, 1)
        c.scatter([0.10], [0.54], s=90, facecolor="white", edgecolor=GREEN, linewidth=1.7)
        c.plot((0.065, 0.135), (0.58, 0.50), color=GREEN, lw=1.5)
        for y in (0.26, 0.40, 0.56, 0.72):
            c.plot([0.15, 0.39, 0.62, 0.78], [0.54, y, 0.50 + (y - 0.5) * 0.5, 0.54],
                   color=GREEN, alpha=0.4, lw=1.3)
            c.scatter([0.39, 0.62], [y, 0.50 + (y - 0.5) * 0.5], color=GREEN, alpha=0.45, s=18)
        c.plot((0.78, 0.94), (0.54, 0.54), color=GREEN, lw=2)
        c.scatter([0.94], [0.54], color=GREEN, s=38)
        c.text(0.10, 0.85, "Virtual\nsilencing", ha="center", fontsize=8)
        c.text(0.54, 0.85, "Deterministic\npropagation", ha="center", fontsize=8)
        c.text(0.70, 0.16, "Structural endpoint proxy", ha="center", fontsize=8, color=GREEN)
        c.text(0.50, -0.04, "Schematic · not physiology", ha="center", fontsize=8, fontweight="bold")

        _panel(e, "D", "Study phases")
        e.set(xlim=(-4.4, 4.4), ylim=(-1.08, 1.08), aspect="equal")
        # Equal angles and radii encode reading order only, not phase duration,
        # sample size or effect magnitude. Saturation emphasizes the formal phase.
        phases = (
            (36, 144, WINE, "Phase 2A", "Formal benchmark",
             f'{d["glm_runs"]} × {d["charged_attempts_per_run"]}', -2.35, 0.71),
            (-84, 24, BLUE, "Phase 2B", "Post-hoc behaviour", "M1–M13", 2.35, 0.32),
            (156, 264, ORANGE, "Phase 2C.1", "Matched controls",
             f'{d["matched_arms"]} × {d["matched_runs_per_arm"]}', -2.35, -0.28),
        )
        for start, stop, color, phase, name, count, x, y in phases:
            e.add_patch(Wedge((0, 0), 1.04, start, stop, width=0.10,
                              facecolor=color, edgecolor="none", alpha=1 if color == WINE else 0.75))
            e.text(x, y, phase, color=color, ha="center", fontsize=7, fontweight="bold")
            e.text(x, y - 0.28, name, ha="center", fontsize=7)
            e.text(x, y - 0.56, count, color=color, ha="center", fontsize=8, fontweight="bold")
            # Clockwise arrows point towards each arc's lower angle endpoint.
            angles = np.deg2rad([start + 14, start + 2])
            xy = 0.99 * np.column_stack((np.cos(angles), np.sin(angles)))
            e.annotate("", xy[1], xy[0], arrowprops=dict(arrowstyle="-|>", color="white",
                        lw=0.7, mutation_scale=7, connectionstyle="arc3,rad=-0.1"))
        for angle in (150, 30, -90):
            theta = np.deg2rad(angle)
            e.scatter([0.99 * np.cos(theta)], [0.99 * np.sin(theta)], s=10,
                      color=MID, edgecolors="white", linewidths=0.4, zorder=4)
        e.text(0, 0, "FORMAL → POST-HOC", ha="center", va="center", fontsize=7, fontweight="bold")
        return fig
