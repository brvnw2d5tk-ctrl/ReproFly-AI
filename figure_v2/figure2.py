"""Measured run endpoints and null landmarks; no inferred attempt histories."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .data_contract import load_rows
from .figure1 import BLUE, LIGHT, MID, ORANGE, STYLE, WINE, _panel


def _score_axis(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=LIGHT, alpha=0.6, lw=0.6, zorder=0)
    ax.set_xlabel("AUTSRC")


def build_figure2():
    """Render the four frozen formal-benchmark comparisons at publication width."""
    runs = load_rows("Fig2a_formal_runs.csv")
    summary = load_rows("Fig2a_summary.csv")[0]
    quantiles = {r["series"]: float(r["AUTSRC"]) for r in load_rows("Fig2b_random_null_quantiles.csv")}
    overlay = load_rows("Fig2b_glm_overlay.csv")
    references = load_rows("Fig2c_graph_baselines.csv")
    glm_reference = load_rows("Fig2c_glm_runs.csv")
    targets = load_rows("Fig2d_target_recovery.csv")
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(183 / 25.4, 7.2))
        a = fig.add_axes((0.13, 0.59, 0.345, 0.31))
        b = fig.add_axes((0.61, 0.59, 0.34, 0.31))
        c = fig.add_axes((0.18, 0.13, 0.295, 0.30))
        e = fig.add_axes((0.61, 0.13, 0.34, 0.30))
        _panel(a, "A", "Formal-run endpoints", subtitle=True)
        ordered = sorted(runs, key=lambda r: (float(r["AUTSRC"]), r["run_id"]))
        for y, row in enumerate(ordered):
            score = float(row["AUTSRC"])
            a.plot([0, score], [y, y], color=WINE, lw=1.5,
                   gid="run-strip-" + row["run_id"], alpha=0.68)
            a.scatter([score], [y], s=25, facecolor=WINE if score else "white",
                      edgecolor=WINE, linewidth=0.8, zorder=3)
        a.set_yticks(range(len(ordered)), [r["run_id"].replace("AI_RUN_", "R") for r in ordered])
        med, lo, hi = [float(summary[k]) for k in ("median_AUTSRC", "iqr_low", "iqr_high")]
        a.plot([lo, hi], [-1.4, -1.4], lw=4, color=WINE, alpha=0.25, solid_capstyle="butt")
        a.scatter([med], [-1.4], marker="D", s=26, color=WINE, zorder=4)
        a.text(0.335, -1.4, f"{med:.4f}", ha="right", va="center", fontsize=7, color=WINE)
        a.text(0.00, -2.7, "Median + IQR", fontsize=7, color=MID)
        a.text(0.0, 1.055, "score strips; not attempt trajectories", transform=a.transAxes, fontsize=7, color=MID)
        a.set_xlim(-0.007, 0.34)
        a.set_ylim(-3, len(ordered) - 0.4)
        a.set_xticks((0, 0.1, 0.2, 0.3))
        _score_axis(a)

        _panel(b, "B", "Random-null landmarks", subtitle=True)
        keys = ("random_null_median", "random_null_p95", "random_null_ci95_high", "random_null_max")
        values = [quantiles[k] for k in keys]
        ys = (3.6, 2.9, 2.2, 1.5)
        b.scatter(values, ys, color=MID, s=30, marker="D", gid="null-landmarks", zorder=4)
        for name, value, y in zip(("median", "p95", "p97.5", "max"), values, ys):
            b.plot([value, value], [0.62, y], color=LIGHT, lw=0.9, ls=(0, (2, 2)))
            b.text(value + (-0.025 if name == "max" else 0.025), y + 0.12,
                   f"{name} {value:.3f}", ha="right" if name == "max" else "left", fontsize=7)
        scores = np.asarray([float(r["AUTSRC"]) for r in overlay])
        # A narrow deterministic stack exposes repeated zeros without implying time.
        repeated = {}
        offsets = []
        for value in scores:
            index = repeated.get(value, 0)
            offsets.append(0.13 + index * 0.105)
            repeated[value] = index + 1
        b.scatter(scores, offsets, s=26, color=WINE, edgecolors="white", linewidths=0.35, zorder=5)
        b.text(0, -0.23, "GLM", color=WINE, fontsize=8)
        above_median = int(np.sum(scores > values[0]))
        above_p95 = int(np.sum(scores > values[1]))
        b.text(0, 1.055, f"{above_median}/{len(scores)} > median; {above_p95}/{len(scores)} > p95",
               transform=b.transAxes, fontsize=7, color=MID)
        b.set_xlim(-0.03, max(values) * 1.06)
        b.set_ylim(-0.45, 4.05)
        b.set_xticks((0, 0.2, 0.4, 0.6, 0.8))
        b.set_yticks([])
        _score_axis(b)

        _panel(c, "C", "Full-information references", subtitle=True)
        c.axhspan(0.55, 3.25, facecolor="#EEF2F5", zorder=0)
        c.axhline(0.5, color=MID, ls=(0, (3, 3)), lw=0.8)
        labels = {"weighted_strength": "Strength", "weighted_pagerank": "PageRank",
                  "baseline_visit_mass": "Visit mass", "GLM": "GLM"}
        positions = {"weighted_strength": 3, "weighted_pagerank": 2, "baseline_visit_mass": 1, "GLM": 0}
        for row in references:
            y, score = positions[row["series"]], float(row["AUTSRC"])
            color = WINE if row["kind"] == "tool_mediated" else BLUE
            if row["kind"] == "tool_mediated":
                c.plot([float(row["iqr_low"]), float(row["iqr_high"])], [y, y], lw=4, color=color, alpha=0.25)
            c.scatter([score], [y], s=38, marker="D", color=color, edgecolor="white", linewidth=0.5, zorder=4)
        c.scatter([float(r["AUTSRC"]) for r in glm_reference], np.full(len(glm_reference), -0.24),
                  s=13, color=WINE, alpha=0.65, zorder=3)
        c.text(0, 1.055, "information-asymmetric / not matched", transform=c.transAxes, fontsize=7, color=MID)
        c.set_yticks([0, 1, 2, 3], ["GLM", "Visit mass", "PageRank", "Strength"])
        c.set_ylim(-0.6, 3.6)
        c.set_xlim(-0.01, 0.32)
        c.set_xticks((0, 0.1, 0.2, 0.3))
        _score_axis(c)

        _panel(e, "D", "Target recovery", subtitle=True)
        colors = {"PCD": BLUE, "LC10a": ORANGE, "CRZ": MID}
        for y, row in zip((2, 1, 0), targets):
            n = int(row["n"])
            hit = int(row["runs_recovered"].split("/")[0])
            x = np.linspace(0.04, 0.79, n)
            color = colors[row["target"]]
            e.scatter(x, np.full(n, y), s=34, facecolors="white", edgecolors=LIGHT, linewidths=1)
            e.scatter(x[:hit], np.full(hit, y), s=34, color=color, zorder=3)
            e.text(0.04, y + 0.30, row["target"], color=color, fontsize=9, fontweight="bold")
            e.text(0.96, y, row["runs_recovered"], ha="right", va="center", fontsize=9, color=color)
        e.text(0.04, -0.59, "Each circle = one run\nFilled = recovered; unfilled = not recovered",
               fontsize=7, color=MID, linespacing=1.5)
        e.text(0, 1.055, "Exact target-specific run fractions", transform=e.transAxes, fontsize=7, color=MID)
        e.set_xlim(0, 1)
        e.set_ylim(-0.8, 2.8)
        e.set_axis_off()
        return fig
