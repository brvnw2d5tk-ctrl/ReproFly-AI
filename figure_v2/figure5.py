"""Frozen target coverage, rank landscape, policy recovery and record profiles.

Member slots expand marginal counts independently. Slot alignment does not infer
member identity, within-run histories, transitions or causal selection paths.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np

from .data_contract import load_rows, numeric_matrix
from .figure1 import BLUE, ORANGE, MID, LIGHT, INK, STYLE, _panel
from .visual_primitives import draw_termination


TARGETS = ("PCD", "LC10a", "CRZ")
STAGES = ("visible_top10", "inspected", "valid_perturbed")
ARMS = ("GLM", "C1", "C2", "C3", "D1", "D2a", "D2b", "D3")
COLORS = {"PCD": BLUE, "LC10a": ORANGE, "CRZ": MID}
BASELINES = ("weighted_strength", "weighted_pagerank", "baseline_visit_mass")


def _coverage():
    records = load_rows("Fig5a_accessibility.csv")
    by_key = {}
    sizes = {}
    for row in records:
        target, stage = row["target"], row["stage"]
        if target not in TARGETS or stage not in STAGES or (target, stage) in by_key:
            raise ValueError("unexpected or duplicate member-state key")
        size, count = float(row["set_size"]), float(row["count"])
        if not np.isfinite([size, count]).all() or not size.is_integer() or not count.is_integer():
            raise ValueError("set sizes and counts must be finite integers")
        if size < 1 or not 0 <= count <= size or sizes.get(target, size) != size:
            raise ValueError("invalid member-state size or count")
        sizes[target] = int(size)
        by_key[target, stage] = int(count)
    if set(by_key) != {(target, stage) for target in TARGETS for stage in STAGES}:
        raise ValueError("incomplete member-state table")
    if [sizes[target] for target in TARGETS] != [13, 1, 1]:
        raise ValueError("unexpected frozen target set sizes")
    return sizes, by_key


def member_state_array() -> np.ndarray:
    """Return 15 × 3 display-only slots, independently filled for each state."""
    sizes, counts = _coverage()
    blocks = []
    for target in TARGETS:
        block = np.zeros((sizes[target], len(STAGES)), dtype=int)
        for column, stage in enumerate(STAGES):
            block[:counts[target, stage], column] = 1
        blocks.append(block)
    return np.concatenate(blocks)


def build_figure5():
    """Draw four source-backed panels at the final 183-mm publication width."""
    sizes, counts = _coverage()
    states = member_state_array()
    ranks = load_rows("Fig5b_ranks.csv")
    expected_keys = {(target, baseline) for target in TARGETS for baseline in BASELINES}
    if len(ranks) != len(expected_keys) or {(r["target"], r["baseline"]) for r in ranks} != expected_keys:
        raise ValueError("unexpected or duplicate rank keys")
    universes = {int(row["universe_n"]) for row in ranks}
    if len(universes) != 1:
        raise ValueError("rank universes must agree")
    universe = universes.pop()
    if any(not 1 <= float(row["median_rank"]) <= universe for row in ranks):
        raise ValueError("rank outside the source universe")
    recovery = numeric_matrix("Fig5c_policy_recovery.csv", "arm", TARGETS, ARMS)
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(183 / 25.4, 8.8))
        a = fig.add_axes((0.17, 0.765, 0.78, 0.13))
        b = fig.add_axes((0.17, 0.465, 0.77, 0.155))
        c = fig.add_axes((0.12, 0.105, 0.31, 0.245))
        d = fig.add_axes((0.54, 0.09, 0.42, 0.26))

        _panel(a, "A", "Pooled member-state coverage", subtitle=True)
        a.text(0, 1.11, "Independent count expansions · each slot is display-only", transform=a.transAxes,
               fontsize=7, color=MID)
        xs = np.r_[np.arange(13), 15, 18]
        offsets = {"PCD": (0, 13), "LC10a": (13, 14), "CRZ": (14, 15)}
        for column, stage in enumerate(STAGES):
            y = 2 - column
            a.scatter(xs, np.full(15, y), s=21, facecolors="white", edgecolors=LIGHT,
                      linewidths=0.8, gid="slots-" + stage, zorder=1)
            for target, (start, stop) in offsets.items():
                active = states[start:stop, column].astype(bool)
                a.scatter(xs[start:stop][active], np.full(active.sum(), y), s=21,
                          color=COLORS[target], edgecolors="white", linewidths=0.4, zorder=2)
                a.text(xs[stop - 1] + 0.55, y, f"{counts[target, stage]}/{sizes[target]}",
                       fontsize=7, va="center", color=COLORS[target])
        for target, center in zip(TARGETS, (6, 15.3, 18.3)):
            a.text(center, 2.65, target, ha="center", fontsize=8, color=COLORS[target], fontweight="bold")
        a.set_yticks([2, 1, 0], ["Top-10 visible", "Inspected", "Valid perturbation"])
        a.set_ylim(-0.5, 3)
        a.set_xlim(-0.5, 20.2)
        a.set_xticks([])
        a.tick_params(axis="y", length=0)
        a.spines[:].set_visible(False)
        a.text(0, -0.18, "Open: unrecorded · no member identity correspondence across states",
               transform=a.transAxes, fontsize=7, color=MID)
        a.text(0, -0.34, "Slot order is neither time nor a within-run transition sequence.",
               transform=a.transAxes, fontsize=7, color=MID)

        _panel(b, "B", "Full-information rank landscape", subtitle=True)
        b.set_title("Full-information rank landscape", pad=42)
        b.text(0, 1.09, "Median target rank · lower is better", transform=b.transAxes, fontsize=7, color=MID)
        markers = dict(zip(BASELINES, ("o", "s", "D")))
        y_offsets = dict(zip(BASELINES, (0.19, 0, -0.19)))
        for row in ranks:
            target, baseline = row["target"], row["baseline"]
            y = 2 - TARGETS.index(target) + y_offsets[baseline]
            rank = float(row["median_rank"])
            b.scatter(rank, y, marker=markers[baseline], s=27, color=COLORS[target],
                      edgecolors="white", linewidths=0.4, gid=target + "-" + baseline, zorder=3)
        b.set_xscale("log")
        b.set_xlim(1, universe)
        b.set_ylim(-0.5, 2.5)
        b.set_yticks([2, 1, 0], TARGETS)
        b.set_xticks([1, 10, 100, 1000, universe], ["1", "10", "100", "1,000", f"{universe:,}"])
        b.minorticks_off()
        b.grid(axis="x", color=LIGHT, lw=0.6, zorder=0)
        b.axvline(universe, color=MID, lw=0.8)
        b.set_xlabel("Full-information rank (log scale)")
        b.spines[["top", "left", "right"]].set_visible(False)
        b.tick_params(axis="y", length=0)
        handles = [Line2D([], [], marker=markers[key], color="none", markerfacecolor=INK,
                          markeredgecolor="white", markersize=5, label=label)
                   for key, label in zip(BASELINES, ("Weighted strength", "Weighted PageRank", "Baseline visit mass"))]
        b.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.18), ncol=3,
                 frameon=False, fontsize=7, handletextpad=0.3, columnspacing=0.8, borderaxespad=0)

        _panel(c, "C", "Policy recovery", subtitle=True)
        c.text(0, 1.08, "Frozen arm order · mean fraction", transform=c.transAxes, fontsize=7, color=MID)
        cmap = LinearSegmentedColormap.from_list("recovery", ["#F6F8FA", BLUE])
        im = c.imshow(recovery, vmin=0, vmax=1, cmap=cmap, aspect="auto", interpolation="none")
        for row, col in np.ndindex(recovery.shape):
            c.text(col, row, f"{recovery[row, col]:.2f}", ha="center", va="center", fontsize=7, color=INK)
        c.set_xticks(range(3), TARGETS)
        c.set_yticks(range(8), ARMS)
        for text, target in zip(c.get_xticklabels(), TARGETS):
            text.set_color(COLORS[target])
            text.set_fontweight("bold")
        for boundary in (0.5, 3.5):
            c.axhline(boundary, color=MID, lw=0.8)
        c.tick_params(length=0)
        c.spines[:].set_visible(False)
        color_axis = fig.add_axes((0.12, 0.045, 0.31, 0.012))
        cb = fig.colorbar(im, cax=color_axis, orientation="horizontal", ticks=[0, 0.5, 1])
        cb.ax.tick_params(labelsize=7, length=2)
        cb.outline.set_visible(False)

        _panel(d, "D", "Recorded target profiles — GLM", subtitle=True)
        d.text(0, 1.08, "display-only · no identity or time correspondence",
               transform=d.transAxes, fontsize=7, color=MID)
        stage_x = (0.40, 0.65, 0.90)
        for x, label in zip(stage_x, ("exposed", "selected", "recovered")):
            d.text(x, 0.94, label, ha="center", fontsize=7, color=MID)
        statuses = ("limited coverage", "not selected", "not exposed")
        for target, y, status in zip(TARGETS, (0.76, 0.46, 0.16), statuses):
            d.text(0, y + 0.035, target, va="center", fontsize=8,
                   fontweight="bold", color=COLORS[target])
            d.text(0, y - 0.045, status, va="center", fontsize=7, color=COLORS[target])
            for x, stage in zip(stage_x, STAGES):
                count = counts[target, stage]
                # Each marginal count gets its own compact cluster. Neither dot
                # placement nor alignment assigns member identity across states.
                slots = np.arange(count)
                columns = min(count, 4)
                gx = x + (slots % max(columns, 1) - (columns - 1) / 2) * 0.035
                gy = y + ((max(count - 1, 0) // 4) / 2 - slots // 4) * 0.04
                d.scatter(gx, gy, s=13, color=COLORS[target], edgecolors="none",
                          gid=f"state-{target}-{stage}", zorder=3)
                d.text(x, y - 0.105, str(count), ha="center", va="center",
                       fontsize=7, color=COLORS[target], gid=f"count-{target}-{stage}")
        if counts["LC10a", "inspected"] == 0:
            for mark in draw_termination(d, (stage_x[1], 0.46), "cross", ORANGE):
                mark.set_gid("LC10a-selection-stop")
        if counts["CRZ", "visible_top10"] == 0:
            for mark in draw_termination(d, (0.27, 0.16), "boundary", MID):
                mark.set_gid("CRZ-pre-exposure-stop")
        d.text(0.54, 0.012,
               "pooled descriptive records · not transition probabilities or a causal funnel",
               transform=fig.transFigure, ha="center", fontsize=7, color=MID)
        d.set_xlim(-0.02, 1.04)
        d.set_ylim(-0.25, 1)
        d.set_axis_off()
        return fig
