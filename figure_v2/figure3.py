"""Multiscale association evidence, with explicitly illustrative search geometry."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .data_contract import load_rows
from .figure1 import BLUE, ORANGE, WINE, MID, LIGHT, STYLE, _panel
from .visual_primitives import schematic_manifold, draw_manifold, draw_progressive_path, draw_termination


def build_figure3():
    """Build four panels from frozen associations, paired values and sensitivity."""
    associations = load_rows("Fig3a_associations.csv")
    runs = load_rows("Fig3b_per_run.csv")
    tiles = load_rows("Fig3c_m13_tiles.csv")
    levels = ("L5", "L13", "ge5", "overlap_excluded")
    measures = ("primary", "control_1", "control_2", "control_3")
    behaviors = ("NEXT_LOCAL", "NEXT_DE_NOVO")
    expected = [(level, measure + "|" + behavior) for level in levels
                for measure in measures for behavior in behaviors]
    if [(r["relabel_level"], r["contrast"]) for r in tiles] != expected:
        raise ValueError("sensitivity source must retain its exact 4-by-8 frozen order")
    concordance = np.asarray([float(r["sign_concordance_fraction"]) for r in tiles]).reshape(4, 8)
    if not np.isfinite(concordance).all() or not ((concordance >= 0) & (concordance <= 1)).all():
        raise ValueError("invalid sign concordance")
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(183 / 25.4, 9.1))
        a = fig.add_axes((0.15, 0.66, 0.33, 0.265))
        b = fig.add_axes((0.62, 0.66, 0.33, 0.265))
        c = fig.add_axes((0.22, 0.365, 0.70, 0.155))
        d_local = fig.add_axes((0.10, 0.04, 0.37, 0.185))
        d_novel = fig.add_axes((0.57, 0.04, 0.37, 0.185))

        _panel(a, "A", "Prediction–behaviour association", subtitle=True)
        a.text(0, 1.05, "Median + IQR; computable runs", transform=a.transAxes, fontsize=7, color=MID)
        a.axvline(0, color=MID, lw=0.8, ls=(0, (3, 3)))
        by_key = {(r["response_measure"], r["next_behavior"]): r for r in associations}
        labels, positions = [], []
        for behavior, top, color, label in (("NEXT_LOCAL", 9, BLUE, "Local continuation"),
                                             ("NEXT_DE_NOVO", 3, ORANGE, "De-novo exploration")):
            a.text(-0.98, top + 0.85, label, color=color, fontsize=8, fontweight="bold")
            for index, measure in enumerate(measures):
                row = by_key[measure, behavior]
                y = top - index
                key = measure + "-" + behavior
                primary = row["measure_class"] == "primary"
                a.axhspan(y - 0.37, y + 0.37, color="#F3F5F7" if primary else "white", zorder=0)
                a.plot([float(row["iqr_low"]), float(row["iqr_high"])], [y, y],
                       lw=3 if primary else 1.4, color=color, alpha=1 if primary else 0.6, gid="iqr-" + key)
                a.plot(float(row["median_assoc"]), y, marker="D" if primary else "o",
                       ms=5 if primary else 4, color=color, mfc=color if primary else "white", gid="median-" + key)
                a.text(0.80, y, f'n={row["n"]}', ha="right", va="center", fontsize=7, color=MID)
                positions.append(y)
                labels.append("Primary" if primary else "Control " + measure[-1])
        a.set_yticks(positions, labels)
        a.set_xlim(-1, 0.85)
        a.set_ylim(-0.7, 10.4)
        a.set_xticks([-1, -0.5, 0, 0.5])
        a.set_xlabel("Run-level association")

        _panel(b, "B", "Primary association by run", subtitle=True)
        b.text(0, 1.05, "paired values; not time series", transform=b.transAxes, fontsize=7, color=MID)
        b.axvline(0, color=MID, lw=0.8, ls=(0, (3, 3)))
        for y, row in enumerate(runs):
            local = float(row["primary_NEXT_LOCAL"])
            value = row["primary_NEXT_DE_NOVO"]
            missing = value == "NOT_COMPUTABLE_INSUFFICIENT_GROUP_SIZE"
            xs = [local] if missing else [float(value), local]
            b.plot(xs, [y] * len(xs), color=LIGHT, lw=1.1, gid="paired-" + row["run_id"])
            b.scatter([local], [y], color=BLUE, s=24, zorder=4)
            if missing:
                b.text(-0.95, y, "not computable", fontsize=7, color=MID, va="center")
            else:
                b.scatter([float(value)], [y], color=ORANGE, s=24, marker="s", zorder=4)
        b.set_yticks(range(len(runs)), [r["run_id"].replace("AI_RUN_", "R") for r in runs])
        b.set_ylim(len(runs) - 0.4, -0.7)
        b.set_xlim(-1, 0.60)
        b.set_xticks([-1, -0.5, 0, 0.5])
        b.set_xlabel("Run-level association")
        b.text(0.00, -0.28, "■ De novo", color=ORANGE, transform=b.transAxes, fontsize=7)
        b.text(0.66, -0.28, "● Local", color=BLUE, transform=b.transAxes, fontsize=7)
        for ax in (a, b):
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)

        _panel(c, "C", "Oracle-relabeling sensitivity", subtitle=True)
        c.text(0, 1.16, "Primary", transform=c.transAxes, fontsize=8, fontweight="bold", ha="left", color=WINE)
        for i, name in enumerate(("Control 1", "Control 2", "Control 3"), 1):
            c.text((2 * i + 1) / 8, 1.16, name, transform=c.transAxes, fontsize=8, ha="center", color=MID)
        image = c.imshow(concordance, cmap="Blues_r", vmin=0.7, vmax=1.0, aspect="auto")
        for (row, col), value in np.ndenumerate(concordance):
            c.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                   color="white" if value < 0.82 else "#202428")
        c.set_xticks(range(8), ["Local", "De novo"] * 4)
        c.set_yticks(range(4), ["L5", "L13", "≥5", "Overlap excluded"])
        c.tick_params(length=0, pad=5)
        for x in (1.5, 3.5, 5.5):
            c.axvline(x, lw=1.5 if x == 1.5 else 0.8, color=MID if x == 1.5 else "white")
        c.spines[:].set_visible(False)
        color_ax = fig.add_axes((0.38, 0.305, 0.35, 0.012))
        cb = fig.colorbar(image, cax=color_ax, orientation="horizontal", ticks=(0.7, 0.8, 0.9, 1.0))
        cb.set_label("Sign concordance (darker = lower)", fontsize=7, labelpad=2)
        cb.outline.set_visible(False)

        _panel(d_local, "D", "Search-pattern interpretation", subtitle=True)
        d_local.texts[-1].set_x(-0.25)
        landscape = schematic_manifold(seed=42)
        # These node choices shape display-only paths; they are not run histories.
        for ax, label, indices in ((d_local, "local", [16, 12, 3, 11, 4, 15]),
                                   (d_novel, "de-novo", [16, 12, 3, 11, 43, 26, 27])):
            draw_manifold(ax, landscape, contours=True)
            ax.collections[-1].set_gid(label + "-space")
            points = landscape.nodes[indices]
            path = draw_progressive_path(ax, points, WINE)
            for index, segment in enumerate(path.segments):
                segment.set_gid(f"{label}-path-{index}")
            draw_termination(ax, points[0], "start", WINE)
            draw_termination(ax, points[-1], "current", WINE)
        d_local.text(0.5, 1.02, "higher prediction", transform=d_local.transAxes,
                     ha="center", fontsize=8, color=WINE)
        d_local.text(0.28, 0.06, "local continuation", transform=d_local.transAxes,
                     ha="center", fontsize=7, color=WINE)
        d_novel.text(0.5, 1.02, "lower prediction", transform=d_novel.transAxes,
                     ha="center", fontsize=8, color=WINE)
        d_novel.text(0.64, 0.06, "de-novo jump", transform=d_novel.transAxes,
                     ha="center", fontsize=7, color=WINE)
        d_local.text(0, -0.16, "schematic association · not causal", transform=d_local.transAxes,
                     fontsize=7, color=MID)
        return fig
