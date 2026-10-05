"""Matched policy schematics and frozen distribution, interval and effect evidence."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .data_contract import load_rows
from .figure1 import BLUE, ORANGE, WINE, MID, LIGHT, STYLE, _panel
from .visual_primitives import (
    schematic_manifold, draw_manifold, draw_progressive_path, draw_termination, raincloud_layer,
)


def build_figure4():
    """Render illustrative policy operations separately from measured outcomes."""
    raw = load_rows("Fig4b_raincloud.csv")
    estimates = load_rows("Fig4c_anyhit_ci.csv")
    effects = load_rows("Fig4d_effects.csv")
    colors = {"GLM": WINE, "C2": BLUE, "D1": ORANGE}
    with mpl.rc_context(STYLE):
        fig = plt.figure(figsize=(183 / 25.4, 8.4))
        c2 = fig.add_axes((0.12, 0.70, 0.36, 0.20))
        d1 = fig.add_axes((0.58, 0.70, 0.36, 0.20))
        b = fig.add_axes((0.17, 0.365, 0.74, 0.215))
        c = fig.add_axes((0.17, 0.085, 0.31, 0.16))
        d = fig.add_axes((0.64, 0.085, 0.30, 0.16))
        _panel(c2, "A", "Matched policy geometries", subtitle=True)
        c2.texts[-1].set_x(-0.25)
        c2.text(0, 1.04, "C2 · greedy local", transform=c2.transAxes, color=BLUE, fontsize=8)
        d1.text(0, 1.04, "D1 · record-minimum restart", transform=d1.transAxes, color=ORANGE, fontsize=8)
        landscape = schematic_manifold(seed=42)
        # Identical background, distinct illustrative operations: no observed history.
        for ax, arm, indices in ((c2, "C2", [16, 12, 3, 11, 4, 15]),
                                 (d1, "D1", [16, 12, 3, 11, 43, 26, 27])):
            draw_manifold(ax, landscape, contours=True)
            ax.collections[-1].set_gid(arm + "-space")
            points = landscape.nodes[indices]
            paths = draw_progressive_path(ax, points, colors[arm],
                                           dashed_segment=(3, 4) if arm == "D1" else None)
            for index, segment in enumerate(paths.segments):
                segment.set_gid(f"{arm}-path-{index}")
            draw_termination(ax, points[0], "start", colors[arm])
            for point in points[1:4]:
                draw_termination(ax, point, "inspection", colors[arm])
            draw_termination(ax, points[-1], "current", colors[arm])
            if paths.restart is not None:
                paths.restart.set_gid("D1-restart")
                draw_termination(ax, points[4], "restart", colors[arm])
                ax.text(0.52, 0.17, "restart", color=ORANGE, fontsize=7, ha="center")
            else:
                ax.text(0.28, 0.10, "local-only", color=BLUE, fontsize=7, ha="center")
        c2.text(0, -0.11, "illustrative policy geometry · histories are not identical", transform=c2.transAxes,
                color=MID, fontsize=7)

        _panel(b, "B", "Matched-control distributions", subtitle=True)
        b.text(0, 1.05, "All runs · density + observations + IQR + median", transform=b.transAxes, color=MID, fontsize=7)
        rng = np.random.default_rng(42)
        all_values = []
        for y, arm in zip((2, 1, 0), ("GLM", "C2", "D1")):
            values = np.asarray([float(r["AUTSRC"]) for r in raw if r["arm"] == arm])
            all_values.extend(values)
            layer = raincloud_layer(b, values, y, colors[arm])
            layer.density.set_gid(arm + "-density")
            # Keep a half-density above and raw observations below each summary.
            for path in layer.density.get_paths():
                path.vertices[:, 1] = np.maximum(path.vertices[:, 1], y)
            jitter = rng.uniform(-0.24, -0.08, len(values))
            layer.strip.set_offsets(np.column_stack([values, y + jitter]))
            layer.strip.set_sizes(np.full(len(values), 12))
            layer.strip.set_gid(arm + "-raw")
            layer.median.set_gid(arm + "-median")
            b.plot(np.quantile(values, [0.25, 0.75]), [y, y], lw=4, color=colors[arm], alpha=0.7,
                   solid_capstyle="butt", gid=arm + "-iqr", zorder=3)
            zero_count = np.count_nonzero(values == 0)
            b.text(0.98, (y + 0.73) / 3, f"{zero_count}/{len(values)} zero", transform=b.transAxes,
                   ha="right", fontsize=7, color=colors[arm])
        b.axvline(0, color=MID, lw=0.7, ls=(0, (2, 2)), zorder=0)
        b.set_xlim(-0.01, max(all_values) * 1.12)
        b.set_xticks([0, 0.1, 0.2, 0.3])
        b.set_ylim(-0.4, 2.6)
        b.set_yticks([2, 1, 0], [f'{arm} (n={sum(r["arm"] == arm for r in raw)})' for arm in ("GLM", "C2", "D1")])
        b.set_xlabel("AUTSRC")

        _panel(c, "C", "Any-hit estimates", subtitle=True)
        c.text(0, 1.06, "Exact binomial 95% CI", transform=c.transAxes, fontsize=7, color=MID)
        for y, row in zip((2, 1, 0), estimates):
            arm = row["arm"]
            c.plot([float(row["ci95_low"]), float(row["ci95_high"])], [y, y], color=colors[arm], lw=1.6, gid=arm + "-ci")
            c.plot(float(row["any_hit_frac"]), y, marker="D", ms=5, color=colors[arm], gid=arm + "-hit")
        c.set_yticks([2, 1, 0], [f'{r["arm"]}  {r["n_with_hit"]}/{r["n"]}' for r in estimates])
        c.set_xlim(0, 1)
        c.set_ylim(-0.55, 2.5)
        c.set_xticks([0, 0.5, 1])
        c.set_xlabel("Fraction of runs with any hit")

        _panel(d, "D", "Effect-size context", subtitle=True)
        d.text(0, 1.06, "GLM versus matched policies", transform=d.transAxes, fontsize=7, color=MID)
        d.axvline(0, color=MID, lw=0.8, ls=(0, (3, 3)))
        for y, row in zip((1.5, 0), effects):
            arm = row["contrast"].split("_vs_")[1]
            value = float(row["cliffs_delta"])
            d.plot(value, y, marker="D", ms=6, color=colors[arm], gid=row["contrast"] + "-delta")
            d.text(value, y + 0.27, f"δ = {value:.3f}", fontsize=7, ha="center", color=colors[arm])
            d.text(0.01, y - 0.37, f'GLM median: {float(row["glm_median_percentile"]):.0f}th percentile', fontsize=7, color=MID)
            if row["perm_p_two_sided"]:
                d.text(0.01, y - 0.67, f'Exploratory p = {float(row["perm_p_two_sided"]):.3f}', fontsize=7, color=MID)
            else:
                d.text(0.01, y - 0.67, "Secondary non-adaptive contrast", fontsize=7, color=MID)
        d.set_yticks([1.5, 0], ["GLM–D1", "GLM–C2"])
        d.set_xlim(-0.01, 0.42)
        d.set_ylim(-0.9, 2.1)
        d.set_xticks([0, 0.2, 0.4])
        d.set_xlabel("Cliff's δ")
        for ax in (b, c, d):
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.tick_params(axis="y", length=0)
        return fig
