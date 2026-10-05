"""Small deterministic drawing primitives shared by the v2 figure modules.

The primitives draw geometry only.  Figure modules own all scientific labels and
interpretation so the same mark can be used without changing its meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.text import Text


@dataclass(frozen=True)
class Landscape:
    """Display-only candidate geometry, local connections and optional regions."""

    nodes: np.ndarray
    edges: tuple[tuple[int, int], ...]
    region_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class PathArtists:
    """Path fragments and an optional dashed restart jump.

    ``draw_search_path`` retains its legacy solid-only ``segments`` contract;
    ``draw_progressive_path`` includes every segment in traversal order, with
    ``restart`` referencing the dashed member when requested.
    """

    segments: tuple[Line2D, ...]
    restart: Line2D | None


@dataclass(frozen=True)
class RaincloudArtists:
    """Artists comprising one compact distribution layer."""

    density: PolyCollection
    strip: PathCollection
    median: Line2D


def schematic_manifold(seed: int = 42) -> Landscape:
    """Return three schematic islands, never measured candidate coordinates.

    A local generator controls node placement only. Region layout and display
    scale are fixed illustrative choices, with no quantitative axis semantics.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0.25, 0.35], [0.73, 0.37], [0.51, 0.75]])
    scales = np.array([[0.20, 0.12], [0.18, 0.13], [0.20, 0.11]])
    angles = (-0.30, 0.40, -0.15)
    count = 24
    regions = []
    edges = set()
    for region, (center, scale, angle) in enumerate(zip(centers, scales, angles)):
        theta = rng.uniform(0, 2 * np.pi, count)
        radius = np.sqrt(rng.uniform(0.04, 1.0, count))
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        offsets = np.column_stack((np.cos(theta), np.sin(theta))) * radius[:, None] * scale
        nodes = center + offsets @ rotation.T
        regions.append(nodes)
        distances = np.sum((nodes[:, None] - nodes[None, :]) ** 2, axis=2)
        np.fill_diagonal(distances, np.inf)
        for index in range(count):
            for neighbor in np.argsort(distances[index], kind="stable")[:3]:
                edges.add(tuple(sorted((region * count + index, region * count + int(neighbor)))))
    return Landscape(
        nodes=np.concatenate(regions),
        edges=tuple(sorted(edges)),
        region_ids=tuple(np.repeat(np.arange(3), count).tolist()),
    )


def draw_manifold(ax: Axes, landscape: Landscape, *, contours: bool) -> None:
    """Draw an unlabelled schematic background; covariance ellipses are glyphs.

    Contours have no numeric levels and indicate only the illustrative island
    shapes. Calling this function does not mutate the supplied geometry.
    """
    if contours:
        ids = np.asarray(landscape.region_ids)
        for region in sorted(set(landscape.region_ids)):
            nodes = landscape.nodes[ids == region]
            if len(nodes) < 3:
                continue
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(nodes.T))
            order = np.argsort(eigenvalues)[::-1]
            radii = np.sqrt(np.maximum(eigenvalues[order], 0))
            direction = eigenvectors[:, order[0]]
            angle = np.degrees(np.arctan2(direction[1], direction[0]))
            for level in (2.6, 2.1, 1.6):
                ax.add_patch(Ellipse(
                    xy=nodes.mean(axis=0), width=2 * level * radii[0],
                    height=2 * level * radii[1], angle=angle,
                    facecolor="#C9D1DA", edgecolor="#AAB6C4", alpha=0.12,
                    linewidth=0.55, zorder=0,
                ))
    for start, end in landscape.edges:
        ax.plot(landscape.nodes[[start, end], 0], landscape.nodes[[start, end], 1],
                color="#B8C2CD", alpha=0.48, lw=0.45, zorder=1)
    ax.scatter(landscape.nodes[:, 0], landscape.nodes[:, 1], s=9,
               facecolor="#AAB6C4", edgecolors="white", linewidths=0.25,
               alpha=0.70, zorder=2)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_axis_off()


def draw_progressive_path(
    ax: Axes, xy: np.ndarray, color: str, *, dashed_segment: tuple[int, int] | None = None
) -> PathArtists:
    """Draw smooth display paths with increasing width and opacity.

    ``dashed_segment=(i, i + 1)`` identifies exactly one adjacent restart
    segment. Its curved bridge is schematic, not an observed trajectory.
    Every adjacent segment is included in ``segments`` in traversal order.
    """
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("xy must be a finite (n, 2) array with n >= 2")
    if dashed_segment is not None:
        if (not isinstance(dashed_segment, tuple) or len(dashed_segment) != 2
                or not all(isinstance(index, (int, np.integer)) for index in dashed_segment)
                or not 0 <= dashed_segment[0] < len(points) - 1
                or dashed_segment[1] != dashed_segment[0] + 1):
            raise ValueError("dashed_segment must name two adjacent path indices")
    tangents = np.empty_like(points)
    tangents[0], tangents[-1] = points[1] - points[0], points[-1] - points[-2]
    tangents[1:-1] = (points[2:] - points[:-2]) / 2
    t = np.linspace(0, 1, 32)[:, None]
    segments = []
    restart = None
    for index, (start, end) in enumerate(zip(points[:-1], points[1:])):
        dashed = dashed_segment == (index, index + 1)
        if dashed:
            delta = end - start
            control = (start + end) / 2 + 0.30 * np.array([-delta[1], delta[0]])
            curve = (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end
        else:
            # Restart bridges should not bend their adjoining local fragments.
            left_tangent = tangents[index]
            right_tangent = tangents[index + 1]
            if dashed_segment is not None:
                if index == dashed_segment[1]:
                    left_tangent = end - start
                if index + 1 == dashed_segment[0]:
                    right_tangent = end - start
            curve = ((2 * t ** 3 - 3 * t ** 2 + 1) * start
                     + (t ** 3 - 2 * t ** 2 + t) * left_tangent
                     + (-2 * t ** 3 + 3 * t ** 2) * end
                     + (t ** 3 - t ** 2) * right_tangent)
        progress = index / max(len(points) - 2, 1)
        line = ax.plot(curve[:, 0], curve[:, 1], color=color,
                       alpha=0.42 + 0.58 * progress, lw=1.1 + 1.25 * progress,
                       linestyle="--" if dashed else "-", solid_capstyle="round", zorder=4)[0]
        segments.append(line)
        if dashed:
            restart = line
    return PathArtists(segments=tuple(segments), restart=restart)


def draw_termination(ax: Axes, xy: Sequence[float], kind: str, color: str) -> tuple[Line2D, ...]:
    """Draw a shared state glyph at a display coordinate, without text labels.

    Kinds: filled ``start``, open ``inspection``, haloed ``current``, diamond
    ``restart``, termination ``cross``, and boundary-stop ``boundary``.
    """
    styles = {
        "start": ("o", 4.0, color), "inspection": ("o", 4.0, "white"),
        "current": ("o", 4.5, color), "restart": ("D", 4.5, "white"),
        "cross": ("x", 6.0, color), "boundary": ("|", 9.0, color),
    }
    point = np.asarray(xy, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("xy must be one finite (x, y) coordinate")
    if kind not in styles:
        raise ValueError(f"kind must be one of {tuple(styles)}")
    artists = []
    if kind == "current":
        artists.append(ax.plot(*point, marker="o", markersize=9, markerfacecolor="none",
                               markeredgecolor=color, markeredgewidth=1.2, alpha=0.30,
                               linestyle="none", zorder=5)[0])
    marker, size, face = styles[kind]
    artists.append(ax.plot(*point, marker=marker, markersize=size, markerfacecolor=face,
                           markeredgecolor=color, markeredgewidth=1.1,
                           linestyle="none", zorder=6)[0])
    return tuple(artists)


def candidate_landscape(ax: Axes, seed: int, regions: int) -> Landscape:
    """Draw a seeded, region-local candidate landscape and return its exact geometry."""
    if regions < 1:
        raise ValueError("regions must be at least one")

    rng = np.random.default_rng(seed)
    nodes_per_region = 7
    centers = rng.uniform(0.16, 0.84, size=(regions, 2))
    offsets = rng.normal(0.0, 0.075, size=(regions, nodes_per_region, 2))
    nodes = np.clip((centers[:, None, :] + offsets).reshape(-1, 2), 0.04, 0.96)
    edges = tuple(
        (region * nodes_per_region + index, region * nodes_per_region + index + 1)
        for region in range(regions)
        for index in range(nodes_per_region - 1)
    )

    for start, end in edges:
        ax.plot(nodes[[start, end], 0], nodes[[start, end], 1], color="#B9C2CC", lw=0.7, zorder=1)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=18, color="#5A6B7C", edgecolors="white", linewidths=0.45, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return Landscape(nodes=nodes, edges=edges)


def draw_search_path(
    ax: Axes, xy: np.ndarray, color: str, restart_at: int | None = None
) -> PathArtists:
    """Draw a polyline, separating an optional restart by a dashed jump mark."""
    points = np.asarray(xy, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("xy must be a finite (n, 2) array with n >= 2")
    if not np.isfinite(points).all():
        raise ValueError("xy must contain only finite values")
    if restart_at is not None and not 0 < restart_at < len(points):
        raise ValueError("restart_at must split two adjacent path points")

    fragments = (points,) if restart_at is None else (points[:restart_at], points[restart_at:])
    segments = tuple(
        ax.plot(fragment[:, 0], fragment[:, 1], color=color, lw=1.8, solid_capstyle="round", zorder=4)[0]
        for fragment in fragments
        if len(fragment) >= 2
    )
    restart = None
    if restart_at is not None:
        restart = ax.plot(
            points[restart_at - 1 : restart_at + 1, 0],
            points[restart_at - 1 : restart_at + 1, 1],
            color=color,
            lw=1.4,
            linestyle="--",
            zorder=3,
        )[0]
    return PathArtists(segments=segments, restart=restart)


def state_dot_matrix(ax: Axes, states: Sequence[object], colors: Sequence[str]) -> tuple[PathCollection, ...]:
    """Place display-only dots, accepting either state names or a member-state matrix.

    A one-dimensional sequence produces one dot per state.  A two-dimensional
    binary-like matrix uses rows as display-only members and columns as states;
    only active cells receive a dot.
    """
    values = np.asarray(states)
    if values.ndim == 1:
        if len(values) != len(colors):
            raise ValueError("one-dimensional states and colors must have equal length")
        return tuple(
            ax.scatter(index, 0, s=42, color=color, edgecolors="white", linewidths=0.45, zorder=2)
            for index, color in enumerate(colors)
        )
    if values.ndim != 2 or values.shape[1] != len(colors):
        raise ValueError("matrix states must have one column per color")
    artists: list[PathCollection] = []
    for column, color in enumerate(colors):
        active_rows = np.flatnonzero(values[:, column].astype(bool))
        artists.append(
            ax.scatter(
                np.full(active_rows.shape, column),
                active_rows,
                s=30,
                color=color,
                edgecolors="white",
                linewidths=0.4,
                zorder=2,
            )
        )
    return tuple(artists)


def raincloud_layer(ax: Axes, values: Sequence[float], y: float, color: str) -> RaincloudArtists:
    """Draw a horizontal density, raw-observation strip, and median at ``y``."""
    numeric = np.asarray(values, dtype=float)
    if numeric.ndim != 1 or numeric.size == 0 or not np.isfinite(numeric).all():
        raise ValueError("values must be a non-empty finite one-dimensional sequence")
    violin = ax.violinplot(numeric, positions=[y], orientation="horizontal", widths=0.48, showextrema=False)
    density = violin["bodies"][0]
    density.set_facecolor(color)
    density.set_edgecolor("none")
    density.set_alpha(0.28)
    strip = ax.scatter(numeric, np.full(numeric.shape, y), s=16, color=color, edgecolors="white", linewidths=0.35, zorder=3)
    median = ax.plot(np.median(numeric), y, marker="|", markersize=12, markeredgewidth=1.5, color="#1C2630", zorder=4)[0]
    return RaincloudArtists(density=density, strip=strip, median=median)


def semantic_separator(ax: Axes, x_or_y: float, label: str) -> Text:
    """Add a vertical semantic boundary with a concise, unboxed label."""
    ax.axvline(x_or_y, color="#8B96A3", lw=0.8, linestyle=(0, (2, 2)), zorder=0)
    return ax.text(
        x_or_y,
        0.98,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8,
        color="#56616E",
        clip_on=False,
    )
