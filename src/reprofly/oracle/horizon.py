"""Ground-truth-free horizon calibration (Phase 1B charter, section 18).

Rule: for each channel, compute directed shortest-path distances from the
source set S_c to every effective target t in T_effective_c on the
primary ALL-WEIGHT graph, using EDGE EXISTENCE ONLY (weights are ignored
by construction — BFS, not weighted shortest path).

    d_c(t) = min_{s in S_c} distance(s, t)      (finite distances only)

L95 = the smallest integer L such that at least 95% of REACHABLE target
nodes satisfy d <= L (nearest-rank quantile). All quantiles below use the
same nearest-rank convention for consistency.

If any channel has ZERO reachable targets, L95 is undefined → STOP
condition E (HorizonUndefinedError). Endpoints are never changed to fix
this. Horizon selection never consults ground truth or known circuits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from .errors import HorizonUndefinedError

QUANTILES = (("median", 0.50), ("q75", 0.75), ("q90", 0.90), ("q95", 0.95))


@dataclass(frozen=True)
class HorizonStats:
    ai_name: str
    target_count: int
    reachable_target_count: int
    reachable_fraction: float
    histogram: dict[str, int]        # distance -> count (reachable only)
    median: int
    q75: int
    q90: int
    q95: int                         # == L95 for this channel
    max_finite_distance: int


def multi_source_shortest_distances(
    PT: csr_matrix, source_mask: np.ndarray, max_levels: int = 512
) -> np.ndarray:
    """Directed BFS distances from the source set (edge existence only).

    PT = P.T (CSR); a frontier indicator vector f propagated as
    nb = PT @ f marks every node reachable in one step from the frontier.
    Returns int32 array: 0 at sources, t at nodes first reached at step t,
    -1 for unreachable nodes.
    """
    n = len(source_mask)
    dist = np.full(n, -1, dtype=np.int32)
    frontier = np.zeros(n, dtype=np.float64)
    frontier[source_mask] = 1.0
    dist[source_mask] = 0

    for t in range(1, max_levels + 1):
        nb = PT @ frontier
        new = (nb > 0.0) & (dist < 0)
        if not new.any():
            break
        dist[new] = t
        frontier[:] = 0.0
        frontier[new] = 1.0
    return dist


def _nearest_rank(sorted_finite: np.ndarray, fraction: float) -> int:
    """Smallest value L in the sorted array such that at least
    `fraction` of the values are <= L (nearest-rank convention)."""
    n = len(sorted_finite)
    k = int(np.ceil(fraction * n))
    k = max(1, min(k, n))
    return int(sorted_finite[k - 1])


def distance_stats(
    ai_name: str, dist: np.ndarray, target_effective_mask: np.ndarray
) -> HorizonStats:
    """Per-channel distance statistics and L95."""
    target_dist = dist[target_effective_mask]
    finite = np.sort(target_dist[target_dist >= 0])
    reachable_n = int(len(finite))
    target_n = int(len(target_dist))
    if reachable_n == 0:
        raise HorizonUndefinedError(
            f"channel {ai_name!r}: 0 reachable targets out of {target_n}; "
            "L95 is undefined (STOP condition E). Endpoints must not be "
            "changed to resolve this — report and await human review."
        )
    hist: dict[str, int] = {}
    for d in finite:
        hist[str(int(d))] = hist.get(str(int(d)), 0) + 1
    q = {name: _nearest_rank(finite, f) for name, f in QUANTILES}
    return HorizonStats(
        ai_name=ai_name,
        target_count=target_n,
        reachable_target_count=reachable_n,
        reachable_fraction=reachable_n / target_n if target_n else 0.0,
        histogram=hist,
        median=q["median"],
        q75=q["q75"],
        q90=q["q90"],
        q95=q["q95"],
        max_finite_distance=int(finite[-1]),
    )


def propose_horizon(l95_by_channel: dict[str, int]) -> int:
    """L_proposed = max over channels of L95. PROPOSAL ONLY — the formal
    horizon is locked exclusively by human review (Phase 1C)."""
    if not l95_by_channel:
        raise ValueError("no channels supplied")
    return int(max(l95_by_channel.values()))


def candidate_horizons(l_proposed: int, span: int = 2) -> list[int]:
    """Sensitivity candidate set {L_proposed-2 .. L_proposed+2} with
    non-positive values removed."""
    return [L for L in range(l_proposed - span, l_proposed + span + 1) if L > 0]


def stats_payload(stats: HorizonStats) -> dict:
    """JSON-safe payload for the private calibration artefact."""
    return {
        "ai_name": stats.ai_name,
        "target_count": stats.target_count,
        "reachable_target_count": stats.reachable_target_count,
        "reachable_fraction": stats.reachable_fraction,
        "distance_histogram": stats.histogram,
        "median": stats.median,
        "q75": stats.q75,
        "q90": stats.q90,
        "q95_L95": stats.q95,
        "max_finite_distance": stats.max_finite_distance,
    }


__all__ = [
    "multi_source_shortest_distances",
    "distance_stats",
    "propose_horizon",
    "candidate_horizons",
    "stats_payload",
    "HorizonStats",
]
