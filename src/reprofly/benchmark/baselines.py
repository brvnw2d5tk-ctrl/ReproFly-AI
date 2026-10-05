"""Non-LLM comparator baselines (Phase 1E §31-35).

All four baselines are STATIC or seeded-stochastic rankings over the
SAME candidate universe with the SAME budget (300) as the GLM Scientist.
Phase 1E implements and unit-tests them on SYNTHETIC graphs only; the
formal rankings are NOT generated in this phase (charter §36, §45).

Information asymmetry (documented, not hidden): static baselines may
use the full type-level graph to rank, whereas the GLM Scientist only
has tool-mediated LOCAL structural inspection. The comparison is
therefore information-conservative FOR THE GLM; fairness comes from the
same universe / budget / scorer / checkpoints, not from equal graph
access (charter §35).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl

from .budgets import PERTURBATION_BUDGET, RANDOM_MASTER_SEED, RANDOM_PERMUTATIONS

# ---- PageRank parameters (LOCKED, explicit — never library defaults) -----
PAGERANK_ALPHA = 0.85
PAGERANK_TOL = 1e-10
PAGERANK_MAX_ITER = 1000


class PageRankConvergenceError(RuntimeError):
    """PageRank failed to converge with the LOCKED parameters. Charter
    §33: STOP — do not retune alpha for nicer results."""


# ---------------------------------------------------------------------------
# Baseline 1 — Random (seeded permutations, no replacement)
# ---------------------------------------------------------------------------

def random_permutation_seed(permutation_index: int,
                            master_seed: int = RANDOM_MASTER_SEED) -> int:
    """Deterministic per-permutation seed derived from the locked master."""
    return (master_seed * 1_000_003 + permutation_index * 7 + 11) & 0x7FFFFFFF


def random_ranking(
    eligible_blind_ids: list[str], permutation_index: int,
    master_seed: int = RANDOM_MASTER_SEED,
) -> list[str]:
    """Uniform sampling WITHOUT replacement over the candidate universe,
    deterministic per (master_seed, permutation_index)."""
    rng = np.random.default_rng(
        random_permutation_seed(permutation_index, master_seed)
    )
    ids = list(eligible_blind_ids)
    order = rng.permutation(len(ids))
    return [ids[i] for i in order]


# ---------------------------------------------------------------------------
# Baseline 2 — Weighted Strength (static)
# ---------------------------------------------------------------------------

def weighted_strength_ranking(
    type_edges: pl.DataFrame,
    eligible_real_types: set[str],
    blind_map: dict[str, str],
) -> list[str]:
    """score(type) = weighted in-strength + weighted out-strength on the
    type-level graph; ranked within the candidate universe; ties broken
    by blind-ID lexical order. Uses NO oracle results."""
    in_s = (
        type_edges.group_by("target_type").agg(
            pl.col("total_synapse_weight").sum().alias("in_strength")
        )
    )
    out_s = (
        type_edges.group_by("source_type").agg(
            pl.col("total_synapse_weight").sum().alias("out_strength")
        )
    )
    rows = []
    for t in eligible_real_types:
        i = (in_s.filter(pl.col("target_type") == t)["in_strength"]
             .to_list() or [0])[0]
        o = (out_s.filter(pl.col("source_type") == t)["out_strength"]
             .to_list() or [0])[0]
        rows.append((t, float(i) + float(o)))
    rows.sort(key=lambda x: (-x[1], blind_map[x[0]]))
    return [blind_map[t] for t, _ in rows]


# ---------------------------------------------------------------------------
# Baseline 3 — Weighted PageRank (static, locked parameters)
# ---------------------------------------------------------------------------

def pagerank_ranking(
    node_list: list[str],
    src_idx: np.ndarray,
    dst_idx: np.ndarray,
    weights: np.ndarray,
    eligible_mask: np.ndarray,
    blind_ids: list[str],
    alpha: float = PAGERANK_ALPHA,
    tol: float = PAGERANK_TOL,
    max_iter: int = PAGERANK_MAX_ITER,
) -> list[str]:
    """Weighted PageRank on the directed type graph (explicit locked
    parameters; power iteration; deterministic). Ranks only eligible
    nodes; ties by blind-ID lexical order. Raises
    PageRankConvergenceError on non-convergence (STOP condition)."""
    n = len(node_list)
    w = weights.astype(np.float64)
    out_sum = np.bincount(src_idx, weights=w, minlength=n)
    if np.any(out_sum == 0):
        raise ValueError("type graph has sink nodes; dangling handling "
                         "must be specified before use")
    P = np.zeros((n, n), dtype=np.float64)
    np.add.at(P, (src_idx, dst_idx), w / out_sum[src_idx])

    r = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(max_iter):
        r_new = alpha * (P.T @ r) + (1.0 - alpha) / n
        if np.max(np.abs(r_new - r)) < tol:
            r = r_new
            break
        r = r_new
    else:
        raise PageRankConvergenceError(
            f"PageRank did not converge in {max_iter} iterations at "
            f"tol={tol:g} (alpha={alpha}); LOCKED parameters — report, "
            "do not retune (charter §33)"
        )
    pairs = [
        (r[i], blind_ids[i]) for i in range(n) if eligible_mask[i]
    ]
    pairs.sort(key=lambda x: (-x[0], x[1]))
    return [b for _s, b in pairs]


# ---------------------------------------------------------------------------
# Baseline 4 — Baseline Visit Mass (static, task-aware, non-perturbational)
# ---------------------------------------------------------------------------

def visit_mass_ranking(
    per_type_visit_mass: dict[str, float],
    eligible_real_types: set[str],
    blind_map: dict[str, str],
) -> list[str]:
    """Ranks eligible types by the cumulative active structural mass
    their neurons RECEIVED during the frozen L=10 all-weight primary
    baseline propagation (oracle v1.0 baseline; NO silencing results).
    Endpoint-aware structural routing baseline. Ties by blind-ID lexical
    order. Ground truth is never used to tune anything."""
    pairs = [
        (float(per_type_visit_mass.get(t, 0.0)), blind_map[t])
        for t in eligible_real_types
    ]
    pairs.sort(key=lambda x: (-x[0], x[1]))
    return [b for _s, b in pairs]


__all__ = [
    "random_permutation_seed", "random_ranking",
    "weighted_strength_ranking", "pagerank_ranking", "visit_mass_ranking",
    "PAGERANK_ALPHA", "PAGERANK_TOL", "PAGERANK_MAX_ITER",
    "PageRankConvergenceError", "RANDOM_PERMUTATIONS", "PERTURBATION_BUDGET",
]
