"""Ground-truth-free candidate universe construction (Phase 1E §9-11).

Eligibility rule (LOCKED, oracle-horizon-anchored, no ground truth):

  PRIMARY_PATH_ELIGIBLE neuron v  iff
      d_source(v) finite AND d_target(v) finite AND
      d_source(v) + d_target(v) <= 10        (primary horizon L=10)

  d_source(v) : shortest directed hop distance from the primary source
                set to v (edge existence only)
  d_target(v) : shortest directed hop distance from v to the primary
                target set (reverse BFS from targets)

A cell type is eligible iff it contains >= 1 PRIMARY_PATH_ELIGIBLE
neuron. A type containing ANY primary source or target neuron is
ENDPOINT_ASSOCIATED and is excluded from the primary discovery pool
(kept in the PRIVATE registry for future sensitivity) — the AI must not
win by simply shutting off the experiment's own endpoints.

This module computes the universe deterministically from the frozen
Phase 0.5/1A/1B artefacts. It performs NO ranking and uses NO ground
truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from ..graph.official_rules import valid_type_expr
from ..oracle.horizon import multi_source_shortest_distances

PRIMARY_HORIZON = 10          # LOCKED (oracle v1.0)
UNIVERSE_MIN = 500            # charter §11 sanity band
UNIVERSE_MAX = 10_000


@dataclass(frozen=True)
class UniverseAggregates:
    typed_types_total: int
    path_eligible_types: int
    endpoint_associated_excluded: int
    final_eligible_types: int
    eligible_neurons: int
    within_band: bool
    note: str = ""


@dataclass
class UniverseBuild:
    neurons: pl.DataFrame          # per-neuron eligibility (PRIVATE: real types)
    types: pl.DataFrame            # per-type eligibility (PRIVATE)
    aggregates: UniverseAggregates


def compute_path_eligibility(
    P, PT, source_mask: np.ndarray, target_mask: np.ndarray,
    horizon: int = PRIMARY_HORIZON,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (d_source, d_target, eligible_mask) over the node index.

    BFS depth capped at `horizon`: nodes farther than `horizon` hops can
    never satisfy d_source + d_target <= horizon, so capping is exact for
    eligibility."""
    d_source = multi_source_shortest_distances(PT, source_mask,
                                               max_levels=horizon)
    d_target = multi_source_shortest_distances(P, target_mask,
                                               max_levels=horizon)
    eligible = (
        (d_source >= 0) & (d_target >= 0)
        & (d_source + d_target <= horizon)
    )
    return d_source, d_target, eligible


def build_universe(
    canonical_neurons: pl.DataFrame,
    node_ids: np.ndarray,
    P,
    PT,
    primary_source_ids,
    primary_target_ids,
    horizon: int = PRIMARY_HORIZON,
) -> UniverseBuild:
    """Build the candidate universe from frozen artefacts (no ranking)."""
    typed = canonical_neurons.filter(valid_type_expr())
    if typed.height == 0:
        raise ValueError("no typed canonical neurons found")

    def _mask(ids) -> np.ndarray:
        m = np.zeros(len(node_ids), dtype=bool)
        arr = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.int64)
        idx = np.searchsorted(node_ids, arr)
        if not np.array_equal(node_ids[idx], arr):
            raise ValueError("endpoint ids not in node index")
        m[idx] = True
        return m

    src_mask = _mask(primary_source_ids)
    tgt_mask = _mask(primary_target_ids)
    d_source, d_target, eligible = compute_path_eligibility(
        P, PT, src_mask, tgt_mask, horizon
    )

    neurons = typed.with_columns(
        pl.col("bodyId").map_elements(
            lambda b: int(d_source[np.searchsorted(node_ids, int(b))]),
            return_dtype=pl.Int32,
        ).alias("min_hops_from_primary_source"),
        pl.col("bodyId").map_elements(
            lambda b: int(d_target[np.searchsorted(node_ids, int(b))]),
            return_dtype=pl.Int32,
        ).alias("min_hops_to_primary_target"),
    ).with_columns(
        pl.col("bodyId").map_elements(
            lambda b: bool(eligible[np.searchsorted(node_ids, int(b))]),
            return_dtype=pl.Boolean,
        ).alias("primary_path_eligible"),
        pl.col("bodyId").is_in(
            [int(i) for i in primary_source_ids]
            + [int(i) for i in primary_target_ids]
        ).alias("is_endpoint_neuron"),
    )

    type_rows = (
        neurons.group_by("type")
        .agg(
            pl.len().alias("neuron_count"),
            pl.col("primary_path_eligible").any().alias("has_eligible_neuron"),
            pl.col("is_endpoint_neuron").any().alias("endpoint_associated"),
            pl.col("min_hops_from_primary_source")
              .filter(pl.col("min_hops_from_primary_source") >= 0).min()
              .alias("type_min_hops_from_source"),
            pl.col("min_hops_to_primary_target")
              .filter(pl.col("min_hops_to_primary_target") >= 0).min()
              .alias("type_min_hops_to_target"),
        )
        .with_columns(
            (
                pl.col("has_eligible_neuron") & ~pl.col("endpoint_associated")
            ).alias("final_eligible"),
        )
        .sort("type")
    )

    agg = UniverseAggregates(
        typed_types_total=type_rows.height,
        path_eligible_types=int(type_rows["has_eligible_neuron"].sum()),
        endpoint_associated_excluded=int(
            type_rows.filter(
                pl.col("endpoint_associated") & pl.col("has_eligible_neuron")
            ).height
        ),
        final_eligible_types=int(type_rows["final_eligible"].sum()),
        eligible_neurons=int(eligible.sum()),
        within_band=UNIVERSE_MIN <= int(type_rows["final_eligible"].sum())
        <= UNIVERSE_MAX,
        note=(
            "aggregates only — no type names, no ordering"
        ),
    )
    return UniverseBuild(neurons=neurons, types=type_rows, aggregates=agg)


__all__ = [
    "build_universe",
    "compute_path_eligibility",
    "UniverseBuild",
    "UniverseAggregates",
    "PRIMARY_HORIZON",
    "UNIVERSE_MIN",
    "UNIVERSE_MAX",
]
