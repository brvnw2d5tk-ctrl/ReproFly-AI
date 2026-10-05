"""PRIVATE cell-type graph — pure aggregation, no scoring.

From canonical neurons with a VALID real type annotation (official rule),
aggregate neuron-level canonical edges to type-level edges. This module
computes ONLY counts and weight sums. It is forbidden (and unimplemented)
to compute importance, centrality, PageRank, candidate scores, or any
reproductive/functional score here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .official_rules import valid_type_expr


@dataclass
class TypeGraphResult:
    nodes: pl.DataFrame
    edges: pl.DataFrame
    ge5_edges: pl.DataFrame
    stats: dict = field(default_factory=dict)


# Exact output schemas — deliberately aggregation-only. Any addition of a
# ranking/scoring column is a protocol violation.
TYPE_NODE_COLUMNS = {
    "real_type", "neuron_count",
    "superclasses", "n_superclasses",
    "soma_sides", "n_with_entry_nerve", "n_with_exit_nerve",
    "main_soma_neuromere",
}
TYPE_EDGE_COLUMNS = {
    "source_type", "target_type",
    "total_synapse_weight",
    "source_neuron_count", "target_neuron_count", "neuron_pair_count",
}


def typed_canonical_neurons(canonical_neurons: pl.DataFrame) -> pl.DataFrame:
    """Canonical neurons carrying a valid real type (official rule)."""
    return canonical_neurons.filter(valid_type_expr())


def build_type_nodes(typed: pl.DataFrame) -> pl.DataFrame:
    """Aggregate typed canonical neurons per real type. PRIVATE table."""
    nodes = (
        typed.group_by(pl.col("type").alias("real_type"))
        .agg(
            pl.len().alias("neuron_count"),
            pl.col("superclass").drop_nulls().unique().sort()
              .str.join(";").alias("superclasses"),
            pl.col("superclass").drop_nulls().n_unique().alias("n_superclasses"),
            pl.col("somaSide").drop_nulls().unique().sort()
              .str.join("/").alias("soma_sides"),
            pl.col("entryNerve").is_not_null().sum().alias("n_with_entry_nerve"),
            pl.col("exitNerve").is_not_null().sum().alias("n_with_exit_nerve"),
            pl.col("somaNeuromere").drop_nulls().mode().first()
              .alias("main_soma_neuromere"),
        )
        .sort("real_type")
    )
    assert set(nodes.columns) == TYPE_NODE_COLUMNS, (
        f"type_nodes schema drift: {sorted(nodes.columns)}"
    )
    return nodes


def _build_type_edges(edges: pl.DataFrame, typed: pl.DataFrame) -> pl.DataFrame:
    """Aggregate neuron-level edges to type-level edges. PRIVATE table.

    Only edges whose BOTH endpoints carry a valid type are kept (official
    notebook semantics: type aggregation over typed neurons).
    """
    src = typed.select(
        pl.col("bodyId").alias("source_bodyId"),
        pl.col("type").alias("source_type"),
    )
    tgt = typed.select(
        pl.col("bodyId").alias("target_bodyId"),
        pl.col("type").alias("target_type"),
    )
    typed_edges = (
        edges.join(src, on="source_bodyId", how="inner")
             .join(tgt, on="target_bodyId", how="inner")
    )
    out = (
        typed_edges.group_by(["source_type", "target_type"])
        .agg(
            pl.col("weight").sum().alias("total_synapse_weight"),
            pl.col("source_bodyId").n_unique().alias("source_neuron_count"),
            pl.col("target_bodyId").n_unique().alias("target_neuron_count"),
            pl.struct(["source_bodyId", "target_bodyId"])
              .n_unique().alias("neuron_pair_count"),
        )
        .sort(["source_type", "target_type"])
    )
    assert set(out.columns) == TYPE_EDGE_COLUMNS, (
        f"type_edges schema drift: {sorted(out.columns)}"
    )
    return out


def _n_types_in_edges(edges: pl.DataFrame) -> int:
    if edges.height == 0:
        return 0
    all_types = pl.concat([
        edges.select(pl.col("source_type").alias("t")),
        edges.select(pl.col("target_type").alias("t")),
    ])
    return int(all_types["t"].n_unique())


def build_type_graph(
    canonical_neurons: pl.DataFrame,
    canonical_edges: pl.DataFrame,
    canonical_ge5_edges: pl.DataFrame,
) -> TypeGraphResult:
    """Build the PRIVATE type-level graph from canonical outputs."""
    typed = typed_canonical_neurons(canonical_neurons)
    nodes = build_type_nodes(typed)
    edges = _build_type_edges(canonical_edges, typed)
    ge5 = _build_type_edges(canonical_ge5_edges, typed)
    stats = {
        "typed_canonical_neurons": typed.height,
        "type_node_count": nodes.height,
        "type_edge_count": edges.height,
        "type_edge_ge5_count": ge5.height,
        "types_in_type_edges_ge5": _n_types_in_edges(ge5),
    }
    return TypeGraphResult(nodes=nodes, edges=edges, ge5_edges=ge5, stats=stats)
