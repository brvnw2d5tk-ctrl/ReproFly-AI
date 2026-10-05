"""Canonical MaleCNS v1.0 neuron graph builder.

Engineering aggregation ONLY — no ranking, no centrality, no scoring.
The neuron set and edge filter follow the official paper logic exactly
(see official_rules.py); nothing here invents thresholds.

Edge source: connectome-weights (aggregated directed pairs, 151.9M rows),
read lazily/streaming and semi-joined against the canonical body ids.
Never passes through pandas; the 1.1 GB table is never fully materialised
into an intermediate copy (only the canonical subset is collected).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from .official_rules import canonical_neuron_mask, valid_superclass_expr

# Annotation columns carried into the PRIVATE canonical neuron table.
CANONICAL_NEURON_COLUMNS = [
    "bodyId", "type", "class", "subclass", "superclass", "supertype",
    "instance", "status", "statusLabel",
    "somaSide", "rootSide", "somaNeuromere",
    "entryNerve", "exitNerve", "receptorType",
    "fruDsx", "dimorphism", "birthtime", "vfbId",
]

GE5_THRESHOLD = 5  # official sensitivity threshold from the notebook


@dataclass
class CanonicalGraphResult:
    neurons: pl.DataFrame          # canonical bodies + annotations + has_valid_edge
    edges: pl.DataFrame            # source_bodyId, target_bodyId, weight
    ge5_edges: pl.DataFrame        # same schema, weight >= 5
    stats: dict = field(default_factory=dict)


def build_canonical_graph(
    annotations: pl.DataFrame,
    weights: pl.LazyFrame,
) -> CanonicalGraphResult:
    """Build the canonical neuron graph from official annotations + weights.

    Parameters
    ----------
    annotations : eager body-annotations table (must contain at least
        bodyId / superclass, plus any annotation columns to carry).
    weights : LAZY connectome-weights frame (body_pre, body_post, weight).
    """
    for col in ("bodyId", "superclass"):
        if col not in annotations.columns:
            raise ValueError(f"annotations must contain {col!r}")

    # --- canonical neuron set (official rule) --------------------------
    carry = [c for c in CANONICAL_NEURON_COLUMNS if c in annotations.columns]
    canonical = annotations.filter(canonical_neuron_mask(annotations)).select(carry)

    canon_ids = canonical.select("bodyId")

    # --- canonical edges: both endpoints canonical, streaming ----------
    canon_ids_lf = canonical.select("bodyId").lazy()
    edges_lf = (
        weights.select(["body_pre", "body_post", "weight"])
        .join(canon_ids_lf, left_on="body_pre", right_on="bodyId", how="semi")
        .join(canon_ids_lf, left_on="body_post", right_on="bodyId", how="semi")
    )
    edges = edges_lf.collect(engine="streaming")

    # integrity: weights must be strictly positive
    min_weight = edges["weight"].min()
    if min_weight is None or min_weight <= 0:
        raise ValueError(
            f"Canonical edges contain non-positive weight (min={min_weight}); "
            "refusing to build graph — investigate the source table."
        )

    # integrity: every endpoint must be a real annotation bodyId
    ann_ids = set(annotations["bodyId"].to_list())
    endpoint_ids = set(edges["body_pre"].to_list()) | set(edges["body_post"].to_list())
    fabricated = endpoint_ids - ann_ids
    if fabricated:
        raise ValueError(
            f"Canonical edges reference {len(fabricated)} body ids absent from "
            "annotations (e.g. sample: "
            f"{sorted(fabricated)[:5]}); refusing to build graph."
        )

    edges = edges.rename({
        "body_pre": "source_bodyId",
        "body_post": "target_bodyId",
    }).select(["source_bodyId", "target_bodyId", "weight"])

    ge5 = edges.filter(pl.col("weight") >= GE5_THRESHOLD)

    # --- has_valid_edge flag on neurons ---------------------------------
    edge_connected = (
        pl.concat([
            edges.select(pl.col("source_bodyId").alias("bodyId")),
            edges.select(pl.col("target_bodyId").alias("bodyId")),
        ]).unique()
    )
    connected_set = set(edge_connected["bodyId"].to_list())
    canonical = canonical.with_columns(
        pl.col("bodyId").is_in(sorted(connected_set)).alias("has_valid_edge")
    )

    # --- stats -----------------------------------------------------------
    raw_rows = int(weights.select(pl.len()).collect(engine="streaming").item())
    n_edge_connected = len(connected_set)
    stats = {
        "annotation_bodies": annotations.height,
        "canonical_neuronal_bodies": canonical.height,
        "raw_connectivity_rows": raw_rows,
        "canonical_neuronal_edges": edges.height,
        "canonical_ge5_edges": ge5.height,
        "edge_connected_canonical_neurons": n_edge_connected,
        "canonical_bodies_without_valid_edge": canonical.height - n_edge_connected,
        "min_weight": int(min_weight),
        "max_weight": int(edges["weight"].max()),
        "total_weight": int(edges["weight"].sum()),
    }
    return CanonicalGraphResult(
        neurons=canonical, edges=edges, ge5_edges=ge5, stats=stats
    )
