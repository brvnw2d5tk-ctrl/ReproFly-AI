"""Transition matrix construction and validation.

Mathematical definition (Phase 1B charter, sections 4-5):

    G = (V, E, W), V = 166,400 canonical connected neurons,
    W_ij = synapse-count weight of the directed edge i -> j (all weights,
    no thresholding).

    For every neuron i with at least one outgoing edge:
        P_ij = W_ij / sum_k W_ik          (row-normalised, float64)
    so  sum_j P_ij = 1.

    Neurons with no outgoing edge are STRUCTURAL SINKS: their row is all
    zero. Sinks receive no self-loop, no epsilon, no re-normalisation —
    signal mass that enters a sink is simply lost.

Hard guarantees:
  * W_ij > 0, finite (no negative / NaN / inf) — validated, never coerced
  * (source, target) directed pairs are UNIQUE — a duplicate raises
    DuplicatePairError (STOP condition C); duplicates are never silently
    re-aggregated
  * scipy.sparse CSR only — a dense 166,400 x 166,400 matrix is forbidden
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix, csr_matrix, issparse

from .errors import DuplicatePairError, InvalidWeightError

EDGE_COLUMNS = ("source_bodyId", "target_bodyId", "weight")


@dataclass(frozen=True)
class TransitionQC:
    n_nodes: int
    n_edges: int
    min_weight: int
    max_weight: int
    sink_count: int
    max_row_sum_error: float
    dtype: str
    format: str


def validate_edge_table(edges: pl.DataFrame) -> dict:
    """Validate the canonical edge table. Raises on any violation; never
    repairs data."""
    for col in EDGE_COLUMNS:
        if col not in edges.columns:
            raise ValueError(f"edge table must contain {col!r}")

    w = edges["weight"]
    if w.min() is None or w.min() <= 0:
        raise InvalidWeightError(
            f"edge table contains non-positive weight (min={w.min()}); "
            "W_ij > 0 is required."
        )
    # int64 weights cannot be NaN/inf, but guard against float payloads too
    if w.dtype in (pl.Float32, pl.Float64):
        if edges.filter(
            ~pl.col("weight").is_finite() | pl.col("weight").is_null()
        ).height:
            raise InvalidWeightError("non-finite or null weight found")

    dup = (
        edges.group_by(["source_bodyId", "target_bodyId"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if dup.height:
        sample = dup.head(3).rows(named=True)
        raise DuplicatePairError(
            f"edge table contains {dup.height} duplicate directed pairs "
            f"(e.g. {sample}). The connectome-weights table is documented "
            "as one row per directed pair; duplicates are a provenance "
            "failure (STOP condition C) and are never silently aggregated."
        )
    return {
        "rows": edges.height,
        "min_weight": int(w.min()),
        "max_weight": int(w.max()),
    }


def build_node_index(edges: pl.DataFrame) -> np.ndarray:
    """Sorted unique node ids (int64) covered by the edge table."""
    ids = pl.concat([
        edges.select(pl.col("source_bodyId").alias("bodyId")),
        edges.select(pl.col("target_bodyId").alias("bodyId")),
    ]).unique()["bodyId"].sort()
    return ids.to_numpy()


def _to_index(values: np.ndarray, node_ids: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(node_ids, values)
    if not np.array_equal(node_ids[idx], values):
        missing = values[node_ids[idx] != values][:5]
        raise ValueError(f"edge endpoints not in node index (e.g. {missing})")
    return idx


def build_transition(
    edges: pl.DataFrame, node_ids: np.ndarray
) -> tuple[csr_matrix, np.ndarray, TransitionQC]:
    """Build the row-normalised float64 CSR transition matrix.

    Returns (P, sink_mask, qc). sink_mask[i] is True iff node i has no
    outgoing edge (row of exact zeros).
    """
    src = edges["source_bodyId"].to_numpy()
    dst = edges["target_bodyId"].to_numpy()
    w = edges["weight"].to_numpy().astype(np.float64)

    n = len(node_ids)
    src_idx = _to_index(src, node_ids)
    dst_idx = _to_index(dst, node_ids)

    row_sums = np.bincount(src_idx, weights=w, minlength=n)
    sink_mask = row_sums == 0.0

    data = w / row_sums[src_idx]
    P = coo_matrix(
        (data, (src_idx, dst_idx)), shape=(n, n), dtype=np.float64
    ).tocsr()
    P.sort_indices()

    # row-stochastic verification (non-sink rows sum to 1 within tolerance)
    rs = np.asarray(P.sum(axis=1)).ravel()
    non_sink = ~sink_mask
    max_row_err = float(np.max(np.abs(rs[non_sink] - 1.0))) if non_sink.any() else 0.0
    if max_row_err > 1e-12:
        raise InvalidWeightError(
            f"row-stochastic check failed: max |rowsum-1| = {max_row_err:.3e}"
        )
    if sink_mask.any() and np.any(rs[sink_mask] != 0.0):
        raise InvalidWeightError("sink rows must be exactly zero")

    qc = TransitionQC(
        n_nodes=n,
        n_edges=int(edges.height),
        min_weight=int(edges["weight"].min()),
        max_weight=int(edges["weight"].max()),
        sink_count=int(sink_mask.sum()),
        max_row_sum_error=max_row_err,
        dtype=str(P.dtype),
        format=P.format,
    )
    return P, sink_mask, qc


def assert_sparse_csr(P) -> None:
    if not issparse(P) or P.format != "csr":
        raise ValueError(f"transition matrix must be scipy CSR, got {type(P)}")
    if P.dtype != np.float64:
        raise ValueError(f"transition matrix must be float64, got {P.dtype}")


__all__ = [
    "validate_edge_table",
    "build_node_index",
    "build_transition",
    "assert_sparse_csr",
    "TransitionQC",
    "EDGE_COLUMNS",
]
