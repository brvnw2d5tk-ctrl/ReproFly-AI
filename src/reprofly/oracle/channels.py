"""Channel definitions and mask construction.

Four transmission channels over the frozen Phase 1A endpoints plus one
new purely annotation-based broad peripheral control:

    primary    : R_IN_ABNT  -> R_OUT_ABNT
                 (AbNT-associated reproductive-enriched peripheral
                  input-to-output structural transmission)
    control_1  : CTRL_ABDOMINAL_IN -> CTRL_ABDOMINAL_OUT
                 (abdominal location-matched structural control)
    control_2  : CTRL_LEG_IN -> CTRL_LEG_OUT
                 (thoracic leg sensorimotor structural control)
    control_3  : CTRL_GLOBAL_IN -> CTRL_GLOBAL_OUT
                 (broad peripheral input-to-output structural control;
                  every primary-graph neuron with a non-empty entryNerve /
                  exitNerve annotation — annotation presence ONLY)

AI-facing names are exactly: primary, control_1, control_2, control_3.
The mapping from AI-facing names to real anatomy exists ONLY in the
PRIVATE channel registry; it is never exposed to any Scientist-facing
interface (Phase 1B charter, section 15).

Source/target semantics (sections 7-8):
  * x0_i = 1/|S| for i in S — equal source mass, NEVER re-weighted by
    synapse count / degree / type size / centrality
  * T_effective = T \\ S — at least one synaptic step is required;
    zero-hop success is impossible
  * source_target_overlap_n = |S n T| is always reported; original
    endpoint files are never modified by this rule
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import polars as pl

CHANNEL_AI_NAMES = ("primary", "control_1", "control_2", "control_3")

_PRIMARY_ENDPOINT = ("R_IN_ABNT", "R_OUT_ABNT")
_ABDOMINAL_ENDPOINT = ("CTRL_ABDOMINAL_IN", "CTRL_ABDOMINAL_OUT")
_LEG_ENDPOINT = ("CTRL_LEG_IN", "CTRL_LEG_OUT")

# Human-facing (Methods) labels — PRIVATE side only.
CHANNEL_LABELS = {
    "primary": (
        "AbNT-associated reproductive-enriched peripheral input-to-output "
        "structural transmission"
    ),
    "control_1": "abdominal location-matched structural control",
    "control_2": "thoracic leg sensorimotor structural control",
    "control_3": "broad peripheral input-to-output structural control",
}


@dataclass(frozen=True)
class ChannelMasks:
    """Boolean masks over the node index for one channel."""

    ai_name: str
    source_mask: np.ndarray
    target_mask: np.ndarray
    target_effective_mask: np.ndarray    # T \\ S
    source_n: int
    target_n: int
    target_effective_n: int
    source_target_overlap_n: int

    @property
    def labels(self) -> dict[str, str]:
        return {"label": CHANNEL_LABELS[self.ai_name]}


def _mask_from_ids(node_ids: np.ndarray, ids) -> np.ndarray:
    m = np.zeros(len(node_ids), dtype=bool)
    if ids is None or len(ids) == 0:
        return m
    arr = np.asarray(sorted(set(int(i) for i in ids)), dtype=np.int64)
    idx = np.searchsorted(node_ids, arr)
    idx_clipped = np.clip(idx, 0, max(len(node_ids) - 1, 0))
    ok = np.array_equal(node_ids[idx_clipped], arr)
    if not ok:
        missing = sorted(set(arr.tolist()) - set(node_ids.tolist()))[:5]
        raise ValueError(
            f"channel ids not present in the node index "
            f"(e.g. {missing}). All channel endpoints must be nodes of "
            "the primary graph."
        )
    m[idx] = True
    return m


def build_channel_masks(
    ai_name: str,
    node_ids: np.ndarray,
    source_ids,
    target_ids,
) -> ChannelMasks:
    src = _mask_from_ids(node_ids, source_ids)
    tgt = _mask_from_ids(node_ids, target_ids)
    t_eff = tgt & ~src
    return ChannelMasks(
        ai_name=ai_name,
        source_mask=src,
        target_mask=tgt,
        target_effective_mask=t_eff,
        source_n=int(src.sum()),
        target_n=int(tgt.sum()),
        target_effective_n=int(t_eff.sum()),
        source_target_overlap_n=int((src & tgt).sum()),
    )


def primary_ids_from_endpoint(df: pl.DataFrame) -> list[int]:
    """bodyIds of the PRIMARY-GRAPH tier of an endpoint membership table."""
    return (
        df.filter(pl.col("canonical_graph_membership") == "primary_graph")
        ["bodyId"].to_list()
    )


def build_global_controls(canonical: pl.DataFrame) -> tuple[list[int], list[int]]:
    """CTRL_GLOBAL_IN / CTRL_GLOBAL_OUT.

    Definition (section 14, CONTROL 3): every canonical CONNECTED neuron
    (primary graph) whose entryNerve / exitNerve annotation is non-empty.
    Annotation presence only — no nerve identity, no literature, no
    ground truth. NOT a whole-brain channel (only peripherally annotated
    neurons are included).
    """
    if "has_valid_edge" not in canonical.columns:
        raise ValueError("canonical table must carry has_valid_edge")

    def _ids(field: str) -> list[int]:
        return (
            canonical.filter(
                pl.col("has_valid_edge")
                & pl.col(field).is_not_null()
                & (pl.col(field).str.strip_chars() != "")
            )["bodyId"].to_list()
        )

    return _ids("entryNerve"), _ids("exitNerve")


def build_all_channel_masks(
    node_ids: np.ndarray,
    endpoint_frames: dict[str, pl.DataFrame],
    canonical: pl.DataFrame,
    primary_tier: Callable[[pl.DataFrame], list[int]] = primary_ids_from_endpoint,
) -> dict[str, ChannelMasks]:
    """Build all four channels from the frozen Phase 1A endpoint tables
    plus the canonical annotation table."""
    for gid in (*_PRIMARY_ENDPOINT, *_ABDOMINAL_ENDPOINT, *_LEG_ENDPOINT):
        if gid not in endpoint_frames:
            raise ValueError(f"missing endpoint table: {gid}")

    glob_in, glob_out = build_global_controls(canonical)

    return {
        "primary": build_channel_masks(
            "primary", node_ids,
            primary_tier(endpoint_frames[_PRIMARY_ENDPOINT[0]]),
            primary_tier(endpoint_frames[_PRIMARY_ENDPOINT[1]]),
        ),
        "control_1": build_channel_masks(
            "control_1", node_ids,
            primary_tier(endpoint_frames[_ABDOMINAL_ENDPOINT[0]]),
            primary_tier(endpoint_frames[_ABDOMINAL_ENDPOINT[1]]),
        ),
        "control_2": build_channel_masks(
            "control_2", node_ids,
            primary_tier(endpoint_frames[_LEG_ENDPOINT[0]]),
            primary_tier(endpoint_frames[_LEG_ENDPOINT[1]]),
        ),
        "control_3": build_channel_masks(
            "control_3", node_ids, glob_in, glob_out,
        ),
    }


def registry_payload(
    masks: dict[str, ChannelMasks],
    node_ids: np.ndarray,
    endpoint_anatomy: dict[str, str],
) -> dict:
    """PRIVATE channel registry payload (never Scientist-visible).

    Contains the real anatomy mapping AND the raw bodyId membership so a
    future session can rebuild masks without touching endpoint parquets.
    """
    def ids_of(mask: np.ndarray) -> list[int]:
        # map boolean mask positions back to REAL bodyIds (not indices)
        return [int(v) for v in node_ids[mask]]

    channels = {
        ai: {
            "ai_name": ai,
            "anatomy": endpoint_anatomy.get(ai, ""),
            "label": CHANNEL_LABELS[ai],
            "source_bodyIds": ids_of(m.source_mask),
            "target_bodyIds": ids_of(m.target_mask),
            "counts": {
                "source_n": m.source_n,
                "target_n": m.target_n,
                "target_effective_n": m.target_effective_n,
                "source_target_overlap_n": m.source_target_overlap_n,
            },
        }
        for ai, m in masks.items()
    }
    return {
        "private": True,
        "warning": (
            "PRIVATE channel registry: contains real anatomy names and "
            "bodyIds. Never expose to Scientist-facing interfaces."
        ),
        "channels": channels,
    }


__all__ = [
    "CHANNEL_AI_NAMES",
    "CHANNEL_LABELS",
    "ChannelMasks",
    "build_channel_masks",
    "primary_ids_from_endpoint",
    "build_global_controls",
    "build_all_channel_masks",
    "registry_payload",
]
