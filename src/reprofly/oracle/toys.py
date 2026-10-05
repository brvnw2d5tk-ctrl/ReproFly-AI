"""Canonical toy graphs for oracle validation (Phase 1B charter, §27).

Thirteen deterministic toy scenarios pinning the exact mathematical
semantics: chain arrival horizon, weighted branching, absorption vs
double counting, recurrent cycles, silencing of intermediates / sources /
targets, sink mass loss, multi-source equal mass, source-target overlap,
monotonicity, sparse-vs-dense equivalence, and determinism.

Every toy is a tiny explicit graph; no biological meaning is attached to
any toy node id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from .channels import build_channel_masks
from .engine import OracleEngine
from .transition import build_node_index, build_transition


@dataclass(frozen=True)
class ToyGraph:
    name: str
    edges: tuple[tuple[int, int, int], ...]   # (source, target, weight)
    sources: tuple[int, ...]
    targets: tuple[int, ...]
    silenced: tuple[int, ...] = ()
    note: str = ""
    expected: dict = field(default_factory=dict)


def _edges_df(toy: ToyGraph) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "source_bodyId": [e[0] for e in toy.edges],
            "target_bodyId": [e[1] for e in toy.edges],
            "weight": [e[2] for e in toy.edges],
        },
        schema={"source_bodyId": pl.Int64, "target_bodyId": pl.Int64,
                "weight": pl.Int64},
    )


def dense_weights(toy: ToyGraph) -> np.ndarray:
    """Dense (N, N) raw-weight matrix for the independent reference."""
    node_ids = build_node_index(_edges_df(toy))
    pos = {int(v): i for i, v in enumerate(node_ids)}
    W = np.zeros((len(node_ids), len(node_ids)), dtype=np.float64)
    for s, t, w in toy.edges:
        W[pos[s], pos[t]] = float(w)
    return W


def build_engine(toy: ToyGraph, channel_name: str = "primary") -> OracleEngine:
    edges = _edges_df(toy)
    node_ids = build_node_index(edges)
    P, _sink_mask, _qc = build_transition(edges, node_ids)
    masks = {
        channel_name: build_channel_masks(
            channel_name, node_ids, toy.sources, toy.targets
        )
    }
    return OracleEngine(node_ids, P, masks)


# ---------------------------------------------------------------------------
# TEST 1 — simple chain: S -> A -> T
# ---------------------------------------------------------------------------

def chain_toy() -> ToyGraph:
    return ToyGraph(
        name="chain",
        edges=((1, 2, 1), (2, 3, 1)),
        sources=(1,),
        targets=(3,),
        note="L=1: 0 success; L=2: 1.0 success",
        expected={"L1": 0.0, "L2": 1.0},
    )


# ---------------------------------------------------------------------------
# TEST 2 — weighted branching: S -> T (3), S -> A (1), A -> T (1)
# ---------------------------------------------------------------------------

def weighted_branch_toy() -> ToyGraph:
    return ToyGraph(
        name="weighted_branch",
        edges=((1, 2, 3), (1, 3, 1), (3, 2, 1)),
        sources=(1,),
        targets=(2,),
        note="step1 absorbed 0.75; step2 additional 0.25; L2 cumulative 1.0",
        expected={"step1": 0.75, "step2_additional": 0.25, "L2": 1.0},
    )


# ---------------------------------------------------------------------------
# TEST 3 — absorption prevents double counting: S -> T, T -> T
# ---------------------------------------------------------------------------

def selfloop_absorption_toy() -> ToyGraph:
    return ToyGraph(
        name="selfloop_absorption",
        edges=((1, 2, 1), (2, 2, 1)),   # real self-edge preserved (section 13)
        sources=(1,),
        targets=(2,),
        note="arrival at T must not re-accumulate via T's self-loop",
        expected={"L2": 1.0, "L10": 1.0},
    )


# ---------------------------------------------------------------------------
# TEST 4 — recurrent cycle: S -> A, A -> S, A -> T
# ---------------------------------------------------------------------------

def recurrent_cycle_toy() -> ToyGraph:
    return ToyGraph(
        name="recurrent_cycle",
        edges=((1, 2, 1), (2, 1, 1), (2, 3, 1)),
        sources=(1,),
        targets=(3,),
        note="first-passage with recurrence: T(2)=0.5, T(4)=0.75, T(inf)=1",
        expected={"L1": 0.0, "L2": 0.5, "L3": 0.5, "L4": 0.75, "L6": 0.875},
    )


# ---------------------------------------------------------------------------
# TEST 5 — silenced intermediate: S -> A -> T, silence A
# ---------------------------------------------------------------------------

def silenced_intermediate_toy() -> ToyGraph:
    return ToyGraph(
        name="silenced_intermediate",
        edges=((1, 2, 1), (2, 3, 1)),
        sources=(1,),
        targets=(3,),
        silenced=(2,),
        note="silence A => transmission 0 at any horizon",
        expected={"transmission": 0.0},
    )


# ---------------------------------------------------------------------------
# TEST 6 — silenced source: two sources, silence S1 (no re-normalisation)
# ---------------------------------------------------------------------------

def silenced_source_toy() -> ToyGraph:
    return ToyGraph(
        name="silenced_source",
        edges=((1, 3, 1), (2, 3, 1)),
        sources=(1, 2),
        targets=(3,),
        silenced=(1,),
        note="initial total mass 0.5 (NOT re-normalised to 1.0)",
        expected={"initial_mass": 0.5, "transmission": 0.5},
    )


# ---------------------------------------------------------------------------
# TEST 7 — silenced target: silence T
# ---------------------------------------------------------------------------

def silenced_target_toy() -> ToyGraph:
    return ToyGraph(
        name="silenced_target",
        edges=((1, 2, 1),),
        sources=(1,),
        targets=(2,),
        silenced=(2,),
        note="arrival at a silenced target never counts as success",
        expected={"transmission": 0.0},
    )


# ---------------------------------------------------------------------------
# TEST 8 — sink node: S -> Z (Z has no outgoing edges)
# ---------------------------------------------------------------------------

def sink_toy() -> ToyGraph:
    # Graph: 1 -> 2 (mass enters structural sink 2 and is lost);
    #        3 -> 2 (node 3 exists in the graph via its own edge to the
    #        sink; from S=1 the target 3 is unreachable).
    # The target set is {3}: unreachable, so transmission stays 0 and the
    # full mass is accounted as sink loss.
    return ToyGraph(
        name="sink",
        edges=((1, 2, 1), (3, 2, 1)),
        sources=(1,),
        targets=(3,),
        note="mass entering a structural sink is lost; accounting holds",
        expected={"transmission": 0.0, "sink_loss": 1.0},
    )


# ---------------------------------------------------------------------------
# TEST 9 — multiple sources with equal initial mass
# ---------------------------------------------------------------------------

def multiple_sources_toy() -> ToyGraph:
    return ToyGraph(
        name="multiple_sources",
        edges=((1, 4, 2), (2, 4, 1), (3, 4, 1), (4, 5, 1)),
        sources=(1, 2, 3),
        targets=(5,),
        note="each source starts with exactly 1/3 mass",
        expected={"initial_mass": 1.0, "L2": 1.0},
    )


# ---------------------------------------------------------------------------
# TEST 10 — source/target overlap: zero-hop never counts
# ---------------------------------------------------------------------------

def overlap_toy() -> ToyGraph:
    return ToyGraph(
        name="source_target_overlap",
        edges=((1, 3, 1), (2, 3, 1)),
        sources=(1, 2),
        targets=(1, 3),                 # node 1 is both source and target
        note="T_effective = T\\S = {3}; the mass starting on node 1 (in T) "
             "is NOT counted zero-hop",
        expected={"overlap_n": 1, "target_effective_n": 1, "L1": 1.0},
    )


# ---------------------------------------------------------------------------
# TEST 11/12/13 base — deterministic pseudo-random toy for monotonicity,
# sparse-vs-dense equivalence and determinism
# ---------------------------------------------------------------------------

def random_toy(seed: int = 42, n_nodes: int = 12, n_edges: int = 30) -> ToyGraph:
    rng = np.random.default_rng(seed)
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n_edges:
        s = int(rng.integers(0, n_nodes))
        t = int(rng.integers(0, n_nodes))
        if s != t:
            pairs.add((s, t))
    edges = tuple((s, t, int(rng.integers(1, 10))) for s, t in sorted(pairs))
    # sources: 3 fixed nodes; targets: 3 other fixed nodes
    return ToyGraph(
        name=f"random_seed{seed}",
        edges=edges,
        sources=(0, 1, 2),
        targets=(9, 10, 11),
        note="deterministic pseudo-random graph (seeded); no self-loops",
    )


ALL_TOYS: tuple[ToyGraph, ...] = (
    chain_toy(),
    weighted_branch_toy(),
    selfloop_absorption_toy(),
    recurrent_cycle_toy(),
    silenced_intermediate_toy(),
    silenced_source_toy(),
    silenced_target_toy(),
    sink_toy(),
    multiple_sources_toy(),
    overlap_toy(),
    random_toy(),
)

__all__ = [
    "ToyGraph",
    "ALL_TOYS",
    "chain_toy",
    "weighted_branch_toy",
    "selfloop_absorption_toy",
    "recurrent_cycle_toy",
    "silenced_intermediate_toy",
    "silenced_source_toy",
    "silenced_target_toy",
    "sink_toy",
    "multiple_sources_toy",
    "overlap_toy",
    "random_toy",
    "build_engine",
    "dense_weights",
]
