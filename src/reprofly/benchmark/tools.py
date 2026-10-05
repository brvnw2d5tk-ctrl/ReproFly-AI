"""Formal Scientist tool contracts (Phase 1E §13-23).

The six-tool allowlist for the formal GLM-5.3 benchmark. These are
SCHEMAS + deterministic response builders only in Phase 1E: the formal
MCP wiring happens in the benchmark phase after protocol lock. None of
these tools is added to the PAPER MCP allowlist in this phase.

Privacy invariants (tested):
  * every payload carries blind IDs only — never real type names, cell
    counts, NT identity, fruDsx, dimorphism, anatomy, bodyIds
  * inspect_candidate returns LOCAL anonymous structure only — never
    oracle predictions, centrality, or ground-truth proximity
  * run_silencing requires a structured precommitment logged BEFORE
    execution; outputs use AI-facing channel names only
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

# The exhaustive formal-Scientist tool allowlist (charter §13).
SCIENTIST_TOOL_ALLOWLIST: tuple[str, ...] = (
    "get_run_status",
    "list_candidate_ids",
    "inspect_candidate",
    "run_silencing",
    "get_experiment_history",
    "submit_final_candidates",
)

NEIGHBOR_TOP_K = 10


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class GetRunStatusArgs(BaseModel):
    model_config = {"extra": "forbid"}


class ListCandidateIdsArgs(BaseModel):
    model_config = {"extra": "forbid"}
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=100, ge=1, le=500)


class InspectCandidateArgs(BaseModel):
    model_config = {"extra": "forbid"}
    blind_id: str = Field(..., pattern=r"^BLIN-\d{6}$")


class Precommitment(BaseModel):
    """Structured precommitment — logged append-only BEFORE the oracle
    executes (charter §17). Immutable once submitted."""

    model_config = {"extra": "forbid"}

    hypothesis_id: str = Field(..., min_length=1, max_length=64)
    target_blind_id: str = Field(..., pattern=r"^BLIN-\d{6}$")
    hypothesis: str = Field(..., min_length=10, max_length=4000)
    predicted_primary_direction: Literal[
        "decrease", "increase", "no_change", "uncertain"
    ]
    predicted_control_pattern: str = Field(
        ..., min_length=5, max_length=2000,
        description="predicted pattern across control_1/2/3",
    )
    alternative_explanation: str = Field(..., min_length=10, max_length=4000)
    falsification_criterion: str = Field(..., min_length=10, max_length=4000)
    reason_for_experiment: str = Field(..., min_length=10, max_length=4000)


class RunSilencingArgs(BaseModel):
    model_config = {"extra": "forbid"}
    precommitment: Precommitment


class GetExperimentHistoryArgs(BaseModel):
    model_config = {"extra": "forbid"}


class FinalCandidateEntry(BaseModel):
    model_config = {"extra": "forbid"}
    rank: int = Field(..., ge=1, le=50)
    blind_id: str = Field(..., pattern=r"^BLIN-\d{6}$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    supporting_experiment_ids: list[str] = Field(
        ..., min_length=1, max_length=50,
    )
    rationale: str = Field(..., min_length=10, max_length=2000)

    @model_validator(mode="after")
    def _rank_one_first(self) -> "FinalCandidateEntry":
        if self.rank == 1 and self.confidence <= 0:
            raise ValueError("rank-1 entry requires positive confidence")
        return self


class SubmitFinalCandidatesArgs(BaseModel):
    model_config = {"extra": "forbid"}
    candidates: list[FinalCandidateEntry] = Field(..., min_length=1, max_length=50)

    @model_validator(mode="after")
    def _unique_ranks_and_ids(self) -> "SubmitFinalCandidatesArgs":
        ranks = [c.rank for c in self.candidates]
        ids = [c.blind_id for c in self.candidates]
        if len(set(ranks)) != len(ranks):
            raise ValueError("duplicate ranks in submission")
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate blind ids in submission")
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("ranks must be exactly 1..N contiguous")
        return self


TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "get_run_status": GetRunStatusArgs,
    "list_candidate_ids": ListCandidateIdsArgs,
    "inspect_candidate": InspectCandidateArgs,
    "run_silencing": RunSilencingArgs,
    "get_experiment_history": GetExperimentHistoryArgs,
    "submit_final_candidates": SubmitFinalCandidatesArgs,
}


# ---------------------------------------------------------------------------
# Response builders (deterministic; private inputs -> blind outputs)
# ---------------------------------------------------------------------------

def list_candidate_ids_response(
    eligible_blind_ids: list[str], page: int, page_size: int,
) -> dict[str, Any]:
    """Fixed deterministic ordering = blind-ID lexical order (charter
    §14: no degree/effect/distance/importance/ground-truth ordering).
    IDs only; free of charge."""
    ordered = sorted(eligible_blind_ids)
    start = (page - 1) * page_size
    chunk = ordered[start:start + page_size]
    return {
        "page": page,
        "page_size": page_size,
        "total_eligible": len(ordered),
        "total_pages": (len(ordered) + page_size - 1) // page_size,
        "blind_ids": chunk,
    }


def inspect_candidate_response(
    blind_id: str,
    tested_status: bool,
    min_hops_from_source: Optional[int],
    min_hops_to_target: Optional[int],
    incoming: list[tuple[str, float]],   # (blind_neighbor_id, weight)
    outgoing: list[tuple[str, float]],
) -> dict[str, Any]:
    """Structural Blind Level 1 payload (charter §15-16).

    incoming/outgoing: type-level aggregate structural edge weights,
    top-10 by weight descending. edge_weight_fraction = neighbour edge
    weight / total (incoming or outgoing) weight of the queried type.
    Contains NO oracle results, NO centrality, NO ground-truth
    proximity, NO real identities or counts.
    """
    def _top(edges: list[tuple[str, float]]) -> list[dict[str, Any]]:
        total = sum(w for _b, w in edges) or 1.0
        top = sorted(edges, key=lambda x: (-x[1], x[0]))[:NEIGHBOR_TOP_K]
        return [
            {"blind_id": b, "edge_weight_fraction": w / total}
            for b, w in top
        ]

    return {
        "blind_id": blind_id,
        "tested_status": tested_status,
        "min_hops_from_primary_source": min_hops_from_source,
        "min_hops_to_primary_target": min_hops_to_target,
        "top_incoming_blind_neighbors": _top(incoming),
        "top_outgoing_blind_neighbors": _top(outgoing),
    }


def run_silencing_response(
    readouts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Four-channel oracle readout, AI-facing channel names only
    (charter §18). Each channel: baseline / silenced / absolute_change /
    retention / relative_loss / status (BASELINE_ZERO handling; never an
    epsilon). L=10 all-weight primary oracle ONLY — sensitivities are
    computed post-hoc by the deterministic pipeline, never shown live."""
    if set(readouts) != {"primary", "control_1", "control_2", "control_3"}:
        raise ValueError("run_silencing must return exactly 4 channels")
    return {"oracle_release": "v1.0", "horizon": 10, "channels": readouts}


__all__ = [
    "SCIENTIST_TOOL_ALLOWLIST",
    "TOOL_SCHEMAS",
    "Precommitment",
    "list_candidate_ids_response",
    "inspect_candidate_response",
    "run_silencing_response",
    "NEIGHBOR_TOP_K",
]
