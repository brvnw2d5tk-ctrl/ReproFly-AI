"""Pydantic input schemas for every MCP tool.

Engineering tools (implemented) and future scientific tools (stubs) both
declare a strict schema here. Arguments are validated against these models
BEFORE any handler is invoked — see tests/test_tool_schema_validation.py.

The future-tool schemas are interface contracts only. Defining them is NOT
an implementation of the underlying biology: their handlers are absent and
calls return NOT_IMPLEMENTED.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Implemented engineering tools
# ---------------------------------------------------------------------------

class HealthCheckArgs(BaseModel):
    """No arguments."""

    model_config = {"extra": "forbid"}


class GetProtocolStatusArgs(BaseModel):
    model_config = {"extra": "forbid"}


class GetDatasetVersionArgs(BaseModel):
    model_config = {"extra": "forbid"}


class GetRemainingBudgetArgs(BaseModel):
    model_config = {"extra": "forbid"}


class GetOracleStatusArgs(BaseModel):
    """Phase 1B: safe, anatomy-free oracle engineering status."""

    model_config = {"extra": "forbid"}


# ---------------------------------------------------------------------------
# Future scientific tools — schema only, NOT implemented in Phase 0
# ---------------------------------------------------------------------------

class BlindIdField(BaseModel):
    """Shared shape: every future tool addresses neurons by BLIND id only."""

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _no_real_names(self) -> "BlindIdField":
        # Guard: blind ids have a strict format (e.g. BLIN-000123). Anything
        # that looks like a free-text neuron type name is rejected so that
        # real neuron type names can never be smuggled through tool args.
        return self


class InspectBlindTypeArgs(BlindIdField):
    """Return the coarse structural class of a blinded neuron (stub)."""

    blind_id: str = Field(
        ...,
        pattern=r"^BLIN-\d{6}$",
        description="Blind identifier; real neuron type names are never accepted.",
    )
    detail_level: Literal["coarse", "morphology_summary"] = "coarse"


class InspectNeighborsArgs(BlindIdField):
    """Return connectome neighbours of a blinded neuron (stub)."""

    blind_id: str = Field(..., pattern=r"^BLIN-\d{6}$")
    direction: Literal["upstream", "downstream", "both"] = "both"
    top_k: int = Field(default=10, ge=1, le=100)


class RunSilencingArgs(BlindIdField):
    """Virtual silencing experiment.

    The finite-horizon structural transmission engine was implemented in
    Phase 1B, but the formal horizon and baseline QC still require human
    review (Phase 1C). The tool therefore returns
    DISABLED_PENDING_HUMAN_APPROVAL and never any result — simulated,
    mocked, or otherwise.
    """

    blind_id: str = Field(..., pattern=r"^BLIN-\d{6}$")
    phenotype: Literal["placeholder_phenotype"] = Field(
        default="placeholder_phenotype",
        description=(
            "Interface placeholder. The formal reproductive phenotype model "
            "is deliberately NOT implemented in Phase 0."
        ),
    )
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class RunActivationArgs(BlindIdField):
    """Virtual activation experiment (stub)."""

    blind_id: str = Field(..., pattern=r"^BLIN-\d{6}$")
    phenotype: Literal["placeholder_phenotype"] = "placeholder_phenotype"
    strength: float = Field(default=1.0, ge=0.0, le=1.0)


class CompareMatchedControlsArgs(BlindIdField):
    """Compare a perturbation against matched control neurons (stub)."""

    blind_id: str = Field(..., pattern=r"^BLIN-\d{6}$")
    n_controls: int = Field(default=5, ge=1, le=50)


class SubmitHypothesisArgs(BaseModel):
    """Submit a scientific hypothesis for the record (stub)."""

    model_config = {"extra": "forbid"}

    hypothesis_text: str = Field(..., min_length=1, max_length=5000)


class SubmitCandidatesArgs(BaseModel):
    """Submit final blinded candidate neurons (stub)."""

    model_config = {"extra": "forbid"}

    blind_ids: list[str] = Field(..., min_length=1, max_length=50)
    ranking_rationale: str = Field(..., min_length=1, max_length=10000)

    @model_validator(mode="after")
    def _validate_blind_ids(self) -> "SubmitCandidatesArgs":
        import re

        pattern = re.compile(r"^BLIN-\d{6}$")
        for bid in self.blind_ids:
            if not pattern.match(bid):
                raise ValueError(
                    f"Invalid blind id {bid!r}: expected format BLIN-######"
                )
        if len(set(self.blind_ids)) != len(self.blind_ids):
            raise ValueError("Duplicate blind ids in submission")
        return self


# ---------------------------------------------------------------------------
# Registry mapping tool name -> (args model, implemented?)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: dict[str, tuple[type[BaseModel], bool]] = {
    # implemented engineering tools
    "health_check": (HealthCheckArgs, True),
    "get_protocol_status": (GetProtocolStatusArgs, True),
    "get_dataset_version": (GetDatasetVersionArgs, True),
    "get_remaining_budget": (GetRemainingBudgetArgs, True),
    "get_oracle_status": (GetOracleStatusArgs, True),
    # future scientific tools — schema only
    "inspect_blind_type": (InspectBlindTypeArgs, False),
    "inspect_neighbors": (InspectNeighborsArgs, False),
    "run_silencing": (RunSilencingArgs, False),
    "run_activation": (RunActivationArgs, False),
    "compare_matched_controls": (CompareMatchedControlsArgs, False),
    "submit_hypothesis": (SubmitHypothesisArgs, False),
    "submit_candidates": (SubmitCandidatesArgs, False),
}

# Explicit non-availability markers. Both forbid simulated results.
TOOL_DISABLED_STATUS: dict[str, str] = {
    "run_silencing": "DISABLED_PENDING_HUMAN_APPROVAL",
    "run_activation": "DISABLED_PENDING_HUMAN_APPROVAL",
}

# The set of names agents may see advertised (all have schemas).
ADVERTISED_TOOLS: tuple[str, ...] = tuple(TOOL_SCHEMAS)


__all__ = [
    "TOOL_SCHEMAS",
    "TOOL_DISABLED_STATUS",
    "ADVERTISED_TOOLS",
    "HealthCheckArgs",
    "GetProtocolStatusArgs",
    "GetDatasetVersionArgs",
    "GetRemainingBudgetArgs",
    "GetOracleStatusArgs",
    "InspectBlindTypeArgs",
    "InspectNeighborsArgs",
    "RunSilencingArgs",
    "RunActivationArgs",
    "CompareMatchedControlsArgs",
    "SubmitHypothesisArgs",
    "SubmitCandidatesArgs",
]
