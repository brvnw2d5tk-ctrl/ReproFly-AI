"""fastmcp adapter around ReproflyServerCore.

This module is deliberately thin: every security-relevant decision happens
in the framework-independent core (src/reprofly/mcp/core.py), which the
pytest suite exercises directly. If fastmcp is not installed, this import
fails — the core and the tests remain fully functional without it.
"""

from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from ..config import load_settings
from ..exceptions import ProtocolError, ReproflyError, SecurityError
from ..logging.logger import ExperimentLogger
from ..schemas.tools import TOOL_SCHEMAS
from .core import ReproflyServerCore, SchemaValidationError

mcp: FastMCP = FastMCP("reprofly")

# One server process == one protocol run.
_settings = load_settings()
_logger = ExperimentLogger("logs/experiments.sqlite")
_core = ReproflyServerCore(
    settings=_settings, logger=_logger, experiment_id="phase0-mcp-server"
)


def _dispatch(tool: str, arguments: dict[str, Any]) -> Any:
    """Common wrapper: call as the 'engineer' role and translate protocol
    denials into clean MCP-level error results (never fake data)."""
    try:
        return _core.call_tool(role="engineer", tool=tool, arguments=arguments)
    except SchemaValidationError as exc:
        return {"status": "SCHEMA_ERROR", "detail": str(exc)}
    except (ProtocolError, SecurityError) as exc:
        return {"status": "DENIED", "reason": type(exc).__name__, "detail": str(exc)}
    except ReproflyError as exc:
        return {"status": "ERROR", "reason": type(exc).__name__, "detail": str(exc)}


@mcp.tool
def health_check() -> dict[str, Any]:
    """Liveness and phase status of the ReproFly-AI MCP shell."""
    return _dispatch("health_check", {})


@mcp.tool
def get_protocol_status() -> dict[str, Any]:
    """Protocol version, mode, tool allowlist, and budgets."""
    return _dispatch("get_protocol_status", {})


@mcp.tool
def get_dataset_version() -> dict[str, Any]:
    """Pinned dataset version string."""
    return _dispatch("get_dataset_version", {})


@mcp.tool
def get_remaining_budget() -> dict[str, Any]:
    """Remaining query/perturbation budgets for this run."""
    return _dispatch("get_remaining_budget", {})


@mcp.tool
def get_oracle_status() -> dict[str, Any]:
    """Oracle engineering status: engine built / horizon locked / paper run_silencing enabled."""
    return _dispatch("get_oracle_status", {})


# ---------------------------------------------------------------------------
# Future scientific tools: registered so clients can introspect their
# schemas, but every call routes to the core and yields an explicit
# non-availability status (NOT_IMPLEMENTED / DISABLED_PENDING_HUMAN_APPROVAL).
# ---------------------------------------------------------------------------

@mcp.tool
def inspect_blind_type(blind_id: str, detail_level: str = "coarse") -> dict[str, Any]:
    """[FUTURE/STUB] Coarse structural class of a blinded neuron. NOT implemented."""
    return _dispatch("inspect_blind_type", {"blind_id": blind_id, "detail_level": detail_level})


@mcp.tool
def inspect_neighbors(
    blind_id: str, direction: str = "both", top_k: int = 10
) -> dict[str, Any]:
    """[FUTURE/STUB] Connectome neighbours of a blinded neuron. NOT implemented."""
    return _dispatch(
        "inspect_neighbors", {"blind_id": blind_id, "direction": direction, "top_k": top_k}
    )


@mcp.tool
def run_silencing(
    blind_id: str, phenotype: str = "placeholder_phenotype", strength: float = 1.0
) -> dict[str, Any]:
    """[DISABLED] Virtual silencing — structural engine built; horizon awaits human review."""
    return _dispatch(
        "run_silencing", {"blind_id": blind_id, "phenotype": phenotype, "strength": strength}
    )


@mcp.tool
def run_activation(
    blind_id: str, phenotype: str = "placeholder_phenotype", strength: float = 1.0
) -> dict[str, Any]:
    """[DISABLED] Virtual activation — awaiting human-approved specification."""
    return _dispatch(
        "run_activation", {"blind_id": blind_id, "phenotype": phenotype, "strength": strength}
    )


@mcp.tool
def compare_matched_controls(blind_id: str, n_controls: int = 5) -> dict[str, Any]:
    """[FUTURE/STUB] Matched-control comparison. NOT implemented."""
    return _dispatch(
        "compare_matched_controls", {"blind_id": blind_id, "n_controls": n_controls}
    )


@mcp.tool
def submit_hypothesis(hypothesis_text: str) -> dict[str, Any]:
    """[FUTURE/STUB] Submit a hypothesis. NOT implemented."""
    return _dispatch("submit_hypothesis", {"hypothesis_text": hypothesis_text})


@mcp.tool
def submit_candidates(blind_ids: list[str], ranking_rationale: str) -> dict[str, Any]:
    """[FUTURE/STUB] Submit final blinded candidates. NOT implemented."""
    return _dispatch(
        "submit_candidates",
        {"blind_ids": blind_ids, "ranking_rationale": ranking_rationale},
    )


__all__ = ["mcp", "_core"]
