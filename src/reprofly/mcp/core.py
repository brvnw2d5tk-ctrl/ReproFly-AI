"""Framework-independent MCP server core.

`ReproflyServerCore` wires together:
  * the frozen ToolRegistry (endpoint definitions),
  * the ProtocolController (authorisation + budgets + logging),
  * pydantic schema validation of every call's arguments.

It contains NO MCP framework dependency on purpose: the fastmcp adapter in
server.py is a thin shell over this core, and the pytest suite exercises
the core directly. That keeps the security properties testable even when
fastmcp is not installed.

Implemented engineering tools (no biology):
  health_check / get_protocol_status / get_dataset_version / get_remaining_budget

Future tools (schema-only, always NOT_IMPLEMENTED, never simulated):
  inspect_blind_type / inspect_neighbors / run_silencing / run_activation /
  compare_matched_controls / submit_hypothesis / submit_candidates
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Settings, load_settings
from ..exceptions import ReproflyError
from ..logging.logger import ExperimentLogger
from ..protocol.controller import (
    ProtocolController,
    ToolDefinition,
    ToolRegistry,
)
from ..schemas.tools import TOOL_DISABLED_STATUS, TOOL_SCHEMAS
from pydantic import BaseModel, ValidationError


class SchemaValidationError(ReproflyError):
    """Tool arguments failed pydantic validation."""


# ---------------------------------------------------------------------------
# handlers for the four implemented engineering tools (closures get the
# controller / settings injected at build time)
# ---------------------------------------------------------------------------

def _health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "phase": 0,
        "biology_implemented": False,
        "message": (
            "ReproFly-AI MCP shell is alive. Phase 0: engineering tools only; "
            "no biological analysis is implemented."
        ),
    }


def _oracle_status() -> dict[str, Any]:
    """Safe, anatomy-free oracle engineering status.

    Phase 1D: the oracle is LOCKED at v1.0 with primary horizon 10 and
    sensitivity horizons [5, 13]. The formal benchmark protocol is NOT
    locked, so PAPER run_silencing stays disabled. Contains ONLY
    engineering fields — never anatomical naming, private membership, or
    ground-truth information.
    """
    return {
        "oracle_engine_built": True,
        "phase1c_calibration_complete": True,
        "oracle_locked": True,
        "oracle_release": "v1.0",
        "primary_horizon": 10,
        "sensitivity_horizons": [5, 13],
        "horizon_locked": True,
        "paper_run_silencing_enabled": False,
        "benchmark_protocol_locked": False,
        "phase": "1D",
        "note": (
            "Structural transmission oracle locked at v1.0 (primary "
            "horizon 10; sensitivities 5 and 13). The formal benchmark "
            "protocol is not yet pre-registered, so run_silencing remains "
            "disabled and never returns simulated results."
        ),
    }


class ReproflyServerCore:
    def __init__(
        self,
        settings: Settings,
        logger: ExperimentLogger,
        experiment_id: str = "phase0-scaffold",
        run_id: str | None = None,
    ):
        self.settings = settings
        self.logger = logger
        if settings.protocol is None:
            raise ReproflyError("Settings were loaded without a ProtocolSpec")
        self.registry = self._build_registry()
        self.registry.freeze()
        self.controller = ProtocolController(
            spec=settings.protocol,
            registry=self.registry,
            logger=logger,
            mode=settings.mode,
            experiment_id=experiment_id,
            run_id=run_id,
        )

    # ------------------------------------------------------------------
    # registry construction
    # ------------------------------------------------------------------

    def _build_registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        # --- implemented engineering tools -------------------------------
        registry.register(
            ToolDefinition(
                name="health_check",
                description="Liveness + phase status (engineering only).",
                handler=_health_check,
                implemented=True,
            )
        )
        registry.register(
            ToolDefinition(
                name="get_protocol_status",
                description="Report protocol version, mode, allowlist, budgets.",
                handler=self._tool_protocol_status,
                implemented=True,
            )
        )
        registry.register(
            ToolDefinition(
                name="get_dataset_version",
                description="Report the pinned dataset version string.",
                handler=self._tool_dataset_version,
                implemented=True,
            )
        )
        registry.register(
            ToolDefinition(
                name="get_remaining_budget",
                description="Report remaining query/perturbation budgets.",
                handler=self._tool_remaining_budget,
                implemented=True,
            )
        )
        registry.register(
            ToolDefinition(
                name="get_oracle_status",
                description=(
                    "Oracle engineering status: engine built / horizon "
                    "locked / paper run_silencing enabled. No anatomy."
                ),
                handler=_oracle_status,
                implemented=True,
            )
        )

        # --- future scientific tools: schema-only stubs ------------------
        future_descriptions = {
            "inspect_blind_type": (
                "[FUTURE/STUB] Coarse structural class of a blinded neuron. "
                "Not implemented; returns NOT_IMPLEMENTED."
            ),
            "inspect_neighbors": (
                "[FUTURE/STUB] Connectome neighbours of a blinded neuron. "
                "Not implemented; returns NOT_IMPLEMENTED."
            ),
            "run_silencing": (
                "[DISABLED] Virtual silencing via the finite-horizon "
                "first-passage structural transmission oracle. The engine "
                "is implemented but the formal horizon and baseline QC "
                "await human review (Phase 1C); returns "
                "DISABLED_PENDING_HUMAN_APPROVAL and never simulated data."
            ),
            "run_activation": (
                "[DISABLED] Virtual activation. Awaiting human-approved "
                "specification; returns DISABLED_PENDING_HUMAN_APPROVAL."
            ),
            "compare_matched_controls": (
                "[FUTURE/STUB] Matched-control comparison. Returns NOT_IMPLEMENTED."
            ),
            "submit_hypothesis": (
                "[FUTURE/STUB] Submit a hypothesis. Returns NOT_IMPLEMENTED."
            ),
            "submit_candidates": (
                "[FUTURE/STUB] Submit final blinded candidates. Returns NOT_IMPLEMENTED."
            ),
        }
        for name, description in future_descriptions.items():
            registry.register(
                ToolDefinition(
                    name=name,
                    description=description,
                    handler=self._future_stub,   # never returns data
                    implemented=False,
                    schema=getattr(
                        TOOL_SCHEMAS[name][0], "model_json_schema", lambda: None
                    )(),
                    unimplemented_status=TOOL_DISABLED_STATUS.get(
                        name, "NOT_IMPLEMENTED"
                    ),
                )
            )
        return registry

    # --- engineering handler methods ------------------------------------

    def _tool_protocol_status(self) -> dict[str, Any]:
        c = self.controller
        return {
            "mode": c.mode,
            "protocol_version": c.protocol_version,
            "dataset_version": c.dataset_version,
            "tool_allowlist": list(c.tool_allowlist),
            "remaining_query_budget": c.remaining_query_budget,
            "remaining_perturbation_budget": c.remaining_perturbation_budget,
            "phase": 0,
            "note": "Protocol is immutable at runtime; budgets only decrease.",
        }

    def _tool_dataset_version(self) -> dict[str, Any]:
        return {"dataset_version": self.controller.dataset_version}

    def _tool_remaining_budget(self) -> dict[str, Any]:
        c = self.controller
        return {
            "query": c.remaining_query_budget,
            "perturbation": c.remaining_perturbation_budget,
        }

    def _future_stub(self, **_: Any) -> dict[str, Any]:
        # Unreachable in practice: the controller short-circuits unimplemented
        # tools to NOT_IMPLEMENTED before invoking any handler. Kept as a
        # belt-and-braces guard that can never produce data.
        return {
            "status": "NOT_IMPLEMENTED",
            "message": "No result will ever be simulated for this tool.",
        }

    # ------------------------------------------------------------------
    # the single dispatch path used by both tests and the MCP adapter
    # ------------------------------------------------------------------

    def call_tool(
        self,
        role: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Validate arguments against the tool schema, then hand the call to
        the protocol controller (authorisation, budget, execution, logging)."""
        arguments = dict(arguments or {})

        schema_entry = TOOL_SCHEMAS.get(tool)
        if schema_entry is None:
            # Unknown tool: let the controller raise + log the violation.
            return self.controller.call_tool(role, tool, arguments)

        args_model: type[BaseModel] = schema_entry[0]
        try:
            args_model(**arguments)
        except ValidationError as exc:
            # Log the invalid attempt as a protocol violation, then raise a
            # schema error (denied calls never consume budget).
            self.controller._violation(
                "VIOLATION_SCHEMA_VALIDATION",
                {"role": role, "tool": tool, "errors": exc.errors(include_url=False)},
            )
            raise SchemaValidationError(
                f"Arguments for {tool!r} failed schema validation: "
                f"{exc.errors(include_url=False)}"
            ) from exc

        return self.controller.call_tool(role, tool, arguments)

    def close(self) -> None:
        self.controller.close()


# ---------------------------------------------------------------------------
# bootstrap helper
# ---------------------------------------------------------------------------

def build_core(
    data_root: str | None = None,
    mode: str | None = None,
    log_db: Path | None = None,
) -> ReproflyServerCore:
    """Convenience factory reading env vars, used by scripts and tests."""
    import os

    settings = load_settings(mode=mode, data_root=data_root)
    db = log_db or Path(
        os.environ.get("REPROFLY_LOG_DB", "logs/experiments.sqlite")
    )
    logger = ExperimentLogger(db)
    return ReproflyServerCore(settings=settings, logger=logger)


__all__ = ["ReproflyServerCore", "SchemaValidationError", "build_core"]
