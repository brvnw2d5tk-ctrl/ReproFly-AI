"""Protocol Controller — the runtime governor for every experiment run.

Guarantees (tested in tests/test_protocol_budget.py):
  * run_id / experiment_id / budgets / allowlist / versions are fixed at
    run start and cannot be mutated by any agent.
  * In PAPER mode budgets can only DECREASE. There is no increase API at
    all; in DEV mode the operator may *reset* a budget, and every reset is
    written to the append-only log as a protocol event (BUDGET_RESET).
  * The protocol (ProtocolSpec) and endpoint (tool registry) definitions
    are frozen objects with no mutation path.
  * Every call — allowed or denied — is logged with role/tool/args/status/
    duration/protocol version. Violations additionally produce
    VIOLATION_* protocol events.
  * Budget accounting happens BEFORE the tool runs; a denied call never
    consumes budget but always consumes a log row.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from ..exceptions import (
    BudgetExceededError,
    ProtocolViolationError,
    RegistryFrozenError,
    ToolDeniedError,
    UnknownToolError,
)
from ..logging.logger import ExperimentLogger
from .spec import ProtocolSpec

# Tools whose invocation consumes query budget.
BUDGETED_TOOLS = frozenset(
    {
        "health_check",
        "get_protocol_status",
        "get_dataset_version",
        "get_remaining_budget",
        "get_oracle_status",
        "inspect_blind_type",
        "inspect_neighbors",
        "run_silencing",
        "run_activation",
        "compare_matched_controls",
        "submit_hypothesis",
        "submit_candidates",
    }
)

OPERATOR_ROLE = "operator"
VALID_MODES = ("DEV", "PAPER")


@dataclass(frozen=True)
class ToolDefinition:
    """Immutable endpoint definition. Registration happens once at startup;
    the registry is then frozen (RegistryFrozenError on any later change)."""

    name: str
    description: str
    handler: Callable[..., Any]
    implemented: bool = True          # False => never returns data
    schema: dict[str, Any] | None = None
    # Status marker returned for unimplemented tools. Default is
    # NOT_IMPLEMENTED; tools whose engine exists but is gated behind human
    # approval return DISABLED_PENDING_HUMAN_APPROVAL instead. Both markers
    # guarantee: no simulated, mocked, or fabricated results, ever.
    unimplemented_status: str = "NOT_IMPLEMENTED"


class ToolRegistry:
    """Frozen-after-start tool registry (the 'endpoint definitions')."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def register(self, definition: ToolDefinition) -> None:
        if self._frozen:
            raise RegistryFrozenError(
                f"Tool registry is frozen; cannot add {definition.name!r}"
            )
        if definition.name in self._tools:
            raise ValueError(f"Tool already registered: {definition.name}")
        self._tools[definition.name] = definition

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(f"Unknown tool: {name!r}") from None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def __contains__(self, name: str) -> bool:
        return name in self._tools


class ProtocolController:
    """One instance per run. Holds the frozen spec plus budget state that
    can only move downward, and mediates every tool call."""

    def __init__(
        self,
        spec: ProtocolSpec,
        registry: ToolRegistry,
        logger: ExperimentLogger,
        mode: str,
        experiment_id: str,
        run_id: str | None = None,
    ):
        if mode not in VALID_MODES:
            raise ProtocolViolationError(f"Invalid mode: {mode!r}")
        self._spec = spec                      # frozen dataclass — no setters
        self._registry = registry
        self._logger = logger
        self._mode = mode
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self.experiment_id = experiment_id
        self._remaining_query = spec.query_budget
        self._remaining_perturbation = spec.perturbation_budget
        self._closed = False
        self._logger.start_run(
            experiment_id=experiment_id,
            mode=mode,
            protocol_version=spec.protocol_version,
            dataset_version=spec.dataset_version,
            run_id=self.run_id,
        )
        self._logger.log_protocol_event(
            self.run_id,
            "RUN_STARTED",
            {
                "mode": mode,
                "query_budget": self._remaining_query,
                "perturbation_budget": self._remaining_perturbation,
                "tool_allowlist": list(spec.tool_allowlist),
            },
        )

    # ------------------------------------------------------------------
    # read-only views (agents can look, never touch)
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def spec(self) -> ProtocolSpec:
        return self._spec

    @property
    def remaining_query_budget(self) -> int:
        return self._remaining_query

    @property
    def remaining_perturbation_budget(self) -> int:
        return self._remaining_perturbation

    @property
    def tool_allowlist(self) -> tuple[str, ...]:
        return self._spec.tool_allowlist

    @property
    def dataset_version(self) -> str:
        return self._spec.dataset_version

    @property
    def protocol_version(self) -> str:
        return self._spec.protocol_version

    # ------------------------------------------------------------------
    # budget operations
    # ------------------------------------------------------------------

    def _decrease(self, what: str, amount: int) -> None:
        if amount < 0:
            raise ProtocolViolationError(
                f"Budget change amount must be non-negative, got {amount}"
            )
        if amount == 0:
            return
        if what == "query":
            if amount > self._remaining_query:
                raise BudgetExceededError(
                    f"Query budget exceeded: requested {amount}, "
                    f"remaining {self._remaining_query}"
                )
            self._remaining_query -= amount
            remaining = self._remaining_query
        elif what == "perturbation":
            if amount > self._remaining_perturbation:
                raise BudgetExceededError(
                    f"Perturbation budget exceeded: requested {amount}, "
                    f"remaining {self._remaining_perturbation}"
                )
            self._remaining_perturbation -= amount
            remaining = self._remaining_perturbation
        else:  # pragma: no cover - internal misuse guard
            raise ValueError(f"Unknown budget: {what}")
        self._logger.log_protocol_event(
            self.run_id,
            "BUDGET_DECREASED",
            {"budget": what, "amount": amount, "remaining": remaining},
        )

    def operator_reset_budget(self, role: str, what: str, new_value: int) -> None:
        """Operator-only budget reset. Forbidden entirely in PAPER mode.

        Agents can never call this successfully: the role must be the
        operator, and even in DEV mode the reset is permanently recorded.
        """
        if role != OPERATOR_ROLE:
            self._violation(
                "VIOLATION_BUDGET_MODIFICATION_BY_AGENT",
                {"role": role, "action": "operator_reset_budget", "budget": what},
            )
            raise ProtocolViolationError(
                "Only the operator may reset budgets; agents cannot modify budgets"
            )
        if self._mode == "PAPER":
            self._violation(
                "VIOLATION_BUDGET_RESET_IN_PAPER_MODE",
                {"role": role, "what": what, "new_value": new_value},
            )
            raise ProtocolViolationError(
                "PAPER mode: budgets can only decrease; reset is disabled"
            )
        # DEV-mode operator reset (exploration); logged indelibly.
        if what == "query":
            self._remaining_query = int(new_value)
        elif what == "perturbation":
            self._remaining_perturbation = int(new_value)
        else:  # pragma: no cover
            raise ValueError(f"Unknown budget: {what}")
        self._logger.log_protocol_event(
            self.run_id,
            "BUDGET_RESET",
            {"by": role, "budget": what, "new_value": new_value},
        )

    # ------------------------------------------------------------------
    # the single entry point for tool execution
    # ------------------------------------------------------------------

    def call_tool(
        self,
        role: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Mediate one tool call: authorise -> budget -> execute -> log."""
        arguments = dict(arguments or {})
        t0 = time.perf_counter()
        status: str

        # 1) existence
        try:
            definition = self._registry.get(tool)
        except UnknownToolError:
            self._log_call(role, tool, arguments, "ERROR", t0)
            self._violation("VIOLATION_UNKNOWN_TOOL", {"role": role, "tool": tool})
            raise

        # 2) allowlist
        if not self._spec.tool_allowed(tool):
            self._log_call(role, tool, arguments, "DENIED", t0)
            self._violation(
                "VIOLATION_TOOL_NOT_ALLOWLISTED",
                {"role": role, "tool": tool, "mode": self._mode},
            )
            raise ToolDeniedError(
                f"Tool {tool!r} is not allowed in this protocol run (mode={self._mode})"
            )

        # 3) unimplemented future tools: explicit contract, never fake data
        if not definition.implemented:
            status = definition.unimplemented_status
            self._log_call(role, tool, arguments, status, t0)
            return {
                "status": status,
                "tool": tool,
                "message": (
                    f"{tool} is not available: {status}. "
                    "No result is returned and no result will ever be "
                    "simulated."
                ),
            }

        # 4) budget (query-style tools cost 1 unit)
        if tool in BUDGETED_TOOLS:
            if self._remaining_query < 1:
                self._log_call(role, tool, arguments, "DENIED", t0)
                self._violation(
                    "VIOLATION_BUDGET_EXHAUSTED", {"role": role, "tool": tool}
                )
                raise BudgetExceededError("Query budget exhausted for this run")
            self._decrease("query", 1)

        # 5) execute
        try:
            result = definition.handler(**arguments)
        except Exception as exc:  # noqa: BLE001 - log everything, re-raise
            self._log_call(role, tool, arguments, "ERROR", t0)
            self._logger.log_error(
                source=f"ProtocolController.call_tool:{tool}",
                error_type=type(exc).__name__,
                message=str(exc),
                run_id=self.run_id,
            )
            raise
        self._log_call(role, tool, arguments, "OK", t0)
        return result

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _log_call(
        self, role: str, tool: str, arguments: dict[str, Any],
        status: str, t0: float,
    ) -> None:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        self._logger.log_tool_call(
            run_id=self.run_id,
            agent_role=role,
            tool=tool,
            arguments=arguments,
            result_status=status,
            duration_ms=duration_ms,
            protocol_version=self._spec.protocol_version,
        )

    def _violation(self, event_type: str, detail: dict[str, Any]) -> None:
        """Every violation leaves an indelible log row."""
        self._logger.log_protocol_event(self.run_id, event_type, detail)

    def close(self) -> None:
        if not self._closed:
            self._logger.finish_run(self.run_id)
            self._closed = True


__all__ = ["ProtocolController", "ToolRegistry", "ToolDefinition", "BUDGETED_TOOLS"]
