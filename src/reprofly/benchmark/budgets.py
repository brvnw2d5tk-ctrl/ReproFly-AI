"""Formal benchmark budget policy (Phase 1E, LOCKED constants)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---- locked benchmark constants (charter §4-6, §22-23, §31) -------------
N_AI_RUNS = 10                     # independent fresh-context GLM-5.3 runs
PERTURBATION_BUDGET = 300          # per run: valid + duplicate + invalid
STRUCTURAL_INSPECTION_BUDGET = 600  # per run: successful + repeated inspects
CHECKPOINTS = (50, 100, 200, 300)
TOP_N_FINAL = 50                   # max final submitted candidates
RANDOM_PERMUTATIONS = 10_000       # random baseline null distribution
RANDOM_MASTER_SEED = 20260925      # locked master seed
TECHNICAL_RETRIES_MAX = 3          # per provider/API request
SCIENTIST_MODEL = "GLM-5.3"

DUPLICATE_PERTURBATION = "DUPLICATE_PERTURBATION"   # costs 1, not re-executed
INVALID_TARGET = "INVALID_TARGET"                   # costs 1
INVALID_PRECOMMITMENT = "INVALID_PRECOMMITMENT"     # costs 1 (charter §15)
ALREADY_INSPECTED = "ALREADY_INSPECTED"             # costs 1, no new info
TECHNICAL_FAILURE = "TECHNICAL_FAILURE"
PROTOCOL_SAFETY_TERMINATION = "PROTOCOL_SAFETY_TERMINATION"

# Cost-0 technical failures (charter §16): must be recorded in the
# machine error log; the Scientist can never self-declare these.
TRANSPORT_FAILURE = "TRANSPORT_FAILURE"             # provider/API, pre-execution
ORACLE_TECHNICAL_FAILURE = "ORACLE_TECHNICAL_FAILURE"  # MCP/oracle, pre-result


class BudgetExhaustedError(RuntimeError):
    """A budgeted action was requested with zero remaining budget."""


@dataclass
class BudgetLedger:
    """Per-run budget accounting (append-style; deterministic).

    Perturbation budget counts EVERY attempt (valid, duplicate, invalid)
    so experimental inefficiency is real (charter §19-22). Inspection
    budget counts every inspect call including repeats (charter §6)."""

    run_id: str
    perturbation_budget: int = PERTURBATION_BUDGET
    inspection_budget: int = STRUCTURAL_INSPECTION_BUDGET
    _perturbation_used: int = 0
    _inspection_used: int = 0
    _tested_blind_ids: set[str] = field(default_factory=set)
    _inspected_blind_ids: set[str] = field(default_factory=set)
    events: list[dict[str, Any]] = field(default_factory=list)

    # -- perturbations ------------------------------------------------------

    @property
    def perturbation_remaining(self) -> int:
        return self.perturbation_budget - self._perturbation_used

    @property
    def perturbation_exhausted(self) -> bool:
        return self.perturbation_remaining <= 0

    def charge_perturbation(
        self, blind_id: str, status: str, detail: str = ""
    ) -> None:
        if self.perturbation_exhausted:
            raise BudgetExhaustedError(
                f"run {self.run_id}: perturbation budget exhausted "
                f"({self.perturbation_budget})"
            )
        self._perturbation_used += 1
        self.events.append({
            "kind": "perturbation", "blind_id": blind_id, "status": status,
            "used": self._perturbation_used, "detail": detail,
        })

    def register_valid_perturbation(self, blind_id: str) -> None:
        if blind_id in self._tested_blind_ids:
            raise ValueError(
                "internal error: valid perturbation of already-tested "
                "target must be intercepted as DUPLICATE_PERTURBATION"
            )
        self._tested_blind_ids.add(blind_id)

    def is_tested(self, blind_id: str) -> bool:
        return blind_id in self._tested_blind_ids

    # -- inspections --------------------------------------------------------

    @property
    def inspection_remaining(self) -> int:
        return self.inspection_budget - self._inspection_used

    @property
    def inspection_exhausted(self) -> bool:
        return self.inspection_remaining <= 0

    def charge_inspection(self, blind_id: str, status: str) -> None:
        if self.inspection_exhausted:
            raise BudgetExhaustedError(
                f"run {self.run_id}: structural inspection budget "
                f"exhausted ({self.inspection_budget})"
            )
        self._inspection_used += 1
        self.events.append({
            "kind": "inspection", "blind_id": blind_id, "status": status,
            "used": self._inspection_used,
        })

    def is_inspected(self, blind_id: str) -> bool:
        return blind_id in self._inspected_blind_ids

    def register_inspection(self, blind_id: str) -> None:
        self._inspected_blind_ids.add(blind_id)

    # -- technical failures (cost 0, machine-logged; §16) ---------------

    def log_technical_failure(
        self, blind_id: str, kind: str, detail: str = ""
    ) -> None:
        """Record a cost-0 technical failure. ONLY TRANSPORT_FAILURE and
        ORACLE_TECHNICAL_FAILURE are valid kinds; both require a machine
        error log entry (this method IS that log). The Scientist can
        never self-declare a technical failure — only the harness may
        call this."""
        if kind not in (TRANSPORT_FAILURE, ORACLE_TECHNICAL_FAILURE):
            raise ValueError(
                f"invalid technical-failure kind {kind!r}; only "
                f"{TRANSPORT_FAILURE}/{ORACLE_TECHNICAL_FAILURE} are "
                "cost-0, and the Scientist cannot self-declare them"
            )
        self.events.append({
            "kind": "technical_failure", "blind_id": blind_id,
            "status": kind, "used_perturbation": self._perturbation_used,
            "detail": detail,
        })

    # -- summary -------------------------------------------------------------

    def checkpoint_status(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "perturbation_used": self._perturbation_used,
            "perturbation_remaining": self.perturbation_remaining,
            "inspection_used": self._inspection_used,
            "inspection_remaining": self.inspection_remaining,
            "checkpoints": {
                str(c): self._perturbation_used >= c for c in CHECKPOINTS
            },
            "run_complete": self.perturbation_exhausted,
        }


__all__ = [
    "N_AI_RUNS", "PERTURBATION_BUDGET", "STRUCTURAL_INSPECTION_BUDGET",
    "CHECKPOINTS", "TOP_N_FINAL", "RANDOM_PERMUTATIONS",
    "RANDOM_MASTER_SEED", "TECHNICAL_RETRIES_MAX", "SCIENTIST_MODEL",
    "DUPLICATE_PERTURBATION", "INVALID_TARGET", "INVALID_PRECOMMITMENT",
    "ALREADY_INSPECTED", "TECHNICAL_FAILURE",
    "TRANSPORT_FAILURE", "ORACLE_TECHNICAL_FAILURE",
    "PROTOCOL_SAFETY_TERMINATION",
    "BudgetLedger", "BudgetExhaustedError",
]
