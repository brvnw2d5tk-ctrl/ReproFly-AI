"""Mass accounting data structures and validation.

Every propagation step must satisfy the mass-conservation identity

    active_mass_before ≈ sink_loss + silenced_loss + target_absorbed
                        + active_mass_after

within floating-point tolerance. No unexplained signal creation is
permitted (Phase 1B charter, section 20).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

DEFAULT_BALANCE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class StepAccount:
    """Per-step mass ledger for one propagation step t."""

    t: int
    active_before: float
    sink_loss: float
    silenced_loss: float
    target_absorbed: float
    active_after: float

    @property
    def balance_error(self) -> float:
        return self.active_before - (
            self.sink_loss + self.silenced_loss + self.target_absorbed
            + self.active_after
        )


@dataclass(frozen=True)
class PropagationResult:
    """Full result of one first-passage propagation run."""

    transmission: float                 # cumulative absorbed mass (<= 1)
    initial_mass: float                 # sum of x0 AFTER silencing init
    initial_silenced_source_mass: float # mass removed from x0 at init
    steps: tuple[StepAccount, ...] = field(default_factory=tuple)

    @property
    def max_balance_error(self) -> float:
        return max((abs(s.balance_error) for s in self.steps), default=0.0)

    @property
    def totals(self) -> dict[str, float]:
        return {
            "sink_loss": sum(s.sink_loss for s in self.steps),
            "silenced_loss": sum(s.silenced_loss for s in self.steps),
            "target_absorbed": sum(s.target_absorbed for s in self.steps),
        }


def assert_mass_balance(
    result: PropagationResult,
    tolerance: float = DEFAULT_BALANCE_TOLERANCE,
) -> None:
    """Raise OracleViolationError if any step violates conservation."""
    from .errors import OracleViolationError

    for s in result.steps:
        if abs(s.balance_error) > tolerance:
            raise OracleViolationError(
                f"mass balance violated at step {s.t}: "
                f"|error|={abs(s.balance_error):.3e} > {tolerance:.1e} "
                f"(before={s.active_before:.6f}, sink={s.sink_loss:.6f}, "
                f"silenced={s.silenced_loss:.6f}, absorbed={s.target_absorbed:.6f}, "
                f"after={s.active_after:.6f})"
            )


def balance_table(steps: Iterable[StepAccount]) -> list[dict[str, float]]:
    """JSON-safe ledger of all steps (for QC reports / MANIFEST)."""
    return [
        {
            "t": s.t,
            "active_before": s.active_before,
            "sink_loss": s.sink_loss,
            "silenced_loss": s.silenced_loss,
            "target_absorbed": s.target_absorbed,
            "active_after": s.active_after,
            "balance_error": s.balance_error,
        }
        for s in steps
    ]


__all__ = [
    "StepAccount",
    "PropagationResult",
    "assert_mass_balance",
    "balance_table",
    "DEFAULT_BALANCE_TOLERANCE",
]
