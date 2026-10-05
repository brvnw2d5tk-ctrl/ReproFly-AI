"""Oracle exceptions — each maps to a Phase 1B STOP condition."""

from __future__ import annotations

from ..exceptions import ReproflyError


class OracleError(ReproflyError):
    """Base class for oracle failures."""


class DuplicatePairError(OracleError):
    """Duplicate directed (source, target) pair in the edge table.

    STOP condition C: the connectome-weights table is documented as an
    already-aggregated per-pair table; duplicates must never be silently
    re-aggregated."""


class InvalidWeightError(OracleError):
    """Non-positive / non-finite weight found (violates W_ij > 0)."""


class HorizonUndefinedError(OracleError):
    """A channel has zero reachable targets, so L95 is undefined.

    STOP condition E: endpoints must not be changed to fix this."""


class OracleViolationError(OracleError):
    """Numerical/protocol violation inside the oracle.

    Includes S_c > B_c beyond tolerance (STOP condition H) and mass
    balance breach (STOP condition F)."""


__all__ = [
    "OracleError",
    "DuplicatePairError",
    "InvalidWeightError",
    "HorizonUndefinedError",
    "OracleViolationError",
]
