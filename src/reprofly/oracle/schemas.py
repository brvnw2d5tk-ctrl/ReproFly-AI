"""Oracle output schemas (pydantic) — auditability of every readout.

The future run_silencing contract (Phase 1B charter, section 21):

    baseline            B_c
    silenced            S_c
    absolute_change     S_c - B_c
    retention           S_c / B_c        (only if B_c > 0)
    relative_loss       1 - S_c / B_c    (only if B_c > 0)
    BASELINE_ZERO flag  if B_c == 0 (ratio fields are NA; no epsilon is
                       ever added to force a ratio)

Because silencing never redistributes signal, S_c <= B_c must hold within
numerical tolerance; a violation is a protocol failure (STOP condition H).

No composite reproductive score exists: every perturbation returns the
four INDEPENDENT channel readouts only (section 16).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class TransmissionResultModel(BaseModel):
    """One propagation run (baseline or silenced)."""

    model_config = {"extra": "forbid"}

    channel: str
    horizon: int = Field(ge=0)
    transmission: float = Field(ge=0.0, le=1.0 + 1e-9)
    initial_mass: float = Field(ge=0.0, le=1.0 + 1e-9)
    initial_silenced_source_mass: float = Field(ge=0.0, le=1.0 + 1e-9)
    max_balance_error: float = Field(ge=0.0)
    n_steps: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_initial(self) -> "TransmissionResultModel":
        if self.initial_mass + self.initial_silenced_source_mass > 1.0 + 1e-9:
            raise ValueError("initial mass accounting exceeds 1.0")
        return self


class SilencingReadoutModel(BaseModel):
    """One channel's baseline-vs-silenced comparison (section 21)."""

    model_config = {"extra": "forbid"}

    channel: str
    horizon: int = Field(ge=0)
    status: Literal["OK", "BASELINE_ZERO"]
    baseline: float = Field(ge=0.0, le=1.0 + 1e-9)
    silenced: float = Field(ge=0.0, le=1.0 + 1e-9)
    absolute_change: float
    retention: Optional[float] = None
    relative_loss: Optional[float] = None
    baseline_zero: bool

    @model_validator(mode="after")
    def _check_consistency(self) -> "SilencingReadoutModel":
        if self.baseline_zero:
            if self.retention is not None or self.relative_loss is not None:
                raise ValueError("BASELINE_ZERO readout must have NA ratios")
            if self.status != "BASELINE_ZERO":
                raise ValueError("baseline_zero=True requires status BASELINE_ZERO")
        else:
            if self.retention is None or self.relative_loss is None:
                raise ValueError("ratios required when baseline > 0")
        # silencing never redistributes mass: S <= B (+ numerical slack)
        if self.silenced > self.baseline + 1e-9:
            raise ValueError(
                f"silenced ({self.silenced}) > baseline ({self.baseline}): "
                "protocol violation (STOP condition H)"
            )
        return self


class OracleStatusModel(BaseModel):
    """Safe, anatomy-free oracle status for AI-facing surfaces."""

    model_config = {"extra": "forbid"}

    oracle_engine_built: bool
    horizon_locked: bool
    paper_run_silencing_enabled: bool
    phase: str


__all__ = [
    "TransmissionResultModel",
    "SilencingReadoutModel",
    "OracleStatusModel",
]
