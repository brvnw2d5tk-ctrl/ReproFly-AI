"""Phase 2C.1 — trajectory event recording (SPEC v1.0 §15).

Event-order locks (all enforced by construction in cd_policies):
  * every STEP-1 draw emits ONE `restart` event immediately followed by
    ONE `seed` event (restart.blind_id = the newly drawn center);
  * restart reasons: initial_seed | record_minimum | percentile_gate |
    frontier_exhausted;
  * each `silence` event carries primary_relative_loss and the D-gate
    outcome (restart_flag; false for all C variants);
  * `inspect` events are emitted only for actually-performed
    inspections (first-OK; C/D never re-inspect by construction);
  * end.restarts = count of ALL restart events including initial_seed;
  * D3 emits its mapping inspect events before any restart/seed event.

Serialization (implementation lock §18.5):
  json.dumps(event, sort_keys=True, separators=(",", ":")) one per line
  — deterministic key order, no whitespace.
"""

from __future__ import annotations

import json
from typing import IO, Optional

RESTART_REASONS = ("initial_seed", "record_minimum",
                   "percentile_gate", "frontier_exhausted")


class TrajectoryEvents:
    """Collects and serializes one (variant, rep) trajectory."""

    def __init__(self, variant: str, stream: Optional[IO[str]] = None):
        self.variant = variant
        self.stream = stream
        self.lines: list[str] = []
        self._n_restart = 0
        self._n_inspect = 0
        self._n_silence = 0

    # -- writer ---------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True,
                          separators=(",", ":"))
        self.lines.append(line)
        if self.stream is not None:
            self.stream.write(line + "\n")

    # -- policy callbacks (cd_policies) --------------------------------------
    def restart(self, blind_id: str, reason: str) -> None:
        if reason not in RESTART_REASONS:
            raise ValueError(f"invalid restart reason {reason!r}")
        self._n_restart += 1
        self._emit({"event": "restart", "blind_id": blind_id,
                    "reason": reason})

    def seed(self, blind_id: str, pool_size: int) -> None:
        self._emit({"event": "seed", "blind_id": blind_id,
                    "pool_size": pool_size})

    def inspect(self, blind_id: str,
                inspections_used: int) -> None:
        self._n_inspect += 1
        self._emit({"event": "inspect", "blind_id": blind_id,
                    "inspections_used": inspections_used,
                    "status": "OK"})

    def silence(self, blind_id: str, charged_attempt: int,
                primary_relative_loss: float,
                restart_flag: bool) -> None:
        self._n_silence += 1
        self._emit({"event": "silence", "blind_id": blind_id,
                    "charged_attempt": charged_attempt,
                    "primary_relative_loss": primary_relative_loss,
                    "restart_flag": bool(restart_flag)})

    def end(self, st) -> None:
        self._emit({
            "event": "end",
            "charged_attempts": st.charged_attempts,
            "valid": len(st.valid_perturbed),
            "inspections_used": st.inspections_used,
            "restarts": self._n_restart,
        })

    # -- readback helpers (tests / manifests) ---------------------------------
    @property
    def restarts(self) -> int:
        return self._n_restart

    def parsed(self) -> list[dict]:
        return [json.loads(line) for line in self.lines]


def write_trajectory(events: TrajectoryEvents, out_path) -> None:
    """Write the JSONL file with LF newlines (deterministic bytes)."""
    from pathlib import Path
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(events.lines) + "\n",
                        encoding="utf-8", newline="\n")
