"""Immutable protocol specification.

A ProtocolSpec is a frozen (hashable) value object loaded from config/ at
run start. Agents can never mutate it at runtime: there is no setter, the
dataclass is frozen, and the ProtocolController only exposes read access.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProtocolSpec:
    """Everything an experiment run is governed by. Immutable once loaded."""

    protocol_version: str
    dataset_version: str
    perturbation_budget: int
    query_budget: int
    tool_allowlist: tuple[str, ...] = field(default_factory=tuple)
    agent_roles: tuple[str, ...] = (
        "operator",   # human operator / harness (not an AI agent)
        "scientist",  # blinded Scientist agent (Phase >= 1)
        "engineer",   # engineering/maintenance role (Phase 0 only role besides operator)
    )

    def tool_allowed(self, tool_name: str) -> bool:
        return tool_name in self.tool_allowlist
