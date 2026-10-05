"""ReproFly-AI exception hierarchy.

Every failure mode that matters for safety (private-path access, budget
abuse, unauthorised tools, large-file misuse) has a dedicated exception so
that guards, the protocol controller, and tests can react precisely.
"""

from __future__ import annotations


class ReproflyError(Exception):
    """Base class for all ReproFly-AI errors."""


class ConfigurationError(ReproflyError):
    """Configuration files / environment are invalid or incomplete."""


class DataRootNotSetError(ConfigurationError):
    """REPROFLY_DATA_ROOT is not set and no data root could be resolved."""


class SecurityError(ReproflyError):
    """Base class for security-relevant refusals."""


class PrivatePathError(SecurityError):
    """Attempt to read a PRIVATE ground-truth path or a path outside the data root."""


class LargeFileError(ReproflyError):
    """An eager in-memory read was requested for a file above the size limit."""


class ReadOnlyViolationError(SecurityError):
    """An operation attempted to modify the read-only data repository."""


# --- protocol / tools ---

class ProtocolError(ReproflyError):
    """Base class for protocol violations."""


class ProtocolViolationError(ProtocolError):
    """An agent attempted something the protocol forbids (e.g. raising a budget in PAPER mode)."""


class BudgetExceededError(ProtocolError):
    """A budgeted action was requested but the remaining budget is insufficient."""


class ToolDeniedError(ProtocolError):
    """The requested tool is not on the protocol allowlist for this run."""


class UnknownToolError(ProtocolError):
    """The requested tool does not exist in the registry."""


class RegistryFrozenError(ProtocolError):
    """Attempted to modify the tool registry after it was frozen (endpoint definitions are immutable at runtime)."""
