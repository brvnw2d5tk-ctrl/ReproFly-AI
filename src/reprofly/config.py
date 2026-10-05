"""Configuration loading for ReproFly-AI.

Layering (later wins):
  1. config/default.toml          — machine-independent defaults
  2. config/{dev,paper}.toml      — mode overlay selected by REPROFLY_MODE
  3. environment variables        — only for secrets / machine-specific paths

Hard rules implemented here:
  * The data ROOT is never hardcoded: it comes from REPROFLY_DATA_ROOT.
  * Secrets are only ever read from the environment (never from TOML, so
    they cannot accidentally enter source control).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import ConfigurationError, DataRootNotSetError
from .protocol.spec import ProtocolSpec

VALID_MODES = ("DEV", "PAPER")
CONFIG_ENV_VAR = "REPROFLY_MODE"
DATA_ROOT_ENV_VAR = "REPROFLY_DATA_ROOT"

# Defence-in-depth duplicate of config default.toml [security].private_dirs.
# Kept in code so that even a misconfigured/deleted TOML cannot silently
# disable private-path blocking.
PRIVATE_DIRS_CODE_DEFAULT: tuple[str, ...] = (
    "05_BrainGonad_groundtruth",
    "13_Reproductive_Circuit_GroundTruth",
    "15_Project_Documentation/04_Ground_Truth_PRIVATE",
)


@dataclass(frozen=True)
class Settings:
    """Frozen, fully-resolved application settings."""

    mode: str
    data_root: Path
    private_dirs: tuple[str, ...]
    eager_read_max_bytes: int
    dataset_paths: dict[str, str] = field(default_factory=dict)
    protocol: ProtocolSpec | None = None

    # -- derived helpers -------------------------------------------------

    def resolve(self, rel_posix: str) -> Path:
        """Resolve a POSIX-style relative path against the data root, then
        verify it neither escapes the root nor enters a private dir."""
        from .security.guards import assert_path_public  # local import: avoid cycle

        return assert_path_public(self.data_root, rel_posix)


def _load_toml(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _find_config_dir() -> Path:
    """Locate the config/ directory relative to this file (installed or
    editable package). src/reprofly/config.py -> ../../.. = project root."""
    return Path(__file__).resolve().parents[2] / "config"


def load_settings(
    mode: str | None = None,
    data_root: str | None = None,
    config_dir: Path | None = None,
) -> Settings:
    """Load and validate settings.

    Parameters mirror the environment variables so tests can inject values
    without touching os.environ.
    """
    mode = (mode or os.environ.get(CONFIG_ENV_VAR, "DEV")).upper()
    if mode not in VALID_MODES:
        raise ConfigurationError(
            f"REPROFLY_MODE must be one of {VALID_MODES}, got {mode!r}"
        )

    cfg_dir = config_dir or _find_config_dir()
    merged: dict[str, Any] = _load_toml(cfg_dir / "default.toml")
    overlay_name = "dev.toml" if mode == "DEV" else "paper.toml"
    overlay_path = cfg_dir / overlay_name
    if overlay_path.exists():
        merged = _deep_merge(merged, _load_toml(overlay_path))

    root_raw = data_root or os.environ.get(DATA_ROOT_ENV_VAR, "")
    if not root_raw:
        raise DataRootNotSetError(
            f"{DATA_ROOT_ENV_VAR} is not set. Point it at the read-only "
            "Drosophila_AI_Repro repository (see .env.example)."
        )
    data_root = Path(root_raw).resolve()
    if not data_root.is_dir():
        raise ConfigurationError(f"REPROFLY_DATA_ROOT is not a directory: {data_root}")

    security = merged.get("security", {})
    private_dirs = tuple(security.get("private_dirs", ()))
    # Defence in depth: union with the in-code defaults.
    private_dirs = tuple(dict.fromkeys(PRIVATE_DIRS_CODE_DEFAULT + private_dirs))

    limits = merged.get("limits", {})
    eager_max = int(limits.get("eager_read_max_bytes", 268435456))

    datasets = dict(merged.get("datasets", {}))

    proto_cfg = merged.get("protocol", {})
    protocol = ProtocolSpec(
        protocol_version=str(proto_cfg.get("protocol_version", "unknown")),
        dataset_version=str(proto_cfg.get("dataset_version", "unknown")),
        perturbation_budget=int(proto_cfg.get("perturbation_budget", 0)),
        query_budget=int(proto_cfg.get("query_budget", 0)),
        tool_allowlist=tuple(proto_cfg.get("tool_allowlist", ())),
    )

    return Settings(
        mode=mode,
        data_root=data_root,
        private_dirs=private_dirs,
        eager_read_max_bytes=eager_max,
        dataset_paths=datasets,
        protocol=protocol,
    )


def get_secret(name: str) -> str | None:
    """Read a secret from the environment only. Returns None if unset/empty.

    Secrets are deliberately never part of Settings so they can never be
    serialised into logs or the protocol.
    """
    value = os.environ.get(name, "")
    return value if value else None
