"""HMAC-SHA256 blind identifier mapping for cell types (Phase 1E §12).

The blinding secret is generated once, stored ONLY under
derived_private/blinding/ (gitignored), and never leaves the private
compartment. Blind IDs are assigned by ranking types in HMAC order and
numbering sequentially (BLIN-000001 ...), which is:

  * deterministic given (secret, real type set)
  * collision-free by construction
  * free of any biological ordering signal (HMAC order is pseudorandom
    with respect to type identity)

Scientists see ONLY blind IDs. The reverse mapping (blind -> real) lives
exclusively in the PRIVATE map file and the ground-truth scorer.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets as pysecrets
from pathlib import Path

BLIND_PREFIX = "BLIN-"
BLIND_ID_WIDTH = 6
SECRET_FILENAME = "blind_secret_v1.key"
MAP_FILENAME = "blind_type_map_private.json"


def generate_secret(path: Path) -> str:
    """Generate and persist a fresh 256-bit hex secret (PRIVATE)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = pysecrets.token_hex(32)
    path.write_text(secret, encoding="utf-8")
    return secret


def load_or_create_secret(private_dir: Path) -> str:
    secret_path = private_dir / SECRET_FILENAME
    if secret_path.is_file():
        return secret_path.read_text(encoding="utf-8").strip()
    return generate_secret(secret_path)


def _hmac_hex(secret: str, real_type: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), real_type.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def build_blind_map(secret: str, real_types) -> dict[str, str]:
    """Assign BLIN-000001.. sequentially in HMAC-sorted order.

    Returns {real_type: blind_id}. Deterministic; collision-free."""
    types = sorted(set(str(t) for t in real_types))
    ordered = sorted(types, key=lambda t: _hmac_hex(secret, t))
    return {
        real: f"{BLIND_PREFIX}{i:0{BLIND_ID_WIDTH}d}"
        for i, real in enumerate(ordered, start=1)
    }


def save_blind_map(private_dir: Path, blind_map: dict[str, str]) -> Path:
    private_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "private": True,
        "warning": (
            "PRIVATE blind mapping. Never expose to Scientist-facing "
            "interfaces or commit to Git."
        ),
        "scheme": (
            "BLIN-<6-digit sequential rank in HMAC-SHA256(secret, type) "
            "order>; collision-free; no biological ordering signal"
        ),
        "n_types": len(blind_map),
        "blind_to_real": {b: r for r, b in blind_map.items()},
    }
    out = private_dir / MAP_FILENAME
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_blind_map(private_dir: Path) -> dict[str, str]:
    payload = json.loads(
        (private_dir / MAP_FILENAME).read_text(encoding="utf-8")
    )
    return {real: blind for blind, real in payload["blind_to_real"].items()}


def is_blind_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(BLIND_PREFIX)
        and len(value) == len(BLIND_PREFIX) + BLIND_ID_WIDTH
        and value[len(BLIND_PREFIX):].isdigit()
    )


__all__ = [
    "generate_secret",
    "load_or_create_secret",
    "build_blind_map",
    "save_blind_map",
    "load_blind_map",
    "is_blind_id",
    "BLIND_PREFIX",
]
