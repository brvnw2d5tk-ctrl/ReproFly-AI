"""Phase 2C.1 — spec guard (SPEC v1.0 §16 launch assertions).

The formal runner MUST assert the on-disk SPEC v1.0 files match the
locked hashes below before any real execution. Hashes come from
PHASE2C1_SPEC_LINEAGE.md (human-approved B-G0 lock, 2026-09-26).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_DIR = REPO_ROOT / "docs" / "benchmark"

SPEC_V10_MD = "PHASE2C1_MATCHED_LOCAL_BASELINES_SPEC_v1.0.md"
SPEC_V10_JSON = "PHASE2C1_MATCHED_LOCAL_BASELINES_SPEC_v1.0.json"

SPEC_V10_MD_SHA256 = (
    "ba178be550891e34cc599ce14894fcc3bd3e7fcc640e50e69e2c8e80605fb6c6")
SPEC_V10_JSON_SHA256 = (
    "54322bd644784bf0b2b8789fa73edbbc53a324ead97aaff5816ea7fe01ab3102")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_spec_v1_0() -> dict:
    """Assert both SPEC v1.0 files match the locked hashes.

    Returns a dict of {file: (ok, actual)} for reporting; raises
    RuntimeError on any mismatch (STOP semantics).
    """
    results = {}
    for name, expected in ((SPEC_V10_MD, SPEC_V10_MD_SHA256),
                           (SPEC_V10_JSON, SPEC_V10_JSON_SHA256)):
        path = SPEC_DIR / name
        actual = sha256_file(path)
        ok = actual == expected
        results[name] = (ok, actual)
        if not ok:
            raise RuntimeError(
                f"SPEC v1.0 hash mismatch for {name}: {actual} != "
                f"{expected} — STOP (SPEC §16 launch assertion)")
    return results


def spec_hashes() -> dict:
    """The locked hash constants (for embedding in manifests)."""
    return {
        "spec_md": SPEC_V10_MD_SHA256,
        "spec_json": SPEC_V10_JSON_SHA256,
    }
