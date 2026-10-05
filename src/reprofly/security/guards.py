"""Security guards: private-path blocking and read-only enforcement.

These functions are the single choke point every data access goes through.
They are pure (no I/O mutation), raise on violation, and are covered by
tests/test_private_path_blocking.py.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from urllib.parse import unquote

from ..exceptions import PrivatePathError

# Normalised (lowercase, forward-slash) private directory markers.
_DEFAULT_PRIVATE_DIRS: tuple[str, ...] = (
    "05_braingonad_groundtruth",
    "13_reproductive_circuit_groundtruth",
    "15_project_documentation/04_ground_truth_private",
)

# URL-encoded / mixed-separator variants are normalised away by
# _normalise(), so a single canonical list suffices.
_WINDOWS_DRIVE_RE = re.compile(r"^[a-z]:[\\/]?", re.IGNORECASE)


def _normalise(raw: str) -> str:
    """Normalise any user-supplied path string for containment checks:
    decode URL escapes, unify separators, strip drive letters and dots,
    lowercase. NOT used for actual filesystem access — only for vetting."""
    s = unquote(str(raw))
    s = s.replace("\\", "/")
    s = _WINDOWS_DRIVE_RE.sub("", s)
    parts = [p for p in PurePosixPath(s).parts if p not in (".", "")]
    # Drop '..' segments conservatively: any '..' is suspicious by itself,
    # but we keep it in place so that "a/../05_private" still contains the
    # private marker; the containment check below rejects escapes anyway.
    norm = "/".join(parts).lower()
    return norm


def _as_posix_rel(data_root: Path, candidate: Path) -> str:
    """Return the candidate's path relative to data_root as a normalised
    POSIX string. Raises PrivatePathError if the candidate is outside the
    root (after symlink resolution)."""
    root_resolved = data_root.resolve()
    cand_resolved = candidate.resolve()
    try:
        rel = cand_resolved.relative_to(root_resolved)
    except ValueError:
        raise PrivatePathError(
            f"Path escapes the data root: {candidate} "
            f"(root: {data_root}). Access denied."
        ) from None
    return "/".join(rel.parts).lower()


def _private_dirs_from(data_root: Path) -> tuple[str, ...]:
    """Read the configured private dirs from a marker file shipped with the
    data repository, falling back to the in-code defaults. The repo-level
    file wins because it is maintained by the data owners."""
    marker = data_root / "00_manifest" / "reprofly_private_dirs.txt"
    dirs: list[str] = []
    if marker.is_file():
        try:
            for line in marker.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    dirs.append(_normalise(line))
        except OSError:
            dirs = []
    if not dirs:
        dirs = list(_DEFAULT_PRIVATE_DIRS)
    return tuple(dirs)


def is_private_path(data_root: Path, rel_posix: str | Path) -> bool:
    """True if rel_posix (relative to data_root) points into a PRIVATE
    ground-truth directory or escapes the root entirely."""
    rel_norm = _normalise(str(rel_posix))
    for private in _private_dirs_from(data_root):
        if rel_norm == private or rel_norm.startswith(private + "/"):
            return True
    if rel_norm == ".." or rel_norm.startswith("../"):
        return True
    return False


def assert_path_public(data_root: Path, rel_posix: str | Path) -> Path:
    """Vet a relative path and return the absolute, resolved public path.

    Raises PrivatePathError on:
      * any private ground-truth directory
      * traversal outside the data root
      * absolute paths / Windows drive letters / URL-encoded traversal
    """
    raw = str(rel_posix)
    norm = _normalise(raw)

    if _WINDOWS_DRIVE_RE.match(raw) or raw.startswith("/") or raw.startswith("\\\\"):
        raise PrivatePathError(
            f"Absolute paths are not accepted; pass a path relative to the "
            f"data root. Got: {raw!r}"
        )
    if ".." in norm.split("/"):
        raise PrivatePathError(f"Parent traversal is not allowed: {raw!r}")

    if is_private_path(data_root, raw):
        raise PrivatePathError(
            f"Access to PRIVATE ground truth is denied: {raw!r}"
        )

    candidate = (data_root / raw).resolve()
    rel_after_resolution = _as_posix_rel(data_root, candidate)
    # Re-check after symlink resolution against the private list.
    for private in _private_dirs_from(data_root):
        if rel_after_resolution == private or rel_after_resolution.startswith(private + "/"):
            raise PrivatePathError(
                f"Resolved path points into PRIVATE ground truth: {raw!r} -> {candidate}"
            )
    return candidate


def assert_readonly_file_operation(path: Path, mode: str) -> None:
    """Guard for any file open() in the DAL: only read-like modes allowed
    against the data root."""
    read_like = {"r", "rb", "rt"} | {m for m in ("br", "tr")}
    if mode not in read_like:
        raise PrivatePathError(
            f"Write-mode open() against the data repository is denied: {path} ({mode!r})"
        )


__all__ = [
    "is_private_path",
    "assert_path_public",
    "assert_readonly_file_operation",
]
