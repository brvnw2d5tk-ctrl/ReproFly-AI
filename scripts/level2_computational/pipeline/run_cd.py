"""Phase 2C.1 — formal C/D runner (B-G6a execution entry; SPEC §11/§14).

Usage (ONLY after human-approved gate B-G6a, with a versioned GO record):
    python run_cd.py --real --go-record <path> --variant C2 --reps 0 1 2
    python run_cd.py --list-variants

Authorization semantics (HOTFIX 002, human 2026-09-26):
  Execution authorization binds to ONE specific versioned GO record
  (e.g. PHASE2C1_EXECUTION_GO_v1.3.json), not to the mere existence of
  historical files. The gate:
    A. loads the explicitly-provided GO record (--go-record);
    B. hashes it (SHA256);
    C. validates its content: HUMAN_GO TRUE, scope
       TRAJECTORY_GENERATION_ONLY, GT_SCORING_AUTHORIZED FALSE,
       AUTSRC_AUTHORIZED FALSE, git HEAD == current HEAD, SPEC hashes
       == frozen v1.0 values, pre-execution manifest hash == the
       record's named authoritative manifest (recomputed on disk),
       execution-surface (runner) hashes == the manifest's;
    D. scans supersession records: the CURRENT authorization is void
       ONLY if some supersession record explicitly references
       CURRENT_GO_RECORD_SHA256 (historical supersessions of old
       sentinels do not affect a new GO record);
    E. rejects on: no GO record, malformed record, any hash/HEAD/
       manifest mismatch, current record superseded, or multiple
       conflicting active GO records (more than one non-superseded
       record present in the phase2c1 root).

  The legacy sentinel file (PHASE2C1_EXECUTION_GO) is retained as
  history, permanently superseded by SUPERSESSION_001, and can never
  re-acquire authorization (OLD_SENTINEL_ACTIVE = FALSE).

Parallelism: each (variant, rep) is an independent single-threaded
process; scheduling order cannot affect any trajectory (SPEC §5).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_DIR))

from cd_policies import RUNNERS, VARIANT_IDS, make_rng  # noqa: E402
from cd_trajectories import TrajectoryEvents, write_trajectory  # noqa: E402

PHASE2C1_ROOT = CODE_DIR.parent
# repo root derived from the phase2c1 root's own depth:
# <repo>/derived_private/phase2c1 -> parents[0]=derived_private,
# parents[1]=<repo>. In test sandboxes PHASE2C1_ROOT is patched to
# <tmp>/phase2c1, so this constant must be derived lazily from the
# (possibly patched) PHASE2C1_ROOT inside authorize().
SPEC_DOCS_SUBPATH = ("docs", "benchmark")
LEGACY_SENTINEL = PHASE2C1_ROOT / "PHASE2C1_EXECUTION_GO"
SUPERSESSION_GLOB = "PHASE2C1_EXECUTION_GO_SUPERSESSION_*.json"
GO_RECORD_GLOB = "PHASE2C1_EXECUTION_GO_v*.json"


class AuthorizationError(RuntimeError):
    """Execution authorization failure (HOTFIX 002 gate)."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_git_head() -> str:
    import subprocess
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PHASE2C1_ROOT,
                       capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise AuthorizationError("cannot determine current git HEAD")
    return r.stdout.strip()


def authorize(go_record_path: Path) -> dict:
    """HOTFIX 002 authorization gate. Returns the validated record;
    raises AuthorizationError on ANY failure mode (human-approved specification §3.E)."""
    import spec_guard

    # ---- A. load the explicitly named GO record ----------------------
    if go_record_path is None:
        raise AuthorizationError(
            "no GO record provided — real execution requires "
            "--go-record <path> (human-approved gate B-G6a, HOTFIX 002 semantics)")
    p = Path(go_record_path)
    if not p.is_file():
        raise AuthorizationError(f"GO record not found: {p}")
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuthorizationError(f"malformed GO record: {exc}") from exc

    # ---- B. hash it ---------------------------------------------------
    go_sha = sha256_file(p)

    # ---- C. validate content -------------------------------------------
    required = {
        "authorization_version": str,
        "created_utc": str,
        "HUMAN_GO": bool,
        "scope": str,
        "GT_SCORING_AUTHORIZED": bool,
        "AUTSRC_AUTHORIZED": bool,
        "git_head": str,
        "spec_md_sha256": str,
        "spec_json_sha256": str,
        "pre_execution_manifest_sha256": str,
        "runner_hashes": dict,
    }
    for key, typ in required.items():
        if key not in rec or not isinstance(rec[key], typ):
            raise AuthorizationError(
                f"GO record missing/invalid field {key!r}")
    if rec["HUMAN_GO"] is not True:
        raise AuthorizationError("GO record HUMAN_GO != TRUE")
    if rec["scope"] != "TRAJECTORY_GENERATION_ONLY":
        raise AuthorizationError(
            f"GO record scope {rec['scope']!r} != TRAJECTORY_GENERATION_ONLY")
    if rec["GT_SCORING_AUTHORIZED"] is not False:
        raise AuthorizationError(
            "GT_SCORING_AUTHORIZED != FALSE — Stage-1 runner refuses "
            "scoring-scoped records")
    if rec["AUTSRC_AUTHORIZED"] is not False:
        raise AuthorizationError(
            "AUTSRC_AUTHORIZED != FALSE — Stage-1 runner refuses "
            "AUTSRC-scoped records")

    # git HEAD
    head = _current_git_head()
    if rec["git_head"] != head:
        raise AuthorizationError(
            f"GO record git HEAD {rec['git_head'][:12]}… != current "
            f"{head[:12]}… — record is stale; a new recorded human approval is required")

    # SPEC hashes: the GO record's declared SPEC hashes must match the
    # SPEC pins of the AUTHORITATIVE manifest the record binds to
    # (manifest v1.4+ pins SPEC v1.0.1 — the authoritative spec since
    # CORRECTION 003; earlier manifests pinned v1.0). The on-disk spec
    # files must also match those pins. (HOTFIX-002-gate adaptation for
    # CORRECTION 003: spec validation follows the manifest's
    # authoritative pins instead of hard-coding one spec version.)
    # SPEC validation is performed AFTER the manifest match below, so
    # the check binds to the manifest's authoritative spec pins rather
    # than a hard-coded spec version (HOTFIX-002-gate adaptation for
    # CORRECTION 003: manifest v1.4+ pins SPEC v1.0.1).

    # pre-execution manifest: the record names its authoritative
    # manifest by sha256; locate it among the versioned manifests and
    # verify the on-disk file matches
    manifest_files = sorted(
        PHASE2C1_ROOT.glob("PHASE2C1_PRE_EXECUTION_MANIFEST_v*.json"))
    match = [m for m in manifest_files
             if sha256_file(m) == rec["pre_execution_manifest_sha256"]]
    if len(match) != 1:
        raise AuthorizationError(
            "GO record pre-execution manifest hash does not match "
            "exactly one on-disk manifest — STOP")
    manifest = json.loads(match[0].read_text(encoding="utf-8"))

    # SPEC validation (performed after the manifest match): the record's
    # declared SPEC hashes must equal the manifest's authoritative spec
    # pins, and the on-disk spec files must match those pins. The spec
    # dir is resolved from the SANDBOX-AWARE repo root: in the real
    # layout PHASE2C1_ROOT=<repo>/derived_private/phase2c1 so the repo
    # root is parents[1]; in test sandboxes PHASE2C1_ROOT=<tmp>/phase2c1
    # so the root is parents[0]. Resolve by locating the docs/benchmark
    # directory that actually contains the pinned spec file.
    def _spec_dir() -> Path:
        for ancestor in PHASE2C1_ROOT.parents[:3]:
            cand = ancestor.joinpath(*SPEC_DOCS_SUBPATH)
            if cand.is_dir():
                return cand
        raise AuthorizationError(
            "cannot locate docs/benchmark spec directory — STOP")
    md_name = manifest["spec_pins"].get(
        "authoritative_for_execution_md",
        "PHASE2C1_MATCHED_LOCAL_BASELINES_SPEC_v1.0.md")
    md_path = _spec_dir() / md_name
    json_path = _spec_dir() / md_name.replace(".md", ".json")
    if rec["spec_md_sha256"] != manifest["spec_pins"][md_name]:
        raise AuthorizationError("GO record SPEC MD hash mismatch "
                                 "(vs authoritative manifest pin)")
    if rec["spec_json_sha256"] != manifest["spec_pins"][
            md_name.replace(".md", ".json")]:
        raise AuthorizationError("GO record SPEC JSON hash mismatch "
                                 "(vs authoritative manifest pin)")
    if sha256_file(md_path) != rec["spec_md_sha256"]:
        raise AuthorizationError("on-disk SPEC MD does not match the "
                                 "authoritative pin — STOP")
    if sha256_file(json_path) != rec["spec_json_sha256"]:
        raise AuthorizationError("on-disk SPEC JSON does not match the "
                                 "authoritative pin — STOP")

    # execution-surface hashes == the manifest's runner pins
    for runner_file, expected in manifest["runner_code"].items():
        actual = sha256_file(CODE_DIR / runner_file)
        if actual != expected:
            raise AuthorizationError(
                f"execution-surface hash mismatch for {runner_file}: "
                f"{actual} != {expected} — the authorized runner "
                "surface has changed; a new recorded human approval is required")

    # ---- D. supersession scan (SPECIFIC, not global) -------------------
    # the current authorization is void only if a supersession record
    # explicitly references CURRENT_GO_RECORD_SHA256
    for sup_path in sorted(PHASE2C1_ROOT.glob(SUPERSESSION_GLOB)):
        try:
            sup = json.loads(sup_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise AuthorizationError(
                f"malformed supersession record {sup_path.name} — STOP")
        referenced = sup.get("superseded_go_record_sha256") or \
            sup.get("superseded_sentinel_sha256") or ""
        if referenced == go_sha:
            raise AuthorizationError(
                f"GO record is SUPERSEDED (referenced by "
                f"{sup_path.name}) — new recorded human approval required")

    # ---- E. conflict detection: exactly one active GO record -----------
    active = []
    for cand in sorted(PHASE2C1_ROOT.glob(GO_RECORD_GLOB)):
        cand_sha = sha256_file(cand)
        superseded = False
        for sup_path in PHASE2C1_ROOT.glob(SUPERSESSION_GLOB):
            try:
                sup = json.loads(sup_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if sup.get("superseded_go_record_sha256") == cand_sha:
                superseded = True
                break
        if not superseded:
            active.append(cand)
    if len(active) != 1:
        raise AuthorizationError(
            f"{len(active)} non-superseded GO records present "
            f"({[a.name for a in active]}) — exactly one active "
            "authorization is required; conflicting/multiple active "
            "GO records are a STOP condition")

    return {"record": rec, "go_record_sha256": go_sha,
            "manifest_path": match[0].name}


def run_one(variant: str, rep: int, surface, out_path: Path | None):
    """Run one (variant, rep) against the given surface."""
    if variant not in RUNNERS:
        raise ValueError(f"unknown variant {variant}")
    events = TrajectoryEvents(variant)
    rng = make_rng(variant, rep)
    RUNNERS[variant](surface, rng, events)
    if out_path is not None:
        write_trajectory(events, out_path)
    return events


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=sorted(RUNNERS))
    ap.add_argument("--reps", type=int, nargs="*", default=None)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--go-record", type=Path, default=None,
                    help="path to the versioned GO record "
                         "(PHASE2C1_EXECUTION_GO_vX.Y.json) that "
                         "authorizes this execution")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--list-variants", action="store_true")
    args = ap.parse_args(argv)

    if args.list_variants:
        for name in sorted(RUNNERS, key=lambda v: VARIANT_IDS[v]):
            print(f"{VARIANT_IDS[name]} {name}")
        return 0

    if not args.variant:
        ap.error("--variant is required (or --list-variants)")

    if args.real:
        auth = authorize(args.go_record)   # HOTFIX 002 gate (raises)
        from cd_surfaces import RealSurface
        surface_factory = lambda: RealSurface()  # noqa: E731
        print(f"authorization OK: GO record "
              f"{auth['go_record_sha256'][:12]}… against manifest "
              f"{auth['manifest_path']}")
    else:
        # synthetic path is for tests only; refuse accidental misuse
        raise RuntimeError(
            "run_cd.py real execution is gated behind --real + a "
            "versioned GO record; synthetic runs go through the test "
            "suite (tests/test_cd_synthetic.py)")

    reps = args.reps if args.reps is not None else list(range(100))
    out_dir = args.out_dir or (PHASE2C1_ROOT / "trajectories")
    for rep in reps:
        surface = surface_factory()   # fresh surface state per rep
        out = out_dir / f"traj_{args.variant}_rep{rep:03d}.jsonl"
        events = run_one(args.variant, rep, surface, out)
        print(f"{args.variant} rep{rep:03d}: "
              f"{events.restarts} restarts, written {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
