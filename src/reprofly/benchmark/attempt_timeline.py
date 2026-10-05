"""Authoritative charged-attempt timeline reconstruction (scoring v1.1).

BACKGROUND (Phase 2A scoring correction v1.1, human-approved 2026-09-26):
The v1.0 sealed scorer rebuilt each run's 300-charged-attempt timeline
from ``tool_calls.jsonl`` alone. ``FormalRunHarness.dispatch()`` returns
INVALID_PRECOMMITMENT BEFORE the tool-call logging wrapper, so those
attempts (charged by the BudgetLedger but never written to
tool_calls.jsonl) existed only in ``conversation.jsonl``. The scorer's
timeline was therefore silently compressed for any run with >=1 such
attempt (43 attempts across 8 of 10 formal runs), shifting later
attempts — including ground-truth hits — earlier.

LOCKED SEMANTICS (unchanged by this correction — only the
reconstruction bug is fixed):
  * the AUTSRC x-axis is the 300 CHARGED perturbation attempts
    (VALID + DUPLICATE_PERTURBATION + INVALID_TARGET +
    INVALID_PRECOMMITMENT; BENCHMARK_PROTOCOL_v1.0 §5, budgets.py);
  * a HIT requires a VALID silencing of a ground-truth member
    (metrics.py: "Duplicate/invalid attempts are not hits — they are
    attempts"); INVALID_PRECOMMITMENT occupies a timeline slot but can
    NEVER constitute a GT hit.

This module is the single authoritative reconstruction for any future
scoring. It rebuilds the timeline from ``conversation.jsonl`` (every
charged run_silencing tool-result in order) and cross-validates against
``tool_calls.jsonl`` / ``precommitments.jsonl`` / ``perturbation_log.jsonl``
/ ``run_meta.json``. All functions are pure and synthetic-testable: no
private assets are read here (see
tests/test_scoring_timeline_correction.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

# Every run_silencing outcome that is charged to the perturbation budget
# and therefore occupies exactly one attempt slot on the scoring
# timeline (budgets.py; BENCHMARK_PROTOCOL_v1.0 §5 "Every attempt
# counts").
CHARGED_RUN_SILENCING_STATUSES: tuple[str, ...] = (
    "OK",
    "DUPLICATE_PERTURBATION",
    "INVALID_TARGET",
    "INVALID_PRECOMMITMENT",
)

# Compact status names used in corrected artefacts.
STATUS_ABBR: dict[str, str] = {
    "OK": "VALID",
    "DUPLICATE_PERTURBATION": "DUPLICATE",
    "INVALID_TARGET": "INVALID_TARGET",
    "INVALID_PRECOMMITMENT": "INVALID_PRECOMMITMENT",
}


class TimelineIntegrityError(RuntimeError):
    """Any violation of the authoritative-timeline invariants: STOP."""


@dataclass(frozen=True)
class ChargedAttempt:
    """One charged perturbation attempt on the scoring timeline."""

    index: int                      # 1..300 (charge order)
    status: str                     # raw status (see CHARGED_…_STATUSES)
    attempted_blind_id: str         # extracted target (may be UNKNOWN)
    conversation_line: int          # 1-based line in conversation.jsonl
    call_id: Optional[str]          # provider tool-call id

    @property
    def status_abbr(self) -> str:
        return STATUS_ABBR[self.status]


def parse_tool_call_arguments(raw: str | None) -> dict[str, Any]:
    """Parse a tool-call arguments JSON string the same way the run-time
    client does (llm_client.chat): unparseable payloads are preserved
    under ``_invalid_json`` rather than dropped."""
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_invalid_json": raw}
    return args if isinstance(args, dict) else {"_invalid_json": raw}


def extract_args_from_assistant(msg: dict[str, Any]) -> dict[str, dict]:
    """Map call_id -> parsed arguments for one assistant message."""
    out: dict[str, dict] = {}
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        out[tc.get("id")] = parse_tool_call_arguments(
            fn.get("arguments"))
    return out


def attempted_blind_id(args: dict[str, Any] | None) -> str:
    """The attempted target of a run_silencing call, mirroring the
    extraction the v1.0 scorer used (precommitment.target_blind_id,
    falling back to blind_id, else UNKNOWN)."""
    if not isinstance(args, dict):
        return "UNKNOWN"
    pc = args.get("precommitment", {})
    if isinstance(pc, dict) and pc.get("target_blind_id"):
        return pc["target_blind_id"]
    if args.get("blind_id"):
        return args["blind_id"]
    return "UNKNOWN"


def rebuild_charged_attempts(
    conversation_records: Iterable[dict[str, Any]],
) -> list[ChargedAttempt]:
    """Rebuild the charged-attempt timeline in charge order from
    conversation records (parsed conversation.jsonl lines: ``dir`` in
    {assistant, tool-result, user-nudge, …}, ``msg`` payload).

    Every run_silencing tool-result whose status is charged appends
    exactly one attempt. Non-charged statuses (BUDGET_EXHAUSTED etc.)
    are ignored — they consume no budget.
    """
    attempts: list[ChargedAttempt] = []
    args_by_call: dict[str, dict] = {}
    for line_idx, rec in enumerate(conversation_records, start=1):
        direction = rec.get("dir")
        if direction == "assistant":
            args_by_call.update(extract_args_from_assistant(
                rec.get("msg", {})))
            continue
        if direction != "tool-result":
            continue
        msg = rec.get("msg", {})
        if msg.get("tool") != "run_silencing":
            continue
        status = msg.get("result", {}).get("status", "?")
        if status not in CHARGED_RUN_SILENCING_STATUSES:
            continue
        call_id = msg.get("call_id")
        attempts.append(ChargedAttempt(
            index=len(attempts) + 1,
            status=status,
            attempted_blind_id=attempted_blind_id(
                args_by_call.get(call_id)),
            conversation_line=line_idx,
            call_id=call_id,
        ))
    return attempts


def timeline_composition(attempts: Sequence[ChargedAttempt]) -> dict[str, int]:
    """Counts per compact status name (VALID / DUPLICATE /
    INVALID_TARGET / INVALID_PRECOMMITMENT)."""
    comp: dict[str, int] = {v: 0 for v in STATUS_ABBR.values()}
    for a in attempts:
        comp[a.status_abbr] += 1
    return comp


def assert_valid_timeline(
    attempts: Sequence[ChargedAttempt],
    *,
    expected_len: int = 300,
    run_id: str = "",
) -> dict[str, int]:
    """Assert the authoritative-timeline invariants; raise
    TimelineIntegrityError (STOP) on any violation. Returns the
    composition."""
    where = f"run {run_id}: " if run_id else ""
    if len(attempts) != expected_len:
        raise TimelineIntegrityError(
            f"{where}charged-attempt timeline has {len(attempts)} "
            f"entries, expected exactly {expected_len}")
    for i, a in enumerate(attempts, start=1):
        if a.index != i:
            raise TimelineIntegrityError(
                f"{where}non-contiguous attempt indices at {i}")
    comp = timeline_composition(attempts)
    if sum(comp.values()) != len(attempts):
        raise TimelineIntegrityError(
            f"{where}composition does not sum to timeline length")
    return comp


def sentinel_hit_sequence(
    attempts: Sequence[ChargedAttempt],
    valid_blind_ids: Iterable[str],
) -> list[str]:
    """Scoring sequence under the locked hit rule: ONLY a VALID
    silencing of a target can be a ground-truth hit.

      * VALID attempts keep their raw blind id (a valid silencing
        executed, so the id is in valid_blind_ids by construction);
      * DUPLICATE attempts keep the raw id only if it was validly
        silenced (ledger semantics: is_tested implies an earlier VALID
        attempt of the same id — the raw id can never create an EARLIER
        hit than that VALID attempt);
      * INVALID_TARGET and INVALID_PRECOMMITMENT attempts NEVER keep a
        raw id: no silencing executed, so they can never be hits even
        if the extracted target id happens to be a GT member that is
        (later) validly silenced.

    Non-hit ids are mapped to a ``~invalid:`` sentinel that matches no
    GT member, mirroring the frozen v1.0 sentinel convention.
    """
    valid = set(valid_blind_ids)
    seq: list[str] = []
    first_valid_pos: dict[str, int] = {}
    for a in attempts:
        if a.status == "OK":
            first_valid_pos.setdefault(a.attempted_blind_id, a.index)
    for a in attempts:
        if a.status == "OK":
            seq.append(a.attempted_blind_id)
        elif (a.status == "DUPLICATE_PERTURBATION"
              and a.attempted_blind_id in valid):
            # defensive invariant: the raw id must have been validly
            # silenced EARLIER (otherwise a duplicate position could
            # masquerade as an earlier hit)
            if first_valid_pos.get(a.attempted_blind_id, 10**9) > a.index:
                raise TimelineIntegrityError(
                    f"DUPLICATE attempt {a.index} of "
                    f"{a.attempted_blind_id} precedes its first VALID "
                    f"silencing — ledger invariant violated")
            seq.append(a.attempted_blind_id)
        else:
            seq.append(f"~invalid:{a.attempted_blind_id}")
    return seq


def cross_validate_timeline(
    attempts: Sequence[ChargedAttempt],
    *,
    tool_calls_records: Sequence[dict[str, Any]],
    perturbation_records: Sequence[dict[str, Any]],
    precommitment_records: Sequence[dict[str, Any]],
    run_meta: dict[str, Any],
) -> dict[str, Any]:
    """Cross-validate the conversation-rebuilt timeline against every
    other run ledger. Returns a check dict (all values True on success);
    any False is an integrity STOP for the caller."""
    checks: dict[str, Any] = {}

    # 1) tool_calls.jsonl must be the exact timeline subsequence with
    #    INVALID_PRECOMMITMENT entries removed (they were charged but
    #    not logged there — the v1.0 defect).
    tc_rs = [c for c in tool_calls_records
             if c.get("tool") == "run_silencing"]
    non_invpre = [a for a in attempts
                  if a.status != "INVALID_PRECOMMITMENT"]
    ok = len(tc_rs) == len(non_invpre)
    if ok:
        for c, a in zip(tc_rs, non_invpre):
            tc_bid = ((c.get("args", {}).get("precommitment", {})
                       .get("target_blind_id"))
                      or c.get("args", {}).get("blind_id", "UNKNOWN"))
            if c.get("status") != a.status or tc_bid != a.attempted_blind_id:
                ok = False
                break
    checks["tool_calls_exact_subsequence_without_invpre"] = ok
    checks["n_tool_calls_run_silencing"] = len(tc_rs)
    checks["n_timeline_minus_invpre"] = len(non_invpre)

    # 2) VALID attempts must match perturbation_log and precommitments
    #    order, ids and experiment pairing.
    valid_attempts = [a for a in attempts if a.status == "OK"]
    ok_p = (len(perturbation_records) == len(valid_attempts)
            == len(precommitment_records))
    if ok_p:
        for a, p, pc in zip(valid_attempts, perturbation_records,
                            precommitment_records):
            if (a.attempted_blind_id != p.get("target_blind_id")
                    or a.attempted_blind_id != pc.get("target_blind_id")
                    or p.get("experiment_id") != pc.get("experiment_id")):
                ok_p = False
                break
    checks["valid_order_matches_perturbation_and_precommitments"] = ok_p
    checks["n_valid"] = len(valid_attempts)

    # 3) run_meta budget agreement (the run-time ledger counted every
    #    charged attempt — including INVALID_PRECOMMITMENT).
    bf = run_meta.get("budget_final", {})
    checks["run_meta_perturbation_used_eq_timeline_len"] = (
        bf.get("perturbation_used") == len(attempts))
    checks["run_meta_valid_experiments_eq_n_valid"] = (
        bf.get("valid_experiments") == len(valid_attempts))
    return checks


__all__ = [
    "CHARGED_RUN_SILENCING_STATUSES",
    "STATUS_ABBR",
    "ChargedAttempt",
    "TimelineIntegrityError",
    "parse_tool_call_arguments",
    "extract_args_from_assistant",
    "attempted_blind_id",
    "rebuild_charged_attempts",
    "timeline_composition",
    "assert_valid_timeline",
    "sentinel_hit_sequence",
    "cross_validate_timeline",
]
