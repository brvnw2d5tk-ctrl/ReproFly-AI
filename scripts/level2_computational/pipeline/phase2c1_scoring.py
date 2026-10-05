"""PHASE 2C.1 OFFLINE GT SCORING (B-G6b GO, 2026-09-27).

Post-hoc matched-information control experiment: score the 700 frozen
Attempt-003 blind trajectories against the Phase 2A frozen v1 GT using
the v1.1 corrected scoring semantics (AUTSRC on the 300-charged-attempt
axis; only VALID silencings can hit).

Scorer-integrity gate (human-approved specification §5): before scoring any C/D trajectory,
re-score the 10 Phase 2A GLM formal runs with THIS wrapper and
reproduce the authoritative v1.1 values exactly (AUTSRC per run,
median, corrected PCD first hits).

C/D trajectories are all-VALID 300-length sequences; VE-AUTSRC must
equal AUTSRC (verified invariant).

Usage:
    python phase2c1_scoring.py            # gate + score all 700
    python phase2c1_scoring.py --gate-only
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT /
                       "derived_private/formal_execution/harness"))

from reprofly.benchmark.attempt_timeline import (  # noqa: E402
    rebuild_charged_attempts, sentinel_hit_sequence,
)
from reprofly.benchmark.scoring_v11 import (  # noqa: E402
    autsrc, crz_excluded_sensitivity, full_target_set_score,
    target_set_recall_curve,
)
from sealed_scoring import (  # noqa: E402
    ALL_RUNS, load_target_sets, run_submission,
)

RUNS_DIR = (PROJECT_ROOT / "derived_private/formal_execution"
            / "runs")


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8")
            .splitlines()]


def glm_v11_sequence(run_id: str) -> list[str]:
    """Authoritative v1.1 scoring sequence: conversation-based charged
    timeline + sentinel hit rule (only VALID silencings can hit)."""
    conv = _read_jsonl(RUNS_DIR / run_id / "conversation.jsonl")
    attempts = rebuild_charged_attempts(conv)
    assert len(attempts) == 300, run_id
    valid_ids = {a.attempted_blind_id for a in attempts
                 if a.status == "OK"}
    return sentinel_hit_sequence(attempts, valid_ids)

P2C1_ROOT = PROJECT_ROOT / "derived_private" / "phase2c1"
TRAJ_DIR = P2C1_ROOT / "trajectories"

# authoritative v1.1 expected values — loaded from the FROZEN corrected
# scoring artifact (comparison targets only, never scoring inputs)
V11_FROZEN = json.loads(
    (PROJECT_ROOT / "derived_private/formal_execution"
     / "SEALED_SCORING_RESULTS_v1.1_CORRECTED.json")
    .read_text(encoding="utf-8"))
EXPECTED_V11 = {
    "AUTSRC": {r: V11_FROZEN["runs"][r]["AUTSRC"] for r in V11_FROZEN["runs"]},
    "median": V11_FROZEN["glm_summary"]["median_AUTSRC"],
    "IQR": V11_FROZEN["glm_summary"]["iqr_AUTSRC"]
    if "iqr_AUTSRC" in V11_FROZEN["glm_summary"] else [0.0, 0.1222],
    "PCD_first_hits": {
        r: V11_FROZEN["runs"][r]["PerTargetSetDiscovery"]["GTSET_PCD"]
        for r in V11_FROZEN["runs"]
        if "GTSET_PCD" in V11_FROZEN["runs"][r]["PerTargetSetDiscovery"]},
}
TOL = 1e-9
VARIANTS = ["C1", "C2", "C3", "D1", "D2a", "D2b", "D3"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def gate() -> dict:
    """Scorer reproduction gate (FROZEN, human-approved abort GO §4):

    Path A (CORRECTED): re-score the 10 Phase 2A runs with the
    conversation-based charged-attempt timeline; reproduce the
    authoritative v1.1 FULL-PRECISION values (read from the frozen
    SEALED_SCORING_RESULTS_v1.1_CORRECTED.json — comparison targets
    only, never scoring inputs): AUTSRC all 10 runs, median, IQR,
    TargetSetRecall checkpoints, FirstGroundTruthHit, PCD first hits.

    Path B (OLD): the superseded v1.0 scorer path (tool_calls-only
    valid-silencing sequence) must reproduce the superseded v1.0
    values — proving the gate discriminates the two timelines.
    """
    target_sets = load_target_sets()
    relevant = sorted({b for ts in target_sets for b in ts.blind_ids})

    # ---- Path A: corrected conversation-based timeline --------------
    out = {}
    for run_id in ALL_RUNS:
        seq = glm_v11_sequence(run_id)
        submitted = run_submission(run_id)
        s = full_target_set_score(target_sets, seq, submitted, relevant)
        s["CRZ_EXCLUDED"] = crz_excluded_sensitivity(target_sets, seq)
        out[run_id] = s

    for run_id, exp in EXPECTED_V11["AUTSRC"].items():
        got = out[run_id]["AUTSRC"]
        assert abs(got - exp) <= TOL, f"{run_id} AUTSRC {got} != {exp}"
    vals = sorted(out[r]["AUTSRC"] for r in ALL_RUNS)
    n = len(vals)
    median = (vals[n // 2 - 1] + vals[n // 2]) / 2 if n % 2 == 0 \
        else vals[n // 2]
    assert abs(median - EXPECTED_V11["median"]) <= TOL, median
    iqr = [vals[2], vals[7]]  # 10 values: 25th/75th by numpy linear
    import numpy as np
    iqr = [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))]
    exp_iqr = EXPECTED_V11["IQR"]
    assert abs(iqr[0] - exp_iqr[0]) <= TOL and \
        abs(iqr[1] - exp_iqr[1]) <= TOL, iqr
    for run_id in ALL_RUNS:
        frozen = V11_FROZEN["runs"][run_id]
        got = out[run_id]
        for cp in ("TargetSetRecall@50", "TargetSetRecall@100",
                   "TargetSetRecall@200", "TargetSetRecall@300",
                   "FirstGroundTruthHit"):
            assert got[cp] == frozen[cp], f"{run_id} {cp} {got[cp]} != {frozen[cp]}"
        for k, v in frozen["PerTargetSetDiscovery"].items():
            assert got["PerTargetSetDiscovery"].get(k) == v, \
                f"{run_id} PTD {k}"
    for run_id, exp_h in EXPECTED_V11["PCD_first_hits"].items():
        got_h = out[run_id]["PerTargetSetDiscovery"].get("GTSET_PCD")
        assert got_h == exp_h, f"{run_id} PCD first hit {got_h} != {exp_h}"
    for run_id in ALL_RUNS:
        if run_id not in EXPECTED_V11["PCD_first_hits"]:
            assert "GTSET_PCD" not in out[run_id][
                "PerTargetSetDiscovery"], run_id

    # ---- Path B: OLD scorer path reproduces superseded v1.0 ---------
    v10 = json.loads(
        (PROJECT_ROOT / "derived_private/formal_execution"
         / "SEALED_SCORING_RESULTS.json").read_text(encoding="utf-8"))
    from sealed_scoring import valid_first_silencing_sequence
    for run_id in ALL_RUNS:
        seq_old = valid_first_silencing_sequence(run_id)
        s_old = full_target_set_score(
            target_sets, seq_old, run_submission(run_id), relevant)
        assert abs(s_old["AUTSRC"]
                   - v10["runs"][run_id]["AUTSRC"]) <= TOL, \
            f"OLD path {run_id} {s_old['AUTSRC']} != v1.0"

    return {"status": "PASS", "glm_rescored": {
        r: out[r]["AUTSRC"] for r in ALL_RUNS}, "median": median,
        "old_path_v1_0_replicated": True}


def score_trajectory(path: Path, target_sets, relevant) -> dict:
    """Score one C/D trajectory: all-VALID 300-length silence
    sequence on the charged-attempt axis."""
    events = [json.loads(l) for l in
              path.read_text(encoding="utf-8").splitlines()]
    silences = [e for e in events if e["event"] == "silence"]
    assert len(silences) == 300, path
    assert [e["charged_attempt"] for e in silences] == list(range(1, 301))
    seq = [e["blind_id"] for e in silences]
    # submission metrics are GLM-only (v1 convention); C/D get no
    # submission — pass empty
    s = full_target_set_score(target_sets, seq, [], relevant)
    s["CRZ_EXCLUDED"] = crz_excluded_sensitivity(target_sets, seq)
    # VE-AUTSRC: valid-experiment-indexed recall — for all-VALID
    # sequences it must equal AUTSRC exactly (invariant check)
    curve = target_set_recall_curve(target_sets, seq)
    ve = autsrc(curve)
    assert abs(ve - s["AUTSRC"]) <= TOL, f"VE-AUTSRC != AUTSRC: {path}"
    s["VE_AUTSRC"] = ve
    return s


def main() -> int:
    # ---- scorer-integrity gate ----------------------------------------
    g = gate()
    print("SCORER-INTEGRITY GATE: PASS "
          f"(10 GLM runs reproduced; median {g['median']:.4f})")

    target_sets = load_target_sets()
    relevant = sorted({b for ts in target_sets for b in ts.blind_ids})

    results = {"record_kind": "PHASE2C1_SCORING_RESULTS_v1.0",
               "classification": "POST-HOC MATCHED-INFORMATION "
                                 "CONTROL EXPERIMENT (not preregistered "
                                 "confirmatory)",
               "scoring_semantics": "Phase 2A scoring v1.1 corrected: "
                                    "AUTSRC on the 300-charged-attempt "
                                    "axis; only VALID silencings hit",
               "gate": g,
               "trajectories": {}}
    n = 0
    for variant in VARIANTS:
        for rep in range(100):
            p = TRAJ_DIR / f"traj_{variant}_rep{rep:03d}.jsonl"
            s = score_trajectory(p, target_sets, relevant)
            s["sha256"] = sha(p)
            results["trajectories"][f"{variant}/rep{rep:03d}"] = s
            n += 1
            if n % 100 == 0:
                print(f"  scored {n}/700")
    assert n == 700

    out = P2C1_ROOT / "PHASE2C1_SCORING_RESULTS_v1.0.json"
    assert not out.exists(), "refusing to overwrite"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8",
                   newline="\n")
    print(f"written {out}")
    print("PHASE2C1_SCORING_RESULTS_SHA256 =", sha(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
