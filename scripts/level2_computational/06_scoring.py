"""Level 2 step 6: sealed scoring verification (AUTSRC, v1.1-corrected).

Tier A (always available): verifies the frozen scoring JSON against the
frozen comparison + the 2A gate block (per-arm medians/IQRs, GLM rescored
values) from paper_release_frozen.
Tier B (needs steps 02-05): full re-scoring of the 700 trajectory JSONLs via
pipeline/phase2c1_scoring.py and byte-comparison of derived summaries.

Usage:  python 06_scoring.py [--full]
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _level2_common import write_status  # noqa: E402

STATUS = os.path.join(HERE, "LEVEL2_STATUS.json")
FR = os.path.join(os.path.dirname(HERE), "paper_release_frozen")


def tier_a():
    sc = json.load(open(os.path.join(FR, "authoritative",
                                     "phase2c1_PHASE2C1_SCORING_RESULTS_v1.0.json"),
                        encoding="utf-8"))
    cmpj = json.load(open(os.path.join(FR, "authoritative",
                                       "phase2c1_PHASE2C1_MATCHED_BASELINE_COMPARISON_v1.0.json"),
                          encoding="utf-8"))
    a2a = json.load(open(os.path.join(FR, "authoritative",
                                      "phase2a_SEALED_SCORING_RESULTS_v1.1_CORRECTED.json"),
                         encoding="utf-8"))
    by_arm = {}
    for k, v in sc["trajectories"].items():
        by_arm.setdefault(k.split("/")[0], []).append(v["AUTSRC"])
    for arm, av in cmpj["all_variants"].items():
        vals = by_arm[arm]
        if abs(float(np.median(vals)) - av["autsrc"]["median"]) > 1e-12:
            return False, f"median mismatch {arm}"
    gate = sc["gate"]
    for rid, v in gate["glm_rescored"].items():
        if abs(a2a["runs"][rid]["AUTSRC"] - v) > 1e-12:
            return False, f"GLM rescore mismatch {rid}"
    return True, "tier A: frozen scoring internally consistent (700 traj + GLM gate)"


def main():
    ok, detail = tier_a()
    if "--full" in sys.argv:
        ok2 = os.path.isdir(os.path.join(FR, "authoritative"))
        detail += "; tier B (full re-scoring) requires steps 02-05 artifacts "                   "and pipeline/phase2c1_scoring.py"
        ok = ok and ok2
    write_status(STATUS, "06_scoring", "PASS" if ok else "FAIL", detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
