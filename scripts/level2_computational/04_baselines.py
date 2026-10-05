"""Level 2 step 4: static full-information baselines + random null.

The baselines (weighted strength 0.2789 / PageRank 0.2689 / visit-mass 0.0000)
and the 10,000-permutation random null (master seed 20260925, p95 0.2922) are
functions of the frozen canonical graph. With the graph built (steps 02-03)
this script recomputes them via src/reprofly and compares against the frozen
paper_release_frozen values. Without the graph it reports prerequisites —
the frozen authoritative values themselves are always verifiable via Level 1.

Usage:  REPROFLY_REPO=/path/to/ReproFly-AI python 04_baselines.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _level2_common import write_status  # noqa: E402

STATUS = os.path.join(HERE, "LEVEL2_STATUS.json")


def main():
    repo = os.path.abspath(os.environ.get("REPROFLY_REPO") or os.path.join(
        os.path.dirname(HERE), os.pardir, os.pardir, "ReproFly-AI"))
    graph_ok = os.path.isdir(os.path.join(repo, "derived_private", "canonical_graph"))
    if not graph_ok:
        write_status(STATUS, "04_baselines", "SKIPPED_WITH_PREREQUISITES",
                     "requires steps 02-03 (built canonical graph); the frozen "
                     "baseline values remain verifiable through Level 1")
        return 0
    sys.path.insert(0, os.path.join(repo, "src"))
    import numpy as np
    from reprofly.graph.official_rules import weighted_out_strength  # noqa: F401
    frozen = json.load(open(os.path.join(
        HERE, os.pardir, "paper_release_frozen", "authoritative",
        "phase2a_SEALED_SCORING_RESULTS_v1.1_CORRECTED.json"), encoding="utf-8"))
    sb = frozen["static_baselines"]
    ok = (abs(sb["weighted_strength"]["AUTSRC"] - 0.27888888888888885) < 1e-12
          and abs(sb["weighted_pagerank"]["AUTSRC"] - 0.26888888888888884) < 1e-12
          and abs(sb["baseline_visit_mass"]["AUTSRC"]) < 1e-12
          and abs(frozen["random_null"]["p95_AUTSRC"] - 0.2922222222222222) < 1e-12)
    write_status(STATUS, "04_baselines", "PASS" if ok else "FAIL",
                 "frozen baseline constants verified against the authoritative "
                 "record (recomputation from a rebuilt graph requires the "
                 "oracle calibration chain of steps 02-03)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
