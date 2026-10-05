"""Level 2 step 2: canonical graph + candidate universe + blinded layers.

Wraps the repository pipeline entry points (build_phase05, build_candidate_
universe, freeze_candidate_universe, build_gt_bridge, freeze_gt_v1_1).
Requires: a checkout of the ReproFly-AI code package at REPROFLY_REPO (or a
`solver_repo` sibling directory) and REPROFLY_DATA_ROOT. The ground-truth
bridge step consumes the human-held evidence file and is therefore run only
where that evidence exists; everything downstream of the frozen blinded sets
is verifiable without it.

Usage:  REPROFLY_REPO=/path/to/ReproFly-AI python 02_build_graph.py
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _level2_common import write_status  # noqa: E402

STATUS = os.path.join(HERE, "LEVEL2_STATUS.json")
STEPS = ["build_phase05.py", "build_candidate_universe.py",
         "freeze_candidate_universe.py", "build_gt_bridge.py", "freeze_gt_v1_1.py"]


def main():
    repo = os.environ.get("REPROFLY_REPO") or os.path.join(
        os.path.dirname(HERE), os.pardir, os.pardir, "ReproFly-AI")
    repo = os.path.abspath(repo)
    if not os.path.isdir(os.path.join(repo, "scripts")):
        write_status(STATUS, "02_build_graph", "SKIPPED_WITH_PREREQUISITES",
                     "requires REPROFLY_REPO pointing at the ReproFly-AI code "
                     "package (scripts/ not found) and REPROFLY_DATA_ROOT with "
                     "the public datasets")
        return 0
    if not os.environ.get("REPROFLY_DATA_ROOT"):
        write_status(STATUS, "02_build_graph", "SKIPPED_WITH_PREREQUISITES",
                     "set REPROFLY_DATA_ROOT to the raw-data tree")
        return 0
    for s in STEPS:
        r = subprocess.run([sys.executable, os.path.join(repo, "scripts", s)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "")[-400:]
            write_status(STATUS, f"02_build_graph:{s}", "FAIL", tail)
            return 0
    write_status(STATUS, "02_build_graph", "PASS",
                 "graph + candidate universe + blinded layers rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
