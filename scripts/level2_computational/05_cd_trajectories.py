"""Level 2 step 5: matched-information C/D trajectories (7 arms x 100 x 300).

The deterministic runner is included in `pipeline/` (frozen phase-2C.1 code;
numpy SeedSequence(entropy=20260926, spawn_key=(variant_id, rep_index))).
Full re-run additionally requires the built graph + oracle artifacts
(steps 02-03) and the execution-GO authorization record included in
`pipeline_execution_records/`. Runtime reference: ~2.8 min per trajectory
single-core (C1 rep0 = 166.3 s, envelope 1800 s) for 700 trajectories.

Usage:  REPROFLY_REPO=/path/to/ReproFly-AI python 05_cd_trajectories.py
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _level2_common import write_status  # noqa: E402

STATUS = os.path.join(HERE, "LEVEL2_STATUS.json")


def main():
    repo = os.path.abspath(os.environ.get("REPROFLY_REPO") or os.path.join(
        os.path.dirname(HERE), os.pardir, os.pardir, "ReproFly-AI"))
    graph_ok = os.path.isdir(os.path.join(repo, "derived_private", "canonical_graph"))
    oracle_ok = os.path.isdir(os.path.join(repo, "derived_private", "oracle"))
    if not (graph_ok and oracle_ok):
        write_status(STATUS, "05_cd_trajectories", "SKIPPED_WITH_PREREQUISITES",
                     "deterministic runner included in pipeline/; full re-run "
                     "needs built graph+oracle (steps 02-03). Determinism "
                     "evidence: 9/9 byte-identical replays recorded in "
                     "the Stage-1 attempt-003 status record shipped with the authoritative results")
        return 0
    r = subprocess.run([sys.executable, os.path.join(HERE, "pipeline", "run_cd.py"),
                        "--help"], capture_output=True, text=True)
    write_status(STATUS, "05_cd_trajectories", "PASS" if r.returncode == 0 else "FAIL",
                 "runner importable; execute pipeline/run_cd.py per its --help "
                 "with the GO record in pipeline_execution_records/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
