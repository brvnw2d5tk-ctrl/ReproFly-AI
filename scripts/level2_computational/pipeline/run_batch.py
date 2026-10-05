"""Phase 2C.1 Stage-1 batch orchestrator (B-G6a GO record v1.3).

Generates the full closed family: 7 variants x 100 reps = 700
trajectories. Each (variant, rep) runs in an isolated worker PROCESS
(single-threaded, own RealSurface/engine) so scheduling order cannot
affect any trajectory (SPEC §5). Worker count is chosen from machine
resources ONLY (memory ~1.2 GB/worker, CPU count), never from
trajectory content.

Usage:
    python run_batch.py --go-record <path> [--workers N] [--dry-plan]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PHASE2C1_ROOT = CODE_DIR.parent
REPO_ROOT = PHASE2C1_ROOT.parents[0]
TRAJ_DIR = PHASE2C1_ROOT / "trajectories"

VARIANTS = ["C1", "C2", "C3", "D1", "D2a", "D2b", "D3"]
REPS = list(range(100))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(task: tuple[str, int]) -> dict:
    """Run one (variant, rep) in this process (isolated state).

    BLAS/OMP threads are pinned to 1 INSIDE the worker before numpy
    loads: 12 multi-threaded BLAS runtimes oversubscribed the CPU in
    the first batch attempt (single-rep 140s -> ~700s); single-threaded
    workers also match SPEC §5's per-rep single-threaded semantics.
    """
    import os
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    variant, rep = task
    t0 = time.perf_counter()
    out = TRAJ_DIR / f"traj_{variant}_rep{rep:03d}.jsonl"
    if out.exists():
        return {"variant": variant, "rep": rep, "status": "ALREADY_DONE",
                "path": str(out), "runtime_s": 0.0}
    import sys as _sys
    _sys.path.insert(0, str(CODE_DIR))
    from cd_surfaces import RealSurface
    from cd_policies import RUNNERS, make_rng
    from cd_trajectories import TrajectoryEvents, write_trajectory

    events = TrajectoryEvents(variant)
    rng = make_rng(variant, rep)
    surface = RealSurface()
    RUNNERS[variant](surface, rng, events)
    write_trajectory(events, out)
    dt = time.perf_counter() - t0
    end = events.parsed()[-1]
    return {"variant": variant, "rep": rep, "status": "OK",
            "path": str(out), "runtime_s": round(dt, 1),
            "charged": end["charged_attempts"], "valid": end["valid"],
            "inspections": end["inspections_used"],
            "restarts": end["restarts"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go-record", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-plan", action="store_true")
    args = ap.parse_args(argv)

    # authorization gate (same as run_cd.py --real)
    sys.path.insert(0, str(CODE_DIR))
    import run_cd
    run_cd.authorize(args.go_record)

    tasks = [(v, r) for v in VARIANTS for r in REPS]
    # skip already-produced official trajectories (C1/rep0 from B-G5
    # and any prior completed file — never regenerated)
    todo = [t for t in tasks
            if not (TRAJ_DIR / f"traj_{t[0]}_rep{t[1]:03d}.jsonl").exists()]
    print(f"plan: {len(tasks)} total, {len(tasks)-len(todo)} already on "
          f"disk, {len(todo)} to run, workers={args.workers}")
    if args.dry_plan:
        return 0

    t0 = time.perf_counter()
    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(worker, t): t for t in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            v, r = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                if i % 25 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] {v}/rep{r:03d} "
                          f"{res['status']} {res['runtime_s']}s")
            except Exception as exc:  # infrastructure failure => STOP
                failures.append({"variant": v, "rep": r, "error": repr(exc)})
                print(f"  FAILURE {v}/rep{r:03d}: {exc!r}")
                # STOP batch immediately: do not silently continue
                for f in futs:
                    f.cancel()
                break
    dt = time.perf_counter() - t0
    print(f"batch done in {dt:.0f}s wall; {len(results)} ok, "
          f"{len(failures)} failures")

    # persist batch log (engineering only)
    log = {
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_s": round(dt, 1),
        "workers": args.workers,
        "results": sorted(results, key=lambda x: (x["variant"], x["rep"])),
        "failures": failures,
    }
    (PHASE2C1_ROOT / "PHASE2C1_BATCH_LOG_v1.3.json").write_text(
        json.dumps(log, indent=1), encoding="utf-8")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
