"""Level 2 step 3: oracle + endpoint proxies (wraps build_oracle +
build_phase1a + build_phase1c). Same prerequisites as step 2.

Usage:  REPROFLY_REPO=/path/to/ReproFly-AI python 03_build_oracle.py
"""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _level2_common import write_status  # noqa: E402

STATUS = os.path.join(HERE, "LEVEL2_STATUS.json")
STEPS = ["build_oracle.py", "build_phase1a.py", "build_phase1c.py"]


def main():
    repo = os.path.abspath(os.environ.get("REPROFLY_REPO") or os.path.join(
        os.path.dirname(HERE), os.pardir, os.pardir, "ReproFly-AI"))
    if not os.path.isdir(os.path.join(repo, "scripts")) or not os.environ.get("REPROFLY_DATA_ROOT"):
        write_status(STATUS, "03_build_oracle", "SKIPPED_WITH_PREREQUISITES",
                     "requires REPROFLY_REPO code package + REPROFLY_DATA_ROOT "
                     "(and step 02 completed)")
        return 0
    for s in STEPS:
        r = subprocess.run([sys.executable, os.path.join(repo, "scripts", s)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            write_status(STATUS, f"03_build_oracle:{s}", "FAIL",
                         (r.stderr or r.stdout or "")[-400:])
            return 0
    write_status(STATUS, "03_build_oracle", "PASS", "oracle + endpoints rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
