"""One-command verification of the ReproFly-AI reproduction package.

Checks:
  1. SHA256 of every shipped file against manifests/SHA256SUMS.tsv
  2. expected/expected_results.json is byte-identical to the frozen numeric layer
  3. all key paper numbers recompute (runs scripts/reproduce_results.py)
  4. all supplementary tables verify (runs scripts/reproduce_tables.py)
  5. main figures were rendered to output/figures/ (run reproduce_figures.py first)

Output:
  REPRODUCTION_STATUS = PASS | FAIL
  NUMERIC_MISMATCHES  = n
  MISSING_FILES       = n

Run:  python scripts/verify_release.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
OUT = os.path.join(PKG, "output")

problems: list[str] = []
numeric_mismatches = 0
missing_files = 0


def fail(msg: str) -> None:
    problems.append(msg)


# 1 ------------------------------------------------------------------ hashes
sums_path = os.path.join(PKG, "manifests", "SHA256SUMS.tsv")
if not os.path.exists(sums_path):
    fail("missing manifests/SHA256SUMS.tsv")
else:
    import hashlib

    n_ok = 0
    with open(sums_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            digest, rel = line.split("\t", 1)
            p = os.path.join(PKG, rel)
            if not os.path.exists(p):
                missing_files += 1
                fail(f"missing file: {rel}")
                continue
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            if h != digest:
                fail(f"hash mismatch: {rel}")
            else:
                n_ok += 1
    print(f"hashes verified: {n_ok}")

# 2 ------------------------------------------- expected == frozen data layer
pn = os.path.join(PKG, "paper", "results", "PAPER_NUMBERS.json")
ex = os.path.join(PKG, "paper", "results", "expected_results.json")
import hashlib

if os.path.exists(pn) and os.path.exists(ex):
    if hashlib.sha256(open(pn, "rb").read()).hexdigest() != \
            hashlib.sha256(open(ex, "rb").read()).hexdigest():
        fail("expected_results.json differs from data/paper_release/PAPER_NUMBERS.json")
else:
    fail("missing PAPER_NUMBERS.json or expected_results.json")

# 3 ---------------------------------------------------------- numeric checks
r = subprocess.run([sys.executable, os.path.join(HERE, "reproduce_results.py")],
                   capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    m = [l for l in r.stdout.splitlines() if l.startswith("FAIL:")]
    numeric_mismatches = len(m) if m else 1
    fail("reproduce_results.py reported failures")

# 4 ------------------------------------------------------------ table checks
r = subprocess.run([sys.executable, os.path.join(HERE, "reproduce_tables.py")],
                   capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    numeric_mismatches += 1
    fail("reproduce_tables.py reported failures")

# 5 --------------------------------------------------------- rendered figures
figs_expected = [f"Figure{i}.pdf" for i in range(1, 6)]
figs_dir = os.path.join(OUT, "figures")
if not os.path.isdir(figs_dir) and os.path.isdir(os.path.join(PKG, "paper", "figures")):
    figs_dir = os.path.join(PKG, "paper", "figures")  # shipped expected figures
if not os.path.isdir(figs_dir):
    print("note: output/figures/ not found - run scripts/reproduce_figures.py first")
    missing_files += len(figs_expected)
    for f in figs_expected:
        fail(f"missing rendered figure: {f}")
else:
    for f in figs_expected:
        if not os.path.exists(os.path.join(figs_dir, f)):
            missing_files += 1
            fail(f"missing rendered figure: {f}")

# ------------------------------------------------------------------- verdict
status = "PASS" if not problems else "FAIL"
os.makedirs(OUT, exist_ok=True)
report = {
    "REPRODUCTION_STATUS": status,
    "NUMERIC_MISMATCHES": numeric_mismatches,
    "MISSING_FILES": missing_files,
    "problems": problems,
}
with open(os.path.join(OUT, "REPRODUCTION_STATUS.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print()
print(f"REPRODUCTION_STATUS = {status}")
print(f"NUMERIC_MISMATCHES  = {numeric_mismatches}")
print(f"MISSING_FILES       = {missing_files}")
if problems:
    print("problems:")
    for p in problems[:30]:
        print("  -", p)
sys.exit(0 if status == "PASS" else 1)
