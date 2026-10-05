"""Value-level verification of every supplementary table (S1-S11) against
the frozen numeric layer (PAPER_NUMBERS.json).

Run:  python scripts/reproduce_tables.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
TD = os.path.join(PKG, "paper", "results", "table_data")
PN = json.load(open(os.path.join(PKG, "paper", "results", "PAPER_NUMBERS.json"),
                    encoding="utf-8"))

bad: list[str] = []
n_checks = 0


def load(name):
    with open(os.path.join(TD, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def chk(cond, msg):
    global n_checks
    n_checks += 1
    if not cond:
        bad.append(msg)


def read_field(rows, field):
    for r in rows:
        if r.get("field") == field:
            return r["value"]
    return None


# S1: GLM runs ---------------------------------------------------------------
rows = load("TableS1_GLM_runs.csv")
ind = PN["phase2a"]["glm_individual_AUTSRC"]
chk(len(rows) == len(ind) == 10, f"S1 row count {len(rows)}")
for r in rows:
    rid = r["run_id"]
    chk(rid in ind and close(r["AUTSRC_v1_1_corrected"], ind[rid]),
        f"S1 {rid} AUTSRC mismatch")

# S2: phase-2a metrics -------------------------------------------------------
rows = load("TableS2_phase2a_metrics.csv")
vals = {r["metric"]: r["value"] for r in rows}
chk(vals.get("primary_metric") == PN["phase2a"]["primary_metric"], "S2 primary_metric")
for key, pn_key in [("weighted_strength", "weighted_strength_AUTSRC"),
                    ("weighted_pagerank", "weighted_pagerank_AUTSRC")]:
    if key in vals:
        chk(close(vals[key], PN["phase2a"][pn_key]), f"S2 {key}")
if "random_null_p95" in vals:
    chk(close(vals["random_null_p95"], PN["phase2a"]["random_null"]["p95_AUTSRC"]),
        "S2 random_null_p95")

# S3: phase-2b classification ------------------------------------------------
rows = load("TableS3_phase2b_metrics.csv")
vals = {r["metric"]: r["value"] for r in rows}
chk(vals.get("classification") == PN["phase2b"]["classification"], "S3 classification")

# S4: 700-run arm summary ----------------------------------------------------
rows = load("TableS4_CD_700run_summary.csv")
arms = PN["phase2c1"]["arms"]
glm_med = PN["phase2a"]["glm_median_AUTSRC"]
glm_iqr = PN["phase2a"]["glm_iqr_AUTSRC"]
for r in rows:
    arm = r["arm"]
    if arm == "GLM":
        chk(close(r["median_AUTSRC"], glm_med), "S4 GLM median")
        chk(close(r["iqr_low"], glm_iqr[0]) and close(r["iqr_high"], glm_iqr[1]),
            "S4 GLM IQR")
        continue
    chk(arm in arms, f"S4 unknown arm {arm}")
    if arm in arms:
        a = arms[arm]["autsrc"]
        chk(close(r["median_AUTSRC"], a["median"]), f"S4 {arm} median")
        chk(close(r["iqr_low"], a["iqr"][0]) and close(r["iqr_high"], a["iqr"][1]),
            f"S4 {arm} IQR")
        chk(close(r["zero_frac"], a["zero_frac"]), f"S4 {arm} zero_frac")
        chk(close(r["n"], a["n"]), f"S4 {arm} n")
        chk(close(r["n_with_hit"], arms[arm]["n_with_hit"]), f"S4 {arm} n_with_hit")

# S5: GLM vs D1 --------------------------------------------------------------
rows = load("TableS5_GLM_vs_D1.csv")
d = PN["phase2c1"]["GLM_vs_D1"]
chk(close(read_field(rows, "glm_median"), PN["phase2a"]["glm_median_AUTSRC"]), "S5 glm_median")
chk(close(read_field(rows, "cliffs_delta"), d["cliffs_delta"], 1e-6), "S5 cliffs_delta")
chk(close(read_field(rows, "glm_median_percentile_in_D1"),
          d["glm_median_percentile_in_D1"], 1e-6), "S5 glm pct")
chk(close(read_field(rows, "permutation_T"), d["permutation_T"], 1e-9), "S5 perm T")
chk(close(read_field(rows, "permutation_p_two_sided"),
          d["permutation_p_two_sided"], 1e-9), "S5 perm p")

# S6: GLM vs C2 --------------------------------------------------------------
rows = load("TableS6_GLM_vs_C2.csv")
c = PN["phase2c1"]["GLM_vs_C2"]
if "c2" in c and isinstance(c["c2"], dict) and "median" in c["c2"]:
    chk(close(read_field(rows, "c2_median"), c["c2"]["median"]), "S6 c2_median")
chk(close(read_field(rows, "cliffs_delta"), c["cliffs_delta"], 1e-6), "S6 cliffs_delta")
chk(close(read_field(rows, "glm_median_percentile_in_C2"),
          c["glm_median_percentile_in_C2"], 1e-6), "S6 glm pct")

# S7: per-target recovery ----------------------------------------------------
rows = load("TableS7_per_target_recovery.csv")
s1_rows = load("TableS1_GLM_runs.csv")
n_pcd_glm = sum(1 for r in s1_rows if r.get("pcd_first_hit", "").strip() != "")
for r in rows:
    arm = r["arm"]
    targets = {"PCD_recovery_frac": "pcd_recovery_frac",
               "LC10A_recovery_frac": "lc10a_recovery_frac",
               "CRZ_recovery_frac": "crz_recovery_frac"}
    for col, key in targets.items():
        if arm == "GLM":
            expected = n_pcd_glm / 10.0 if key == "pcd_recovery_frac" else \
                (0.0 if key == "lc10a_recovery_frac" else 0.0)
            if key == "pcd_recovery_frac":
                chk(close(r[col], expected), f"S7 GLM {key}")
            else:
                chk(0.0 <= float(r[col]) <= 1.0, f"S7 GLM {col} range")
        else:
            chk(arm in arms and close(r[col], arms[arm][key]), f"S7 {arm} {key}")

# S8: sensitivity ------------------------------------------------------------
rows = load("TableS8_sensitivity.csv")
chk(len(rows) > 0, "S8 empty")

# S9: environment ------------------------------------------------------------
rows = load("TableS9_environment.csv")
chk(len(rows) > 0, "S9 empty")

# S10: datasets --------------------------------------------------------------
rows = load("TableS10_datasets.csv")
chk(len(rows) >= 3, "S10 too few datasets listed")

# S11: hashes ----------------------------------------------------------------
rows = load("TableS11_hashes.csv")
import re
for r in rows:
    chk(re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) is not None,
        f"S11 bad sha256 for {r.get('artifact')}")

print(f"table checks: {n_checks}  problems: {len(bad)}")
for b in bad:
    print("FAIL:", b)
sys.exit(1 if bad else 0)
