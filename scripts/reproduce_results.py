"""Reproduce the paper's main results from the frozen paper-release data layer.

Level-1 reproduction (no LLM, no network):
  1. Recompute every deterministic key statistic from the frozen
     authoritative result copies in  data/paper_release/authoritative/
  2. Compare against the frozen numeric layer (PAPER_NUMBERS.json, byte-verified against expected/expected_results.json by verify_release.py)
  3. Verify the recorded locked-permutation output is self-consistent
  4. Write  output/EXPECTED_PAPER_RESULTS_MATCH  =  PASS | FAIL

This script is standalone: it needs only numpy + the frozen files.
Run:  python scripts/reproduce_results.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
FROZEN = os.path.join(PKG, "paper", "results")
AUTH = os.path.join(FROZEN, "authoritative")

d2a = json.load(open(os.path.join(AUTH, "phase2a_SEALED_SCORING_RESULTS_v1.1_CORRECTED.json"),
                     encoding="utf-8"))
d2b = json.load(open(os.path.join(AUTH, "phase2b_PHASE2B_BEHAVIOR_RESULTS_attempt_002.json"),
                     encoding="utf-8"))
d2c1 = json.load(open(os.path.join(AUTH, "phase2c1_PHASE2C1_SCORING_RESULTS_v1.0.json"),
                      encoding="utf-8"))
cmp1 = json.load(open(os.path.join(AUTH, "phase2c1_PHASE2C1_MATCHED_BASELINE_COMPARISON_v1.0.json"),
                      encoding="utf-8"))
stage1 = json.load(open(os.path.join(AUTH, "phase2c1_STAGE1_ATTEMPT_003_STATUS.json"),
                        encoding="utf-8"))
PN = json.load(open(os.path.join(FROZEN, "PAPER_NUMBERS.json"), encoding="utf-8"))

CHECKS: list[dict] = []


def check(name: str, recomputed, expected, tol: float = 1e-12) -> None:
    if isinstance(expected, (int, float)) and isinstance(recomputed, (int, float)):
        ok = abs(float(recomputed) - float(expected)) <= tol
    else:
        ok = recomputed == expected
    CHECKS.append({"check": name, "recomputed": repr(recomputed),
                   "expected": repr(expected), "pass": bool(ok)})


def med_iqr(vals):
    v = np.sort(np.asarray(vals, dtype=float))
    return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))


# ---- Phase 2A --------------------------------------------------------------
glm_vals = [d2a["runs"][r]["AUTSRC"] for r in d2a["runs"]]
m, lo, hi = med_iqr(glm_vals)
check("2A.glm_median", m, PN["phase2a"]["glm_median_AUTSRC"])
check("2A.glm_iqr_low", lo, PN["phase2a"]["glm_iqr_AUTSRC"][0])
check("2A.glm_iqr_high", hi, PN["phase2a"]["glm_iqr_AUTSRC"][1])
check("2A.glm_n_runs", len(glm_vals), PN["phase2a"]["glm_n_runs"])
for rid, v in d2a["runs"].items():
    check(f"2A.{rid}.AUTSRC", v["AUTSRC"],
          PN["phase2a"]["glm_individual_AUTSRC"][rid])
for b in ["weighted_strength", "weighted_pagerank", "baseline_visit_mass"]:
    check(f"2A.{b}", d2a["static_baselines"][b]["AUTSRC"],
          PN["phase2a"][f"{b}_AUTSRC"])
rn = d2a["random_null"]
check("2A.null_median", rn["median_AUTSRC"], PN["phase2a"]["random_null"]["median_AUTSRC"])
check("2A.null_p95", rn["p95_AUTSRC"], PN["phase2a"]["random_null"]["p95_AUTSRC"])
check("2A.null_perms", rn["permutations"], PN["phase2a"]["random_null"]["permutations"])
check("2A.null_seed", rn["master_seed"], PN["phase2a"]["random_null"]["master_seed"])

# ---- Phase 2B (M8 cross-run medians; M12 counts) ---------------------------
grid = {rid: d2b["results"]["M8"]["per_run"][rid]["grid"] for rid in
        d2b["results"]["M8"]["per_run"]}
for key, st in PN["phase2b"]["M8_cross_run"].items():
    comp, beh = key.split("|")
    vals = [g[key]["r_rb"] for g in grid.values()
            if key in g and isinstance(g[key].get("r_rb"), (int, float))]
    m = float(np.median(vals))
    check(f"2B.M8.{key}.median", m, st["median"])
    check(f"2B.M8.{key}.n", st["n"], len(vals))
for st, v in PN["phase2b"]["M12_observability"].items():
    src = d2b["results"]["M12"]["sets"][st]
    check(f"2B.M12.{st}.inspected", src["members_ever_inspected"],
          v["members_ever_inspected"])
    check(f"2B.M12.{st}.visible", src["members_ever_visible_top10_neighbor"],
          v["members_ever_visible_top10_neighbor"])
    check(f"2B.M12.{st}.perturbed", src["members_ever_valid_perturbed"],
          v["members_ever_valid_perturbed"])

# ---- Phase 2C.1 (700 trajectories + contrasts) -----------------------------
traj = d2c1["trajectories"]
check("2C1.n_traj", len(traj), PN["phase2c1"]["n_trajectories"])
check("2C1.stage1_complete",
      stage1["PHASE2C1_STAGE1_ATTEMPT_003_STATUS"], "COMPLETE")
check("2C1.bundle_hash", stage1["PHASE2C1_TRAJECTORY_BUNDLE_HASH"],
      PN["phase2c1"]["trajectory_bundle_hash"])

by_arm: dict[str, list[float]] = {}
for k, v in traj.items():
    by_arm.setdefault(k.split("/")[0], []).append(v["AUTSRC"])
for arm, av in PN["phase2c1"]["arms"].items():
    vals = by_arm[arm]
    m, lo, hi = med_iqr(vals)
    check(f"2C1.{arm}.n", len(vals), av["autsrc"]["n"])
    check(f"2C1.{arm}.median", m, av["autsrc"]["median"])
    check(f"2C1.{arm}.iqr_low", lo, av["autsrc"]["iqr"][0])
    check(f"2C1.{arm}.iqr_high", hi, av["autsrc"]["iqr"][1])
    check(f"2C1.{arm}.zero_frac", float(np.mean(np.asarray(vals) == 0.0)),
          av["autsrc"]["zero_frac"])
    check(f"2C1.{arm}.max", float(max(vals)), av["autsrc"]["max"])

glm_vals_c1 = cmp1["glm_values"]
d1_vals = by_arm["D1"]
c2_vals = by_arm["C2"]

# Cliff's delta GLM vs D1 (deterministic): (n_glm>d1 - n_glm<d1) / (10*100)
n_gt = sum(1 for g in glm_vals_c1 for d in d1_vals if g > d)
n_lt = sum(1 for g in glm_vals_c1 for d in d1_vals if g < d)
n_tie = sum(1 for g in glm_vals_c1 for d in d1_vals if g == d)
delta = (n_gt - n_lt) / (len(glm_vals_c1) * len(d1_vals))
check("2C1.GLM_vs_D1.cliffs_delta", delta, PN["phase2c1"]["GLM_vs_D1"]["cliffs_delta"])
# probabilistic superiority counts ties as one half (midrank convention)
p_sup = (n_gt + 0.5 * n_tie) / (len(glm_vals_c1) * len(d1_vals))
check("2C1.GLM_vs_D1.p_superiority_midrank_ties_half", round(p_sup, 4),
      PN["phase2c1"]["GLM_vs_D1"]["probability_superiority"])
check("2C1.GLM_vs_D1.p_superiority_display_rounding",
      round(p_sup, 3), PN["phase2c1"]["GLM_vs_D1"]["probability_superiority_display_rounding"])

# GLM median percentile in D1 (midrank)
def midrank_percentile(x, ref):
    r = np.sort(np.asarray(ref, dtype=float))
    below = np.searchsorted(r, x, side="left")
    above = np.searchsorted(r, x, side="right")
    return 100.0 * (below + above) / (2 * len(r))


pct = midrank_percentile(np.median(glm_vals_c1), d1_vals)
check("2C1.GLM_median_pct_in_D1", pct, PN["phase2c1"]["GLM_vs_D1"]["glm_median_percentile_in_D1"])
for i, g in enumerate(glm_vals_c1):
    check(f"2C1.glm_run{i+1:03d}_pct_in_D1", midrank_percentile(g, d1_vals),
          PN["phase2c1"]["GLM_vs_D1"]["glm_run_percentiles_in_D1"][i])

n_gt2 = sum(1 for g in glm_vals_c1 for c in c2_vals if g > c)
n_lt2 = sum(1 for g in glm_vals_c1 for c in c2_vals if g < c)
delta2 = (n_gt2 - n_lt2) / (len(glm_vals_c1) * len(c2_vals))
check("2C1.GLM_vs_C2.cliffs_delta", delta2, PN["phase2c1"]["GLM_vs_C2"]["cliffs_delta"])
pct2 = midrank_percentile(np.median(glm_vals_c1), c2_vals)
check("2C1.GLM_median_pct_in_C2", pct2, PN["phase2c1"]["GLM_vs_C2"]["glm_median_percentile_in_C2"])

# recorded locked-permutation output must be self-consistent:
#   T == observed median difference;  p sits on the exact (k+1)/(n+1) grid
perm = PN["phase2c1"]["GLM_vs_D1"]
T = perm["permutation_T"]
check("2C1.perm_T_equals_observed_median_diff", T,
      float(np.median(glm_vals_c1)) - float(np.median(d1_vals)))
p = perm["permutation_p_two_sided"]
n_perms = 10000
check("2C1.perm_p_on_exact_grid", round(p * (n_perms + 1)) / (n_perms + 1), p)
check("2C1.perm_p_in_range", True, 0.0 < p <= 1.0)

# per-target recovery spot-check: runs with any GT hit (FirstGroundTruthHit)
for arm, av in PN["phase2c1"]["arms"].items():
    n_hit = sum(1 for v in (traj[k] for k in traj if k.split("/")[0] == arm)
                if v["FirstGroundTruthHit"] != "NOT_REACHED")
    check(f"2C1.{arm}.n_with_hit", n_hit, av["n_with_hit"])

# ---- verdict ----------------------------------------------------------------
failed = [c for c in CHECKS if not c["pass"]]
out_dir = os.path.join(PKG, "output")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "EXPECTED_PAPER_RESULTS_MATCH")
with open(out_path, "w", encoding="utf-8", newline="\n") as f:
    f.write(f"EXPECTED_PAPER_RESULTS_MATCH = {'PASS' if not failed else 'FAIL'}\n"
            f"n_checks = {len(CHECKS)}\nn_failed = {len(failed)}\n")
with open(os.path.join(PKG, "output", "REPRODUCTION_CHECKS.json"), "w", encoding="utf-8",
          newline="\n") as f:
    json.dump({"record_kind": "REPRODUCTION_CHECKS", "n_checks": len(CHECKS),
               "n_failed": len(failed), "checks": CHECKS}, f, indent=1,
              ensure_ascii=False)
    f.write("\n")

print(f"checks: {len(CHECKS)}  failed: {len(failed)}")
for c in failed[:20]:
    print("FAIL:", c)
print(open(out_path).read())
return_code = 0 if not failed else 1
sys.exit(return_code)
