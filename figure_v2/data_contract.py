"""Strict accessors for the frozen v2 figure-data release."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from itertools import product
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np


_MODULE_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROJECT_ROOT = Path(os.environ.get("REPROFLY_PROJECT_ROOT", _MODULE_REPO_ROOT))
SOURCE_DIR = Path(os.environ.get("REPROFLY_FIGURE_DATA_DIR", str(_PROJECT_ROOT / "release" / "paper_release" / "figure_data_v2"))).resolve()


RUNS = tuple(f"AI_RUN_{number:03d}" for number in range(1, 11))
ARMS = ("GLM", "C1", "C2", "C3", "D1", "D2a", "D2b", "D3")
TARGETS = ("PCD", "LC10a", "CRZ")
BASELINES = ("weighted_strength", "weighted_pagerank", "baseline_visit_mass")
MEASURES = ("primary", "control_1", "control_2", "control_3")
BEHAVIORS = ("NEXT_LOCAL", "NEXT_DE_NOVO")
LEVELS = ("L5", "L13", "ge5", "overlap_excluded")
STAGES = ("visible_top10", "inspected", "valid_perturbed")
NOT_COMPUTABLE = "NOT_COMPUTABLE_INSUFFICIENT_GROUP_SIZE"


@dataclass(frozen=True)
class SourceContract:
    """Explicit schema and frozen row identity/order, independent of CSV contents.

    Distribution exports lack run IDs: repeated (arm, AUTSRC) values are legal.
    For these alone, ``unique`` is false and row counts/arm blocks are checked.
    Missing tokens are authorized by exact (row key, column), never globally.
    """

    columns: tuple[str, ...]
    numeric: tuple[str, ...]
    key: tuple[str, ...]
    expected_rows: tuple[tuple[str, ...], ...]
    unique: bool = True
    integers: tuple[str, ...] = ()
    categories: dict[str, tuple[str, ...]] = field(default_factory=dict)
    missing: dict[tuple[tuple[str, ...], str], tuple[str, ...]] = field(default_factory=dict)


def _contract(columns, numeric, key, rows, **kwargs) -> SourceContract:
    return SourceContract(tuple(columns.split(",")), tuple(numeric.split(",")),
                          tuple(key.split(",")),
                          tuple((row,) if isinstance(row, str) else tuple(row) for row in rows),
                          **kwargs)


# A required-file registry, not a glob: deleting any source must fail preflight.
# The four additional release tables also have explicit contracts, so callers
# cannot bypass validation by using a currently undisplayed quantitative CSV.
SOURCE_CONTRACTS = {
    "Fig1a_design_constants.csv": _contract(
        "item,value,note", "value", "item",
        ("candidate_universe_n", "glm_runs", "charged_attempts_per_run", "gt_pcd_m",
         "gt_lc10a_m", "gt_crz_m", "random_null_permutations", "random_null_master_seed",
         "matched_arms", "matched_runs_per_arm", "matched_trajectories", "c2_anyhit",
         "d1_anyhit", "glm_anyhit"), integers=("value",)),
    **{name: _contract("run_id,AUTSRC", "AUTSRC", "run_id", RUNS) for name in (
        "Fig2a_formal_runs.csv", "Fig2b_glm_overlay.csv", "Fig2c_glm_runs.csv")},
    "Fig2a_summary.csv": _contract(
        "series,median_AUTSRC,iqr_low,iqr_high,n", "median_AUTSRC,iqr_low,iqr_high,n",
        "series", ("GLM",), integers=("n",)),
    "Fig2b_random_null_quantiles.csv": _contract(
        "series,AUTSRC", "AUTSRC", "series",
        ("random_null_median", "random_null_mean", "random_null_p95", "random_null_ci95_high",
         "random_null_max", "random_null_permutations", "random_null_master_seed")),
    "Fig2c_graph_baselines.csv": _contract(
        "series,AUTSRC,iqr_low,iqr_high,kind", "AUTSRC,iqr_low,iqr_high", "series",
        (*BASELINES, "GLM"), categories={"kind": ("full_information", "tool_mediated")},
        missing={((series,), column): ("",) for series in BASELINES
                 for column in ("iqr_low", "iqr_high")}),
    "Fig2d_target_recovery.csv": _contract(
        "target,set_size,recovery_frac,runs_recovered,n", "set_size,recovery_frac,n",
        "target", TARGETS, integers=("set_size", "n")),
    "Fig3a_associations.csv": _contract(
        "response_measure,measure_class,next_behavior,median_assoc,iqr_low,iqr_high,n",
        "median_assoc,iqr_low,iqr_high,n", "response_measure,next_behavior",
        product(MEASURES, BEHAVIORS), integers=("n",),
        categories={"measure_class": ("primary", "control")}),
    "Fig3b_per_run.csv": _contract(
        "run_id,primary_NEXT_LOCAL,primary_NEXT_DE_NOVO",
        "primary_NEXT_LOCAL,primary_NEXT_DE_NOVO", "run_id", RUNS,
        missing={((run,), "primary_NEXT_DE_NOVO"): (NOT_COMPUTABLE,)
                 for run in ("AI_RUN_003", "AI_RUN_006")}),
    "Fig3c_m13_tiles.csv": _contract(
        "relabel_level,contrast,contrast_class,median_abs_difference,sign_concordance_fraction",
        "median_abs_difference,sign_concordance_fraction", "relabel_level,contrast",
        product(LEVELS, (f"{measure}|{behavior}" for measure, behavior in product(MEASURES, BEHAVIORS))),
        categories={"contrast_class": ("primary", "control")}),
    **{name: _contract(
        "arm,AUTSRC", "AUTSRC", "arm",
        (arm for arm in ARMS for _ in range(10 if arm == "GLM" else 100)), unique=False)
       for name in ("Fig4b_raincloud.csv", "SuppS9_family_distributions.csv")},
    "Fig4c_anyhit_ci.csv": _contract(
        "arm,n,n_with_hit,any_hit_frac,ci95_low,ci95_high,ci_method",
        "n,n_with_hit,any_hit_frac,ci95_low,ci95_high,ci_method", "arm", ("GLM", "C2", "D1"),
        integers=("n", "n_with_hit")),
    "Fig4d_effects.csv": _contract(
        "contrast,cliffs_delta,glm_median_percentile,perm_T,perm_p_two_sided,label",
        "cliffs_delta,glm_median_percentile,perm_T,perm_p_two_sided", "contrast",
        ("GLM_vs_D1", "GLM_vs_C2"),
        categories={"label": ("POST-HOC / EXPLORATORY ONLY",
                              "SECONDARY matched-information non-adaptive contrast")},
        missing={(("GLM_vs_C2",), column): ("",) for column in ("perm_T", "perm_p_two_sided")}),
    "Fig4e_target_matrix.csv": _contract(
        "arm,PCD,LC10a,CRZ", "PCD,LC10a,CRZ", "arm", ("GLM", "C2", "D1")),
    "Fig5a_accessibility.csv": _contract(
        "gt_set,target,set_size,stage,count", "set_size,count", "target,stage",
        product(("LC10a", "PCD", "CRZ"), STAGES), integers=("set_size", "count"),
        categories={"gt_set": ("GTSET_LC10A", "GTSET_PCD", "GTSET_CRZ")}),
    "Fig5b_ranks.csv": _contract(
        "gt_set,target,baseline,median_rank,universe_n", "median_rank,universe_n", "target,baseline",
        product(("LC10a", "PCD", "CRZ"), BASELINES), integers=("universe_n",),
        categories={"gt_set": ("GTSET_LC10A", "GTSET_PCD", "GTSET_CRZ")}),
    "Fig5c_policy_recovery.csv": _contract("arm,PCD,LC10a,CRZ", "PCD,LC10a,CRZ", "arm", ARMS),
    "SuppS10_recovery_full.csv": _contract(
        "arm,PCD_recovery_frac,LC10A_recovery_frac,CRZ_recovery_frac",
        "PCD_recovery_frac,LC10A_recovery_frac,CRZ_recovery_frac", "arm", ARMS),
    "SuppS8_checkpoints.csv": _contract(
        "series,checkpoint,TargetSetRecall,kind", "checkpoint,TargetSetRecall", "series,checkpoint",
        product((*RUNS, *BASELINES, "GLM_10runs_median"), ("50", "100", "200", "300")),
        integers=("checkpoint",), categories={"kind": ("GLM_run", "baseline_context")}),
}


def _validate_records(name: str, rows: list[dict[str, str]], contract: SourceContract) -> None:
    categories = dict(contract.categories)
    for index, column in enumerate(contract.key):
        categories[column] = tuple(dict.fromkeys(row[index] for row in contract.expected_rows))
    keys = []
    seen = set()
    for number, row in enumerate(rows, 2):
        key = tuple(row[column] for column in contract.key)
        if contract.unique and key in seen:
            raise ValueError(f"duplicate row key {key!r} in {name}")
        seen.add(key)
        keys.append(key)
        for column, allowed in categories.items():
            if row[column] not in allowed:
                raise ValueError(f"unknown category {row[column]!r} at {name}:{number}/{column}")
        for column in contract.numeric:
            raw = row[column]
            if raw in contract.missing.get((key, column), ()):
                continue
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"non-numeric value {raw!r} at {name}:{number}/{column}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite value at {name}:{number}/{column}")
            if column in contract.integers and not value.is_integer():
                raise ValueError(f"non-integer value at {name}:{number}/{column}")
        for column in set(contract.columns) - set(contract.numeric):
            if not row[column].strip():
                raise ValueError(f"empty text at {name}:{number}/{column}")
        _validate_categories_and_counts(name, row)
    if tuple(keys) != contract.expected_rows:
        raise ValueError(f"unexpected row count/order in {name}")


def _validate_categories_and_counts(name: str, row: dict[str, str]) -> None:
    """Check dependent labels, rather than allowing valid labels on wrong rows."""
    expected = {}
    if name == "Fig3a_associations.csv":
        expected["measure_class"] = "primary" if row["response_measure"] == "primary" else "control"
    elif name == "Fig3c_m13_tiles.csv":
        expected["contrast_class"] = "primary" if row["contrast"].startswith("primary|") else "control"
    elif name == "Fig2c_graph_baselines.csv":
        expected["kind"] = "tool_mediated" if row["series"] == "GLM" else "full_information"
    elif name == "SuppS8_checkpoints.csv":
        expected["kind"] = "GLM_run" if row["series"] in RUNS else "baseline_context"
    elif name in ("Fig5a_accessibility.csv", "Fig5b_ranks.csv"):
        expected["gt_set"] = "GTSET_" + row["target"].upper()
    elif name == "Fig4d_effects.csv":
        expected["label"] = ("POST-HOC / EXPLORATORY ONLY" if row["contrast"] == "GLM_vs_D1"
                             else "SECONDARY matched-information non-adaptive contrast")
    for column, value in expected.items():
        if row[column] != value:
            raise ValueError(f"inconsistent category at {name}/{column}")
    if name == "Fig2d_target_recovery.csv":
        try:
            numerator, denominator = (int(value) for value in row["runs_recovered"].split("/"))
        except ValueError as exc:
            raise ValueError(f"invalid runs_recovered fraction in {name}") from exc
        if (not 0 <= numerator <= denominator or denominator <= 0 or
                denominator != float(row["n"]) or
                not math.isclose(numerator / denominator, float(row["recovery_frac"]))):
            raise ValueError(f"inconsistent runs_recovered fraction in {name}")


def _source_path(name: str) -> Path:
    """Resolve a source name while keeping access inside the frozen directory."""
    path = SOURCE_DIR / name
    if path.parent != SOURCE_DIR or not path.is_file():
        raise FileNotFoundError(path)
    path = path.resolve()
    if path.parent != SOURCE_DIR:
        raise FileNotFoundError(path)
    return path


def load_rows(name: str) -> list[dict[str, str]]:
    """Load records only after exact schema, identity and value validation."""
    path = _source_path(name)
    if name not in SOURCE_CONTRACTS:
        raise ValueError(f"unregistered source contract: {name}")
    contract = SOURCE_CONTRACTS[name]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(field is None or not field for field in reader.fieldnames):
            raise ValueError(f"missing header: {name}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"duplicate header: {name}")
        if set(reader.fieldnames) != set(contract.columns):
            raise ValueError(f"unexpected columns in {name}: expected {contract.columns}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty source: {name}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"malformed row: {name}")
    _validate_records(name, rows, contract)
    return rows


def load_design_constants() -> dict[str, int]:
    """Return the integer constants recorded in Fig1a's frozen contract."""
    rows = load_rows("Fig1a_design_constants.csv")
    required = {"item", "value"}
    if not required <= set(rows[0]):
        raise ValueError("design constants require item and value columns")
    result: dict[str, int] = {}
    for row in rows:
        item = row["item"]
        if item in result:
            raise ValueError(f"duplicate design constant: {item}")
        try:
            value = float(row["value"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric design constant: {item}") from exc
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError(f"design constant is not a finite integer: {item}")
        result[item] = int(value)
    return result


def numeric_matrix(
    name: str,
    row_key: str,
    column_keys: Sequence[str],
    expected_rows: Sequence[str],
) -> np.ndarray:
    """Extract a finite numeric matrix with an exact, frozen row order."""
    records = load_rows(name)
    if not records or row_key not in records[0]:
        raise ValueError(f"missing row key {row_key!r} in {name}")
    missing_columns = [key for key in column_keys if key not in records[0]]
    if missing_columns:
        raise ValueError(f"missing columns in {name}: {missing_columns}")

    by_key: dict[str, dict[str, str]] = {}
    for record in records:
        key = record[row_key]
        if key in by_key:
            raise ValueError(f"duplicate {row_key} {key!r} in {name}")
        by_key[key] = record
    actual_order = list(by_key)
    expected_order = list(expected_rows)
    if actual_order != expected_order:
        raise ValueError(f"unexpected {row_key} order in {name}: {actual_order}")

    values: list[list[float]] = []
    for key in expected_order:
        row_values: list[float] = []
        for column in column_keys:
            try:
                value = float(by_key[key][column])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric value at {key}/{column} in {name}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite value at {key}/{column} in {name}")
            row_values.append(value)
        values.append(row_values)
    return np.asarray(values, dtype=float)


def validate_all_sources() -> None:
    """Fail closed on missing, unregistered or schema-invalid release sources."""
    for name in SOURCE_CONTRACTS:
        load_rows(name)
    unexpected = {path.name for path in SOURCE_DIR.glob("*.csv")} - SOURCE_CONTRACTS.keys()
    if unexpected:
        raise ValueError(f"unregistered source contracts: {sorted(unexpected)}")
