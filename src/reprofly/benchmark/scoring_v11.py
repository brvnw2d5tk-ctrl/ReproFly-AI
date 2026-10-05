"""Phase 1F.2 FINAL scorer upgrade — OPTION B target-set scoring.

Human-selected architecture: primary scoring units are UNIQUE APPROVED
MALECNS TYPE SETS (GTSET_*), not literature records or functional
labels. Primary metric AUTSRC (renamed from AUGRC to avoid silent
terminology reuse — charter §9). Biological GroupRecall becomes
SECONDARY. CRZ_EXCLUDED sensitivity preregistered. Size-adjusted
random nulls (RandomFirstHitPercentile) retained.

Synthetic tests only — NO real result is scored (none exist).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .budgets import CHECKPOINTS, PERTURBATION_BUDGET
from .bridge import GTEntityScoringRecord, random_first_hit_percentile
from .metrics import (
    NOT_REACHED,
    first_ground_truth_hit,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    type_recall_at,
)


@dataclass(frozen=True)
class TargetSet:
    """One approved primary scoring unit (deduplicated type set)."""

    gt_set_id: str
    approved_types: tuple[str, ...]        # real names (PRIVATE)
    blind_ids: tuple[str, ...]             # blind equivalents
    biological_groups: tuple[str, ...]     # provenance only (secondary)
    caveats: str = ""

    @property
    def size(self) -> int:
        return len(self.approved_types)

    def recovered_by(self, perturbed_blind_ids: set[str]) -> bool:
        return any(b in perturbed_blind_ids for b in self.blind_ids)


def target_set_recall_curve(
    target_sets: list[TargetSet],
    attempt_blind_ids: list[str],
    max_k: Optional[int] = None,
) -> list[float]:
    """TargetSetRecall(k) = fraction of the N approved primary target
    sets for which at least one member type has been successfully
    perturbed by attempt k (charter §9)."""
    if not target_sets:
        raise ValueError("no target sets")
    limit = max_k or PERTURBATION_BUDGET
    hit: set[str] = set()
    perturbed: set[str] = set()
    curve: list[float] = []
    n = len(target_sets)
    for k in range(1, limit + 1):
        if k <= len(attempt_blind_ids):
            perturbed.add(attempt_blind_ids[k - 1])
            for ts in target_sets:
                if ts.recovered_by(perturbed):
                    hit.add(ts.gt_set_id)
        curve.append(len(hit) / n)
    return curve


def autsrc(curve: list[float]) -> float:
    """Area Under Target-Set Recall Curve: (1/300) sum TargetSetRecall(k).
    Discrete mean over k = 1..K (identical estimator shape to the old
    AUGRC, new name + new recall unit — never silent reuse)."""
    if not curve:
        raise ValueError("empty TargetSetRecall curve")
    return sum(curve) / len(curve)


def biological_group_recall_curve_secondary(
    target_sets: list[TargetSet],
    attempt_blind_ids: list[str],
    max_k: Optional[int] = None,
) -> list[float]:
    """SECONDARY biological GroupRecall: a biological group is
    recovered when at least one target set listing that group is
    recovered (multifunctional membership recorded, no duplicate
    PRIMARY credit — charter §8)."""
    limit = max_k or PERTURBATION_BUDGET
    groups = {g for ts in target_sets for g in ts.biological_groups}
    if not groups:
        return [0.0] * limit
    perturbed: set[str] = set()
    hit_groups: set[str] = set()
    curve: list[float] = []
    for k in range(1, limit + 1):
        if k <= len(attempt_blind_ids):
            perturbed.add(attempt_blind_ids[k - 1])
            for ts in target_sets:
                if ts.recovered_by(perturbed):
                    hit_groups.update(ts.biological_groups)
        curve.append(len(hit_groups) / len(groups))
    return curve


def experiments_to_50pct_target_set_recall(curve: list[float]) -> Any:
    for k, v in enumerate(curve, start=1):
        if v >= 0.5:
            return k
    return NOT_REACHED


def per_target_set_discovery(
    target_sets: list[TargetSet],
    attempt_blind_ids: list[str],
) -> dict[str, int]:
    """First attempt index at which each target set is recovered
    (blinded reporting uses gt_set_id, not biology names)."""
    perturbed: set[str] = set()
    first_hit: dict[str, int] = {}
    for k, bid in enumerate(attempt_blind_ids, start=1):
        perturbed.add(bid)
        for ts in target_sets:
            if ts.gt_set_id not in first_hit and ts.recovered_by(perturbed):
                first_hit[ts.gt_set_id] = k
    return first_hit


def full_target_set_score(
    target_sets: list[TargetSet],
    attempt_blind_ids: list[str],
    submitted_blind_ids: list[str],
    relevant_blind_ids: Iterable[str],
) -> dict[str, Any]:
    """The complete deterministic scorer output for one run under
    OPTION B (primary AUTSRC + secondaries incl. size-adjusted nulls)."""
    curve = target_set_recall_curve(target_sets, attempt_blind_ids)
    group_curve = biological_group_recall_curve_secondary(
        target_sets, attempt_blind_ids
    )
    relevant = set(relevant_blind_ids)
    out: dict[str, Any] = {
        "AUTSRC": autsrc(curve),
        "ExperimentsTo50PercentTargetSetRecall":
            experiments_to_50pct_target_set_recall(curve),
        "FirstGroundTruthHit": first_ground_truth_hit(
            attempt_blind_ids, relevant
        ),
        "PerTargetSetDiscovery": per_target_set_discovery(
            target_sets, attempt_blind_ids
        ),
        "RandomFirstHitPercentile_per_set": {
            ts.gt_set_id: random_first_hit_percentile(
                n=11_657, m=ts.size,
                observed_k=per_target_set_discovery(
                    target_sets, attempt_blind_ids
                ).get(ts.gt_set_id, PERTURBATION_BUDGET + 1),
            )
            for ts in target_sets
        },
    }
    for k in CHECKPOINTS:
        out[f"TargetSetRecall@{k}"] = (
            curve[k - 1] if len(curve) >= k else None
        )
        out[f"GroupRecall@{k}"] = (
            group_curve[k - 1] if len(group_curve) >= k else None
        )
        out[f"TypeRecall@{k}"] = type_recall_at(
            attempt_blind_ids, relevant, k
        )
    out["Precision@20"] = precision_at_k(submitted_blind_ids, relevant, 20)
    out["Recall@50"] = recall_at_k(submitted_blind_ids, relevant, 50)
    out["NDCG@50"] = ndcg_at_k(submitted_blind_ids, relevant, 50)
    return out


def crz_excluded_sensitivity(
    target_sets: list[TargetSet],
    attempt_blind_ids: list[str],
) -> dict[str, Any]:
    """Preregistered CRZ_EXCLUDED sensitivity: removes GTSET_CRZ from
    the scoring universe; remaining recovery reported DESCRIPTIVELY
    (only 2 independent sets remain — cannot replace primary)."""
    remaining = [ts for ts in target_sets if ts.gt_set_id != "GTSET_CRZ"]
    curve = target_set_recall_curve(remaining, attempt_blind_ids)
    return {
        "sensitivity": "CRZ_EXCLUDED (preregistered)",
        "n_remaining_sets": len(remaining),
        "target_set_recall_curve": curve,
        "descriptive_autsrc": autsrc(curve) if curve else None,
        "note": "descriptive/supporting only — cannot replace the "
                "primary benchmark",
    }


__all__ = [
    "TargetSet", "target_set_recall_curve", "autsrc",
    "biological_group_recall_curve_secondary",
    "experiments_to_50pct_target_set_recall", "per_target_set_discovery",
    "full_target_set_score", "crz_excluded_sensitivity",
]
