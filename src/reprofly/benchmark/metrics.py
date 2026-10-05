"""Deterministic benchmark metrics (Phase 1E §28-30, §50).

All scorers are pure deterministic Python. The LLM never computes its
own score. Definitions:

  hit(k)          : the k-th perturbation attempt was a VALID silencing
                    of an evaluable ground-truth type
  GroupRecall(k)  : fraction of evaluable functional groups with >= 1
                    hit among attempts 1..k
  AUGRC           : (1/300) * sum_{k=1..300} GroupRecall(k)

Secondary: GroupRecall@{50,100,200,300}, TypeRecall@{50,100,200,300},
FirstGroundTruthHit, ExperimentsTo50PercentGroupRecall (>300 ->
NOT_REACHED), and — on the final top-50 submission — Precision@20,
Recall@50, NDCG@50 with binary relevance (evaluable GT type = 1).
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

from .budgets import CHECKPOINTS, PERTURBATION_BUDGET

NOT_REACHED = "NOT_REACHED"


def _hits_by_attempt(
    attempt_blind_ids: Sequence[str],
    evaluable_gt_blind_ids: Iterable[str],
) -> list[bool]:
    """hit[k-1] = True iff attempt k validly silenced an evaluable GT
    type. (Duplicate/invalid attempts are not hits — they are attempts.)"""
    gt = set(evaluable_gt_blind_ids)
    return [bid in gt for bid in attempt_blind_ids]


def group_recall_curve(
    attempt_blind_ids: Sequence[str],
    evaluable_gt_blind_ids: Iterable[str],
    gt_blind_to_group: dict[str, str],
    n_evaluable_groups: int,
    max_k: Optional[int] = None,
) -> list[float]:
    """GroupRecall(k) for k = 1..max_k (default the perturbation budget)."""
    if n_evaluable_groups <= 0:
        raise ValueError("n_evaluable_groups must be positive")
    limit = max_k or PERTURBATION_BUDGET
    hits = _hits_by_attempt(attempt_blind_ids, evaluable_gt_blind_ids)
    groups_hit: set[str] = set()
    curve: list[float] = []
    for k in range(1, limit + 1):
        if k <= len(hits) and hits[k - 1]:
            groups_hit.add(gt_blind_to_group[attempt_blind_ids[k - 1]])
        curve.append(len(groups_hit) / n_evaluable_groups)
    return curve


def augrc(group_recall: Sequence[float]) -> float:
    """Area Under Group Recall Curve: discrete mean over k = 1..K."""
    if not group_recall:
        raise ValueError("empty GroupRecall curve")
    return sum(group_recall) / len(group_recall)


def type_recall_at(
    attempt_blind_ids: Sequence[str],
    evaluable_gt_blind_ids: Iterable[str],
    k: int,
) -> float:
    gt = set(evaluable_gt_blind_ids)
    if len(gt) == 0:
        raise ValueError("no evaluable GT types")
    hit_types = {b for b in attempt_blind_ids[:k] if b in gt}
    return len(hit_types) / len(gt)


def first_ground_truth_hit(
    attempt_blind_ids: Sequence[str],
    evaluable_gt_blind_ids: Iterable[str],
) -> Any:
    """1-based attempt index of the first GT hit, or NOT_REACHED."""
    hits = _hits_by_attempt(attempt_blind_ids, evaluable_gt_blind_ids)
    for k, h in enumerate(hits, start=1):
        if h:
            return k
    return NOT_REACHED


def experiments_to_50pct_group_recall(
    group_recall: Sequence[float],
) -> Any:
    """Smallest k with GroupRecall(k) >= 0.5, or NOT_REACHED."""
    for k, gr in enumerate(group_recall, start=1):
        if gr >= 0.5:
            return k
    return NOT_REACHED


# ---------------------------------------------------------------------------
# Final-submission metrics (binary relevance over the top-50 list)
# ---------------------------------------------------------------------------

def precision_at_k(submitted_blind_ids: Sequence[str],
                   relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    topk = submitted_blind_ids[:k]
    if not topk:
        return 0.0
    return sum(1 for b in topk if b in rel) / len(topk)


def recall_at_k(submitted_blind_ids: Sequence[str],
                relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("no relevant items")
    topk = set(submitted_blind_ids[:k])
    return len(topk & rel) / len(rel)


def ndcg_at_k(submitted_blind_ids: Sequence[str],
              relevant: Iterable[str], k: int) -> float:
    """Binary-relevance NDCG@k (no gains beyond relevance=1)."""
    rel = set(relevant)
    import math

    def dcg(ids: Sequence[str]) -> float:
        return sum(
            (1.0 if b in rel else 0.0) / math.log2(i + 2)
            for i, b in enumerate(ids[:k])
        )

    ideal = sorted(
        [b for b in submitted_blind_ids if b in rel]
        + [b for b in rel if b not in set(submitted_blind_ids)],
        key=lambda _b: 0,  # binary: any order of relevant-first is ideal
    )[:k]
    ideal_ids = ([1] * min(len(rel), k))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(ideal_ids)))
    if idcg == 0.0:
        return 0.0
    return dcg(submitted_blind_ids) / idcg


def full_score(
    attempt_blind_ids: Sequence[str],
    evaluable_gt_blind_ids: Iterable[str],
    gt_blind_to_group: dict[str, str],
    n_evaluable_groups: int,
    submitted_blind_ids: Sequence[str],
) -> dict[str, Any]:
    """The complete deterministic scorer output for one run."""
    gt = list(evaluable_gt_blind_ids)
    curve = group_recall_curve(
        attempt_blind_ids, gt, gt_blind_to_group, n_evaluable_groups,
        max_k=PERTURBATION_BUDGET,
    )
    out: dict[str, Any] = {
        "AUGRC": augrc(curve),
        "FirstGroundTruthHit": first_ground_truth_hit(attempt_blind_ids, gt),
        "ExperimentsTo50PercentGroupRecall": experiments_to_50pct_group_recall(curve),
    }
    for k in CHECKPOINTS:
        out[f"GroupRecall@{k}"] = (
            curve[k - 1] if len(curve) >= k else None
        )
        out[f"TypeRecall@{k}"] = type_recall_at(attempt_blind_ids, gt, k)
    out["FinalCandidatePrecision@20"] = precision_at_k(
        submitted_blind_ids, gt, 20
    )
    out["FinalCandidateRecall@50"] = recall_at_k(
        submitted_blind_ids, gt, 50
    )
    out["NDCG@50"] = ndcg_at_k(submitted_blind_ids, gt, 50)
    return out


__all__ = [
    "group_recall_curve", "augrc", "type_recall_at",
    "first_ground_truth_hit", "experiments_to_50pct_group_recall",
    "precision_at_k", "recall_at_k", "ndcg_at_k", "full_score",
    "NOT_REACHED",
]
