"""Benchmark package — formal AI benchmark protocol (Phase 1E).

Design / implementation / tests / frozen protocol ONLY. Phase 1E
executes ZERO real biological perturbations, ZERO GLM Scientist runs,
ZERO formal baseline rankings, and ZERO ground-truth scoring. The
formal benchmark runs only after the protocol is human-locked (1F).
"""

from __future__ import annotations

from . import baselines, blinding, budgets, gt_audit, metrics, tools, universe
from .baselines import (
    PAGERANK_ALPHA,
    PAGERANK_MAX_ITER,
    PAGERANK_TOL,
    PageRankConvergenceError,
    pagerank_ranking,
    random_ranking,
    visit_mass_ranking,
    weighted_strength_ranking,
)
from .blinding import (
    build_blind_map,
    is_blind_id,
    load_blind_map,
    load_or_create_secret,
    save_blind_map,
)
from .budgets import (
    ALREADY_INSPECTED,
    CHECKPOINTS,
    DUPLICATE_PERTURBATION,
    INVALID_TARGET,
    N_AI_RUNS,
    PERTURBATION_BUDGET,
    RANDOM_MASTER_SEED,
    RANDOM_PERMUTATIONS,
    SCIENTIST_MODEL,
    STRUCTURAL_INSPECTION_BUDGET,
    TECHNICAL_FAILURE,
    TECHNICAL_RETRIES_MAX,
    TOP_N_FINAL,
    BudgetExhaustedError,
    BudgetLedger,
)
from .gt_audit import (
    VALID_STATUSES,
    WORKSHEET_COLUMNS,
    GroundTruthAudit,
    audit_mapping_worksheet,
)
from .metrics import (
    NOT_REACHED,
    augrc,
    experiments_to_50pct_group_recall,
    first_ground_truth_hit,
    full_score,
    group_recall_curve,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    type_recall_at,
)
from .tools import (
    SCIENTIST_TOOL_ALLOWLIST,
    TOOL_SCHEMAS,
    Precommitment,
    inspect_candidate_response,
    list_candidate_ids_response,
    run_silencing_response,
)
from .universe import (
    PRIMARY_HORIZON,
    UNIVERSE_MAX,
    UNIVERSE_MIN,
    UniverseAggregates,
    UniverseBuild,
    build_universe,
    compute_path_eligibility,
)

__all__ = [
    "blinding", "universe", "budgets", "tools", "metrics", "baselines",
    "gt_audit",
    # blinding
    "build_blind_map", "load_blind_map", "load_or_create_secret",
    "save_blind_map", "is_blind_id",
    # universe
    "build_universe", "compute_path_eligibility", "UniverseBuild",
    "UniverseAggregates", "PRIMARY_HORIZON", "UNIVERSE_MIN", "UNIVERSE_MAX",
    # budgets
    "BudgetLedger", "BudgetExhaustedError", "N_AI_RUNS",
    "PERTURBATION_BUDGET", "STRUCTURAL_INSPECTION_BUDGET", "CHECKPOINTS",
    "TOP_N_FINAL", "RANDOM_PERMUTATIONS", "RANDOM_MASTER_SEED",
    "TECHNICAL_RETRIES_MAX", "SCIENTIST_MODEL", "DUPLICATE_PERTURBATION",
    "INVALID_TARGET", "ALREADY_INSPECTED", "TECHNICAL_FAILURE",
    # tools
    "SCIENTIST_TOOL_ALLOWLIST", "TOOL_SCHEMAS", "Precommitment",
    "list_candidate_ids_response", "inspect_candidate_response",
    "run_silencing_response",
    # metrics
    "group_recall_curve", "augrc", "type_recall_at",
    "first_ground_truth_hit", "experiments_to_50pct_group_recall",
    "precision_at_k", "recall_at_k", "ndcg_at_k", "full_score",
    "NOT_REACHED",
    # baselines
    "random_ranking", "weighted_strength_ranking", "pagerank_ranking",
    "visit_mass_ranking", "PAGERANK_ALPHA", "PAGERANK_TOL",
    "PAGERANK_MAX_ITER", "PageRankConvergenceError",
    # gt audit
    "audit_mapping_worksheet", "GroundTruthAudit", "WORKSHEET_COLUMNS",
    "VALID_STATUSES",
]
