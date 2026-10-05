"""Finite-Horizon First-Passage Structural Transmission Oracle.

Phase 1B instrument: a transparent, deterministic, auditable structural
transmission engine over the frozen MaleCNS v1.0 primary graph. This is a
STRUCTURAL model — not neural activity, not functional connectivity, not
a biophysical/LIF simulation (see docs/oracle/ASSUMPTIONS_AND_LIMITATIONS).

Boundary (Phase 1B): the engine is exercised on toy graphs and on the
real graph for baseline calibration / horizon calibration / a synthetic
runtime benchmark ONLY. No biological cell-type or neuron silencing
analysis is executed. PAPER MCP run_silencing stays disabled pending
human approval (Phase 1C).
"""

from __future__ import annotations

from .calibration import (
    SweepResult,
    audit_compartment_exposure,
    build_ge5_transition,
    marginal_gains,
    overlap_excluded_mask,
    run_horizon_sweep,
    validate_bounds,
    validate_hits_sum,
    validate_mass_balance_ceiling,
    validate_monotonicity,
    window_metrics,
)
from .channels import (
    CHANNEL_AI_NAMES,
    CHANNEL_LABELS,
    ChannelMasks,
    build_all_channel_masks,
    build_channel_masks,
    build_global_controls,
    primary_ids_from_endpoint,
    registry_payload,
)
from .engine import OracleEngine
from .errors import (
    DuplicatePairError,
    HorizonUndefinedError,
    InvalidWeightError,
    OracleError,
    OracleViolationError,
)
from .first_passage import propagate_dense, propagate_sparse
from .horizon import (
    HorizonStats,
    candidate_horizons,
    distance_stats,
    multi_source_shortest_distances,
    propose_horizon,
    stats_payload,
)
from .mass_balance import (
    DEFAULT_BALANCE_TOLERANCE,
    PropagationResult,
    StepAccount,
    assert_mass_balance,
    balance_table,
)
from .schemas import (
    OracleStatusModel,
    SilencingReadoutModel,
    TransmissionResultModel,
)
from .transition import (
    TransitionQC,
    build_node_index,
    build_transition,
    validate_edge_table,
)

__all__ = [
    "OracleEngine",
    "CHANNEL_AI_NAMES",
    "CHANNEL_LABELS",
    "ChannelMasks",
    "build_channel_masks",
    "build_all_channel_masks",
    "build_global_controls",
    "primary_ids_from_endpoint",
    "registry_payload",
    "propagate_sparse",
    "propagate_dense",
    "validate_edge_table",
    "build_node_index",
    "build_transition",
    "TransitionQC",
    "multi_source_shortest_distances",
    "distance_stats",
    "propose_horizon",
    "candidate_horizons",
    "HorizonStats",
    "stats_payload",
    "StepAccount",
    "PropagationResult",
    "assert_mass_balance",
    "balance_table",
    "DEFAULT_BALANCE_TOLERANCE",
    "TransmissionResultModel",
    "SilencingReadoutModel",
    "OracleStatusModel",
    "OracleError",
    "DuplicatePairError",
    "InvalidWeightError",
    "HorizonUndefinedError",
    "OracleViolationError",
    # Phase 1C calibration
    "SweepResult",
    "run_horizon_sweep",
    "marginal_gains",
    "window_metrics",
    "build_ge5_transition",
    "overlap_excluded_mask",
    "audit_compartment_exposure",
    "validate_monotonicity",
    "validate_bounds",
    "validate_hits_sum",
    "validate_mass_balance_ceiling",
]
