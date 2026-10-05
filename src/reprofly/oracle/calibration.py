"""Phase 1C calibration: horizon sweep, window metrics, sensitivity
variants, and neutral compartment-exposure audit.

Everything here is deterministic instrument calibration over the FROZEN
Phase 1A endpoints and the FROZEN Phase 1B oracle mathematics. No
biological perturbation, no candidate ranking, no ground truth, no
composite score. Horizon decisions belong to a human (Phase 1D).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from .channels import ChannelMasks
from .engine import OracleEngine
from .errors import OracleViolationError
from .mass_balance import DEFAULT_BALANCE_TOLERANCE, assert_mass_balance
from .transition import build_transition

# ---------------------------------------------------------------------------
# Horizon sweep + first-passage hit distribution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SweepResult:
    ai_name: str
    horizons: tuple[int, ...]
    baselines: tuple[float, ...]        # B_c(L), cumulative
    hits: tuple[float, ...]             # hit_c(t), first-passage mass at step t
    max_balance_error: float
    initial_mass: float


def run_horizon_sweep(
    engine: OracleEngine,
    ai_name: str,
    max_horizon: int,
    silenced_body_ids=None,
) -> SweepResult:
    """Run propagation to max_horizon ONCE and read off B_c(L) for
    L = 1..max_horizon as the cumulative sum of per-step hits.

    Running once and cumulating (instead of one run per L) guarantees
    exact internal consistency: sum_{t<=L} hit(t) == B_c(L) by
    construction, which is exactly the identity Phase 1C requires.
    """
    result = engine.run(ai_name, max_horizon, silenced_body_ids)
    hits = tuple(float(s.target_absorbed) for s in result.steps)
    baselines: list[float] = []
    running = 0.0
    for h in hits:
        running += h
        baselines.append(running)
    # sanity: cumulative must equal the engine's own transmission at L_max
    if abs(baselines[-1] - result.transmission) > 1e-12:
        raise OracleViolationError(
            f"{ai_name}: cumulative hits {baselines[-1]:.12f} != engine "
            f"transmission {result.transmission:.12f} (STOP condition F)"
        )
    return SweepResult(
        ai_name=ai_name,
        horizons=tuple(range(1, max_horizon + 1)),
        baselines=tuple(baselines),
        hits=hits,
        max_balance_error=result.max_balance_error,
        initial_mass=result.initial_mass,
    )


def marginal_gains(baselines) -> tuple[tuple[float, ...], tuple]:
    """Delta_c(L) = B(L) - B(L-1) with B(0)=0, and
    relative_increment_c(L) = Delta(L)/B(L) when B(L) > 0 else None."""
    deltas: list[float] = []
    prev = 0.0
    for b in baselines:
        deltas.append(float(b) - prev)
        prev = float(b)
    rel: list = [
        (d / b if b > 0 else None) for d, b in zip(deltas, baselines)
    ]
    return tuple(deltas), tuple(rel)


# ---------------------------------------------------------------------------
# Within-window capture metrics (B(20) = "20-step reference", NOT an
# asymptote — the naming is deliberate and binding)
# ---------------------------------------------------------------------------

def window_metrics(
    ai_name: str, baselines, reference_index: int = 19
) -> dict:
    """capture20_c(L) = B(L)/B(20); L90/L95/L99_window = minimal L whose
    capture20 reaches the fraction. Requires the sweep to cover at least
    20 steps (reference_index = index of L=20)."""
    if len(baselines) <= reference_index:
        raise ValueError("sweep must cover L=1..20 for window metrics")
    b20 = float(baselines[reference_index])
    if b20 <= 0:
        raise OracleViolationError(
            f"{ai_name}: B(20) == 0; window metrics undefined"
        )
    captures = [float(b) / b20 for b in baselines]
    out = {
        "ai_name": ai_name,
        "B20_reference": b20,
        "capture20": captures,
    }
    for frac, key in ((0.90, "L90_window"), (0.95, "L95_window"),
                      (0.99, "L99_window")):
        reached = None
        for L, c in enumerate(captures, start=1):
            if c >= frac:
                reached = L
                break
        out[key] = reached  # None if never reached within window (reported)
    return out


# ---------------------------------------------------------------------------
# ge5 sensitivity: same node universe, fixed endpoints, thresholded edges
# ---------------------------------------------------------------------------

def build_ge5_transition(
    edges: pl.DataFrame, node_ids: np.ndarray, threshold: int = 5
) -> tuple[csr_matrix, np.ndarray, dict]:
    """Row-normalised transition over the SAME 166,400-node index universe
    with the edge set restricted to weight >= threshold.

    Endpoint membership is NEVER redefined here: sources/targets that lose
    all edges simply become structural sinks or receive no mass — they are
    kept. The node universe comes from the caller (all-weight index)."""
    ge5 = edges.filter(pl.col("weight") >= threshold)
    P, sink_mask, qc = build_transition(ge5, node_ids)
    info = {
        "ge5_edges": int(ge5.height),
        "ge5_sink_count": int(sink_mask.sum()),
        "threshold": threshold,
        "node_universe": int(len(node_ids)),
    }
    return P, sink_mask, info


# ---------------------------------------------------------------------------
# Overlap-excluded endpoint sensitivity (Phase 1C charter, section 13)
# ---------------------------------------------------------------------------

def overlap_excluded_mask(
    masks: dict[str, ChannelMasks], overlap_body_ids: list[int],
    node_ids: np.ndarray,
) -> dict[str, ChannelMasks]:
    """Sensitivity-only variant: remove the overlap node(s) from the
    primary AND control_1 TARGET sets simultaneously (both sides, never
    one-sided). Source sets are untouched. Other channels pass through
    unchanged."""
    import copy

    if not overlap_body_ids:
        raise ValueError("overlap_excluded requires a non-empty overlap set")
    remove = np.zeros(len(node_ids), dtype=bool)
    for bid in overlap_body_ids:
        i = int(np.searchsorted(node_ids, int(bid)))
        i_clipped = min(i, len(node_ids) - 1)
        if node_ids[i_clipped] != int(bid):
            raise ValueError(f"overlap bodyId {bid} not in node index")
        remove[i] = True

    out: dict[str, ChannelMasks] = {}
    for ai, m in masks.items():
        if ai in ("primary", "control_1"):
            tgt = m.target_mask & ~remove
        else:
            tgt = m.target_mask
        # T_effective = T \ S recomputed on the new target set
        t_eff = tgt & ~m.source_mask
        out[ai] = ChannelMasks(
            ai_name=m.ai_name,
            source_mask=m.source_mask,
            target_mask=tgt,
            target_effective_mask=t_eff,
            source_n=m.source_n,
            target_n=int(tgt.sum()),
            target_effective_n=int(t_eff.sum()),
            source_target_overlap_n=int((m.source_mask & tgt).sum()),
        )
    return out


# ---------------------------------------------------------------------------
# Neutral compartment-exposure audit (official annotations only)
# ---------------------------------------------------------------------------

def audit_compartment_exposure(
    PT: csr_matrix,
    source_mask: np.ndarray,
    target_effective_mask: np.ndarray,
    sink_mask: np.ndarray,
    compartment_labels: np.ndarray,
    horizon: int,
) -> dict[str, list[float]]:
    """Per-step active signal mass aggregated by a neutral official
    annotation (superclass or somaNeuromere). Descriptive audit only —
    NOT candidate ranking, NOT functional interpretation.

    Records the active mass distribution at t=0..horizon: t=0 is the
    initial source distribution; x_t for t>=1 is the post-step active
    vector (after absorption / sink removal). Silencing is always empty
    here (audit is baseline-only).
    """
    n = len(source_mask)
    labels = np.asarray(compartment_labels, dtype=object)
    unique = sorted({str(x) for x in labels})
    pos = {lab: i for i, lab in enumerate(unique)}
    lab_idx = np.array([pos[str(x)] for x in labels], dtype=np.int64)

    x = np.zeros(n, dtype=np.float64)
    x[source_mask] = 1.0 / max(int(source_mask.sum()), 1)

    mass = np.zeros((horizon + 1, len(unique)), dtype=np.float64)
    for t in range(horizon + 1):
        counts = np.bincount(lab_idx, weights=x, minlength=len(unique))
        mass[t] = counts
        if t == horizon:
            break
        x[sink_mask] = 0.0
        y = PT @ x
        y[target_effective_mask] = 0.0
        y[sink_mask] = 0.0
        x = y
    return {"labels": unique, "mass_by_step": mass.tolist()}


# ---------------------------------------------------------------------------
# Validation helpers (STOP conditions D/E/F/G)
# ---------------------------------------------------------------------------

def validate_monotonicity(sweep: SweepResult, tolerance: float = 1e-12) -> None:
    for i in range(1, len(sweep.baselines)):
        if sweep.baselines[i] < sweep.baselines[i - 1] - tolerance:
            raise OracleViolationError(
                f"{sweep.ai_name}: baseline decreased with larger L "
                f"(B({i+1})={sweep.baselines[i]:.12f} < "
                f"B({i})={sweep.baselines[i-1]:.12f}) — STOP condition D"
            )


def validate_bounds(sweep: SweepResult, tolerance: float = 1e-12) -> None:
    for L, b in zip(sweep.horizons, sweep.baselines):
        if b < -tolerance or b > 1.0 + tolerance:
            raise OracleViolationError(
                f"{sweep.ai_name}: B({L})={b:.12f} outside [0,1] — "
                "STOP condition E"
            )


def validate_hits_sum(sweep: SweepResult, tolerance: float = 1e-12) -> None:
    running = 0.0
    for t, h in enumerate(sweep.hits, start=1):
        running += h
        if abs(running - sweep.baselines[t - 1]) > tolerance:
            raise OracleViolationError(
                f"{sweep.ai_name}: sum of hits through t={t} "
                f"({running:.12f}) != B({t}) ({sweep.baselines[t-1]:.12f}) "
                "— STOP condition F"
            )


def validate_mass_balance_ceiling(
    sweeps, phase1b_reference: float = 3.4e-16,
) -> float:
    """Max mass-balance error across all sweeps; must stay in the same
    floating-noise regime as Phase 1B (no tolerance tweaking allowed)."""
    worst = max(s.max_balance_error for s in sweeps)
    if worst > max(phase1b_reference * 10.0, DEFAULT_BALANCE_TOLERANCE):
        raise OracleViolationError(
            f"mass balance error {worst:.3e} substantially exceeds the "
            f"Phase 1B regime ({phase1b_reference:.1e}) — STOP condition G"
        )
    return worst


__all__ = [
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
