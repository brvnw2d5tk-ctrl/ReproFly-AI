"""Finite-horizon first-passage propagation.

Mathematical definition (Phase 1B charter, sections 9-12):

    For step t = 1..L:
        y        = x_t @ P                       (one propagation step)
        hit_t    = sum(y[j] for j in T_effective) (first-passage arrivals)
        y[T_effective] = 0                        (absorbed mass is removed)
        y[K]     = 0  (already applied before absorption)
        x_{t+1}  = y

    Transmission(L) = sum_{t=1..L} hit_t   in [0, 1]

Ordering inside one step (fixed, tested):
  1. mass sitting at structural sinks cannot propagate -> sink loss
  2. y = PT @ x
  3. mass ARRIVING at silenced neurons K is lost (before target absorption,
     so a silenced target never counts as success — section 12)
  4. mass arriving at T_effective is absorbed (counts as hit)
  5. mass arriving at other sinks is lost
  6. remainder stays active

Silencing (section 10-11): x0[K] = 0 at initialisation; y[K] = 0 after
every propagation. NO re-normalisation of surviving nodes, NO
redistribution of lost mass — silencing causes true signal loss.

Zero-hop success is impossible by construction: absorption only happens
after >= 1 propagation step (section 8).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from .errors import OracleViolationError
from .mass_balance import PropagationResult, StepAccount

NUMERICAL_TOLERANCE = 1e-12


def _initialise(
    x0: np.ndarray, silenced_mask: np.ndarray
) -> tuple[np.ndarray, float]:
    x = np.asarray(x0, dtype=np.float64).copy()
    removed = float(x[silenced_mask].sum())
    x[silenced_mask] = 0.0
    return x, removed


def propagate_sparse(
    PT: csr_matrix,
    x0: np.ndarray,
    target_effective_mask: np.ndarray,
    silenced_mask: np.ndarray,
    sink_mask: np.ndarray,
    horizon: int,
) -> PropagationResult:
    """Production sparse propagation. PT = P.T in CSR so one step is a
    single sparse matvec (y = PT @ x). No matrix copies are made."""
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    x, initial_removed = _initialise(x0, silenced_mask)
    initial_mass = float(x.sum())
    steps: list[StepAccount] = []
    transmission = 0.0

    for t in range(1, horizon + 1):
        active_before = float(x.sum())

        # 1. mass already sitting at sinks cannot propagate
        sink_stay = float(x[sink_mask].sum())
        x[sink_mask] = 0.0

        # 2. propagate
        y = PT @ x

        # 3. arrivals at silenced neurons are lost (before absorption)
        silenced_loss = float(y[silenced_mask].sum())
        y[silenced_mask] = 0.0

        # 4. first-passage absorption at effective targets
        hit = float(y[target_effective_mask].sum())
        y[target_effective_mask] = 0.0

        # 5. arrivals at (other) structural sinks are lost
        sink_arrival = float(y[sink_mask].sum())
        y[sink_mask] = 0.0

        active_after = float(y.sum())
        steps.append(
            StepAccount(
                t=t,
                active_before=active_before,
                sink_loss=sink_stay + sink_arrival,
                silenced_loss=silenced_loss,
                target_absorbed=hit,
                active_after=active_after,
            )
        )
        transmission += hit
        x = y

    if not np.isfinite(transmission):
        raise OracleViolationError(
            f"non-finite transmission encountered (STOP condition D): "
            f"{transmission}"
        )
    if transmission < -NUMERICAL_TOLERANCE or transmission > 1.0 + NUMERICAL_TOLERANCE:
        raise OracleViolationError(
            f"transmission outside [0, 1] (got {transmission:.12f}); "
            "unexplained signal creation or double counting (STOP condition F/H)."
        )
    transmission = min(max(transmission, 0.0), 1.0)

    return PropagationResult(
        transmission=transmission,
        initial_mass=initial_mass,
        initial_silenced_source_mass=initial_removed,
        steps=tuple(steps),
    )


def propagate_dense(
    W: np.ndarray,
    x0: np.ndarray,
    target_effective_mask: np.ndarray,
    silenced_mask: np.ndarray,
    sink_mask: np.ndarray,
    horizon: int,
) -> PropagationResult:
    """INDEPENDENT dense reference implementation — toy graphs only.

    Deliberately written with explicit loops and its own row-normalisation
    so that a direction/transpose bug in the sparse engine cannot be
    mirrored here. Must agree with propagate_sparse to rtol<=1e-10,
    atol<=1e-12 on every toy graph.
    """
    n = W.shape[0]
    Pn = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        rs = 0.0
        for j in range(n):
            rs += W[i, j]
        if rs > 0.0:
            for j in range(n):
                Pn[i, j] = W[i, j] / rs

    x = np.asarray(x0, dtype=np.float64).copy()
    initial_removed = float(x[silenced_mask].sum())
    x[silenced_mask] = 0.0
    initial_mass = float(x.sum())
    steps: list[StepAccount] = []
    transmission = 0.0

    for t in range(1, horizon + 1):
        active_before = float(x.sum())

        sink_stay = 0.0
        for i in range(n):
            if sink_mask[i]:
                sink_stay += x[i]
                x[i] = 0.0

        y = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if x[i] != 0.0:
                for j in range(n):
                    if Pn[i, j] != 0.0:
                        y[j] += x[i] * Pn[i, j]

        silenced_loss = 0.0
        for i in range(n):
            if silenced_mask[i] and y[i] != 0.0:
                silenced_loss += y[i]
                y[i] = 0.0

        hit = 0.0
        for i in range(n):
            if target_effective_mask[i] and y[i] != 0.0:
                hit += y[i]
                y[i] = 0.0

        sink_arrival = 0.0
        for i in range(n):
            if sink_mask[i] and y[i] != 0.0:
                sink_arrival += y[i]
                y[i] = 0.0

        active_after = float(y.sum())
        steps.append(
            StepAccount(
                t=t,
                active_before=active_before,
                sink_loss=sink_stay + sink_arrival,
                silenced_loss=silenced_loss,
                target_absorbed=hit,
                active_after=active_after,
            )
        )
        transmission += hit
        x = y

    return PropagationResult(
        transmission=transmission,
        initial_mass=initial_mass,
        initial_silenced_source_mass=initial_removed,
        steps=tuple(steps),
    )


__all__ = [
    "propagate_sparse",
    "propagate_dense",
    "NUMERICAL_TOLERANCE",
]
