"""Phase 2C.1 — C/D policies (SPEC v1.0 exact state machines).

Implements the seven closed-family variants:
  C1 (variant 1) — random-order inspect-and-perturb
  C2 (variant 2) — greedy local expansion (POST-HOC BEHAVIOR-MATCHED
                   LOCAL NON-ADAPTIVE BASELINE)
  C3 (variant 3) — global-frontier greedy
  D1 (variant 4) — record-minimum restart (POST-HOC BEHAVIOR-
                   DISTILLATION LOCAL ADAPTIVE BASELINE)
  D2a (variant 5) / D2b (variant 6) — percentile-threshold gates
  D3 (variant 7) — map-then-mine (K=100) then D1

The policy layer receives ONLY the LocalSurface protocol (blind ids,
inspection payloads, primary.relative_loss, budgets) — never real
names, GT, the blind key, or the full graph (SPEC §12.1; tested in
the information-firewall contract).

RNG (SPEC §5): SeedSequence(entropy=20260926, spawn_key=(variant_id,
rep_index)) -> Generator(PCG64); seed draws via
rng.choice(sorted_pool, size=None, replace=False); permutations via
rng.permutation applied to an index arange over the lexically sorted
eligible array (implementation lock §18: deterministic, no
object-array semantics). No other randomness.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cd_surfaces import LocalSurface

MASTER_SEED = 20260926
VARIANT_IDS = {"C1": 1, "C2": 2, "C3": 3, "D1": 4,
               "D2a": 5, "D2b": 6, "D3": 7}
PERTURBATION_BUDGET = 300
INSPECTION_BUDGET = 600
D3_MAPPING_K = 100


def make_rng(variant: str, rep_index: int) -> np.random.Generator:
    """SPEC §5 spawn strategy (implementation lock §18.1)."""
    ss = np.random.SeedSequence(
        entropy=MASTER_SEED,
        spawn_key=(VARIANT_IDS[variant], rep_index))
    return np.random.Generator(np.random.PCG64(ss))


def draw_seed(rng: np.random.Generator, sorted_pool: list[str]) -> str:
    """STEP 1 uniform draw (implementation lock §18.1)."""
    return str(rng.choice(sorted_pool, size=None, replace=False))


def permuted(rng: np.random.Generator, sorted_pool: list[str]) -> list[str]:
    """Full-universe permutation (implementation lock §18.2): index
    permutation over the lexically sorted pool — no object-array
    semantics, byte-identical replay guaranteed."""
    order = rng.permutation(len(sorted_pool))
    return [sorted_pool[i] for i in order]


# ---------------------------------------------------------------------------
# shared engine (C2 local expansion; D1 inherits it EXACTLY)
# ---------------------------------------------------------------------------

def _candidate_score(payload, other_id: str, eligible_set: set[str],
                     valid_perturbed: set[str]) -> Optional[float]:
    """Score `other_id` from ONE inspected node's delivered lists.

    None if not a delivered neighbor / not eligible / already perturbed;
    else the max-aggregated fraction (SPEC 4b convention).
    """
    if other_id in valid_perturbed or other_id not in eligible_set:
        return None
    return payload.best_fraction(other_id)


def _select_from_entries(entries: list[tuple[float, str]]) -> Optional[str]:
    """Order by (score DESC, blind_id ASC); return the first id."""
    if not entries:
        return None
    entries.sort(key=lambda e: (-e[0], e[1]))
    return entries[0][1]


def _neighbor_ids(payload) -> set[str]:
    ids = set(b for b, _ in payload.top_incoming)
    ids.update(b for b, _ in payload.top_outgoing)
    return ids


def _center_local_list(payload, eligible_set, valid_perturbed):
    entries = []
    for other in _neighbor_ids(payload):
        score = _candidate_score(payload, other, eligible_set,
                                 valid_perturbed)
        if score is not None:
            entries.append((score, other))
    return entries


def _frontier_list(inspected_payloads: dict, eligible_set,
                   valid_perturbed):
    entries = []
    for payload in inspected_payloads.values():
        for other in _neighbor_ids(payload):
            score = _candidate_score(payload, other, eligible_set,
                                     valid_perturbed)
            if score is not None:
                entries.append((score, other))
    return entries


class _RunState:
    """Shared bookkeeping for all C/D variants (SPEC §1.1).

    Holds the policy-visible state and implements STEP 2/3/4 exactly
    once, so D1 provably inherits the C2 engine.
    """

    def __init__(self, surface: LocalSurface, events):
        self.surface = surface
        self.events = events
        self.eligible_list: list[str] = list(surface.eligible_ids())
        self.eligible_set: set[str] = set(self.eligible_list)
        self.inspected: dict[str, object] = {}   # blind_id -> payload
        self.valid_perturbed: set[str] = set()
        self.charged_attempts = 0

    @property
    def inspections_used(self) -> int:
        return self.surface.inspections_used

    # STEP 2 — inspect the center (budget-conditional, idempotent)
    def step2_inspect(self, center: str) -> None:
        if center in self.inspected:
            return
        if self.inspections_used >= INSPECTION_BUDGET:
            return  # skip rule (SPEC §1.2 STEP 2)
        self.inspected[center] = self.surface.inspect(center)
        self.events.inspect(center, self.inspections_used)

    # STEP 3 — perturb the center (event emitted by the RUNNER after
    # the D-gate is decided, so restart_flag is correct; C runners
    # emit immediately with flag=False)
    def step3_perturb(self, center: str) -> float:
        readout = self.surface.silence(center)
        self.charged_attempts += 1
        self.valid_perturbed.add(center)
        return readout.primary_relative_loss

    def emit_silence(self, center: str, r: float,
                     restart_flag: bool) -> None:
        self.events.silence(center, self.charged_attempts, r,
                            restart_flag)

    # STEP 4 — select the next center. use_center_local=False gives the
    # C3 frontier-only rule; D engines call it with the perturbed
    # center (C2 rule) whenever the gate did not fire.
    def step4_select(self, center: Optional[str],
                     *, use_center_local: bool = True) -> Optional[str]:
        if use_center_local and center is not None \
                and center in self.inspected:
            local = _center_local_list(
                self.inspected[center], self.eligible_set,
                self.valid_perturbed)
            nxt = _select_from_entries(local)
            if nxt is not None:
                return nxt
        frontier = _frontier_list(
            self.inspected, self.eligible_set, self.valid_perturbed)
        return _select_from_entries(frontier)   # None -> STEP 4f restart

    def reseed(self, rng: np.random.Generator, reason: str,
               first: bool) -> Optional[str]:
        """STEP 1 — uniform draw + restart/seed events (SPEC §15 order)."""
        pool = sorted(self.eligible_set - self.valid_perturbed)
        if not pool:
            return None
        center = draw_seed(rng, pool)
        self.events.restart(
            center, "initial_seed" if first else reason)
        self.events.seed(center, len(pool))
        return center


# ---------------------------------------------------------------------------
# C-family
# ---------------------------------------------------------------------------

def run_c2(surface: LocalSurface, rng: np.random.Generator,
           events) -> None:
    """C2 greedy local expansion — SPEC §1.2 cycle."""
    st = _RunState(surface, events)
    center: Optional[str] = None
    first = True

    while st.charged_attempts < PERTURBATION_BUDGET:
        if center is None:                       # STEP 1
            center = st.reseed(rng, "frontier_exhausted", first)
            if center is None:
                break                            # universe exhaustion
            first = False

        st.step2_inspect(center)                 # STEP 2
        r = st.step3_perturb(center)             # STEP 3
        st.emit_silence(center, r, False)

        if st.charged_attempts >= PERTURBATION_BUDGET:
            break

        center = st.step4_select(center)         # STEP 4 (4f -> None)

    events.end(st)


def run_c1(surface: LocalSurface, rng: np.random.Generator,
           events) -> None:
    """C1 random-order inspect-and-perturb (SPEC §10)."""
    st = _RunState(surface, events)
    for blind_id in permuted(rng, st.eligible_list):
        if st.charged_attempts >= PERTURBATION_BUDGET:
            break
        st.step2_inspect(blind_id)
        r = st.step3_perturb(blind_id)
        st.emit_silence(blind_id, r, False)
    events.end(st)


def run_c3(surface: LocalSurface, rng: np.random.Generator,
           events) -> None:
    """C3 global-frontier greedy (SPEC §10): always the STEP 4e rule."""
    st = _RunState(surface, events)
    center: Optional[str] = None
    first = True

    while st.charged_attempts < PERTURBATION_BUDGET:
        if center is None:                       # STEP 1
            center = st.reseed(rng, "frontier_exhausted", first)
            if center is None:
                break
            first = False

        st.step2_inspect(center)
        r = st.step3_perturb(center)
        st.emit_silence(center, r, False)

        if st.charged_attempts >= PERTURBATION_BUDGET:
            break

        center = st.step4_select(None)           # never center-local

    events.end(st)


# ---------------------------------------------------------------------------
# D-family (local engine = C2 EXACTLY + adaptive gate at STEP 3.5)
# ---------------------------------------------------------------------------

def _d_engine(surface: LocalSurface, rng: np.random.Generator, events,
              *, gate, gate_reason: str) -> None:
    """D1/D2a/D2b common runner.

    gate(r_t, history) -> bool decides RESTART after each valid
    perturbation (SPEC §2.1 STEP 3.5 / §10 D2 definitions). When the
    gate fires, STEP 4 is skipped ENTIRELY (frontier not consulted).
    """
    st = _RunState(surface, events)
    center: Optional[str] = None
    first = True
    history: list[float] = []

    while st.charged_attempts < PERTURBATION_BUDGET:
        if center is None:                       # STEP 1
            center = st.reseed(rng, gate_reason, first)
            if center is None:
                break
            first = False

        st.step2_inspect(center)                 # STEP 2
        r = st.step3_perturb(center)             # STEP 3
        history.append(r)

        restart_flag = gate(r, history)          # STEP 3.5
        st.emit_silence(center, r, restart_flag)

        if st.charged_attempts >= PERTURBATION_BUDGET:
            break

        if restart_flag:
            center = None                        # override STEP 4
            continue

        center = st.step4_select(center)         # STEP 4 (C2 engine)

    events.end(st)


def run_d1(surface: LocalSurface, rng: np.random.Generator,
           events) -> None:
    """D1 record-minimum restart (SPEC §2.1).

    t==1 -> restart; t>1 -> restart iff r_t < min(r_1..r_{t-1})
    (strict <, raw float64, ties are not records).
    """
    def gate(r_t: float, history: list[float]) -> bool:
        prior = history[:-1]
        if not prior:
            return True                          # t == 1
        return r_t < min(prior)
    _d_engine(surface, rng, events, gate=gate, gate_reason="record_minimum")


def run_d2a(surface: LocalSurface, rng: np.random.Generator,
           events) -> None:
    """D2a — percentile gate p=0.05 (SPEC §10; numpy linear quantile)."""
    def gate(r_t: float, history: list[float]) -> bool:
        prior = history[:-1]
        if not prior:
            return True
        return r_t < float(np.quantile(prior, 0.05))
    _d_engine(surface, rng, events, gate=gate, gate_reason="percentile_gate")


def run_d2b(surface: LocalSurface, rng: np.random.Generator,
           events) -> None:
    """D2b — percentile gate p=0.10 (SPEC §10)."""
    def gate(r_t: float, history: list[float]) -> bool:
        prior = history[:-1]
        if not prior:
            return True
        return r_t < float(np.quantile(prior, 0.10))
    _d_engine(surface, rng, events, gate=gate, gate_reason="percentile_gate")


def run_d3(surface: LocalSurface, rng: np.random.Generator,
           events) -> None:
    """D3 map-then-mine (SPEC §10): K=100 mapping inspections in own
    permutation order (no perturbations), then D1 with the mapped
    lists available to STEP 4e from the start."""
    st = _RunState(surface, events)
    for blind_id in permuted(rng, st.eligible_list)[:D3_MAPPING_K]:
        if st.inspections_used >= INSPECTION_BUDGET:
            break
        st.step2_inspect(blind_id)

    # ---- D1 loop on the same surface state (record-minimum gate) ----
    center: Optional[str] = None
    first = True
    history: list[float] = []

    while st.charged_attempts < PERTURBATION_BUDGET:
        if center is None:
            center = st.reseed(rng, "record_minimum", first)
            if center is None:
                break
            first = False

        st.step2_inspect(center)
        r = st.step3_perturb(center)
        history.append(r)

        prior = history[:-1]
        restart_flag = (not prior) or r < min(prior)
        st.emit_silence(center, r, restart_flag)

        if st.charged_attempts >= PERTURBATION_BUDGET:
            break

        if restart_flag:
            center = None
            continue

        center = st.step4_select(center)

    events.end(st)


RUNNERS = {
    "C1": run_c1, "C2": run_c2, "C3": run_c3,
    "D1": run_d1, "D2a": run_d2a, "D2b": run_d2b, "D3": run_d3,
}
