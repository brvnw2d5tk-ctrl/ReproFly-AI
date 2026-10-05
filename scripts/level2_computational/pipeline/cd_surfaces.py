"""Phase 2C.1 — surfaces (information-surface adapters).

POLICY-FACING CONTRACT (SPEC v1.0 §12.1):
a LocalSurface exposes to the policy ONLY:
  * eligible_ids()            -> tuple of eligible blind ids (lexical order)
  * inspect(blind_id)         -> InspectionPayload (top-10 lists + hops)
  * silence(blind_id)         -> SilencingReadout (primary.relative_loss)
  * budget counters (inspections_used / charged_attempts)

Implementations:
  * SyntheticSurface — toy graph, deterministic, for B-G1 tests.
  * RealSurface (private side, same trust domain as the Phase 2A MCP
    tool backend) — loads frozen blind-keyed assets + the real oracle
    engine. The policy NEVER sees real_type / GT / blind key through
    this module: the real-surface reader uses an explicit column
    allowlist and never touches the `real_type` column (unit-tested in
    the information-firewall contract).

No filesystem-order dependence: every set/list exposed to the policy is
lexically sorted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Protocol

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_PRIVATE = REPO_ROOT / "derived_private" / "benchmark"

# SPEC §12.2 — real-surface reader column allowlist (real_type NEVER read).
INSPECT_COLUMNS_ALLOWLIST = (
    "blind_id",
    "min_hops_from_primary_source",
    "min_hops_to_primary_target",
    "top_incoming_ids",
    "top_incoming_weights",
    "top_outgoing_ids",
    "top_outgoing_weights",
)


@dataclass(frozen=True)
class InspectionPayload:
    blind_id: str
    min_hops_from_primary_source: Optional[int]
    min_hops_to_primary_target: Optional[int]
    top_incoming: tuple[tuple[str, float], ...]   # (blind_id, weight)
    top_outgoing: tuple[tuple[str, float], ...]

    def best_fraction(self, other_blind_id: str) -> Optional[float]:
        """Max-aggregated delivered salience of other_blind_id (SPEC 4b).

        Returns the incoming fraction, outgoing fraction, or the max of
        both if the id appears in both lists; None if in neither.
        PREDEFINED DETERMINISTIC DIRECTIONAL-SALIENCE AGGREGATION
        CONVENTION (max of the two directional fractions).
        """
        inc = out = None
        for bid, w in self.top_incoming:
            if bid == other_blind_id:
                inc = w
                break
        for bid, w in self.top_outgoing:
            if bid == other_blind_id:
                out = w
                break
        if inc is None and out is None:
            return None
        if inc is None:
            return out
        if out is None:
            return inc
        return inc if inc >= out else out


@dataclass(frozen=True)
class SilencingReadout:
    blind_id: str
    primary_relative_loss: float


class BudgetExhausted(Exception):
    """Raised by surfaces when the charged-attempt cap is hit."""


class LocalSurface(Protocol):
    def eligible_ids(self) -> tuple[str, ...]: ...
    def inspect(self, blind_id: str) -> InspectionPayload: ...
    def silence(self, blind_id: str) -> SilencingReadout: ...
    @property
    def inspections_used(self) -> int: ...
    @property
    def charged_attempts(self) -> int: ...
    @property
    def inspection_budget(self) -> int: ...
    @property
    def perturbation_budget(self) -> int: ...


# ---------------------------------------------------------------------------
# Synthetic surface (B-G1 tests ONLY — never the real graph)
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSurface:
    """Toy delivered-surface + toy oracle.

    neighbors: blind_id -> (incoming tuple, outgoing tuple) of
    (neighbor_id, edge_weight) pairs; weights must be > 0.
    oracle: blind_id -> primary_relative_loss (finite float); defaults
    to a deterministic injective function of the id if omitted.
    eligible: the universe (defaults to all ids mentioned anywhere);
    extra eligible ids with no neighbors are allowed (isolate seeds).
    """

    neighbors: dict[str, tuple[tuple[tuple[str, float], ...],
                                tuple[tuple[str, float], ...]]]
    oracle_losses: dict[str, float] = field(default_factory=dict)
    eligible: tuple[str, ...] | None = None
    inspection_budget: int = 600
    perturbation_budget: int = 300

    _inspections_used: int = field(init=False, default=0)
    _charged_attempts: int = field(init=False, default=0)
    _inspected: set = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        # Universe semantics mirror the real interface: the silenceable
        # universe is EXACTLY the eligible set; delivered neighbor lists
        # may reference blind ids OUTSIDE the universe (ineligible ids
        # such as endpoint-associated types) — policies must filter
        # them, and silence() rejects them.
        if self.eligible is not None:
            universe = set(self.eligible)
            unknown_keys = set(self.neighbors) - universe
            if unknown_keys:
                raise ValueError(
                    f"synthetic neighbors keys outside eligible "
                    f"universe: {sorted(unknown_keys)}")
        else:
            universe = set(self.neighbors.keys())
            for inc, out in self.neighbors.values():
                universe.update(b for b, _w in inc)
                universe.update(b for b, _w in out)
        self._universe: tuple[str, ...] = tuple(sorted(universe))
        for bid, (inc, out) in self.neighbors.items():
            for lst in (inc, out):
                seen: set[str] = set()
                for b, w in lst:
                    if w <= 0:
                        raise ValueError(f"non-positive weight for {bid}->{b}")
                    if b in seen:
                        raise ValueError(f"duplicate neighbor {b} in {bid} list")
                    seen.add(b)

    # -- LocalSurface protocol ------------------------------------------------
    def eligible_ids(self) -> tuple[str, ...]:
        return self._universe

    @property
    def inspections_used(self) -> int:
        return self._inspections_used

    @property
    def charged_attempts(self) -> int:
        return self._charged_attempts

    def _payload(self, blind_id: str) -> InspectionPayload:
        inc, out = self.neighbors.get(blind_id, ((), ()))
        return InspectionPayload(
            blind_id=blind_id,
            min_hops_from_primary_source=2,
            min_hops_to_primary_target=2,
            top_incoming=tuple(inc),
            top_outgoing=tuple(out),
        )

    def inspect(self, blind_id: str) -> InspectionPayload:
        if blind_id not in self._universe:
            raise ValueError(
                f"inspect of non-eligible id {blind_id} (INVALID_TARGET "
                "semantics; C/D policies never do this)")
        if blind_id in self._inspected:
            # C/D policies never re-inspect by construction; a repeat
            # would indicate a policy bug — surface mirrors the frozen
            # charging rule (ALREADY_INSPECTED costs 1, no new info).
            self._inspections_used += 1
            return self._last_payload
        if self._inspections_used >= self.inspection_budget:
            return self._last_payload  # unreachable: policies check first
        self._inspections_used += 1
        self._inspected.add(blind_id)
        self._last_payload = self._payload(blind_id)
        return self._last_payload

    def silence(self, blind_id: str) -> SilencingReadout:
        if self._charged_attempts >= self.perturbation_budget:
            raise BudgetExhausted
        self._charged_attempts += 1
        if blind_id not in self._universe:
            raise ValueError(f"silencing non-eligible id {blind_id}")
        if blind_id in self.oracle_losses:
            r = float(self.oracle_losses[blind_id])
        else:
            r = (int(blind_id.split("-")[1]) % 977 + 1) / 1000.0
        if r != r or r in (float("inf"), float("-inf")):  # NaN/inf guard
            raise ValueError(f"non-finite toy loss for {blind_id}")
        return SilencingReadout(blind_id=blind_id, primary_relative_loss=r)


# ---------------------------------------------------------------------------
# Real surface (PRIVATE side; B-G6a execution ONLY — not used in tests)
# ---------------------------------------------------------------------------

class RealSurface:
    """Frozen-asset surface for the real MaleCNS run (SPEC §13 pins).

    Same trust-domain design as the Phase 2A MCP tool backend: this
    object holds the blind->real mapping and the real oracle engine,
    and exposes to the policy only the LocalSurface protocol (blind ids,
    inspection payloads, primary.relative_loss, budgets). The
    `real_type` column of the inspect-features asset is NEVER read
    (column allowlist above; unit-tested).
    """

    ASSET_PINS = {
        "scientist_eligible_blind_ids_v1.json":
            "462cc43cad4fabdddab62d19fb02d129f530d99c00bf2a3208e9c0acb61f50b9",
        "candidate_universe_v1.parquet":
            "a2ec9c251f3e8981e79571b8dd69d05cf1b0a4db0ec5cf15546844b0d9b1f822",
        "inspect_features_private.parquet":
            "7ba103783fbab07393a35e11d8b48a61db0b825f0a4e716a3140d6f67ea857f1",
        "oracle_baseline_primary_L10_v1.json":
            "d1c2e94b70be1e2a4a5f7d151d1c24b5e22fdd53bba375557ea36f078e3b76d4",
    }

    def __init__(self, verify_pins: bool = True) -> None:
        import hashlib

        import polars as pl

        # HOTFIX 001 (human decision, plan B, 2026-09-26): the eligible universe
        # is loaded from candidate_universe_v1.parquet — the SAME
        # authoritative source and field semantics as the Phase 2A
        # formal harness (formal_tools.py: eligible==True rows,
        # blind_id column). scientist_eligible_blind_ids_v1.json is
        # retained ONLY as a cross-check (its top level is structured
        # metadata, not a bare id array — misreading it as one was
        # failure 001).
        universe_path = BENCH_PRIVATE / "candidate_universe_v1.parquet"
        ids_path = BENCH_PRIVATE / "scientist_eligible_blind_ids_v1.json"
        feats_path = BENCH_PRIVATE / "inspect_features_private.parquet"
        base_path = (REPO_ROOT / "derived_private" / "oracle" / "baselines"
                     / "oracle_baseline_primary_L10_v1.json")

        if verify_pins:
            for path in (universe_path, ids_path, feats_path, base_path):
                name = path.name
                if name not in self.ASSET_PINS:
                    raise RuntimeError(f"unpinned asset {path}")
                got = hashlib.sha256(path.read_bytes()).hexdigest()
                if got != self.ASSET_PINS[name]:
                    raise RuntimeError(
                        f"asset pin mismatch for {name}: {got} != "
                        f"{self.ASSET_PINS[name]} — STOP (SPEC §12.3)")

        universe = pl.read_parquet(universe_path)
        self._eligible: tuple[str, ...] = tuple(sorted(
            universe.filter(pl.col("eligible"))["blind_id"].to_list()))
        self._eligible_set = set(self._eligible)

        # cross-check (not the runtime source): the JSON's blind_ids
        # must equal the parquet-derived universe
        ids_payload = json.loads(ids_path.read_text(encoding="utf-8"))
        json_ids = set(ids_payload["blind_ids"])
        if json_ids != self._eligible_set:
            raise RuntimeError(
                "eligible-universe cross-check FAILED: parquet "
                f"(candidate_universe_v1) and scientist_eligible_blind_"
                "ids_v1.json[blind_ids] disagree — STOP")

        # SPEC §12.2 — allowlist read; real_type never loaded.
        feats = pl.read_parquet(
            feats_path, columns=list(INSPECT_COLUMNS_ALLOWLIST))
        self._feats: dict[str, dict] = {
            r["blind_id"]: r for r in feats.to_dicts()
        }
        missing = self._eligible_set - set(self._feats)
        if missing:
            raise RuntimeError(
                f"inspect-features missing {len(missing)} eligible ids")

        payload = json.loads(base_path.read_text(encoding="utf-8"))
        b_primary = float(payload["baselines"]["10"]["primary"])
        if not (b_primary > 0.0):
            raise RuntimeError("frozen B_primary(L=10) must be > 0")

        from reprofly.oracle.engine import OracleEngine
        self._engine = OracleEngine.from_derived_dir(
            REPO_ROOT / "derived_private" / "oracle")

        # private-side blind -> real mapping (same asset as the Phase
        # 2A formal harness backend)
        blind_map = json.loads(
            (REPO_ROOT / "derived_private" / "blinding"
             / "blind_type_map_private.json").read_text(encoding="utf-8"))
        self._blind_to_real: dict[str, str] = blind_map["blind_to_real"]

        # CORRECTION 003 (human decision, 2026-09-26): universe-wide Oracle-
        # domain membership rule. For every candidate type:
        #   TYPE_MEMBERS_ORACLE(t) = TYPE_MEMBERS_RAW(t) ∩ NODE_INDEX
        # perturbations silence exactly TYPE_MEMBERS_ORACLE(t).
        # Deterministic ordering (sorted), intersection precomputed at
        # construction (before any trajectory execution), zero-member
        # types trigger a fail-fast preflight STOP, and NO per-type
        # runtime special case exists anywhere.
        neuron_flags = pl.read_parquet(
            BENCH_PRIVATE / "universe_neuron_flags_private.parquet")
        elig_flags = neuron_flags.filter(pl.col("primary_path_eligible"))
        self._raw_members_by_type: dict[str, list[int]] = {
            r["type"]: r["bodyIds"]
            for r in elig_flags.group_by("type")
            .agg(pl.col("bodyId").sort().alias("bodyIds"))
            .to_dicts()
        }
        node_index = pl.read_parquet(
            REPO_ROOT / "derived_private" / "oracle"
            / "node_index_private.parquet")
        self._oracle_nodes = frozenset(node_index["bodyId"].to_list())
        self._neurons_by_type: dict[str, list[int]] = {
            real: sorted(b for b in body_ids
                         if b in self._oracle_nodes)
            for real, body_ids in self._raw_members_by_type.items()
        }
        # fail-fast preflight: every candidate type must resolve to at
        # least one strict-oracle node (audit: 0 zero-member types)
        zero = [real for real, ids in self._neurons_by_type.items()
                if not ids]
        if zero:
            raise RuntimeError(
                f"{len(zero)} candidate types resolve to ZERO oracle-"
                "domain members — preflight STOP (SPEC v1.0.1 §0.5; "
                "the human operator required immediate STOP; do not alter the "
                "candidate universe)")

        self.inspection_budget = 600
        self.perturbation_budget = 300
        self._inspections_used = 0
        self._charged_attempts = 0
        self._inspected: set[str] = set()

    # -- LocalSurface protocol ------------------------------------------------
    def eligible_ids(self) -> tuple[str, ...]:
        return self._eligible

    @property
    def inspections_used(self) -> int:
        return self._inspections_used

    @property
    def charged_attempts(self) -> int:
        return self._charged_attempts

    def inspect(self, blind_id: str) -> InspectionPayload:
        if self._inspections_used >= self.inspection_budget:
            raise RuntimeError("inspection budget exhausted (policy bug)")
        self._inspections_used += 1
        self._inspected.add(blind_id)
        r = self._feats[blind_id]
        return InspectionPayload(
            blind_id=blind_id,
            min_hops_from_primary_source=r[
                "min_hops_from_primary_source"],
            min_hops_to_primary_target=r["min_hops_to_primary_target"],
            top_incoming=tuple(
                zip(r["top_incoming_ids"], r["top_incoming_weights"])),
            top_outgoing=tuple(
                zip(r["top_outgoing_ids"], r["top_outgoing_weights"])),
        )

    def silence(self, blind_id: str) -> SilencingReadout:
        if self._charged_attempts >= self.perturbation_budget:
            raise BudgetExhausted
        self._charged_attempts += 1
        real = self._blind_to_real[blind_id]
        body_ids = self._neurons_by_type.get(real, [])
        if not body_ids:
            # unreachable after the CORRECTION-003 preflight (every
            # candidate type resolves to >=1 oracle-domain member);
            # retained as a defensive fail-fast
            raise RuntimeError(
                f"zero oracle-domain members for universe type "
                f"{blind_id} — preflight should have caught this; STOP")
        readout = self._engine.compare(
            "primary", 10, silenced_body_ids=body_ids)
        r = readout.relative_loss
        if r is None or r != r or r in (float("inf"), float("-inf")):
            # SPEC §2.2 — STOP rule: no field substitution.
            raise RuntimeError(
                f"primary.relative_loss non-finite/None for {blind_id} — "
                "SPEC v1.0 §2.2 STOP rule; escalate to the human operator")
        return SilencingReadout(blind_id=blind_id, primary_relative_loss=float(r))
