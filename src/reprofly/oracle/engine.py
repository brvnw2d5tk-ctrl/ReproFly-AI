"""OracleEngine — the production structural transmission engine.

Efficiency contract (Phase 1B charter, section 25):
  * the CSR transition matrix is built ONCE; silencing is implemented via
    boolean masks (active signal on K = 0), so a future silencing run NEVER
    copies the 25.6M-edge matrix
  * every propagation step is a single sparse matvec (y = PT @ x) on CPU

Scientist-visibility contract (section 15): engine outputs reference
channels ONLY by their AI-facing names (primary / control_1 / control_2 /
control_3). Real anatomy lives exclusively in the private registry.

Phase 1B execution boundary (section 22): the engine may be exercised on
TOY graphs in tests and on the real graph for baseline calibration and a
synthetic runtime benchmark; NO biological cell-type or neuron silencing
analysis is executed in this phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz

from .channels import CHANNEL_AI_NAMES, ChannelMasks, build_channel_masks
from .errors import OracleViolationError
from .first_passage import NUMERICAL_TOLERANCE, propagate_sparse
from .mass_balance import PropagationResult, assert_mass_balance
from .schemas import SilencingReadoutModel, TransmissionResultModel
from .transition import assert_sparse_csr


@dataclass(frozen=True)
class EngineInfo:
    n_nodes: int
    n_edges: int
    sink_count: int
    channels: tuple[str, ...]


class OracleEngine:
    def __init__(
        self,
        node_ids: np.ndarray,
        P: csr_matrix,
        channel_masks: dict[str, ChannelMasks],
    ):
        assert_sparse_csr(P)
        if P.shape != (len(node_ids), len(node_ids)):
            raise ValueError("node index / matrix shape mismatch")
        missing = set(CHANNEL_AI_NAMES) - set(channel_masks)
        # toy engines may define a subset; production requires all four
        self._require_all = not missing
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.P = P
        self.PT = P.T.tocsr()          # single precomputed transpose
        self.PT.sort_indices()
        row_sums = np.asarray(P.sum(axis=1)).ravel()
        self.sink_mask = row_sums == 0.0
        self.channels = dict(channel_masks)

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------

    @property
    def info(self) -> EngineInfo:
        return EngineInfo(
            n_nodes=len(self.node_ids),
            n_edges=int(self.P.nnz),
            sink_count=int(self.sink_mask.sum()),
            channels=tuple(sorted(self.channels)),
        )

    def channel_summary(self, ai_name: str) -> dict:
        """AI-facing channel summary — counts only, no anatomy."""
        m = self._channel(ai_name)
        return {
            "channel": m.ai_name,
            "source_n": m.source_n,
            "target_n": m.target_n,
            "target_effective_n": m.target_effective_n,
            "source_target_overlap_n": m.source_target_overlap_n,
        }

    # ------------------------------------------------------------------
    # propagation runs
    # ------------------------------------------------------------------

    def _channel(self, ai_name: str) -> ChannelMasks:
        try:
            return self.channels[ai_name]
        except KeyError:
            raise ValueError(
                f"unknown channel {ai_name!r}; available: "
                f"{sorted(self.channels)} (AI-facing names only)"
            ) from None

    def _silenced_mask(self, silenced_body_ids) -> np.ndarray:
        m = np.zeros(len(self.node_ids), dtype=bool)
        if silenced_body_ids is None:
            return m
        try:
            ids = [int(i) for i in silenced_body_ids]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "silencing requires an explicit iterable of bodyIds "
                "(integers). Cell-type names or annotation strings are not "
                "accepted anywhere in the oracle."
            ) from exc
        if ids:
            arr = np.asarray(sorted(set(ids)), dtype=np.int64)
            idx = np.searchsorted(self.node_ids, arr)
            if not np.array_equal(self.node_ids[idx], arr):
                raise ValueError(
                    "silenced bodyIds not present in the node index — "
                    "silencing accepts bodyIds only"
                )
            m[idx] = True
        return m

    def _x0(self, m: ChannelMasks) -> np.ndarray:
        if m.source_n == 0:
            raise ValueError(f"channel {m.ai_name!r} has an empty source set")
        x0 = np.zeros(len(self.node_ids), dtype=np.float64)
        x0[m.source_mask] = 1.0 / m.source_n   # equal source mass (section 7)
        return x0

    def run(
        self,
        ai_name: str,
        horizon: int,
        silenced_body_ids=None,
    ) -> PropagationResult:
        """One first-passage propagation run (baseline if no silencing)."""
        m = self._channel(ai_name)
        silenced = self._silenced_mask(silenced_body_ids)
        # section 12: silenced targets cannot count as arrival
        t_eff = m.target_effective_mask & ~silenced
        result = propagate_sparse(
            PT=self.PT,
            x0=self._x0(m),
            target_effective_mask=t_eff,
            silenced_mask=silenced,
            sink_mask=self.sink_mask,
            horizon=horizon,
        )
        assert_mass_balance(result)
        return result

    def baseline(self, ai_name: str, horizon: int) -> PropagationResult:
        return self.run(ai_name, horizon)

    def result_model(self, ai_name: str, horizon: int, result: PropagationResult) -> TransmissionResultModel:
        return TransmissionResultModel(
            channel=ai_name,
            horizon=horizon,
            transmission=result.transmission,
            initial_mass=result.initial_mass,
            initial_silenced_source_mass=result.initial_silenced_source_mass,
            max_balance_error=result.max_balance_error,
            n_steps=len(result.steps),
        )

    # ------------------------------------------------------------------
    # section-21 comparison contract (future run_silencing readout)
    # ------------------------------------------------------------------

    def compare(
        self,
        ai_name: str,
        horizon: int,
        silenced_body_ids,
    ) -> SilencingReadoutModel:
        """Baseline-vs-silenced comparison for ONE channel.

        Returns independent channel readouts only — no composite
        reproductive score of any kind (section 16).
        """
        base = self.baseline(ai_name, horizon)
        sil = self.run(ai_name, horizon, silenced_body_ids)
        B, S = base.transmission, sil.transmission
        if S > B + NUMERICAL_TOLERANCE:
            raise OracleViolationError(
                f"channel {ai_name!r}: silenced transmission {S:.12f} > "
                f"baseline {B:.12f} (STOP condition H — silencing must not "
                "increase transmission)"
            )
        baseline_zero = B == 0.0
        readout = SilencingReadoutModel(
            channel=ai_name,
            horizon=horizon,
            status="BASELINE_ZERO" if baseline_zero else "OK",
            baseline=B,
            silenced=S,
            absolute_change=S - B,
            retention=None if baseline_zero else S / B,
            relative_loss=None if baseline_zero else 1.0 - S / B,
            baseline_zero=baseline_zero,
        )
        return readout

    # ------------------------------------------------------------------
    # persistence (Phase 1C will reload these PRIVATE artefacts)
    # ------------------------------------------------------------------

    @classmethod
    def from_derived_dir(cls, path: Path) -> "OracleEngine":
        """Rebuild the engine from derived_private/oracle/ artefacts."""
        import json

        import polars as pl

        path = Path(path)
        node_index = pl.read_parquet(path / "node_index_private.parquet")
        node_ids = node_index["bodyId"].to_numpy()
        P = load_npz(path / "transition_matrix_primary.npz").tocsr()
        registry = json.loads(
            (path / "channel_registry_private.json").read_text(encoding="utf-8")
        )
        masks = {
            ai: build_channel_masks(
                ai, node_ids, entry["source_bodyIds"], entry["target_bodyIds"]
            )
            for ai, entry in registry["channels"].items()
        }
        return cls(node_ids, P, masks)


__all__ = ["OracleEngine", "EngineInfo"]
