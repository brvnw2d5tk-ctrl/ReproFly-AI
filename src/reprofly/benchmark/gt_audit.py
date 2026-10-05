"""Ground-truth mapping worksheet schema + aggregate audit (Phase 1E
§25-26, §46).

⚠️ EXECUTION STATUS: NOT EXECUTED IN PHASE 1E.

The Phase 1E charter (§26, §46) authorises reading the PRIVATE
ground-truth tables to build a mapping worksheet and run an
aggregate-only evaluability audit. However the authoritative
PROTOCOL_RULES.md #3 states 'Ground truth only used AFTER blinded runs
for scoring' and PROJECT_STATE.md records the ground truth as
'never read'. Per the charter's own conflict rule (§0: conflicts →
STOP, do not self-resolve), the audit CODE is implemented here but is
NOT executed against the sealed directories. The human must either
amend PROTOCOL_RULES #3 or provide an authorised ground-truth
worksheet; until then the audit numbers are NOT AVAILABLE.

The audit, once authorised, reads a PRIVATE worksheet (TSV) whose
columns are pinned below, and outputs AGGREGATE COUNTS ONLY — never
type names, blind identities, or rankings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

WORKSHEET_COLUMNS = (
    "gt_record_id",          # stable id of the ground-truth record
    "functional_group",      # human-assigned functional group label
    "raw_type_label",        # verbatim label from the GT source
    "mapping_status",        # EXACT | ALIAS_SUPPORTED | AMBIGUOUS |
                             # UNMAPPED | OUTSIDE_CANDIDATE_UNIVERSE
    "mapped_malecns_type",   # only for EXACT / approved ALIAS_SUPPORTED
)

VALID_STATUSES = (
    "EXACT", "ALIAS_SUPPORTED", "AMBIGUOUS", "UNMAPPED",
    "OUTSIDE_CANDIDATE_UNIVERSE",
)

# Only these enter the formal scorer (ALIAS_SUPPORTED additionally
# requires recorded human approval per record).
EVALUABLE_STATUSES = ("EXACT", "ALIAS_SUPPORTED")


@dataclass(frozen=True)
class GroundTruthAudit:
    records_total: int
    groups_total: int
    exact: int
    alias_supported_pending: int
    ambiguous: int
    unmapped: int
    outside_candidate_universe: int
    evaluable: int
    evaluable_groups: int
    biologically_underpowered: bool  # evaluable_groups < 3


def audit_mapping_worksheet(
    worksheet_path: Path,
    candidate_universe_types: set[str],
    alias_approved: bool = False,
) -> GroundTruthAudit:
    """Deterministic aggregate audit of the PRIVATE mapping worksheet.

    Parameters
    ----------
    worksheet_path : PRIVATE TSV with the pinned columns.
    candidate_universe_types : real type names of the final eligible
        universe (PRIVATE input, used only for membership checks).
    alias_approved : whether ALIAS_SUPPORTED records carry recorded
        human approval. If False, ALIAS_SUPPORTED counts as pending and
        is NOT evaluable yet.

    Outputs are counts ONLY (charter §46)."""
    ws = pl.read_csv(worksheet_path, separator="\t")
    missing = [c for c in WORKSHEET_COLUMNS if c not in ws.columns]
    if missing:
        raise ValueError(f"worksheet missing columns: {missing}")
    bad = ws.filter(~pl.col("mapping_status").is_in(list(VALID_STATUSES)))
    if bad.height:
        raise ValueError(
            f"worksheet has {bad.height} rows with invalid mapping_status"
        )

    counts = {
        s: ws.filter(pl.col("mapping_status") == s).height
        for s in VALID_STATUSES
    }

    evaluable_rows = ws.filter(
        (pl.col("mapping_status") == "EXACT")
        | (
            (pl.col("mapping_status") == "ALIAS_SUPPORTED")
            & pl.lit(alias_approved)
        )
    ).filter(
        pl.col("mapped_malecns_type").is_not_null()
        & pl.col("mapped_malecns_type").is_in(list(candidate_universe_types))
    )
    evaluable = evaluable_rows.height
    evaluable_groups = evaluable_rows["functional_group"].n_unique()

    return GroundTruthAudit(
        records_total=ws.height,
        groups_total=ws["functional_group"].n_unique(),
        exact=counts["EXACT"],
        alias_supported_pending=counts["ALIAS_SUPPORTED"],
        ambiguous=counts["AMBIGUOUS"],
        unmapped=counts["UNMAPPED"],
        outside_candidate_universe=counts["OUTSIDE_CANDIDATE_UNIVERSE"],
        evaluable=evaluable,
        evaluable_groups=evaluable_groups,
        biologically_underpowered=evaluable_groups < 3,
    )


__all__ = [
    "audit_mapping_worksheet",
    "GroundTruthAudit",
    "WORKSHEET_COLUMNS",
    "VALID_STATUSES",
    "EVALUABLE_STATUSES",
]
