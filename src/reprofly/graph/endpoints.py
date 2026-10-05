"""Phase 1A — Anatomical Endpoint Freeze: membership construction ONLY.

The endpoint policy is a frozen human decision (docs/state/LOCKED_DECISIONS.md
#9/#10, docs/endpoints/ANATOMICAL_ENDPOINT_POLICY.md). This module implements
it verbatim: it must not re-discuss, replace, re-derive, or re-weight the
selection, and it must not adjust membership to force disjoint groups.

What this module does:
  * token-aware exact matching of comma-separated nerve annotations
  * bodyId-level membership tables for six endpoint groups
  * QC counts and the four dictated pairwise overlaps

What this module must never contain:
  candidate scores, centrality, PageRank, reproductive importance, rankings,
  effect sizes, or any biological interpretation of any node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import polars as pl

from .official_rules import valid_type_expr

# ---------------------------------------------------------------------------
# Frozen endpoint definitions (human decision — LOCKED_DECISIONS.md #9)
# ---------------------------------------------------------------------------

MEMBERSHIP_PRIMARY = "primary_graph"            # canonical AND edge-connected (166,400 set)
MEMBERSHIP_CANONICAL_UNCONNECTED = "canonical_unconnected"  # canonical, no valid edge
MEMBERSHIP_ANNOTATION_ONLY = "annotation_only"  # nerve annotation but not in canonical set


@dataclass(frozen=True)
class EndpointGroupDef:
    group_id: str
    direction: str            # "entry" | "exit"
    tokens: tuple[str, ...]
    label: str
    role: str


ENDPOINT_DEFINITIONS: dict[str, EndpointGroupDef] = {
    "R_IN_ABNT": EndpointGroupDef(
        group_id="R_IN_ABNT", direction="entry", tokens=("AbNT",),
        label="AbNT-associated reproductive-enriched input proxy",
        role="reproductive-enriched anatomical proxy (a priori)",
    ),
    "R_OUT_ABNT": EndpointGroupDef(
        group_id="R_OUT_ABNT", direction="exit", tokens=("AbNT",),
        label="AbNT-associated reproductive-enriched output proxy",
        role="reproductive-enriched anatomical proxy (a priori)",
    ),
    "CTRL_ABDOMINAL_IN": EndpointGroupDef(
        group_id="CTRL_ABDOMINAL_IN", direction="entry",
        tokens=("AbN1", "AbN2", "AbN3", "AbN4"),
        label="Abdominal location-matched input control",
        role="abdominal non-AbNT control",
    ),
    "CTRL_ABDOMINAL_OUT": EndpointGroupDef(
        group_id="CTRL_ABDOMINAL_OUT", direction="exit",
        tokens=("AbN1", "AbN2", "AbN3", "AbN4"),
        label="Abdominal location-matched output control",
        role="abdominal non-AbNT control",
    ),
    "CTRL_LEG_IN": EndpointGroupDef(
        group_id="CTRL_LEG_IN", direction="entry",
        tokens=("ProLN", "MesoLN", "MetaLN"),
        label="Thoracic leg sensorimotor input control",
        role="general sensorimotor control",
    ),
    "CTRL_LEG_OUT": EndpointGroupDef(
        group_id="CTRL_LEG_OUT", direction="exit",
        tokens=("ProLN", "MesoLN", "MetaLN"),
        label="Thoracic leg sensorimotor output control",
        role="general sensorimotor control",
    ),
}

# The four dictated overlap QC pairs (matched direction).
OVERLAP_PAIRS: tuple[tuple[str, str], ...] = (
    ("R_IN_ABNT", "CTRL_ABDOMINAL_IN"),
    ("R_OUT_ABNT", "CTRL_ABDOMINAL_OUT"),
    ("R_IN_ABNT", "CTRL_LEG_IN"),
    ("R_OUT_ABNT", "CTRL_LEG_OUT"),
)

# Exact output schema. Aggregation-free membership record only.
ENDPOINT_FILE_COLUMNS: tuple[str, ...] = (
    "bodyId",
    "endpoint_group",
    "source_annotation",          # raw nerve annotation, verbatim
    "matched_tokens",             # comma-joined sorted tokens that matched
    "canonical_graph_membership", # primary_graph | canonical_unconnected | annotation_only
    "typed_canonical",            # official valid type among canonical neurons (bool)
)

# Columns that must NEVER exist in endpoint outputs (spec 2.5).
FORBIDDEN_COLUMN_MARKERS: tuple[str, ...] = (
    "score", "centrality", "pagerank", "rank", "importance",
    "effect_size", "effectsize", "candidate", "reproductive",
    "priority", "weight_norm", "prediction",
)


# ---------------------------------------------------------------------------
# Token-aware exact matching (LOCKED_DECISIONS.md #10)
# ---------------------------------------------------------------------------

def split_nerve_tokens(annotation: str | None) -> list[str]:
    """Split a possibly comma-separated nerve annotation into stripped,
    non-empty tokens. Null/empty -> []. This is exact-token logic; callers
    must never fall back to substring matching."""
    if annotation is None:
        return []
    if not isinstance(annotation, str):
        annotation = str(annotation)
    return [t.strip() for t in annotation.split(",") if t.strip()]


def annotation_matches_tokens(
    annotation: str | None, tokens: Iterable[str]
) -> list[str]:
    """Return the sorted unique tokens of `annotation` that exactly equal an
    allowed token. Empty list = no membership."""
    allowed = set(tokens)
    return sorted({t for t in split_nerve_tokens(annotation) if t in allowed})


# ---------------------------------------------------------------------------
# Membership construction
# ---------------------------------------------------------------------------

def _assert_no_forbidden_columns(frame: pl.DataFrame) -> None:
    for col in frame.columns:
        low = col.lower()
        for marker in FORBIDDEN_COLUMN_MARKERS:
            if marker in low:
                raise ValueError(
                    f"Forbidden endpoint column {col!r} (marker {marker!r}); "
                    "endpoint files are membership records only."
                )


def build_endpoint_groups(
    full_annotations: pl.DataFrame,
    canonical_neurons: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Build the six frozen endpoint membership tables.

    Parameters
    ----------
    full_annotations : the complete body-annotations table (all bodies,
        e.g. 211,577 rows for MaleCNS v1.0). Must contain bodyId, entryNerve,
        exitNerve.
    canonical_neurons : the canonical neuron table from Phase 0.5
        (derived_private/canonical_graph/canonical_neurons_private.parquet).
        Must contain bodyId, type, has_valid_edge.
    """
    for col in ("bodyId", "entryNerve", "exitNerve"):
        if col not in full_annotations.columns:
            raise ValueError(f"full_annotations must contain {col!r}")
    for col in ("bodyId", "type", "has_valid_edge"):
        if col not in canonical_neurons.columns:
            raise ValueError(f"canonical_neurons must contain {col!r}")

    # integrity: canonical ids must be real annotation bodies
    ann_ids = set(full_annotations["bodyId"].to_list())
    orphan = set(canonical_neurons["bodyId"].to_list()) - ann_ids
    if orphan:
        raise ValueError(
            f"canonical_neurons contains {len(orphan)} body ids absent from "
            f"full_annotations (e.g. {sorted(orphan)[:5]}); refusing to build."
        )

    canon = canonical_neurons.select(
        "bodyId",
        pl.col("has_valid_edge").cast(pl.Boolean).alias("_edge"),
        valid_type_expr().alias("_typed"),
    )

    groups: dict[str, pl.DataFrame] = {}
    for gid, d in ENDPOINT_DEFINITIONS.items():
        field = "entryNerve" if d.direction == "entry" else "exitNerve"
        candidates = (
            full_annotations
            .filter(
                pl.col(field).is_not_null()
                & (pl.col(field).str.strip_chars() != "")
            )
            .select(
                pl.col("bodyId"),
                pl.col(field).alias("source_annotation"),
            )
            .with_columns(
                pl.col("source_annotation").map_elements(
                    lambda s, _t=d.tokens: ",".join(
                        annotation_matches_tokens(s, _t)
                    ),
                    return_dtype=pl.String,
                ).alias("matched_tokens")
            )
            .filter(pl.col("matched_tokens") != "")
            .with_columns(pl.lit(gid).alias("endpoint_group"))
        )
        merged = candidates.join(canon, on="bodyId", how="left").with_columns(
            pl.when(pl.col("_edge") == True)          # noqa: E712
              .then(pl.lit(MEMBERSHIP_PRIMARY))
              .when(pl.col("_edge") == False)         # noqa: E712
              .then(pl.lit(MEMBERSHIP_CANONICAL_UNCONNECTED))
              .otherwise(pl.lit(MEMBERSHIP_ANNOTATION_ONLY))
              .alias("canonical_graph_membership"),
            pl.col("_typed").fill_null(False).alias("typed_canonical"),
        )
        out = merged.select(ENDPOINT_FILE_COLUMNS).sort("bodyId")
        # integrity: no fabricated ids
        if not set(out["bodyId"].to_list()) <= ann_ids:
            raise ValueError(f"{gid}: fabricated body ids detected")
        _assert_no_forbidden_columns(out)
        groups[gid] = out
    return groups


# ---------------------------------------------------------------------------
# QC counts + dictated overlaps
# ---------------------------------------------------------------------------

def _primary_members(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("canonical_graph_membership") == MEMBERSHIP_PRIMARY)


def endpoint_qc(groups: dict[str, pl.DataFrame]) -> dict:
    """Per-group counts (annotation-level / primary-graph / typed / missing)
    plus the four dictated pairwise overlaps. Counts only — no scores."""
    per_group: dict[str, dict] = {}
    for gid, df in groups.items():
        pg = _primary_members(df)
        typed_n = int(pg["typed_canonical"].sum()) if pg.height else 0
        per_group[gid] = {
            "annotation_n": df.height,
            "primary_graph_n": pg.height,
            "typed_n": typed_n,
            "missing_type_n": pg.height - typed_n,
            "canonical_unconnected_n": df.filter(
                pl.col("canonical_graph_membership") == MEMBERSHIP_CANONICAL_UNCONNECTED
            ).height,
            "annotation_only_n": df.filter(
                pl.col("canonical_graph_membership") == MEMBERSHIP_ANNOTATION_ONLY
            ).height,
            "multivalued_n": df.filter(
                pl.col("source_annotation").str.contains(",")
            ).height,
        }

    overlaps: dict[str, dict] = {}
    for a, b in OVERLAP_PAIRS:
        pa = _primary_members(groups[a])
        pb = _primary_members(groups[b])
        ids_a = set(pa["bodyId"].to_list())
        ids_b = set(pb["bodyId"].to_list())
        inter = ids_a & ids_b
        patterns: list[str] = []
        if inter:
            inter_list = sorted(inter)
            patterns = sorted(set(
                groups[a]
                .filter(pl.col("bodyId").is_in(inter_list))
                .filter(pl.col("canonical_graph_membership") == MEMBERSHIP_PRIMARY)
                ["source_annotation"].to_list()
            ))
        overlaps[f"{a}__vs__{b}"] = {
            "primary_graph_overlap_n": len(inter),
            "annotation_level_overlap_n": len(
                set(groups[a]["bodyId"].to_list())
                & set(groups[b]["bodyId"].to_list())
            ),
            "overlap_source_annotations": patterns,
        }

    return {"per_group": per_group, "overlaps": overlaps}


__all__ = [
    "ENDPOINT_DEFINITIONS",
    "OVERLAP_PAIRS",
    "ENDPOINT_FILE_COLUMNS",
    "FORBIDDEN_COLUMN_MARKERS",
    "MEMBERSHIP_PRIMARY",
    "MEMBERSHIP_CANONICAL_UNCONNECTED",
    "MEMBERSHIP_ANNOTATION_ONLY",
    "split_nerve_tokens",
    "annotation_matches_tokens",
    "build_endpoint_groups",
    "endpoint_qc",
]
