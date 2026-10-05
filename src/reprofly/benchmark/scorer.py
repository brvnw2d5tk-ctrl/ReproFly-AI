"""PRIVATE GroundTruthScorer — sealed benchmark scoring component.

ISOLATION CONTRACT (Phase 1F charter §4, PROTOCOL_RULES #3):
  * This component MAY read: the PRIVATE ground truth, the PRIVATE blind
    mapping, and the candidate-universe PRIVATE registry.
  * It MUST NEVER: be callable from the Scientist MCP; return mapping
    information to the Scientist; return live recall / hit information
    to the Scientist; influence a running Scientist's choices. Its
    filesystem and tool access are fully isolated from the Scientist
    surface (it is not imported by any MCP module — enforced by tests).
  * Actual benchmark scores are computed only AFTER blinded runs
    complete. In Phase 1F only the MAPPING AUDIT and denominator FREEZE
    run (authorized pre-benchmark exception).

MECHANICAL MAPPING POLICY (no semantic guessing — charter §5, STOP I):
  Each ground-truth record's `neuron_or_cell_type` label is tokenised;
  tokens are matched against (a) exact MaleCNS v1.0 type names and
  (b) the annotations' OFFICIAL `synonyms` column (with paper citation).
  * EXACT — a token is literally a type name; unique single candidate
  * ALIAS_SUPPORTED — a token appears in a type's official synonyms
    (enters denominator ONLY after recorded human approval)
  * AMBIGUOUS — a token maps to >1 type (e.g. two types cite the same
    literature alias); NEVER auto-resolved
  * UNMAPPED — no mechanical match (organ/animal-level records,
    genetic-class labels, unnamed populations)
  * OUTSIDE_CANDIDATE_UNIVERSE — maps to a typed type outside the frozen
    eligible universe
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import polars as pl

# Functional-group assignment from ground_truth_id prefix (verified
# against 13_Reproductive_Circuit_GroundTruth/G1..G4 directory names
# and 05_BrainGonad_groundtruth = the Nat Commun 2026 paper).
GT_PREFIX_TO_GROUP = {
    "GT-CRZEJAC": "G1_Crz_Ejaculation_2012",
    "GT-DSXCOP": "G2_dsx_Copulation_2016",
    "GT-P1PCD": "G3_P1_pCd_2020",
    "GT-LC10-P1": "G4_P1_LC10a_2021",
    "GT-AXIS": "G5_BrainGonad_20E_2026",
}

STATUS_EXACT = "EXACT"
STATUS_ALIAS = "ALIAS_SUPPORTED"
STATUS_AMBIGUOUS = "AMBIGUOUS"
STATUS_UNMAPPED = "UNMAPPED"
STATUS_OUTSIDE = "OUTSIDE_CANDIDATE_UNIVERSE"

# tokens that are generic words and must never be treated as type labels
_STOPWORDS = {
    "neurons", "neuron", "INs", "PNs", "PN", "IN", "cells", "cell",
    "motor", "visual", "projection", "interneurons", "serotonergic",
    "glutamatergic", "GABAergic", "male", "female", "fly", "mouse",
    "brain", "VNC", "AG", "EB", "TE", "SV", "AED", "ventral", "nerve",
    "cord", "abdominal", "ganglion", "ventral", "accessory", "glands",
    "gland", "intestine", "testes", "system", "reproductive", "whole",
    "animal", "pathway", "subset", "four", "circuits", "circuit",
    "positive", "corazonin", "peptides", "peptide", "hormone", "20E",
    "and", "or", "the", "of", "in", "to", "with", "not", "local",
}


@dataclass
class MappingAuditResult:
    records_total: int
    groups_total: int
    counts: dict = field(default_factory=dict)
    evaluable_types: set = field(default_factory=set)
    evaluable_groups: set = field(default_factory=set)
    group_type_sets: dict = field(default_factory=dict)
    cross_group_shared_types: dict = field(default_factory=dict)


def parse_synonym_entries(synonyms: str | None) -> list[tuple[str, str]]:
    """Parse the official synonyms string into (paper, alias) pairs.
    Format observed: 'Kimura 2008, Kohatsu 2010: P2b; Cachero 2010:
    pIP-a; Yu 2010: pIP1'."""
    if not synonyms:
        return []
    out: list[tuple[str, str]] = []
    for chunk in synonyms.split(";"):
        if ":" in chunk:
            paper, alias = chunk.split(":", 1)
            out.append((paper.strip(), alias.strip()))
    return out


def build_alias_index(annotations: pl.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """alias -> [(type, paper), ...] from the OFFICIAL synonyms column."""
    idx: dict[str, list[tuple[str, str]]] = {}
    rows = annotations.filter(pl.col("synonyms").is_not_null())
    for r in rows.iter_rows(named=True):
        t = r["type"]
        if not t:
            continue
        for paper, alias in parse_synonym_entries(r["synonyms"]):
            idx.setdefault(alias, []).append((t, paper))
    return idx


def extract_label_tokens(label: str) -> list[str]:
    """Conservative tokenisation of a neuron_or_cell_type label."""
    raw = re.split(r"[^A-Za-z0-9_\-\.]+", label)
    tokens = []
    for t in raw:
        t = t.strip(".-_")
        if not t or t in _STOPWORDS or t.isdigit():
            continue
        tokens.append(t)
    # de-duplicate, preserve order
    seen, out = set(), []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def audit_mapping(
    gt_table: pl.DataFrame,
    typed_types: set[str],
    eligible_types: set[str],
    alias_index: dict[str, list[tuple[str, str]]],
) -> tuple[pl.DataFrame, MappingAuditResult]:
    """Mechanical mapping audit. Returns (worksheet frame, aggregates).

    The worksheet frame carries PRIVATE columns (real labels, candidate
    types) and is written ONLY to derived_private/.
    """
    rows = []
    for r in gt_table.iter_rows(named=True):
        gt_id = r["ground_truth_id"]
        label = r["neuron_or_cell_type"]
        group = next(
            (g for p, g in GT_PREFIX_TO_GROUP.items()
             if gt_id.startswith(p)), "UNKNOWN"
        )
        tokens = extract_label_tokens(label)

        exact_hits: dict[str, str] = {}     # type -> token
        alias_hits: dict[str, tuple[str, str]] = {}  # type -> (token, paper)
        ambiguous_tokens: dict[str, list[str]] = {}

        for tok in tokens:
            if tok in typed_types:
                exact_hits.setdefault(tok, tok)  # token IS the type name
            if tok in alias_index:
                cands = alias_index[tok]
                if len({t for t, _p in cands}) == 1:
                    t, paper = cands[0]
                    alias_hits.setdefault(t, (tok, paper))
                else:
                    ambiguous_tokens[tok] = [
                        f"{t} ({p})" for t, p in cands
                    ]

        mapped_type: Optional[str] = None
        status: str
        evidence = ""

        if ambiguous_tokens:
            status = STATUS_AMBIGUOUS
            evidence = "; ".join(
                f"{tok} -> {', '.join(v)}"
                for tok, v in ambiguous_tokens.items()
            )
        elif exact_hits:
            if len(exact_hits) == 1:
                mapped_type = next(iter(exact_hits))
                status = STATUS_EXACT
                evidence = f"label token '{mapped_type}' is an exact " \
                           "MaleCNS v1.0 type name"
            else:
                status = STATUS_AMBIGUOUS
                evidence = "multiple exact-mapped populations: " + \
                    ", ".join(sorted(exact_hits))
        elif alias_hits:
            if len(alias_hits) == 1:
                mapped_type, (tok, paper) = next(iter(alias_hits.items()))
                status = STATUS_ALIAS
                evidence = f"label token '{tok}' is an official synonym " \
                           f"of {mapped_type} ({paper}) — requires recorded human " \
                           "APPROVAL before entering denominator"
            else:
                status = STATUS_AMBIGUOUS
                evidence = "multiple alias-mapped populations: " + \
                    ", ".join(sorted(alias_hits))
        else:
            status = STATUS_UNMAPPED
            evidence = "no mechanical match of any label token against " \
                       "MaleCNS type names or official synonyms"

        if mapped_type is not None and status in (STATUS_EXACT, STATUS_ALIAS):
            if mapped_type not in eligible_types:
                status = STATUS_OUTSIDE
                evidence += " | mapped type is OUTSIDE the frozen " \
                            "eligible candidate universe"

        rows.append({
            "gt_record_id": gt_id,
            "functional_group": group,
            "raw_type_label": label,                      # PRIVATE
            "mapping_status": status,
            "mapped_malecns_type": mapped_type,           # PRIVATE
            "mapped_blind_id": None,                      # filled by caller
            "mechanical_evidence": evidence,              # PRIVATE
            "human_approval": "PENDING" if status == STATUS_ALIAS else (
                "NOT_APPLICABLE"),
            "paper": r["publication"][:80],
            "doi": r["doi"],
            "confidence_in_paper": r["confidence"],
        })

    ws = pl.DataFrame(rows)

    # aggregates — EXACT only is evaluable at freeze (aliases pending
    # human approval; ambiguous never auto-resolved)
    evaluable = ws.filter(
        (pl.col("mapping_status") == STATUS_EXACT)
        & pl.col("mapped_malecns_type").is_not_null()
    )
    eval_types = set(evaluable["mapped_malecns_type"].to_list())
    eval_groups = set(evaluable["functional_group"].to_list())

    group_type_sets: dict[str, set] = {}
    for g in GT_PREFIX_TO_GROUP.values():
        group_type_sets[g] = set(
            evaluable.filter(pl.col("functional_group") == g)
            ["mapped_malecns_type"].to_list()
        )
    shared: dict[str, list[str]] = {}
    for t in eval_types:
        gs = [g for g, s in group_type_sets.items() if t in s]
        if len(gs) > 1:
            shared[t] = gs

    counts = {s: ws.filter(pl.col("mapping_status") == s).height
              for s in (STATUS_EXACT, STATUS_ALIAS, STATUS_AMBIGUOUS,
                        STATUS_UNMAPPED, STATUS_OUTSIDE)}
    result = MappingAuditResult(
        records_total=ws.height,
        groups_total=ws["functional_group"].n_unique(),
        counts=counts,
        evaluable_types=eval_types,
        evaluable_groups=eval_groups,
        group_type_sets=group_type_sets,
        cross_group_shared_types=shared,
    )
    return ws, result


# ---------------------------------------------------------------------------
# Cross-group sensitivity metric (charter §8)
# ---------------------------------------------------------------------------

def deduplicated_group_recall(
    attempt_blind_ids: list[str],
    gt_blind_to_group: dict[str, str],
    group_type_sets: dict[str, set],
    real_to_blind: dict[str, str],
) -> dict:
    """DEDUPLICATED_GROUP_RECALL sensitivity metric: functional groups
    whose EVALUABLE GT type sets are exactly identical are merged into
    one before computing group recall. Group structure itself is NEVER
    modified by benchmark results."""
    # canonical signature per group (frozenset of real types)
    sig_map: dict[frozenset, list[str]] = {}
    for g, types in group_type_sets.items():
        if types:
            sig_map.setdefault(frozenset(types), []).append(g)
    merged_groups = {sig: gs[0] for sig, gs in sig_map.items()}
    n_groups = len(merged_groups)

    hit_groups: set[frozenset] = set()
    curve = []
    for k, bid in enumerate(attempt_blind_ids, start=1):
        g = gt_blind_to_group.get(bid)
        if g is not None:
            for sig, gs in sig_map.items():
                if g in gs:
                    hit_groups.add(sig)
        curve.append(len(hit_groups) / n_groups if n_groups else 0.0)
    return {
        "n_merged_groups": n_groups,
        "n_original_groups": len(
            [g for g, s in group_type_sets.items() if s]
        ),
        "deduplicated_group_recall_curve": curve,
        "deduplicated_augrc": (
            sum(curve) / len(curve) if curve else 0.0
        ),
    }


__all__ = [
    "audit_mapping", "build_alias_index", "extract_label_tokens",
    "parse_synonym_entries", "deduplicated_group_recall",
    "MappingAuditResult", "GT_PREFIX_TO_GROUP",
    "STATUS_EXACT", "STATUS_ALIAS", "STATUS_AMBIGUOUS",
    "STATUS_UNMAPPED", "STATUS_OUTSIDE",
]
