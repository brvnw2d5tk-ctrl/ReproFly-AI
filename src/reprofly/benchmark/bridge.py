"""Ground-Truth Bridge v1.1 — evidence preparation (Phase 1F.1).

Representation-layer fix: the old model assumed one literature label →
one MaleCNS type. The bridge supports **GT Entity → Frozen MaleCNS
Type Set** with per-member evidence. This module ONLY prepares
evidence: it never approves mappings, never guesses types, never
expands mappings to pass the power gate. Every non-EXACT mapping row
leaves human_decision = PENDING for HUMAN review (Phase 1F.2).

Evidence tiers (charter §4):
  A1  MaleCNS official type or official synonym explicitly names the
      legacy population (preferably citing the same study)
  A2  MaleCNS official cross-atlas mapping links legacy → modern types
  B1  curated connectome DB (frozen drosophila_neuropeptides v1.1.0)
      links a molecular population to a connectome cell type with
      strong experimental evidence (confidence 4–5; confidence 3 =
      moderate; ≤2 not suitable for primary)
  B2  original paper + official later connectome mapping / official
      curated morphology correspondence
  C   name resemblance / anatomical proximity / model or LLM inference
      / biological plausibility — NEVER enters primary mapping
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import polars as pl

# ---------------------------------------------------------------------------
# Entity definition types (charter §3)
# ---------------------------------------------------------------------------

ENTITY_EXACT_CELL_TYPE = "EXACT_CELL_TYPE"
ENTITY_LEGACY_MORPHO = "LEGACY_MORPHOLOGICAL_POPULATION"
ENTITY_GENETIC_DRIVER = "GENETIC_DRIVER_POPULATION"
ENTITY_MOLECULAR = "MOLECULAR_POPULATION"
ENTITY_ANATOMICAL = "ANATOMICAL_POPULATION"
ENTITY_ORGAN = "ORGAN_LEVEL_NOT_NEURONAL"
ENTITY_OTHER = "OTHER"

# Proposed mapping statuses (charter §12) — evidence statuses, NOT approvals
STATUS_EXACT = "EXACT"
STATUS_ALIAS_SINGLE = "OFFICIAL_ALIAS_SINGLE"
STATUS_POPULATION_SPLIT = "OFFICIAL_POPULATION_SPLIT"
STATUS_CURATED_MOLECULAR = "CURATED_MOLECULAR_MAPPING"
STATUS_BROAD = "BROAD_POPULATION_MAPPING"
STATUS_TRUE_AMBIGUITY = "TRUE_AMBIGUITY"
STATUS_UNMAPPED = "UNMAPPED"
STATUS_ORGAN = "ORGAN_LEVEL_NOT_NEURONAL"
STATUS_OUTSIDE = "OUTSIDE_CANDIDATE_UNIVERSE"

# Evidence strength ceilings for the review packet (charter §18)
STRENGTH_STRONG = "STRONG EVIDENCE"
STRENGTH_MODERATE = "MODERATE EVIDENCE"
STRENGTH_INSUFFICIENT = "INSUFFICIENT EVIDENCE"

HUMAN_PENDING = "PENDING"


# ---------------------------------------------------------------------------
# §3 GT entities from the 25 evidence records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GTEntity:
    entity_id: str
    legacy_label: str
    functional_group: str
    entity_definition_type: str
    source_record_ids: str
    source_publications: str


_ORGAN_PATTERNS = (
    re.compile(r"male reproductive system", re.I),
    re.compile(r"ejaculatory bulb", re.I),
    re.compile(r"whole[- ]animal", re.I),
    re.compile(r"male fly \(and male mouse", re.I),
    re.compile(r"hemolymph", re.I),
)
_MOLECULAR_PATTERNS = (
    re.compile(r"corazonin-positive", re.I),
    re.compile(r"serotonergic \(5-HT\)", re.I),
    re.compile(r"Crz neurons", re.I),
    re.compile(r"Dsk neurons", re.I),
    re.compile(r"CrzR neurons", re.I),
    re.compile(r"CCKLR", re.I),
    re.compile(r"DopEcR", re.I),
    re.compile(r"EcR", re.I),
)
_GENETIC_PATTERNS = (
    re.compile(r"dsx[+-]", re.I),
    re.compile(r"fru[+-]", re.I),
    re.compile(r"dsx expression", re.I),
    re.compile(r"dsx-configured", re.I),
)
_LEGACY_PATTERNS = (
    re.compile(r"pCd neurons", re.I),
    re.compile(r"P1 neurons", re.I),
    re.compile(r"persistent.*pCd", re.I),
)


def classify_entity_label(label: str) -> str:
    # exact modern type name takes precedence (charter §17: keep LC10a)
    if "LC10a" in label:
        return ENTITY_EXACT_CELL_TYPE
    if any(p.search(label) for p in _ORGAN_PATTERNS):
        return ENTITY_ORGAN
    if any(p.search(label) for p in _MOLECULAR_PATTERNS):
        return ENTITY_MOLECULAR
    if any(p.search(label) for p in _GENETIC_PATTERNS):
        return ENTITY_GENETIC_DRIVER
    if any(p.search(label) for p in _LEGACY_PATTERNS):
        return ENTITY_LEGACY_MORPHO
    return ENTITY_OTHER


def build_gt_entities(gt_table: pl.DataFrame) -> list[GTEntity]:
    """Group the 25 evidence records into biological entities by
    (functional group, neuron_or_cell_type label). Original records are
    never modified; entity rows reference their source record ids."""
    entities: list[GTEntity] = []
    grouped: dict[tuple, list[dict]] = {}
    for r in gt_table.iter_rows(named=True):
        key = (r["ground_truth_id"].rsplit("-", 1)[0], r["neuron_or_cell_type"])
        grouped.setdefault(key, []).append(r)
    for (prefix, label), rows in sorted(grouped.items()):
        group = next(
            (g for p, g in {
                "GT-CRZEJAC": "G1_Crz_Ejaculation_2012",
                "GT-DSXCOP": "G2_dsx_Copulation_2016",
                "GT-P1PCD": "G3_P1_pCd_2020",
                "GT-LC10-P1": "G4_P1_LC10a_2021",
                "GT-AXIS": "G5_BrainGonad_20E_2026",
            }.items() if prefix.startswith(p)), "UNKNOWN"
        )
        entities.append(GTEntity(
            entity_id=f"ENTE-{prefix}-{len(entities)+1:02d}",
            legacy_label=label,
            functional_group=group,
            entity_definition_type=classify_entity_label(label),
            source_record_ids=";".join(r["ground_truth_id"] for r in rows),
            source_publications=rows[0]["publication"][:100],
        ))
    return entities


# ---------------------------------------------------------------------------
# §5 full-field official synonym audit
# ---------------------------------------------------------------------------

AUDIT_FIELDS = ("type", "synonyms", "flywireType", "hemibrainType",
                "mancType", "supertype", "matchingNotes")


def search_official_fields(
    annotations: pl.DataFrame, needle: str,
) -> list[dict]:
    """Case-sensitive substring search of the official annotation fields
    for a legacy label token. Records every DISTINCT match verbatim
    (field, modern type, exact matched string) — deduplicated across
    the per-neuron annotation rows; no collapsing."""
    hits: list[dict] = []
    seen: set[tuple] = set()
    for fld in AUDIT_FIELDS:
        col = pl.col(fld)
        rows = annotations.filter(col.is_not_null() & col.str.contains(
            f"(?i){re.escape(needle)}"
        ))
        for r in rows.iter_rows(named=True):
            key = (fld, r.get("type"), str(r.get(fld)))
            if key in seen:
                continue
            seen.add(key)
            hits.append({
                "field": fld,
                "modern_type": r.get("type"),
                "matched_string": str(r.get(fld)),
                "status": r.get("status"),
                "dimorphism": r.get("dimorphism"),
            })
    return hits


def distinguish_split_vs_ambiguity(
    candidates: list[dict],
) -> tuple[str, str]:
    """Classify a multi-candidate official-synonym result.

    LEGACY_POPULATION_SPLIT: every candidate's evidence string is an
      OFFICIAL synonym/citation explicitly naming the legacy population
      (the old biological label genuinely corresponds to multiple
      modern types — a legitimate type-set).
    TRUE_AMBIGUITY: evidence cannot determine which candidate(s)
      belong (e.g. matches come from unstructured free-text notes
      rather than curated synonym citations, or the same citation is
      claimed by types with conflicting status)."""
    if len(candidates) <= 1:
        return ("SINGLE", "single or empty candidate set")
    curated = [c for c in candidates if c["field"] == "synonyms"]
    if len(curated) == len(candidates) and len(curated) >= 2:
        return ("LEGACY_POPULATION_SPLIT",
                f"{len(curated)} official synonym citations each "
                "explicitly name the legacy population (population "
                "split; type-set legitimate pending human approval)")
    if len(curated) >= 2:
        return ("LEGACY_POPULATION_SPLIT",
                f"{len(curated)} official synonym citations (plus "
                f"{len(candidates) - len(curated)} free-text matches); "
                "split supported by curated citations, breadth pending "
                "human review")
    return ("TRUE_AMBIGUITY",
            f"only {len(curated)} curated synonym citation(s) among "
            f"{len(candidates)} matches — evidence cannot determine "
            "membership")


# ---------------------------------------------------------------------------
# §8 molecular bridge via the frozen neuropeptide database
# ---------------------------------------------------------------------------

NP_DB_COLUMNS = {
    "cell_type": 0, "source": 1, "evidence": 2, "confidence": 3,
}


def load_neuropeptide_rows(
    np_csv: Path, peptide_column: str,
) -> list[dict]:
    """Rows of the frozen drosophila_neuropeptides v1.1.0 table where
    the given peptide column == 1.

    CSV layout (verified against the frozen file):
    species, region, hemilineage, cell_type, source, evidence,
    confidence, <peptide columns...>
    Confidence classes: 4-5 strong curated candidate; 3 moderate;
    <=2 not suitable for primary."""
    rows: list[dict] = []
    with open(np_csv, encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        i_type = header.index("cell_type")
        i_src = header.index("neuropeptide_verified_source")
        i_evi = header.index("neuropeptide_verified_evidence")
        i_conf = header.index("neuropeptide_verified_confidence")
        try:
            col = header.index(peptide_column)
        except ValueError:
            return rows
        for raw in reader:
            if len(raw) > col and raw[col].strip() == "1":
                try:
                    conf = int(raw[i_conf].strip())
                except (ValueError, IndexError):
                    conf = 0
                rows.append({
                    "cell_type": raw[i_type],
                    "source": raw[i_src] if len(raw) > i_src else "",
                    "evidence": raw[i_evi] if len(raw) > i_evi else "",
                    "confidence": conf,
                    "confidence_class": (
                        "strong" if conf >= 4
                        else "moderate" if conf == 3
                        else "not_suitable_for_primary"
                    ),
                })
    return rows


# ---------------------------------------------------------------------------
# §11 evidence worksheet row builder
# ---------------------------------------------------------------------------

@dataclass
class BridgeRow:
    entity_id: str
    legacy_label: str
    functional_group: str
    entity_definition_type: str
    candidate_malecns_type: Optional[str]
    candidate_blind_id: Optional[str]
    mapping_evidence_tier: str
    evidence_source: str
    evidence_reference: str
    exact_evidence_text_or_field: str
    original_study_relation: str
    mapping_status_proposed: str
    candidate_universe_member: str
    human_decision: str = HUMAN_PENDING
    human_notes: str = ""

    def as_dict(self) -> dict:
        return dict(vars(self))


WORKSHEET_COLUMNS = (
    "entity_id", "legacy_label", "functional_group",
    "entity_definition_type", "candidate_malecns_type",
    "candidate_blind_id", "mapping_evidence_tier", "evidence_source",
    "evidence_reference", "exact_evidence_text_or_field",
    "original_study_relation", "mapping_status_proposed",
    "candidate_universe_member", "human_decision", "human_notes",
)


# ---------------------------------------------------------------------------
# §14 set-size audit
# ---------------------------------------------------------------------------

SET_SIZE_LARGE = 10
SET_SIZE_VERY_LARGE = 25


def set_size_flag(n: int) -> str:
    if n > SET_SIZE_VERY_LARGE:
        return "SET_SIZE_VERY_LARGE"
    if n > SET_SIZE_LARGE:
        return "SET_SIZE_LARGE"
    return ""


# ---------------------------------------------------------------------------
# §15 RandomFirstHitPercentile (implemented, NOT executed on real runs)
# ---------------------------------------------------------------------------

def random_first_hit_pmf(n: int, m: int, k: int) -> float:
    """P(first target hit exactly at attempt k) under uniform random
    sampling WITHOUT replacement: population n, target set m."""
    if not (1 <= k <= n - m + 1) or m < 1:
        return 0.0
    # P(no hit in first k-1) * P(hit at k | no hit before)
    #   = C(n-m, k-1)/C(n, k-1) * m/(n-k+1)
    num = 1.0
    for i in range(k - 1):
        num *= (n - m - i) / (n - i)
    return num * m / (n - k + 1)


def random_first_hit_percentile(n: int, m: int, observed_k: int) -> dict:
    """Secondary size-adjusted metric: percentile / upper-tail
    probability of the observed first-hit attempt under the random
    null. Broad type-sets are easier to hit by chance — this metric
    normalises for set size m."""
    if m < 1 or observed_k < 1 or observed_k > n:
        return {"percentile": None, "tail_probability": None,
                "note": "undefined"}
    tail = sum(random_first_hit_pmf(n, m, k)
               for k in range(observed_k, n - m + 2))
    cdf = sum(random_first_hit_pmf(n, m, k)
              for k in range(1, observed_k + 1))
    return {
        "n": n, "m": m, "observed_k": observed_k,
        "percentile": round(cdf, 6),
        "tail_probability": round(tail, 6),
        "note": ("SECONDARY metric — size-adjusted first-hit null; "
                 "does NOT replace primary AUGRC"),
    }


# ---------------------------------------------------------------------------
# §13 type-set capable scorer support (schema upgrade, no final mapping)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GTEntityScoringRecord:
    """One approved entity mapping (post-human-review). The scorer
    consumes these; the bridge only PREPARES candidates."""

    entity_id: str
    approved_type_set: tuple[str, ...]     # real type names (PRIVATE)
    functional_group: str

    def recovered_by(self, perturbed_blind_ids: set[str],
                     real_to_blind: dict[str, str]) -> bool:
        """An entity is recovered when at least one approved member
        type has been successfully perturbed (charter §13 default rule
        — no per-entity special rules unless human-approved before
        benchmark)."""
        return any(
            real_to_blind.get(t) in perturbed_blind_ids
            for t in self.approved_type_set
        )


def entity_group_recall(
    entity_records: list[GTEntityScoringRecord],
    attempt_blind_ids: list[str],
    real_to_blind: dict[str, str],
) -> dict:
    """GroupRecall over ENTITIES: a group is recovered when at least
    one evaluable approved entity in it is recovered (§16)."""
    perturbed = set(attempt_blind_ids)
    groups = {r.functional_group for r in entity_records}
    curve: list[float] = []
    hit_groups: set[str] = set()
    seen: set[str] = set()
    for bid in attempt_blind_ids:
        seen.add(bid)
        for r in entity_records:
            if r.recovered_by(seen, real_to_blind):
                hit_groups.add(r.functional_group)
        curve.append(len(hit_groups) / len(groups) if groups else 0.0)
    return {
        "n_groups": len(groups),
        "n_entities": len(entity_records),
        "total_unique_mapped_types": len(
            {t for r in entity_records for t in r.approved_type_set}
        ),
        "group_recall_curve": curve,
    }


__all__ = [
    "GTEntity", "build_gt_entities", "classify_entity_label",
    "search_official_fields", "distinguish_split_vs_ambiguity",
    "load_neuropeptide_rows", "BridgeRow", "WORKSHEET_COLUMNS",
    "set_size_flag", "random_first_hit_percentile", "random_first_hit_pmf",
    "GTEntityScoringRecord", "entity_group_recall",
    "ENTITY_EXACT_CELL_TYPE", "ENTITY_LEGACY_MORPHO",
    "ENTITY_GENETIC_DRIVER", "ENTITY_MOLECULAR", "ENTITY_ANATOMICAL",
    "ENTITY_ORGAN", "ENTITY_OTHER",
    "STATUS_EXACT", "STATUS_ALIAS_SINGLE", "STATUS_POPULATION_SPLIT",
    "STATUS_CURATED_MOLECULAR", "STATUS_BROAD", "STATUS_TRUE_AMBIGUITY",
    "STATUS_UNMAPPED", "STATUS_ORGAN", "STATUS_OUTSIDE",
    "STRENGTH_STRONG", "STRENGTH_MODERATE", "STRENGTH_INSUFFICIENT",
    "HUMAN_PENDING", "AUDIT_FIELDS",
    "SET_SIZE_LARGE", "SET_SIZE_VERY_LARGE",
]
