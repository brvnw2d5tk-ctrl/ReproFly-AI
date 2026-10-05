"""Official MaleCNS validity rules — verbatim port of the paper's own logic.

Provenance (do not modify without a new official source):
  11_MaleCNS_Official/raw/2025malecns/supplemental_data/
  quantify-neuron-connections.ipynb  (Berg et al., repo commit 67767d2)

Official code (v0.9 notebook, cell 3):

    valid_superclasses = [sc for sc in annotations['superclass'].unique()
                          if sc and 'tbc' not in sc]
    valid_types = [t for t in annotations['type'].unique()
                   if t and 'unclear' not in t and t != "Pm7_Li28"]

These rules are applied to v1.0 annotations here. Note the notebook's
printed counts (20 superclasses / 11,691 types / 25,563,426 edges /
166,391 connected neurons / 6,237,402 ge5 edges / 929,735 type edges)
are v0.9 reference numbers; v1.0 differs slightly (see
docs/PHASE05_GRAPH_ANATOMY_REPORT.md).

Reference counts from the paper abstract (bioRxiv 2025.10.09.680999):
166,691 neurons, 11,691 types.
"""

from __future__ import annotations

import polars as pl

# Officially excluded type name (verbatim from the notebook).
OFFICIAL_EXCLUDED_TYPE = "Pm7_Li28"

# v0.9 official notebook reference counts (for comparison reporting only).
OFFICIAL_V09_REFERENCE: dict[str, int] = {
    "valid_superclasses": 20,
    "valid_types": 11691,
    "edges_valid_superclass": 25_563_426,
    "connected_neurons": 166_391,
    "ge5_edges": 6_237_402,
    "type_edges_ge5": 929_735,
    "connected_types_ge5": 11_687,
}

# Paper-abstract headline counts (v0.9 preprint text).
PAPER_ABSTRACT_REFERENCE: dict[str, int] = {
    "neurons": 166_691,
    "types": 11_691,
}


def valid_superclass_expr() -> pl.Expr:
    """Superclass is non-null and does not contain the substring 'tbc'
    (the official 'to-be-confirmed' marker)."""
    return (
        pl.col("superclass").is_not_null()
        & ~pl.col("superclass").str.contains("tbc")
    )


def valid_type_expr() -> pl.Expr:
    """Type is non-null, does not contain 'unclear', and is not the
    officially excluded name Pm7_Li28."""
    return (
        pl.col("type").is_not_null()
        & ~pl.col("type").str.contains("unclear")
        & (pl.col("type") != OFFICIAL_EXCLUDED_TYPE)
    )


def canonical_neuron_mask(annotations: pl.DataFrame) -> pl.Expr:
    """The official notion of a neuronal body for analysis: a body whose
    superclass passes the official validity rule. Glia carry a null
    superclass and are thereby excluded, as are 'tbc' bodies."""
    return valid_superclass_expr()


def official_excluded_type_name() -> str:
    return OFFICIAL_EXCLUDED_TYPE


__all__ = [
    "valid_superclass_expr",
    "valid_type_expr",
    "canonical_neuron_mask",
    "official_excluded_type_name",
    "OFFICIAL_V09_REFERENCE",
    "PAPER_ABSTRACT_REFERENCE",
]
