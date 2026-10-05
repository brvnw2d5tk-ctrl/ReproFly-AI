"""Graph package — Phase 0.5 canonical graph & anatomy builders.

Everything here is data engineering / aggregation. There are NO graph
algorithms (no centrality, PageRank, diffusion, GNN, LIF) and NO scoring.
"""

from __future__ import annotations

from .anatomy import AnatomyInventory, build_anatomy_inventory
from .canonical import CanonicalGraphResult, build_canonical_graph
from .endpoints import (
    ENDPOINT_DEFINITIONS,
    ENDPOINT_FILE_COLUMNS,
    FORBIDDEN_COLUMN_MARKERS,
    OVERLAP_PAIRS,
    annotation_matches_tokens,
    build_endpoint_groups,
    endpoint_qc,
    split_nerve_tokens,
)
from .official_rules import (
    OFFICIAL_V09_REFERENCE,
    PAPER_ABSTRACT_REFERENCE,
    valid_superclass_expr,
    valid_type_expr,
)
from .type_graph import TypeGraphResult, build_type_graph

__all__ = [
    "build_canonical_graph",
    "build_type_graph",
    "build_anatomy_inventory",
    "build_endpoint_groups",
    "endpoint_qc",
    "CanonicalGraphResult",
    "TypeGraphResult",
    "AnatomyInventory",
    "ENDPOINT_DEFINITIONS",
    "ENDPOINT_FILE_COLUMNS",
    "FORBIDDEN_COLUMN_MARKERS",
    "OVERLAP_PAIRS",
    "split_nerve_tokens",
    "annotation_matches_tokens",
    "valid_superclass_expr",
    "valid_type_expr",
    "OFFICIAL_V09_REFERENCE",
    "PAPER_ABSTRACT_REFERENCE",
]
