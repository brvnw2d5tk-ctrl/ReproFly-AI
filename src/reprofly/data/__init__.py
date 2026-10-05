"""Read-only Data Access Layer."""

from __future__ import annotations

from .access import (
    describe_feather_schema,
    h5ad_backed_summary,
    load_banc_metadata,
    load_malecns_annotations,
    load_malecns_connectome_weights,
    load_malecns_edges,
    load_malecns_neurotransmitters,
    load_malecns_synapse_partners,
    open_fca_h5ad_backed,
    read_project_asset_index,
)

__all__ = [
    "load_malecns_annotations",
    "load_malecns_synapse_partners",
    "load_malecns_edges",
    "load_malecns_connectome_weights",
    "load_malecns_neurotransmitters",
    "load_banc_metadata",
    "open_fca_h5ad_backed",
    "read_project_asset_index",
    "describe_feather_schema",
    "h5ad_backed_summary",
]
