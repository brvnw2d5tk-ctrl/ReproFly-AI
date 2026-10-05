"""Data Access Layer (DAL) — read-only, schema-inspection-only.

Phase 0 contract:
  * Reading public assets is allowed (Feather via pyarrow/polars, H5AD via
    anndata backed mode, TSV via polars).
  * Every path goes through security.assert_path_public (private-path and
    containment checks).
  * Files above settings.eager_read_max_bytes may never be read eagerly:
    they are returned as lazy frames (polars) or backed AnnData.
  * No biological computation happens here — loading, head/schema info,
    and row counts only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from ..config import Settings
from ..exceptions import LargeFileError
from ..security.guards import assert_path_public


def _file_size(path: Path) -> int:
    return path.stat().st_size


def _read_feather_eager(path: Path) -> "pl.DataFrame":
    """Eager feather read via pyarrow -> polars (zero-copy Arrow interchange,
    never through pandas).

    pyarrow is used as the physical reader because the real MaleCNS/BANC
    feathers contain dictionary-encoded columns that polars' native IPC
    reader cannot decode ("dictionary key must fit in a usize, but -1").
    """
    import pyarrow.feather as feather

    table = feather.read_table(path)
    return pl.from_arrow(table)  # type: ignore[return-value]


def _public_path(settings: Settings, config_key: str) -> Path:
    """Resolve a configured dataset path and vet it."""
    rel = settings.dataset_paths.get(config_key)
    if rel is None:
        raise KeyError(
            f"Dataset key {config_key!r} not present in config [datasets]"
        )
    return assert_path_public(settings.data_root, rel)


# --------------------------------------------------------------------------
# MaleCNS v1.0 (public connectome assets, feather format)
# --------------------------------------------------------------------------

def load_malecns_annotations(settings: Settings) -> pl.DataFrame:
    """Load body annotations (small, ~14 MB) eagerly as a polars DataFrame.

    Purely structural data: cell ids, hemilineage / side / position etc.
    No candidate selection or scoring is performed here.
    """
    path = _public_path(settings, "malecns_annotations_rel")
    if _file_size(path) > settings.eager_read_max_bytes:
        raise LargeFileError(
            f"{path.name} grew beyond the eager-read limit; refusing full load"
        )
    return _read_feather_eager(path)


def load_malecns_synapse_partners(settings: Settings) -> pl.LazyFrame:
    """Open the syn-partners connectome table (6.8 GB) as a LazyFrame.

    Per-synapse partner rows. Since Phase 0.5 this is NOT the default edge
    source for the canonical graph — it is reserved for neuropil-specific
    validation and single-synapse drill-down. The canonical graph uses
    load_malecns_connectome_weights() instead.
    """
    path = _public_path(settings, "malecns_edges_rel")
    return pl.scan_ipc(path)


# Backwards-compatible alias (Phase 0 name; kept so no existing caller breaks).
load_malecns_edges = load_malecns_synapse_partners


def load_malecns_connectome_weights(settings: Settings) -> pl.LazyFrame:
    """Open the aggregated connectome weights table (1.1 GB, 151.9M directed
    pairs, columns body_pre/body_post/weight) as a LazyFrame.

    This is the PRIMARY connectivity source for the MaleCNS v1.0 canonical
    graph (Phase 0.5 charter). Always lazy/streaming-safe: the frame is
    never materialised by this loader; callers must filter/aggregate before
    collecting. Never passes through pandas.
    """
    path = _public_path(settings, "malecns_connectome_weights_rel")
    return pl.scan_ipc(path)


def load_malecns_neurotransmitters(settings: Settings) -> pl.DataFrame:
    """Load per-body neurotransmitter assignments (small) eagerly."""
    path = _public_path(settings, "malecns_neurotransmitters_rel")
    if _file_size(path) > settings.eager_read_max_bytes:
        raise LargeFileError(
            f"{path.name} grew beyond the eager-read limit; refusing full load"
        )
    return _read_feather_eager(path)


# --------------------------------------------------------------------------
# BANC (female adult nerve cord, v888)
# --------------------------------------------------------------------------

def load_banc_metadata(settings: Settings) -> pl.DataFrame:
    """Load BANC v888 per-cell metadata (55 MB) eagerly."""
    path = _public_path(settings, "banc_metadata_rel")
    if _file_size(path) > settings.eager_read_max_bytes:
        raise LargeFileError(
            f"{path.name} grew beyond the eager-read limit; refusing full load"
        )
    return _read_feather_eager(path)


# --------------------------------------------------------------------------
# Fly Cell Atlas — full-gene 'All' object (33.8 GB H5AD)
# --------------------------------------------------------------------------

def open_fca_h5ad_backed(settings: Settings):
    """Open the FCA All full-gene H5AD in backed read-only mode.

    Returns an AnnData with X backed on disk ('r'); no dense matrix is
    loaded. Only .shape / obs / var inspection is cheap; anything touching
    backed X rows reads chunks from disk.
    """
    import anndata as ad  # imported lazily: heavy dependency

    path = _public_path(settings, "fca_all_h5ad_rel")
    return ad.read_h5ad(path, backed="r")


# --------------------------------------------------------------------------
# Project asset index
# --------------------------------------------------------------------------

def read_project_asset_index(settings: Settings) -> pl.DataFrame:
    """Read 00_manifest/master_manifest.tsv (the download ledger)."""
    path = _public_path(settings, "asset_index_rel")
    return pl.read_csv(path, separator="\t", infer_schema_length=2000)


# --------------------------------------------------------------------------
# Generic schema inspection (never materialises data)
# --------------------------------------------------------------------------

def describe_feather_schema(path_rel: str, settings: Settings) -> dict[str, Any]:
    """Return schema metadata (names/dtypes) for any public feather file
    without loading its data columns."""
    path = assert_path_public(settings.data_root, path_rel)
    import pyarrow.ipc as ipc

    with open(path, "rb") as fh:
        reader = ipc.open_file(fh)   # reads footer/schema only, no records
        schema = reader.schema
        names = [f.name for f in schema]
        types = [str(f.type) for f in schema]
    return {
        "path": path.name,
        "columns": names,
        "dtypes": types,
        "size_bytes": _file_size(path),
    }


def h5ad_backed_summary(adata) -> dict[str, Any]:
    """Summarise a backed AnnData (shape + obs/var heads). No biology."""
    return {
        "shape": tuple(adata.shape),
        "n_obs": adata.n_obs,
        "n_vars": adata.n_vars,
        "obs_columns": list(adata.obs.columns),
        "var_columns": list(adata.var.columns),
        "backed": bool(getattr(adata, "isbacked", False)),
        "X_dtype": str(adata.X.dtype) if adata.X is not None else None,
    }


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
