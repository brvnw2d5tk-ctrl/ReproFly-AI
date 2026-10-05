"""Neutral anatomy inventory over canonical annotations.

Counts ONLY. Every nerve / neuromere is recorded by its official
annotation code verbatim. No nerve is labelled a 'key reproductive
pathway', no node selection, no ranking. Names that themselves contain
genital/reproductive/abdominal/visceral are recorded as-is without any
functional interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .official_rules import valid_type_expr

NERVE_INVENTORY_COLUMNS = [
    "nerve", "neuron_count", "type_count", "main_neuromere", "coverage",
]
NEUROMERE_INVENTORY_COLUMNS = [
    "soma_neuromere", "neuron_count", "type_count", "coverage",
]
PERIPHERAL_INVENTORY_COLUMNS = [
    "annotation_field", "value", "neuron_count", "type_count",
]

# Annotation fields enumerated in the peripheral (long-format) inventory.
# Purely structural/annotation fields; counts only.
PERIPHERAL_FIELDS = [
    "class", "superclass", "receptorType", "rootSide",
    "somaSide", "dimorphism", "fruDsx", "birthtime",
]


@dataclass
class AnatomyInventory:
    entry_nerves: pl.DataFrame
    exit_nerves: pl.DataFrame
    neuromeres: pl.DataFrame
    peripheral: pl.DataFrame
    stats: dict


def _nerve_inventory(
    canonical: pl.DataFrame, field_name: str
) -> pl.DataFrame:
    """Per-nerve neutral counts: neuron count, distinct valid types,
    main soma neuromere, coverage (fraction of all canonical neurons)."""
    total = canonical.height
    typed = canonical.filter(valid_type_expr())
    inv = (
        canonical.filter(pl.col(field_name).is_not_null())
        .group_by(pl.col(field_name).alias("nerve"))
        .agg(
            pl.len().alias("neuron_count"),
            # distinct valid types among this nerve's neurons
            pl.col("type").filter(
                pl.col("type").is_not_null()
                & ~pl.col("type").str.contains("unclear")
                & (pl.col("type") != "Pm7_Li28")
            ).n_unique().alias("type_count"),
            pl.col("somaNeuromere").drop_nulls().mode().first()
              .alias("main_neuromere"),
        )
        .with_columns(
            (pl.col("neuron_count") / total).alias("coverage"),
        )
        .sort("nerve")
    )
    assert set(inv.columns) == set(NERVE_INVENTORY_COLUMNS)
    return inv


def build_anatomy_inventory(canonical_neurons: pl.DataFrame) -> AnatomyInventory:
    """Build all four neutral inventories from the canonical neuron table."""
    total = canonical_neurons.height
    typed = canonical_neurons.filter(valid_type_expr())

    entry = _nerve_inventory(canonical_neurons, "entryNerve")
    exit_ = _nerve_inventory(canonical_neurons, "exitNerve")

    neuro = (
        canonical_neurons.filter(pl.col("somaNeuromere").is_not_null())
        .group_by(pl.col("somaNeuromere").alias("soma_neuromere"))
        .agg(
            pl.len().alias("neuron_count"),
            pl.col("type").filter(
                pl.col("type").is_not_null()
                & ~pl.col("type").str.contains("unclear")
                & (pl.col("type") != "Pm7_Li28")
            ).n_unique().alias("type_count"),
        )
        .with_columns((pl.col("neuron_count") / total).alias("coverage"))
        .sort("soma_neuromere")
    )
    assert set(neuro.columns) == set(NEUROMERE_INVENTORY_COLUMNS)

    frames = []
    for field_name in PERIPHERAL_FIELDS:
        if field_name not in canonical_neurons.columns:
            continue
        per = (
            canonical_neurons.filter(pl.col(field_name).is_not_null())
            .group_by(pl.col(field_name).alias("value"))
            .agg(
                pl.len().alias("neuron_count"),
                pl.col("type").filter(
                    pl.col("type").is_not_null()
                    & ~pl.col("type").str.contains("unclear")
                    & (pl.col("type") != "Pm7_Li28")
                ).n_unique().alias("type_count"),
            )
            .with_columns(pl.lit(field_name).alias("annotation_field"))
            .select(PERIPHERAL_INVENTORY_COLUMNS)
        )
        frames.append(per)
    peripheral = (
        pl.concat(frames).sort(["annotation_field", "value"])
        if frames else pl.DataFrame(schema={
            c: pl.String for c in PERIPHERAL_INVENTORY_COLUMNS
        })
    )

    stats = {
        "canonical_bodies": total,
        "typed_canonical_bodies": typed.height,
        "entry_nerve_unique": entry.height,
        "entry_nerve_annotated": int(entry["neuron_count"].sum()),
        "exit_nerve_unique": exit_.height,
        "exit_nerve_annotated": int(exit_["neuron_count"].sum()),
        "soma_neuromere_unique": neuro.height,
        "soma_neuromere_annotated": int(neuro["neuron_count"].sum()),
        "entry_nerve_coverage": (
            float(entry["neuron_count"].sum() / total) if total else 0.0
        ),
        "exit_nerve_coverage": (
            float(exit_["neuron_count"].sum() / total) if total else 0.0
        ),
        "soma_neuromere_coverage": (
            float(neuro["neuron_count"].sum() / total) if total else 0.0
        ),
    }
    return AnatomyInventory(
        entry_nerves=entry,
        exit_nerves=exit_,
        neuromeres=neuro,
        peripheral=peripheral,
        stats=stats,
    )
