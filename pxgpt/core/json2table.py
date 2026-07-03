"""Flatten per-cultivar Stage 3 phenotype JSON results into a wide table.

Each ``Result_Stage3/<cultivar_id>.json`` is ``{group: {trait: {rationale,
value}}}``. This module builds trait metadata (scale_type / unit / ordinal
levels) from the master schema (authoritative) with the shard schemas as a
best-effort fallback for traits master doesn't cover, then flattens all
cultivar records into one row-per-cultivar table: nominal traits stay plain
strings, quantitative traits become numeric columns named
``<trait>_<unit>``, and ordinal traits are reconstructed from their integer
level code into the schema-defined label.
"""

import glob
import json
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd

NA_SENTINEL = "not_assessable"

logger = logging.getLogger(__name__)


def _sanitize_unit(unit):
    """Lowercase, superscript-normalize and collapse *unit* to a column-safe token."""
    if not unit:
        return ""
    token = unit.strip().lower()
    token = token.replace("²", "2").replace("³", "3")
    token = re.sub(r"[^a-z0-9]+", "_", token)
    return token.strip("_")


def _column_name(trait_name, meta):
    if meta["scale_type"] == "quantitative":
        unit_token = _sanitize_unit(meta.get("unit"))
        return f"{trait_name}_{unit_token}" if unit_token else trait_name
    return trait_name


def _levels_from_master_trait(trait):
    """Return an ``OrderedDict(code -> label)`` sorted by level, or ``None``."""
    if trait.get("scale_type") != "ordinal" or not trait.get("values"):
        return None
    ordered = sorted(trait["values"], key=lambda v: v["level"])
    return OrderedDict((v["level"], v["label"]) for v in ordered)


def discover_shard_schema_paths(shard_dir):
    """Return sorted ``shard_*.schema.json`` paths under *shard_dir*."""
    if not shard_dir:
        return []
    return sorted(glob.glob(os.path.join(shard_dir, "shard_*.schema.json")))


def build_trait_metadata(master_schema_path, shard_schema_paths=None):
    """Return ``(trait_order, trait_meta)`` describing every trait master/shards know.

    ``trait_order`` lists trait names in master-schema order, then any
    shard-only traits (shard file order). ``trait_meta[trait_name]`` is
    ``{"scale_type", "unit", "levels"}`` where ``levels`` is an
    ``OrderedDict(code -> label)`` for ordinal traits (``None`` otherwise).
    Master is authoritative; a shard-only trait is filled in on a best-effort
    basis (shard schemas carry no unit, and ordinal shard enums carry only
    integer codes with no labels).
    """
    trait_order = []
    trait_meta = {}

    with open(master_schema_path, encoding="utf-8") as f:
        master = json.load(f)
    for group in master.get("trait_groups", {}).values():
        for trait in group.get("traits", []):
            name = trait["trait_name"]
            trait_meta[name] = {
                "scale_type": trait["scale_type"],
                "unit": trait.get("unit"),
                "levels": _levels_from_master_trait(trait),
            }
            trait_order.append(name)

    for shard_path in shard_schema_paths or []:
        with open(shard_path, encoding="utf-8") as f:
            shard = json.load(f)
        for group_schema in shard.get("properties", {}).values():
            for trait_name, trait_schema in group_schema.get("properties", {}).items():
                if trait_name in trait_meta:
                    continue
                value_enum = (
                    trait_schema.get("properties", {}).get("value", {}).get("enum", [])
                )
                codes = sorted(v for v in value_enum if isinstance(v, int))
                if codes:
                    logger.warning(
                        "trait %r found only in shard schema %s; ordinal labels "
                        "are unavailable there, using the numeric code as the "
                        "label", trait_name, shard_path,
                    )
                    levels = OrderedDict((c, str(c)) for c in codes)
                    trait_meta[trait_name] = {
                        "scale_type": "ordinal", "unit": None, "levels": levels,
                    }
                else:
                    trait_meta[trait_name] = {
                        "scale_type": "nominal", "unit": None, "levels": None,
                    }
                trait_order.append(trait_name)

    return trait_order, trait_meta


def load_cultivar_records(result_dir):
    """Return ``(records, trait_names_seen)``.

    ``records`` is ``OrderedDict(cultivar_id -> dict(trait_name -> raw_value))``
    in sorted-filename order; ``trait_names_seen`` is the set of every trait
    name encountered across all files.
    """
    records = OrderedDict()
    trait_names_seen = set()
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        cultivar_id = Path(path).stem
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        flat = {}
        for traits in data.values():
            if not isinstance(traits, dict):
                continue
            for trait_name, obj in traits.items():
                if not isinstance(obj, dict) or "value" not in obj:
                    continue
                flat[trait_name] = obj["value"]
                trait_names_seen.add(trait_name)
        records[cultivar_id] = flat
    return records, trait_names_seen


def _convert_value(raw, meta, cultivar_id, trait_name, warnings):
    """Return the (identical) csv/feather-ready value for *raw*, or ``None`` for NA."""
    if raw is None or raw == NA_SENTINEL:
        return None

    scale_type = meta["scale_type"]
    if scale_type == "quantitative":
        try:
            return float(raw)
        except (TypeError, ValueError):
            warnings.append(
                f"{cultivar_id}: trait {trait_name!r} has non-numeric value {raw!r}; using NA"
            )
            return None

    if scale_type == "ordinal":
        levels = meta.get("levels")
        if levels is None:
            return str(raw)
        label = levels.get(raw)
        if label is None:
            warnings.append(
                f"{cultivar_id}: trait {trait_name!r} has unknown ordinal code "
                f"{raw!r}; using NA"
            )
            return None
        return label

    return str(raw)


def build_table(result_dir, master_schema_path, shard_dir=None):
    """Flatten *result_dir*'s per-cultivar JSON into (csv_df, feather_df, warnings).

    ``csv_df``: plain values (ordinal as label strings, nominal as strings,
    quantitative numeric) — write directly with ``to_csv``.
    ``feather_df``: identical except ordinal columns are ordered
    ``pd.Categorical`` over the full schema-defined level set.
    """
    warnings = []
    shard_paths = discover_shard_schema_paths(shard_dir)
    trait_order, trait_meta = build_trait_metadata(master_schema_path, shard_paths)

    records, trait_names_seen = load_cultivar_records(result_dir)

    unknown_traits = sorted(trait_names_seen - set(trait_meta))
    for name in unknown_traits:
        warnings.append(
            f"trait {name!r} found in results but absent from master schema and "
            f"shard schemas; treating as a plain string column"
        )
        trait_meta[name] = {"scale_type": "nominal", "unit": None, "levels": None}
    trait_order = trait_order + unknown_traits

    present_traits = [t for t in trait_order if t in trait_names_seen]

    columns = OrderedDict()  # trait_name -> column name (dedup, first-seen order)
    for trait_name in present_traits:
        columns[trait_name] = _column_name(trait_name, trait_meta[trait_name])

    csv_data = OrderedDict()
    csv_data["cultivar_id"] = list(records.keys())
    for trait_name, col_name in columns.items():
        meta = trait_meta[trait_name]
        values = []
        for cultivar_id, flat in records.items():
            raw = flat.get(trait_name)
            values.append(_convert_value(raw, meta, cultivar_id, trait_name, warnings))
        csv_data[col_name] = values

    csv_df = pd.DataFrame(csv_data)

    feather_df = csv_df.copy()
    for trait_name, col_name in columns.items():
        meta = trait_meta[trait_name]
        if meta["scale_type"] != "ordinal":
            continue
        levels = meta.get("levels")
        categories = list(levels.values()) if levels else None
        feather_df[col_name] = pd.Categorical(
            feather_df[col_name], categories=categories, ordered=True
        )

    return csv_df, feather_df, warnings


def write_table(csv_df, feather_df, out_prefix):
    """Write ``<out_prefix>.csv`` and ``<out_prefix>.feather``."""
    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df.to_csv(f"{out_prefix}.csv", index=False, encoding="utf-8")
    feather_df.to_feather(f"{out_prefix}.feather", version=2)
