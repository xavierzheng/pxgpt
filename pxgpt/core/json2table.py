"""Flatten per-cultivar Stage 3 phenotype JSON results into a wide table.

Each ``Result_Stage3/<cultivar_id>.json`` is ``{group: {trait: {rationale,
value}}}``. This module builds trait metadata (scale_type / unit / ordinal
levels) from the master schema (authoritative) with the shard schemas as a
best-effort fallback for traits master doesn't cover, then flattens all
cultivar records into one row-per-cultivar table: nominal traits stay plain
strings, quantitative traits become numeric columns named
``<trait>_<unit>``, and ordinal traits are reconstructed from their integer
level code into the schema-defined label.

Two (or more) traits can compute to the same final column name — e.g. the
same leaf key ``length`` assessed under both a ``leaf`` and a ``petal``
group. Every candidate column is tracked by its full dotted source path
(group + ... + trait name) so such collisions can be detected and resolved
via ``on_collision`` / ``rename_map`` before either output is written.

Provenance travels with the data: each record's ``_provenance`` block (see
:mod:`pxgpt.core.provenance`) becomes the ``provider`` / ``model`` /
``schema_version`` columns of BOTH outputs, and the whole block is repeated in
the feather file's Arrow schema metadata under ``pxgpt_provenance``.  Records
from two different runs in one directory are refused unless the caller says
otherwise, because a table whose rows come from two models is only useful when
that is on purpose.
"""

import glob
import json
import logging
import os
import re
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pyarrow import feather as pa_feather

from .provenance import (
    PROVENANCE_KEY,
    RESERVED_TABLE_COLUMNS,
    block_key,
    identity_tuple,
    is_reserved_record_key,
)

NA_SENTINEL = "not_assessable"

#: Per-row provenance columns, inserted right after ``cultivar_id``.
PROVENANCE_COLUMNS = ("provider", "model", "schema_version")

#: Arrow schema metadata key holding the JSON-encoded provenance.
FEATHER_PROVENANCE_KEY = b"pxgpt_provenance"

logger = logging.getLogger(__name__)


class ColumnCollisionError(Exception):
    """Raised when ``on_collision="error"`` and duplicate final column names remain.

    Also raised, in every mode, when a trait resolves onto one of the reserved
    provenance column names: those are never renamed away from under the reader.
    """

    def __init__(self, clashes, reserved=None):
        # clashes: OrderedDict[final_name -> list[(path_tuple, meta_dict)]]
        self.clashes = clashes
        self.reserved = set(reserved or ())
        super().__init__(format_collision_message(clashes, self.reserved))


class MixedProvenanceError(Exception):
    """Raised when one result dir holds records from more than one run."""

    def __init__(self, by_identity):
        # by_identity: OrderedDict[(provider, model, schema_version) -> [cultivar_id]]
        self.by_identity = by_identity
        lines = ["Refusing to flatten records from more than one run:"]
        for (provider, model, schema_version), ids in by_identity.items():
            shown = ", ".join(ids[:4]) + (" ..." if len(ids) > 4 else "")
            lines.append(f"  provider={provider!r} model={model!r} "
                         f"schema_version={schema_version!r}  <- {len(ids)} "
                         f"record(s): {shown}")
        lines.append("")
        lines.append(
            "One table whose rows come from different models (or different "
            "schema versions) reads as one experiment and is not one.  Either "
            "split the directory, or pass --allow-mixed-provenance if the "
            "mixture is deliberate — the provider/model/schema_version columns "
            "then carry the truth per row."
        )
        super().__init__("\n".join(lines))


class DuplicateColumnError(Exception):
    """Raised by the final safety net when resolved column names still duplicate."""

    def __init__(self, duplicates):
        # duplicates: OrderedDict[final_name -> list[path_tuple]]
        self.duplicates = duplicates
        lines = ["Duplicate column name(s) after resolution:"]
        for name, paths in duplicates.items():
            lines.append(f" {name!r} <- " + ", ".join(_dotted(p) for p in paths))
        super().__init__("\n".join(lines))


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


def _dotted(path):
    return ".".join(path)


def _descriptor(meta):
    if meta["scale_type"] == "quantitative":
        return f"quantitative, unit={meta.get('unit')}"
    return meta["scale_type"]


def _prefixed_name(path, base_name, depth):
    """Return *base_name* prefixed with up to ``depth - 1`` path components.

    ``depth=1`` returns *base_name* unprefixed. ``depth=len(path)`` returns
    the full underscore-joined path (base_name already carries the trait's
    own unit suffix).
    """
    depth = min(depth, len(path))
    prefix_components = path[-depth:-1]
    if not prefix_components:
        return base_name
    return "_".join(prefix_components) + "_" + base_name


def format_collision_message(clashes, reserved=None):
    """Render the "error" mode message: clash list + a rename_map fill-in template."""
    reserved = set(reserved or ())
    lines = ["Column name collision(s). Unresolved:"]
    template = OrderedDict()
    for final_name, entries in clashes.items():
        parts = [f"{_dotted(path)} ({_descriptor(meta)})" for path, meta in entries]
        lines.append(f" {final_name!r} <- " + ", ".join(parts))
        for path, _meta in entries:
            template[_dotted(path)] = ""

    lines.append("")
    if reserved:
        lines.append(
            "Reserved column name(s): "
            + ", ".join(repr(n) for n in sorted(reserved))
            + ".  " + ", ".join(repr(n) for n in RESERVED_TABLE_COLUMNS)
            + " belong to the provenance carried in every row and are never "
              "renamed; rename the trait instead."
        )
        lines.append("")
    lines.append(
        "Fill in this rename map (path -> column name) and re-run with --rename-map:"
    )
    lines.append(" {")
    keys = list(template.keys())
    for i, key in enumerate(keys):
        comma = "," if i < len(keys) - 1 else ""
        lines.append(f' {json.dumps(key)}: ""{comma}')
    lines.append(" }")
    return "\n".join(lines)


def resolve_column_names(paths_with_meta, on_collision="error", rename_map=None,
                         reserved=RESERVED_TABLE_COLUMNS):
    """Return ``OrderedDict(path_tuple -> final_column_name)`` for *paths_with_meta*.

    ``paths_with_meta`` is an ordered list of ``(path_tuple, meta_dict)`` for
    every trait present in the results, in traversal/schema order.

    Resolution order: ``rename_map`` overrides (literal, no unit re-appended)
    are applied first; ``on_collision`` then resolves whatever still clashes
    among the un-overridden columns; a final global-uniqueness safety net
    always runs, regardless of mode, and raises ``DuplicateColumnError`` if
    anything still duplicates.
    """
    valid_modes = ("error", "prefix_collided", "prefix_all")
    if on_collision not in valid_modes:
        raise ValueError(f"on_collision must be one of {valid_modes}, got {on_collision!r}")

    rename_map = rename_map or {}
    base_names = OrderedDict()
    meta_by_path = OrderedDict()
    for path, meta in paths_with_meta:
        base_names[path] = _column_name(path[-1], meta)
        meta_by_path[path] = meta

    resolved = OrderedDict()
    overridden = set()
    for path in base_names:
        dotted = _dotted(path)
        if dotted in rename_map:
            resolved[path] = rename_map[dotted]
            overridden.add(path)

    groups = OrderedDict()
    for path in base_names:
        if path in overridden:
            continue
        groups.setdefault(base_names[path], []).append(path)
    clashing = OrderedDict((name, ps) for name, ps in groups.items() if len(ps) > 1)

    if clashing:
        if on_collision == "error":
            clashes = OrderedDict(
                (name, [(p, meta_by_path[p]) for p in ps]) for name, ps in clashing.items()
            )
            raise ColumnCollisionError(clashes)
        if on_collision == "prefix_collided":
            for _name, ps in clashing.items():
                depth = 2
                max_depth = max(len(p) for p in ps)
                while True:
                    candidates = {p: _prefixed_name(p, base_names[p], depth) for p in ps}
                    if len(set(candidates.values())) == len(ps) or depth >= max_depth:
                        break
                    depth += 1
                resolved.update(candidates)

    for path in base_names:
        if path in resolved:
            continue
        if on_collision == "prefix_all":
            resolved[path] = _prefixed_name(path, base_names[path], len(path))
        else:
            resolved[path] = base_names[path]

    # A trait landing on ``cultivar_id`` / ``provider`` / ``model`` /
    # ``schema_version`` would overwrite the row's own identity.  Refuse in every
    # mode, including the prefix ones: silently renaming a reserved column is
    # how a reader ends up plotting a trait as if it were the model name.
    reserved_hits = OrderedDict()
    for path in base_names:
        name = resolved[path]
        if name in (reserved or ()):
            reserved_hits.setdefault(name, []).append((path, meta_by_path[path]))
    if reserved_hits:
        raise ColumnCollisionError(reserved_hits, reserved=set(reserved_hits))

    seen = OrderedDict()
    for path in base_names:
        seen.setdefault(resolved[path], []).append(path)
    duplicates = OrderedDict((name, ps) for name, ps in seen.items() if len(ps) > 1)
    if duplicates:
        raise DuplicateColumnError(duplicates)

    return resolved


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
    """Return ``(key_order, trait_meta)`` describing every trait master/shards know.

    Traits are keyed by ``(group_name, trait_name)`` — the same trait_name
    can legitimately mean different things (e.g. a different unit) under
    different groups. ``key_order`` lists these keys in master-schema order,
    then any shard-only keys (shard file order). ``trait_meta[key]`` is
    ``{"scale_type", "unit", "levels"}`` where ``levels`` is an
    ``OrderedDict(code -> label)`` for ordinal traits (``None`` otherwise).
    Master is authoritative; a shard-only key is filled in on a best-effort
    basis (shard schemas carry no unit, and ordinal shard enums carry only
    integer codes with no labels).
    """
    key_order = []
    trait_meta = {}

    with open(master_schema_path, encoding="utf-8") as f:
        master = json.load(f)
    from . import shard_builder
    master = shard_builder.normalize_master(master)
    for group_name, group in master["trait_groups"].items():
        for trait in group.get("traits", []):
            key = (group_name, trait["trait_name"])
            trait_meta[key] = {
                "scale_type": trait["scale_type"],
                "unit": trait.get("unit"),
                "levels": _levels_from_master_trait(trait),
            }
            key_order.append(key)

    for shard_path in shard_schema_paths or []:
        with open(shard_path, encoding="utf-8") as f:
            shard = json.load(f)
        for group_name, group_schema in shard.get("properties", {}).items():
            for trait_name, trait_schema in group_schema.get("properties", {}).items():
                key = (group_name, trait_name)
                if key in trait_meta:
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
                    trait_meta[key] = {
                        "scale_type": "ordinal", "unit": None, "levels": levels,
                    }
                else:
                    trait_meta[key] = {
                        "scale_type": "nominal", "unit": None, "levels": None,
                    }
                key_order.append(key)

    return key_order, trait_meta


def _flatten_leaves(node, path, out):
    """Recursively collect ``path_tuple -> value`` for every leaf trait dict.

    A leaf is any dict carrying a ``"value"`` key; everything above it
    (group, and any intermediate sub-group keys) becomes part of the path.
    """
    if isinstance(node, dict) and "value" in node:
        out[path] = node["value"]
        return
    if isinstance(node, dict):
        for key, child in node.items():
            _flatten_leaves(child, path + (key,), out)


def load_cultivar_records(result_dir):
    """Return ``(records, paths_seen, provenance)``.

    ``records`` is ``OrderedDict(cultivar_id -> dict(path_tuple -> raw_value))``
    in sorted-filename order; ``paths_seen`` is the set of every full source
    path (``(group, ..., trait_name)``) encountered across all files;
    ``provenance`` maps cultivar_id to that record's ``_provenance`` block for
    the records that carry one.

    ``<cultivar>.gaps.json`` sidecars are skipped: they sit in the same
    directory but describe what is MISSING from a record, so treating one as a
    record adds an all-NA row named ``<cultivar>.gaps``.  Top-level keys
    starting with ``_`` are metadata, never trait groups.
    """
    records = OrderedDict()
    paths_seen = set()
    provenance = {}
    for path in sorted(glob.glob(os.path.join(result_dir, "*.json"))):
        if path.endswith(".gaps.json"):
            continue
        cultivar_id = Path(path).stem
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        block = data.get(PROVENANCE_KEY)
        if isinstance(block, dict):
            provenance[cultivar_id] = block
        flat = {}
        for group_name, group_val in data.items():
            if is_reserved_record_key(group_name):
                continue
            if not isinstance(group_val, dict):
                continue
            _flatten_leaves(group_val, (group_name,), flat)
        records[cultivar_id] = flat
        paths_seen.update(flat.keys())
    return records, paths_seen, provenance


def resolve_provenance(result_dir, cultivar_ids, record_provenance,
                       allow_mixed=False):
    """Return ``(per_cultivar, metadata, warnings)`` for the whole result dir.

    Records written by an older pxGPT carry no block.  Those fall back to the
    run stamp in ``<result_dir>/_partial/.run.json`` when there is one, and to
    real NA when there is not — one warning either way, never a hard failure,
    because a legacy directory must still flatten.

    Raises :class:`MixedProvenanceError` when the records disagree on
    ``(provider, model, schema_version)`` and *allow_mixed* is false.
    """
    warnings = []
    per_cultivar = {cid: record_provenance[cid]
                    for cid in cultivar_ids if cid in record_provenance}

    missing = [cid for cid in cultivar_ids if cid not in per_cultivar]
    if missing:
        stamp = _read_run_stamp(result_dir)
        if stamp is not None:
            fallback = OrderedDict([
                ("provider", stamp.get("provider")),
                ("model", stamp.get("model")),
                ("schema_version", stamp.get("schema_version")),
            ])
            warnings.append(
                f"{len(missing)} record(s) carry no {PROVENANCE_KEY} (written by "
                f"an older pxGPT); filled provider/model from "
                f"{os.path.join(result_dir, '_partial', '.run.json')}"
            )
        else:
            fallback = OrderedDict([(field, None) for field in PROVENANCE_COLUMNS])
            warnings.append(
                f"{len(missing)} record(s) carry no {PROVENANCE_KEY} and there is "
                f"no _partial/.run.json to fall back on; "
                f"{', '.join(PROVENANCE_COLUMNS)} are NA"
            )
        for cid in missing:
            per_cultivar[cid] = fallback

    by_identity = OrderedDict()
    for cid in cultivar_ids:
        by_identity.setdefault(identity_tuple(per_cultivar.get(cid)), []).append(cid)
    if len(by_identity) > 1 and not allow_mixed:
        raise MixedProvenanceError(by_identity)

    blocks = OrderedDict()
    for cid in cultivar_ids:
        blocks.setdefault(block_key(per_cultivar.get(cid)), per_cultivar.get(cid))
    distinct = [b for b in blocks.values()]
    if len(distinct) == 1:
        metadata = distinct[0]
    else:
        # More than one distinct block.  Usually two models; it can also be two
        # merges of the SAME run, which differ only in ``created`` — the
        # identity check above is what decides whether that needed permission.
        metadata = OrderedDict([("mixed", True), ("values", distinct)])
    return per_cultivar, metadata, warnings


def _read_run_stamp(result_dir):
    """Return the parsed ``_partial/.run.json``, or None when absent/unreadable."""
    path = os.path.join(result_dir, "_partial", ".run.json")
    try:
        with open(path, encoding="utf-8") as f:
            stamp = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return stamp if isinstance(stamp, dict) else None


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


def build_table(
    result_dir, master_schema_path, shard_dir=None, on_collision="error",
    rename_map=None, allow_mixed_provenance=False,
):
    """Flatten *result_dir* into ``(csv_df, feather_df, warnings, provenance)``.

    ``csv_df``: plain values (ordinal as label strings, nominal as strings,
    quantitative numeric) — write directly with ``to_csv``.
    ``feather_df``: identical except ordinal columns are ordered
    ``pd.Categorical`` over the full schema-defined level set.
    ``provenance``: the block to embed in the feather metadata — hand it to
    :func:`write_table`.

    Both frames start with ``cultivar_id`` followed by the ``provider`` /
    ``model`` / ``schema_version`` columns, so a CSV reader is never a
    second-class citizen of the provenance.

    Raises ``ColumnCollisionError`` (``on_collision="error"``, the default; also
    any mode when a trait lands on a reserved column name),
    ``DuplicateColumnError`` (final safety net, any mode), or
    ``MixedProvenanceError`` (records from more than one run, unless
    *allow_mixed_provenance*).  No output is produced in those cases.
    """
    warnings = []
    shard_paths = discover_shard_schema_paths(shard_dir)
    key_order, trait_meta = build_trait_metadata(master_schema_path, shard_paths)

    records, paths_seen, record_provenance = load_cultivar_records(result_dir)
    # Before any flattening work: refuse a mixed directory while nothing is written.
    prov_by_cultivar, prov_metadata, prov_warnings = resolve_provenance(
        result_dir, list(records.keys()), record_provenance,
        allow_mixed=allow_mixed_provenance,
    )
    warnings.extend(prov_warnings)

    paths_by_key = OrderedDict()
    for path in sorted(paths_seen):
        key = (path[0], path[-1])
        paths_by_key.setdefault(key, []).append(path)

    unknown_keys = sorted(key for key in paths_by_key if key not in trait_meta)
    for key in unknown_keys:
        warnings.append(
            f"trait {key[1]!r} found in results but absent from master schema and "
            f"shard schemas; treating as a plain string column"
        )
        trait_meta[key] = {"scale_type": "nominal", "unit": None, "levels": None}
    key_order = key_order + unknown_keys

    present = []
    for key in key_order:
        for path in paths_by_key.get(key, []):
            present.append((path, trait_meta[key]))

    resolved_names = resolve_column_names(present, on_collision=on_collision, rename_map=rename_map)

    csv_data = OrderedDict()
    cultivar_ids = list(records.keys())
    csv_data["cultivar_id"] = cultivar_ids
    for column in PROVENANCE_COLUMNS:
        csv_data[column] = [
            (prov_by_cultivar.get(cid) or {}).get(column) for cid in cultivar_ids
        ]
    for path, meta in present:
        col_name = resolved_names[path]
        values = []
        for cultivar_id, flat in records.items():
            raw = flat.get(path)
            values.append(_convert_value(raw, meta, cultivar_id, path[-1], warnings))
        csv_data[col_name] = values

    csv_df = pd.DataFrame(csv_data)

    feather_df = csv_df.copy()
    for path, meta in present:
        if meta["scale_type"] != "ordinal":
            continue
        col_name = resolved_names[path]
        levels = meta.get("levels")
        categories = list(levels.values()) if levels else None
        feather_df[col_name] = pd.Categorical(
            feather_df[col_name], categories=categories, ordered=True
        )

    return csv_df, feather_df, warnings, prov_metadata


def write_table(csv_df, feather_df, out_prefix, provenance=None):
    """Write ``<out_prefix>.csv`` and ``<out_prefix>.feather``.

    *provenance* (from :func:`build_table`) is JSON-encoded into the feather
    file's Arrow schema metadata under ``pxgpt_provenance``, so the block
    survives even if the columns are dropped downstream.  It is added to the
    metadata pandas itself writes, never in place of it: the ``b"pandas"`` entry
    is what makes an ordinal column come back as an ORDERED Categorical.
    """
    out_path = Path(out_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    csv_df.to_csv(f"{out_prefix}.csv", index=False, encoding="utf-8")

    table = pa.Table.from_pandas(feather_df, preserve_index=False)
    if provenance is not None:
        metadata = dict(table.schema.metadata or {})
        metadata[FEATHER_PROVENANCE_KEY] = json.dumps(
            provenance, ensure_ascii=False
        ).encode("utf-8")
        table = table.replace_schema_metadata(metadata)
    pa_feather.write_feather(table, f"{out_prefix}.feather", version=2)
