"""Record-level provenance for merged Stage 3 phenotype records.

``_partial/.run.json`` stamps a *store* with the run that created it (see
:mod:`pxgpt.core.batch_utils`).  That guard dies with the directory: copy
``<line_id>.json`` somewhere else and its origin is gone.  So every merged
record also carries its own ``_provenance`` block, and
``pxgpt json-to-table`` carries that block into both outputs.

The block is written once per run and repeated verbatim into every record of
that run, so two records that disagree really do come from two different runs.

``_provenance`` is a RESERVED top-level key: it is not a trait group.  Any
consumer that iterates a merged record's top-level keys must skip keys that
start with ``_`` — use :func:`is_reserved_record_key` rather than testing for
this one name, so a later ``_something`` block cannot be mistaken for a group.
"""

import json
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from .. import __version__

#: Top-level key holding the provenance block inside a merged record.
PROVENANCE_KEY = "_provenance"

#: Column names ``json-to-table`` owns.  A trait that resolves to one of these
#: is a collision, never a silent overwrite.
RESERVED_TABLE_COLUMNS = ("cultivar_id", "provider", "model", "schema_version")

#: The three fields that decide whether two records are "the same provenance".
#: ``created`` / ``run_id`` differ between two merges of the same run, and
#: ``schema_name`` / ``pxgpt_version`` add nothing the version already says.
IDENTITY_FIELDS = ("provider", "model", "schema_version")

#: Fields of the block, in write order.
BLOCK_FIELDS = ("provider", "model", "schema_name", "schema_version",
                "pxgpt_version", "created", "run_id")


def utc_now() -> str:
    """Return the current UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_provenance(
    provider: str,
    model: str,
    schema_name: Optional[str] = None,
    schema_version: Optional[str] = None,
    run_id: Optional[str] = None,
) -> "OrderedDict[str, Any]":
    """Return the ``_provenance`` block for one run.

    *provider* / *model* must be the same strings the ``.run.json`` stamp uses,
    so a record and the store it came from can be compared field by field.
    ``schema_name`` / ``schema_version`` are None wherever the run never opens a
    master schema (the manifest carries no schema identity — see
    :func:`read_schema_identity`), and ``run_id`` is None for dispatch paths
    that have no batch id.  A null is the honest answer there; a guessed value
    would be worse than none.
    """
    return OrderedDict([
        ("provider", provider),
        ("model", model),
        ("schema_name", schema_name),
        ("schema_version", schema_version),
        ("pxgpt_version", __version__),
        ("created", utc_now()),
        ("run_id", run_id),
    ])


def read_schema_identity(master_path: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(schema_name, schema_version)`` from a master schema, or ``(None, None)``.

    Only the two top-level strings are used, so this is cheap enough to call
    once per run and safe to call on any JSON file: a document without them
    (e.g. a bare JSON Schema handed to ``--schema``) yields ``(None, None)``
    rather than an error.  ``shards_manifest.json`` deliberately is NOT a
    source: its ``version`` field is the manifest format version, and it
    carries no schema name or schema version at all.
    """
    if not master_path:
        return None, None
    try:
        with open(master_path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None
    if not isinstance(doc, dict):
        return None, None
    name = doc.get("schema_name")
    version = doc.get("schema_version")
    return (name if isinstance(name, str) else None,
            version if isinstance(version, str) else None)


def is_reserved_record_key(key: Any) -> bool:
    """True for a merged-record top-level key that is metadata, not a trait group."""
    return isinstance(key, str) and key.startswith("_")


def stamp_record(record: Dict[str, Any], provenance: Dict[str, Any]) -> "OrderedDict[str, Any]":
    """Return *record* with ``_provenance`` first, replacing any block already there.

    Rebuilt at write time from the current run's identity, so re-merging (an
    idempotent ``fetch-results``, a sequential resume) can never leave a stale
    or duplicated block behind.
    """
    out: "OrderedDict[str, Any]" = OrderedDict()
    out[PROVENANCE_KEY] = provenance
    for key, value in record.items():
        if key == PROVENANCE_KEY:
            continue
        out[key] = value
    return out


def identity_tuple(provenance: Optional[Dict[str, Any]]) -> Tuple[Optional[str], ...]:
    """Return ``(provider, model, schema_version)`` for *provenance* (None-safe)."""
    prov = provenance or {}
    return tuple(prov.get(field) for field in IDENTITY_FIELDS)


def block_key(provenance: Optional[Dict[str, Any]]) -> str:
    """Return a stable string key for deduplicating whole provenance blocks."""
    return json.dumps(provenance or {}, sort_keys=True, ensure_ascii=False)
