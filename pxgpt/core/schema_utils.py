"""Normalize a JSON schema for Anthropic structured outputs.

Changes applied:
  - Add ``additionalProperties: false`` to every object node that lacks it.
  - Add an empty ``required: []`` to every object that has a properties dict
    but no required array (existing required lists are left intact).
  - Strip the ``format`` keyword (e.g. "date") — not supported by the API.
  - Strip the root ``$schema`` meta-key (not needed in the API payload).
"""

import copy
import json
from typing import Any, Dict, Optional

_STRIP_ROOT_KEYS = {"$schema"}
_STRIP_NODE_KEYS = {"format"}   # unsupported leaf-level keywords


def normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copy of *schema* with Anthropic structured-output tweaks."""
    schema = copy.deepcopy(schema)
    for k in _STRIP_ROOT_KEYS:
        schema.pop(k, None)
    _walk(schema)
    return schema


def _walk(node: Any) -> None:
    if not isinstance(node, dict):
        return

    for k in _STRIP_NODE_KEYS:
        node.pop(k, None)

    if node.get("type") == "object" or "properties" in node:
        node.setdefault("additionalProperties", False)
        if "properties" in node:
            node.setdefault("required", [])
        for child in node.get("properties", {}).values():
            _walk(child)

    if isinstance(node.get("items"), dict):
        _walk(node["items"])

    for key in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(key, []):
            _walk(sub)


def load_normalized(schema_path: str) -> Dict[str, Any]:
    """Read a JSON schema file and return the normalized dict (file unchanged)."""
    with open(schema_path, encoding="utf-8") as f:
        raw = json.load(f)
    return normalize_schema(raw)


def normalize_and_write(schema_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
    """Normalize *schema_path* and write the result to disk.

    If *output_path* is None the source file is overwritten in-place.
    Returns the normalized schema dict.
    """
    with open(schema_path, encoding="utf-8") as f:
        raw = json.load(f)
    normalized = normalize_schema(raw)
    dest = output_path or schema_path
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
        f.write("\n")
    return normalized
