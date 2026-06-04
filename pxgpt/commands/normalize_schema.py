"""Normalize a JSON schema for Anthropic structured outputs and write it to disk."""

import argparse

from ..core.schema_utils import normalize_and_write


def normalize_schema_command(args):
    output_path = args.output or args.schema
    in_place = not args.output

    try:
        normalized = normalize_and_write(args.schema, output_path)
    except FileNotFoundError:
        print(f"Error: schema file not found: {args.schema}")
        return 1
    except Exception as e:
        print(f"Error normalizing schema: {e}")
        return 1

    obj_count = _count_objects(normalized)
    if in_place:
        print(f"Schema normalized in-place: {args.schema}")
    else:
        print(f"Normalized schema written to: {output_path}")
    print(f"Object nodes patched: {obj_count} "
          f"(additionalProperties: false + required: [] added where missing)")
    return 0


def _count_objects(node, _n=None) -> int:
    """Count how many object nodes are in the schema tree."""
    if _n is None:
        _n = [0]
    if not isinstance(node, dict):
        return _n[0]
    if node.get("type") == "object" or "properties" in node:
        _n[0] += 1
    for child in node.get("properties", {}).values():
        _count_objects(child, _n)
    if isinstance(node.get("items"), dict):
        _count_objects(node["items"], _n)
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(key, []):
            _count_objects(sub, _n)
    return _n[0]


def setup_normalize_schema_parser(subparsers):
    parser = subparsers.add_parser(
        "normalize-schema",
        help="Normalize a JSON schema for Anthropic structured outputs",
        description=(
            "Adds ``additionalProperties: false`` and an empty ``required`` list "
            "to every object node that lacks them, strips the ``format`` keyword "
            "(unsupported by the API), and removes the ``$schema`` meta-key.  "
            "Writes the result back to the same file by default."
        ),
    )
    parser.add_argument(
        "--schema", required=True,
        help="Input JSON schema file path",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path (default: overwrite input file in-place)",
    )
    parser.set_defaults(func=normalize_schema_command)
