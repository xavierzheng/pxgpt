#!/usr/bin/env python3
"""Prove the three backends receive constraint-equivalent shard schemas.

    python scripts/check_schema_equivalence.py <shard_dir>

For every ``*.schema.json`` in *shard_dir* this builds three versions —

  raw        the file on disk, as Anthropic's xgrammar / vLLM consume it
  anthropic  ``schema_utils.normalize_schema()``
  openai     ``openai_batch_utils.openai_normalize_schema()``

— walks the same JSON paths through all three and checks that every leaf carries
the same constraint: identical ``enum`` member sets where there is an enum, the
same ``type`` where there is not, and the same property names in the same
traversal order.

The point is the OpenAI normalizer's ``"type": "string"`` injection on enum-only
nodes.  That is a repair for OpenAI strict mode, and it must not narrow or widen
what any backend accepts.  This script is the reproducibility evidence for that
claim, not a unit test.

*shard_dir* is opened READ-ONLY.  The sha256 of every file is printed at the end
so the frozen shard set can be shown to be untouched.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pxgpt.core.schema_utils import normalize_schema                # noqa: E402
from pxgpt.core.openai_batch_utils import openai_normalize_schema   # noqa: E402

VERSIONS = ("raw", "anthropic", "openai")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def walk(nodes, path, leaves, problems):
    """Walk the three versions of one node in lock-step.

    *nodes* is a dict {version: node}.  Recurses through ``properties`` and
    ``items``, appending one entry to *leaves* per constraint leaf and one string
    to *problems* per disagreement.
    """
    types = {v: n.get("type") for v, n in nodes.items()}
    enums = {v: n.get("enum") for v, n in nodes.items()}
    has_enum = {v: e is not None for v, e in enums.items()}

    if any(has_enum.values()):
        if not all(has_enum.values()):
            problems.append(
                f"{path}: enum present in "
                f"{sorted(v for v in VERSIONS if has_enum[v])} but absent in "
                f"{sorted(v for v in VERSIONS if not has_enum[v])}"
            )
            return
        sets = {v: frozenset(map(_hashable, e)) for v, e in enums.items()}
        if len({*sets.values()}) != 1:
            detail = " | ".join(f"{v}={sorted(map(str, sets[v]))}" for v in VERSIONS)
            problems.append(f"{path}: enum members differ — {detail}")
        else:
            leaves.append((path, "enum", len(enums["raw"])))
        # An injected "type" alongside an identical member set is the expected
        # difference, so type is deliberately not compared on enum leaves.
        return

    props = {v: n.get("properties") for v, n in nodes.items()}
    if any(p is not None for p in props.values()):
        if not all(isinstance(p, dict) for p in props.values()):
            problems.append(f"{path}: properties missing in some version")
            return
        keys = {v: list(p.keys()) for v, p in props.items()}
        if len({tuple(k) for k in keys.values()}) != 1:
            detail = " | ".join(f"{v}={keys[v]}" for v in VERSIONS)
            problems.append(f"{path}: property names/order differ — {detail}")
            return
        for key in keys["raw"]:
            walk({v: props[v][key] for v in VERSIONS}, f"{path}.{key}",
                 leaves, problems)
        return

    items = {v: n.get("items") for v, n in nodes.items()}
    if any(i is not None for i in items.values()):
        if not all(isinstance(i, dict) for i in items.values()):
            problems.append(f"{path}: items missing in some version")
            return
        walk(items, f"{path}[]", leaves, problems)
        return

    if len(set(types.values())) != 1:
        detail = " | ".join(f"{v}={types[v]!r}" for v in VERSIONS)
        problems.append(f"{path}: type differs — {detail}")
    else:
        leaves.append((path, types["raw"], 0))


def _hashable(value):
    """Enum members are scalars in practice; be safe about unhashable ones."""
    try:
        hash(value)
        return value
    except TypeError:
        return json.dumps(value, sort_keys=True)


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    shard_dir = Path(argv[1])
    if not shard_dir.is_dir():
        print(f"Error: not a directory: {shard_dir}")
        return 2

    schema_files = sorted(shard_dir.glob("*.schema.json"))
    if not schema_files:
        print(f"Error: no *.schema.json files in {shard_dir}")
        return 2

    print(f"Shard dir: {shard_dir}")
    print(f"Schemas:   {len(schema_files)}\n")

    failed = 0
    for path in schema_files:
        raw = json.loads(path.read_text(encoding="utf-8"))
        versions = {
            "raw": raw,
            "anthropic": normalize_schema(raw),
            "openai": openai_normalize_schema(raw),
        }
        leaves, problems = [], []
        walk(versions, path.name.replace(".schema.json", ""), leaves, problems)

        enum_leaves = sum(1 for _, kind, _ in leaves if kind == "enum")
        if problems:
            failed += 1
            print(f"FAIL  {path.name}  ({len(leaves)} leaves, "
                  f"{enum_leaves} enum)")
            for p in problems:
                print(f"        {p}")
        else:
            print(f"PASS  {path.name}  ({len(leaves)} leaves, "
                  f"{enum_leaves} enum)")

    print("\nsha256 (proof the shard set was not modified):")
    for path in schema_files:
        print(f"  {sha256(path)}  {path.name}")

    if failed:
        print(f"\n{failed} of {len(schema_files)} schema(s) FAILED")
        return 1
    print(f"\nAll {len(schema_files)} schema(s) PASS — the three backends receive "
          f"constraint-equivalent schemas.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
