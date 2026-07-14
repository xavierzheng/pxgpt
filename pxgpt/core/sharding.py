"""Stage 3 schema-sharding helpers.

The full Stage 3 structured-output schema is too large to compile into a
constrained-decoding grammar (400 "Schema is too complex").  ``build_stage3.py``
(in the analysis tree) shards the master schema by organ group, bin-packed to a
grammar-cost budget, and writes one ``{schema, prompt}`` pair per shard plus a
``shards_manifest.json``.  This module is the *consumer* side used by
``phenotype-batch``:

  - load a shard set (schemas + prompts + trait inventory),
  - live compile-check each distinct shard schema and auto-reshard (re-run the
    generator at a smaller budget) if one still trips the limit,
  - build per-(plant × shard) requests with a cached system prefix and uncached images,
  - parse + merge the per-shard ``{rationale, value}`` objects back into one
    per-plant record and validate coverage against the master schema.
"""

import os
import json
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

from anthropic import APIStatusError, BadRequestError

from . import shard_builder

NA = "not_assessable"
MANIFEST_NAME = "shards_manifest.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_shard_set(shard_dir: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load the manifest and every shard's schema + prompt from *shard_dir*.

    Returns ``(manifest, shards)`` where each shard dict carries
    ``{shard_id, schema (dict), prompt (str), groups, traits}``.
    """
    sdir = Path(shard_dir)
    manifest_path = sdir / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Generate shards first with build_stage3.py."
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    shards: List[Dict[str, Any]] = []
    for entry in manifest["shards"]:
        with open(sdir / entry["schema_file"], encoding="utf-8") as f:
            schema = json.load(f)
        with open(sdir / entry["prompt_file"], encoding="utf-8") as f:
            prompt = f.read()
        shards.append({
            "shard_id": entry["shard_id"],
            "schema": schema,
            "prompt": prompt,
            "groups": entry["groups"],
            "traits": entry["traits"],
        })
    return manifest, shards


def load_system_prompt(shard_dir: str, manifest: Dict[str, Any],
                       override: Optional[str] = None) -> str:
    """Return the shared invariant system prompt (Block A) for the shard set.

    A CLI ``--system-prompt`` override wins; otherwise the manifest's
    ``system_file`` (shared preamble emitted by the generator) is used.
    """
    if override:
        with open(override, encoding="utf-8") as f:
            return f.read()
    system_file = manifest.get("system_file")
    if not system_file:
        raise FileNotFoundError(
            "No system_file in manifest and no --system-prompt given; "
            "regenerate shards with build_stage3.py or pass --system-prompt."
        )
    with open(Path(shard_dir) / system_file, encoding="utf-8") as f:
        return f.read()


def resolve_master_path(manifest: Dict[str, Any], shard_dir: str,
                        override: Optional[str] = None) -> Optional[str]:
    """Resolve the master_schema path (CLI override wins, else manifest-relative)."""
    if override:
        return override
    rel = manifest.get("master_schema")
    if not rel:
        return None
    return os.path.normpath(os.path.join(shard_dir, rel))


def load_master_index(master_path: str):
    """Return ``(group_order, group_traits, trait_meta)`` from a master schema.

    ``group_order``: list of group names in master order.
    ``group_traits``: ``{group: [trait_name, ...]}`` in master order.
    ``trait_meta``: ``{(group, trait): {"scale_type", "unit"}}``.
    """
    with open(master_path, encoding="utf-8") as f:
        master = json.load(f)
    master = shard_builder.normalize_master(master)
    group_order: List[str] = []
    group_traits: "OrderedDict[str, List[str]]" = OrderedDict()
    trait_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for gname, gobj in master["trait_groups"].items():
        group_order.append(gname)
        names = []
        for tr in gobj["traits"]:
            names.append(tr["trait_name"])
            trait_meta[(gname, tr["trait_name"])] = {
                "scale_type": tr["scale_type"],
                "unit": tr.get("unit"),
            }
        group_traits[gname] = names
    return group_order, group_traits, trait_meta


def master_index_from_manifest(manifest: Dict[str, Any]):
    """Fallback master index built from the manifest's ``all_traits`` inventory."""
    group_order: List[str] = []
    group_traits: "OrderedDict[str, List[str]]" = OrderedDict()
    trait_meta: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for t in manifest.get("all_traits", []):
        g, name = t["group"], t["trait"]
        if g not in group_traits:
            group_order.append(g)
            group_traits[g] = []
        group_traits[g].append(name)
        trait_meta[(g, name)] = {"scale_type": t.get("scale_type"), "unit": t.get("unit")}
    return group_order, group_traits, trait_meta


# ---------------------------------------------------------------------------
# Live compile check + auto-reshard
# ---------------------------------------------------------------------------

def _is_complexity_error(err: Exception) -> bool:
    msg = str(getattr(err, "message", "") or err).lower()
    return ("too complex" in msg or "compiled grammar" in msg
            or "grammar is too large" in msg or "reduce the number of strict" in msg)


def compile_check_schema(client, model: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Return ``(ok, error_message)`` for whether *schema* compiles to a grammar.

    Sends a tiny structured-output request (no effort/thinking) and treats a
    400 mentioning grammar complexity as a compile failure.  Other errors
    propagate.
    """
    try:
        client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": "Reply with any schema-valid JSON."}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        return True, None
    except (BadRequestError, APIStatusError) as e:
        status = getattr(e, "status_code", None)
        if status == 400 and _is_complexity_error(e):
            return False, str(getattr(e, "message", "") or e)
        raise


def reshard(master_path: str, shard_dir: str, new_budget: int) -> None:
    """Re-generate the shard set at *new_budget* (in-process, keeps files in sync)."""
    shard_builder.generate_shards(master_path, shard_dir, budget=new_budget,
                                  write_combined=False)


def ensure_compilable(
    client, model: str, shard_dir: str, manifest: Dict[str, Any],
    shards: List[Dict[str, Any]], master_path: Optional[str],
    min_budget: int = 4,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compile-check every shard; auto-reshard at a smaller budget on failure.

    Auto-reshard re-runs the in-process generator (which owns schema + prompt +
    manifest, so regenerating keeps them consistent); it needs the master schema
    on disk.  Returns the (possibly regenerated) ``(manifest, shards)``.  Raises
    ``RuntimeError`` if a shard cannot be made to compile.
    """
    can_reshard = bool(master_path and os.path.exists(master_path))

    while True:
        failed = []
        print(f"\n--- Pre-flight: compile-checking {len(shards)} shard schema(s) ---")
        for s in shards:
            ok, err = compile_check_schema(client, model, s["schema"])
            if ok:
                print(f"  {s['shard_id']}: compiles OK")
            else:
                print(f"  {s['shard_id']}: TOO COMPLEX — {err}")
                failed.append(s["shard_id"])
        if not failed:
            return manifest, shards

        current_budget = int(manifest.get("shard_budget", 0)) or 0
        new_budget = max(min_budget, current_budget // 2) if current_budget else 0
        if not can_reshard or current_budget <= min_budget or new_budget >= current_budget:
            raise RuntimeError(
                "Shard(s) %s exceed the grammar-complexity limit and cannot be "
                "auto-resharded (master schema not found on disk, or budget floor "
                "reached). Regenerate with a smaller --shard-budget (current %s) "
                "via `pxgpt shard-schema` and retry."
                % (", ".join(failed), current_budget)
            )
        print(f"\n  Re-sharding at smaller budget {current_budget} -> {new_budget}")
        reshard(master_path, shard_dir, new_budget)
        manifest, shards = load_shard_set(shard_dir)


# ---------------------------------------------------------------------------
# Request building (cached system; uncached images followed by per-shard prompt)
# ---------------------------------------------------------------------------


def shard_custom_id(line_id: str, shard_id: str) -> str:
    return f"{line_id}__{shard_id}"


def split_custom_id(custom_id: str) -> Tuple[str, str]:
    line_id, _, shard_id = custom_id.rpartition("__")
    return line_id, shard_id


def build_sharded_requests(
    line_image_blocks: Dict[str, List[Dict[str, Any]]],
    shards: List[Dict[str, Any]],
    system_prompt: str,
    config,
    build_request_params,
) -> List[Dict[str, Any]]:
    """Build one batch request per (plant × shard).

    Only the shared system block has a cache breakpoint. Image blocks remain
    ordinary input and stay before the per-shard text prompt, following
    Anthropic's recommended image-then-text layout. Each shard still changes
    ``output_config.format``, so same-format system prefixes may be reused across
    plants without repeatedly cache-writing the much larger image input.
    """
    system_blocks = [{
        "type": "text",
        "text": system_prompt,
        "cache_control": {"type": "ephemeral"},
    }]

    requests: List[Dict[str, Any]] = []
    for line_id, image_blocks in line_image_blocks.items():
        for s in shards:
            content = image_blocks + [{"type": "text", "text": s["prompt"]}]
            output_config = config.stage3_output_config(schema=s["schema"])
            params = build_request_params(
                model=config.anthropic_model,
                max_tokens=config.stage3_max_tokens,
                system=system_blocks,
                messages=[{"role": "user", "content": content}],
                temperature=config.temperature,
                output_config=output_config,
            )
            requests.append({"custom_id": shard_custom_id(line_id, s["shard_id"]),
                             "params": params})
    return requests


# ---------------------------------------------------------------------------
# Parse + merge
# ---------------------------------------------------------------------------

def parse_value(scale_type: Optional[str], value: Any) -> Any:
    """Parse a raw shard value: quantitative strings -> float (or None)."""
    if scale_type == "quantitative":
        if value is None:
            return None
        s = str(value).strip()
        if s == "" or s.lower() == NA:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return value  # nominal/ordinal kept verbatim (NA stays string, ints stay ints)


def merge_plant_record(
    shard_objs: List[Dict[str, Any]],
    group_order: List[str],
    group_traits: Dict[str, List[str]],
    trait_meta: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple["OrderedDict[str, Any]", List[Tuple[str, str]]]:
    """Merge per-shard JSON objects into one per-plant record (master order).

    Returns ``(record, missing)`` where *missing* lists ``(group, trait)`` pairs
    absent from the merged result.  Quantitative values are parsed to numbers.
    """
    raw: Dict[str, Dict[str, Any]] = {}
    for obj in shard_objs:
        if not isinstance(obj, dict):
            continue
        for group, traits in obj.items():
            if isinstance(traits, dict):
                raw.setdefault(group, {}).update(traits)

    record: "OrderedDict[str, Any]" = OrderedDict()
    missing: List[Tuple[str, str]] = []
    for group in group_order:
        grec: "OrderedDict[str, Any]" = OrderedDict()
        for trait in group_traits.get(group, []):
            obj = raw.get(group, {}).get(trait)
            if not isinstance(obj, dict) or "value" not in obj:
                missing.append((group, trait))
                continue
            meta = trait_meta.get((group, trait), {})
            grec[trait] = OrderedDict([
                ("rationale", obj.get("rationale")),
                ("value", parse_value(meta.get("scale_type"), obj.get("value"))),
            ])
        if grec:
            record[group] = grec
    return record, missing
