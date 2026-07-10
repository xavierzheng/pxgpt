"""Build sharded Stage 3 artifacts from a master phenotype schema.

This is the *generator* side of schema sharding (the consumer side lives in
:mod:`pxgpt.core.sharding`).  From a master schema in the Stage 2 format
(``trait_groups -> traits -> {trait_name, description, scale_type, values,
unit, ...}``) it produces, under ``<shard-dir>/``:

  - ``shard_NN.schema.json`` + ``shard_NN.prompt.md`` per shard,
  - ``shards_system.md`` — the shared invariant preamble (cached system block),
  - ``shards_manifest.json`` — shard list + trait inventory (drives merge).

Per trait object: EXACTLY ``{rationale, value}``, rationale FIRST (CoT), then
value.  ``value`` is constrained by ``scale_type``:

  nominal      -> {"enum": [...categories..., "not_assessable"]}
  ordinal      -> {"enum": [...integer level ids..., "not_assessable"]}
  quantitative -> {"type": "string"}   # number-as-string or "not_assessable",
                                       # parsed downstream. NOT anyOf — union
                                       # types blow up the constrained grammar.

Whole organ groups are bin-packed (in master order) up to a grammar-cost
budget; a group that alone exceeds the budget is sub-sharded across its traits.
Grammar-subset safe: no anyOf/union, min/max/length, pattern, $ref, self-ref.
"""

import os
import json
import argparse
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Tuple

NA = "not_assessable"
DEFAULT_SHARD_BUDGET = 40

# Constructs that must never appear in a shard schema (grammar-subset safety).
FORBIDDEN = {
    "anyOf", "oneOf", "allOf", "not",
    "minLength", "maxLength", "minItems", "maxItems",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "pattern", "$ref", "$defs", "definitions",
}


# ---------------------------------------------------------------------------
# Master-schema shape normalization
# ---------------------------------------------------------------------------

def normalize_master(master: Dict[str, Any]) -> Dict[str, Any]:
    """Return a master whose groups live in a canonical ``trait_groups`` dict.

    Master schemas reach us in two shapes that carry identical trait objects but
    differ only in how groups are wrapped:

      * ``trait_groups``: ``{group_name: {"description"?, "traits": [...]}}``
        (canonical; e.g. the Stage 2 generator / ``master_schema_opus4-8.json``).
      * ``groups``: ``[{"group": name, "description"?, "traits": [...]}, ...]``
        (list-of-objects, e.g. LLM+human-synthesized VER_2 ``master_schema.json``).

    Both are accepted so downstream code can rely on a name-keyed
    ``trait_groups`` OrderedDict.  Idempotent: a master already in canonical
    shape is returned unchanged.  Raises on an unrecognizable / malformed shape.
    """
    if isinstance(master.get("trait_groups"), dict):
        return master
    groups = master.get("groups")
    if not isinstance(groups, list):
        raise KeyError(
            "master schema has neither a 'trait_groups' object nor a 'groups' "
            "list; got top-level keys: %s" % sorted(master.keys())
        )
    tg: "OrderedDict[str, Any]" = OrderedDict()
    for grp in groups:
        name = grp.get("group") or grp.get("group_name") or grp.get("name")
        if not name:
            raise ValueError(
                "group entry missing a name field ('group'/'group_name'/'name'): %r"
                % (grp,)
            )
        if name in tg:
            raise ValueError("duplicate group name in master schema: %r" % name)
        tg[name] = OrderedDict([
            ("description", grp.get("description", "")),
            ("traits", grp["traits"]),
        ])
    m = dict(master)
    m["trait_groups"] = tg
    return m


# ---------------------------------------------------------------------------
# Schema building
# ---------------------------------------------------------------------------

def value_schema(trait: Dict[str, Any]) -> Dict[str, Any]:
    st = trait["scale_type"]
    if st == "nominal":
        return {"enum": list(trait["values"]) + [NA]}
    if st == "ordinal":
        return {"enum": [lvl["level"] for lvl in trait["values"]] + [NA]}
    if st == "quantitative":
        # number-as-string (or "not_assessable"); parsed downstream. NOT anyOf.
        return {"type": "string"}
    raise ValueError("unknown scale_type: %r (%s)" % (st, trait["trait_name"]))


def trait_object(trait: Dict[str, Any]) -> "OrderedDict[str, Any]":
    # ORDER MATTERS: rationale declared before value -> CoT under autoregressive decode.
    props = OrderedDict()
    props["rationale"] = {"type": "string"}
    props["value"] = value_schema(trait)
    return OrderedDict([
        ("type", "object"),
        ("properties", props),
        ("required", ["rationale", "value"]),
        ("additionalProperties", False),
    ])


def group_object(traits: List[Dict[str, Any]]) -> "OrderedDict[str, Any]":
    trait_props = OrderedDict()
    for tr in traits:
        trait_props[tr["trait_name"]] = trait_object(tr)
    return OrderedDict([
        ("type", "object"),
        ("properties", trait_props),
        ("required", list(trait_props.keys())),
        ("additionalProperties", False),
    ])


def build_schema_from_groups(groups, title, description) -> "OrderedDict[str, Any]":
    """``groups`` is a list of (group_name, [trait,...]) in display order."""
    group_props = OrderedDict()
    for gname, traits in groups:
        group_props[gname] = group_object(traits)
    return OrderedDict([
        ("$schema", "https://json-schema.org/draft/2020-12/schema"),
        ("title", title),
        ("description", description),
        ("type", "object"),
        ("properties", group_props),
        ("required", list(group_props.keys())),
        ("additionalProperties", False),
    ])


def build_schema(master: Dict[str, Any]) -> "OrderedDict[str, Any]":
    master = normalize_master(master)
    groups = [(g, gobj["traits"]) for g, gobj in master["trait_groups"].items()]
    return build_schema_from_groups(
        groups,
        "stage3_per_plant_phenotype",
        "Stage 3 structured-output schema: one value per trait for ONE plant, "
        "scored from a SET of multiple photographs. Each trait emits {rationale, value} "
        "with rationale FIRST to force chain-of-thought before committing the value.",
    )


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def prompt_preamble() -> str:
    """Shared, plant-invariant instruction block (identical across all shards)."""
    L = []
    A = L.append
    A("# Stage 3 — Per-Plant Visual Phenotyping Instruction\n")
    A("## Your role\n")
    A("You are a careful botanical phenotyper scoring **ONE plant**. You are given "
      "**MULTIPLE PHOTOGRAPHS of that single plant** — different angles, organs, and "
      "magnifications (whole-plant shots, close-ups of leaves, petioles, stems, roots and "
      "the rockwool cube). These are **complementary views of the SAME individual**, not "
      "different plants.\n")
    A("A **10 x 10 x 6.5 cm rockwool cube** appears in the images; use it as the scale "
      "reference for every size estimate (10 cm width, 6.5 cm height).\n")

    A("## How to read the images\n")
    A("- **Integrate evidence across ALL images.** For each trait, use whichever image(s) "
      "show that trait most clearly. A trait that is invisible or ambiguous in one photo may "
      "be clearly visible in another — do not base a judgment on a single image when another "
      "image carries better evidence.\n")
    A("- A trait may be visible in several views that **disagree**; weigh them and decide, and "
      "say so in the rationale.\n")

    A("## Output procedure for EVERY trait\n")
    A("For each trait you must produce an object with two fields, **in this order**:\n")
    A("1. **`rationale`** (write this FIRST): cite the **specific visual evidence** — *what* you "
      "see, *where* on the plant, and **which image(s)** it came from (e.g. \"in the close-up "
      "of the leaf underside\", \"in the top-down whole-plant view\"). State here when views "
      "disagree or when only some images were usable.\n")
    A("2. **`value`** (write this SECOND): choose from the allowed set listed for that trait. "
      "**The value must follow from the rationale, never the reverse.**\n")

    A("## Hard rules\n")
    A("- Use **ONLY** the allowed categories / levels / units given below. **Never invent a "
      "value.**\n")
    A("- For ordinal traits, return the **integer level id** (not the label).\n")
    A("- For quantitative traits, return a **number in the stated unit** as a string (e.g. "
      "\"12.5\"); estimate against the cube scale.\n")
    A("- Set `value` to **`not_assessable` ONLY IF** the trait cannot be judged from **ANY** of "
      "the provided images (organ absent from every view, not present at this developmental "
      "stage, or no angle shows it). **If even one image affords a confident judgment, assess "
      "it.** Guessing a trait that no image supports is an **ERROR**, not helpfulness.\n")
    A("- **Do NOT report a confidence score.** Confidence is computed downstream from embedding "
      "geometry, not self-reported.\n")
    return "\n".join(L) + "\n"


def group_section(gname: str, gdesc: str, traits: List[Dict[str, Any]]) -> str:
    """Human-readable spec for one organ group's subset of traits."""
    L = []
    A = L.append
    A("### %s\n" % gname.replace("_", " "))
    if gdesc:
        A("_%s_\n" % gdesc)
    for tr in traits:
        name = tr["trait_name"]
        st = tr["scale_type"]
        desc = tr.get("description", "")
        A("#### `%s` — %s" % (name, st))
        if desc:
            A("%s" % desc)
        if st == "nominal":
            cats = ", ".join("`%s`" % v for v in tr["values"])
            A("Allowed categories: %s, `%s`." % (cats, NA))
        elif st == "ordinal":
            A("Allowed ordinal levels (return the integer):")
            for lvl in tr["values"]:
                A("- `%s` (%s): %s" % (lvl["level"], lvl["label"], lvl["definition"]))
            A("- `%s`: trait cannot be judged from any image." % NA)
        elif st == "quantitative":
            unit = tr.get("unit") or "(unit unspecified)"
            A("Return a **number in %s** (estimated against the cube scale) as a string, "
              "or `%s`." % (unit, NA))
        A("")  # blank line between traits
    return "\n".join(L)


def shard_body_from_groups(master: Dict[str, Any], groups) -> str:
    """Per-shard prompt BODY (Block C): only the organ sections, no preamble.

    The shared preamble (role / rules — Block A) is emitted once as the system
    prompt so it can be cached and is never re-billed per shard.
    """
    master = normalize_master(master)
    descs = {g: gobj.get("description", "") for g, gobj in master["trait_groups"].items()}
    L = ["## Traits to score\n"]
    L.append("Traits are grouped by plant region. Within each trait, the allowed values "
             "follow. `not_assessable` is always additionally permitted under the rule "
             "above.\n")
    for gname, traits in groups:
        L.append(group_section(gname, descs.get(gname, ""), traits))
    return "\n".join(L) + "\n"


def build_prompt(master: Dict[str, Any]) -> str:
    """Combined (non-sharded) prompt = shared preamble + all organ bodies."""
    master = normalize_master(master)
    groups = [(g, gobj["traits"]) for g, gobj in master["trait_groups"].items()]
    return prompt_preamble() + "\n" + shard_body_from_groups(master, groups)


# ---------------------------------------------------------------------------
# Sharding (bin-pack whole groups; sub-shard a group that alone exceeds budget)
# ---------------------------------------------------------------------------

def trait_cost(trait: Dict[str, Any]) -> int:
    """Rough grammar-cost proxy: 1 (rationale) + 1 (value wrapper) + enum alternatives."""
    st = trait["scale_type"]
    if st in ("nominal", "ordinal"):
        return 2 + len(trait["values"]) + 1  # +1 for not_assessable alternative
    return 2  # quantitative -> bare string


def pack_shards(master: Dict[str, Any], budget: int):
    """Return a list of shards; each shard is a list of (group_name, [trait,...])."""
    shards: List[List[Tuple[str, List[Dict[str, Any]]]]] = []
    current: List[Tuple[str, List[Dict[str, Any]]]] = []
    current_cost = 0

    def flush():
        nonlocal current, current_cost
        if current:
            shards.append(current)
            current = []
            current_cost = 0

    for gname, gobj in master["trait_groups"].items():
        traits = gobj["traits"]
        gcost = sum(trait_cost(t) for t in traits)

        if gcost > budget:
            # Group alone exceeds budget -> flush, then sub-shard its own traits.
            flush()
            sub, sub_cost = [], 0
            for t in traits:
                tc = trait_cost(t)
                if sub and sub_cost + tc > budget:
                    shards.append([(gname, sub)])
                    sub, sub_cost = [], 0
                sub.append(t)
                sub_cost += tc
            if sub:
                shards.append([(gname, sub)])
            continue

        if current and current_cost + gcost > budget:
            flush()
        current.append((gname, traits))
        current_cost += gcost

    flush()
    return shards


def build_shards(master: Dict[str, Any], budget: int) -> List[Dict[str, Any]]:
    """Build shard descriptors with schema + prompt + trait inventory."""
    master = normalize_master(master)
    packed = pack_shards(master, budget)
    descriptors = []
    for i, shard in enumerate(packed):
        sid = "shard_%02d" % (i + 1)
        cost = sum(trait_cost(t) for _, traits in shard for t in traits)
        traits_inv = [
            {"group": g, "trait": t["trait_name"], "scale_type": t["scale_type"],
             "unit": t.get("unit")}
            for g, traits in shard for t in traits
        ]
        schema = build_schema_from_groups(
            shard,
            "stage3_%s" % sid,
            "Stage 3 shard %s of the per-plant phenotype schema (groups: %s). "
            "Each trait emits {rationale, value}, rationale FIRST." %
            (sid, ", ".join(g for g, _ in shard)),
        )
        prompt = shard_body_from_groups(master, shard)
        descriptors.append({
            "shard_id": sid,
            "schema_file": "%s.schema.json" % sid,
            "prompt_file": "%s.prompt.md" % sid,
            "groups": [g for g, _ in shard],
            "cost": cost,
            "traits": traits_inv,
            "_schema": schema,
            "_prompt": prompt,
        })
    return descriptors


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _scan_forbidden(node, path, found):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in FORBIDDEN:
                found.append("%s.%s" % (path, k))
            _scan_forbidden(v, "%s.%s" % (path, k), found)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _scan_forbidden(v, "%s[%d]" % (path, i), found)


def validate_schema_doc(doc: Dict[str, Any], label: str) -> List[str]:
    """Structural + forbidden-construct checks on one (sharded or combined) schema."""
    problems: List[str] = []
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(doc)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        problems.append("%s: not valid JSON Schema: %s" % (label, e))

    found: List[str] = []
    _scan_forbidden(doc, label, found)
    for f in found:
        problems.append("%s: forbidden construct %s" % (label, f))

    for gname, gobj in doc["properties"].items():
        for tname, tobj in gobj["properties"].items():
            keys = list(tobj.get("properties", {}).keys())
            if keys[:2] != ["rationale", "value"]:
                problems.append("%s/%s: key order %s" % (label, tname, keys))
            if tobj.get("required") != ["rationale", "value"]:
                problems.append("%s/%s: required=%s" % (label, tname, tobj.get("required")))
            if tobj.get("additionalProperties") is not False:
                problems.append("%s/%s: additionalProperties=%s" %
                                (label, tname, tobj.get("additionalProperties")))
    return problems


# ---------------------------------------------------------------------------
# Top-level generation
# ---------------------------------------------------------------------------

def generate_shards(
    master_path: str,
    shard_dir: str,
    budget: int = DEFAULT_SHARD_BUDGET,
    write_combined: bool = False,
    combined_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate the shard set (and optionally the combined artifacts).

    Writes ``shards_system.md``, ``shard_NN.{schema.json,prompt.md}`` and
    ``shards_manifest.json`` under *shard_dir*.  Returns a summary dict including
    ``shards`` and a ``problems`` list (empty when validation passes).
    """
    with open(master_path, encoding="utf-8") as f:
        master = json.load(f)
    master = normalize_master(master)
    os.makedirs(shard_dir, exist_ok=True)

    all_traits = [
        {"group": g, "trait": t["trait_name"], "scale_type": t["scale_type"],
         "unit": t.get("unit")}
        for g, gobj in master["trait_groups"].items() for t in gobj["traits"]
    ]

    # combined artifacts (optional; off by default)
    combined_schema = build_schema(master)
    if write_combined:
        cdir = combined_dir or os.path.dirname(os.path.abspath(shard_dir))
        os.makedirs(cdir, exist_ok=True)
        with open(os.path.join(cdir, "stage3_schema.json"), "w", encoding="utf-8") as f:
            json.dump(combined_schema, f, indent=2)
            f.write("\n")
        with open(os.path.join(cdir, "stage3_prompt.md"), "w", encoding="utf-8") as f:
            f.write(build_prompt(master))

    # sharded artifacts
    shards = build_shards(master, budget)
    system_text = prompt_preamble()
    manifest = OrderedDict([
        ("version", 1),
        ("master_schema", os.path.relpath(master_path, shard_dir)),
        ("system_file", "shards_system.md"),
        ("shard_budget", budget),
        ("shard_count", len(shards)),
        ("shards", [
            OrderedDict([
                ("shard_id", s["shard_id"]),
                ("schema_file", s["schema_file"]),
                ("prompt_file", s["prompt_file"]),
                ("groups", s["groups"]),
                ("cost", s["cost"]),
                ("traits", s["traits"]),
            ]) for s in shards
        ]),
        ("all_traits", all_traits),
    ])

    with open(os.path.join(shard_dir, "shards_system.md"), "w", encoding="utf-8") as f:
        f.write(system_text)
    for s in shards:
        with open(os.path.join(shard_dir, s["schema_file"]), "w", encoding="utf-8") as f:
            json.dump(s["_schema"], f, indent=2)
            f.write("\n")
        with open(os.path.join(shard_dir, s["prompt_file"]), "w", encoding="utf-8") as f:
            f.write(s["_prompt"])
    with open(os.path.join(shard_dir, "shards_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # validation
    problems = validate_schema_doc(combined_schema, "combined")
    for s in shards:
        problems += validate_schema_doc(s["_schema"], s["shard_id"])
    master_traits = [(t["group"], t["trait"]) for t in all_traits]
    shard_traits = [(t["group"], t["trait"]) for s in shards for t in s["traits"]]
    dupes = [x for x, c in Counter(shard_traits).items() if c > 1]
    missing = [x for x in master_traits if x not in set(shard_traits)]
    extra = [x for x in shard_traits if x not in set(master_traits)]
    if dupes:
        problems.append("coverage: traits in >1 shard: %s" % dupes)
    if missing:
        problems.append("coverage: traits missing from all shards: %s" % missing)
    if extra:
        problems.append("coverage: shard traits not in master: %s" % extra)

    return {
        "shard_dir": shard_dir,
        "shard_count": len(shards),
        "shard_budget": budget,
        "master_trait_count": len(all_traits),
        "group_count": len(master["trait_groups"]),
        "by_scale": dict(Counter(t["scale_type"] for t in all_traits)),
        "shards": [{"shard_id": s["shard_id"], "groups": s["groups"],
                    "cost": s["cost"], "trait_count": len(s["traits"])} for s in shards],
        "problems": problems,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_cli(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build sharded Stage 3 schema/prompt artifacts from a master schema."
    )
    ap.add_argument("--master", required=True, help="Path to master_schema.json (Stage 2 output).")
    ap.add_argument("--shard-dir", default=None,
                    help="Output directory for shards (default: <master dir>/shards).")
    ap.add_argument("--shard-budget", type=int, default=DEFAULT_SHARD_BUDGET,
                    help="Grammar-cost budget per shard (default: %d)." % DEFAULT_SHARD_BUDGET)
    ap.add_argument("--combined", action="store_true",
                    help="Also write the combined (non-sharded) stage3_schema.json + "
                         "stage3_prompt.md (default: shards only).")
    ap.add_argument("--combined-dir", default=None,
                    help="Directory for the combined artifacts (default: parent of --shard-dir).")
    args = ap.parse_args(argv)

    shard_dir = args.shard_dir or os.path.join(os.path.dirname(os.path.abspath(args.master)), "shards")
    summary = generate_shards(
        args.master, shard_dir, budget=args.shard_budget,
        write_combined=args.combined, combined_dir=args.combined_dir,
    )
    print_summary(summary)
    return 1 if summary["problems"] else 0


def print_summary(summary: Dict[str, Any]) -> None:
    print("=== VALIDATION ===")
    if summary["problems"]:
        print("FAILURES:")
        for p in summary["problems"]:
            print("   -", p)
    else:
        print("[1] combined + all shard schemas valid; no forbidden constructs "
              "(incl. no anyOf/union); rationale-before-value; both required; "
              "additionalProperties:false: OK")
        print("[2] coverage: every master trait in exactly one shard: OK")
    print("[3] SUMMARY")
    print("    master traits:", summary["master_trait_count"],
          " groups:", summary["group_count"])
    for st in ("nominal", "ordinal", "quantitative"):
        print("    %-13s: %d" % (st, summary["by_scale"].get(st, 0)))
    print("    shard budget :", summary["shard_budget"])
    print("    shards       :", summary["shard_count"])
    for s in summary["shards"]:
        print("      %-9s cost=%-3d groups=%s (%d traits)" %
              (s["shard_id"], s["cost"], ",".join(s["groups"]), s["trait_count"]))
    print("\nWrote shard set to:", summary["shard_dir"])


if __name__ == "__main__":
    raise SystemExit(run_cli())
