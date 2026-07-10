"""One-time migration: lift per-category definitions out of ``design_note``
into a structured, phenotyper-facing ``values: [{value, definition}]`` field.

Draft-and-review: writes ``<name>.migrated.json`` + a report of categories the
heuristic could not resolve, which a human fills in before use. Never edits in
place. ``design_note`` is preserved untouched (it stays author-only).
"""
import json
import re
import sys
from collections import OrderedDict


def extract_defs(design_note, categories):
    """Best-effort ``{category: definition}`` from a free-text design_note.

    Matches ``<category> = <definition>`` clauses terminated by ``;`` or ``.``.
    Categories with no match are simply absent from the result.
    """
    defs = {}
    for cat in categories:
        m = re.search(re.escape(cat) + r"\s*=\s*([^;.]+)", design_note or "")
        if m:
            defs[cat] = m.group(1).strip()
    return defs


def _category_tokens(values):
    return [v["value"] if isinstance(v, dict) else v for v in (values or [])]


def migrate_master(master):
    """Return ``(new_master, unresolved)``; only nominal ``values`` change."""
    new = json.loads(json.dumps(master))  # deep copy, order-preserving enough
    unresolved = []
    groups = new.get("trait_groups") or {}
    # normalize 'groups' list shape if present (mirror shard_builder.normalize_master)
    if not groups and isinstance(new.get("groups"), list):
        groups = OrderedDict((g["group"], g) for g in new["groups"])
    for gname, gobj in groups.items():
        for tr in gobj.get("traits", []):
            if tr.get("scale_type") != "nominal":
                continue
            cats = _category_tokens(tr.get("values"))
            defs = extract_defs(tr.get("design_note", ""), cats)
            tr["values"] = [
                {"value": c, "definition": defs[c]} if c in defs else {"value": c}
                for c in cats
            ]
            for c in cats:
                if c not in defs:
                    unresolved.append("%s/%s/%s" % (gname, tr["trait_name"], c))
    return new, unresolved


def main(argv):
    if len(argv) != 2:
        print("usage: migrate_nominal_definitions.py <master_schema.json>")
        return 2
    src = argv[1]
    with open(src, encoding="utf-8") as f:
        master = json.load(f)
    new, unresolved = migrate_master(master)
    out = src.rsplit(".json", 1)[0] + ".migrated.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(new, f, indent=2, ensure_ascii=False)
        f.write("\n")
    report = src.rsplit(".json", 1)[0] + ".migration_report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("Migrated: %s -> %s\n" % (src, out))
        f.write("Categories needing a MANUAL definition (%d):\n" % len(unresolved))
        for u in unresolved:
            f.write("  - %s\n" % u)
    print("Wrote", out)
    print("Unresolved definitions:", len(unresolved), "-> see", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
