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


# A sentence-ending period: one followed by whitespace + a capital letter, or at
# end of string. Deliberately NOT triggered by "e.g."/"i.e." (period + lowercase)
# or decimals like "5.5" (period between digits), so those stay inside a definition.
_CLAUSE_END = re.compile(r"\.(?=\s+[A-Z])|\.\s*$")


def extract_defs(design_note, categories):
    """Best-effort ``{category: definition}`` from a free-text design_note.

    For each category, finds ``<category> = <definition>`` (category anchored on
    word boundaries so a suffix category like ``erect`` does not match inside
    ``semi_erect``), then trims the definition at the first ``;`` or
    sentence-ending period. Internal periods (abbreviations, decimals) are kept.
    Categories with no match are absent from the result.
    """
    defs = {}
    note = design_note or ""
    for cat in categories:
        m = re.search(r"\b" + re.escape(cat) + r"\b\s*=\s*(.+)", note)
        if not m:
            continue
        tail = m.group(1)
        cut = len(tail)
        semi = tail.find(";")
        if semi != -1:
            cut = min(cut, semi)
        mb = _CLAUSE_END.search(tail)
        if mb:
            cut = min(cut, mb.start())
        defs[cat] = tail[:cut].strip().rstrip(".").strip()
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
