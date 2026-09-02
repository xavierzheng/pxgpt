"""Every repo-relative path a doc cites must exist.

Commit 40c7004 ("Split describe/phenotyping prompts by growth stage") renamed
two prompts and retired three others into prompts/old_v0.1.0/. The docs were
never updated, so for a long while ALL FIVE `prompts/` paths in README.md and
user_manual.md pointed at files that did not exist -- every copy-pasteable
example in the project was broken, and nothing noticed:

    prompts/phenotyping_system.txt         -> describe_plant_system.txt
    prompts/describe_plant.txt             -> describe_plant_{mature,seedling}.txt
    prompts/extract_traits.txt             -> retired
    prompts/phenotype_schema.json          -> retired (use --shard-dir)
    prompts/phenotyping_system_schema.txt  -> retired

It surfaced only because an agent copied a path out of README.md into another
document and then checked it.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

DOCS = sorted(
    [REPO / "README.md", REPO / "user_manual.md", REPO / "HANDOFF.md"]
    + list((REPO / "ops").rglob("*.md"))
)

# Directories whose contents are real, checkable repo paths. Deliberately not
# every token that looks like a path: dataset trees live outside this repo, and
# example output names are not files.
CITED_DIRS = ("prompts/", "pxgpt/", "tests/", "ops/", "scripts/", "tools/")

# A trailing character that is punctuation, not part of the filename.
TRAILING = ".,;:)]}`'\"*"


def _cited_paths(text):
    # Strip fenced code blocks? No -- commands in code blocks are exactly what
    # users copy, so those paths matter most.
    for m in re.finditer(r"(?<![\w/.-])((?:%s)[A-Za-z0-9_./-]+)"
                         % "|".join(re.escape(d) for d in CITED_DIRS), text):
        p = m.group(1).rstrip(TRAILING)
        # Brace expansion and placeholders are not literal paths.
        if "{" in p or "<" in p or p.endswith("/"):
            continue
        # Only check things that look like files, not directory prefixes.
        if "." not in Path(p).name:
            continue
        yield p


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_cited_paths_exist(doc):
    missing = sorted({p for p in _cited_paths(doc.read_text())
                      if not (REPO / p).exists()})
    assert not missing, (
        f"{doc.relative_to(REPO)} cites paths that do not exist:\n  "
        + "\n  ".join(missing)
        + "\n\nEither the file was renamed and the doc was not updated, or the "
          "doc invented a filename. Both have happened here."
    )


def test_the_guard_would_have_caught_the_original_bug():
    """A doc citing a retired prompt must fail, not be quietly tolerated."""
    for retired in ("prompts/phenotyping_system.txt",
                    "prompts/extract_traits.txt",
                    "prompts/phenotype_schema.json"):
        assert not (REPO / retired).exists(), (
            f"{retired} exists again; update this test's premise"
        )
        assert list(_cited_paths(f"run it with --system-prompt {retired} now")), (
            f"the extractor does not even see {retired}, so it cannot guard it"
        )
