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

The first version of this guard then failed on a fresh clone, which is the only
place that matters, because its premise was wrong. "Every cited path exists" is
not the rule. A doc may legitimately cite a path that git deliberately ignores:

  - ``ops/local-vllm/.env`` is created by the user (``cp env.example .env``) and
    is gitignored, so it is absent from every clone by design;
  - ``HANDOFF.md`` and ``CLAUDE.md`` are gitignored on purpose (see .gitignore),
    so hardcoding them as documents to scan raised FileNotFoundError.

The rule is: **a cited path must exist unless git ignores it.**
"""

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# Only documents that are actually present. HANDOFF.md and any CLAUDE.md are
# gitignored, so they exist on a working checkout and not on a clone; scan them
# when they are here, and do not demand them when they are not.
DOCS = sorted(
    d for d in (
        [REPO / "README.md", REPO / "user_manual.md", REPO / "HANDOFF.md"]
        + list((REPO / "ops").rglob("*.md"))
    )
    if d.is_file()
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


def _git_ignored(paths):
    """Which of *paths* does git deliberately ignore?

    Those are absent from a clone on purpose -- a user-created .env, a
    machine-local HANDOFF.md -- so citing one is correct and it must not be
    demanded. Uses .gitignore as the source of truth rather than a hand-kept
    allowlist that would drift.
    """
    paths = list(paths)
    if not paths:
        return set()
    try:
        r = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            input="\n".join(paths), cwd=REPO,
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable; this is a repo-hygiene check")
    # 0 = some ignored, 1 = none ignored, anything else = not a checkout
    if r.returncode not in (0, 1):
        pytest.skip("not a git checkout; this is a repo-hygiene check")
    return {line.strip() for line in r.stdout.splitlines() if line.strip()}


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_cited_paths_exist(doc):
    absent = {p for p in _cited_paths(doc.read_text()) if not (REPO / p).exists()}
    missing = sorted(absent - _git_ignored(absent))
    assert not missing, (
        f"{doc.relative_to(REPO)} cites paths that do not exist:\n  "
        + "\n  ".join(missing)
        + "\n\nEither the file was renamed and the doc was not updated, or the "
          "doc invented a filename. Both have happened here.\n"
          "(A path git ignores -- a user-created .env, say -- is exempt and "
          "would not be listed.)"
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
