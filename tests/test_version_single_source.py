"""The version number must exist in exactly one editable place.

It had drifted before this test existed: pxgpt/__init__.py said 0.3.0 while
setup.py and `--version` both said 0.4.0, because each was edited by hand.
"""

import re
import subprocess
import sys
from importlib.metadata import version as installed_version
from pathlib import Path

import pxgpt

REPO = Path(__file__).resolve().parent.parent


def test_package_version_comes_from_the_installed_metadata():
    assert pxgpt.__version__ == installed_version("pxgpt")


def test_cli_reports_the_same_version_as_the_package():
    out = subprocess.run([sys.executable, "-m", "pxgpt.main", "--version"],
                         capture_output=True, text=True, cwd=REPO)
    assert pxgpt.__version__ in (out.stdout + out.stderr)


# The one literal allowed inside the package: the not-installed sentinel.
SENTINEL = "0.0.0+source"

# Leading/trailing underscores must be part of the match -- the drift that
# actually happened was `__version__ = "0.3.0"`, and a pattern anchored on a bare
# `version=` sails straight past it.
VERSION_LITERAL = re.compile(r"""(?i)_*version_*\s*=\s*["']([^"']*\d+\.\d+[^"']*)["']""")


def test_setup_py_is_the_only_hardcoded_copy():
    """setup.py declares it; nothing under pxgpt/ may restate it.

    A second literal is how the drift happened, so any dotted-number assignment
    to a version-ish name inside the package fails here -- except the sentinel
    used when the package is not installed at all.
    """
    offenders = []
    for f in sorted((REPO / "pxgpt").rglob("*.py")):
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            m = VERSION_LITERAL.search(line)
            if m and m.group(1) != SENTINEL:
                offenders.append(f"{f.relative_to(REPO)}:{i}: {line.strip()}")
    assert not offenders, ("hardcoded version literal(s) inside the package:\n"
                           + "\n".join(offenders))


def test_the_guard_itself_catches_a_reintroduced_literal():
    """Guard the guard: a regex that matches nothing would pass silently."""
    assert VERSION_LITERAL.search('__version__ = "0.3.0"')      # the real drift
    assert VERSION_LITERAL.search('    version="0.4.0",')       # setup.py style
    assert VERSION_LITERAL.search("version = '1.2'")
    assert VERSION_LITERAL.search('VERSION = "9.9.9"')
    assert not VERSION_LITERAL.search('__author__ = "PXGPT Team"')
    assert not VERSION_LITERAL.search("from importlib.metadata import version")
    assert VERSION_LITERAL.search(f'__version__ = "{SENTINEL}"').group(1) == SENTINEL


def test_setup_py_still_declares_one():
    m = re.search(r"""version\s*=\s*["']([^"']+)["']""",
                  (REPO / "setup.py").read_text(encoding="utf-8"))
    assert m, "setup.py must declare version="
    # Matches what is installed right now, so a bump without a reinstall is visible.
    assert m.group(1) == installed_version("pxgpt")
