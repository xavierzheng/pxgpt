"""PXGPT: Plant analysis tool with multiple LLM provider support."""

from importlib.metadata import PackageNotFoundError, version as _installed_version

# Single source of truth is setup.py's `version=`.  Reading it back from the
# installed distribution means the number cannot be edited in one place and go
# stale in another -- which is exactly what had happened: this file said 0.3.0
# while setup.py and `--version` both said 0.4.0.
try:
    __version__ = _installed_version("pxgpt")
except PackageNotFoundError:
    # Running from a source tree that was never installed (no `pip install -e .`).
    # Say so rather than inventing a number that might be wrong.
    __version__ = "0.0.0+source"

__author__ = "PXGPT Team"
