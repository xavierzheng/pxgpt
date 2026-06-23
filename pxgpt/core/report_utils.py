"""Extract ``<report>...</report>`` content from chain-of-thought style output.

Backward-compatibility helper for the legacy prompt convention where the model
emits ``<think>...</think><report>...</report>``.  The ``<think>`` reasoning is
discarded; only the ``<report>`` body is kept.  (With the native-reasoning path
— ``--effort`` / ``*_EFFORT`` — no tags are produced and this is not needed.)

Two input shapes are handled:

- **single**  — the whole file is one model response (one ``<report>``).
- **grouped** — the ``describe-batch`` / ``describe-batch-openai`` output format:
  multiple cultivars, each a ``### <id>`` section joined by ``---``.  Each
  section's ``<report>`` is extracted and the section structure is preserved.

The core regex helpers mirror the standalone ``extract_report_tags.py`` script
(kept alive for single-file use); this module adds the grouped-file handling
used by the ``pxgpt extract-report`` command.
"""

import re
from typing import Dict, Tuple

# Matches a "### <id>" section header (the describe-batch grouped format).
_SECTION_HEADER = re.compile(r"(?m)^###[ \t]+(.+?)[ \t]*$")


def parse_tag_content(text: str, tag: str) -> str:
    """Return the inner text of the first ``<tag>...</tag>`` block, or ""."""
    match = re.search(fr"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def ensure_all_tags_closed(text: str) -> str:
    """Append a closing tag for any opened-but-unclosed tag (truncated output)."""
    for tag in re.findall(r"<(\w+)>", text):
        if f"</{tag}>" not in text:
            text += f"</{tag}>"
    return text


def clean_text(text: str) -> str:
    """Strip BOM, normalize newlines, and collapse blank lines."""
    text = text.replace("﻿", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


def is_grouped(text: str) -> bool:
    """True if *text* contains ``### <id>`` section headers (grouped format)."""
    return bool(_SECTION_HEADER.search(text))


def extract_report_single(text: str) -> str:
    """Extract the ``<report>`` body from a single model response."""
    return clean_text(parse_tag_content(ensure_all_tags_closed(text), "report"))


def extract_report_grouped(text: str) -> Tuple[str, Dict[str, int]]:
    """Extract ``<report>`` from every ``### <id>`` section, preserving structure.

    Returns ``(combined_text, stats)`` where ``stats`` is
    ``{"sections": N, "empty": M}`` and *empty* counts sections with no
    ``<report>`` found.
    """
    headers = list(_SECTION_HEADER.finditer(text))
    sections = []
    empty = 0
    for i, h in enumerate(headers):
        line_id = h.group(1).strip()
        start = h.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        report = extract_report_single(text[start:end])
        if not report:
            empty += 1
        sections.append(f"### {line_id}\n\n{report}")
    combined = "\n\n---\n\n".join(sections) + "\n"
    return combined, {"sections": len(headers), "empty": empty}
