"""``pxgpt extract-report`` — pull ``<report>`` content out of chain-of-thought output.

Backward-compatibility path for prompts that wrap output in
``<think>...</think><report>...</report>`` (as opposed to native reasoning via
``--effort``).  Handles both a single-response file and the grouped
multi-cultivar file produced by ``describe-batch`` / ``describe-batch-openai``.

The ``<think>`` reasoning is discarded; only the ``<report>`` body is kept.
"""

import argparse
from pathlib import Path

from ..core.report_utils import (
    is_grouped,
    extract_report_single,
    extract_report_grouped,
)


def extract_report_command(args):
    path = Path(args.input)
    if not path.exists():
        print(f"Error: input file not found: {path}")
        return 1
    text = path.read_text(encoding="utf-8")

    mode = args.mode
    if mode == "auto":
        mode = "grouped" if is_grouped(text) else "single"

    if mode == "grouped":
        if not is_grouped(text):
            print("Warning: no '### <id>' section headers found; "
                  "treating input as a single response.")
            result = extract_report_single(text) + "\n"
        else:
            result, stats = extract_report_grouped(text)
            msg = f"Extracted <report> from {stats['sections']} section(s)"
            if stats["empty"]:
                msg += f"; {stats['empty']} had no <report> (left empty)"
            print(msg)
    else:
        result = extract_report_single(text) + "\n"
        if not result.strip():
            print("Warning: no <report>...</report> found in input.")

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Written to: {args.output}")
    else:
        print(result, end="")
    return 0


def setup_extract_report_parser(subparsers):
    parser = subparsers.add_parser(
        "extract-report",
        help="Extract <report> content from <think>/<report> chain-of-thought output",
        description=(
            "Backward-compatible extractor for prompts that wrap output in "
            "<think>...</think><report>...</report>. Keeps only the <report> body "
            "(the <think> reasoning is discarded). Handles a single-response file "
            "and the grouped multi-cultivar describe-batch output. Not needed when "
            "using native reasoning (--effort / *_EFFORT), which emits no tags."
        ),
    )
    parser.add_argument(
        "--input", required=True,
        help="Input text file (a single response, or a grouped describe output)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file (default: print to stdout)",
    )
    parser.add_argument(
        "--mode", choices=["auto", "grouped", "single"], default="auto",
        help="auto-detect by '### ' headers (default), grouped (per-cultivar "
             "sections), or single (one response)",
    )
    parser.set_defaults(func=extract_report_command)
