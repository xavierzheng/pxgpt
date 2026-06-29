"""Shared helpers for Anthropic batch operations.

Both Stage 1 (describe-batch) and Stage 3 (phenotype-batch) use these to:
  - Build per-request param dicts with the temperature guard applied.
  - Extract plain text from a response content block list (skipping thinking blocks).
  - Poll a batch until it reaches ``ended`` status.
  - Write describe / phenotype results from a completed batch.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Temperature guard
# ---------------------------------------------------------------------------

def build_request_params(
    model: str,
    max_tokens: int,
    system: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    temperature: float,
    output_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a ``MessageCreateParamsNonStreaming``-compatible dict.

    The API rejects a custom temperature when thinking is active
    (``output_config.effort`` is set), so we only include it when safe.
    """
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if output_config:
        params["output_config"] = output_config
    # Only send temperature when thinking is off
    thinking_active = bool(output_config and output_config.get("effort"))
    if not thinking_active:
        params["temperature"] = temperature
    return params


# ---------------------------------------------------------------------------
# Response content extraction
# ---------------------------------------------------------------------------

def extract_text_content(content_blocks) -> str:
    """Return the concatenated text from all TextBlock entries.

    Thinking blocks (type="thinking") are intentionally skipped.
    """
    return "\n".join(
        b.text for b in content_blocks if getattr(b, "type", None) == "text"
    )


def describe_batch_error(error_response) -> str:
    """Return ``"<type>: <message>"`` for a failed batch request.

    A failed result exposes ``result.result.error`` as a ``BetaErrorResponse``
    whose own ``type`` is always the literal ``"error"`` and which has **no**
    ``message``; the actual API error (``invalid_request_error``, etc.) and its
    human-readable text live on the nested ``.error`` object.  This helper
    digs into that nested object, falling back gracefully if the shape differs.
    """
    inner = getattr(error_response, "error", None) or error_response
    etype = getattr(inner, "type", None) or getattr(error_response, "type", None) or "unknown"
    emsg = getattr(inner, "message", None)
    if emsg is None:
        emsg = str(inner)
    return f"{etype}: {emsg}"


def strip_code_fence(text: str) -> str:
    """Remove a leading ```json ... ``` or ``` ... ``` wrapper if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]  # drop ```json or ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# Batch polling
# ---------------------------------------------------------------------------

def poll_batch(client, batch_id: str, interval: int = 30):
    """Block until the batch reaches ``ended`` status.  Prints progress."""
    while True:
        batch = client.beta.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [{batch_id}] {batch.processing_status} | "
            f"succeeded={counts.succeeded}  errored={counts.errored}  "
            f"processing={counts.processing}"
        )
        if batch.processing_status == "ended":
            return batch
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Result writers
# ---------------------------------------------------------------------------

def write_describe_results(
    client,
    batch_id: str,
    line_ids: List[str],
    output_path: str,
) -> Dict[str, int]:
    """Retrieve batch results and write grouped description text.

    Returns token-usage totals: ``{input, output, cache_creation, cache_read}``.
    """
    raw: Dict[str, str] = {}
    totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}

    for result in client.beta.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            raw[cid] = extract_text_content(msg.content)
            u = msg.usage
            totals["input"] += getattr(u, "input_tokens", 0)
            totals["output"] += getattr(u, "output_tokens", 0)
            totals["cache_creation"] += getattr(u, "cache_creation_input_tokens", 0)
            totals["cache_read"] += getattr(u, "cache_read_input_tokens", 0)
        else:
            detail = describe_batch_error(result.result.error)
            raw[cid] = f"[ERROR {detail}]"
            print(f"  WARNING: {cid} failed — {detail}")

    # Build output in original line order
    sections = []
    for lid in line_ids:
        text = raw.get(lid, "[NOT FOUND IN RESULTS]")
        sections.append(f"### {lid}\n\n{text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(sections) + "\n")

    return totals


def write_phenotype_results(
    client,
    batch_id: str,
    line_ids: List[str],
    output_dir: str,
) -> Dict[str, int]:
    """Retrieve batch results and write one JSON file per plant line/cultivar.

    Returns token-usage totals.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}

    written = 0
    errored = 0

    for result in client.beta.messages.batches.results(batch_id):
        cid = result.custom_id
        if result.result.type == "succeeded":
            msg = result.result.message
            text = extract_text_content(msg.content)
            text = strip_code_fence(text)
            u = msg.usage
            totals["input"] += getattr(u, "input_tokens", 0)
            totals["output"] += getattr(u, "output_tokens", 0)
            totals["cache_creation"] += getattr(u, "cache_creation_input_tokens", 0)
            totals["cache_read"] += getattr(u, "cache_read_input_tokens", 0)
            try:
                parsed = json.loads(text)
                dest = out / f"{cid}.json"
                with open(dest, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2)
                    f.write("\n")
                written += 1
            except json.JSONDecodeError:
                # Fallback: save raw text so the user can inspect it
                dest = out / f"{cid}.err.txt"
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(text)
                print(f"  WARNING: {cid} — JSON parse failed; raw text saved to {dest}")
                errored += 1
        else:
            detail = describe_batch_error(result.result.error)
            dest = out / f"{cid}.err.txt"
            with open(dest, "w", encoding="utf-8") as f:
                f.write(f"[ERROR {detail}]\n")
            print(f"  WARNING: {cid} failed — {detail}")
            errored += 1

    print(f"  Wrote {written} JSON files; {errored} errors")
    return totals


def write_phenotype_sharded_results(
    client,
    batch_id: str,
    line_ids: List[str],
    master_index,
    output_dir: str,
) -> Dict[str, int]:
    """Retrieve a SHARDED phenotype batch, merge shards, write one JSON per plant.

    Results have ``custom_id = "<line_id>__<shard_id>"``.  Per plant the shard
    objects are merged into one record keyed by the master organ structure,
    quantitative strings are parsed to numbers, and any missing trait is
    reported.  ``master_index`` is ``(group_order, group_traits, trait_meta)``
    from :mod:`pxgpt.core.sharding`.

    Returns token-usage totals.
    """
    from .sharding import split_custom_id, merge_plant_record

    group_order, group_traits, trait_meta = master_index
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}

    per_line: Dict[str, List[Any]] = {lid: [] for lid in line_ids}
    shard_errors: Dict[str, List[str]] = {}

    for result in client.beta.messages.batches.results(batch_id):
        cid = result.custom_id
        line_id, shard_id = split_custom_id(cid)
        if result.result.type == "succeeded":
            msg = result.result.message
            text = strip_code_fence(extract_text_content(msg.content))
            u = msg.usage
            totals["input"] += getattr(u, "input_tokens", 0)
            totals["output"] += getattr(u, "output_tokens", 0)
            totals["cache_creation"] += getattr(u, "cache_creation_input_tokens", 0)
            totals["cache_read"] += getattr(u, "cache_read_input_tokens", 0)
            try:
                per_line.setdefault(line_id, []).append(json.loads(text))
            except json.JSONDecodeError:
                shard_errors.setdefault(line_id, []).append(f"{shard_id}: JSON parse failed")
                print(f"  WARNING: {cid} — JSON parse failed")
        else:
            detail = describe_batch_error(result.result.error)
            shard_errors.setdefault(line_id, []).append(f"{shard_id}: {detail}")
            print(f"  WARNING: {cid} failed — {detail}")

    written = 0
    plants_with_gaps = 0
    total_gaps = 0
    for lid in line_ids:
        record, missing = merge_plant_record(
            per_line.get(lid, []), group_order, group_traits, trait_meta
        )
        dest = out / f"{lid}.json"
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        written += 1

        errs = shard_errors.get(lid, [])
        if missing or errs:
            plants_with_gaps += 1
            total_gaps += len(missing)
            report = {
                "line_id": lid,
                "missing_traits": [{"group": g, "trait": t} for g, t in missing],
                "shard_errors": errs,
            }
            with open(out / f"{lid}.gaps.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
                f.write("\n")
            print(f"  {lid}: {len(missing)} missing trait(s)"
                  + (f", {len(errs)} shard error(s)" if errs else ""))

    print(f"\n  Wrote {written} merged JSON files; "
          f"{plants_with_gaps} plant(s) with gaps ({total_gaps} missing traits total)")
    if total_gaps or plants_with_gaps:
        print("  (see *.gaps.json next to the affected records)")
    return totals


def print_token_summary(totals: Dict[str, int]) -> None:
    print("\n--- Token usage summary ---")
    print(f"  Input tokens:           {totals['input']:>10,}")
    print(f"  Output tokens:          {totals['output']:>10,}")
    print(f"  Cache creation tokens:  {totals['cache_creation']:>10,}")
    print(f"  Cache read tokens:      {totals['cache_read']:>10,}")
