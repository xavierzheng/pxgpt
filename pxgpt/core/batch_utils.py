"""Shared helpers for Anthropic batch operations.

Both Stage 1 (describe-batch) and Stage 3 (phenotype-batch) use these to:
  - Build per-request param dicts with the temperature guard applied.
  - Extract plain text from a response content block list (skipping thinking blocks).
  - Poll a batch until it reaches ``ended`` status.
  - Write describe / phenotype results from a completed batch.
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .provenance import build_provenance, stamp_record


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write *obj* as pretty JSON to *path* via a temp file + atomic rename.

    Shared by the batch and sequential dispatch paths so a crash mid-write can
    never leave a half-written ``<lid>.json`` / partial behind.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Temperature / thinking guard
# ---------------------------------------------------------------------------

# Model tiers where the API rejects a non-default temperature/top_p/top_k
# unconditionally (not just while thinking is active) and where omitting
# `thinking` now defaults to adaptive thinking ON instead of off. Sonnet 4.6
# and earlier tiers keep the original rule: a custom temperature is fine as
# long as thinking is off, and omitting `thinking` means thinking is off.
_STRICT_TEMPERATURE_GUARD_PREFIXES = (
    "claude-sonnet-5",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
    "claude-mythos-5",
)


def model_uses_strict_temperature_guard(model: str) -> bool:
    """True for model tiers that reject a non-default temperature unconditionally.

    These tiers (Claude Sonnet 5, Opus 4.7/4.8, Fable 5, Mythos 5) also default
    to adaptive thinking ON when ``thinking`` is omitted, unlike Sonnet 4.6 and
    earlier where omitting it means thinking is off.
    """
    return model.startswith(_STRICT_TEMPERATURE_GUARD_PREFIXES)


def build_request_params(
    model: str,
    max_tokens: int,
    system: List[Dict[str, Any]],
    messages: List[Dict[str, Any]],
    temperature: float,
    output_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a ``MessageCreateParamsNonStreaming``-compatible dict.

    Temperature guard: the API rejects a custom temperature while thinking is
    active, on every model tier. On the strict-guard tiers (see
    ``model_uses_strict_temperature_guard``) it rejects a custom temperature
    unconditionally, even with thinking off — and those same tiers default to
    adaptive thinking ON when ``thinking`` is omitted. So on those tiers, when
    effort is off, we send an explicit ``thinking: {"type": "disabled"}`` to
    preserve the "no reasoning" behavior and omit temperature entirely.
    """
    params: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if output_config:
        params["output_config"] = output_config

    thinking_active = bool(output_config and output_config.get("effort"))
    strict_guard = model_uses_strict_temperature_guard(model)

    if not thinking_active:
        if strict_guard:
            params["thinking"] = {"type": "disabled"}
        else:
            params["temperature"] = temperature

    return params


def temperature_guard_status(model: str, effort: str) -> str:
    """Human-readable summary of what ``build_request_params`` will do.

    For status/log messages in the CLI commands — keeps them in sync with the
    actual guard logic above instead of assuming a fixed model tier.
    """
    if effort:
        return "temperature omitted (thinking active)"
    if model_uses_strict_temperature_guard(model):
        return ("temperature omitted; thinking explicitly disabled "
                "(model rejects a non-default temperature even with thinking off)")
    return "temperature sent (thinking off)"


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
    provider: str = "anthropic",
    model: Optional[str] = None,
    schema_name: Optional[str] = None,
    schema_version: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, int]:
    """Retrieve batch results and write one JSON file per plant line/cultivar.

    The unsharded path: one request per plant, so the response IS the record and
    there is nothing to merge.  It still gets the same ``_provenance`` block as
    the sharded path — these files land in the same kind of result directory and
    are read by the same ``json-to-table``.

    Returns token-usage totals.
    """
    prov = build_provenance(provider, model, schema_name, schema_version, run_id)
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
                if isinstance(parsed, dict):
                    parsed = stamp_record(parsed, prov)
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


# ---------------------------------------------------------------------------
# _partial/ provenance
# ---------------------------------------------------------------------------
# Partial files are keyed by ``<line_id>__<shard_id>`` alone — no provider, no
# model — and adoption is an unconditional glob.  So two runs pointed at the
# same ``--output`` would silently merge each other's shards into both outputs.
# ``.run.json`` stamps the store with whoever created it, and the guard below
# refuses a run that does not match.  The dot prefix keeps it out of
# ``glob("*.json")``, but every adoption loop skips it by name as well rather
# than relying on that.
_RUN_META_NAME = ".run.json"


def _run_meta_path(partial_dir) -> Path:
    """Return the path of the provenance stamp inside *partial_dir*."""
    return Path(partial_dir) / _RUN_META_NAME


def read_run_meta(partial_dir) -> Optional[Dict[str, Any]]:
    """Return the parsed ``.run.json``, or None when absent or unreadable."""
    try:
        return json.loads(_run_meta_path(partial_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_run_meta_if_absent(
    partial_dir,
    provider: str,
    model: str,
    schema_name: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> None:
    """Stamp *partial_dir* with the run's identity unless already stamped.

    Creates *partial_dir* when it does not exist yet, so the guard can be called
    before the store has been laid down.  ``schema_name`` / ``schema_version``
    are recorded alongside provider/model: the same model run against two schema
    versions produces two incompatible trait sets, and merging those into one
    output is exactly as wrong as merging two models.
    """
    path = _run_meta_path(partial_dir)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {
        "provider": provider,
        "model": model,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


def assert_partial_provenance(
    partial_dir,
    provider: str,
    model: str,
    schema_name: Optional[str] = None,
    schema_version: Optional[str] = None,
) -> None:
    """Refuse to reuse a ``_partial/`` store another run created.

    Stamps the store on first use.  An unstamped store that already holds
    partials is a legacy one: it is adopted as before, with one warning.  Raises
    ``RuntimeError`` when the stamp names a different provider, model or schema
    version, without writing anything.

    Two kinds of stamp are tolerated rather than refused, because neither is
    evidence of a conflict:

      * a stamp written before ``schema_version`` existed (no such key) — one
        warning, and the missing fields are added in place from this run;
      * a run that cannot name its own schema version (``schema_version=None``,
        e.g. the local ``--shard-dir`` path, whose merge index comes from the
        manifest and never opens a master schema) — nothing to compare, so the
        stamp is left as it is.
    """
    partial_dir = Path(partial_dir)
    meta = read_run_meta(partial_dir)

    if meta is None:
        if partial_dir.is_dir() and any(
            p.name != _RUN_META_NAME for p in partial_dir.glob("*.json")
        ):
            print(f"  WARNING: {partial_dir} holds shard partials but no "
                  f"{_RUN_META_NAME} (legacy store).  Adopting them as "
                  f"provider={provider!r} model={model!r} — if they came from a "
                  f"different run, stop now and use a different --output.")
        write_run_meta_if_absent(partial_dir, provider, model,
                                 schema_name, schema_version)
        return

    if meta.get("provider") != provider or meta.get("model") != model:
        raise RuntimeError(
            f"Refusing to reuse the shard partial store in {partial_dir}\n"
            f"  it was created by : provider={meta.get('provider')!r} "
            f"model={meta.get('model')!r}\n"
            f"  this run is       : provider={provider!r} model={model!r}\n"
            f"Adopting those partials would silently merge another run's results "
            f"into this one's output.  Either:\n"
            f"  - point --output at a different directory, or\n"
            f"  - delete {_run_meta_path(partial_dir)} if you are certain the "
            f"existing partials belong to this run."
        )

    if "schema_version" not in meta:
        # Legacy stamp: provider/model agree, so these partials are ours.  Fill
        # in the fields it predates instead of erroring — same adoption
        # behaviour as an unstamped store.
        print(f"  WARNING: {_run_meta_path(partial_dir)} predates schema "
              f"provenance (no 'schema_version').  Adopting it and recording "
              f"schema_name={schema_name!r} schema_version={schema_version!r} "
              f"from this run.")
        upgraded = dict(meta)
        upgraded.setdefault("schema_name", schema_name)
        upgraded["schema_version"] = schema_version
        write_json_atomic(_run_meta_path(partial_dir), upgraded)
        return

    if schema_version is None or meta.get("schema_version") == schema_version:
        return

    raise RuntimeError(
        f"Refusing to reuse the shard partial store in {partial_dir}\n"
        f"  it was created with schema_version="
        f"{meta.get('schema_version')!r} (schema_name={meta.get('schema_name')!r})\n"
        f"  this run uses       schema_version="
        f"{schema_version!r} (schema_name={schema_name!r})\n"
        f"Same model, different schema: the traits do not line up, so merging "
        f"those partials would produce a record that never came from one "
        f"schema.  Either:\n"
        f"  - point --output at a different directory, or\n"
        f"  - delete {_run_meta_path(partial_dir)} if you are certain the "
        f"existing partials belong to this run."
    )


def merge_sharded_results(
    fresh: Dict[str, Dict[str, Any]],
    shard_errors: Dict[str, List[str]],
    line_ids: List[str],
    master_index,
    output_dir: str,
    provider: str,
    model: str,
    schema_name: Optional[str] = None,
    schema_version: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, int]:
    """Merge freshly-fetched shards with the ``_partial/`` store; write per plant.

    Provider-agnostic half of a sharded fetch.  *fresh* maps
    ``"<line_id>__<shard_id>"`` to the parsed shard object retrieved in THIS run;
    *shard_errors* maps ``line_id`` to ``"<shard_id>: <detail>"`` strings for the
    shards that produced nothing.  ``master_index`` is
    ``(group_order, group_traits, trait_meta)`` from :mod:`pxgpt.core.sharding`.

    Partial-aware + cumulative.  This shares the sequential dispatch's
    ``<output>/_partial/<line_id>__<shard_id>.json`` store so a run that left
    gaps (e.g. a shard hit a transient ``overloaded_error``) can be recovered:

      * per-shard partials already on disk are adopted before merging,
      * each freshly-succeeded shard is persisted as a partial, and
      * the merge uses the UNION of prior partials + this run.

    So re-running ``fetch-results`` is idempotent, and a follow-up
    ``--dispatch sequential`` (whose resume reads the same ``_partial/`` dir)
    re-issues only the still-missing shards.  A trait is only reported in
    ``<lid>.gaps.json`` if it is missing *after* the union; a stale gaps file
    whose traits are now filled is removed.

    Every written record carries a ``_provenance`` block naming this run
    (:func:`pxgpt.core.provenance.build_provenance`), computed once here and
    repeated into each record, so a record that leaves this directory still says
    where it came from.

    Returns ``{"written", "plants_with_gaps", "total_gaps"}``.  Both providers
    share this so the merge, the gap rule and the recovery story can never drift
    apart between them — an asymmetry there would silently break the
    cross-provider comparison this project exists to make.
    """
    from .sharding import split_custom_id, merge_plant_record

    group_order, group_traits, trait_meta = master_index
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    partial_dir = out / "_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    assert_partial_provenance(partial_dir, provider, model,
                              schema_name, schema_version)
    prov = build_provenance(provider, model, schema_name, schema_version, run_id)

    # Per plant, keep one object per shard so a re-fetch overrides cleanly and
    # a shard is never merged twice.  Adopt existing partials first.
    shards_by_line: Dict[str, Dict[str, Any]] = {lid: {} for lid in line_ids}

    adopted = 0
    for p in sorted(partial_dir.glob("*.json")):
        if p.name == _RUN_META_NAME:
            continue  # provenance stamp, not a shard partial
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue  # corrupt/half-written partial -> ignore, may be re-run
        lid, sid = split_custom_id(p.stem)
        shards_by_line.setdefault(lid, {})[sid] = obj
        adopted += 1
    if adopted:
        print(f"  Adopted {adopted} shard partial(s) from {partial_dir}")

    for cid, obj in fresh.items():
        line_id, shard_id = split_custom_id(cid)
        # Persist (crash safety + feeds a later sequential resume).
        write_json_atomic(partial_dir / f"{cid}.json", obj)
        shards_by_line.setdefault(line_id, {})[shard_id] = obj

    written = 0
    plants_with_gaps = 0
    total_gaps = 0
    for lid in line_ids:
        record, missing = merge_plant_record(
            list(shards_by_line.get(lid, {}).values()),
            group_order, group_traits, trait_meta,
        )
        write_json_atomic(out / f"{lid}.json", stamp_record(record, prov))
        written += 1

        gaps_path = out / f"{lid}.gaps.json"
        # Only surface shard errors for shards that produced nothing this run
        # AND left the plant with missing traits — a shard error covered by an
        # adopted partial is no longer a gap.
        errs = shard_errors.get(lid, []) if missing else []
        if missing:
            plants_with_gaps += 1
            total_gaps += len(missing)
            report = {
                "line_id": lid,
                "missing_traits": [{"group": g, "trait": t} for g, t in missing],
                "shard_errors": errs,
            }
            write_json_atomic(gaps_path, report)
            print(f"  {lid}: {len(missing)} missing trait(s)"
                  + (f", {len(errs)} shard error(s)" if errs else ""))
        elif gaps_path.exists():
            gaps_path.unlink()  # a prior run's gap is now filled

    print(f"\n  Wrote {written} merged JSON files; "
          f"{plants_with_gaps} plant(s) with gaps ({total_gaps} missing traits total)")
    if total_gaps or plants_with_gaps:
        print("  (see *.gaps.json next to the affected records; recover them with "
              "`--dispatch sequential` to the same --output)")
    return {"written": written, "plants_with_gaps": plants_with_gaps,
            "total_gaps": total_gaps}


def write_phenotype_sharded_results(
    client,
    batch_id: str,
    line_ids: List[str],
    master_index,
    output_dir: str,
    provider: str,
    model: str,
    schema_name: Optional[str] = None,
    schema_version: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Dict[str, int]:
    """Retrieve a SHARDED Anthropic phenotype batch and merge it per plant.

    Results have ``custom_id = "<line_id>__<shard_id>"``.  This is the
    Anthropic-specific half — iterate the batch, pull text + usage out of the
    Anthropic result shape — and :func:`merge_sharded_results` does everything
    after that, shared with the OpenAI path.

    Returns token-usage totals for the calls made in THIS batch (adopted
    partials contribute nothing to the totals — they were billed earlier).
    """
    from .sharding import split_custom_id

    # Refuse a foreign _partial/ store before spending the results download.
    # merge_sharded_results asserts again; the check is idempotent.
    assert_partial_provenance(Path(output_dir) / "_partial", provider, model,
                              schema_name, schema_version)

    totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}
    fresh: Dict[str, Dict[str, Any]] = {}
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
                fresh[cid] = json.loads(text)
            except json.JSONDecodeError:
                shard_errors.setdefault(line_id, []).append(f"{shard_id}: JSON parse failed")
                print(f"  WARNING: {cid} — JSON parse failed")
        else:
            detail = describe_batch_error(result.result.error)
            shard_errors.setdefault(line_id, []).append(f"{shard_id}: {detail}")
            print(f"  WARNING: {cid} failed — {detail}")

    merge_sharded_results(fresh, shard_errors, line_ids, master_index, output_dir,
                          provider, model, schema_name, schema_version, run_id)
    return totals


def print_token_summary(totals: Dict[str, int]) -> None:
    print("\n--- Token usage summary ---")
    print(f"  Input tokens:           {totals['input']:>10,}")
    print(f"  Output tokens:          {totals['output']:>10,}")
    print(f"  Cache creation tokens:  {totals['cache_creation']:>10,}")
    print(f"  Cache read tokens:      {totals['cache_read']:>10,}")
