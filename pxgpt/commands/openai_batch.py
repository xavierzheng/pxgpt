"""OpenAI Batch API stages: describe-batch-openai and phenotype-batch-openai.

These mirror the Anthropic ``describe-batch`` / ``phenotype-batch`` commands but
run on the OpenAI Batch API using the **Responses** endpoint (``/v1/responses``).
The Responses API is required so images can be referenced by Files-API file_id;
Chat Completions cannot reference uploaded images.  Images are uploaded once via
the OpenAI Files API (``purpose="vision"``) and reused by file_id; pass
``--no-files-api`` (or ``USE_FILES_API=false``) to embed images inline as base64.

Workflow
--------
1. (phenotype only) Load the JSON schema and normalize it for OpenAI strict
   structured output (all properties required, additionalProperties: false).
2. Discover plant-line subdirectories (one per ``custom_id`` / cultivar).
3. Images: upload via the Files API (default) or embed as base64.
4. Build a JSONL request file (one ``/v1/responses`` body per line).
5. Upload the JSONL (``purpose="batch"``) and create the batch.
6. Save a checkpoint and exit (fire-and-forget), or ``--wait`` to poll + write.

Sharded mode (``phenotype-batch-openai --shard-dir``) mirrors ``phenotype-batch``
instead: one request per (plant × shard) with a per-shard schema and prompt,
merged back into one record per plant.  Both dispatch strategies build the exact
same request body, so ``batch`` and ``sequential`` results are comparable and
their ``_partial/`` stores can be mixed for recovery.
"""

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import openai

from ..core.config import Config
from ..core.file_utils import read_file_safely
from ..core.files_manager import IMAGE_EXTENSIONS
from ..core.openai_files_manager import OpenAIFilesManager
from ..core.batch_utils import (
    assert_partial_provenance,
    print_token_summary,
    strip_code_fence,
    write_json_atomic,
    _RUN_META_NAME,
)
from ..core.openai_batch_utils import (
    build_openai_file_id_blocks,
    build_openai_base64_blocks,
    build_openai_sharded_requests,
    build_responses_request_body,
    build_text_format,
    extract_response_text_and_usage,
    openai_compile_probe,
    openai_effort_status,
    openai_normalize_schema,
    write_jsonl_requests,
    poll_openai_batch,
    write_openai_describe_results,
    write_openai_phenotype_results,
    write_openai_phenotype_sharded_results,
)
from ..core import sharding
# Same master-index resolution (and the same "master schema not found" note) as
# the Anthropic path.  Shared rather than copied: a merge index that differed
# between providers would silently make their outputs incomparable.
from .phenotype import _resolve_master_index

# OpenAI Batch endpoint. The Responses API is required so images can be
# referenced by Files-API file_id (Chat Completions cannot do this).
_OPENAI_BATCH_ENDPOINT = "/v1/responses"

# Hard cap OpenAI puts on a batch INPUT file (purpose="batch").
_BATCH_INPUT_LIMIT = 200 * 1024 * 1024
# Refuse a little below it: the limit is on the uploaded object, and a file this
# close to it is a mistake either way.
_BATCH_INPUT_GUARD = 190 * 1024 * 1024


def _make_openai_client(config: Config):
    from openai import OpenAI

    kwargs = {"api_key": config.openai_api_key, "max_retries": 0}
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    return OpenAI(**kwargs)


def _discover_plant_lines(input_dir_arg):
    """Return the sorted plant-line subdirectories, or None after printing why not."""
    input_dir = Path(input_dir_arg)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return None
    plant_lines = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not plant_lines:
        print(f"Error: no subdirectories found in {input_dir}")
        return None
    print(f"Found {len(plant_lines)} plant line(s) in {input_dir}")
    return plant_lines


def _collect_line_image_blocks(args, config, client, plant_lines, use_files_api):
    """Return ``{line_id: [image content block, ...]}`` for the discovered lines."""
    line_image_blocks: Dict[str, List[Dict]] = {}

    if use_files_api:
        print(f"\n--- Uploading images to OpenAI Files API (manifest: {args.manifest}) ---")
        files_mgr = OpenAIFilesManager(client, args.manifest)
        for line_dir in plant_lines:
            images = [p for p in line_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
            if not images:
                print(f"  {line_dir.name}: no images, skipping")
                continue
            already = sum(1 for p in images if files_mgr.get_file_id(str(p)) is not None)
            print(f"  {line_dir.name}: {len(images)} image(s)  "
                  f"({already} cached, {len(images) - already} to upload)")
            file_ids = files_mgr.upload_folder(
                str(line_dir), concurrency=config.upload_concurrency
            )
            line_image_blocks[line_dir.name] = build_openai_file_id_blocks(file_ids)
        if line_image_blocks:
            print(f"\nTotal manifest entries: {files_mgr.stats()['total']}")
    else:
        print("\n--- Files API disabled: embedding images inline as base64 ---")
        for line_dir in plant_lines:
            images = sorted(
                p for p in line_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                print(f"  {line_dir.name}: no images, skipping")
                continue
            print(f"  {line_dir.name}: {len(images)} image(s) embedded inline")
            line_image_blocks[line_dir.name] = build_openai_base64_blocks(images)

    return line_image_blocks


def _human_size(n):
    """Bytes as GB once that reads better than MB."""
    return f"{n / 1024**3:.1f} GB" if n >= 1024**3 else f"{n / 1024**2:.0f} MB"


def _warn_base64_shard_size(plant_lines, shard_count):
    """Warn, before any encoding, that base64 × shards will not fit a batch file.

    Sharding repeats the same images once per shard, so inline base64 multiplies
    the payload by the shard count: a run that fits comfortably with the Files
    API can be tens of GB without it.  Estimated from the image sizes on disk (
    base64 is ~4/3 of the bytes) so nothing has to be encoded — let alone written
    — to find out.
    """
    total = 0
    for line_dir in plant_lines:
        for p in line_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                total += p.stat().st_size
    estimate = int(total * 4 / 3) * shard_count
    print(f"\n  WARNING: --no-files-api with --shard-dir: every image is embedded "
          f"in each\n  of the {shard_count} shard requests, so the batch input "
          f"JSONL would be roughly\n  {_human_size(estimate)} ({estimate:,} bytes) "
          f"against OpenAI's {_BATCH_INPUT_LIMIT // 1024**2} MB limit.\n"
          f"  Drop --no-files-api to upload each image once and reference it by "
          f"file_id.")
    return estimate


def _submit_batch_jsonl(client, config, requests, stage):
    """Write the JSONL, refuse an oversized one, upload it and create the batch.

    Returns ``(batch, input_file_id, jsonl_path)``, or ``None`` after printing a
    hard-fail message when the JSONL exceeds what OpenAI will accept.
    """
    fd, jsonl_path = tempfile.mkstemp(prefix=f"openai_batch_{stage}_", suffix=".jsonl", dir=".")
    os.close(fd)
    write_jsonl_requests(requests, jsonl_path)
    size = os.path.getsize(jsonl_path)
    print(f"Batch input JSONL: {jsonl_path}  ({size:,} bytes)")

    if size > _BATCH_INPUT_GUARD:
        print(
            f"\nError: the batch input JSONL is {size:,} bytes "
            f"({size / 1024**2:.0f} MB); OpenAI rejects a batch input file over "
            f"{_BATCH_INPUT_LIMIT // 1024**2} MB, so it was NOT uploaded.\n"
            f"  That {_BATCH_INPUT_LIMIT // 1024**2} MB cap applies to the batch "
            f"input JSONL (purpose=\"batch\") only — it has nothing to do with "
            f"image uploads (purpose=\"vision\"), which are not part of this "
            f"file.\n"
            f"  Send the images through the Files API instead (drop "
            f"--no-files-api): the JSONL then carries only file_id strings "
            f"rather than the image bytes, which is what keeps it in the MB "
            f"range however many plants and shards there are.\n"
            f"  The JSONL is left at {jsonl_path} for inspection."
        )
        return None

    print(f"\n--- Submitting batch ({len(requests)} requests) ---")
    with open(jsonl_path, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=_OPENAI_BATCH_ENDPOINT,
        completion_window=config.openai_batch_completion_window,
    )
    print(f"Batch ID:  {batch.id}")
    print(f"Status:    {batch.status}")
    return batch, input_file.id, jsonl_path


def _write_checkpoint(checkpoint):
    path = f"checkpoint_{checkpoint['batch_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
        f.write("\n")
    print(f"Checkpoint: {path}")
    return path


def _run_openai_batch(args, stage: str) -> int:
    """Shared runner for both OpenAI batch stages.  *stage* is 'describe' or 'phenotype'."""
    config = Config.from_env()
    if not config.openai_api_key:
        print("Error: OPENAI_API_KEY is not set")
        return 1

    model = config.openai_model
    client = _make_openai_client(config)

    # Reasoning effort mirrors the Anthropic stages: Stage 1 takes DESCRIBE_EFFORT
    # with an --effort override, Stage 3 takes STAGE3_EFFORT and has no flag.
    # "" / off / none all mean off, sent as reasoning.effort "none" downstream.
    if stage == "describe":
        effort = config.describe_effort if args.effort is None else args.effort
        if effort == "off":
            effort = ""
    else:
        effort = config.stage3_effort

    sharded = stage == "phenotype" and bool(getattr(args, "shard_dir", None))

    # ------------------------------------------------------------------
    # Prompts / schema.  In sharded mode both come from the shard set
    # (shared preamble + per-shard {schema, prompt}) and are resolved later.
    # ------------------------------------------------------------------
    system_prompt = None
    user_prompt = None
    text_format = None
    if sharded:
        if args.prompt:
            print("Note: --prompt is ignored in sharded mode; per-shard prompts "
                  "come from --shard-dir.")
    else:
        if stage == "phenotype" and not args.schema:
            print("Error: --schema is required without --shard-dir.")
            return 1
        if not args.system_prompt or not args.prompt:
            print("Error: --system-prompt and --prompt are required "
                  "without --shard-dir.")
            return 1
        try:
            system_prompt = read_file_safely(args.system_prompt, "system prompt")
            user_prompt = read_file_safely(args.prompt, "user prompt")
        except (FileNotFoundError, IOError) as e:
            print(f"File error: {e}")
            return 1

        if stage == "phenotype":
            try:
                with open(args.schema, encoding="utf-8") as f:
                    raw_schema = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Schema error: {e}")
                return 1
            schema = openai_normalize_schema(raw_schema)
            text_format = build_text_format(schema, name="phenotype")
            print(f"Structured output: json_schema (strict) from {args.schema}")

    # ------------------------------------------------------------------
    # Discover plant lines
    # ------------------------------------------------------------------
    plant_lines = _discover_plant_lines(args.input_dir)
    if plant_lines is None:
        return 1

    print(f"Model: {model}")
    print(f"Reasoning effort: {openai_effort_status(model, effort)}")

    use_files_api = config.use_files_api and not args.no_files_api

    if sharded:
        return _run_openai_sharded(args, config, client, model, effort,
                                   plant_lines, use_files_api)

    # ------------------------------------------------------------------
    # Collect images: upload via Files API (default) or embed inline base64
    # ------------------------------------------------------------------
    line_image_blocks = _collect_line_image_blocks(
        args, config, client, plant_lines, use_files_api
    )
    if not line_image_blocks:
        print("Error: no images found to process")
        return 1

    # ------------------------------------------------------------------
    # Build JSONL requests
    # ------------------------------------------------------------------
    max_tokens = config.stage1_max_tokens if stage == "describe" else config.stage3_max_tokens
    print(f"\n--- Building {len(line_image_blocks)} batch request(s) ---")
    requests: List[Dict] = []
    for line_id, image_blocks in line_image_blocks.items():
        body = build_responses_request_body(
            model=model,
            system_prompt=system_prompt,
            image_blocks=image_blocks,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=config.temperature,
            text_format=text_format,
            reasoning_effort=effort,
        )
        requests.append({
            "custom_id": line_id,
            "method": "POST",
            "url": _OPENAI_BATCH_ENDPOINT,
            "body": body,
        })

    line_ids = list(line_image_blocks.keys())

    # ------------------------------------------------------------------
    # Submit batch
    # ------------------------------------------------------------------
    submitted = _submit_batch_jsonl(client, config, requests, stage)
    if submitted is None:
        return 1
    batch, input_file_id, jsonl_path = submitted
    batch_id = batch.id

    # Save checkpoint (provider-tagged so fetch-results can dispatch)
    checkpoint_path = _write_checkpoint({
        "batch_id": batch_id,
        "provider": "openai",
        "stage": stage,
        "output": args.output,
        "line_ids": line_ids,
        "model": model,
        "input_file_id": input_file_id,
        "jsonl_path": jsonl_path,
    })

    if not args.wait:
        print(
            f"\nBatch submitted.  To fetch results when complete:\n"
            f"  pxgpt fetch-results --checkpoint {checkpoint_path}"
        )
        return 0

    # ------------------------------------------------------------------
    # Optional: poll and write immediately
    # ------------------------------------------------------------------
    print("\n--- Polling batch ---")
    batch = poll_openai_batch(client, batch_id)
    if batch.status != "completed":
        print(f"\nBatch ended with status '{batch.status}'.  "
              f"Run fetch-results later or inspect the batch for details.")

    print("\n--- Writing results ---")
    if stage == "describe":
        totals = write_openai_describe_results(client, batch, line_ids, args.output)
        print_token_summary(totals)
        print(f"\nDescriptions written to: {args.output}")
    else:
        totals = write_openai_phenotype_results(client, batch, line_ids, args.output)
        print_token_summary(totals)
        print(f"\nPhenotype JSON files written to: {args.output}/")
    return 0


# ---------------------------------------------------------------------------
# Sharded mode (Stage 3 only)
# ---------------------------------------------------------------------------

def _run_openai_sharded(args, config, client, model, effort, plant_lines,
                        use_files_api):
    """Dispatch per (plant × shard) against a shard set, then merge."""
    shard_dir = args.shard_dir
    try:
        manifest, shards = sharding.load_shard_set(shard_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    try:
        system_prompt = sharding.load_system_prompt(shard_dir, manifest, args.system_prompt)
    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}")
        return 1

    master_path = sharding.resolve_master_path(manifest, shard_dir, args.master_schema)
    print(f"\n--- Sharded mode: {len(shards)} shard(s) from {shard_dir} ---")
    print("Structured output: json_schema (strict), one per shard")
    print(f"system prompt: "
          f"{'--system-prompt override' if args.system_prompt else manifest.get('system_file')}")

    # Pre-flight before touching any image: confirm every shard schema is
    # accepted.  A schema error is per-request on the Batch API, so without this
    # one bad shard bills 7/8 of a run to deliver 7/8 of the traits.
    try:
        manifest, shards = sharding.ensure_compilable(
            client, model, shard_dir, manifest, shards, master_path,
            allow_reshard=args.allow_reshard,
            probe=openai_compile_probe,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    if not use_files_api:
        _warn_base64_shard_size(plant_lines, len(shards))

    line_image_blocks = _collect_line_image_blocks(
        args, config, client, plant_lines, use_files_api
    )
    if not line_image_blocks:
        print("Error: no images found to process")
        return 1

    master_index, master_path = _resolve_master_index(manifest, shard_dir,
                                                      args.master_schema)

    requests = build_openai_sharded_requests(
        line_image_blocks, shards, system_prompt, config
    )
    line_ids = list(line_image_blocks.keys())
    print(f"\n--- Built {len(requests)} request(s): "
          f"{len(line_ids)} plant(s) × {len(shards)} shard(s) ---")

    if args.dispatch == "sequential":
        return _dispatch_openai_sequential(
            args, config, client, model, requests, line_ids, master_index
        )
    return _dispatch_openai_batch(
        args, config, client, model, requests, line_ids, shards, shard_dir,
        master_path, master_index,
    )


def _dispatch_openai_batch(args, config, client, model, requests, line_ids, shards,
                           shard_dir, master_path, master_index):
    jsonl_requests = [
        {"custom_id": r["custom_id"], "method": "POST",
         "url": _OPENAI_BATCH_ENDPOINT, "body": r["body"]}
        for r in requests
    ]
    submitted = _submit_batch_jsonl(client, config, jsonl_requests, "phenotype_sharded")
    if submitted is None:
        return 1
    batch, input_file_id, jsonl_path = submitted
    batch_id = batch.id

    # Field names match the Anthropic sharded checkpoint exactly so
    # fetch-results' _sharded_master_index() serves both providers.
    checkpoint_path = _write_checkpoint({
        "batch_id": batch_id,
        "provider": "openai",
        "stage": "phenotype_sharded",
        "output": args.output,
        "line_ids": line_ids,
        "model": model,
        "input_file_id": input_file_id,
        "jsonl_path": jsonl_path,
        "shard_dir": str(Path(shard_dir).resolve()),
        "master_schema": str(Path(master_path).resolve()) if master_path else None,
        "shard_ids": [s["shard_id"] for s in shards],
    })

    if not args.wait:
        print(f"\nBatch submitted.  To fetch + merge results when complete:\n"
              f"  pxgpt fetch-results --checkpoint {checkpoint_path}")
        return 0

    print("\n--- Polling batch ---")
    batch = poll_openai_batch(client, batch_id)
    if batch.status != "completed":
        print(f"\nBatch ended with status '{batch.status}'.  "
              f"Writing whatever results are available.")

    print("\n--- Writing merged results ---")
    totals = write_openai_phenotype_sharded_results(
        client, batch, line_ids, master_index, args.output, "openai", model,
    )
    print_token_summary(totals)
    print(f"\nMerged phenotype JSON files written to: {args.output}/")
    return 0


# HTTP statuses worth a short in-run retry: rate limit + transient server
# conditions.  A 400 (a schema the model will never accept, an oversized
# request) is a client-side error and is NOT retried here — it surfaces, writes
# no partial, and is retried on the next resume run instead.
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _is_transient(exc) -> bool:
    """True if *exc* is a transient API condition worth retrying in-run."""
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    return getattr(exc, "status_code", None) in _TRANSIENT_STATUS


def _call_with_retry(client, body, i, total, custom_id,
                     max_attempts=3, base_delay=2.0):
    """Issue one Responses API call, retrying only transient errors with
    exponential backoff.  Returns the response, or raises the last exception
    once retries are exhausted / the error is non-transient."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.responses.create(**body)
        except Exception as e:  # noqa: BLE001
            if attempt >= max_attempts or not _is_transient(e):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            code = getattr(e, "status_code", "?")
            print(f"  [{i}/{total}] {custom_id} transient {code}; "
                  f"retry {attempt}/{max_attempts - 1} in {delay:.0f}s", flush=True)
            time.sleep(delay)


def _dispatch_openai_sequential(args, config, client, model, requests, line_ids,
                                master_index):
    """Run each plant's shards as near-synchronous, resumable Responses calls.

    Deliberately ``client.responses.create()`` with the same body the batch path
    writes to its JSONL — not the sync chat-completions provider.  If the two
    dispatches sent different request formats their results would not be
    comparable, and their ``_partial/`` stores could not be mixed to recover a
    batch's gaps, which is the whole point of the shared store.

    Crash-safe + resumable, like the Anthropic sequential path: each successful
    shard's parsed JSON is written immediately to
    ``<output>/_partial/<line_id>__<shard_id>.json`` and, because requests are
    plant-contiguous, a plant's merged ``<line_id>.json`` is written as soon as
    its last shard is attempted.  On restart (``--resume``, default on) any shard
    whose partial already exists and parses is skipped rather than re-billed.
    """
    group_order, group_traits, trait_meta = master_index

    out = Path(args.output)
    partial_dir = out / "_partial"
    out.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    assert_partial_provenance(partial_dir, "openai", model)
    progress_path = partial_dir / "progress.jsonl"

    # Expected shard count per plant (requests are built plant-contiguous).
    expected: Dict[str, int] = {lid: 0 for lid in line_ids}
    for req in requests:
        lid, _ = sharding.split_custom_id(req["custom_id"])
        expected[lid] = expected.get(lid, 0) + 1

    per_line: Dict[str, List] = {lid: [] for lid in line_ids}
    attempted: Dict[str, int] = {lid: 0 for lid in line_ids}
    plant_missing: Dict[str, list] = {}
    written_plants = set()

    def _partial_path(custom_id):
        return partial_dir / f"{custom_id}.json"

    def _log_progress(entry):
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _finalize_plant(lid):
        record, missing = sharding.merge_plant_record(
            per_line.get(lid, []), group_order, group_traits, trait_meta
        )
        write_json_atomic(out / f"{lid}.json", record)
        gaps_path = out / f"{lid}.gaps.json"
        if missing:
            write_json_atomic(gaps_path, {
                "line_id": lid,
                "missing_traits": [{"group": g, "trait": t} for g, t in missing],
            })
        elif gaps_path.exists():
            gaps_path.unlink()  # a prior run's gap was filled this run
        written_plants.add(lid)
        plant_missing[lid] = missing

    def _bump_and_maybe_finalize(lid):
        attempted[lid] = attempted.get(lid, 0) + 1
        if attempted[lid] >= expected.get(lid, 0) and lid not in written_plants:
            _finalize_plant(lid)

    # ---- Resume scan: adopt valid existing partials, skip those calls ----
    resume = getattr(args, "resume", True)
    done = set()
    if resume:
        for req in requests:
            cid = req["custom_id"]
            p = _partial_path(cid)
            if p.name == _RUN_META_NAME:
                continue  # provenance stamp, not a shard partial
            if not p.exists():
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue  # corrupt/partial write -> re-run this shard
            lid, _ = sharding.split_custom_id(cid)
            per_line.setdefault(lid, []).append(obj)
            done.add(cid)
    if done:
        print(f"\n--- Resume: {len(done)} of {len(requests)} shard(s) already "
              f"on disk; skipping those calls ---", flush=True)

    print(f"\n--- Sequential dispatch: {len(requests)} call(s) "
          f"({len(done)} skip, {len(requests) - len(done)} to run) ---", flush=True)

    totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}
    errors = 0
    made = 0
    for i, req in enumerate(requests, 1):
        cid = req["custom_id"]
        line_id, shard_id = sharding.split_custom_id(cid)

        if cid in done:
            _bump_and_maybe_finalize(line_id)
            continue

        try:
            resp = _call_with_retry(client, req["body"], i, len(requests), cid)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(requests)}] {cid} ERROR: {e}", flush=True)
            _log_progress({"i": i, "custom_id": cid, "status": "error",
                           "detail": str(e)})
            errors += 1
            _bump_and_maybe_finalize(line_id)
            continue

        made += 1
        # Same extraction the batch path uses, so a refusal or an empty output is
        # classified identically whichever dispatch produced it.
        content, err, usage = extract_response_text_and_usage(resp)
        cr = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
        totals["input"] += usage.get("input_tokens", 0)
        totals["output"] += usage.get("output_tokens", 0)
        totals["cache_read"] += cr

        if content is None:
            print(f"  [{i}/{len(requests)}] {cid} no usable output: {err}", flush=True)
            _log_progress({"i": i, "custom_id": cid, "status": "error",
                           "detail": err, "cache_read": cr})
            errors += 1
            _bump_and_maybe_finalize(line_id)
            continue

        try:
            obj = json.loads(strip_code_fence(content))
        except json.JSONDecodeError:
            print(f"  [{i}/{len(requests)}] {cid} JSON parse failed", flush=True)
            _log_progress({"i": i, "custom_id": cid, "status": "parse_error",
                           "cache_read": cr})
            errors += 1
            _bump_and_maybe_finalize(line_id)
            continue

        # Persist the shard immediately (crash safety), then record in memory.
        write_json_atomic(_partial_path(cid), obj)
        per_line.setdefault(line_id, []).append(obj)
        _log_progress({"i": i, "custom_id": cid, "status": "ok", "cache_read": cr})
        print(f"  [{i}/{len(requests)}] {cid}  cache_read={cr}", flush=True)
        _bump_and_maybe_finalize(line_id)

    # Safety net: finalize any plant not yet written (e.g. zero requests).
    for lid in line_ids:
        if lid not in written_plants:
            _finalize_plant(lid)

    written = len(written_plants)
    plants_with_gaps = sum(1 for m in plant_missing.values() if m)
    total_gaps = sum(len(m) for m in plant_missing.values())

    print(f"\n  Wrote {written} merged JSON files; {errors} call error(s) this run; "
          f"{len(done)} shard(s) skipped; "
          f"{plants_with_gaps} plant(s) with gaps ({total_gaps} missing traits)",
          flush=True)
    print_token_summary(totals)
    if done:
        print(f"  (token totals cover the {made} call(s) made this run only; "
              f"{len(done)} shard(s) reused from prior partials)", flush=True)
    print(f"\nMerged phenotype JSON files written to: {args.output}/", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Command entry points
# ---------------------------------------------------------------------------

def describe_batch_openai_command(args):
    return _run_openai_batch(args, "describe")


def phenotype_batch_openai_command(args):
    return _run_openai_batch(args, "phenotype")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _add_common_args(parser, prompts_required=True):
    parser.add_argument(
        "--input-dir", required=True,
        help="Root directory; each subdirectory = one plant line",
    )
    parser.add_argument(
        "--system-prompt", required=prompts_required, default=None,
        help="System prompt file path"
        + ("" if prompts_required else
           ". Required in single-schema mode. With --shard-dir it is optional "
           "and overrides the shard set's shared preamble (system block)."),
    )
    parser.add_argument(
        "--prompt", required=prompts_required, default=None,
        help="User prompt file path"
        + ("" if prompts_required else
           ". Required in single-schema mode; ignored with --shard-dir "
           "(per-shard prompts come from the shard set)."),
    )
    parser.add_argument(
        "--manifest", default="openai_file_manifest.json",
        help="Path to the OpenAI Files-API manifest "
             "(default: openai_file_manifest.json); ignored with --no-files-api",
    )
    parser.add_argument(
        "--no-files-api", action="store_true",
        help="Disable the Files API and embed images inline as base64 in each "
             "request (default: use the Files API). Can also be set via "
             "USE_FILES_API=false in the environment / .env",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Poll until batch completes and write output immediately",
    )


def setup_describe_batch_openai_parser(subparsers):
    parser = subparsers.add_parser(
        "describe-batch-openai",
        help="Stage 1 (OpenAI): batch plant descriptions via the OpenAI Batch API",
        description=(
            "Upload plant images once via the OpenAI Files API, then submit an "
            "OpenAI Message Batch for descriptions (one per plant line)."
        ),
    )
    _add_common_args(parser)
    parser.add_argument(
        "--output", required=True,
        help="Output text file (grouped descriptions, one section per plant line)",
    )
    parser.add_argument(
        "--effort",
        choices=["off", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="OpenAI reasoning effort (overrides DESCRIBE_EFFORT). "
             "default = off = none = no reasoning, sent as reasoning.effort "
             "'none'; a level enables reasoning (TEMPERATURE is then omitted, "
             "because only effort 'none' accepts a custom temperature).",
    )
    parser.set_defaults(func=describe_batch_openai_command)


def setup_phenotype_batch_openai_parser(subparsers):
    parser = subparsers.add_parser(
        "phenotype-batch-openai",
        help="Stage 3 (OpenAI): batch structured phenotyping via the OpenAI Batch API",
        description=(
            "Upload plant images (reusing the OpenAI manifest when available) and "
            "submit an OpenAI Message Batch that returns schema-valid JSON for each "
            "plant line using strict structured output."
        ),
    )
    _add_common_args(parser, prompts_required=False)
    schema_source = parser.add_mutually_exclusive_group(required=True)
    schema_source.add_argument(
        "--schema",
        help="JSON schema file path (normalized in memory for OpenAI strict mode; "
             "file not modified). One schema for the whole record; mutually "
             "exclusive with --shard-dir.",
    )
    schema_source.add_argument(
        "--shard-dir",
        help="Enable SHARDED mode: directory of per-shard {schema, prompt} pairs "
             "+ shards_manifest.json produced by build_stage3.py. Each plant is "
             "scored with one small schema per shard and the shard outputs are "
             "merged into one record per plant. Mutually exclusive with --schema.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory; one {line_id}.json file per plant line",
    )
    parser.add_argument(
        "--master-schema", default=None,
        help="Path to the master schema (sharded mode) used to validate the merged "
             "record and report missing traits. Defaults to the path recorded in "
             "the shard manifest.",
    )
    parser.add_argument(
        "--allow-reshard", action="store_true", default=False,
        help="Sharded mode: if a shard schema is rejected for exceeding an OpenAI "
             "schema size limit, let the tool regenerate the whole shard set at a "
             "halved budget. WARNING: this OVERWRITES the schema, prompt and "
             "manifest files inside --shard-dir. Off by default, in which case a "
             "rejected shard aborts the run and leaves --shard-dir untouched.",
    )
    parser.add_argument(
        "--dispatch", choices=("batch", "sequential"), default="batch",
        help="Sharded dispatch strategy: 'batch' (default; one Batch API job for "
             "all plant×shard requests) or 'sequential' (each plant's shards run as "
             "near-synchronous calls with incremental persistence and resume).",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Sequential dispatch only: resume from shards already saved under "
             "<output>/_partial/ instead of re-running (and re-billing) completed "
             "calls (default: --resume). Use --no-resume to force a fresh run.",
    )
    parser.set_defaults(func=phenotype_batch_openai_command)
