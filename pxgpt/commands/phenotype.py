"""Stage 3: batch structured phenotyping via the Anthropic Batch API.

Input layout
------------
Same as Stage 1: ``--input-dir`` with one subdir per plant line.  By default
Stage 3 reuses the file_ids already present in the manifest from Stage 1, so
images are NOT re-uploaded if the manifest is up-to-date.

``--input-dir`` is OPTIONAL in the default (Files API) mode: if it is omitted,
the plant lines and their file_ids are reconstructed entirely from
``--manifest`` (the manifest written by ``describe-batch``), so the original
image tree need not be present on disk and nothing is uploaded.

With ``--no-files-api`` (or ``USE_FILES_API=false``) images are embedded inline
as base64 instead and the manifest is not used — in that mode ``--input-dir``
is required because the image bytes must be read from disk.

Workflow
--------
1. Normalize the user-supplied JSON schema (in memory only; the on-disk file
   is not touched — use ``pxgpt normalize-schema`` for that).
2. Images: load / reuse file_ids from the manifest (upload only if missing),
   or embed inline as base64 when the Files API is disabled.
3. Build one batch request per plant line with:
     output_config = {
         "effort":  "<stage3_effort>",          # adaptive thinking
         "format":  {"type": "json_schema", …}  # structured output
     }
4. Submit with ``files-api-2025-04-14`` beta header (omitted in base64 mode).
5. Save checkpoint and exit (fire-and-forget default).
   With ``--wait``: poll and write one JSON file per plant line/cultivar immediately.

When ``output_config.effort`` is set, temperature is NOT sent (thinking is
active on every model tier). When effort is off, whether a custom temperature
is sent instead depends on the model tier — see
``batch_utils.build_request_params``.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List

from anthropic import Anthropic, APIConnectionError, APITimeoutError

from ..core.config import Config
from ..core.file_utils import read_file_safely
from ..core.image_utils import build_file_id_content_list, build_base64_content_list
from ..core.files_manager import FilesManager, IMAGE_EXTENSIONS
from ..core.schema_utils import load_normalized
from ..core.batch_utils import (
    build_request_params,
    poll_batch,
    write_phenotype_results,
    write_phenotype_sharded_results,
    write_json_atomic,
    print_token_summary,
    extract_text_content,
    strip_code_fence,
    temperature_guard_status,
)
from ..core import sharding


def phenotype_batch_command(args):
    config = Config.from_env()
    if not config.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY is not set")
        return 1

    client = Anthropic(api_key=config.anthropic_api_key, max_retries=0)

    sharded = bool(args.shard_dir)

    # ------------------------------------------------------------------
    # System prompt / schema: in sharded mode these come from the shard set
    # (shared preamble + per-shard {schema, prompt}); resolved later.
    # ------------------------------------------------------------------
    system_prompt = None
    user_prompt = None
    schema = None
    output_config = None
    if sharded:
        if args.prompt or args.schema:
            print("Note: --prompt and --schema are ignored in sharded mode; "
                  "per-shard prompts and schemas come from --shard-dir.")
    else:
        if not args.schema or not args.system_prompt or not args.prompt:
            print("Error: --schema, --system-prompt and --prompt are required "
                  "without --shard-dir.")
            return 1
        try:
            system_prompt = read_file_safely(args.system_prompt, "system prompt")
            user_prompt = read_file_safely(args.prompt, "user prompt")
        except (FileNotFoundError, IOError) as e:
            print(f"File error: {e}")
            return 1
        # Load and normalize schema (in-memory only)
        try:
            schema = load_normalized(args.schema)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Schema error: {e}")
            return 1
        # Build output_config — combines effort + structured format in one dict
        output_config = config.stage3_output_config(schema=schema)
        print(f"output_config.effort: {output_config.get('effort', '(none)')}")
        print(f"output_config.format: json_schema (schema loaded from {args.schema})")
        print(f"Temperature:          "
              f"{temperature_guard_status(config.anthropic_model, output_config.get('effort', ''))}")

    use_files_api = config.use_files_api and not args.no_files_api
    line_image_blocks: Dict[str, List[Dict]] = {}

    # ------------------------------------------------------------------
    # Manifest-only mode: reuse the file_ids uploaded by `describe-batch`
    # without needing the original image tree on disk.  Requires the Files API.
    # ------------------------------------------------------------------
    if args.input_dir is None:
        if not use_files_api:
            print("Error: --input-dir is required when the Files API is disabled "
                  "(--no-files-api / USE_FILES_API=false); there are no uploaded "
                  "file_ids to reuse, so images must be read from disk.")
            return 1

        print(f"\n--- Reusing uploaded images from manifest: {args.manifest} ---")
        files_mgr = FilesManager(client, args.manifest)
        grouped = files_mgr.group_by_line()
        if not grouped:
            print(f"Error: manifest {args.manifest} is empty or missing. Run "
                  f"`pxgpt describe-batch` first, or pass --input-dir to upload images.")
            return 1

        for line_id, file_ids in grouped.items():
            print(f"  {line_id}: {len(file_ids)} image(s) reused from manifest")
            line_image_blocks[line_id] = build_file_id_content_list(file_ids)

        print(f"\nFound {len(line_image_blocks)} plant line(s) in {args.manifest}")
    else:
        # --------------------------------------------------------------
        # Discover plant lines from the image tree
        # --------------------------------------------------------------
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Error: {input_dir} is not a directory")
            return 1

        plant_lines = sorted(d for d in input_dir.iterdir() if d.is_dir())
        if not plant_lines:
            print(f"Error: no subdirectories found in {input_dir}")
            return 1

        print(f"\nFound {len(plant_lines)} plant line(s) in {input_dir}")

        # --------------------------------------------------------------
        # Collect images: reuse/upload via Files API (default) or inline base64
        # --------------------------------------------------------------
        if use_files_api:
            print(f"\n--- Checking / uploading images (manifest: {args.manifest}) ---")
            files_mgr = FilesManager(client, args.manifest)

            for line_dir in plant_lines:
                images = [p for p in line_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
                if not images:
                    print(f"  {line_dir.name}: no images, skipping")
                    continue

                already = sum(
                    1 for p in images
                    if files_mgr.get_file_id(str(p)) is not None
                )
                new_count = len(images) - already
                print(f"  {line_dir.name}: {len(images)} image(s)  "
                      f"({already} cached, {new_count} to upload)")

                file_ids = files_mgr.upload_folder(
                    str(line_dir), concurrency=config.upload_concurrency
                )
                line_image_blocks[line_dir.name] = build_file_id_content_list(file_ids)
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
                line_image_blocks[line_dir.name] = build_base64_content_list(images)

    if not line_image_blocks:
        print("Error: no images found to process")
        return 1

    # ------------------------------------------------------------------
    # Sharded mode: dispatch per (plant × shard) and merge.
    # ------------------------------------------------------------------
    if sharded:
        return _run_sharded(args, config, client, line_image_blocks, use_files_api)

    # ------------------------------------------------------------------
    # Build batch requests (single-schema mode)
    # ------------------------------------------------------------------
    print(f"\n--- Building {len(line_image_blocks)} batch request(s) ---")
    requests: List[Dict] = []
    system_blocks = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]

    for line_id, image_blocks in line_image_blocks.items():
        content = image_blocks + [{"type": "text", "text": user_prompt}]
        messages = [{"role": "user", "content": content}]

        # Temperature/thinking guard applied inside build_request_params.
        params = build_request_params(
            model=config.anthropic_model,
            max_tokens=config.stage3_max_tokens,
            system=system_blocks,
            messages=messages,
            temperature=config.temperature,
            output_config=output_config,
        )
        requests.append({"custom_id": line_id, "params": params})

    # ------------------------------------------------------------------
    # Submit batch
    # ------------------------------------------------------------------
    betas: List[str] = ["files-api-2025-04-14"] if use_files_api else []
    print(f"\n--- Submitting batch ({len(requests)} requests) ---")
    batch = client.beta.messages.batches.create(requests=requests, betas=betas)
    batch_id = batch.id
    line_ids = list(line_image_blocks.keys())

    print(f"Batch ID:  {batch_id}")
    print(f"Status:    {batch.processing_status}")

    # Save checkpoint
    checkpoint_path = f"checkpoint_{batch_id}.json"
    checkpoint = {
        "batch_id": batch_id,
        "stage": "phenotype",
        "output": args.output,
        "line_ids": line_ids,
        "model": config.anthropic_model,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
        f.write("\n")

    print(f"Checkpoint: {checkpoint_path}")

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
    poll_batch(client, batch_id)

    print("\n--- Writing results ---")
    totals = write_phenotype_results(client, batch_id, line_ids, args.output)
    print_token_summary(totals)
    print(f"\nPhenotype JSON files written to: {args.output}/")
    return 0


def _resolve_master_index(manifest, shard_dir, master_override):
    """Return (master_index, master_path) — load from master schema or manifest."""
    master_path = sharding.resolve_master_path(manifest, shard_dir, master_override)
    if master_path and Path(master_path).exists():
        return sharding.load_master_index(master_path), master_path
    print("  Note: master schema not found; using the shard manifest's trait "
          "inventory for merge/validation.")
    return sharding.master_index_from_manifest(manifest), master_path


def _run_sharded(args, config, client, line_image_blocks, use_files_api):
    """Dispatch per (plant × shard) with a cached system prefix, then merge."""
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
    print(f"output_config.effort: {config.stage3_effort or '(none)'}")
    print(f"Temperature:          "
          f"{temperature_guard_status(config.anthropic_model, config.stage3_effort)}")
    print(f"system prompt: {'--system-prompt override' if args.system_prompt else manifest.get('system_file')}")

    # Pre-flight: confirm each shard schema compiles; auto-reshard on failure.
    try:
        manifest, shards = sharding.ensure_compilable(
            client, config.anthropic_model, shard_dir, manifest, shards, master_path
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    master_index, master_path = _resolve_master_index(manifest, shard_dir, args.master_schema)

    requests = sharding.build_sharded_requests(
        line_image_blocks, shards, system_prompt, config, build_request_params
    )
    line_ids = list(line_image_blocks.keys())
    print(f"\n--- Built {len(requests)} request(s): "
          f"{len(line_ids)} plant(s) × {len(shards)} shard(s) ---")

    if args.dispatch == "sequential":
        return _dispatch_sequential(
            args, config, client, requests, line_ids, master_index, use_files_api
        )
    return _dispatch_batch(
        args, config, client, requests, line_ids, shards, shard_dir, master_path,
        use_files_api,
    )


def _dispatch_batch(args, config, client, requests, line_ids, shards, shard_dir,
                    master_path, use_files_api):
    betas = ["files-api-2025-04-14"] if use_files_api else []
    print(f"\n--- Submitting batch ({len(requests)} requests) ---")
    batch = client.beta.messages.batches.create(requests=requests, betas=betas)
    batch_id = batch.id
    print(f"Batch ID:  {batch_id}")
    print(f"Status:    {batch.processing_status}")

    checkpoint_path = f"checkpoint_{batch_id}.json"
    checkpoint = {
        "batch_id": batch_id,
        "provider": "anthropic",
        "stage": "phenotype_sharded",
        "output": args.output,
        "line_ids": line_ids,
        "model": config.anthropic_model,
        "shard_dir": str(Path(shard_dir).resolve()),
        "master_schema": str(Path(master_path).resolve()) if master_path else None,
        "shard_ids": [s["shard_id"] for s in shards],
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)
        f.write("\n")
    print(f"Checkpoint: {checkpoint_path}")

    if not args.wait:
        print(f"\nBatch submitted.  To fetch + merge results when complete:\n"
              f"  pxgpt fetch-results --checkpoint {checkpoint_path}")
        return 0

    print("\n--- Polling batch ---")
    poll_batch(client, batch_id)
    print("\n--- Writing merged results ---")
    # Resolve master_index from the (possibly resharded) manifest on disk.
    manifest, _ = sharding.load_shard_set(shard_dir)
    master_index, _ = _resolve_master_index(manifest, shard_dir, args.master_schema)
    totals = write_phenotype_sharded_results(
        client, batch_id, line_ids, master_index, args.output
    )
    print_token_summary(totals)
    print(f"\nMerged phenotype JSON files written to: {args.output}/")
    return 0


# HTTP statuses worth a short in-run retry: rate limit + transient server/
# overload conditions (429, 5xx, and Anthropic's 529 "Overloaded").  A 400 such
# as "Grammar compilation timed out" is a client-side error and is NOT retried
# here — it surfaces to the caller, writes no partial, and is retried on the
# next resume run instead.
_TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _is_transient(exc) -> bool:
    """True if *exc* is a transient API condition worth retrying in-run."""
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    return getattr(exc, "status_code", None) in _TRANSIENT_STATUS


def _call_with_retry(client, betas, params, i, total, custom_id,
                     max_attempts=3, base_delay=2.0):
    """Issue one Messages API call, retrying only transient errors with
    exponential backoff.  Returns the response, or raises the last exception
    once retries are exhausted / the error is non-transient."""
    attempt = 0
    while True:
        attempt += 1
        try:
            if betas:
                return client.beta.messages.create(betas=betas, **params)
            return client.messages.create(**params)
        except Exception as e:  # noqa: BLE001
            if attempt >= max_attempts or not _is_transient(e):
                raise
            delay = base_delay * (2 ** (attempt - 1))
            code = getattr(e, "status_code", "?")
            print(f"  [{i}/{total}] {custom_id} transient {code}; "
                  f"retry {attempt}/{max_attempts - 1} in {delay:.0f}s", flush=True)
            time.sleep(delay)


def _dispatch_sequential(args, config, client, requests, line_ids, master_index,
                         use_files_api):
    """Run each plant's shards as near-synchronous, resumable calls.

    Crash-safe + resumable.  Each successful shard's parsed JSON is written
    immediately to ``<output>/_partial/<line_id>__<shard_id>.json`` and, because
    requests are plant-contiguous, a plant's final merged ``<line_id>.json`` is
    written as soon as its last shard is attempted — so a mid-run kill loses
    nothing.  On restart (``--resume``, default on) any shard whose partial
    already exists and parses is skipped rather than re-billed.

    Token totals reflect ONLY the calls made in THIS run; shards skipped from a
    prior run contribute nothing and are reported as a separate count.  Failed /
    unparseable calls write no partial, so they are retried on the next resume.
    A clean uninterrupted run produces the same ``<line_id>.json`` /
    ``<line_id>.gaps.json`` set as before (plus the ``_partial/`` dir alongside).
    """
    betas = ["files-api-2025-04-14"] if use_files_api else []
    group_order, group_traits, trait_meta = master_index

    out = Path(args.output)
    partial_dir = out / "_partial"
    out.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
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

        params = dict(req["params"])
        try:
            resp = _call_with_retry(client, betas, params, i, len(requests), cid)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(requests)}] {cid} ERROR: {e}", flush=True)
            _log_progress({"i": i, "custom_id": cid, "status": "error",
                           "detail": str(e)})
            errors += 1
            _bump_and_maybe_finalize(line_id)
            continue

        made += 1
        u = resp.usage
        cr = getattr(u, "cache_read_input_tokens", 0)
        cc = getattr(u, "cache_creation_input_tokens", 0)
        totals["input"] += getattr(u, "input_tokens", 0)
        totals["output"] += getattr(u, "output_tokens", 0)
        totals["cache_creation"] += cc
        totals["cache_read"] += cr

        text = strip_code_fence(extract_text_content(resp.content))
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            print(f"  [{i}/{len(requests)}] {cid} JSON parse failed", flush=True)
            _log_progress({"i": i, "custom_id": cid, "status": "parse_error",
                           "cache_read": cr, "cache_creation": cc})
            errors += 1
            _bump_and_maybe_finalize(line_id)
            continue

        # Persist the shard immediately (crash safety), then record in memory.
        write_json_atomic(_partial_path(cid), obj)
        per_line.setdefault(line_id, []).append(obj)
        _log_progress({"i": i, "custom_id": cid, "status": "ok",
                       "cache_read": cr, "cache_creation": cc})
        print(f"  [{i}/{len(requests)}] {cid}  "
              f"cache_read={cr} cache_creation={cc}", flush=True)
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


def setup_phenotype_parser(subparsers):
    parser = subparsers.add_parser(
        "phenotype-batch",
        help="Stage 3: batch structured phenotyping via the Files + Batch API",
        description=(
            "Upload plant images (reusing the Stage 1 manifest when available) "
            "and submit a Message Batch that returns schema-valid JSON for each "
            "plant line using native structured output."
        ),
    )
    parser.add_argument(
        "--input-dir", default=None,
        help="Root directory; each subdirectory = one plant line. Optional when "
             "using the Files API: if omitted, plant lines and their file_ids are "
             "reconstructed from --manifest (reusing the images uploaded by "
             "describe-batch, no image tree needed). Required with --no-files-api.",
    )
    parser.add_argument(
        "--schema", default=None,
        help="JSON schema file path (normalized in memory; file not modified). "
             "Required in single-schema mode; ignored with --shard-dir (per-shard "
             "schemas come from the shard set).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory; one {line_id}.json file per plant line",
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help="System prompt file path. Required in single-schema mode. With "
             "--shard-dir it is optional and overrides the shard set's shared "
             "preamble (system block).",
    )
    parser.add_argument(
        "--prompt", default=None,
        help="User prompt file path. Required in single-schema mode; ignored with "
             "--shard-dir (per-shard prompts come from the shard set).",
    )
    parser.add_argument(
        "--shard-dir", default=None,
        help="Enable SHARDED mode: directory of per-shard {schema, prompt} pairs "
             "+ shards_manifest.json produced by build_stage3.py. Each plant is "
             "scored with one small schema per shard (cached system prefix; images "
             "are ordinary input) and the shard outputs are merged into one record "
             "per plant.",
    )
    parser.add_argument(
        "--master-schema", default=None,
        help="Path to the master schema (sharded mode) used to validate the merged "
             "record and report missing traits. Defaults to the path recorded in "
             "the shard manifest.",
    )
    parser.add_argument(
        "--dispatch", choices=("batch", "sequential"), default="batch",
        help="Sharded dispatch strategy: 'batch' (default; one Message Batch for "
             "all plant×shard requests) or 'sequential' (each plant's shards run as "
             "near-synchronous calls with incremental persistence and resume).",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Sequential dispatch only: resume from shards already saved under "
             "<output>/_partial/ instead of re-running (and re-billing) completed "
             "calls (default: --resume). Use --no-resume to force a fresh run.",
    )
    parser.add_argument(
        "--manifest", default="file_manifest.json",
        help="Path to the Files-API manifest (default: file_manifest.json); "
             "ignored when --no-files-api is set",
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
    parser.set_defaults(func=phenotype_batch_command)
