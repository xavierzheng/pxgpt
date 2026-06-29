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

Stage 3 uses ``output_config.effort`` → temperature is NOT sent.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

from anthropic import Anthropic

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
    print_token_summary,
    extract_text_content,
    strip_code_fence,
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

        # Stage 3: effort is set → temperature guard omits temperature
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
    """Dispatch per (plant × shard) with a cached system+image prefix, then merge."""
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


def _dispatch_sequential(args, config, client, requests, line_ids, master_index,
                         use_files_api):
    """Run each plant's shards as near-synchronous calls (reliable image cache)."""
    betas = ["files-api-2025-04-14"] if use_files_api else []
    totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}
    per_line = {lid: [] for lid in line_ids}
    errors = 0

    print(f"\n--- Sequential dispatch: {len(requests)} call(s) ---")
    for i, req in enumerate(requests, 1):
        line_id, shard_id = sharding.split_custom_id(req["custom_id"])
        params = dict(req["params"])
        try:
            if betas:
                resp = client.beta.messages.create(betas=betas, **params)
            else:
                resp = client.messages.create(**params)
        except Exception as e:  # noqa: BLE001
            print(f"  [{i}/{len(requests)}] {req['custom_id']} ERROR: {e}")
            errors += 1
            continue

        u = resp.usage
        totals["input"] += getattr(u, "input_tokens", 0)
        totals["output"] += getattr(u, "output_tokens", 0)
        totals["cache_creation"] += getattr(u, "cache_creation_input_tokens", 0)
        totals["cache_read"] += getattr(u, "cache_read_input_tokens", 0)
        text = strip_code_fence(extract_text_content(resp.content))
        try:
            per_line.setdefault(line_id, []).append(json.loads(text))
        except json.JSONDecodeError:
            print(f"  [{i}/{len(requests)}] {req['custom_id']} JSON parse failed")
            errors += 1
        print(f"  [{i}/{len(requests)}] {req['custom_id']}  "
              f"cache_read={getattr(u, 'cache_read_input_tokens', 0)} "
              f"cache_creation={getattr(u, 'cache_creation_input_tokens', 0)}")

    # Merge + write per plant
    group_order, group_traits, trait_meta = master_index
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    written = total_gaps = plants_with_gaps = 0
    for lid in line_ids:
        record, missing = sharding.merge_plant_record(
            per_line.get(lid, []), group_order, group_traits, trait_meta
        )
        with open(out / f"{lid}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        written += 1
        if missing:
            plants_with_gaps += 1
            total_gaps += len(missing)
            with open(out / f"{lid}.gaps.json", "w", encoding="utf-8") as f:
                json.dump({"line_id": lid,
                           "missing_traits": [{"group": g, "trait": t} for g, t in missing]},
                          f, indent=2)
                f.write("\n")

    print(f"\n  Wrote {written} merged JSON files; {errors} call error(s); "
          f"{plants_with_gaps} plant(s) with gaps ({total_gaps} missing traits)")
    print_token_summary(totals)
    print(f"\nMerged phenotype JSON files written to: {args.output}/")
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
             "scored with one small schema per shard (sharing a cached system+image "
             "prefix) and the shard outputs are merged into one record per plant.",
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
             "near-synchronous calls so the 5-min image cache reliably hits).",
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
