"""Stage 3: batch structured phenotyping via the Anthropic Batch API.

Input layout
------------
Same as Stage 1: ``--input-dir`` with one subdir per plant line.  By default
Stage 3 reuses the file_ids already present in the manifest from Stage 1, so
images are NOT re-uploaded if the manifest is up-to-date.  With
``--no-files-api`` (or ``USE_FILES_API=false``) images are embedded inline as
base64 instead and the manifest is not used.

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
    print_token_summary,
)


def phenotype_batch_command(args):
    config = Config.from_env()
    if not config.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY is not set")
        return 1

    client = Anthropic(api_key=config.anthropic_api_key, max_retries=0)

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

    # ------------------------------------------------------------------
    # Discover plant lines
    # ------------------------------------------------------------------
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return 1

    plant_lines = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not plant_lines:
        print(f"Error: no subdirectories found in {input_dir}")
        return 1

    print(f"\nFound {len(plant_lines)} plant line(s) in {input_dir}")

    # ------------------------------------------------------------------
    # Collect images: reuse/upload via Files API (default) or embed inline base64
    # ------------------------------------------------------------------
    use_files_api = config.use_files_api and not args.no_files_api
    line_image_blocks: Dict[str, List[Dict]] = {}

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
    # Build batch requests
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
        "--input-dir", required=True,
        help="Root directory; each subdirectory = one plant line",
    )
    parser.add_argument(
        "--schema", required=True,
        help="JSON schema file path (normalized in memory; file not modified)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory; one {line_id}.json file per plant line",
    )
    parser.add_argument(
        "--system-prompt", required=True,
        help="System prompt file path",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="User prompt file path",
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
