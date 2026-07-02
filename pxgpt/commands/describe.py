"""Stage 1: batch plant description via the Anthropic Batch API.

Input layout
------------
``--input-dir`` must contain one subdirectory per plant line; the subdir
name becomes the ``custom_id`` and the ``cultivar_id`` in the output.
Each subdir may contain any mix of .jpg / .jpeg / .png / .gif / .webp files.

Workflow
--------
1. Discover plant-line subdirectories.
2. Images: by default upload all images via the Files API (skipping
   already-uploaded files found in the manifest). With ``--no-files-api``
   (or ``USE_FILES_API=false``) skip uploading and embed images inline as
   base64 in each request instead.
3. Build one batch request per plant line (images → file_id or base64
   content blocks + user prompt).
4. Submit the batch with the ``files-api-2025-04-14`` beta header (omitted in
   base64 mode) and optionally ``output-300k-2026-03-24`` when
   ``batch_300k_output`` is True.
5. Save a checkpoint JSON file.
6. Print the batch ID and exit (fire-and-forget default).
   With ``--wait``: poll until complete and write grouped output immediately.

Stage 1 uses NO thinking by default.  Set ``DESCRIBE_EFFORT`` (or pass
``--effort``) to enable Anthropic adaptive thinking; when effort is set, the
temperature guard omits temperature.  Whether a custom temperature is sent
when effort is off depends on the model tier — see
``batch_utils.build_request_params``.
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
from ..core.batch_utils import (
    build_request_params,
    poll_batch,
    write_describe_results,
    print_token_summary,
    temperature_guard_status,
)


def describe_batch_command(args):
    config = Config.from_env()
    if not config.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY is not set")
        return 1

    client = Anthropic(api_key=config.anthropic_api_key, max_retries=0)

    # Resolve thinking effort: --effort overrides DESCRIBE_EFFORT; "off" disables.
    effort = config.describe_effort if args.effort is None else args.effort
    if effort == "off":
        effort = ""
    describe_output_config = config.build_output_config(effort)  # {} when off
    print(f"Thinking effort: {effort or 'off'} "
          f"({temperature_guard_status(config.anthropic_model, effort)})")

    try:
        system_prompt = read_file_safely(args.system_prompt, "system prompt")
        user_prompt = read_file_safely(args.prompt, "user prompt")
    except (FileNotFoundError, IOError) as e:
        print(f"File error: {e}")
        return 1

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

    print(f"Found {len(plant_lines)} plant line(s) in {input_dir}")

    # ------------------------------------------------------------------
    # Collect images: upload via Files API (default) or embed inline as base64
    # ------------------------------------------------------------------
    use_files_api = config.use_files_api and not args.no_files_api
    line_image_blocks: Dict[str, List[Dict]] = {}

    if use_files_api:
        print(f"\n--- Uploading images (manifest: {args.manifest}) ---")
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

        if not line_image_blocks:
            print("Error: no images found to process")
            return 1

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

        # Stage 1: effort off by default → temperature included; the temperature
        # guard in build_request_params drops it automatically if effort is set.
        params = build_request_params(
            model=config.anthropic_model,
            max_tokens=config.stage1_max_tokens,
            system=system_blocks,
            messages=messages,
            temperature=config.temperature,
            output_config=describe_output_config,
        )
        requests.append({"custom_id": line_id, "params": params})

    # ------------------------------------------------------------------
    # Submit batch
    # ------------------------------------------------------------------
    betas: List[str] = []
    if use_files_api:
        betas.append("files-api-2025-04-14")
    if config.batch_300k_output:
        betas.append("output-300k-2026-03-24")
        print(f"Using 300 k output token budget (output-300k-2026-03-24)")

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
        "stage": "describe",
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
    totals = write_describe_results(client, batch_id, line_ids, args.output)
    print_token_summary(totals)
    print(f"\nDescriptions written to: {args.output}")
    return 0


def setup_describe_parser(subparsers):
    parser = subparsers.add_parser(
        "describe-batch",
        help="Stage 1: batch plant descriptions via the Files + Batch API",
        description=(
            "Upload plant images once via the Files API, then submit a "
            "Message Batch for descriptions (one per plant line)."
        ),
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Root directory; each subdirectory = one plant line",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output text file (grouped descriptions, one section per plant line/cultivar)",
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
        "--effort",
        choices=["off", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Anthropic adaptive thinking effort (overrides DESCRIBE_EFFORT). "
             "default = off = none = no reasoning; a level enables reasoning "
             "(temperature is then omitted; whether it's sent when off depends "
             "on the model tier).",
    )
    parser.add_argument(
        "--wait", action="store_true",
        help="Poll until batch completes and write output immediately",
    )
    parser.set_defaults(func=describe_batch_command)
