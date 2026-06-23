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
4. Build a JSONL request file (one ``/v1/chat/completions`` body per line).
5. Upload the JSONL (``purpose="batch"``) and create the batch.
6. Save a checkpoint and exit (fire-and-forget), or ``--wait`` to poll + write.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List

from ..core.config import Config
from ..core.file_utils import read_file_safely
from ..core.files_manager import IMAGE_EXTENSIONS
from ..core.openai_files_manager import OpenAIFilesManager
from ..core.batch_utils import print_token_summary
from ..core.openai_batch_utils import (
    build_openai_file_id_blocks,
    build_openai_base64_blocks,
    build_responses_request_body,
    build_text_format,
    openai_normalize_schema,
    write_jsonl_requests,
    poll_openai_batch,
    write_openai_describe_results,
    write_openai_phenotype_results,
)

# OpenAI Batch endpoint. The Responses API is required so images can be
# referenced by Files-API file_id (Chat Completions cannot do this).
_OPENAI_BATCH_ENDPOINT = "/v1/responses"


def _make_openai_client(config: Config):
    from openai import OpenAI

    kwargs = {"api_key": config.openai_api_key, "max_retries": 0}
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    return OpenAI(**kwargs)


def _run_openai_batch(args, stage: str) -> int:
    """Shared runner for both OpenAI batch stages.  *stage* is 'describe' or 'phenotype'."""
    config = Config.from_env()
    if not config.openai_api_key:
        print("Error: OPENAI_API_KEY is not set")
        return 1

    model = config.openai_model
    client = _make_openai_client(config)

    try:
        system_prompt = read_file_safely(args.system_prompt, "system prompt")
        user_prompt = read_file_safely(args.prompt, "user prompt")
    except (FileNotFoundError, IOError) as e:
        print(f"File error: {e}")
        return 1

    # Phenotype: load + normalize schema for OpenAI strict structured output
    text_format = None
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
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return 1

    plant_lines = sorted(d for d in input_dir.iterdir() if d.is_dir())
    if not plant_lines:
        print(f"Error: no subdirectories found in {input_dir}")
        return 1

    print(f"Found {len(plant_lines)} plant line(s) in {input_dir}")
    print(f"Model: {model}")

    # ------------------------------------------------------------------
    # Collect images: upload via Files API (default) or embed inline base64
    # ------------------------------------------------------------------
    use_files_api = config.use_files_api and not args.no_files_api
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
            reasoning_effort=config.openai_reasoning_effort,
        )
        requests.append({
            "custom_id": line_id,
            "method": "POST",
            "url": _OPENAI_BATCH_ENDPOINT,
            "body": body,
        })

    line_ids = list(line_image_blocks.keys())

    # Write JSONL to a temp file and upload it
    fd, jsonl_path = tempfile.mkstemp(prefix=f"openai_batch_{stage}_", suffix=".jsonl", dir=".")
    os.close(fd)
    write_jsonl_requests(requests, jsonl_path)
    print(f"Batch input JSONL: {jsonl_path}")

    # ------------------------------------------------------------------
    # Submit batch
    # ------------------------------------------------------------------
    print(f"\n--- Submitting batch ({len(requests)} requests) ---")
    with open(jsonl_path, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=_OPENAI_BATCH_ENDPOINT,
        completion_window=config.openai_batch_completion_window,
    )
    batch_id = batch.id
    print(f"Batch ID:  {batch_id}")
    print(f"Status:    {batch.status}")

    # Save checkpoint (provider-tagged so fetch-results can dispatch)
    checkpoint_path = f"checkpoint_{batch_id}.json"
    checkpoint = {
        "batch_id": batch_id,
        "provider": "openai",
        "stage": stage,
        "output": args.output,
        "line_ids": line_ids,
        "model": model,
        "input_file_id": input_file.id,
        "jsonl_path": jsonl_path,
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
# Command entry points
# ---------------------------------------------------------------------------

def describe_batch_openai_command(args):
    return _run_openai_batch(args, "describe")


def phenotype_batch_openai_command(args):
    return _run_openai_batch(args, "phenotype")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _add_common_args(parser):
    parser.add_argument(
        "--input-dir", required=True,
        help="Root directory; each subdirectory = one plant line",
    )
    parser.add_argument(
        "--system-prompt", required=True, help="System prompt file path",
    )
    parser.add_argument(
        "--prompt", required=True, help="User prompt file path",
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
    _add_common_args(parser)
    parser.add_argument(
        "--schema", required=True,
        help="JSON schema file path (normalized in memory for OpenAI strict mode; "
             "file not modified)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output directory; one {line_id}.json file per plant line",
    )
    parser.set_defaults(func=phenotype_batch_openai_command)
