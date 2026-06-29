"""Retrieve results for a pending or completed batch.

Loads the checkpoint written by ``describe-batch`` / ``phenotype-batch``
(Anthropic) or ``describe-batch-openai`` / ``phenotype-batch-openai`` (OpenAI),
checks the batch status, and writes the output once the batch has ended.

The checkpoint ``provider`` field selects the backend; checkpoints written
before that field existed default to ``anthropic``.
"""

import json
import argparse
from pathlib import Path

from ..core.config import Config
from ..core.batch_utils import (
    write_describe_results,
    write_phenotype_results,
    write_phenotype_sharded_results,
    print_token_summary,
)
from ..core import sharding


def _sharded_master_index(checkpoint):
    """Rebuild the (group_order, group_traits, trait_meta) index for a sharded run.

    Prefer the master schema recorded in the checkpoint; fall back to the shard
    manifest's trait inventory.  Returns ``None`` (after printing) if neither is
    available.
    """
    master_schema = checkpoint.get("master_schema")
    if master_schema and Path(master_schema).exists():
        return sharding.load_master_index(master_schema)

    shard_dir = checkpoint.get("shard_dir")
    if shard_dir and (Path(shard_dir) / sharding.MANIFEST_NAME).exists():
        manifest, _ = sharding.load_shard_set(shard_dir)
        print("  Note: master schema not found; using the shard manifest's trait "
              "inventory for merge/validation.")
        return sharding.master_index_from_manifest(manifest)

    print("Error: cannot locate the master schema or shard manifest recorded in "
          "the checkpoint; needed to merge sharded results.")
    return None


def _fetch_anthropic(config, checkpoint, output):
    from anthropic import Anthropic

    if not config.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY is not set")
        return 1

    batch_id = checkpoint["batch_id"]
    stage = checkpoint["stage"]
    line_ids = checkpoint["line_ids"]
    client = Anthropic(api_key=config.anthropic_api_key, max_retries=0)

    batch = client.beta.messages.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"Batch:  {batch_id}")
    print(f"Stage:  {stage}  (provider: anthropic)")
    print(f"Status: {batch.processing_status}  "
          f"(succeeded={counts.succeeded}, errored={counts.errored}, "
          f"processing={counts.processing})")

    if batch.processing_status != "ended":
        print("\nBatch is still processing.  Run this command again when it finishes.")
        return 0

    print(f"\n--- Writing results to: {output} ---")
    if stage == "describe":
        totals = write_describe_results(client, batch_id, line_ids, output)
        print(f"Descriptions written to: {output}")
    elif stage == "phenotype":
        totals = write_phenotype_results(client, batch_id, line_ids, output)
        print(f"Phenotype JSON files written to: {output}/")
    elif stage == "phenotype_sharded":
        master_index = _sharded_master_index(checkpoint)
        if master_index is None:
            return 1
        totals = write_phenotype_sharded_results(
            client, batch_id, line_ids, master_index, output
        )
        print(f"Merged phenotype JSON files written to: {output}/")
    else:
        print(f"Error: unknown stage in checkpoint: {stage!r}")
        return 1

    print_token_summary(totals)
    return 0


def _fetch_openai(config, checkpoint, output):
    from openai import OpenAI
    from ..core.openai_batch_utils import (
        write_openai_describe_results,
        write_openai_phenotype_results,
    )

    if not config.openai_api_key:
        print("Error: OPENAI_API_KEY is not set")
        return 1

    batch_id = checkpoint["batch_id"]
    stage = checkpoint["stage"]
    line_ids = checkpoint["line_ids"]
    kwargs = {"api_key": config.openai_api_key, "max_retries": 0}
    if config.openai_base_url:
        kwargs["base_url"] = config.openai_base_url
    client = OpenAI(**kwargs)

    batch = client.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"Batch:  {batch_id}")
    print(f"Stage:  {stage}  (provider: openai)")
    print(f"Status: {batch.status}  "
          f"(completed={getattr(counts, 'completed', 0)}, "
          f"failed={getattr(counts, 'failed', 0)}, "
          f"total={getattr(counts, 'total', 0)})")

    if batch.status not in {"completed", "failed", "expired", "cancelled"}:
        print("\nBatch is still processing.  Run this command again when it finishes.")
        return 0
    if batch.status != "completed":
        print(f"\nNote: batch ended with status '{batch.status}'; "
              f"writing whatever results are available.")

    print(f"\n--- Writing results to: {output} ---")
    if stage == "describe":
        totals = write_openai_describe_results(client, batch, line_ids, output)
        print(f"Descriptions written to: {output}")
    elif stage == "phenotype":
        totals = write_openai_phenotype_results(client, batch, line_ids, output)
        print(f"Phenotype JSON files written to: {output}/")
    else:
        print(f"Error: unknown stage in checkpoint: {stage!r}")
        return 1

    print_token_summary(totals)
    return 0


def fetch_results_command(args):
    config = Config.from_env()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint file not found: {checkpoint_path}")
        return 1

    with open(checkpoint_path, encoding="utf-8") as f:
        checkpoint = json.load(f)

    provider = checkpoint.get("provider", "anthropic")
    output = args.output or checkpoint["output"]

    if provider == "anthropic":
        return _fetch_anthropic(config, checkpoint, output)
    elif provider == "openai":
        return _fetch_openai(config, checkpoint, output)
    else:
        print(f"Error: unknown provider in checkpoint: {provider!r}")
        return 1


def setup_fetch_results_parser(subparsers):
    parser = subparsers.add_parser(
        "fetch-results",
        help="Retrieve results for a submitted batch (Anthropic or OpenAI)",
        description=(
            "Load a checkpoint file (written by any of the *-batch commands) "
            "and write the results if the batch has ended."
        ),
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to the checkpoint_<batch_id>.json file",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override the output path from the checkpoint",
    )
    parser.set_defaults(func=fetch_results_command)
