"""Retrieve results for a pending or completed batch.

Loads the checkpoint written by ``describe-batch`` or ``phenotype-batch``,
checks the batch status, and writes the output when the batch has ended.
"""

import json
import argparse
from pathlib import Path

from anthropic import Anthropic

from ..core.config import Config
from ..core.batch_utils import (
    write_describe_results,
    write_phenotype_results,
    print_token_summary,
)


def fetch_results_command(args):
    config = Config.from_env()
    if not config.anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY is not set")
        return 1

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: checkpoint file not found: {checkpoint_path}")
        return 1

    with open(checkpoint_path, encoding="utf-8") as f:
        checkpoint = json.load(f)

    batch_id: str = checkpoint["batch_id"]
    stage: str = checkpoint["stage"]
    output: str = args.output or checkpoint["output"]
    line_ids: list = checkpoint["line_ids"]

    client = Anthropic(api_key=config.anthropic_api_key, max_retries=0)

    batch = client.beta.messages.batches.retrieve(batch_id)
    counts = batch.request_counts
    print(f"Batch:  {batch_id}")
    print(f"Stage:  {stage}")
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
    else:
        print(f"Error: unknown stage in checkpoint: {stage!r}")
        return 1

    print_token_summary(totals)
    return 0


def setup_fetch_results_parser(subparsers):
    parser = subparsers.add_parser(
        "fetch-results",
        help="Retrieve results for a submitted batch",
        description=(
            "Load a checkpoint file (written by describe-batch or phenotype-batch) "
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
