"""Build a sharded Stage 3 schema/prompt set from a master schema.

Splits a master phenotype schema (Stage 2 format) into per-organ-group shards,
bin-packed to a grammar-cost budget, so each Stage 3 request carries a small,
compilable structured-output schema instead of one schema too large to compile.
The shard set produced here is consumed by ``phenotype-batch --shard-dir``.
"""

import os

from ..core import shard_builder


def shard_schema_command(args):
    if not os.path.exists(args.master):
        print(f"Error: master schema not found: {args.master}")
        return 1

    shard_dir = args.shard_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.master)), "shards"
    )
    print(f"--- Sharding {args.master} -> {shard_dir} (budget {args.shard_budget}) ---")
    summary = shard_builder.generate_shards(
        args.master, shard_dir, budget=args.shard_budget,
        write_combined=args.combined, combined_dir=args.combined_dir,
    )
    shard_builder.print_summary(summary)
    if summary["problems"]:
        print("\nValidation failed; see FAILURES above.")
        return 1
    print("\nNext: pxgpt phenotype-batch --shard-dir", shard_dir,
          "--output <dir> --manifest <file_manifest.json>")
    return 0


def setup_shard_schema_parser(subparsers):
    parser = subparsers.add_parser(
        "shard-schema",
        help="Split a master schema into compilable Stage 3 shards (+ prompts)",
        description=(
            "Generate a sharded Stage 3 schema/prompt set from a master schema. "
            "Each shard is a small structured-output schema for one or more organ "
            "groups (bin-packed to a grammar-cost budget); the shared invariant "
            "preamble is written once as the cached system prompt. Consumed by "
            "`phenotype-batch --shard-dir`."
        ),
    )
    parser.add_argument(
        "--master", required=True,
        help="Path to the master schema JSON (Stage 2 output: trait_groups -> traits).",
    )
    parser.add_argument(
        "--shard-dir", default=None,
        help="Output directory for the shard set (default: <master dir>/shards).",
    )
    parser.add_argument(
        "--shard-budget", type=int, default=shard_builder.DEFAULT_SHARD_BUDGET,
        help=f"Grammar-cost budget per shard (default: {shard_builder.DEFAULT_SHARD_BUDGET}). "
             "Lower it if a shard still trips the compile limit.",
    )
    parser.add_argument(
        "--combined", action="store_true",
        help="Also write the combined (non-sharded) stage3_schema.json + "
             "stage3_prompt.md (default: shards only).",
    )
    parser.add_argument(
        "--combined-dir", default=None,
        help="Directory for the combined artifacts (default: parent of --shard-dir).",
    )
    parser.set_defaults(func=shard_schema_command)
