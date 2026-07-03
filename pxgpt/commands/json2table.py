"""Flatten Stage 3 per-cultivar phenotype JSON results into a wide table."""

import json
import os

from ..core import json2table as core


def json2table_command(args):
    if not os.path.isdir(args.result_dir):
        print(f"Error: result directory not found: {args.result_dir}")
        return 1
    if not os.path.exists(args.master_schema):
        print(f"Error: master schema not found: {args.master_schema}")
        return 1
    if args.shard_dir and not os.path.isdir(args.shard_dir):
        print(f"Error: shard directory not found: {args.shard_dir}")
        return 1
    rename_map = None
    if args.rename_map:
        if not os.path.exists(args.rename_map):
            print(f"Error: rename map not found: {args.rename_map}")
            return 1
        with open(args.rename_map, encoding="utf-8") as f:
            rename_map = json.load(f)

    print(f"--- Flattening {args.result_dir} -> {args.out_prefix}.{{csv,feather}} ---")
    try:
        csv_df, feather_df, warnings = core.build_table(
            args.result_dir, args.master_schema, shard_dir=args.shard_dir,
            on_collision=args.on_collision, rename_map=rename_map,
        )
    except (core.ColumnCollisionError, core.DuplicateColumnError) as e:
        print(str(e))
        return 1

    for w in warnings:
        print(f"  WARNING: {w}")

    core.write_table(csv_df, feather_df, args.out_prefix)

    print(f"  Rows: {len(csv_df)}   Columns: {len(csv_df.columns)}")
    print(f"  Wrote {args.out_prefix}.csv")
    print(f"  Wrote {args.out_prefix}.feather")
    return 0


def setup_json2table_parser(subparsers):
    parser = subparsers.add_parser(
        "json-to-table",
        help="Flatten Stage 3 per-cultivar JSON results into a wide CSV/feather table",
        description=(
            "Flattens `Result_Stage3/<cultivar_id>.json` files into one row per "
            "cultivar. Trait metadata (scale_type / unit / ordinal levels) comes "
            "from the master schema (authoritative), falling back to the shard "
            "schemas for traits master doesn't cover. Nominal traits stay plain "
            "strings, quantitative traits become numeric `<trait>_<unit>` "
            "columns, and ordinal traits are reconstructed from their integer "
            "level code into the schema-defined label. Writes both a CSV (labels "
            "as strings) and an Arrow IPC feather file (ordinal columns as "
            "ordered pandas Categoricals, so R's arrow::read_feather() reads "
            "them as ordered factors)."
        ),
    )
    parser.add_argument(
        "--result-dir", required=True,
        help="Directory of per-cultivar Stage 3 result JSON files.",
    )
    parser.add_argument(
        "--master-schema", required=True,
        help="Path to the master phenotype schema JSON (trait_groups -> traits).",
    )
    parser.add_argument(
        "--shard-dir", default=None,
        help="Optional shard set directory (shard_*.schema.json), used only as a "
             "fallback for traits absent from the master schema.",
    )
    parser.add_argument(
        "--out-prefix", required=True,
        help="Output path prefix; writes <prefix>.csv and <prefix>.feather.",
    )
    parser.add_argument(
        "--on-collision", choices=["error", "prefix_collided", "prefix_all"],
        default="error",
        help="How to resolve two traits computing to the same output column "
             "name (e.g. the same leaf key under two organ groups). 'error' "
             "(default) stops and prints a --rename-map fill-in template; "
             "'prefix_collided' auto-prefixes only the clashing columns with "
             "the minimal path prefix needed; 'prefix_all' uses every "
             "column's full path (a safe, ugly debug escape hatch).",
    )
    parser.add_argument(
        "--rename-map", default=None,
        help="Path to a JSON file mapping dotted source path (e.g. "
             "'leaf.length') -> desired column name, used verbatim (no unit "
             "re-appended). Applied before --on-collision; see the template "
             "printed by the default 'error' mode.",
    )
    parser.set_defaults(func=json2table_command)
