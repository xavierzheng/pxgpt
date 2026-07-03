"""Main CLI entry point for PXGPT."""

import argparse
import sys
import os
from pathlib import Path

from .commands.analyze import setup_analyze_parser
from .commands.schema import setup_schema_parser
from .commands.describe import setup_describe_parser
from .commands.phenotype import setup_phenotype_parser
from .commands.openai_batch import (
    setup_describe_batch_openai_parser,
    setup_phenotype_batch_openai_parser,
)
from .commands.fetch_results import setup_fetch_results_parser
from .commands.cleanup_files import setup_cleanup_files_parser
from .commands.extract_report import setup_extract_report_parser
from .commands.normalize_schema import setup_normalize_schema_parser
from .commands.shard_schema import setup_shard_schema_parser
from .commands.json2table import setup_json2table_parser


def load_env_file():
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    if key and value:
                        os.environ.setdefault(key.strip(), value.strip())


def main():
    load_env_file()

    parser = argparse.ArgumentParser(
        prog="pxgpt",
        description="Plant analysis tool — multi-provider LLM with Files + Batch API support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="PXGPT 0.3.0")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    subparsers = parser.add_subparsers(
        title="Commands",
        description="Available commands",
        help="Use pxgpt <command> --help for details",
        dest="command",
        required=True,
    )

    setup_analyze_parser(subparsers)
    setup_schema_parser(subparsers)
    setup_describe_parser(subparsers)
    setup_phenotype_parser(subparsers)
    setup_describe_batch_openai_parser(subparsers)
    setup_phenotype_batch_openai_parser(subparsers)
    setup_fetch_results_parser(subparsers)
    setup_cleanup_files_parser(subparsers)
    setup_extract_report_parser(subparsers)
    setup_normalize_schema_parser(subparsers)
    setup_shard_schema_parser(subparsers)
    setup_json2table_parser(subparsers)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    try:
        exit_code = args.func(args)
        sys.exit(exit_code or 0)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
