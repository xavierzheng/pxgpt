"""Delete files uploaded via the Files API (Anthropic or OpenAI).

Uploaded images live on the provider's servers after a batch run.  Anthropic
keeps them free for a while, but OpenAI bills for stored files, so they should
be removed once results are fetched.

This command deletes every file_id recorded in a manifest (auto-detecting the
provider) and, for OpenAI, optionally the batch input/output/error files
referenced by one or more checkpoints.  Successfully deleted entries are pruned
from the manifest so it can be reused safely.

Examples
--------
    # delete all images in an Anthropic manifest
    pxgpt cleanup-files --manifest file_manifest.json

    # delete OpenAI images AND the batch input/output/error files
    pxgpt cleanup-files --manifest openai_file_manifest.json \
        --checkpoint checkpoint_batch_xxx.json

    # preview only, delete nothing
    pxgpt cleanup-files --manifest openai_file_manifest.json --dry-run
"""

import json
import argparse
from pathlib import Path
from typing import List

from ..core.config import Config


def _detect_provider(manifest_path: Path) -> str:
    """Return the provider recorded in a manifest ('openai' or 'anthropic')."""
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return "anthropic"
    # OpenAI manifests tag themselves; Anthropic manifests do not.
    return data.get("provider", "anthropic")


def _make_client(provider: str, config: Config):
    if provider == "anthropic":
        if not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        from anthropic import Anthropic
        return Anthropic(api_key=config.anthropic_api_key, max_retries=2)
    elif provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        from openai import OpenAI
        kwargs = {"api_key": config.openai_api_key, "max_retries": 2}
        if config.openai_base_url:
            kwargs["base_url"] = config.openai_base_url
        return OpenAI(**kwargs)
    raise ValueError(f"unknown provider: {provider!r}")


def _delete_openai_batch_files(client, checkpoint_paths: List[str], dry_run: bool):
    """Delete OpenAI batch input/output/error files referenced by checkpoints."""
    from ..core.files_manager import _is_not_found

    file_ids = set()
    for cp in checkpoint_paths:
        cp_path = Path(cp)
        if not cp_path.exists():
            print(f"  checkpoint not found, skipping: {cp}")
            continue
        data = json.loads(cp_path.read_text(encoding="utf-8"))
        if data.get("provider") != "openai":
            print(f"  not an OpenAI checkpoint, skipping: {cp}")
            continue
        if data.get("input_file_id"):
            file_ids.add(data["input_file_id"])
        batch_id = data.get("batch_id")
        if batch_id:
            try:
                b = client.batches.retrieve(batch_id)
                for attr in ("input_file_id", "output_file_id", "error_file_id"):
                    fid = getattr(b, attr, None)
                    if fid:
                        file_ids.add(fid)
            except Exception as e:  # noqa: BLE001
                print(f"  could not retrieve batch {batch_id}: {str(e)[:80]}")

    if not file_ids:
        return 0, 0
    print(f"\n--- Deleting {len(file_ids)} OpenAI batch file(s) ---")
    ok = fail = 0
    for fid in sorted(file_ids):
        if dry_run:
            print(f"  [dry-run] would delete {fid}")
            ok += 1
            continue
        try:
            client.files.delete(fid)
            ok += 1
        except Exception as e:  # noqa: BLE001
            if _is_not_found(e):
                ok += 1
            else:
                fail += 1
                print(f"  FAIL {fid}: {str(e)[:80]}")
    return ok, fail


def cleanup_files_command(args):
    config = Config.from_env()

    if not args.manifest and not args.checkpoint:
        print("Error: provide --manifest and/or --checkpoint")
        return 1

    deleted_total = failed_total = 0

    # ------------------------------------------------------------------
    # Manifest images
    # ------------------------------------------------------------------
    client = None
    provider = None
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            print(f"Error: manifest not found: {manifest_path}")
            return 1

        provider = args.provider if args.provider != "auto" else _detect_provider(manifest_path)
        print(f"Manifest:  {manifest_path}")
        print(f"Provider:  {provider}")

        try:
            client = _make_client(provider, config)
        except ValueError as e:
            print(f"Error: {e}")
            return 1

        if provider == "anthropic":
            from ..core.files_manager import FilesManager
            mgr = FilesManager(client, str(manifest_path))
        else:
            from ..core.openai_files_manager import OpenAIFilesManager
            mgr = OpenAIFilesManager(client, str(manifest_path))

        n = mgr.stats()["total"]
        print(f"\n--- Deleting {n} uploaded image file(s)"
              f"{' (dry-run)' if args.dry_run else ''} ---")
        deleted, failed = mgr.delete_all(dry_run=args.dry_run)
        deleted_total += len(deleted)
        failed_total += len(failed)
        for fid, err in failed:
            print(f"  FAIL {fid}: {err[:80]}")
        print(f"  {'would delete' if args.dry_run else 'deleted'}: {len(deleted)}; "
              f"failed: {len(failed)}")

    # ------------------------------------------------------------------
    # OpenAI batch files (from checkpoints)
    # ------------------------------------------------------------------
    if args.checkpoint:
        # Need an OpenAI client; reuse the one above only if it is OpenAI.
        oa_client = client if provider == "openai" else None
        if oa_client is None:
            try:
                oa_client = _make_client("openai", config)
            except ValueError as e:
                print(f"Error: {e}")
                return 1
        ok, fail = _delete_openai_batch_files(oa_client, args.checkpoint, args.dry_run)
        deleted_total += ok
        failed_total += fail
        print(f"  batch files {'would delete' if args.dry_run else 'deleted'}: {ok}; "
              f"failed: {fail}")

    print(f"\nTotal {'would delete' if args.dry_run else 'deleted'}: "
          f"{deleted_total};  failed: {failed_total}")
    return 1 if failed_total else 0


def setup_cleanup_files_parser(subparsers):
    parser = subparsers.add_parser(
        "cleanup-files",
        help="Delete files uploaded via the Files API (Anthropic or OpenAI)",
        description=(
            "Delete uploaded images (by file_id) recorded in a manifest, and "
            "optionally the OpenAI batch input/output/error files referenced by "
            "checkpoints.  OpenAI bills for stored files, so clean up after "
            "fetching results."
        ),
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Manifest of uploaded images to delete (provider auto-detected)",
    )
    parser.add_argument(
        "--provider", choices=["auto", "anthropic", "openai"], default="auto",
        help="Override the provider (default: auto-detect from the manifest)",
    )
    parser.add_argument(
        "--checkpoint", action="append", default=None,
        help="OpenAI checkpoint whose batch input/output/error files to delete "
             "(repeatable)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without deleting anything",
    )
    parser.set_defaults(func=cleanup_files_command)
