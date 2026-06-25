"""OpenAI Files API manager — upload images once, reuse file_ids via a manifest.

Mirrors ``core.files_manager.FilesManager`` but targets the OpenAI Files API
(``client.files.create(file=..., purpose="vision")``) instead of Anthropic's.
OpenAI file_ids live in a different namespace, so this uses its own manifest
file (default ``openai_file_manifest.json``) and must never share one with the
Anthropic manager.

The manifest maps absolute image path → OpenAI file_id and is written after
every successful upload so a partial run can be resumed without re-uploading.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

# Reuse the shared image-extension set, not-found check, and upload retry helper.
from .files_manager import IMAGE_EXTENSIONS, _is_not_found, _upload_with_retry

_MANIFEST_VERSION = 1
# OpenAI uploads images for vision with this purpose.
_VISION_PURPOSE = "vision"


class OpenAIFilesManager:
    """Upload images via ``client.files.create``; persist file_ids in a manifest."""

    def __init__(self, client, manifest_path: str):
        self._client = client
        self._manifest_path = Path(manifest_path)
        self._lock = threading.Lock()
        self._manifest: Dict[str, str] = self._load()

    # ------------------------------------------------------------------
    # Manifest I/O
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, str]:
        if self._manifest_path.exists():
            try:
                data = json.loads(self._manifest_path.read_text(encoding="utf-8"))
                return data.get("files", {})
            except (json.JSONDecodeError, KeyError):
                return {}
        return {}

    def _save(self) -> None:
        """Overwrite manifest on disk.  Must be called while holding self._lock."""
        data = {"version": _MANIFEST_VERSION, "provider": "openai", "files": self._manifest}
        self._manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_file_id(self, image_path: str) -> Optional[str]:
        """Return the cached file_id for *image_path*, or None if not uploaded."""
        key = str(Path(image_path).resolve())
        with self._lock:
            return self._manifest.get(key)

    def upload_image(self, image_path: str) -> str:
        """Upload *image_path* and return its file_id (cached if already uploaded)."""
        abs_path = str(Path(image_path).resolve())

        with self._lock:
            if abs_path in self._manifest:
                return self._manifest[abs_path]

        path = Path(image_path)

        def _do_upload():
            # Reopen the file on every attempt — a failed upload consumes it.
            with open(path, "rb") as f:
                return self._client.files.create(file=f, purpose=_VISION_PURPOSE)

        response = _upload_with_retry(_do_upload, path.name)

        with self._lock:
            self._manifest[abs_path] = response.id
            self._save()

        return response.id

    def upload_folder(self, folder_path: str, concurrency: int = 10) -> Dict[str, str]:
        """Upload all images in *folder_path* and return ``{filename: file_id}``.

        Already-uploaded images are returned immediately from the manifest.
        New uploads run in parallel up to *concurrency* threads.
        """
        folder = Path(folder_path)
        images: List[Path] = sorted(
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            return {}

        result: Dict[str, str] = {}
        to_upload: List[Path] = []

        for img in images:
            abs_path = str(img.resolve())
            with self._lock:
                cached = self._manifest.get(abs_path)
            if cached:
                result[img.name] = cached
            else:
                to_upload.append(img)

        if not to_upload:
            return result

        def _upload(img: Path):
            return img.name, self.upload_image(str(img))

        with ThreadPoolExecutor(max_workers=min(concurrency, len(to_upload))) as pool:
            futures = {pool.submit(_upload, img): img for img in to_upload}
            for fut in as_completed(futures):
                name, file_id = fut.result()
                result[name] = file_id

        # Preserve sorted order in the returned dict
        return {img.name: result[img.name] for img in images if img.name in result}

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def stats(self) -> Dict[str, int]:
        """Return ``{total: N}`` count of cached entries."""
        with self._lock:
            return {"total": len(self._manifest)}

    def delete_all(self, dry_run: bool = False):
        """Delete every uploaded file in the manifest via the OpenAI Files API.

        Returns ``(deleted_ids, failed)`` where *failed* is a list of
        ``(file_id, error)`` tuples.  A file that is already gone (404) counts
        as deleted.  Successfully deleted entries are pruned from the manifest.
        """
        with self._lock:
            items = list(self._manifest.items())

        deleted: List[str] = []
        failed: List = []
        for path, fid in items:
            if dry_run:
                deleted.append(fid)
                continue
            try:
                self._client.files.delete(fid)
            except Exception as e:  # noqa: BLE001 - report, keep going
                if not _is_not_found(e):
                    failed.append((fid, str(e)))
                    continue
            deleted.append(fid)
            with self._lock:
                self._manifest.pop(path, None)
                self._save()
        return deleted, failed
