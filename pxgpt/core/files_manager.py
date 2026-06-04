"""Files API manager — upload images once, reuse file_ids via a persistent manifest.

The manifest is a JSON file mapping absolute image path → file_id.  It is
written after every successful upload so a partial run can always be resumed
without re-uploading already-stored files.

Concurrent uploads are handled with a ThreadPoolExecutor; a threading.Lock
protects the in-memory manifest and all disk writes.
"""

import json
import mimetypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from anthropic import Anthropic

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
_MANIFEST_VERSION = 1


class FilesManager:
    """Upload images via ``client.beta.files``; persist file_ids in a manifest."""

    def __init__(self, client: Anthropic, manifest_path: str):
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
        data = {"version": _MANIFEST_VERSION, "files": self._manifest}
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
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "image/jpeg"

        with open(path, "rb") as f:
            response = self._client.beta.files.upload(file=(path.name, f, mime_type))

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
            file_id = self.upload_image(str(img))
            return img.name, file_id

        with ThreadPoolExecutor(max_workers=min(concurrency, len(to_upload))) as pool:
            futures = {pool.submit(_upload, img): img for img in to_upload}
            for fut in as_completed(futures):
                name, file_id = fut.result()
                result[name] = file_id

        # Preserve sorted order in returned dict
        return {img.name: result[img.name] for img in images if img.name in result}

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def stats(self) -> Dict[str, int]:
        """Return ``{total: N}`` count of cached entries."""
        with self._lock:
            return {"total": len(self._manifest)}
