"""Image processing utilities."""

import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Iterable


# Explicit extension → media_type map, and the single source of truth for which
# image formats pxGPT accepts anywhere.  Plant photographs are .jpg/.jpeg/.png;
# .gif and .webp are deliberately NOT supported.  Resolving the media type here
# rather than through mimetypes keeps it consistent across platforms.
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
}

# Every image-discovery path filters on this set (matched against
# ``p.suffix.lower()``, so upper-case extensions are accepted too).
IMAGE_EXTENSIONS = frozenset(_MEDIA_TYPES)


def get_base64_encoded_image(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_base64_content_list(image_paths: Iterable) -> List[Dict[str, Any]]:
    """Return base64 image content blocks for the given image paths.

    Used by the batch stages when the Files API is disabled: each image is
    embedded inline in the request rather than referenced by file_id. The
    media_type is derived per file, so .png is labelled correctly rather than
    being sent as jpeg. Input order is preserved, so callers should pass an
    already-sorted list.
    """
    blocks: List[Dict[str, Any]] = []
    for p in image_paths:
        p = Path(p)
        media_type = _MEDIA_TYPES.get(p.suffix.lower())
        if not media_type:
            media_type, _ = mimetypes.guess_type(str(p))
        if not media_type:
            media_type = "image/jpeg"
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": get_base64_encoded_image(str(p)),
                },
            }
        )
    return blocks


def create_image_content_list(folder_path: str) -> List[Dict[str, Any]]:
    """Return base64 image content blocks for every supported image in *folder_path*.

    Uses the same IMAGE_EXTENSIONS filter and per-file media_type as the batch
    stages, so the sync ``analyze`` / ``schema`` commands accept exactly the same
    formats. Sorted by filename for a stable image order.
    """
    image_paths = sorted(
        p for p in Path(folder_path).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return build_base64_content_list(image_paths)


def create_multi_image_message(folder_path: str, prompt_text: str) -> List[Dict[str, Any]]:
    """Return a messages list with base64 images followed by the text prompt."""
    content = create_image_content_list(folder_path)
    content.append({"type": "text", "text": prompt_text})
    return [{"role": "user", "content": content}]


def build_file_id_content_list(file_ids: Dict[str, str]) -> List[Dict[str, Any]]:
    """Return image content blocks referencing Files-API file_ids.

    Used in beta batch requests where images have already been uploaded.
    Preserves insertion order of *file_ids* (filename → file_id mapping).
    """
    return [
        {
            "type": "image",
            "source": {"type": "file", "file_id": fid},
        }
        for fid in file_ids.values()
    ]
