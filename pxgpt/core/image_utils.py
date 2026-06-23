"""Image processing utilities."""

import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Any, Iterable


# Explicit extension → media_type map. Some systems' mimetypes DB does not know
# .webp (and occasionally .gif), so we resolve known image types ourselves and
# only fall back to mimetypes/jpeg for anything else.
_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def get_base64_encoded_image(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_base64_content_list(image_paths: Iterable) -> List[Dict[str, Any]]:
    """Return base64 image content blocks for the given image paths.

    Used by the batch stages when the Files API is disabled: each image is
    embedded inline in the request rather than referenced by file_id. The
    media_type is derived per file so .png/.webp/.gif are handled correctly
    (unlike ``create_image_content_list``, which is jpeg-only). Input order is
    preserved, so callers should pass an already-sorted list.
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
    """Return base64 image content blocks for every .jpg in *folder_path*."""
    image_paths = list(Path(folder_path).glob("*.jpg"))
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": get_base64_encoded_image(str(p)),
            },
        }
        for p in image_paths
    ]


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
