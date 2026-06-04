"""Image processing utilities."""

import base64
from pathlib import Path
from typing import Dict, List, Any


def get_base64_encoded_image(image_path: str) -> str:
    """Convert image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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
