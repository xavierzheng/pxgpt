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


def build_file_uri_content_list(image_paths: Iterable) -> List[Dict[str, Any]]:
    """Return ``file://`` URI image content blocks for the given image paths.

    Shaped as an Anthropic url source block (Anthropic natively supports
    ``source.type == "url"``), which ``OpenAICompatProvider._to_openai_messages``
    turns into an OpenAI ``image_url``.  This is the transport the local vLLM
    server wants: the bytes are never sent, the server reads them off the same
    mount, so a whole plant line costs one path per photo instead of megabytes
    of base64.

    Paths are resolved to absolute.  Whether the path is readable by the server
    is deliberately NOT checked here -- pxGPT does not know the server's mount
    layout, and guessing wrong is worse than letting the server say so.  Input
    order is preserved, so callers should pass an already-sorted list.
    """
    return [
        {
            "type": "image",
            "source": {"type": "url", "url": f"file://{Path(p).resolve()}"},
        }
        for p in image_paths
    ]


IMAGE_TRANSPORTS = ("base64", "file")

_TRANSPORT_BUILDERS = {
    "base64": build_base64_content_list,
    "file": build_file_uri_content_list,
}


def list_images(folder_path: str) -> List[Path]:
    """Return every supported image in *folder_path*, sorted by filename.

    The single image-discovery point for the sync commands.  Sorting matters
    beyond tidiness: the same plant's shards must present their photos in the
    same order every time or the server's prefix cache misses from the first
    differing block onward.
    """
    folder = Path(folder_path)
    images = sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        # Never return an empty list.  A request built from zero images is still
        # a valid request: the model answers the prompt from nothing, every shard
        # still validates against its schema, and the run looks perfectly
        # healthy while the data is invented.  The usual cause is pointing at the
        # tree of plant folders instead of one plant, so say so.
        subdirs = sorted(d.name for d in folder.iterdir() if d.is_dir())
        hint = ""
        if subdirs:
            hint = (f"  It holds {len(subdirs)} subdirector(y/ies) "
                    f"({', '.join(subdirs[:3])}{', ...' if len(subdirs) > 3 else ''})"
                    f" — this looks like a tree of plant folders, so point at one "
                    f"of them instead.")
        raise ValueError(
            f"No images ({', '.join(sorted(IMAGE_EXTENSIONS))}) directly inside "
            f"{folder}.{hint}"
        )
    return images


def create_image_content_list(folder_path: str,
                              transport: str = "base64") -> List[Dict[str, Any]]:
    """Return image content blocks for every supported image in *folder_path*.

    Uses the same IMAGE_EXTENSIONS filter and per-file media_type as the batch
    stages, so the sync ``analyze`` / ``schema`` commands accept exactly the same
    formats. Sorted by filename for a stable image order.  *transport* selects
    the block builder: ``base64`` (inline bytes, works everywhere) or ``file``
    (``file://`` URI, local vLLM only).
    """
    try:
        builder = _TRANSPORT_BUILDERS[transport]
    except KeyError:
        raise ValueError(
            f"Unknown image transport {transport!r}; expected one of "
            f"{', '.join(IMAGE_TRANSPORTS)}"
        ) from None
    return builder(list_images(folder_path))


def create_multi_image_message(folder_path: str, prompt_text: str,
                               transport: str = "base64") -> List[Dict[str, Any]]:
    """Return a messages list with the images followed by the text prompt.

    Image blocks come first: both the Gemma model card and Anthropic's own
    guidance want image-then-text, and the batch stages already lay requests out
    this way.
    """
    content = create_image_content_list(folder_path, transport)
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
