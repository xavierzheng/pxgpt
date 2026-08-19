"""Image discovery, ordering and the two transports.

Ordering is not cosmetic here.  A plant's nine shards send the same photos nine
times; if the order differs between them the server's prefix cache misses from
the first differing block onward and the 97-99 % hit rate this whole run is
costed on collapses.  So the order is pinned by a test rather than left to
whatever ``iterdir`` returns on the day.
"""

from pathlib import Path

import pytest

from pxgpt.core.image_utils import (
    build_file_uri_content_list,
    create_image_content_list,
    create_multi_image_message,
    list_images,
)


PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d494844520000000100000001080600000"
    "01f15c4890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082"
)


def _make_folder(tmp_path):
    """A folder whose creation order is deliberately NOT the sorted order."""
    folder = tmp_path / "s0019"
    folder.mkdir()
    for name in ("c_third.png", "a_first.jpg", "b_second.jpeg", "notes.txt"):
        (folder / name).write_bytes(PNG_1x1 if name.endswith(".png") else b"\xff\xd8\xff")
    return folder


def test_all_supported_extensions_are_included_and_sorted(tmp_path):
    folder = _make_folder(tmp_path)

    names = [p.name for p in list_images(str(folder))]

    assert names == ["a_first.jpg", "b_second.jpeg", "c_third.png"]  # .txt dropped


def test_order_is_identical_across_calls(tmp_path):
    folder = _make_folder(tmp_path)

    first = create_image_content_list(str(folder), "file")
    second = create_image_content_list(str(folder), "file")

    assert first == second


def test_media_type_follows_the_extension(tmp_path):
    folder = _make_folder(tmp_path)

    blocks = create_image_content_list(str(folder))

    assert [b["source"]["media_type"] for b in blocks] == [
        "image/jpeg", "image/jpeg", "image/png",
    ]


def test_file_transport_emits_absolute_file_uris(tmp_path):
    folder = _make_folder(tmp_path)

    blocks = create_image_content_list(str(folder), "file")

    assert [b["source"]["type"] for b in blocks] == ["url", "url", "url"]
    for b in blocks:
        url = b["source"]["url"]
        assert url.startswith("file:///")           # absolute, not relative
        assert Path(url[len("file://"):]).is_file()
    # No bytes ride along — that is the whole point of the transport.
    assert all("data" not in b["source"] for b in blocks)


def test_file_transport_resolves_relative_paths(tmp_path):
    folder = _make_folder(tmp_path)
    relative = Path(folder.name) / "a_first.jpg"

    import os
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        blocks = build_file_uri_content_list([relative])
    finally:
        os.chdir(cwd)

    assert blocks[0]["source"]["url"] == f"file://{(folder / 'a_first.jpg').resolve()}"


@pytest.mark.parametrize("transport", ["base64", "file"])
def test_images_come_before_the_text_block(tmp_path, transport):
    folder = _make_folder(tmp_path)

    messages = create_multi_image_message(str(folder), "describe this", transport)

    content = messages[0]["content"]
    types = [item["type"] for item in content]
    assert types == ["image", "image", "image", "text"]
    # Stated as an invariant, not just a happy-path index: no text may precede
    # any image.  Gemma's model card and pxGPT's batch layout both require it.
    last_image = max(i for i, t in enumerate(types) if t == "image")
    first_text = min(i for i, t in enumerate(types) if t == "text")
    assert last_image < first_text


def test_unknown_transport_is_rejected(tmp_path):
    folder = _make_folder(tmp_path)

    with pytest.raises(ValueError, match="Unknown image transport"):
        create_image_content_list(str(folder), "http")
