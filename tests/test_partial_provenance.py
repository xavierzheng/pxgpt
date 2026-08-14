"""Provenance guard on the ``_partial/`` shard store.

The store is keyed by ``<line_id>__<shard_id>`` alone, so without a stamp two
runs sharing an ``--output`` would silently merge each other's shards.  These
cover the three states ``assert_partial_provenance`` distinguishes.
"""

import json

import pytest

from pxgpt.core.batch_utils import (
    _RUN_META_NAME,
    assert_partial_provenance,
    read_run_meta,
    write_run_meta_if_absent,
)


def test_model_mismatch_raises_and_names_both_models(tmp_path):
    partial = tmp_path / "_partial"
    partial.mkdir()
    write_run_meta_if_absent(partial, "anthropic", "claude-sonnet-5")

    with pytest.raises(RuntimeError) as exc:
        assert_partial_provenance(partial, "anthropic", "claude-opus-4-8")

    msg = str(exc.value)
    assert "claude-sonnet-5" in msg      # what the store was created with
    assert "claude-opus-4-8" in msg      # what this run is
    assert str(partial) in msg           # which directory
    assert _RUN_META_NAME in msg         # the escape hatch


def test_provider_mismatch_raises(tmp_path):
    partial = tmp_path / "_partial"
    partial.mkdir()
    write_run_meta_if_absent(partial, "anthropic", "same-model")

    with pytest.raises(RuntimeError) as exc:
        assert_partial_provenance(partial, "openai", "same-model")

    assert "anthropic" in str(exc.value)
    assert "openai" in str(exc.value)


def test_mismatch_writes_nothing(tmp_path):
    """A refused run must not touch the store."""
    partial = tmp_path / "_partial"
    partial.mkdir()
    write_run_meta_if_absent(partial, "anthropic", "model-a")
    (partial / "LINE01__shard_01.json").write_text('{"x": 1}', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in partial.iterdir()}

    with pytest.raises(RuntimeError):
        assert_partial_provenance(partial, "anthropic", "model-b")

    after = {p.name: p.read_bytes() for p in partial.iterdir()}
    assert after == before


def test_empty_dir_passes_and_stamps(tmp_path, capsys):
    partial = tmp_path / "_partial"
    partial.mkdir()

    assert_partial_provenance(partial, "openai", "gpt-5.6-luna")

    meta = read_run_meta(partial)
    assert meta["provider"] == "openai"
    assert meta["model"] == "gpt-5.6-luna"
    assert meta["created"].endswith("Z")
    assert capsys.readouterr().out == ""      # silent on a fresh store


def test_legacy_store_warns_then_stamps(tmp_path, capsys):
    partial = tmp_path / "_partial"
    partial.mkdir()
    (partial / "LINE01__shard_01.json").write_text('{"x": 1}', encoding="utf-8")

    assert_partial_provenance(partial, "anthropic", "claude-sonnet-5")

    out = capsys.readouterr().out
    assert "WARNING" in out
    assert _RUN_META_NAME in out
    meta = read_run_meta(partial)
    assert (meta["provider"], meta["model"]) == ("anthropic", "claude-sonnet-5")
    # the legacy partial itself is untouched
    assert json.loads((partial / "LINE01__shard_01.json").read_text()) == {"x": 1}


def test_matching_stamp_is_silent_and_stable(tmp_path, capsys):
    partial = tmp_path / "_partial"
    partial.mkdir()
    assert_partial_provenance(partial, "anthropic", "claude-sonnet-5")
    created_first = read_run_meta(partial)["created"]
    capsys.readouterr()

    assert_partial_provenance(partial, "anthropic", "claude-sonnet-5")

    assert capsys.readouterr().out == ""
    assert read_run_meta(partial)["created"] == created_first  # not re-stamped


def test_missing_dir_passes(tmp_path):
    """No dir yet is the normal first-run case for the sequential path."""
    assert_partial_provenance(tmp_path / "_partial", "anthropic", "claude-sonnet-5")


def test_pathlib_glob_does_match_the_stamp(tmp_path):
    """Why the adoption loops must skip the stamp by name.

    Unlike shell globbing (and ``glob.glob``), ``Path.glob("*.json")`` DOES
    return dot-prefixed files.  The dot prefix is therefore not protection on its
    own: without the explicit skip the stamp would be adopted as a shard and
    ``split_custom_id(".run")`` would yield the junk line_id ``""``.
    """
    partial = tmp_path / "_partial"
    partial.mkdir()
    (partial / _RUN_META_NAME).write_text("{}", encoding="utf-8")
    (partial / "LINE01__shard_01.json").write_text("{}", encoding="utf-8")

    assert _RUN_META_NAME in {p.name for p in partial.glob("*.json")}


def test_read_run_meta_returns_none_when_absent_or_corrupt(tmp_path):
    partial = tmp_path / "_partial"
    partial.mkdir()
    assert read_run_meta(partial) is None
    (partial / _RUN_META_NAME).write_text("{not json", encoding="utf-8")
    assert read_run_meta(partial) is None
