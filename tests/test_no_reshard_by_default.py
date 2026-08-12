"""Regression test: a failed compile-check must not rewrite a frozen shard set."""

import hashlib
import json

import pytest

from pxgpt.core import sharding


def _dir_digest(path):
    """Map every file under *path* to its sha256 (relative path -> digest)."""
    return {
        p.relative_to(path).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.rglob("*")) if p.is_file()
    }


def _write_shard_set(shard_dir):
    schema = {
        "type": "object",
        "properties": {"leaf": {"type": "string"}},
        "required": ["leaf"],
        "additionalProperties": False,
    }
    (shard_dir / "shard_01.schema.json").write_text(json.dumps(schema), encoding="utf-8")
    (shard_dir / "shard_01.prompt.md").write_text("Score shard 1.", encoding="utf-8")
    (shard_dir / "shards_system.md").write_text("Shared system prompt.", encoding="utf-8")
    manifest = {
        "version": 1,
        "system_file": "shards_system.md",
        "shard_budget": 40,
        "shard_count": 1,
        "shards": [{
            "shard_id": "shard_01",
            "schema_file": "shard_01.schema.json",
            "prompt_file": "shard_01.prompt.md",
            "groups": ["leaf"],
            "traits": ["leaf"],
        }],
        "all_traits": [{"group": "leaf", "trait": "leaf", "scale_type": "nominal"}],
    }
    (shard_dir / sharding.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_compile_failure_does_not_touch_shard_dir(tmp_path, monkeypatch):
    shard_dir = tmp_path / "shard_master_schema"
    shard_dir.mkdir()
    _write_shard_set(shard_dir)
    master_path = tmp_path / "master_schema.json"
    master_path.write_text(json.dumps({"trait_groups": {}}), encoding="utf-8")

    before = _dir_digest(shard_dir)

    # Every shard trips the grammar-complexity limit.
    monkeypatch.setattr(
        sharding, "compile_check_schema",
        lambda client, model, schema: (False, "Schema is too complex"),
    )
    # Reaching the generator at all is a failure of this test's contract.
    def _explode(*args, **kwargs):
        raise AssertionError("reshard() must not be called with allow_reshard=False")
    monkeypatch.setattr(sharding, "reshard", _explode)

    manifest, shards = sharding.load_shard_set(str(shard_dir))

    with pytest.raises(RuntimeError) as excinfo:
        sharding.ensure_compilable(
            client=None, model="claude-sonnet-test", shard_dir=str(shard_dir),
            manifest=manifest, shards=shards, master_path=str(master_path),
            allow_reshard=False,
        )

    message = str(excinfo.value)
    assert "shard_01" in message
    assert "Schema is too complex" in message
    assert "NOT modified" in message
    assert "--allow-reshard" in message
    assert "shard-schema" in message

    assert _dir_digest(shard_dir) == before
