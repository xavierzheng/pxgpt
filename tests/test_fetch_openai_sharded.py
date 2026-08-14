"""fetch-results must route an OpenAI sharded checkpoint to the sharded writer.

The failure mode this guards is silent: ``stage`` falling through to the
non-sharded ``write_openai_phenotype_results`` would write one file per
``<line_id>__<shard_id>`` custom_id instead of one merged record per plant, and
report success while doing it.

``_sharded_master_index`` is deliberately shared with the Anthropic path — both
providers write the same ``shard_dir`` / ``master_schema`` checkpoint fields — so
it is exercised here through the real checkpoint, not stubbed.
"""

import json
import types

import pytest

from pxgpt.core.config import Config
from pxgpt.commands import fetch_results as fr


SHARD_A = {"g1": {"t1": {"rationale": "r1", "value": "A"}}}
# t2 is quantitative and arrives as a STRING, so the merged record proves the
# shared parse_value ran: it must come back as a number.
SHARD_B = {"g2": {"t2": {"rationale": "r2", "value": "12.5"}}}

MANIFEST = {
    "version": 1,
    "system_file": "shards_system.md",
    "shard_budget": 40,
    "shard_count": 2,
    "shards": [
        {"shard_id": "shard_01", "schema_file": "s1.json", "prompt_file": "p1.md",
         "groups": ["g1"], "traits": ["t1"]},
        {"shard_id": "shard_02", "schema_file": "s2.json", "prompt_file": "p2.md",
         "groups": ["g2"], "traits": ["t2"]},
    ],
    "all_traits": [
        {"group": "g1", "trait": "t1", "scale_type": "nominal"},
        {"group": "g2", "trait": "t2", "scale_type": "quantitative", "unit": "cm"},
    ],
}


def _shard_dir(tmp_path):
    d = tmp_path / "shard_master_schema"
    d.mkdir()
    (d / "shards_manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    for name in ("s1.json", "s2.json"):
        (d / name).write_text("{}", encoding="utf-8")
    for name in ("p1.md", "p2.md", "shards_system.md"):
        (d / name).write_text("x", encoding="utf-8")
    return d


def _client(payloads):
    lines = [json.dumps({
        "custom_id": cid,
        "response": {"status_code": 200, "body": {
            "output": [{"type": "message", "content": [
                {"type": "output_text", "text": json.dumps(obj)}]}],
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }},
    }) for cid, obj in payloads.items()]

    class _Files:
        def content(self, file_id):
            return types.SimpleNamespace(text="\n".join(lines))

    class _Batches:
        def retrieve(self, batch_id):
            return types.SimpleNamespace(
                id=batch_id, status="completed", output_file_id="out",
                error_file_id=None,
                request_counts=types.SimpleNamespace(completed=2, failed=0, total=2))

    return types.SimpleNamespace(files=_Files(), batches=_Batches())


@pytest.fixture
def patched_openai(monkeypatch):
    """Make ``from openai import OpenAI`` inside _fetch_openai return our fake."""
    holder = {}

    def _factory(**kwargs):
        return holder["client"]

    monkeypatch.setattr("openai.OpenAI", _factory, raising=False)
    return holder


def _checkpoint(tmp_path, out, **over):
    ck = {
        "batch_id": "batch_xyz",
        "provider": "openai",
        "stage": "phenotype_sharded",
        "output": str(out),
        "line_ids": ["p1"],
        "model": "gpt-5.6-luna",
        "shard_dir": str(_shard_dir(tmp_path)),
        "master_schema": None,
        "shard_ids": ["shard_01", "shard_02"],
    }
    ck.update(over)
    return ck


def test_sharded_checkpoint_merges_one_record_per_plant(tmp_path, patched_openai):
    out = tmp_path / "out"
    patched_openai["client"] = _client({"p1__shard_01": SHARD_A,
                                        "p1__shard_02": SHARD_B})
    config = Config(openai_api_key="k", openai_model="gpt-5.6-luna")

    rc = fr._fetch_openai(config, _checkpoint(tmp_path, out), str(out))

    assert rc == 0
    # merged per plant, NOT one file per custom_id
    assert (out / "p1.json").exists()
    assert not (out / "p1__shard_01.json").exists()
    record = json.loads((out / "p1.json").read_text())
    assert record["g1"]["t1"]["value"] == "A"
    # quantitative trait parsed from the string "12.5" by the shared merge
    assert record["g2"]["t2"]["value"] == 12.5


def test_sharded_fetch_stamps_the_partial_store_as_openai(tmp_path, patched_openai):
    out = tmp_path / "out"
    patched_openai["client"] = _client({"p1__shard_01": SHARD_A})
    config = Config(openai_api_key="k", openai_model="gpt-5.6-luna")

    fr._fetch_openai(config, _checkpoint(tmp_path, out), str(out))

    meta = json.loads((out / "_partial" / ".run.json").read_text())
    assert (meta["provider"], meta["model"]) == ("openai", "gpt-5.6-luna")
    assert (out / "_partial" / "p1__shard_01.json").exists()


def test_checkpoint_without_a_model_warns_and_falls_back(tmp_path, patched_openai,
                                                         capsys):
    out = tmp_path / "out"
    patched_openai["client"] = _client({"p1__shard_01": SHARD_A})
    config = Config(openai_api_key="k", openai_model="gpt-5.6-luna")

    fr._fetch_openai(config, _checkpoint(tmp_path, out, model=None), str(out))

    assert "WARNING" in capsys.readouterr().out
    meta = json.loads((out / "_partial" / ".run.json").read_text())
    assert meta["model"] == "gpt-5.6-luna"


def test_unresolvable_master_index_fails_before_writing(tmp_path, patched_openai):
    out = tmp_path / "out"
    patched_openai["client"] = _client({"p1__shard_01": SHARD_A})
    config = Config(openai_api_key="k", openai_model="gpt-5.6-luna")
    ck = _checkpoint(tmp_path, out, shard_dir=str(tmp_path / "gone"),
                     master_schema=None)

    rc = fr._fetch_openai(config, ck, str(out))

    assert rc == 1
    assert not out.exists() or not list(out.glob("*.json"))
