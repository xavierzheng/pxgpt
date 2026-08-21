"""Record-level provenance: the ``_provenance`` block and the extended guard.

Run-level provenance (``_partial/.run.json``) dies with the directory it stamps.
These tests pin the block that travels inside each merged record instead, and
the schema-version half of the guard that keeps two schema versions out of one
store.  Everything runs on fixture files — no API keys, no network.
"""

import json
from datetime import datetime, timezone

import pytest

import pxgpt
from pxgpt.core import sharding
from pxgpt.core.batch_utils import (
    _RUN_META_NAME,
    assert_partial_provenance,
    merge_sharded_results,
    read_run_meta,
    write_run_meta_if_absent,
)
from pxgpt.core.provenance import PROVENANCE_KEY, build_provenance, read_schema_identity

BLOCK_KEYS = {"provider", "model", "schema_name", "schema_version",
              "pxgpt_version", "created", "run_id"}

MASTER = {
    "schema_name": "probe_master",
    "schema_version": "v1",
    "trait_groups": {
        "g1": {"traits": [
            {"trait_name": "leaf_color", "scale_type": "nominal",
             "values": ["green", "red"]},
            {"trait_name": "plant_height", "scale_type": "quantitative",
             "unit": "cm"},
        ]},
        "g2": {"traits": [
            {"trait_name": "root_vigor", "scale_type": "nominal",
             "values": ["weak", "strong"]},
        ]},
    },
}


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


@pytest.fixture
def master_path(tmp_path):
    p = tmp_path / "master_schema.json"
    _write_json(p, MASTER)
    return str(p)


@pytest.fixture
def master_index(master_path):
    return sharding.load_master_index(master_path)


def _shard_one(color="green", height=12.5):
    return {"g1": {"leaf_color": {"rationale": "r", "value": color},
                   "plant_height": {"rationale": "r", "value": height}}}


def _shard_two(vigor="strong"):
    return {"g2": {"root_vigor": {"rationale": "r", "value": vigor}}}


def _merge(out, master_index, **kwargs):
    """Merge one complete plant, returning the parsed record."""
    fresh = kwargs.pop("fresh", None)
    if fresh is None:
        fresh = {"p1__shard_01": _shard_one(), "p1__shard_02": _shard_two()}
    shard_errors = kwargs.pop("shard_errors", {})
    params = {"provider": "anthropic", "model": "claude-sonnet-5",
              "schema_name": "probe_master", "schema_version": "v1",
              "run_id": "msgbatch_abc"}
    params.update(kwargs)
    merge_sharded_results(
        fresh, shard_errors, ["p1"], master_index, str(out),
        params["provider"], params["model"], params["schema_name"],
        params["schema_version"], params["run_id"],
    )
    return json.loads((out / "p1.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# A. the block itself
# ---------------------------------------------------------------------------

def test_merged_record_carries_exactly_the_seven_provenance_fields(tmp_path, master_index):
    record = _merge(tmp_path / "out", master_index)

    prov = record[PROVENANCE_KEY]
    assert set(prov) == BLOCK_KEYS
    assert prov["provider"] == "anthropic"
    assert prov["model"] == "claude-sonnet-5"
    assert prov["schema_name"] == "probe_master"
    assert prov["schema_version"] == "v1"
    assert prov["run_id"] == "msgbatch_abc"
    assert prov["pxgpt_version"] == pxgpt.__version__
    # UTC ISO-8601, to the second, with the Z
    parsed = datetime.strptime(prov["created"], "%Y-%m-%dT%H:%M:%SZ")
    assert parsed.replace(tzinfo=timezone.utc) <= datetime.now(timezone.utc)
    # the traits are untouched and the block does not pretend to be a group
    assert list(record) == [PROVENANCE_KEY, "g1", "g2"]
    assert record["g1"]["leaf_color"]["value"] == "green"


def test_provenance_never_shows_up_as_a_missing_trait(tmp_path, master_index):
    out = tmp_path / "out"
    # root_vigor's shard produced nothing: one real gap, and only that one.
    _merge(out, master_index,
           fresh={"p1__shard_01": _shard_one()},
           shard_errors={"p1": ["shard_02: overloaded_error"]})

    report = json.loads((out / "p1.gaps.json").read_text(encoding="utf-8"))
    assert report == {
        "line_id": "p1",
        "missing_traits": [{"group": "g2", "trait": "root_vigor"}],
        "shard_errors": ["shard_02: overloaded_error"],
    }
    assert PROVENANCE_KEY not in json.dumps(report["missing_traits"])


def test_re_merging_leaves_one_current_block(tmp_path, master_index):
    out = tmp_path / "out"
    first = _merge(out, master_index)
    # Same run identity (the guard demands it), later merge, new batch id.
    second = _merge(out, master_index, run_id="msgbatch_def")

    assert [k for k in second if k.startswith("_")] == [PROVENANCE_KEY]
    assert second[PROVENANCE_KEY]["run_id"] == "msgbatch_def"
    assert first[PROVENANCE_KEY]["run_id"] == "msgbatch_abc"   # not retro-edited
    assert second["g1"] == first["g1"]                          # traits unchanged
    raw = (out / "p1.json").read_text(encoding="utf-8")
    assert raw.count(f'"{PROVENANCE_KEY}"') == 1


def test_read_schema_identity_sources_and_refuses_to_guess(tmp_path, master_path):
    assert read_schema_identity(master_path) == ("probe_master", "v1")
    # A bare JSON Schema (what --schema takes) has no identity to give.
    plain = tmp_path / "shard_01.schema.json"
    _write_json(plain, {"type": "object", "properties": {}})
    assert read_schema_identity(plain) == (None, None)
    assert read_schema_identity(None) == (None, None)
    assert read_schema_identity(str(tmp_path / "nope.json")) == (None, None)


def test_build_provenance_records_nulls_rather_than_guesses():
    prov = build_provenance("openai", "gpt-5.6-luna")
    assert prov["schema_name"] is None
    assert prov["schema_version"] is None
    assert prov["run_id"] is None
    assert prov["pxgpt_version"] == pxgpt.__version__


# ---------------------------------------------------------------------------
# B. the extended .run.json guard
# ---------------------------------------------------------------------------

def test_same_model_different_schema_version_is_refused(tmp_path):
    partial = tmp_path / "_partial"
    write_run_meta_if_absent(partial, "anthropic", "claude-sonnet-5",
                             "probe_master", "v1")

    with pytest.raises(RuntimeError) as excinfo:
        assert_partial_provenance(partial, "anthropic", "claude-sonnet-5",
                                  "probe_master", "v2")

    message = str(excinfo.value)
    assert "'v1'" in message and "'v2'" in message
    assert "--output" in message
    # nothing was rewritten
    assert read_run_meta(partial)["schema_version"] == "v1"


def test_legacy_stamp_without_schema_version_is_upgraded_not_refused(tmp_path, capsys):
    partial = tmp_path / "_partial"
    partial.mkdir()
    # A stamp written before schema identity existed.
    _write_json(partial / _RUN_META_NAME, {
        "provider": "anthropic", "model": "claude-sonnet-5",
        "created": "2026-08-01T00:00:00Z",
    })

    assert_partial_provenance(partial, "anthropic", "claude-sonnet-5",
                              "probe_master", "v2")

    out = capsys.readouterr().out
    assert out.count("WARNING") == 1
    assert "schema_version" in out
    meta = read_run_meta(partial)
    assert meta["schema_version"] == "v2"
    assert meta["schema_name"] == "probe_master"
    assert meta["created"] == "2026-08-01T00:00:00Z"   # first-use time preserved


def test_a_run_that_cannot_name_its_schema_version_compares_nothing(tmp_path):
    """The local --shard-dir path merges off the manifest, so it has none to give."""
    partial = tmp_path / "_partial"
    write_run_meta_if_absent(partial, "local", "gemma-4", "probe_master", "v1")

    assert_partial_provenance(partial, "local", "gemma-4")   # no raise

    assert read_run_meta(partial)["schema_version"] == "v1"  # left alone


def test_provider_model_mismatch_still_wins_over_the_schema_check(tmp_path):
    partial = tmp_path / "_partial"
    write_run_meta_if_absent(partial, "anthropic", "claude-sonnet-5",
                             "probe_master", "v1")

    with pytest.raises(RuntimeError) as excinfo:
        assert_partial_provenance(partial, "openai", "gpt-5.6-luna",
                                  "probe_master", "v2")

    assert "provider='openai'" in str(excinfo.value)


def test_first_use_stamps_the_schema_identity(tmp_path):
    partial = tmp_path / "_partial"
    assert_partial_provenance(partial, "anthropic", "claude-sonnet-5",
                              "probe_master", "v1")

    meta = read_run_meta(partial)
    assert (meta["provider"], meta["model"]) == ("anthropic", "claude-sonnet-5")
    assert (meta["schema_name"], meta["schema_version"]) == ("probe_master", "v1")


def test_the_local_single_schema_path_stamps_json_and_passes_junk_through():
    """``pxgpt schema --schema`` writes text; only a JSON object gets the block."""
    from pxgpt.commands.schema import _stamped_text

    prov = build_provenance("local", "gemma-4", "probe_master", "v1")

    stamped = json.loads(_stamped_text(json.dumps({"g1": {"t": {"value": 1}}}), prov))
    assert list(stamped) == [PROVENANCE_KEY, "g1"]
    assert stamped[PROVENANCE_KEY]["model"] == "gemma-4"

    # A truncated / prose-wrapped answer must stay byte-identical so it can be read.
    junk = 'Here is the JSON:\n{"g1": {'
    assert _stamped_text(junk, prov) == junk
    assert _stamped_text("[1, 2, 3]", prov) == "[1, 2, 3]"


def test_merge_refuses_a_store_from_another_schema_version(tmp_path, master_index):
    out = tmp_path / "out"
    _merge(out, master_index)

    with pytest.raises(RuntimeError):
        _merge(out, master_index, schema_version="v2")
