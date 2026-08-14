"""The OpenAI compile-check probe, and what it is allowed to authorise.

``ensure_compilable`` treats a probe's ``False`` as permission to reshard, which
with ``--allow-reshard`` OVERWRITES the shard set on disk.  Resharding only ever
fixes one thing: a schema too big for the backend.  So the OpenAI probe must
return ``False`` for a size-limit rejection and raise for every other schema 400
— otherwise a strict-mode formatting error, which no budget change can fix,
would be enough to overwrite a frozen shard set.

The error strings below are the real ones, measured against gpt-5.6-luna.
"""

import pytest

from pxgpt.core import sharding
from pxgpt.core.openai_batch_utils import (
    is_openai_size_limit_error,
    openai_compile_probe,
)


# --- real API wordings -------------------------------------------------------

DEPTH_400 = (
    "Error code: 400 - {'error': {'message': \"Invalid schema for response_format "
    "'limit_probe': 14 levels of nesting exceeds limit of 10.\", 'type': "
    "'invalid_request_error', 'param': 'text.format.schema', 'code': "
    "'invalid_json_schema'}}"
)
PROPERTIES_400 = (
    "Error code: 400 - {'error': {'message': \"Invalid schema for response_format "
    "'limit_probe': 6000 parameters exceeds limit of 5000.\", 'type': "
    "'invalid_request_error', 'param': 'text.format.schema', 'code': "
    "'invalid_json_schema'}}"
)
STRICT_FORMAT_400 = (
    "Error code: 400 - {'error': {'message': \"Invalid schema for response_format "
    "'stage3_shard_04': In context=(), 'additionalProperties' is required to be "
    "supplied and to be false.\", 'type': 'invalid_request_error', 'param': "
    "'text.format.schema', 'code': 'invalid_json_schema'}}"
)

SHARD_SCHEMA = {
    "title": "stage3_shard_07",
    "type": "object",
    "properties": {"value": {"enum": ["absent", "present"]}},
    "required": ["value"],
    "additionalProperties": False,
}


def test_size_limit_wordings_are_recognised():
    assert is_openai_size_limit_error(DEPTH_400)
    assert is_openai_size_limit_error(PROPERTIES_400)


def test_strict_format_error_is_not_a_size_limit():
    assert not is_openai_size_limit_error(STRICT_FORMAT_400)


def test_probe_returns_false_on_a_size_limit(monkeypatch):
    """False is the only answer that lets ensure_compilable reshard."""
    import pxgpt.core.openai_batch_utils as obu
    monkeypatch.setattr(obu, "openai_compile_check_schema",
                        lambda *a, **k: (False, DEPTH_400))

    ok, err = openai_compile_probe(None, "gpt-5.6-luna", SHARD_SCHEMA)
    assert ok is False
    assert "exceeds limit of" in err


def test_probe_raises_on_a_non_size_schema_error(monkeypatch):
    """A malformed-for-strict-mode 400 must never authorise an overwrite."""
    import pxgpt.core.openai_batch_utils as obu
    monkeypatch.setattr(obu, "openai_compile_check_schema",
                        lambda *a, **k: (False, STRICT_FORMAT_400))

    with pytest.raises(RuntimeError) as exc:
        openai_compile_probe(None, "gpt-5.6-luna", SHARD_SCHEMA)

    msg = str(exc.value)
    assert "not for a size limit" in msg
    assert "NOT modified" in msg
    assert "additionalProperties" in msg      # the underlying error is quoted
    assert "stage3_shard_07" in msg           # and which shard it was


def test_probe_names_the_format_from_the_raw_title(monkeypatch):
    """The name must be read before openai_normalize_schema strips ``title``.

    Getting this backwards is silent: every shard would be called
    ``structured_output`` and an API error could not be traced to a shard.
    """
    seen = {}

    import pxgpt.core.openai_batch_utils as obu

    def _capture(client, model, schema, name="structured_output"):
        seen["name"] = name
        seen["schema"] = schema
        return True, None

    monkeypatch.setattr(obu, "openai_compile_check_schema", _capture)

    ok, err = openai_compile_probe(None, "gpt-5.6-luna", SHARD_SCHEMA)
    assert (ok, err) == (True, None)
    assert seen["name"] == "stage3_shard_07"
    # and the schema it probed was the normalized one, not the raw file
    assert "title" not in seen["schema"]
    assert seen["schema"]["properties"]["value"]["type"] == "string"


def test_probe_does_not_mutate_the_shard_schema(monkeypatch):
    """The shard set is frozen; the in-memory copy of it is not ours to edit."""
    import copy
    import pxgpt.core.openai_batch_utils as obu
    monkeypatch.setattr(obu, "openai_compile_check_schema",
                        lambda *a, **k: (True, None))

    before = copy.deepcopy(SHARD_SCHEMA)
    openai_compile_probe(None, "gpt-5.6-luna", SHARD_SCHEMA)
    assert SHARD_SCHEMA == before


# --- ensure_compilable actually calls the injected probe ---------------------

def _shard_set():
    schema = {"title": "stage3_shard_01", "type": "object",
              "properties": {"leaf": {"type": "string"}}}
    manifest = {"version": 1, "shard_budget": 40, "shard_count": 1,
                "shards": [{"shard_id": "shard_01", "schema_file": "s.json",
                            "prompt_file": "p.md", "groups": ["leaf"],
                            "traits": ["leaf"]}]}
    return manifest, [{"shard_id": "shard_01", "schema": schema, "prompt": "x",
                       "groups": ["leaf"], "traits": ["leaf"]}]


def test_ensure_compilable_uses_the_injected_probe():
    manifest, shards = _shard_set()
    calls = []

    def _probe(client, model, schema):
        calls.append((model, schema["title"]))
        return True, None

    out_manifest, out_shards = sharding.ensure_compilable(
        client=None, model="gpt-5.6-luna", shard_dir="unused",
        manifest=manifest, shards=shards, master_path=None, probe=_probe,
    )
    assert calls == [("gpt-5.6-luna", "stage3_shard_01")]
    assert (out_manifest, out_shards) == (manifest, shards)


def test_injected_probe_failure_still_refuses_to_reshard(tmp_path):
    manifest, shards = _shard_set()

    with pytest.raises(RuntimeError) as exc:
        sharding.ensure_compilable(
            client=None, model="gpt-5.6-luna", shard_dir=str(tmp_path),
            manifest=manifest, shards=shards, master_path=None,
            allow_reshard=False,
            probe=lambda c, m, s: (False, "42 parameters exceeds limit of 10"),
        )

    assert "NOT modified" in str(exc.value)
    assert list(tmp_path.iterdir()) == []
