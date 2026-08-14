"""The two OpenAI sharded dispatches must be interchangeable.

``--dispatch batch`` and ``--dispatch sequential`` exist to be mixed: a batch
that left gaps is recovered by a sequential resume reading the same ``_partial/``
store.  That only holds if both send the *same request* — same model, same
images, same per-shard schema, same reasoning/temperature decision.  If they
diverged, the merged record would be a blend of two different experiments and
the batch-vs-sequential comparison would be meaningless.

So these tests drive the real dispatch functions with fake clients and compare
what actually goes out, rather than asserting that one builder equals itself.
"""

import json
import types
from pathlib import Path

import pytest

from pxgpt.core.config import Config
from pxgpt.core.batch_utils import _RUN_META_NAME, write_run_meta_if_absent
from pxgpt.core.openai_batch_utils import (
    build_openai_sharded_requests,
    write_openai_phenotype_sharded_results,
)
from pxgpt.commands import openai_batch as ob


# --- fixtures ----------------------------------------------------------------

GROUP_ORDER = ["g1", "g2"]
GROUP_TRAITS = {"g1": ["t1"], "g2": ["t2"]}
TRAIT_META = {("g1", "t1"): {"scale_type": "nominal"},
              ("g2", "t2"): {"scale_type": "nominal"}}
MASTER_INDEX = (GROUP_ORDER, GROUP_TRAITS, TRAIT_META)

SHARD_PAYLOAD = {
    "shard_01": {"g1": {"t1": {"rationale": "r1", "value": "A"}}},
    "shard_02": {"g2": {"t2": {"rationale": "r2", "value": "B"}}},
}

SHARDS = [
    {"shard_id": "shard_01", "prompt": "Score group g1.",
     "groups": ["g1"], "traits": ["t1"],
     "schema": {"title": "stage3_shard_01", "type": "object",
                "properties": {"g1": {"type": "object", "properties": {
                    "t1": {"type": "object", "properties": {
                        "rationale": {"type": "string"},
                        "value": {"enum": ["A", "not_assessable"]}}}}}}}},
    {"shard_id": "shard_02", "prompt": "Score group g2.",
     "groups": ["g2"], "traits": ["t2"],
     "schema": {"title": "stage3_shard_02", "type": "object",
                "properties": {"g2": {"type": "object", "properties": {
                    "t2": {"type": "object", "properties": {
                        "rationale": {"type": "string"},
                        "value": {"enum": ["B", "not_assessable"]}}}}}}}},
]

LINE_IMAGE_BLOCKS = {
    "p1": [{"type": "input_image", "file_id": "file-p1-a"},
           {"type": "input_image", "file_id": "file-p1-b"}],
    "p2": [{"type": "input_image", "file_id": "file-p2-a"}],
}

SYSTEM_PROMPT = "Shared invariant preamble."


def _config():
    return Config(openai_model="gpt-5.6-luna", stage3_max_tokens=4096,
                  temperature=0.5, stage3_effort="")


def _args(tmp_path, **over):
    base = dict(output=str(tmp_path / "out"), wait=False, resume=True,
                dispatch="batch", allow_reshard=False, master_schema=None,
                shard_dir="unused", system_prompt=None, prompt=None,
                manifest="m.json", no_files_api=False, input_dir="in")
    base.update(over)
    return types.SimpleNamespace(**base)


def _usage():
    return {"input_tokens": 100, "output_tokens": 200,
            "input_tokens_details": {"cached_tokens": 40}}


# --- fake clients ------------------------------------------------------------

class _BatchCapture:
    """Captures the JSONL a batch submission uploads."""

    def __init__(self):
        self.jsonl = None
        outer = self

        class _Files:
            def create(self, file, purpose):        # noqa: A002
                outer.jsonl = file.read().decode("utf-8")
                outer.purpose = purpose
                return types.SimpleNamespace(id="file-batch-in")

        class _Batches:
            def create(self, **kwargs):
                outer.create_kwargs = kwargs
                return types.SimpleNamespace(id="batch_abc", status="validating")

        self.files = _Files()
        self.batches = _Batches()

    def bodies(self):
        return {json.loads(line)["custom_id"]: json.loads(line)["body"]
                for line in self.jsonl.strip().split("\n")}

    def envelopes(self):
        return [json.loads(line) for line in self.jsonl.strip().split("\n")]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


class _SequentialCapture:
    """Captures every body handed to responses.create and answers with shard JSON."""

    def __init__(self, fail=()):
        self.sent = {}
        self.fail = set(fail)
        outer = self

        class _Responses:
            def create(self, **body):
                # identify the shard from the response-format name
                name = body["text"]["format"]["name"]
                shard_id = name.replace("stage3_", "")
                outer.sent[shard_id] = outer.sent.get(shard_id, 0) + 1
                outer.last = body
                outer.order.append((body["input"][0]["content"][0]["file_id"], shard_id)
                                   if body["input"][0]["content"] else None)
                if shard_id in outer.fail:
                    raise RuntimeError("simulated failure")
                text = json.dumps(SHARD_PAYLOAD[shard_id])
                return _FakeResponse({
                    "output": [{"type": "message",
                                "content": [{"type": "output_text", "text": text}]}],
                    "usage": _usage(),
                })

        self.order = []
        self.responses = _Responses()


class _CaptureAllBodies:
    """Records the body of every sequential call, keyed by (file_id, shard)."""

    def __init__(self):
        self.bodies = {}
        outer = self

        class _Responses:
            def create(self, **body):
                name = body["text"]["format"]["name"]
                shard_id = name.replace("stage3_", "")
                first_image = body["input"][0]["content"][0]["file_id"]
                line_id = "p1" if "p1" in first_image else "p2"
                outer.bodies[f"{line_id}__{shard_id}"] = body
                text = json.dumps(SHARD_PAYLOAD[shard_id])
                return _FakeResponse({
                    "output": [{"type": "message",
                                "content": [{"type": "output_text", "text": text}]}],
                    "usage": _usage(),
                })

        self.responses = _Responses()


def _batch_result_client(payloads, errors=()):
    """Fake client whose files.content() returns a batch output JSONL."""
    lines = []
    for cid, obj in payloads.items():
        lines.append(json.dumps({
            "custom_id": cid,
            "response": {"status_code": 200, "body": {
                "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": json.dumps(obj)}]}],
                "usage": _usage(),
            }},
        }))
    err_lines = [json.dumps({
        "custom_id": cid,
        "error": {"code": "server_error", "message": "transient upstream failure"},
    }) for cid in errors]

    class _Files:
        def content(self, file_id):
            text = "\n".join(lines if file_id == "out" else err_lines)
            return types.SimpleNamespace(text=text)

    client = types.SimpleNamespace(files=_Files())
    batch = types.SimpleNamespace(output_file_id="out",
                                  error_file_id="err" if err_lines else None,
                                  status="completed")
    return client, batch


# --- acceptance 12: identical bodies -----------------------------------------

def test_batch_and_sequential_send_identical_bodies(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config()
    requests = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, config
    )

    batch_client = _BatchCapture()
    ob._dispatch_openai_batch(
        _args(tmp_path), config, batch_client, config.openai_model, requests,
        list(LINE_IMAGE_BLOCKS), SHARDS, str(tmp_path), None, MASTER_INDEX,
    )

    seq_client = _CaptureAllBodies()
    ob._dispatch_openai_sequential(
        _args(tmp_path, dispatch="sequential", output=str(tmp_path / "seq")),
        config, seq_client, config.openai_model, requests,
        list(LINE_IMAGE_BLOCKS), MASTER_INDEX,
    )

    batch_bodies = batch_client.bodies()
    assert set(batch_bodies) == set(seq_client.bodies) == {
        "p1__shard_01", "p1__shard_02", "p2__shard_01", "p2__shard_02"}
    for cid in batch_bodies:
        assert batch_bodies[cid] == seq_client.bodies[cid], cid


def test_batch_envelope_adds_only_custom_id_method_url(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config()
    requests = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, config
    )
    client = _BatchCapture()
    ob._dispatch_openai_batch(
        _args(tmp_path), config, client, config.openai_model, requests,
        list(LINE_IMAGE_BLOCKS), SHARDS, str(tmp_path), None, MASTER_INDEX,
    )

    for env in client.envelopes():
        assert set(env) == {"custom_id", "method", "url", "body"}
        assert env["method"] == "POST"
        assert env["url"] == "/v1/responses"


# --- acceptance 13: one response-format name per shard ------------------------

def test_format_name_is_per_shard_not_generic():
    requests = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config()
    )
    names = {r["custom_id"]: r["body"]["text"]["format"]["name"] for r in requests}
    assert names["p1__shard_01"] == "stage3_shard_01"
    assert names["p1__shard_02"] == "stage3_shard_02"
    assert "structured_output" not in set(names.values())


def test_enum_leaf_gets_a_type_and_strict_is_on():
    requests = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config()
    )
    fmt = requests[0]["body"]["text"]["format"]
    assert fmt["strict"] is True
    value = (fmt["schema"]["properties"]["g1"]["properties"]["t1"]
             ["properties"]["value"])
    assert value["type"] == "string"
    assert value["enum"] == ["A", "not_assessable"]


# --- request ordering --------------------------------------------------------

def test_requests_are_plant_major_shard_minor():
    """One plant's shards stay contiguous so prompt caching can hit the images."""
    requests = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config()
    )
    assert [r["custom_id"] for r in requests] == [
        "p1__shard_01", "p1__shard_02", "p2__shard_01", "p2__shard_02"]


def test_reasoning_off_sends_none_and_keeps_temperature():
    requests = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config()
    )
    body = requests[0]["body"]
    assert body["reasoning"] == {"effort": "none"}
    assert body["temperature"] == 0.5


def test_reasoning_level_drops_temperature():
    config = _config()
    config.stage3_effort = "high"
    body = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, config
    )[0]["body"]
    assert body["reasoning"] == {"effort": "high"}
    assert "temperature" not in body


# --- acceptance 11: the batch input size guard -------------------------------

def test_oversized_jsonl_is_refused_before_upload(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ob, "_BATCH_INPUT_GUARD", 100)  # bytes

    class _NoUpload:
        class files:
            @staticmethod
            def create(**kwargs):
                raise AssertionError("must not upload an oversized JSONL")

        class batches:
            @staticmethod
            def create(**kwargs):
                raise AssertionError("must not create the batch")

    requests = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config()
    )
    result = ob._submit_batch_jsonl(_NoUpload(), _config(), requests, "phenotype_sharded")

    assert result is None
    out = capsys.readouterr().out
    assert "NOT uploaded" in out
    assert "200 MB" in out
    assert 'purpose="batch"' in out       # names which limit this is
    assert 'purpose="vision"' in out      # and which it is not
    assert "--no-files-api" in out        # and the way out


def test_base64_shard_size_warning_scales_with_shard_count(tmp_path, capsys):
    line = tmp_path / "p1"
    line.mkdir()
    (line / "a.jpg").write_bytes(b"x" * 3000)
    (line / "b.png").write_bytes(b"x" * 3000)
    (line / "notes.txt").write_bytes(b"x" * 100000)   # not an image

    estimate = ob._warn_base64_shard_size([line], 8)

    assert estimate == int(6000 * 4 / 3) * 8
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "8 shard requests" in out


# --- acceptance 8: provenance guard on the OpenAI paths ----------------------

def test_sequential_refuses_an_anthropic_partial_store(tmp_path):
    out = tmp_path / "out"
    (out / "_partial").mkdir(parents=True)
    write_run_meta_if_absent(out / "_partial", "anthropic", "claude-sonnet-5")
    before = {p.name: p.read_bytes() for p in (out / "_partial").iterdir()}

    requests = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config()
    )

    class _NoCall:
        class responses:
            @staticmethod
            def create(**body):
                raise AssertionError("must not call the API after refusing")

    with pytest.raises(RuntimeError) as exc:
        ob._dispatch_openai_sequential(
            _args(tmp_path, output=str(out)), _config(), _NoCall(),
            "gpt-5.6-luna", requests, list(LINE_IMAGE_BLOCKS), MASTER_INDEX,
        )

    assert "claude-sonnet-5" in str(exc.value)
    assert "gpt-5.6-luna" in str(exc.value)
    assert {p.name: p.read_bytes() for p in (out / "_partial").iterdir()} == before
    assert not list(out.glob("*.json"))


def test_fetch_refuses_an_anthropic_partial_store(tmp_path):
    out = tmp_path / "out"
    (out / "_partial").mkdir(parents=True)
    write_run_meta_if_absent(out / "_partial", "anthropic", "claude-sonnet-5")

    client, batch = _batch_result_client({"p1__shard_01": SHARD_PAYLOAD["shard_01"]})
    with pytest.raises(RuntimeError):
        write_openai_phenotype_sharded_results(
            client, batch, ["p1"], MASTER_INDEX, str(out), "openai", "gpt-5.6-luna",
        )
    assert not list(out.glob("*.json"))


# --- acceptance 7: cross-dispatch recovery -----------------------------------

def test_batch_gaps_are_filled_by_a_sequential_resume(tmp_path, capsys):
    """The end-to-end reason the two dispatches share a _partial/ store."""
    out = tmp_path / "out"

    # 1. a batch where shard_02 errored for p1
    client, batch = _batch_result_client(
        {"p1__shard_01": SHARD_PAYLOAD["shard_01"],
         "p2__shard_01": SHARD_PAYLOAD["shard_01"],
         "p2__shard_02": SHARD_PAYLOAD["shard_02"]},
        errors=["p1__shard_02"],
    )
    write_openai_phenotype_sharded_results(
        client, batch, ["p1", "p2"], MASTER_INDEX, str(out), "openai", "gpt-5.6-luna",
    )

    gaps = json.loads((out / "p1.gaps.json").read_text())
    assert {"group": "g2", "trait": "t2"} in gaps["missing_traits"]
    assert any("shard_02" in e for e in gaps["shard_errors"])
    assert not (out / "p2.gaps.json").exists()

    # 2. a sequential resume: only the missing shard is re-sent
    requests = build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config()
    )
    seq = _SequentialCapture()
    ob._dispatch_openai_sequential(
        _args(tmp_path, dispatch="sequential", output=str(out)),
        _config(), seq, "gpt-5.6-luna", requests, ["p1", "p2"], MASTER_INDEX,
    )

    assert seq.sent == {"shard_02": 1}, "only the gap should be re-billed"
    assert not (out / "p1.gaps.json").exists()
    record = json.loads((out / "p1.json").read_text())
    assert record["g1"]["t1"]["value"] == "A"   # adopted from the batch
    assert record["g2"]["t2"]["value"] == "B"   # recovered sequentially


def test_no_resume_re_sends_everything(tmp_path):
    out = tmp_path / "out"
    client, batch = _batch_result_client(
        {"p1__shard_01": SHARD_PAYLOAD["shard_01"],
         "p1__shard_02": SHARD_PAYLOAD["shard_02"]})
    write_openai_phenotype_sharded_results(
        client, batch, ["p1"], MASTER_INDEX, str(out), "openai", "gpt-5.6-luna")

    requests = [r for r in build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config())
        if r["custom_id"].startswith("p1__")]
    seq = _SequentialCapture()
    ob._dispatch_openai_sequential(
        _args(tmp_path, dispatch="sequential", output=str(out), resume=False),
        _config(), seq, "gpt-5.6-luna", requests, ["p1"], MASTER_INDEX,
    )
    assert seq.sent == {"shard_01": 1, "shard_02": 1}


def test_sequential_stamps_the_store_and_skips_it_when_resuming(tmp_path):
    out = tmp_path / "out"
    requests = [r for r in build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config())
        if r["custom_id"].startswith("p1__")]

    ob._dispatch_openai_sequential(
        _args(tmp_path, dispatch="sequential", output=str(out)),
        _config(), _SequentialCapture(), "gpt-5.6-luna", requests, ["p1"],
        MASTER_INDEX,
    )

    meta = json.loads((out / "_partial" / _RUN_META_NAME).read_text())
    assert (meta["provider"], meta["model"]) == ("openai", "gpt-5.6-luna")
    # the stamp must not have been mistaken for a shard partial
    record = json.loads((out / "p1.json").read_text())
    assert set(record) == {"g1", "g2"}
    assert not (out / ".json").exists()


def test_failed_shard_writes_no_partial_so_resume_retries_it(tmp_path):
    out = tmp_path / "out"
    requests = [r for r in build_openai_sharded_requests(
        LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT, _config())
        if r["custom_id"].startswith("p1__")]

    ob._dispatch_openai_sequential(
        _args(tmp_path, dispatch="sequential", output=str(out)),
        _config(), _SequentialCapture(fail=["shard_02"]), "gpt-5.6-luna",
        requests, ["p1"], MASTER_INDEX,
    )

    assert (out / "_partial" / "p1__shard_01.json").exists()
    assert not (out / "_partial" / "p1__shard_02.json").exists()
    assert json.loads((out / "p1.gaps.json").read_text())["missing_traits"]

    # a resume re-sends only the failed one, and it now succeeds
    seq = _SequentialCapture()
    ob._dispatch_openai_sequential(
        _args(tmp_path, dispatch="sequential", output=str(out)),
        _config(), seq, "gpt-5.6-luna", requests, ["p1"], MASTER_INDEX,
    )
    assert seq.sent == {"shard_02": 1}
    assert not (out / "p1.gaps.json").exists()


# --- retry policy ------------------------------------------------------------

class _Status(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.status_code = code


def test_400_is_not_retried(monkeypatch):
    monkeypatch.setattr(ob.time, "sleep", lambda s: None)
    calls = []

    class _C:
        class responses:
            @staticmethod
            def create(**body):
                calls.append(1)
                raise _Status(400)

    with pytest.raises(_Status):
        ob._call_with_retry(_C(), {}, 1, 1, "p1__shard_01")
    assert len(calls) == 1, "a 400 will never succeed on retry"


def test_429_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(ob.time, "sleep", lambda s: None)
    calls = []

    class _C:
        class responses:
            @staticmethod
            def create(**body):
                calls.append(1)
                if len(calls) < 2:
                    raise _Status(429)
                return "ok"

    assert ob._call_with_retry(_C(), {}, 1, 1, "p1__shard_01") == "ok"
    assert len(calls) == 2


# --- checkpoint --------------------------------------------------------------

def test_sharded_checkpoint_matches_the_anthropic_field_names(tmp_path, monkeypatch):
    """fetch-results' _sharded_master_index() reads these names for both providers."""
    monkeypatch.chdir(tmp_path)
    master = tmp_path / "master.json"
    master.write_text("{}", encoding="utf-8")

    ob._dispatch_openai_batch(
        _args(tmp_path), _config(), _BatchCapture(), "gpt-5.6-luna",
        build_openai_sharded_requests(LINE_IMAGE_BLOCKS, SHARDS, SYSTEM_PROMPT,
                                      _config()),
        list(LINE_IMAGE_BLOCKS), SHARDS, str(tmp_path), str(master), MASTER_INDEX,
    )

    ck = json.loads(Path("checkpoint_batch_abc.json").read_text())
    assert ck["stage"] == "phenotype_sharded"
    assert ck["provider"] == "openai"
    assert ck["model"] == "gpt-5.6-luna"
    assert ck["shard_dir"] == str(tmp_path.resolve())
    assert ck["master_schema"] == str(master.resolve())
    assert ck["shard_ids"] == ["shard_01", "shard_02"]
    assert ck["line_ids"] == ["p1", "p2"]
