"""``pxgpt schema --shard-dir``: one plant through a whole shard set.

This is the rehearsal that runs before 267 plants and 5.6 h of GPU time, so its
job is to surface the state of EVERY shard in one pass rather than stop at the
first bad one.  These cover what it writes, what it refuses to write, and what
it skips on a second run.
"""

import json
from pathlib import Path

import pytest

from pxgpt.commands import schema as schema_cmd
from pxgpt.core.batch_utils import _RUN_META_NAME
from pxgpt.commands.schema import setup_schema_parser
from pxgpt.providers.base import APIResponse, TokenUsage
from pxgpt.providers.openai_compat_provider import (
    OutputLengthError,
    ThinkingLeakError,
)


def create_parser():
    """Just the `schema` subcommand, wired the same way main() wires it."""
    import argparse
    parser = argparse.ArgumentParser(prog="pxgpt")
    setup_schema_parser(parser.add_subparsers(dest="command", required=True))
    return parser


TRAITS = [
    {"group": "leaf", "trait": "leaf_shape", "scale_type": "nominal", "unit": None},
    {"group": "stem", "trait": "stem_colour", "scale_type": "nominal", "unit": None},
]


def _shard_schema(shard_id, trait):
    return {
        "title": f"stage3_{shard_id}",
        "type": "object",
        "additionalProperties": False,
        "required": ["trait_groups"],
        "properties": {"trait_groups": {"type": "object"}},
        "x_trait": trait,
    }


@pytest.fixture
def shard_dir(tmp_path):
    """A two-shard set shaped like the frozen ones (manifest + schema + prompt)."""
    d = tmp_path / "shard_master_schema"
    d.mkdir()
    (d / "system.txt").write_text("you are a botanist\n", encoding="utf-8")
    shards = []
    for i, t in enumerate(TRAITS, 1):
        sid = f"shard_{i:02d}"
        (d / f"{sid}.schema.json").write_text(
            json.dumps(_shard_schema(sid, t["trait"])), encoding="utf-8")
        (d / f"{sid}.prompt.txt").write_text(f"score {t['trait']}", encoding="utf-8")
        shards.append({
            "shard_id": sid,
            "schema_file": f"{sid}.schema.json",
            "prompt_file": f"{sid}.prompt.txt",
            "groups": [t["group"]],
            "traits": [t["trait"]],
        })
    (d / "shards_manifest.json").write_text(json.dumps({
        "system_file": "system.txt",
        "shards": shards,
        "all_traits": TRAITS,
    }), encoding="utf-8")
    return d


@pytest.fixture
def plant(tmp_path):
    folder = tmp_path / "images" / "s0019"
    folder.mkdir(parents=True)
    for name in ("b.jpg", "a.jpg"):
        (folder / name).write_bytes(b"\xff\xd8\xff")
    return folder


def _answer(shard_id, trait):
    """The shape merge_plant_record consumes: {group: {trait: {value, rationale}}}."""
    group = TRAITS[int(shard_id[-2:]) - 1]["group"]
    return {group: {trait: {"value": "ovate", "rationale": "looks ovate"}}}


class FakeProvider:
    """Stands in for OpenAICompatProvider; records every call it receives."""

    provider_name = "openai-compat-vllm"

    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = outcomes or {}

    def send_request_with_retry(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"][-1]["text"]
        shard_id = "shard_01" if "leaf_shape" in prompt else "shard_02"
        outcome = self.outcomes.get(shard_id)
        if isinstance(outcome, Exception):
            raise outcome
        content = outcome if outcome is not None else json.dumps(
            _answer(shard_id, TRAITS[int(shard_id[-2:]) - 1]["trait"]))
        return APIResponse(
            content=content,
            usage=TokenUsage(input_tokens=1000, output_tokens=42,
                             cache_read_tokens=0 if shard_id == "shard_01" else 990),
            request_id="req",
            model="gemma4-26b",
            finish_reason="stop",
        )


def _args(shard_dir, plant, out, **over):
    parser = create_parser()
    argv = ["schema", "--provider", "vllm",
            "--shard-dir", str(shard_dir),
            "--input-folder", str(plant),
            "--output", str(out),
            "--image-transport", "file"]
    for k, v in over.pop("extra_argv", {}).items():
        argv += [k] if v is None else [k, str(v)]
    args = parser.parse_args(argv)
    for k, v in over.items():
        setattr(args, k, v)
    return args


@pytest.fixture
def run(monkeypatch, shard_dir, plant, tmp_path):
    """Run the command against a FakeProvider; return (exit_code, provider, out)."""
    monkeypatch.setenv("VLLM_MODEL", "gemma4-26b")

    def _run(outcomes=None, out=None, **over):
        out = out or (tmp_path / "smoke")
        provider = FakeProvider(outcomes)

        def _factory(name, config):
            provider.config = config      # what the command actually configured
            return provider

        monkeypatch.setattr(schema_cmd, "create_provider", _factory)
        code = schema_cmd.schema_command(_args(shard_dir, plant, out, **over))
        return code, provider, Path(out)

    return _run


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------

def test_writes_one_partial_per_shard_and_a_merged_record(run):
    code, provider, out = run()

    assert code == 0
    partials = sorted(p.name for p in (out / "_partial").glob("*.json")
                      if p.name != _RUN_META_NAME)
    assert partials == ["s0019__shard_01.json", "s0019__shard_02.json"]
    record = json.loads((out / "s0019.json").read_text())
    assert record["leaf"]["leaf_shape"]["value"] == "ovate"
    assert record["stem"]["stem_colour"]["value"] == "ovate"
    assert not (out / "s0019.gaps.json").exists()


def test_shards_run_in_manifest_order_and_never_in_parallel(run):
    _, provider, _ = run()

    prompts = [c["messages"][0]["content"][-1]["text"] for c in provider.calls]
    assert prompts == ["score leaf_shape", "score stem_colour"]


def test_each_shard_gets_its_own_raw_schema_and_the_shared_system_prompt(run):
    _, provider, _ = run()

    assert [c["json_schema"]["title"] for c in provider.calls] == [
        "stage3_shard_01", "stage3_shard_02",
    ]
    # Raw, not normalized: the extra key survives, proving nothing rewrote it.
    assert provider.calls[0]["json_schema"]["x_trait"] == "leaf_shape"
    assert all(c["system_prompt"] == "you are a botanist\n" for c in provider.calls)


def test_images_precede_the_shard_prompt_and_use_the_chosen_transport(run):
    _, provider, _ = run()

    content = provider.calls[0]["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"]
    assert [Path(c["source"]["url"]).name for c in content[:2]] == ["a.jpg", "b.jpg"]


# --------------------------------------------------------------------------
# resume
# --------------------------------------------------------------------------

def test_second_run_skips_every_shard_and_issues_no_api_call(run, tmp_path):
    out = tmp_path / "smoke"
    run(out=out)

    code, provider, _ = run(out=out)

    assert code == 0
    assert provider.calls == []


def test_no_resume_reruns_every_shard(run, tmp_path):
    out = tmp_path / "smoke"
    run(out=out)

    code, provider, _ = run(out=out, resume=False)

    assert code == 0
    assert len(provider.calls) == 2


def test_a_corrupt_partial_is_rerun_rather_than_adopted(run, tmp_path):
    out = tmp_path / "smoke"
    run(out=out)
    (out / "_partial" / "s0019__shard_01.json").write_text("{ truncated")

    _, provider, _ = run(out=out)

    assert len(provider.calls) == 1  # only the corrupt one


# --------------------------------------------------------------------------
# failures: nothing is written, and the run keeps going
# --------------------------------------------------------------------------

@pytest.mark.parametrize("failure", [
    OutputLengthError("stopped at finish_reason='length'"),
    ThinkingLeakError("backend returned non-empty 'reasoning'"),
    RuntimeError("connection reset"),
    "not json at all",
])
def test_a_failed_shard_writes_no_partial_but_the_run_continues(run, failure):
    code, provider, out = run(outcomes={"shard_01": failure})

    assert code == 1                              # the run reports the failure
    assert len(provider.calls) == 2               # ...after trying shard_02 too
    assert not (out / "_partial" / "s0019__shard_01.json").exists()
    assert (out / "_partial" / "s0019__shard_02.json").exists()


def test_a_leaked_reasoning_response_is_never_stored(run, tmp_path, capsys):
    leak = ThinkingLeakError(
        "backend returned non-empty 'reasoning' ...; first 200 chars: 'hmm'")
    code, _, out = run(outcomes={"shard_01": leak, "shard_02": leak})

    assert code == 1
    stored = [p.name for p in (out / "_partial").glob("*.json")
              if p.name != _RUN_META_NAME]
    assert stored == []
    out_text = capsys.readouterr().out
    assert "reasoning leak" in out_text
    assert "'reasoning'" in out_text               # the field name is reported


def test_a_gaps_file_records_the_traits_the_failed_shard_owned(run):
    _, _, out = run(outcomes={"shard_01": RuntimeError("boom")})

    gaps = json.loads((out / "s0019.gaps.json").read_text())
    assert {"group": "leaf", "trait": "leaf_shape"} in gaps["missing_traits"]


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def test_reusing_another_providers_output_dir_fails_before_writing(run, tmp_path):
    out = tmp_path / "smoke"
    (out / "_partial").mkdir(parents=True)
    (out / "_partial" / _RUN_META_NAME).write_text(json.dumps({
        "provider": "anthropic", "model": "claude-sonnet-5",
    }), encoding="utf-8")

    code, provider, _ = run(out=out)

    assert code == 1
    assert provider.calls == []
    assert not list((out / "_partial").glob("s0019*.json"))
    assert not (out / "s0019.json").exists()


def test_output_pointing_at_an_existing_file_is_rejected(run, tmp_path):
    out = tmp_path / "already-a-file.json"
    out.write_text("{}", encoding="utf-8")

    code, provider, _ = run(out=out)

    assert code == 1
    assert provider.calls == []
    assert out.read_text() == "{}"          # untouched


def test_max_tokens_defaults_to_the_shard_cap(run):
    _, provider, _ = run()

    # 2048: ~3.4x the observed p90 shard answer, so it cannot truncate a real
    # one, but it caps the runaway rationale case at ~50 s instead of ~190 s.
    assert provider.config.max_tokens == 2048


def test_explicit_max_tokens_overrides_the_shard_cap(run):
    _, provider, _ = run(max_tokens=32)

    assert provider.config.max_tokens == 32


# --------------------------------------------------------------------------
# CLI shape
# --------------------------------------------------------------------------

def test_schema_and_shard_dir_are_mutually_exclusive():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["schema", "--input-folder", "f", "--output", "o",
                           "--schema", "s.json", "--shard-dir", "d"])


def test_one_of_schema_or_shard_dir_is_required():
    parser = create_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["schema", "--input-folder", "f", "--output", "o"])


def test_resume_defaults_on_and_no_resume_turns_it_off():
    parser = create_parser()
    base = ["schema", "--input-folder", "f", "--output", "o", "--shard-dir", "d"]

    assert parser.parse_args(base).resume is True
    assert parser.parse_args(base + ["--no-resume"]).resume is False


def test_image_transport_defaults_to_base64():
    parser = create_parser()

    args = parser.parse_args(["schema", "--input-folder", "f", "--output", "o",
                              "--schema", "s.json"])
    assert args.image_transport == "base64"
