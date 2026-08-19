"""``OpenAICompatProvider``: constrained decoding, sampling, and the two aborts.

The failures these cover are the quiet kind.  A backend that ignores
``response_format`` still returns fluent JSON-ish text; a chat template that
leaves thinking on still returns a usable answer; a response cut off at
``max_tokens`` still returns a string.  Each one would produce a run that reads
as healthy and is not, so each is turned into a hard error here.
"""

import json
from pathlib import Path

import pytest

from pxgpt.core.config import Config
from pxgpt.providers.openai_compat_provider import (
    OpenAICompatProvider,
    OutputLengthError,
    ThinkingLeakError,
)


SHARD_SCHEMA = {
    "title": "stage3_shard_02",
    "type": "object",
    "additionalProperties": False,
    "required": ["leaf_shape"],
    "properties": {"leaf_shape": {"enum": ["ovate", "linear", "NA"]}},
}


# --------------------------------------------------------------------------
# A fake OpenAI client: records the params it was called with, returns what the
# test asks it to.
# --------------------------------------------------------------------------

class _Message:
    def __init__(self, content, extra=None):
        self.content = content
        self.model_extra = extra or {}


class _Choice:
    def __init__(self, content, finish_reason, extra=None):
        self.message = _Message(content, extra)
        self.finish_reason = finish_reason


class _Details:
    def __init__(self, cached):
        self.cached_tokens = cached


class _Usage:
    def __init__(self, prompt, completion, cached):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _Details(cached)


class _Response:
    def __init__(self, content, finish_reason="stop", extra=None, cached=0):
        self.id = "req-1"
        self.choices = [_Choice(content, finish_reason, extra)]
        self.usage = _Usage(100, 20, cached)


class FakeClient:
    def __init__(self, response=None, error=None):
        self.calls = []
        self._response = response
        self._error = error

        outer = self

        class _Completions:
            def create(self, **params):
                outer.calls.append(params)
                if outer._error is not None:
                    raise outer._error
                return outer._response

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def _provider(provider="vllm", response=None, error=None, **cfg):
    cfg.setdefault("max_retries", 0)
    config = Config(vllm_model="gemma4-26b", **cfg)
    p = OpenAICompatProvider(config, provider)
    p._client = FakeClient(response or _Response('{"leaf_shape": "ovate"}'), error)
    return p


def _send(p, **kwargs):
    kwargs.setdefault("messages", [{"role": "user", "content": [
        {"type": "text", "text": "shard prompt"},
    ]}])
    kwargs.setdefault("system_prompt", "you are a botanist")
    return p.send_request_with_retry(**kwargs)


# --------------------------------------------------------------------------
# response_format
# --------------------------------------------------------------------------

def test_json_schema_is_sent_verbatim_as_strict_response_format():
    p = _provider()

    _send(p, json_schema=SHARD_SCHEMA)

    rf = p.client.calls[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "stage3_shard_02"   # from the title
    # Verbatim: NOT run through openai_normalize_schema, which would strip the
    # title and rewrite enums.  xgrammar takes standard JSON Schema as-is.
    assert rf["json_schema"]["schema"] == SHARD_SCHEMA


def test_schema_never_reaches_the_system_prompt_on_the_native_path():
    p = _provider()

    _send(p, json_schema=SHARD_SCHEMA)

    system = p.client.calls[0]["messages"][0]
    assert system["role"] == "system"
    assert system["content"] == "you are a botanist"
    assert "leaf_shape" not in system["content"]


def test_legacy_and_native_schema_paths_are_mutually_exclusive():
    p = _provider()

    with pytest.raises(ValueError, match="mutually exclusive"):
        _send(p, json_schema=SHARD_SCHEMA, schema=json.dumps(SHARD_SCHEMA))


def test_backend_rejecting_response_format_raises_and_never_falls_back():
    import openai

    err = openai.BadRequestError(
        "response_format not supported",
        response=type("R", (), {"status_code": 400, "headers": {}, "request": None})(),
        body=None,
    )
    p = _provider(error=err)

    with pytest.raises(openai.BadRequestError):
        _send(p, json_schema=SHARD_SCHEMA)

    # One attempt, and the schema was never re-sent as system-prompt prose.
    assert len(p.client.calls) == 1
    assert "leaf_shape" not in p.client.calls[0]["messages"][0]["content"]


def test_legacy_system_prompt_path_still_exists():
    p = _provider()

    _send(p, schema=json.dumps(SHARD_SCHEMA))

    assert "response_format" not in p.client.calls[0]
    assert "leaf_shape" in p.client.calls[0]["messages"][0]["content"]


# --------------------------------------------------------------------------
# sampling parameters
# --------------------------------------------------------------------------

def test_local_backend_sends_temperature_top_p_and_top_k():
    p = _provider(temperature=1.0, top_p=0.95, top_k=64)

    _send(p, json_schema=SHARD_SCHEMA)

    params = p.client.calls[0]
    assert params["temperature"] == 1.0
    assert params["top_p"] == 0.95
    assert params["extra_body"]["top_k"] == 64


def test_extra_body_never_carries_the_two_forbidden_keys():
    p = _provider()

    extra = p._build_extra_body()

    # mm_processor_kwargs moves the images into a different prefix-cache
    # namespace (silent ~57 s per request); seed collapses the run-to-run
    # variance the consistency study measures.  Neither errors, so neither is
    # catchable at runtime — it has to be a property of the builder.
    assert "mm_processor_kwargs" not in extra
    assert "seed" not in extra
    assert set(extra) == {"chat_template_kwargs", "top_k"}


def test_forbidden_keys_appear_nowhere_in_the_provider_source():
    import pxgpt.providers.openai_compat_provider as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    code = "\n".join(
        line for line in src.splitlines()
        if not line.lstrip().startswith(("#", "*"))
    )
    # Docstrings mention both by name; no executable line may.
    assert '"mm_processor_kwargs"' not in code
    assert '"seed"' not in code


def test_openai_proper_gets_no_local_only_fields():
    p = _provider(provider="openai", openai_api_key="sk-test",
                  openai_model="gpt-4o")

    _send(p, json_schema=SHARD_SCHEMA)

    params = p.client.calls[0]
    assert "extra_body" not in params    # top_k / chat_template_kwargs would 400
    assert "top_p" not in params


# --------------------------------------------------------------------------
# thinking
# --------------------------------------------------------------------------

def test_thinking_is_disabled_explicitly_on_every_request():
    p = _provider()

    _send(p, json_schema=SHARD_SCHEMA)

    # Spelled out rather than left to the chat template default: a default does
    # not appear in the request record, and it is a default somebody else owns.
    assert p.client.calls[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content"])
def test_leaked_reasoning_fails_the_call_and_names_the_field(field):
    leak = "Let me think step by step about the leaf margin" * 20
    p = _provider(response=_Response(
        '{"leaf_shape": "ovate"}', extra={field: leak}
    ))

    with pytest.raises(ThinkingLeakError) as exc:
        _send(p, json_schema=SHARD_SCHEMA)

    msg = str(exc.value)
    assert field in msg                       # which field name leaked
    assert leak[:200] in msg                  # the first 200 chars, for triage
    assert leak[:250] not in msg              # and no more than that


def test_empty_reasoning_field_is_not_a_leak():
    p = _provider(response=_Response(
        '{"leaf_shape": "ovate"}', extra={"reasoning": ""}
    ))

    assert _send(p, json_schema=SHARD_SCHEMA).content == '{"leaf_shape": "ovate"}'


# --------------------------------------------------------------------------
# length
# --------------------------------------------------------------------------

def test_finish_reason_length_is_an_error_not_a_partial_result():
    p = _provider(response=_Response('{"leaf_shape": "ov', finish_reason="length"))

    with pytest.raises(OutputLengthError) as exc:
        _send(p, json_schema=SHARD_SCHEMA)

    assert "length" in str(exc.value)


def test_length_error_is_not_retried():
    p = _provider(response=_Response("{", finish_reason="length"),
                  max_retries=3)

    with pytest.raises(OutputLengthError):
        _send(p, json_schema=SHARD_SCHEMA)

    assert len(p.client.calls) == 1


# --------------------------------------------------------------------------
# usage / transport plumbing
# --------------------------------------------------------------------------

def test_cached_prompt_tokens_are_surfaced():
    p = _provider(response=_Response('{"leaf_shape": "ovate"}', cached=36448))

    resp = _send(p, json_schema=SHARD_SCHEMA)

    assert resp.usage.cache_read_tokens == 36448
    assert resp.finish_reason == "stop"


def test_url_source_becomes_an_openai_image_url():
    p = _provider()

    _send(p, json_schema=SHARD_SCHEMA, messages=[{
        "role": "user",
        "content": [
            {"type": "image",
             "source": {"type": "url", "url": "file:///media/s0019/a.jpg"}},
            {"type": "text", "text": "shard prompt"},
        ],
    }])

    content = p.client.calls[0]["messages"][1]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {"url": "file:///media/s0019/a.jpg"},
    }
    assert content[1]["type"] == "text"     # images still precede text


def test_base64_source_conversion_is_unchanged():
    p = _provider()

    _send(p, json_schema=SHARD_SCHEMA, messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": "QUJD"}},
            {"type": "text", "text": "shard prompt"},
        ],
    }])

    content = p.client.calls[0]["messages"][1]["content"]
    assert content[0]["image_url"]["url"] == "data:image/png;base64,QUJD"


# --------------------------------------------------------------------------
# thinking, deliberately ON (analyze only — Stage 3 stays pinned off)
# --------------------------------------------------------------------------

def test_an_effort_level_turns_thinking_on_for_a_local_backend():
    p = _provider()

    _send(p, output_config={"effort": "high"})

    # Local models have no reasoning levels, only on/off, so any level means on.
    assert p.client.calls[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }


def test_reasoning_is_allowed_through_but_kept_out_of_the_content():
    thoughts = "Thinking Process: the leaves are ruffled, so..."
    p = _provider(response=_Response("A young kale plant.",
                                     extra={"reasoning": thoughts}))

    resp = _send(p, output_config={"effort": "high"})

    # The server's reasoning parser already split them.  content is the answer
    # alone, so a caller writing response.content never saves the reasoning.
    assert resp.content == "A young kale plant."
    assert thoughts not in resp.content


def test_reasoning_still_fails_the_call_when_thinking_was_not_asked_for():
    p = _provider(response=_Response("answer", extra={"reasoning": "leaked"}))

    with pytest.raises(ThinkingLeakError):
        _send(p, output_config={"effort": ""})


def test_truncation_is_still_an_error_with_thinking_on():
    p = _provider(response=_Response("half", finish_reason="length"))

    with pytest.raises(OutputLengthError):
        _send(p, output_config={"effort": "high"})
