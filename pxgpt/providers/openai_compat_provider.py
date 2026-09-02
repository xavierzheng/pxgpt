"""OpenAI-SDK provider for every OpenAI-wire-protocol backend."""

from typing import Dict, Any, List, Optional
import openai
from openai import OpenAI

from ..core.openai_batch_utils import schema_format_name
from .base import BaseProvider, APIResponse, TokenUsage


def _cached_prompt_tokens(usage) -> int:
    """Return ``prompt_tokens_details.cached_tokens``, or 0 when absent.

    This is the prefix-cache hit count.  It is the number that shows whether a
    plant's 9 shards really pay one cold prefill between them, so it is read off
    every response rather than inferred from wall-clock timing.
    """
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    value = getattr(details, "cached_tokens", None)
    if value is None and isinstance(details, dict):
        value = details.get("cached_tokens")
    return value or 0


class ThinkingLeakError(RuntimeError):
    """The backend returned reasoning text although thinking was disabled.

    Raised, never swallowed: stripping the field would turn a server/chat-template
    misconfiguration into an invisible data-cleaning step, and the run would look
    healthy while the model was spending its budget somewhere the paper cannot
    account for.
    """


class OutputLengthError(RuntimeError):
    """The response stopped at ``finish_reason == "length"``.

    A grammar constrains the *shape* of the output, not its length, so a runaway
    ``rationale`` string can hit the cap mid-object.  What comes back is not a
    partial result worth keeping -- it is unparseable -- so it is an error.
    """


class OpenAICompatProvider(BaseProvider):
    """OpenAI SDK provider for OpenAI, Ollama, LM Studio and vLLM.

    All four speak the OpenAI wire protocol, so one ``openai.OpenAI`` client
    with the right ``base_url`` serves them all.  The model name is sent
    verbatim -- no route prefixes.

    - openai    : ``config.openai_base_url``   (None -> the SDK default)
    - ollama     : ``config.ollama_base_url`` + ``/v1``
    - lmstudio  : ``config.lmstudio_base_url`` (already ends in ``/v1``)
    - vllm      : ``config.vllm_base_url``     (already ends in ``/v1``)

    .. deprecated::
       **``ollama`` and ``lmstudio`` are slated for removal in a future major
       release.  vLLM is the supported local backend.**  Both remain fully
       functional today; nothing here is gated or warned at runtime.

       The reason is visual tokenization, which is the whole ballgame for
       phenotyping.  vLLM exposes a per-image budget --
       ``--mm-processor-kwargs '{"max_soft_tokens": N}'``, ladder
       ``70 / 140 / 280 / 560 / 1120`` -- so the deployment in
       ``ops/local-vllm/`` pins 1120 to land close to Anthropic Sonnet 5's
       per-image tokenization and keep local and cloud runs comparable.
       Ollama and LM Studio expose no equivalent knob: whatever downsampling
       they apply is not settable, not reportable, and free to change under a
       backend or model update.  A trait like petiole cross-section or leaf
       margin lives or dies on that detail, and an uncontrolled backend cannot
       be held to a measurement.

       This has NOT been measured against either backend -- the objection is
       the missing control, not a benchmarked loss.  Measuring it is what would
       overturn the decision.
    """

    def __init__(self, config, provider: str):
        super().__init__(config)
        self.llm_provider = provider
        self.base_model = config.get_model(provider)
        self.model = self.base_model
        self.api_base = self._resolve_api_base(provider)
        self.api_key = self._resolve_api_key(provider)

        if provider in ("lmstudio", "vllm") and not self.base_model:
            raise ValueError(
                f"No model configured for '{provider}'. Set "
                f"{'VLLM_MODEL' if provider == 'vllm' else 'LMSTUDIO_MODEL'}."
            )

    def _resolve_api_base(self, provider: str):
        if provider == "openai":
            return self.config.openai_base_url  # may be None (real OpenAI)
        if provider == "ollama":
            # Ollama's OpenAI-compatible surface lives under /v1, and
            # OLLAMA_BASE_URL conventionally omits it. Append only when needed so
            # a user who already wrote /v1 doesn't get /v1/v1.
            base = (self.config.ollama_base_url or "").rstrip("/")
            return base if base.endswith("/v1") else f"{base}/v1"
        if provider == "lmstudio":
            return self.config.lmstudio_base_url
        if provider == "vllm":
            return self.config.vllm_base_url
        return None

    def _resolve_api_key(self, provider: str):
        # The SDK rejects None/"" outright, and Ollama has no key setting.
        if provider == "ollama":
            return "ollama"
        return self.config.get_api_key(provider)

    def _is_openai_reasoning_model(self) -> bool:
        m = self.base_model.lower()
        return "gpt-5" in m or m.startswith(("o1", "o3", "o4"))

    @property
    def provider_name(self) -> str:
        return f"openai-compat-{self.llm_provider}"

    def _create_client(self):
        """Build the OpenAI client; retries are BaseProvider's job, not the SDK's."""
        kwargs = {
            "api_key": self.api_key,
            "max_retries": 0,
            "timeout": self.config.timeout,
        }
        if self.api_base:
            kwargs["base_url"] = self.api_base
        return OpenAI(**kwargs)

    def supports_caching(self) -> bool:
        """OpenAI-compatible providers don't support prompt caching"""
        return False

    def _build_system_prompt(self, system_prompt: str, schema: Optional[str] = None) -> str:
        """Build combined system prompt since caching isn't supported"""
        if schema:
            return f"{system_prompt}\n\nUse this JSON schema for your response:\n{schema}"
        return system_prompt

    def _to_openai_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic-style messages to OpenAI format"""
        converted_messages = []

        for message in messages:
            if message["role"] == "user":
                content = []
                for item in message["content"]:
                    if item["type"] == "text":
                        content.append({
                            "type": "text",
                            "text": item["text"]
                        })
                    elif item["type"] == "image":
                        source = item["source"]
                        if source["type"] == "url":
                            # file:// (or http) URI — the server fetches the
                            # bytes itself; nothing is embedded in the request.
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": source["url"]},
                            })
                        else:
                            # Convert Anthropic base64 format to OpenAI format
                            image_data = source["data"]
                            media_type = source["media_type"]
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{image_data}"
                                }
                            })

                converted_messages.append({
                    "role": "user",
                    "content": content
                })

        return converted_messages

    def _build_extra_body(self, enable_thinking: bool = False) -> Dict[str, Any]:
        """Return the non-OpenAI request fields for the local backends.

        The ONLY place ``extra_body`` is assembled, and it can therefore be read
        as a guarantee about what is never sent:

        * **no** ``mm_processor_kwargs`` — the vision token budget is pinned
          server-side.  Passing it per request puts the images in a different
          prefix-cache namespace: identical tokenization, but the cached entry is
          not reused.  Measured on one 26-photo line: 14.8 s when the key
          matched, 71.6 s when it did not.  No error, just a silent ~57 s penalty
          on every request.
        * **no** ``seed`` — the consistency study measures run-to-run variation on
          the same plant.  A fixed seed returns byte-identical output, collapses
          the variance to zero, and reports no error while doing it.

        ``top_k`` rides here because the OpenAI wire protocol has no field for it,
        and ``enable_thinking`` is spelled out either way rather than left to the
        chat template's default: a default does not appear in the request record,
        and it is a default somebody else can change.
        """
        return {
            "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
            "top_k": self.config.top_k,
        }

    def _assert_no_reasoning(self, message) -> None:
        """Fail the call if the backend returned reasoning text.

        vLLM 0.24 puts it in ``message.reasoning``; other builds use
        ``reasoning_content``.  Neither is modelled by the OpenAI SDK, so both
        land in ``model_extra``.  Both names are checked.
        """
        extra = getattr(message, "model_extra", None) or {}
        for field in ("reasoning", "reasoning_content"):
            value = getattr(message, field, None)
            if value is None:
                value = extra.get(field)
            if value:
                head = str(value)[:200]
                raise ThinkingLeakError(
                    f"backend returned non-empty {field!r} although "
                    f"chat_template_kwargs.enable_thinking=False was sent; "
                    f"first 200 chars: {head!r}"
                )

    def _send_request(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        schema: Optional[str] = None,
        output_config: Optional[Dict[str, Any]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:
        """Send request via the OpenAI SDK.

        Two mutually exclusive schema paths:

        * *json_schema* — a schema dict sent as native structured output
          (``response_format``), i.e. real constrained decoding.  The schema is
          sent verbatim; it is NOT run through ``openai_normalize_schema``,
          because vLLM's xgrammar backend takes standard JSON Schema and the
          frozen shard schemas already carry ``additionalProperties: false`` and
          full ``required`` lists.
        * *schema* — the legacy path: schema text appended to the system prompt,
          with no decoding constraint at all.

        They never combine.  When *json_schema* is given the schema is kept out
        of the system prompt entirely, so the prompt stays byte-identical to what
        the other providers see and the run remains cross-provider comparable.
        If the backend rejects ``response_format`` the error is raised: there is
        deliberately no fallback to the legacy path, because a silent downgrade
        produces output that looks fine and is in fact completely unconstrained.

        Only ``output_config["effort"]`` is honoured here, and only for OpenAI
        reasoning models, where it becomes ``reasoning_effort``.  The Anthropic
        ``format`` key has no equivalent on this wire protocol and is ignored;
        the local backends ignore effort as well.
        """
        if json_schema is not None and schema:
            raise ValueError(
                "json_schema (native structured output) and schema (legacy "
                "system-prompt text) are mutually exclusive; the schema must "
                "appear in exactly one place."
            )

        # Build combined system prompt (no caching support)
        combined_system = self._build_system_prompt(system_prompt, schema)

        # Convert messages to OpenAI format
        converted_messages = self._to_openai_messages(messages)

        # Add system message at the beginning
        full_messages = [{"role": "system", "content": combined_system}] + converted_messages

        # Prepare request parameters
        params = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": self.config.max_tokens,
        }

        if json_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_format_name(json_schema),
                    "schema": json_schema,
                    "strict": True,
                },
            }

        # Gemma-class local models have no reasoning *levels*, only on/off, so
        # any effort at all means on.  Stage 3 never sets one; `analyze` may.
        thinking_on = bool((output_config or {}).get("effort")) \
            and self.llm_provider != "openai"

        if self.llm_provider != "openai":
            # Sampling knobs the local servers understand.  temperature/top_p go
            # in the body below; top_k and the thinking switch have no OpenAI
            # field, so they ride in extra_body.  The server's
            # --override-generation-config already sets the same values -- that
            # is deliberate redundancy, because only the client-side copy shows
            # up in the request record the paper has to cite.
            params["top_p"] = self.config.top_p
            params["extra_body"] = self._build_extra_body(thinking_on)
            print(f"## Thinking: {'on' if thinking_on else 'off'} "
                  f"(chat_template_kwargs.enable_thinking)")

        if self.llm_provider == "openai" and self._is_openai_reasoning_model():
            # These models reject `max_tokens` outright, and omitting
            # `reasoning_effort` leaves reasoning ON at the model's own default
            # (medium on gpt-5.6) — so both have to be spelled out.  A custom
            # temperature is accepted only with reasoning fully off.
            params.pop("max_tokens")
            params["max_completion_tokens"] = self.config.max_tokens
            effort = (output_config or {}).get("effort") or "none"
            params["reasoning_effort"] = effort
            if effort == "none":
                params["temperature"] = self.config.temperature
                print(f"## Reasoning effort: none (off); "
                      f"temperature {self.config.temperature}")
            else:
                print(f"## Reasoning effort: {effort}; temperature omitted "
                      f"(only effort 'none' accepts a custom value)")
        else:
            params["temperature"] = self.config.temperature

        # Send request
        response = self.client.chat.completions.create(**params)

        choice = response.choices[0]

        # Extract usage information
        usage = TokenUsage(
            input_tokens=getattr(response.usage, 'prompt_tokens', 0),
            output_tokens=getattr(response.usage, 'completion_tokens', 0),
            cache_read_tokens=_cached_prompt_tokens(response.usage),
        )

        # Print usage stats
        request_id = getattr(response, 'id', 'N/A')
        print(f"## Request ID: {request_id}")
        print(f"## Provider: {self.llm_provider}")
        print(f"## Model: {self.model}")
        print(f"## Input tokens: {usage.input_tokens}")
        print(f"## Output tokens: {usage.output_tokens}")

        finish_reason = getattr(choice, "finish_reason", None)

        # A truncated response is an error, not a partial result: the grammar
        # constrains shape, not length, so what came back cannot be parsed.
        if finish_reason == "length":
            raise OutputLengthError(
                f"response stopped at finish_reason='length' after "
                f"{usage.output_tokens} completion token(s) (max_tokens="
                f"{self.config.max_tokens}); treat this call as failed"
            )

        if not thinking_on:
            # Only an assertion while thinking is meant to be off.  With it on,
            # reasoning is expected: the server's reasoning parser keeps it in
            # its own field, `content` holds the final answer alone, and the
            # caller simply never writes the reasoning anywhere.
            self._assert_no_reasoning(choice.message)

        return APIResponse(
            content=choice.message.content,
            usage=usage,
            request_id=request_id,
            model=self.model,
            finish_reason=finish_reason,
        )

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error"""
        return isinstance(error, openai.RateLimitError)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable"""
        if isinstance(error, (openai.APIConnectionError, openai.APITimeoutError)):
            return True
        if isinstance(error, openai.APIStatusError):
            return error.status_code in (500, 502, 503, 504)
        return False
