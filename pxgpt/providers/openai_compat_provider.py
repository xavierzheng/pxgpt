"""OpenAI-SDK provider for every OpenAI-wire-protocol backend."""

from typing import Dict, Any, List, Optional
import openai
from openai import OpenAI

from .base import BaseProvider, APIResponse, TokenUsage


class OpenAICompatProvider(BaseProvider):
    """OpenAI SDK provider for OpenAI, Ollama, LM Studio and vLLM.

    All four speak the OpenAI wire protocol, so one ``openai.OpenAI`` client
    with the right ``base_url`` serves them all.  The model name is sent
    verbatim -- no route prefixes.

    - openai    : ``config.openai_base_url``   (None -> the SDK default)
    - ollama     : ``config.ollama_base_url`` + ``/v1``
    - lmstudio  : ``config.lmstudio_base_url`` (already ends in ``/v1``)
    - vllm      : ``config.vllm_base_url``     (already ends in ``/v1``)
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
                        # Convert Anthropic base64 format to OpenAI format
                        image_data = item["source"]["data"]
                        media_type = item["source"]["media_type"]
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

    def _send_request(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        schema: Optional[str] = None,
        output_config: Optional[Dict[str, Any]] = None,  # Anthropic-only
    ) -> APIResponse:
        """Send request via the OpenAI SDK"""

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

        # OpenAI reasoning models (gpt-5 / o-series) only accept the default temperature.
        if self.llm_provider == "openai" and self._is_openai_reasoning_model():
            print("## Note: OpenAI reasoning model — using default temperature")
        else:
            params["temperature"] = self.config.temperature

        # Send request
        response = self.client.chat.completions.create(**params)

        # Extract usage information
        usage = TokenUsage(
            input_tokens=getattr(response.usage, 'prompt_tokens', 0),
            output_tokens=getattr(response.usage, 'completion_tokens', 0)
        )

        # Print usage stats
        request_id = getattr(response, 'id', 'N/A')
        print(f"## Request ID: {request_id}")
        print(f"## Provider: {self.llm_provider}")
        print(f"## Model: {self.model}")
        print(f"## Input tokens: {usage.input_tokens}")
        print(f"## Output tokens: {usage.output_tokens}")

        return APIResponse(
            content=response.choices[0].message.content,
            usage=usage,
            request_id=request_id,
            model=self.model
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
