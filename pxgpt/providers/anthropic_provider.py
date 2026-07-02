"""Anthropic provider — sync single-request path with prompt caching.

For batch operations (Stage 1 / Stage 3) the calling commands use the
Anthropic client directly; this provider covers the ``analyze`` and ``schema``
CLI commands.

Temperature / thinking guard
-----------------------------
``_build_kwargs`` delegates to ``batch_utils.build_request_params`` so the
sync and batch code paths apply identical, model-aware rules — see that
function's docstring for the guard logic per model tier.
"""

from typing import Any, Dict, List, Optional

from anthropic import Anthropic, RateLimitError, APIConnectionError, APIStatusError

from .base import BaseProvider, APIResponse, TokenUsage
from ..core.batch_utils import build_request_params


class AnthropicProvider(BaseProvider):

    def __init__(self, config):
        super().__init__(config)
        self.model_name = config.get_model("anthropic")

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _create_client(self):
        api_key = self.config.get_api_key("anthropic")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        return Anthropic(api_key=api_key, max_retries=0)  # retries handled by base class

    def supports_caching(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def estimate_tokens(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        schema: Optional[str] = None,
    ) -> Optional[int]:
        try:
            system = self._build_system(system_prompt, schema)
            count = self.client.beta.messages.count_tokens(
                model=self.model_name,
                system=system,
                messages=messages,
            )
            return count.input_tokens
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system(
        self,
        system_prompt: str,
        schema_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the system message list.

        When *schema_text* is provided (legacy path) it is appended as a
        separate block with prompt-cache control.  When native structured
        output is used via ``output_config.format`` the schema is NOT passed
        here; only the system prompt itself is included (also cached).
        """
        blocks: List[Dict[str, Any]] = [{"type": "text", "text": system_prompt}]
        if schema_text:
            blocks.append({
                "type": "text",
                "text": schema_text,
                "cache_control": {"type": "ephemeral"},
            })
        else:
            # Cache the system prompt so repeated calls (e.g. testing) hit the
            # cache.  This is a no-op if the prompt is too short to be cached.
            blocks[0]["cache_control"] = {"type": "ephemeral"}
        return blocks

    def _build_kwargs(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        schema_text: Optional[str],
        output_config: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assemble keyword arguments for ``messages.create``.

        Temperature/thinking guard is applied by ``build_request_params``
        (see module docstring).
        """
        return build_request_params(
            model=self.model_name,
            max_tokens=self.config.max_tokens,
            system=self._build_system(system_prompt, schema_text),
            messages=messages,
            temperature=self.config.temperature,
            output_config=output_config,
        )

    @staticmethod
    def _extract_text(content_blocks) -> str:
        """Return concatenated text from TextBlock entries (skip thinking blocks)."""
        return "\n".join(
            b.text for b in content_blocks if getattr(b, "type", None) == "text"
        )

    # ------------------------------------------------------------------
    # BaseProvider implementation
    # ------------------------------------------------------------------

    def _send_request(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        schema: Optional[str] = None,
        output_config: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:

        estimated = self.estimate_tokens(messages, system_prompt, schema)
        if estimated:
            print(f"## Estimated input tokens: {estimated}")

        kwargs = self._build_kwargs(messages, system_prompt, schema, output_config)
        response = self.client.messages.create(**kwargs)

        usage = TokenUsage(
            input_tokens=getattr(response.usage, "input_tokens", 0),
            output_tokens=getattr(response.usage, "output_tokens", 0),
            cache_creation_tokens=getattr(response.usage, "cache_creation_input_tokens", 0),
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
        )

        print(f"## Request ID:                  {response._request_id}")
        print(f"## Model:                       {self.model_name}")
        print(f"## cache_creation_input_tokens: {usage.cache_creation_tokens}")
        print(f"## cache_read_input_tokens:     {usage.cache_read_tokens}")
        print(f"## Actual input tokens:         {usage.input_tokens}")
        print(f"## Actual output tokens:        {usage.output_tokens}")

        return APIResponse(
            content=self._extract_text(response.content),
            usage=usage,
            request_id=response._request_id,
            model=self.model_name,
        )

    def _is_rate_limit_error(self, error: Exception) -> bool:
        return isinstance(error, RateLimitError)

    def _is_retryable_error(self, error: Exception) -> bool:
        if isinstance(error, APIConnectionError):
            return True
        if isinstance(error, APIStatusError):
            return error.status_code in {429, 502, 503, 504}
        return False
