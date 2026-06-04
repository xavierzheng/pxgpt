"""Abstract base provider for LLM services."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import time
import random


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class APIResponse:
    content: str
    usage: TokenUsage
    request_id: Optional[str] = None
    model: Optional[str] = None


class BaseProvider(ABC):

    def __init__(self, config):
        self.config = config
        self._client = None

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def _create_client(self):
        pass

    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @abstractmethod
    def _send_request(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        schema: Optional[str] = None,
        output_config: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:
        """Send a single request to the provider.

        *schema* is the legacy path (schema text appended to system prompt).
        *output_config* is the native Anthropic path (passed directly to the
        API).  Providers that do not support output_config should ignore it.
        """

    def send_request_with_retry(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        schema: Optional[str] = None,
        output_config: Optional[Dict[str, Any]] = None,
    ) -> APIResponse:
        """Send request with unified error-handling and retry logic."""
        max_retries = self.config.max_retries
        base_delay = 1

        for attempt in range(max_retries + 1):
            try:
                return self._send_request(messages, system_prompt, schema, output_config)
            except Exception as e:
                if attempt == max_retries:
                    raise
                if self._is_rate_limit_error(e):
                    sleep_time = self.config.rate_limit_sleep
                    print(f"Rate limit hit, sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)
                elif self._is_retryable_error(e):
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Retryable error (attempt {attempt + 1}/{max_retries + 1}), "
                          f"retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    raise

    @abstractmethod
    def _is_rate_limit_error(self, error: Exception) -> bool:
        pass

    @abstractmethod
    def _is_retryable_error(self, error: Exception) -> bool:
        pass

    def supports_caching(self) -> bool:
        return False

    def estimate_tokens(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        schema: Optional[str] = None,
    ) -> Optional[int]:
        return None
