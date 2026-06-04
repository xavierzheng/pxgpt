"""Configuration management for PXGPT."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


VALID_EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}


@dataclass
class Config:
    """Configuration for PXGPT."""

    # Provider
    provider: str = "anthropic"

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Base URLs for local providers
    openai_base_url: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"

    # Model names
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-5-2025-08-07"
    google_model: str = "gemini-2.5-pro"
    ollama_model: str = "gemma3:12b"

    # Sync API settings
    max_retries: int = 3
    timeout: int = 300
    # Temperature is only sent when thinking is off (Stage 1 / analyze / schema).
    # When output_config.effort is set, the API forces the default temperature and
    # rejects custom values; the temperature guard in anthropic_provider enforces this.
    temperature: float = 0.5
    max_tokens: int = 16384  # sync API; claude-sonnet-4-6 supports up to 64 k

    # Batch-specific token budgets
    stage1_max_tokens: int = 16384   # Stage 1 descriptions (raise to 300 k if needed)
    stage3_max_tokens: int = 16384   # Stage 3 structured JSON

    # Thinking / effort level for Stage 3 and the schema command.
    # Valid values: "low", "medium", "high", "xhigh", "max"
    # Set to empty string "" to disable thinking entirely.
    stage3_effort: str = "medium"

    # Set True to request the output-300k-2026-03-24 beta header on Stage 1
    # batches, raising the per-response output cap from 64 k to 300 k tokens.
    batch_300k_output: bool = False

    # Concurrency for parallel image uploads via Files API
    upload_concurrency: int = 10

    # Rate limiting
    rate_limit_sleep: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        stage3_effort = os.getenv("STAGE3_EFFORT", "medium")
        if stage3_effort not in VALID_EFFORT_LEVELS and stage3_effort != "":
            raise ValueError(
                f"STAGE3_EFFORT must be one of {VALID_EFFORT_LEVELS} or '' (empty), "
                f"got: {stage3_effort!r}"
            )
        return cls(
            provider=os.getenv("DEFAULT_PROVIDER", "anthropic"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-2025-08-07"),
            google_model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout=int(os.getenv("TIMEOUT", "300")),
            temperature=float(os.getenv("TEMPERATURE", "0.5")),
            max_tokens=int(os.getenv("MAX_TOKENS", "16384")),
            stage1_max_tokens=int(os.getenv("STAGE1_MAX_TOKENS", "16384")),
            stage3_max_tokens=int(os.getenv("STAGE3_MAX_TOKENS", "16384")),
            stage3_effort=stage3_effort,
            batch_300k_output=os.getenv("BATCH_300K_OUTPUT", "false").lower() in ("1", "true", "yes"),
            upload_concurrency=int(os.getenv("UPLOAD_CONCURRENCY", "10")),
            rate_limit_sleep=int(os.getenv("RATE_LIMIT_SLEEP", "60")),
        )

    def get_api_key(self, provider: str) -> Optional[str]:
        if provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "openai":
            return self.openai_api_key
        elif provider == "google":
            return self.google_api_key
        return None

    def get_model(self, provider: str) -> str:
        if provider == "anthropic":
            return self.anthropic_model
        elif provider == "openai":
            return self.openai_model
        elif provider == "google":
            return self.google_model
        elif provider == "ollama":
            return self.ollama_model
        return provider

    def validate_provider(self, provider: str) -> bool:
        if provider == "anthropic":
            return self.anthropic_api_key is not None
        elif provider == "openai":
            return self.openai_api_key is not None
        elif provider == "google":
            return self.google_api_key is not None
        elif provider == "ollama":
            return True
        return False

    def stage3_output_config(self, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return the output_config dict for Stage 3 / schema command requests.

        Combines the thinking effort level with an optional structured-output
        format.  Callers should pass the result directly as output_config= to the
        provider or batch-request builder.
        """
        cfg: Dict[str, Any] = {}
        if self.stage3_effort:
            cfg["effort"] = self.stage3_effort
        if schema is not None:
            cfg["format"] = {"type": "json_schema", "schema": schema}
        return cfg
