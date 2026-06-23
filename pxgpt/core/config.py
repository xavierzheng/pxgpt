"""Configuration management for PXGPT."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


VALID_EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}
# OpenAI reasoning models accept a different set; "" disables the param.
VALID_OPENAI_EFFORT_LEVELS = {"minimal", "low", "medium", "high"}


@dataclass
class Config:
    """Configuration for PXGPT."""

    # Provider
    provider: str = "anthropic"

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Base URLs for local / OpenAI-compatible providers
    openai_base_url: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    vllm_base_url: str = "http://localhost:8000/v1"

    # API keys for OpenAI-compatible local servers (usually a dummy placeholder)
    lmstudio_api_key: str = "lm-studio"
    vllm_api_key: str = "EMPTY"

    # Model names
    anthropic_model: str = "claude-sonnet-4-6"
    openai_model: str = "gpt-5-2025-08-07"
    google_model: str = "gemini-2.5-pro"
    ollama_model: str = "gemma3:12b"
    lmstudio_model: str = "local-model"   # name as shown in LM Studio
    vllm_model: str = ""                   # the served model name (required for vllm)

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
    # Default "" (off): Stage 3 runs WITHOUT reasoning and DOES send temperature.
    # Reasoning is opt-in via STAGE3_EFFORT or the --effort flag.
    stage3_effort: str = ""

    # Thinking / effort level for the sync `analyze` command (Anthropic only).
    # Default "" (off) preserves the original non-thinking behavior; the
    # --effort CLI flag overrides this per run.
    analyze_effort: str = ""

    # Set True to request the output-300k-2026-03-24 beta header on Stage 1
    # batches, raising the per-response output cap from 64 k to 300 k tokens.
    batch_300k_output: bool = False

    # When True (default), batch stages upload each image once via the Files API
    # and reference it by file_id (cheaper, no re-upload across stages). When
    # False, images are embedded inline as base64 in every request and the
    # files-api beta header is omitted. The --no-files-api CLI flag overrides this.
    use_files_api: bool = True

    # Concurrency for parallel image uploads via Files API
    upload_concurrency: int = 10

    # --- OpenAI batch settings (describe-batch-openai / phenotype-batch-openai) ---
    # Reasoning effort for OpenAI reasoning models (gpt-5, o-series). Valid:
    # "minimal", "low", "medium", "high", or "" to omit the param entirely.
    openai_reasoning_effort: str = ""
    # Completion window passed to the OpenAI Batch API.
    openai_batch_completion_window: str = "24h"

    # Rate limiting
    rate_limit_sleep: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        stage3_effort = os.getenv("STAGE3_EFFORT", "")
        if stage3_effort not in VALID_EFFORT_LEVELS and stage3_effort != "":
            raise ValueError(
                f"STAGE3_EFFORT must be one of {VALID_EFFORT_LEVELS} or '' (empty), "
                f"got: {stage3_effort!r}"
            )
        openai_reasoning_effort = os.getenv("OPENAI_REASONING_EFFORT", "")
        if openai_reasoning_effort not in VALID_OPENAI_EFFORT_LEVELS and openai_reasoning_effort != "":
            raise ValueError(
                f"OPENAI_REASONING_EFFORT must be one of {VALID_OPENAI_EFFORT_LEVELS} "
                f"or '' (empty), got: {openai_reasoning_effort!r}"
            )
        analyze_effort = os.getenv("ANALYZE_EFFORT", "")
        if analyze_effort not in VALID_EFFORT_LEVELS and analyze_effort != "":
            raise ValueError(
                f"ANALYZE_EFFORT must be one of {VALID_EFFORT_LEVELS} or '' (empty), "
                f"got: {analyze_effort!r}"
            )
        return cls(
            provider=os.getenv("DEFAULT_PROVIDER", "anthropic"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            lmstudio_api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            vllm_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-2025-08-07"),
            google_model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            lmstudio_model=os.getenv("LMSTUDIO_MODEL", "local-model"),
            vllm_model=os.getenv("VLLM_MODEL", ""),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout=int(os.getenv("TIMEOUT", "300")),
            temperature=float(os.getenv("TEMPERATURE", "0.5")),
            max_tokens=int(os.getenv("MAX_TOKENS", "16384")),
            stage1_max_tokens=int(os.getenv("STAGE1_MAX_TOKENS", "16384")),
            stage3_max_tokens=int(os.getenv("STAGE3_MAX_TOKENS", "16384")),
            stage3_effort=stage3_effort,
            analyze_effort=analyze_effort,
            batch_300k_output=os.getenv("BATCH_300K_OUTPUT", "false").lower() in ("1", "true", "yes"),
            use_files_api=os.getenv("USE_FILES_API", "true").lower() in ("1", "true", "yes"),
            upload_concurrency=int(os.getenv("UPLOAD_CONCURRENCY", "10")),
            openai_reasoning_effort=openai_reasoning_effort,
            openai_batch_completion_window=os.getenv("OPENAI_BATCH_COMPLETION_WINDOW", "24h"),
            rate_limit_sleep=int(os.getenv("RATE_LIMIT_SLEEP", "60")),
        )

    def get_api_key(self, provider: str) -> Optional[str]:
        if provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "openai":
            return self.openai_api_key
        elif provider == "google":
            return self.google_api_key
        elif provider == "lmstudio":
            return self.lmstudio_api_key
        elif provider == "vllm":
            return self.vllm_api_key
        return None

    def get_model(self, provider: str) -> str:
        if provider == "anthropic":
            return self.anthropic_model
        elif provider == "openai":
            return self.openai_model
        elif provider == "google":
            return self.google_model
        elif provider == "lmstudio":
            return self.lmstudio_model
        elif provider == "vllm":
            return self.vllm_model
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
        elif provider == "lmstudio":
            return bool(self.lmstudio_base_url)
        elif provider == "vllm":
            # vLLM needs both an endpoint and an explicit served model name.
            return bool(self.vllm_base_url and self.vllm_model)
        return False

    def build_output_config(
        self,
        effort: str = "",
        schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return an ``output_config`` dict from an effort level and optional schema.

        *effort* enables adaptive thinking (omit / pass "" to disable).  *schema*
        adds native structured output via ``output_config.format``.  Returns an
        empty dict when neither is set (caller may treat that as "no
        output_config").
        """
        cfg: Dict[str, Any] = {}
        if effort:
            cfg["effort"] = effort
        if schema is not None:
            cfg["format"] = {"type": "json_schema", "schema": schema}
        return cfg

    def stage3_output_config(self, schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return the output_config dict for Stage 3 / schema command requests."""
        return self.build_output_config(self.stage3_effort, schema)
