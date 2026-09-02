"""Configuration management for PXGPT."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any


# One effort vocabulary for both providers: Anthropic (claude-sonnet-5 and the
# other adaptive-thinking tiers) and OpenAI (gpt-5.x / o-series) both accept
# these five levels.  ""/off/none all mean OFF, spelled per provider — Anthropic
# gets thinking disabled, OpenAI gets reasoning.effort "none".
# Note: OpenAI's older "minimal" level is not offered; gpt-5.6 dropped it.
VALID_EFFORT_LEVELS = {"low", "medium", "high", "xhigh", "max"}


def _normalize_effort(value: str) -> str:
    """Map the "no reasoning" spellings to the empty string.

    ``off``, ``none`` and ``""`` (any case) all mean: no reasoning. Whether a
    custom temperature is then sent depends on the model tier — see
    ``batch_utils.build_request_params``. This lets the env vars and the
    ``--effort off`` CLI flag agree.
    """
    v = (value or "").strip()
    return "" if v.lower() in ("", "off", "none") else v


@dataclass
class Config:
    """Configuration for PXGPT."""

    # Provider
    provider: str = "anthropic"

    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Base URLs for local / OpenAI-compatible providers.
    # DEPRECATED: ollama_* and lmstudio_* are slated for removal in a future
    # major release -- vLLM is the supported local backend, because it is the
    # only one that lets you pin the per-image visual token budget
    # (max_soft_tokens; see ops/local-vllm/ and OpenAICompatProvider).
    # Still fully functional; no runtime warning is emitted.
    openai_base_url: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    lmstudio_base_url: str = "http://localhost:1234/v1"
    vllm_base_url: str = "http://localhost:8000/v1"

    # API keys for OpenAI-compatible local servers (usually a dummy placeholder)
    lmstudio_api_key: str = "lm-studio"
    vllm_api_key: str = "EMPTY"

    # Model names
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-5.6-luna"
    ollama_model: str = "gemma3:12b"
    lmstudio_model: str = "local-model"   # name as shown in LM Studio
    vllm_model: str = ""                   # the served model name (required for vllm)

    # Sync API settings
    max_retries: int = 3
    timeout: int = 300
    # Temperature is only sent when thinking is off, and only on model tiers
    # that allow a custom temperature while thinking is off (Sonnet 4.6 and
    # earlier). Newer tiers (Sonnet 5, Opus 4.7/4.8, Fable 5) reject a custom
    # temperature unconditionally — see batch_utils.build_request_params.
    temperature: float = 0.5
    # Nucleus / top-k sampling for the OpenAI-compatible backends.  Defaults are
    # the Gemma 4 checkpoint's own generation_config.json values.  The local
    # server is started with the same numbers via --override-generation-config;
    # sending them from the client too is deliberate duplication, because only
    # what the client sends appears in the request record.  Anthropic ignores
    # both (its batch path builds its own params).
    top_p: float = 0.95
    top_k: int = 64
    max_tokens: int = 16384  # sync API; current models support up to 128 k with streaming

    # Batch-specific token budgets
    stage1_max_tokens: int = 16384   # Stage 1 descriptions (raise to 300 k if needed)
    stage3_max_tokens: int = 16384   # Stage 3 structured JSON

    # Adaptive thinking effort (Anthropic). For all three knobs below:
    #   default = "" = off = none = NO reasoning. Whether a custom temperature
    #   is then sent depends on the model tier (see `temperature` above): sent
    #   on Sonnet 4.6 and earlier, omitted on the strict-guard tiers.
    # Enable reasoning with a level: "low" | "medium" | "high" | "xhigh" | "max"
    # (env vars also accept "off"/"none"; the --effort flag overrides per run).

    # Stage 3 (phenotype-batch) and the schema command — env STAGE3_EFFORT.
    stage3_effort: str = ""
    # Sync `analyze` command — env ANALYZE_EFFORT.
    analyze_effort: str = ""
    # Stage 1 (describe-batch) — env DESCRIBE_EFFORT.
    describe_effort: str = ""

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
    # Reasoning effort reuses the shared per-stage knobs above: describe_effort
    # for Stage 1 (overridable with --effort) and stage3_effort for Stage 3.
    # Off is sent to OpenAI as an explicit reasoning.effort "none" — omitting the
    # param would leave the model on its own default (medium on gpt-5.6), i.e.
    # reasoning silently ON.
    # Completion window passed to the OpenAI Batch API.
    openai_batch_completion_window: str = "24h"

    # Rate limiting
    rate_limit_sleep: int = 60

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        # Effort env vars accept off/none/"" — all mean: no reasoning (whether a
        # custom temperature is then sent is decided per model tier downstream).
        stage3_effort = _normalize_effort(os.getenv("STAGE3_EFFORT", ""))
        if stage3_effort not in VALID_EFFORT_LEVELS and stage3_effort != "":
            raise ValueError(
                f"STAGE3_EFFORT must be one of {VALID_EFFORT_LEVELS} or off/none, "
                f"got: {stage3_effort!r}"
            )
        analyze_effort = _normalize_effort(os.getenv("ANALYZE_EFFORT", ""))
        if analyze_effort not in VALID_EFFORT_LEVELS and analyze_effort != "":
            raise ValueError(
                f"ANALYZE_EFFORT must be one of {VALID_EFFORT_LEVELS} or off/none, "
                f"got: {analyze_effort!r}"
            )
        describe_effort = _normalize_effort(os.getenv("DESCRIBE_EFFORT", ""))
        if describe_effort not in VALID_EFFORT_LEVELS and describe_effort != "":
            raise ValueError(
                f"DESCRIBE_EFFORT must be one of {VALID_EFFORT_LEVELS} or off/none, "
                f"got: {describe_effort!r}"
            )
        return cls(
            provider=os.getenv("DEFAULT_PROVIDER", "anthropic"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            lmstudio_base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
            vllm_base_url=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
            lmstudio_api_key=os.getenv("LMSTUDIO_API_KEY", "lm-studio"),
            vllm_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            lmstudio_model=os.getenv("LMSTUDIO_MODEL", "local-model"),
            vllm_model=os.getenv("VLLM_MODEL", ""),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout=int(os.getenv("TIMEOUT", "300")),
            temperature=float(os.getenv("TEMPERATURE", "0.5")),
            top_p=float(os.getenv("TOP_P", "0.95")),
            top_k=int(os.getenv("TOP_K", "64")),
            max_tokens=int(os.getenv("MAX_TOKENS", "16384")),
            stage1_max_tokens=int(os.getenv("STAGE1_MAX_TOKENS", "16384")),
            stage3_max_tokens=int(os.getenv("STAGE3_MAX_TOKENS", "16384")),
            stage3_effort=stage3_effort,
            analyze_effort=analyze_effort,
            describe_effort=describe_effort,
            batch_300k_output=os.getenv("BATCH_300K_OUTPUT", "false").lower() in ("1", "true", "yes"),
            use_files_api=os.getenv("USE_FILES_API", "true").lower() in ("1", "true", "yes"),
            upload_concurrency=int(os.getenv("UPLOAD_CONCURRENCY", "10")),
            openai_batch_completion_window=os.getenv("OPENAI_BATCH_COMPLETION_WINDOW", "24h"),
            rate_limit_sleep=int(os.getenv("RATE_LIMIT_SLEEP", "60")),
        )

    def get_api_key(self, provider: str) -> Optional[str]:
        if provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "openai":
            return self.openai_api_key
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
