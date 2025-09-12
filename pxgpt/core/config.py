"""Configuration management for PXGPT."""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class Config:
    """Configuration for PXGPT"""
    
    # Provider settings
    provider: str = "anthropic"
    
    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None  
    google_api_key: Optional[str] = None
    
    # Base URLs for local providers
    openai_base_url: Optional[str] = None  # For LM Studio compatibility
    ollama_base_url: str = "http://localhost:11434"
    
    # Model configurations
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    openai_model: str = "gpt-5-2025-08-07"
    google_model: str = "gemini-2.5-pro"
    ollama_model: str = "gemma3:12b"
    
    # Request settings
    max_retries: int = 3
    timeout: int = 300
    temperature: float = 0.5
    max_tokens: int = 8192
    
    # Rate limiting
    rate_limit_sleep: int = 60
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables"""
        return cls(
            provider=os.getenv("DEFAULT_PROVIDER", "anthropic"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),  # LM Studio: http://localhost:1234/v1
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-2025-08-07"),
            google_model=os.getenv("GOOGLE_MODEL", "gemini-2.5-pro"),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:12b"),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            timeout=int(os.getenv("TIMEOUT", "300")),
            temperature=float(os.getenv("TEMPERATURE", "0.5")),
            max_tokens=int(os.getenv("MAX_TOKENS", "8192")),
            rate_limit_sleep=int(os.getenv("RATE_LIMIT_SLEEP", "60"))
        )
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for specific provider"""
        if provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "openai":
            return self.openai_api_key
        elif provider == "google":
            return self.google_api_key
        return None
    
    def get_model(self, provider: str) -> str:
        """Get model name for specific provider"""
        if provider == "anthropic":
            return self.anthropic_model
        elif provider == "openai":
            return self.openai_model
        elif provider == "google":
            return self.google_model
        elif provider == "ollama":
            return self.ollama_model
        return provider  # fallback to provider name
    
    def validate_provider(self, provider: str) -> bool:
        """Check if provider has required configuration"""
        if provider == "anthropic":
            return self.anthropic_api_key is not None
        elif provider == "openai":
            return self.openai_api_key is not None
        elif provider == "google":
            return self.google_api_key is not None
        elif provider == "ollama":
            return True  # Local, no API key needed
        return False