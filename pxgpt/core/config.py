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
    ollama_base_url: str = "http://localhost:11434"
    
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
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
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