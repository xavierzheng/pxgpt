"""Anthropic provider with prompt caching support."""

from typing import Dict, Any, List, Optional
from anthropic import Anthropic, RateLimitError, APIConnectionError, APIStatusError

from .base import BaseProvider, APIResponse, TokenUsage


class AnthropicProvider(BaseProvider):
    """Anthropic provider with caching support"""
    
    def __init__(self, config):
        super().__init__(config)
        self.model_name = config.get_model("anthropic")
    
    @property
    def provider_name(self) -> str:
        return "anthropic"
    
    def _create_client(self):
        """Create Anthropic client"""
        api_key = self.config.get_api_key("anthropic")
        if not api_key:
            raise ValueError("Anthropic API key not provided")
        return Anthropic(api_key=api_key, max_retries=0)  # We handle retries ourselves
    
    def supports_caching(self) -> bool:
        return True
    
    def estimate_tokens(self, messages: List[Dict[str, Any]], 
                       system_prompt: str, schema: Optional[str] = None) -> Optional[int]:
        """Estimate input tokens using Anthropic's count API"""
        try:
            system_messages = self._build_system_messages(system_prompt, schema)
            count = self.client.beta.messages.count_tokens(
                model=self.model_name,
                system=system_messages,
                messages=messages
            )
            return count.input_tokens
        except Exception:
            return None
    
    def _build_system_messages(self, system_prompt: str, schema: Optional[str] = None) -> List[Dict[str, Any]]:
        """Build system messages with caching"""
        system_messages = [{"type": "text", "text": system_prompt}]
        
        if schema:
            system_messages.append({
                "type": "text",
                "text": schema,
                "cache_control": {"type": "ephemeral"}
            })
        
        return system_messages
    
    def _send_request(self, messages: List[Dict[str, Any]], 
                      system_prompt: str, schema: Optional[str] = None) -> APIResponse:
        """Send request to Anthropic API"""
        
        # Estimate tokens
        estimated_tokens = self.estimate_tokens(messages, system_prompt, schema)
        if estimated_tokens:
            print(f"## Estimated input tokens: {estimated_tokens}")
        
        # Build system messages with caching
        system_messages = self._build_system_messages(system_prompt, schema)
        
        # Send request
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_messages,
            messages=messages
        )
        
        # Extract usage information
        usage = TokenUsage(
            input_tokens=getattr(response.usage, 'input_tokens', 0),
            output_tokens=getattr(response.usage, 'output_tokens', 0),
            cache_creation_tokens=getattr(response.usage, 'cache_creation_input_tokens', 0),
            cache_read_tokens=getattr(response.usage, 'cache_read_input_tokens', 0)
        )
        
        # Print usage stats
        print(f"## Request ID: {response._request_id}")
        print(f"## Model: {self.model_name}")
        print(f"## cache_creation_input_tokens: {usage.cache_creation_tokens}")
        print(f"## cache_read_input_tokens: {usage.cache_read_tokens}")
        print(f"## Actual input tokens: {usage.input_tokens}")
        print(f"## Actual output tokens: {usage.output_tokens}")
        
        return APIResponse(
            content=response.content[0].text,
            usage=usage,
            request_id=response._request_id,
            model=self.model_name
        )
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error"""
        return isinstance(error, RateLimitError)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable"""
        if isinstance(error, APIConnectionError):
            return True
        if isinstance(error, APIStatusError):
            # Some status codes are retryable
            return error.status_code in [429, 502, 503, 504]
        return False