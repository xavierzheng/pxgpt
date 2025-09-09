"""LiteLLM provider for OpenAI, Google, Ollama and other providers."""

from typing import Dict, Any, List, Optional
import litellm
from litellm import RateLimitError, APIConnectionError, APIError

from .base import BaseProvider, APIResponse, TokenUsage


class LiteLLMProvider(BaseProvider):
    """LiteLLM provider for multiple LLM services"""
    
    # Model mappings for different providers
    MODELS = {
        "openai": "gpt-4-vision-preview",
        "google": "gemini-1.5-pro-latest", 
        "ollama": "llama3.2-vision"
    }
    
    def __init__(self, config, provider: str):
        super().__init__(config)
        self.llm_provider = provider
        self.model = self.MODELS.get(provider, provider)
        
        # Set up provider-specific configuration
        self._setup_provider()
    
    @property
    def provider_name(self) -> str:
        return f"litellm-{self.llm_provider}"
    
    def _setup_provider(self):
        """Set up provider-specific configuration"""
        if self.llm_provider == "openai":
            api_key = self.config.get_api_key("openai")
            if api_key:
                litellm.api_key = api_key
        
        elif self.llm_provider == "google":
            api_key = self.config.get_api_key("google") 
            if api_key:
                litellm.api_key = api_key
        
        elif self.llm_provider == "ollama":
            litellm.api_base = self.config.ollama_base_url
    
    def _create_client(self):
        """LiteLLM doesn't need explicit client creation"""
        return None
    
    def supports_caching(self) -> bool:
        """LiteLLM providers don't support prompt caching"""
        return False
    
    def _build_system_prompt(self, system_prompt: str, schema: Optional[str] = None) -> str:
        """Build combined system prompt since caching isn't supported"""
        if schema:
            return f"{system_prompt}\n\nUse this JSON schema for your response:\n{schema}"
        return system_prompt
    
    def _convert_messages_for_litellm(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Anthropic-style messages to LiteLLM format"""
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
                        # Convert Anthropic base64 format to LiteLLM format
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
    
    def _send_request(self, messages: List[Dict[str, Any]], 
                      system_prompt: str, schema: Optional[str] = None) -> APIResponse:
        """Send request via LiteLLM"""
        
        # Build combined system prompt (no caching support)
        combined_system = self._build_system_prompt(system_prompt, schema)
        
        # Convert messages to LiteLLM format
        converted_messages = self._convert_messages_for_litellm(messages)
        
        # Add system message at the beginning
        full_messages = [{"role": "system", "content": combined_system}] + converted_messages
        
        # Send request
        response = litellm.completion(
            model=self.model,
            messages=full_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout
        )
        
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
        return isinstance(error, RateLimitError)
    
    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error is retryable"""
        if isinstance(error, APIConnectionError):
            return True
        if isinstance(error, APIError):
            # Check for retryable status codes in the error message
            error_str = str(error).lower()
            retryable_errors = ['502', '503', '504', 'timeout', 'connection']
            return any(err in error_str for err in retryable_errors)
        return False