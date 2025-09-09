"""Structured JSON analysis command with schema validation."""

import argparse
from typing import Optional

from ..core.config import Config
from ..core.image_utils import create_multi_image_message
from ..core.file_utils import read_file_safely, write_file_safely
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.litellm_provider import LiteLLMProvider


def create_provider(provider_name: str, config: Config):
    """Factory function to create appropriate provider"""
    if provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name in ["openai", "google", "ollama"]:
        return LiteLLMProvider(config, provider_name)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


def schema_command(args):
    """Execute the schema command"""
    
    # Load configuration
    config = Config.from_env()
    
    # Override provider if specified
    provider_name = args.provider or config.provider
    
    # Validate provider configuration
    if not config.validate_provider(provider_name):
        raise ValueError(f"Provider '{provider_name}' is not properly configured. Check your API keys.")
    
    # Read input files
    try:
        system_prompt = read_file_safely(args.system_prompt, "system prompt")
        prompt_text = read_file_safely(args.prompt, "prompt")
        json_schema = read_file_safely(args.schema, "JSON schema")
    except (FileNotFoundError, IOError) as e:
        print(f"File error: {e}")
        return 1
    
    # Create messages
    try:
        messages = create_multi_image_message(args.input_folder, prompt_text)
    except Exception as e:
        print(f"Error processing images: {e}")
        return 1
    
    # Create provider and send request
    try:
        provider = create_provider(provider_name, config)
        print(f"Using provider: {provider.provider_name}")
        
        # Show caching support info
        if provider.supports_caching():
            print("## Prompt caching: ENABLED")
        else:
            print("## Prompt caching: NOT SUPPORTED (using combined system prompt)")
        
        response = provider.send_request_with_retry(
            messages=messages,
            system_prompt=system_prompt,
            schema=json_schema
        )
        
        # Write output
        write_file_safely(args.output, response.content, "output")
        print(f"Results successfully written to file: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error during schema analysis: {e}")
        return 1


def setup_schema_parser(subparsers):
    """Set up the schema command parser"""
    parser = subparsers.add_parser(
        'schema',
        help='Structured JSON analysis with schema validation',
        description='Generate structured JSON output for plant images using AI with schema validation'
    )
    
    parser.add_argument(
        '--input-folder',
        required=True,
        help='Path to folder containing images'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Output file path'
    )
    
    parser.add_argument(
        '--system-prompt',
        required=True, 
        help='System prompt file path'
    )
    
    parser.add_argument(
        '--schema',
        required=True,
        help='JSON schema file path'
    )
    
    parser.add_argument(
        '--prompt',
        required=True,
        help='User prompt file path'
    )
    
    parser.add_argument(
        '--provider',
        choices=['anthropic', 'openai', 'google', 'ollama'],
        help='LLM provider to use (overrides config/env)'
    )
    
    parser.set_defaults(func=schema_command)