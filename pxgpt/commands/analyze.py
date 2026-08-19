"""Basic image analysis command."""

import argparse
from typing import Optional

from ..core.config import Config
from ..core.image_utils import create_multi_image_message, IMAGE_TRANSPORTS
from ..core.file_utils import read_file_safely, write_file_safely
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.openai_compat_provider import OpenAICompatProvider


OPENAI_COMPAT_PROVIDERS = ["openai", "ollama", "lmstudio", "vllm"]


def create_provider(provider_name: str, config: Config):
    """Factory function to create appropriate provider"""
    if provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name in OPENAI_COMPAT_PROVIDERS:
        return OpenAICompatProvider(config, provider_name)
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")


def analyze_command(args):
    """Execute the analyze command"""
    
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
    except (FileNotFoundError, IOError) as e:
        print(f"File error: {e}")
        return 1
    
    # Create messages
    try:
        messages = create_multi_image_message(
            args.input_folder, prompt_text, args.image_transport
        )
    except Exception as e:
        print(f"Error processing images: {e}")
        return 1
    
    # Resolve thinking effort: --effort overrides ANALYZE_EFFORT; "off" disables.
    effort = config.analyze_effort if args.effort is None else args.effort
    if effort == "off":
        effort = ""

    output_config = None
    if effort:
        if provider_name == "anthropic":
            output_config = config.build_output_config(effort)
            print(f"Thinking effort: {effort} (temperature omitted while thinking)")
        elif provider_name == "openai":
            # The provider turns this into reasoning_effort; off is sent as "none".
            output_config = config.build_output_config(effort)
            print(f"Reasoning effort: {effort}")
        else:
            print(f"Note: --effort is only supported for the anthropic and openai "
                  f"providers; ignoring for '{provider_name}'")

    # Create provider and send request
    try:
        provider = create_provider(provider_name, config)
        print(f"Using provider: {provider.provider_name}")
        print(f"Image transport: {args.image_transport}")

        response = provider.send_request_with_retry(
            messages=messages,
            system_prompt=system_prompt,
            output_config=output_config,
        )

        # Write output
        write_file_safely(args.output, response.content, "output")
        print(f"Results successfully written to file: {args.output}")

        return 0

    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1


def setup_analyze_parser(subparsers):
    """Set up the analyze command parser"""
    parser = subparsers.add_parser(
        'analyze',
        help='Basic image analysis with text output',
        description='Analyze images using AI and generate text descriptions'
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
        '--prompt',
        required=True,
        help='User prompt file path'
    )
    
    parser.add_argument(
        '--provider',
        choices=['anthropic', 'openai', 'ollama', 'lmstudio', 'vllm'],
        help='LLM provider to use (overrides config/env)'
    )

    parser.add_argument(
        '--image-transport',
        choices=list(IMAGE_TRANSPORTS),
        default='base64',
        help="How images reach the model.  'base64' embeds the bytes in the "
             "request and works everywhere (default).  'file' sends file:// "
             "URIs and is the recommended path for a local vLLM server, which "
             "must have the image directory mounted at the very same path.  "
             "(default: base64)"
    )

    parser.add_argument(
        '--effort',
        choices=['off', 'low', 'medium', 'high', 'xhigh', 'max'],
        default=None,
        help='Reasoning effort (overrides ANALYZE_EFFORT). default = off = none '
             '= no reasoning; a level enables it. Anthropic adaptive thinking or '
             'OpenAI reasoning_effort, depending on the provider; whether a '
             'custom temperature is sent when off depends on the model. '
             'Ignored by the local providers.'
    )

    parser.set_defaults(func=analyze_command)