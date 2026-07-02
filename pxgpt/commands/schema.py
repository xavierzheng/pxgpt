"""Single-request structured JSON analysis (Stage 3, non-batch).

Useful for testing a schema against one folder before running the full batch.
For Anthropic the schema is passed via native structured output
(``output_config.format``); for other providers it is appended to the system
prompt (legacy path).
"""

import json
import argparse
from typing import Optional

from ..core.config import Config
from ..core.image_utils import create_multi_image_message
from ..core.file_utils import read_file_safely, write_file_safely
from ..core.schema_utils import load_normalized
from ..core.batch_utils import temperature_guard_status
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.litellm_provider import LiteLLMProvider


LITELLM_PROVIDERS = {"openai", "google", "ollama", "lmstudio", "vllm"}


def create_provider(provider_name: str, config: Config):
    if provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name in LITELLM_PROVIDERS:
        return LiteLLMProvider(config, provider_name)
    raise ValueError(f"Unsupported provider: {provider_name}")


def schema_command(args):
    config = Config.from_env()
    provider_name = args.provider or config.provider

    if not config.validate_provider(provider_name):
        raise ValueError(
            f"Provider '{provider_name}' is not properly configured.  "
            "Check your API keys."
        )

    try:
        system_prompt = read_file_safely(args.system_prompt, "system prompt")
        user_prompt = read_file_safely(args.prompt, "user prompt")
    except (FileNotFoundError, IOError) as e:
        print(f"File error: {e}")
        return 1

    try:
        messages = create_multi_image_message(args.input_folder, user_prompt)
    except Exception as e:
        print(f"Error processing images: {e}")
        return 1

    provider = create_provider(provider_name, config)
    print(f"Using provider: {provider.provider_name}")

    try:
        if provider_name == "anthropic":
            # Native structured output — schema is NOT in the system prompt
            schema_dict = load_normalized(args.schema)
            # --effort overrides STAGE3_EFFORT; "off" disables thinking.
            effort = config.stage3_effort if args.effort is None else args.effort
            if effort == "off":
                effort = ""
            output_config = config.build_output_config(effort, schema=schema_dict)
            print(f"Structured output: native (output_config.format)")
            print(f"Thinking effort:   {output_config.get('effort', 'off')}")
            print(f"Temperature:       {temperature_guard_status(config.anthropic_model, effort)}")

            response = provider.send_request_with_retry(
                messages=messages,
                system_prompt=system_prompt,
                output_config=output_config,
            )
        else:
            # Legacy: schema text in system prompt
            schema_text = read_file_safely(args.schema, "JSON schema")
            print(f"Structured output: schema in system prompt (legacy)")
            response = provider.send_request_with_retry(
                messages=messages,
                system_prompt=system_prompt,
                schema=schema_text,
            )
    except Exception as e:
        print(f"Error during schema analysis: {e}")
        return 1

    write_file_safely(args.output, response.content, "output")
    print(f"Results written to: {args.output}")
    return 0


def setup_schema_parser(subparsers):
    parser = subparsers.add_parser(
        "schema",
        help="Single-request structured JSON analysis (Stage 3, non-batch)",
        description=(
            "Analyze one folder of images against a JSON schema.  "
            "For Anthropic, uses native structured output via output_config.format."
        ),
    )
    parser.add_argument(
        "--input-folder", required=True,
        help="Path to folder containing images",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output file path",
    )
    parser.add_argument(
        "--system-prompt", required=True,
        help="System prompt file path",
    )
    parser.add_argument(
        "--schema", required=True,
        help="JSON schema file path",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="User prompt file path",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "google", "ollama", "lmstudio", "vllm"],
        help="LLM provider (overrides config/env)",
    )
    parser.add_argument(
        "--effort",
        choices=["off", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Anthropic adaptive thinking effort (overrides STAGE3_EFFORT). "
             "default = off = none = no reasoning; whether a custom temperature "
             "is sent when off depends on the model tier; a level enables "
             "reasoning. Ignored for non-anthropic providers.",
    )
    parser.set_defaults(func=schema_command)
