"""Basic image analysis command."""

import argparse
from typing import Optional

from ..core.config import Config
from pathlib import Path

from ..core.image_utils import create_multi_image_message, IMAGE_TRANSPORTS
from .schema import add_image_source_args, resolve_plants
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
    
    # One plant (--input-folder) or a tree of them (--input-dir).
    try:
        plants = resolve_plants(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    multi = bool(args.input_dir)
    out = Path(args.output)
    if multi and out.exists() and not out.is_dir():
        print(f"Error: --output must be a DIRECTORY with --input-dir (one "
              f"description per plant), but {out} is an existing file.")
        return 1

    # Resolve thinking effort: --effort overrides ANALYZE_EFFORT; "off" disables.
    effort = config.analyze_effort if args.effort is None else args.effort
    if effort == "off":
        effort = ""

    output_config = None
    if effort:
        output_config = config.build_output_config(effort)
        if provider_name == "anthropic":
            print(f"Thinking effort: {effort} (temperature omitted while thinking)")
        elif provider_name == "openai":
            # The provider turns this into reasoning_effort; off is sent as "none".
            print(f"Reasoning effort: {effort}")
        else:
            # Gemma-class local models have no reasoning levels, only on/off, so
            # every level means the same thing here.  The server's reasoning
            # parser keeps the thinking in its own response field and leaves
            # `content` holding the final answer alone, so only the answer is
            # written to --output -- the reasoning is never saved.
            print(f"Thinking: on ('{effort}' -> on; local models have no levels). "
                  f"Reasoning stays in the response's own field and is NOT "
                  f"written to --output.")

    # Create provider and send request
    try:
        provider = create_provider(provider_name, config)
    except Exception as e:
        print(f"Error creating provider: {e}")
        return 1
    print(f"Using provider: {provider.provider_name}")
    print(f"Image transport: {args.image_transport}")

    if multi:
        out.mkdir(parents=True, exist_ok=True)
        print(f"Plants: {len(plants)}")

    failures = 0
    for i, plant in enumerate(plants, 1):
        dest = (out / f"{plant.name}.txt") if multi else out
        if multi:
            # Resume is the default: a description already on disk is not
            # re-billed.  A 277-plant run that dies at plant 200 restarts cheap.
            if args.resume and dest.exists():
                print(f"[{i}/{len(plants)}] {plant.name}  skip (cached)", flush=True)
                continue
            print(f"[{i}/{len(plants)}] {plant.name}", flush=True)

        try:
            messages = create_multi_image_message(
                str(plant), prompt_text, args.image_transport
            )
            response = provider.send_request_with_retry(
                messages=messages,
                system_prompt=system_prompt,
                output_config=output_config,
            )
        except Exception as e:  # noqa: BLE001
            print(f"Error during analysis ({plant.name}): {e}")
            failures += 1
            if not multi:
                return 1
            continue

        write_file_safely(str(dest), response.content, "output")
        if not multi:
            print(f"Results successfully written to file: {dest}")

    if multi:
        print(f"\nWrote {len(plants) - failures} of {len(plants)} description(s) "
              f"to {out}/")
        if failures:
            print(f"{failures} plant(s) failed; re-run to retry only those.")
    return 1 if failures else 0


def setup_analyze_parser(subparsers):
    """Set up the analyze command parser"""
    parser = subparsers.add_parser(
        'analyze',
        help='Basic image analysis with text output',
        description='Analyze images using AI and generate text descriptions'
    )
    
    add_image_source_args(parser)

    parser.add_argument(
        '--output',
        required=True,
        help='Output FILE for a single plant; a DIRECTORY with --input-dir '
             '(one <line_id>.txt per plant)'
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
        help=(
            "LLM provider (overrides config/env). "
            "Local backends: 'vllm' is the recommended one. It is the only "
            "local server that lets you pin the per-image visual token "
            "budget (max_soft_tokens -- see ops/local-vllm/), which is what "
            "keeps fine-grained traits legible and keeps local runs "
            "comparable with the cloud ones. "
            "'ollama' and 'lmstudio' are NOT recommended for phenotyping: "
            "neither exposes that control, so their image downsampling is "
            "neither settable nor reportable. Both still work today and are "
            "slated for removal in a future major release."
        ),
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
             '= no reasoning; a level enables it. Anthropic adaptive thinking, '
             'OpenAI reasoning_effort, or enable_thinking on the local backends '
             '-- those have no levels, so any level simply means on. Either way '
             'only the final text is written to --output; the reasoning stays in '
             'its own response field. Expect it to be several times slower.'
    )

    parser.add_argument(
        '--resume', dest='resume', action='store_true', default=True,
        help='With --input-dir, skip plants whose output file already exists '
             'without re-billing them (default)'
    )

    parser.add_argument(
        '--no-resume', dest='resume', action='store_false',
        help='Re-run every plant even if its output file exists'
    )

    parser.set_defaults(func=analyze_command)