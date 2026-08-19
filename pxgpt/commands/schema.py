"""Single-request structured JSON analysis (Stage 3, non-batch).

Two modes, one command:

``--schema``
    One folder of images against one schema, written to one output FILE.  This
    is the original behaviour.

``--shard-dir``
    One folder of images against a whole shard set, written to an output
    DIRECTORY laid out exactly like ``phenotype-batch``'s (``_partial/`` store
    plus a merged ``<line_id>.json``).  This is the cheap rehearsal: before
    committing hours of GPU time to 267 plants, run one plant through every
    shard and see whether the frozen shard set actually works on this model.

Either way the schema reaches the model as a real decoding constraint, never as
prose: Anthropic gets ``output_config.format``, the OpenAI-wire backends get
``response_format`` ``json_schema``.  The legacy "schema appended to the system
prompt" path still exists in the provider but this command no longer selects it.
"""

import json
from pathlib import Path

from ..core.config import Config
from ..core.image_utils import (
    create_image_content_list,
    create_multi_image_message,
    IMAGE_TRANSPORTS,
)
from ..core.file_utils import read_file_safely, write_file_safely
from ..core.schema_utils import load_normalized, normalize_schema
from ..core.batch_utils import (
    temperature_guard_status,
    assert_partial_provenance,
    merge_sharded_results,
    strip_code_fence,
    write_json_atomic,
)
from ..core import sharding
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.openai_compat_provider import (
    OpenAICompatProvider,
    OutputLengthError,
    ThinkingLeakError,
)


OPENAI_COMPAT_PROVIDERS = {"openai", "ollama", "lmstudio", "vllm"}

# Per-shard output cap.  Observed p90 for a shard answer is 607 completion
# tokens, so 2048 cannot truncate a sane response; what it does cut short is the
# runaway case (a `rationale` string that will not stop), which was measured at
# ~190 s and 8192 tokens and drops to ~50 s under this cap.  Only the sharded
# mode defaults to it — a single whole-master schema answer is far larger, so
# --schema mode keeps MAX_TOKENS.
SHARD_MAX_TOKENS = 2048

# How much of the first good response to echo for eyeballing.
RAW_PREVIEW_CHARS = 2000


def create_provider(provider_name: str, config: Config):
    if provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name in OPENAI_COMPAT_PROVIDERS:
        return OpenAICompatProvider(config, provider_name)
    raise ValueError(f"Unsupported provider: {provider_name}")


def schema_command(args):
    config = Config.from_env()
    provider_name = args.provider or config.provider

    if not config.validate_provider(provider_name):
        raise ValueError(
            f"Provider '{provider_name}' is not properly configured.  "
            "Check your API keys."
        )

    if args.shard_dir:
        return _run_sharded(args, config, provider_name)
    return _run_single(args, config, provider_name)


# ---------------------------------------------------------------------------
# --schema : one folder, one schema, one output file  (unchanged behaviour)
# ---------------------------------------------------------------------------

def _run_single(args, config, provider_name):
    if not args.system_prompt or not args.prompt:
        print("Error: --system-prompt and --prompt are required with --schema.")
        return 1

    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens

    try:
        system_prompt = read_file_safely(args.system_prompt, "system prompt")
        user_prompt = read_file_safely(args.prompt, "user prompt")
    except (FileNotFoundError, IOError) as e:
        print(f"File error: {e}")
        return 1

    try:
        messages = create_multi_image_message(
            args.input_folder, user_prompt, args.image_transport
        )
    except Exception as e:
        print(f"Error processing images: {e}")
        return 1

    provider = create_provider(provider_name, config)
    print(f"Using provider: {provider.provider_name}")
    print(f"Image transport: {args.image_transport}")

    # --effort overrides STAGE3_EFFORT; "off" disables reasoning.
    effort = config.stage3_effort if args.effort is None else args.effort
    if effort == "off":
        effort = ""

    try:
        if provider_name == "anthropic":
            # Native structured output — schema is NOT in the system prompt
            schema_dict = load_normalized(args.schema)
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
            # Native structured output on the OpenAI wire too: response_format
            # json_schema, i.e. constrained decoding.  The schema goes in exactly
            # one place, so the system prompt stays comparable across providers.
            with open(args.schema, encoding="utf-8") as f:
                schema_dict = json.load(f)
            print(f"Structured output: native (response_format json_schema, strict)")
            # OpenAI reasoning models read the effort off output_config; the local
            # backends ignore it.
            legacy_output_config = config.build_output_config(effort) if effort else None
            response = provider.send_request_with_retry(
                messages=messages,
                system_prompt=system_prompt,
                output_config=legacy_output_config,
                json_schema=schema_dict,
            )
    except Exception as e:
        print(f"Error during schema analysis: {e}")
        return 1

    write_file_safely(args.output, response.content, "output")
    print(f"Results written to: {args.output}")
    return 0


# ---------------------------------------------------------------------------
# --shard-dir : one plant, every shard, an output directory
# ---------------------------------------------------------------------------

def _run_sharded(args, config, provider_name):
    """Run one plant through every shard of a shard set, sequentially.

    Sequential on purpose, not for simplicity.  The whole economic premise of
    the local run is that a plant's first shard pays the cold prefill and the
    remaining eight ride the prefix cache; running them in order is what makes
    that visible in the printed ``cached_tokens`` column instead of something you
    have to take on trust.
    """
    out = Path(args.output)
    if out.exists() and not out.is_dir():
        print(f"Error: --output must be a DIRECTORY in --shard-dir mode "
              f"(it holds _partial/ plus one merged file per plant), but "
              f"{out} is an existing file.")
        return 1

    if args.prompt:
        print("Note: --prompt is ignored in --shard-dir mode; each shard brings "
              "its own prompt.")

    try:
        manifest, shards = sharding.load_shard_set(args.shard_dir)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return 1

    try:
        system_prompt = sharding.load_system_prompt(
            args.shard_dir, manifest, args.system_prompt
        )
    except (FileNotFoundError, IOError) as e:
        print(f"Error: {e}")
        return 1

    # The merge index comes from the manifest, unconditionally.  It carries the
    # same (group_order, group_traits, trait_meta) shape as a master schema,
    # built from the manifest's own all_traits inventory — so it describes the
    # shards that were actually run.  A separately-pointed master could have
    # moved on since the set was frozen and would order fields against shards
    # that no longer exist.
    master_index = sharding.master_index_from_manifest(manifest)

    # Per-shard cap: small enough to bound a runaway rationale, far above any
    # real answer.  See SHARD_MAX_TOKENS.
    config.max_tokens = args.max_tokens if args.max_tokens is not None else SHARD_MAX_TOKENS

    line_id = Path(args.input_folder).name

    try:
        image_blocks = create_image_content_list(
            args.input_folder, args.image_transport
        )
    except Exception as e:
        print(f"Error processing images: {e}")
        return 1

    provider = create_provider(provider_name, config)
    model = config.get_model(provider_name)

    print(f"Using provider: {provider.provider_name}")
    print(f"Model:           {model}")
    print(f"Image transport: {args.image_transport}  ({len(image_blocks)} image(s))")
    print(f"Shard set:       {len(shards)} shard(s) from {args.shard_dir}")
    print(f"Plant line:      {line_id}")
    print(f"max_tokens:      {config.max_tokens}")
    print(f"timeout:         {config.timeout}s")
    print(f"Resume:          {'on' if args.resume else 'off'}")
    print(f"system prompt:   "
          f"{'--system-prompt override' if args.system_prompt else manifest.get('system_file')}")

    partial_dir = out / "_partial"
    out.mkdir(parents=True, exist_ok=True)
    partial_dir.mkdir(parents=True, exist_ok=True)
    try:
        assert_partial_provenance(partial_dir, provider.provider_name, model)
    except RuntimeError as e:
        print(f"Error: {e}")
        return 1

    # --effort overrides STAGE3_EFFORT; "off" disables reasoning.
    effort = config.stage3_effort if args.effort is None else args.effort
    if effort == "off":
        effort = ""

    fresh = {}
    shard_errors = {line_id: []}
    first_raw = None
    rows = []

    print(f"\n--- Running {len(shards)} shard(s) sequentially ---", flush=True)
    for s in shards:
        shard_id = s["shard_id"]
        custom_id = sharding.shard_custom_id(line_id, shard_id)
        partial_path = partial_dir / f"{custom_id}.json"

        if args.resume and partial_path.exists():
            try:
                json.loads(partial_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass  # corrupt partial -> re-run this shard
            else:
                rows.append((shard_id, "skip (cached)", "-", "-"))
                print(f"  {shard_id:<12} skip (cached)", flush=True)
                continue

        messages = [{
            "role": "user",
            "content": image_blocks + [{"type": "text", "text": s["prompt"]}],
        }]

        try:
            if provider_name == "anthropic":
                output_config = config.build_output_config(
                    effort, schema=normalize_schema(s["schema"])
                )
                response = provider.send_request_with_retry(
                    messages=messages,
                    system_prompt=system_prompt,
                    output_config=output_config,
                )
            else:
                # Raw shard schema, deliberately un-normalized: xgrammar takes
                # standard JSON Schema and the frozen shards already carry
                # additionalProperties:false and full required lists.
                response = provider.send_request_with_retry(
                    messages=messages,
                    system_prompt=system_prompt,
                    output_config=config.build_output_config(effort) if effort else None,
                    json_schema=s["schema"],
                )
        except OutputLengthError as e:
            _fail(rows, shard_errors, line_id, shard_id, "length", str(e))
            continue
        except ThinkingLeakError as e:
            _fail(rows, shard_errors, line_id, shard_id, "reasoning leak", str(e))
            continue
        except Exception as e:  # noqa: BLE001
            _fail(rows, shard_errors, line_id, shard_id, "api error", str(e))
            continue

        completion = response.usage.output_tokens
        cached = response.usage.cache_read_tokens
        raw = response.content or ""

        try:
            obj = json.loads(strip_code_fence(raw))
        except json.JSONDecodeError as e:
            _fail(rows, shard_errors, line_id, shard_id, "parse error", str(e),
                  completion, cached)
            continue

        # Persist immediately: a kill after this point loses nothing.
        write_json_atomic(partial_path, obj)
        fresh[custom_id] = obj
        if first_raw is None:
            first_raw = (shard_id, raw)

        rows.append((shard_id, "ok", completion, cached))
        print(f"  {shard_id:<12} ok             "
              f"completion={completion:<6} cached_tokens={cached}", flush=True)

    _print_shard_table(rows)

    if first_raw is not None:
        shard_id, raw = first_raw
        print(f"\n--- Raw response text, {shard_id} (first {RAW_PREVIEW_CHARS} chars) ---")
        print("Check by eye for: a ``` code fence, prose around the JSON, or a "
              "cut-off tail.  Leaked reasoning is asserted on, not eyeballed.")
        print(raw[:RAW_PREVIEW_CHARS])
        if len(raw) > RAW_PREVIEW_CHARS:
            print(f"... [{len(raw) - RAW_PREVIEW_CHARS} more chars]")

    print("\n--- Merging ---")
    stats = merge_sharded_results(
        fresh, shard_errors, [line_id], master_index, str(out),
        provider.provider_name, model,
    )

    print(f"\nPartial store:  {partial_dir}")
    print(f"Merged record:  {out / (line_id + '.json')}")
    gaps = out / f"{line_id}.gaps.json"
    if gaps.exists():
        print(f"Gaps:           {gaps}")
    failed = len(shard_errors[line_id])
    print(f"Shards ok this run: {len(fresh)}   failed: {failed}   "
          f"merged files written: {stats.get('written', 0)}")
    return 1 if failed else 0


def _fail(rows, shard_errors, line_id, shard_id, status, detail,
          completion="-", cached="-"):
    """Record a failed shard: no partial written, run continues.

    Not aborting is the point — this command exists to show the state of EVERY
    shard in one pass, and stopping at the first failure hides the rest.
    """
    shard_errors[line_id].append(f"{shard_id}: {detail}")
    rows.append((shard_id, status, completion, cached))
    print(f"  {shard_id:<12} {status:<14} {detail}", flush=True)


def _print_shard_table(rows):
    print(f"\n{'shard_id':<14}{'status':<16}{'completion':>12}{'cached_tokens':>16}")
    for shard_id, status, completion, cached in rows:
        print(f"{shard_id:<14}{status:<16}{str(completion):>12}{str(cached):>16}")
    print("cached_tokens is prompt_tokens_details.cached_tokens: shard_01 should "
          "be ~0 and the rest 97-99 % of prompt tokens.  If shard_02 onward is "
          "still ~0 the prefix cache is not being reused — check the image order "
          "and that no mm_processor_kwargs is being sent.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_schema_parser(subparsers):
    parser = subparsers.add_parser(
        "schema",
        help="Single-request structured JSON analysis (Stage 3, non-batch)",
        description=(
            "Analyze one folder of images against a JSON schema (--schema) or "
            "against a whole shard set (--shard-dir).  The schema is always sent "
            "as native structured output: output_config.format for Anthropic, "
            "response_format json_schema for the OpenAI-wire backends."
        ),
    )
    parser.add_argument(
        "--input-folder", required=True,
        help="Path to folder containing images (its basename is the line_id "
             "in --shard-dir mode)",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output FILE with --schema; output DIRECTORY with --shard-dir "
             "(holding _partial/ and the merged <line_id>.json)",
    )
    parser.add_argument(
        "--system-prompt",
        help="System prompt file path.  Required with --schema; with "
             "--shard-dir it overrides the shard set's own system prompt.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--schema",
        help="JSON schema sent to the model as the output constraint. "
             "For schemas small enough to compile; use --shard-dir otherwise.",
    )
    target.add_argument(
        "--shard-dir",
        help="Shard set directory. Each shard's own schema is sent to the "
             "model; the merge index comes from the shard manifest.",
    )
    parser.add_argument(
        "--prompt",
        help="User prompt file path.  Required with --schema; ignored with "
             "--shard-dir, where each shard carries its own prompt.",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama", "lmstudio", "vllm"],
        help="LLM provider (overrides config/env)",
    )
    parser.add_argument(
        "--image-transport",
        choices=list(IMAGE_TRANSPORTS),
        default="base64",
        help="How images reach the model.  'base64' embeds the bytes in the "
             "request and works everywhere (default).  'file' sends "
             "file:// URIs and is the recommended path for a local vLLM "
             "server, which must have the image directory mounted at the very "
             "same path.  (default: base64)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None,
        help=f"Output token cap.  Default: {SHARD_MAX_TOKENS} with --shard-dir "
             f"(bounds a runaway rationale; a normal shard answer is ~600 "
             f"tokens), MAX_TOKENS otherwise.  A response that stops at "
             f"finish_reason 'length' counts as a failed shard, not a partial "
             f"result.",
    )
    parser.add_argument(
        "--resume", dest="resume", action="store_true", default=True,
        help="Skip shards whose _partial/<line_id>__<shard_id>.json already "
             "parses, without re-billing them (default)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Re-run every shard even if a partial already exists",
    )
    parser.add_argument(
        "--effort",
        choices=["off", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Reasoning effort (overrides STAGE3_EFFORT). default = off = none "
             "= no reasoning; a level enables it. Anthropic adaptive thinking or "
             "OpenAI reasoning_effort, depending on the provider; whether a "
             "custom temperature is sent when off depends on the model. "
             "Ignored by the local providers, which are always sent "
             "enable_thinking=False and fail the shard if reasoning comes back.",
    )
    parser.set_defaults(func=schema_command)
