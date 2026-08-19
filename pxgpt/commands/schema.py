"""Structured JSON analysis (Stage 3, non-batch).

Two axes, chosen independently:

*What the model is asked for* — ``--schema`` (one schema) or ``--shard-dir``
(a whole shard set, one request per shard, merged into one record per plant).

*How many plants* — ``--input-folder`` (one plant; its images sit directly
inside) or ``--input-dir`` (a tree with one subfolder per plant).  The two names
match the batch stages: ``describe-batch`` / ``phenotype-batch`` also take the
tree as ``--input-dir``.

``--shard-dir --input-dir`` is the local production path: without the providers'
Batch APIs, which a local server does not offer, it is the only way to run a
whole dataset through a shard set.

Either way the schema reaches the model as a real decoding constraint, never as
prose: Anthropic gets ``output_config.format``, the OpenAI-wire backends get
``response_format`` ``json_schema``.  The legacy "schema appended to the system
prompt" path still exists in the provider but this command no longer selects it.
"""

import json
import time
from pathlib import Path

from ..core.config import Config
from ..core.image_utils import (
    create_image_content_list,
    create_multi_image_message,
    IMAGE_EXTENSIONS,
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

# Backends whose thinking switch is a chat-template flag rather than a hosted
# reasoning setting.  Stage 3 pins them to thinking OFF: the shard schemas
# already require a `rationale` field, so reasoning text would only restate it
# at several times the cost, and the setting has to stay identical across the
# whole run for the results to be comparable.  `analyze` may turn it on.
LOCAL_BACKENDS = {"ollama", "lmstudio", "vllm"}

# Per-shard output cap.  Observed p90 for a shard answer is 607 completion
# tokens, so 2048 cannot truncate a sane response; what it does cut short is the
# runaway case (a `rationale` string that will not stop), which was measured at
# ~190 s and 8192 tokens and drops to ~50 s under this cap.  Only the sharded
# mode defaults to it — a single whole-master schema answer is far larger, so
# --schema mode keeps MAX_TOKENS.
SHARD_MAX_TOKENS = 2048

# How much of the first good response to echo for eyeballing.
RAW_PREVIEW_CHARS = 2000


def _resolve_effort(args, config, provider_name: str) -> str:
    """Return the reasoning effort for this run; "" means off.

    ``--effort`` overrides ``STAGE3_EFFORT``, and "off" means off.  On the local
    backends the answer is always "" -- see LOCAL_BACKENDS for why Stage 3 keeps
    thinking pinned off there.
    """
    effort = config.stage3_effort if args.effort is None else args.effort
    if effort == "off":
        effort = ""
    if effort and provider_name in LOCAL_BACKENDS:
        print(f"Note: effort {effort!r} ignored — Stage 3 runs with thinking off "
              f"on '{provider_name}'; the shard schemas already require a "
              f"rationale field.  Use `pxgpt analyze --effort` if you want it.")
        return ""
    return effort


def create_provider(provider_name: str, config: Config):
    if provider_name == "anthropic":
        return AnthropicProvider(config)
    elif provider_name in OPENAI_COMPAT_PROVIDERS:
        return OpenAICompatProvider(config, provider_name)
    raise ValueError(f"Unsupported provider: {provider_name}")


def resolve_plants(args):
    """Return the plant folders to run, in a stable order.

    ``--input-folder`` is one plant and is returned as-is.  ``--input-dir`` is a
    tree: every immediate subdirectory that actually holds images, sorted by
    name.  Subdirectories without images are reported and skipped rather than
    failing the whole run, but a tree with no usable plant at all is an error —
    that almost always means the wrong level was given.
    """
    if args.input_folder:
        return [Path(args.input_folder)]

    root = Path(args.input_dir)
    if not root.is_dir():
        raise ValueError(f"--input-dir is not a directory: {root}")

    plants, empty = [], []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in d.iterdir()):
            plants.append(d)
        else:
            empty.append(d.name)

    if empty:
        print(f"Note: {len(empty)} subdirector(y/ies) hold no images and are "
              f"skipped: {', '.join(empty[:5])}"
              f"{', ...' if len(empty) > 5 else ''}")
    if not plants:
        raise ValueError(
            f"No plant folder under {root} holds any images "
            f"({', '.join(sorted(IMAGE_EXTENSIONS))}).  --input-dir wants a tree "
            f"with one subfolder per plant; for a single plant whose images sit "
            f"directly inside, use --input-folder."
        )
    return plants


def schema_command(args):
    config = Config.from_env()
    provider_name = args.provider or config.provider

    if not config.validate_provider(provider_name):
        raise ValueError(
            f"Provider '{provider_name}' is not properly configured.  "
            "Check your API keys."
        )

    # --output is a directory whenever this run can produce more than one file:
    # any sharded run (it owns a _partial/ store beside the merged records) or
    # any multi-plant run (one result per plant).
    out = Path(args.output)
    if (args.shard_dir or args.input_dir) and out.exists() and not out.is_dir():
        why = ("--shard-dir keeps a _partial/ store beside the merged records"
               if args.shard_dir else "--input-dir writes one file per plant")
        print(f"Error: --output must be a DIRECTORY here ({why}), but {out} is "
              f"an existing file.")
        return 1

    try:
        plants = resolve_plants(args)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    if args.shard_dir:
        return _run_sharded(args, config, provider_name, plants)
    return _run_single(args, config, provider_name, plants)


# ---------------------------------------------------------------------------
# --schema : one schema per plant
# ---------------------------------------------------------------------------

def _run_single(args, config, provider_name, plants):
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

    provider = create_provider(provider_name, config)
    print(f"Using provider: {provider.provider_name}")
    print(f"Image transport: {args.image_transport}")

    effort = _resolve_effort(args, config, provider_name)

    try:
        if provider_name == "anthropic":
            schema_dict = load_normalized(args.schema)
            print(f"Structured output: native (output_config.format)")
            print(f"Thinking effort:   {effort or 'off'}")
            print(f"Temperature:       "
                  f"{temperature_guard_status(config.anthropic_model, effort)}")
        else:
            # Native structured output on the OpenAI wire too: response_format
            # json_schema, i.e. constrained decoding.  The schema goes in exactly
            # one place, so the system prompt stays comparable across providers.
            with open(args.schema, encoding="utf-8") as f:
                schema_dict = json.load(f)
            print(f"Structured output: native (response_format json_schema, strict)")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Schema error: {e}")
        return 1

    multi = bool(args.input_dir)
    out = Path(args.output)
    if multi:
        out.mkdir(parents=True, exist_ok=True)
        print(f"Plants:          {len(plants)}")

    failures = 0
    for i, plant in enumerate(plants, 1):
        dest = (out / f"{plant.name}.json") if multi else out
        if multi:
            if args.resume and dest.exists():
                print(f"[{i}/{len(plants)}] {plant.name}  skip (cached)", flush=True)
                continue
            print(f"[{i}/{len(plants)}] {plant.name}", flush=True)

        try:
            messages = create_multi_image_message(
                str(plant), user_prompt, args.image_transport
            )
            if provider_name == "anthropic":
                response = provider.send_request_with_retry(
                    messages=messages,
                    system_prompt=system_prompt,
                    output_config=config.build_output_config(effort, schema=schema_dict),
                )
            else:
                response = provider.send_request_with_retry(
                    messages=messages,
                    system_prompt=system_prompt,
                    output_config=config.build_output_config(effort) if effort else None,
                    json_schema=schema_dict,
                )
        except Exception as e:  # noqa: BLE001
            print(f"Error during schema analysis ({plant.name}): {e}")
            failures += 1
            if not multi:
                return 1
            continue

        write_file_safely(str(dest), response.content, "output")
        if not multi:
            print(f"Results written to: {dest}")

    if multi:
        print(f"\nWrote {len(plants) - failures} of {len(plants)} plant(s) to {out}/")
        if failures:
            print(f"{failures} plant(s) failed; re-run to retry only those.")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# --shard-dir : every shard of a shard set, per plant
# ---------------------------------------------------------------------------

def _run_sharded(args, config, provider_name, plants):
    """Run each plant through every shard of a shard set, sequentially.

    Sequential on purpose, not for simplicity.  The whole economic premise of
    the local run is that a plant's first shard pays the cold prefill and the
    rest ride the prefix cache; running them in order is what makes that visible
    in the reported ``cached_tokens`` instead of something taken on trust.

    Crash-safe by the mechanism the batch stages already use: each shard's parsed
    JSON is written to ``<output>/_partial/<line_id>__<shard_id>.json`` the
    moment it succeeds, so a kill loses at most the shard in flight.  The merged
    per-plant records are written once at the end — a re-run re-reads the
    partials, issues no API call for what is already done, and merges again.
    """
    out = Path(args.output)

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

    provider = create_provider(provider_name, config)
    model = config.get_model(provider_name)
    effort = _resolve_effort(args, config, provider_name)

    print(f"Using provider: {provider.provider_name}")
    print(f"Model:           {model}")
    print(f"Image transport: {args.image_transport}")
    print(f"Shard set:       {len(shards)} shard(s) from {args.shard_dir}")
    print(f"Plants:          {len(plants)}"
          f"{'' if args.input_folder else f' (from {args.input_dir})'}")
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

    single = len(plants) == 1
    fresh = {}
    shard_errors = {}
    first_raw = None
    failed_total = 0
    started = time.time()

    print(f"\n--- Running {len(shards)} shard(s) x {len(plants)} plant(s), "
          f"sequentially ---", flush=True)

    for i, plant in enumerate(plants, 1):
        line_id = plant.name
        shard_errors.setdefault(line_id, [])
        t0 = time.time()

        try:
            image_blocks = create_image_content_list(str(plant), args.image_transport)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(plants)}] {line_id}  image error: {e}", flush=True)
            shard_errors[line_id].append(f"(all shards): {e}")
            failed_total += len(shards)
            continue

        if not single:
            print(f"[{i}/{len(plants)}] {line_id}  ({len(image_blocks)} images)",
                  flush=True)

        rows, raw = _run_plant_shards(
            args, config, provider, provider_name, effort, system_prompt,
            shards, line_id, image_blocks, partial_dir, fresh, shard_errors,
            verbose=single,
        )
        if first_raw is None and raw is not None:
            first_raw = raw

        ok = sum(1 for r in rows if r[1] == "ok")
        cached = sum(1 for r in rows if r[1] == "skip (cached)")
        bad = len(rows) - ok - cached
        failed_total += bad

        if single:
            _print_shard_table(rows)
        else:
            print(f"           {ok} ok, {cached} cached, {bad} failed"
                  f"   [{time.time() - t0:.1f}s]", flush=True)

    if first_raw is not None:
        line_id, shard_id, raw = first_raw
        print(f"\n--- Raw response text, {line_id} {shard_id} "
              f"(first {RAW_PREVIEW_CHARS} chars) ---")
        print("Check by eye for: a ``` code fence, prose around the JSON, or a "
              "cut-off tail.  Leaked reasoning is asserted on, not eyeballed.")
        print(raw[:RAW_PREVIEW_CHARS])
        if len(raw) > RAW_PREVIEW_CHARS:
            print(f"... [{len(raw) - RAW_PREVIEW_CHARS} more chars]")

    print("\n--- Merging ---")
    stats = merge_sharded_results(
        fresh, shard_errors, [p.name for p in plants], master_index, str(out),
        provider.provider_name, model,
    )

    print(f"\nPartial store:  {partial_dir}")
    print(f"Merged records: {out}/<line_id>.json")
    print(f"Shards ok this run: {len(fresh)}   failed: {failed_total}   "
          f"merged files written: {stats.get('written', 0)}   "
          f"elapsed: {time.time() - started:.0f}s")
    return 1 if failed_total else 0


def _run_plant_shards(args, config, provider, provider_name, effort, system_prompt,
                      shards, line_id, image_blocks, partial_dir, fresh,
                      shard_errors, verbose):
    """Run one plant's shards in manifest order.  Returns ``(rows, first_raw)``."""
    rows = []
    first_raw = None

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
                if verbose:
                    print(f"  {shard_id:<12} skip (cached)", flush=True)
                continue

        messages = [{
            "role": "user",
            "content": image_blocks + [{"type": "text", "text": s["prompt"]}],
        }]

        try:
            if provider_name == "anthropic":
                response = provider.send_request_with_retry(
                    messages=messages,
                    system_prompt=system_prompt,
                    output_config=config.build_output_config(
                        effort, schema=normalize_schema(s["schema"])),
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
            _fail(rows, shard_errors, line_id, shard_id, "length", str(e), verbose)
            continue
        except ThinkingLeakError as e:
            _fail(rows, shard_errors, line_id, shard_id, "reasoning leak", str(e), verbose)
            continue
        except Exception as e:  # noqa: BLE001
            _fail(rows, shard_errors, line_id, shard_id, "api error", str(e), verbose)
            continue

        completion = response.usage.output_tokens
        cached = response.usage.cache_read_tokens
        raw = response.content or ""

        try:
            obj = json.loads(strip_code_fence(raw))
        except json.JSONDecodeError as e:
            _fail(rows, shard_errors, line_id, shard_id, "parse error", str(e),
                  verbose, completion, cached)
            continue

        # Persist immediately: a kill after this point loses nothing.
        write_json_atomic(partial_path, obj)
        fresh[custom_id] = obj
        if first_raw is None:
            first_raw = (line_id, shard_id, raw)

        rows.append((shard_id, "ok", completion, cached))
        if verbose:
            print(f"  {shard_id:<12} ok             "
                  f"completion={completion:<6} cached_tokens={cached}", flush=True)

    return rows, first_raw


def _fail(rows, shard_errors, line_id, shard_id, status, detail, verbose,
          completion="-", cached="-"):
    """Record a failed shard: no partial written, run continues.

    Not aborting is the point — a run exists to surface the state of every shard,
    and stopping at the first failure hides the rest.
    """
    shard_errors.setdefault(line_id, []).append(f"{shard_id}: {detail}")
    rows.append((shard_id, status, completion, cached))
    prefix = "  " if verbose else f"           {line_id} "
    print(f"{prefix}{shard_id:<12} {status:<14} {detail}", flush=True)


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

def add_image_source_args(parser):
    """Add the mutually exclusive ``--input-folder`` / ``--input-dir`` pair.

    Shared with ``analyze`` so the two commands cannot drift: one plant or a
    tree, never a guess about which was meant.  The names match the batch
    stages, where ``--input-dir`` is already the tree.
    """
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--input-folder",
        help="ONE plant's folder — the images sit directly inside it.",
    )
    src.add_argument(
        "--input-dir",
        help="A TREE of plant folders — one subfolder per plant, images inside "
             "each.  Same meaning as --input-dir on describe-batch / "
             "phenotype-batch.  Plants run in sorted order and --output becomes "
             "a directory holding one result per plant.",
    )
    return src


def setup_schema_parser(subparsers):
    parser = subparsers.add_parser(
        "schema",
        help="Structured JSON analysis (Stage 3, non-batch)",
        description=(
            "Analyze one plant (--input-folder) or a whole tree of plants "
            "(--input-dir) against a JSON schema (--schema) or a shard set "
            "(--shard-dir).  The schema is always sent as native structured "
            "output: output_config.format for Anthropic, response_format "
            "json_schema for the OpenAI-wire backends."
        ),
    )
    add_image_source_args(parser)
    parser.add_argument(
        "--output", required=True,
        help="Output FILE for a single plant with --schema; otherwise a "
             "DIRECTORY (one result per plant, plus _partial/ when sharded)",
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
        help="Skip work already on disk without re-billing it: shards whose "
             "_partial/<line_id>__<shard_id>.json parses, or plants whose "
             "output file already exists (default)",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Re-run everything even if results already exist",
    )
    parser.add_argument(
        "--effort",
        choices=["off", "low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Reasoning effort (overrides STAGE3_EFFORT). default = off = none "
             "= no reasoning; a level enables it. Anthropic adaptive thinking or "
             "OpenAI reasoning_effort, depending on the provider; whether a "
             "custom temperature is sent when off depends on the model. "
             "Ignored on the local backends, which Stage 3 pins to thinking off.",
    )
    parser.set_defaults(func=schema_command)
