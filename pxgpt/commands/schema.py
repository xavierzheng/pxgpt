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

Where the dispatch defaults come from
------------------------------------
Every number below was measured on THIS machine (GB10, 128 GB unified memory).
None of them is derived from the shard count, and none should be assumed to
travel to other hardware.

===================  =======  ==========================================  ============================================
knob                 default  source                                      on new hardware / dataset
===================  =======  ==========================================  ============================================
``--concurrency``    8        GB10 memory pressure, NOT ``n_shards - 1``   re-measure; before raising it run
                                                                          ``--limit 4`` and watch MemAvailable
``--pipeline-depth`` 2        GB10; depth 3 hit 7.37 GiB against an        re-measure; >2 is refused
                              8 GiB stop line for only 5 % more speed
``--mem-floor-gib``  5        specific to GB10 / 128 GB unified            must be reset
``--max-tokens``     2048     5.4x the p90 of 381 at temperature 0.5       re-confirm from the run summary after a
                                                                          model or temperature change
circuit breaker      3        safety net, not a tuning knob                unchanged
hit-rate warning     50 %     canary threshold                            unchanged
===================  =======  ==========================================  ============================================

Effective fan-out width is ``min(--concurrency, n_shards - 1)`` and the global
request ceiling is ``--concurrency + 1``.  Neither is derived from the shard
count: a 30-shard set still fans out at most ``--concurrency`` requests.
"""

import argparse
import json
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

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

# Per-shard output cap.  Measured p90 for a shard answer is 381 completion tokens
# at TEMPERATURE=0.5 (n=10, one plant of 02_mature_v1 on a clean container); the
# older 607 figure was taken at temperature 1.0.  So 2048 is 5.4x p90 and cannot
# truncate a sane response.  What it does cut short is the runaway case (a
# `rationale` string that will not stop), measured at ~190 s / 8192 tokens and
# dropping to ~50 s under this cap.  The run summary reprints the real
# distribution for the dataset actually run -- re-confirm from there after any
# model or temperature change.  Only the sharded mode defaults to this; a single
# whole-master schema answer is far larger, so --schema mode keeps MAX_TOKENS.
SHARD_MAX_TOKENS = 2048

# Consecutive plants with zero successful shards before the run gives up.  A
# safety net, deliberately NOT a CLI flag: exposing it invites setting it so high
# that it never fires, which is the same as not having it.
CIRCUIT_BREAKER_PLANTS = 3

# Warn when a plant's warm shards average less than this prefix-cache hit rate.
# Healthy is 97-99 %, so 50 % is unambiguous rather than borderline.
WARM_HIT_WARN_PCT = 50.0

# Hard ceiling on plants in flight.  See _pipeline_depth for why.
MAX_PIPELINE_DEPTH = 2

# All-serial reference for one plant, measured on this box, used only to report
# the speed-up actually achieved.
SERIAL_SECONDS_PER_PLANT = 161.6

# How much of the first good response to echo for eyeballing.
RAW_PREVIEW_CHARS = 2000


def _positive_int(value):
    """argparse type: an int >= 1."""
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or greater, got {n}")
    return n


def _pipeline_depth(value):
    """argparse type: 1 or 2 only.

    Refused rather than clamped, so asking for 3 is an error you see instead of
    a setting that silently did not apply.  Depth 3 was measured: it drove host
    MemAvailable to 7.37 GiB against an 8 GiB stop line to buy 5 %, and
    exhausting the unified pool hard locks the machine (power cycle).
    """
    n = int(value)
    if n < 1 or n > MAX_PIPELINE_DEPTH:
        raise argparse.ArgumentTypeError(
            f"--pipeline-depth must be 1 or {MAX_PIPELINE_DEPTH}, got {n}.  "
            f"Depth 3 reached 7.37 GiB MemAvailable against an 8 GiB stop line "
            f"for 5% more speed, and exhausting the unified memory pool hard "
            f"locks this machine."
        )
    return n


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

    if args.limit is not None and args.limit < len(plants):
        print(f"Note: --limit {args.limit} — running the first {args.limit} of "
              f"{len(plants)} plant(s).")
        plants = plants[:args.limit]

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

@dataclass
class ShardResult:
    """One shard's outcome.  Returned by :func:`_send_one`, never shared state.

    ``_send_one`` runs on a worker thread, so it touches nothing outside itself
    and hands everything back here; the main thread is the only writer of
    ``fresh`` / ``shard_errors``.  That is what keeps the concurrent path free of
    locks rather than merely unlikely to race.
    """
    shard_id: str
    status: str                  # ok | parse error | api error | length | reasoning leak
    obj: Optional[dict] = None
    error: Optional[str] = None
    completion_tokens: Any = "-"
    cached_tokens: Any = "-"
    prompt_tokens: Any = "-"
    wall_seconds: float = 0.0
    raw: Optional[str] = None

    @property
    def hit_pct(self) -> Optional[float]:
        """cached_tokens / prompt_tokens, or None when either is unknown."""
        if isinstance(self.prompt_tokens, int) and isinstance(self.cached_tokens, int) \
                and self.prompt_tokens:
            return 100.0 * self.cached_tokens / self.prompt_tokens
        return None


class InflightGate:
    """Global cap on concurrent requests, across every in-flight plant.

    This is the limit that protects host memory, and it does NOT follow from the
    pipeline depth.  Depth 2 with a within-plant width of 8 could otherwise put
    two plants in their warm phase at the same moment — 16 concurrent requests,
    a pressure never measured here and exactly the server's ``MAX_NUM_SEQS``.
    The measured depth-2 shape is one plant's cold prefill overlapping another's
    warm fan-out, so the ceiling is ``concurrency + 1``.

    Tracks its own peak so the invariant can be asserted in a test and reported
    at the end, instead of being taken on trust.
    """

    def __init__(self, limit: int):
        self.limit = limit
        self._sem = threading.Semaphore(limit)
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __enter__(self):
        self._sem.acquire()
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        return self

    def __exit__(self, *exc):
        with self._lock:
            self.current -= 1
        self._sem.release()
        return False


def mem_available_gib() -> Optional[float]:
    """Host ``MemAvailable`` in GiB, or None where /proc/meminfo does not exist.

    Whole-machine number, so it includes whatever else is running.  That makes it
    a conservative signal rather than a precise one, which is the point: the
    guard has to hold even if every measured figure here is optimistic.
    """
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None
    return None


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:  # NaN-safe
        return "--"
    h, rem = divmod(int(seconds), 3600)
    m = rem // 60
    return f"{h}h{m:02d}m" if h else f"{m}m"


def _send_one(ctx, line_id, image_blocks, shard) -> ShardResult:
    """Issue one shard request and return its outcome.  Never raises.

    Writes the parsed JSON to ``_partial/`` the moment it succeeds, so crash
    safety is unchanged by the move to threads: a kill loses at most the requests
    actually in flight.
    """
    shard_id = shard["shard_id"]
    custom_id = sharding.shard_custom_id(line_id, shard_id)
    messages = [{
        "role": "user",
        "content": image_blocks + [{"type": "text", "text": shard["prompt"]}],
    }]
    t0 = time.time()
    try:
        with ctx.gate:
            if ctx.provider_name == "anthropic":
                response = ctx.provider.send_request_with_retry(
                    messages=messages,
                    system_prompt=ctx.system_prompt,
                    output_config=ctx.config.build_output_config(
                        ctx.effort, schema=normalize_schema(shard["schema"])),
                )
            else:
                # Raw shard schema, deliberately un-normalized: xgrammar takes
                # standard JSON Schema and the frozen shards already carry
                # additionalProperties:false and full required lists.
                response = ctx.provider.send_request_with_retry(
                    messages=messages,
                    system_prompt=ctx.system_prompt,
                    output_config=(ctx.config.build_output_config(ctx.effort)
                                   if ctx.effort else None),
                    json_schema=shard["schema"],
                )
    except OutputLengthError as e:
        return _failed(shard_id, "length", e, t0)
    except ThinkingLeakError as e:
        return _failed(shard_id, "reasoning leak", e, t0)
    except Exception as e:  # noqa: BLE001
        return _failed(shard_id, "api error", e, t0)

    wall = time.time() - t0
    raw = response.content or ""
    common = dict(
        completion_tokens=response.usage.output_tokens,
        cached_tokens=response.usage.cache_read_tokens,
        prompt_tokens=response.usage.input_tokens,
        wall_seconds=wall,
    )
    try:
        obj = json.loads(strip_code_fence(raw))
    except json.JSONDecodeError as e:
        return ShardResult(shard_id, "parse error", error=str(e), raw=raw, **common)

    write_json_atomic(ctx.partial_dir / f"{custom_id}.json", obj)
    return ShardResult(shard_id, "ok", obj=obj, raw=raw, **common)


def _failed(shard_id, status, exc, t0) -> ShardResult:
    return ShardResult(shard_id, status, error=str(exc), wall_seconds=time.time() - t0)


def _run_plant_shards(ctx, line_id, image_blocks):
    """Run one plant: the first pending shard alone, then the rest fanned out.

    The lone first request is the whole point.  A plant's shards share a prefix
    of system prompt plus every image, and only the first one to arrive pays to
    build it; firing them all at once would have each miss the cache and pay the
    prefill separately.  So the head is sent and awaited, and only then does the
    remainder fan out onto the warm prefix.

    The head is *the first pending shard*, not ``shard_01``.  On a resumed run
    ``shard_01`` may already be on disk, but a disk partial says nothing about
    the server's KV cache — a restarted container has an empty one.  Whichever
    shard is actually sent first is the one that pays.

    A failed head still lets the rest fan out: ``length``, ``reasoning leak`` and
    ``parse error`` all happen *after* the prefill, so the prefix is cached
    regardless.

    Returns ``(results, cold, warm_results, warm_wall)``.
    """
    pending, results = [], []
    for s in ctx.shards:
        partial_path = ctx.partial_dir / f"{sharding.shard_custom_id(line_id, s['shard_id'])}.json"
        if ctx.resume and partial_path.exists():
            try:
                json.loads(partial_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pending.append(s)      # corrupt partial -> re-run this shard
            else:
                results.append(ShardResult(s["shard_id"], "skip (cached)"))
                if ctx.verbose:
                    print(f"  {s['shard_id']:<12} skip (cached)", flush=True)
        else:
            pending.append(s)

    cold = None
    warm_results, warm_wall = [], 0.0

    if pending:
        head, *rest = pending
        cold = _send_one(ctx, line_id, image_blocks, head)
        results.append(cold)
        _report(ctx, line_id, cold)

        if rest:
            width = min(ctx.concurrency, len(rest))
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=width,
                                    thread_name_prefix="shard") as ex:
                for r in ex.map(lambda s: _send_one(ctx, line_id, image_blocks, s), rest):
                    warm_results.append(r)
                    _report(ctx, line_id, r)
            warm_wall = time.time() - t0
            results.extend(warm_results)

    # Manifest order, so the table is comparable run to run even though the
    # completion order is not.
    order = {s["shard_id"]: i for i, s in enumerate(ctx.shards)}
    results.sort(key=lambda r: order.get(r.shard_id, 1 << 30))
    return results, cold, warm_results, warm_wall


def _report(ctx, line_id, r: ShardResult):
    """Print a shard's outcome as it lands.

    Failures print immediately even in multi-plant mode -- on a run measured in
    hours you need to see them when they happen, not in the summary -- and carry
    both ids because the lines from concurrent plants interleave.
    """
    if r.status == "ok":
        if ctx.verbose:
            print(f"  {r.shard_id:<12} ok             "
                  f"completion={r.completion_tokens:<6} "
                  f"cached_tokens={r.cached_tokens}", flush=True)
        return
    prefix = "  " if ctx.verbose else f"           {line_id} "
    print(f"{prefix}{r.shard_id:<12} {r.status:<14} {r.error}", flush=True)


def _run_sharded(args, config, provider_name, plants):
    """Run every plant through every shard of a shard set.

    Dispatch shape, measured on this GB10 (see the portability table in the
    module docstring): all-serial 161.6 s per plant, lone cold shard then a
    fanned-out remainder 101.6 s, and the same with two plants in flight 75.1 s.
    Depth 3 bought a further 5 % while driving host ``MemAvailable`` down to
    7.37 GiB against an 8 GiB stop line, and exhausting the unified pool hard
    locks the machine -- so depth is capped at 2.

    Merging stays a single pass at the end.  Per-plant merging would re-glob the
    whole ``_partial/`` store for every plant and would race between threads;
    the store itself is the crash-safety mechanism, and an interrupted run still
    merges what finished (see the KeyboardInterrupt path).
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
    # built from the manifest's own all_traits inventory -- so it describes the
    # shards that were actually run.  A separately-pointed master could have
    # moved on since the set was frozen and would order fields against shards
    # that no longer exist.
    master_index = sharding.master_index_from_manifest(manifest)

    config.max_tokens = args.max_tokens if args.max_tokens is not None else SHARD_MAX_TOKENS

    provider = create_provider(provider_name, config)
    model = config.get_model(provider_name)
    effort = _resolve_effort(args, config, provider_name)

    single = len(plants) == 1
    # Width is a CAP on hardware pressure, never derived from the shard count:
    # a 30-shard set must not fan out 29 requests at once.  See the portability
    # table -- 8 came from memory measurements on this box, and it is only a
    # coincidence that a 9-shard set has 8 in its remainder.
    max_width = min(args.concurrency, max(1, len(shards) - 1))
    gate = InflightGate(args.concurrency + 1)
    depth = 1 if single else args.pipeline_depth

    print(f"Using provider: {provider.provider_name}")
    print(f"Model:           {model}")
    print(f"Image transport: {args.image_transport}")
    print(f"Shard set:       {len(shards)} shard(s) from {args.shard_dir}")
    print(f"Plants:          {len(plants)}"
          f"{'' if args.input_folder else f' (from {args.input_dir})'}")
    print(f"Dispatch:        1 cold + up to {max_width} concurrent "
          f"(--concurrency {args.concurrency}), {depth} plant(s) in flight, "
          f"global cap {gate.limit} request(s)")
    print(f"max_tokens:      {config.max_tokens}")
    print(f"timeout:         {config.timeout}s")
    print(f"Resume:          {'on' if args.resume else 'off'}")
    print(f"Memory floor:    {args.mem_floor_gib} GiB")
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

    # Build the client here, on the main thread.  BaseProvider.client is a lazy
    # `if self._client is None` init -- the one piece of mutable state on the
    # send path -- so touching it once before any fan-out removes the race by
    # ordering instead of by adding a lock.
    getattr(provider, "client", None)

    ctx = SimpleNamespace(
        config=config, provider=provider, provider_name=provider_name,
        effort=effort, system_prompt=system_prompt, shards=shards,
        partial_dir=partial_dir, resume=args.resume, gate=gate,
        concurrency=max_width, verbose=single,
    )

    if mem_available_gib() is None:
        print("Note: /proc/meminfo is unavailable, so the memory guard is off "
              "(not an error -- it simply cannot read MemAvailable here).")

    fresh, shard_errors = {}, {}
    first_raw = None
    status_counts = Counter()
    completions = []
    cold_hits, warm_hits = [], []
    mem_low = mem_available_gib()
    guard_trips = 0
    plants_done = 0
    plants_with_no_success = 0
    aborted = None
    started = time.time()

    print(f"\n--- Running {len(shards)} shard(s) x {len(plants)} plant(s) ---",
          flush=True)

    def _finish(idx, plant, results, cold, warm, warm_wall, plant_wall, depth_now):
        """Fold one finished plant into the run totals.  Main thread only."""
        nonlocal first_raw, plants_done, plants_with_no_success, mem_low
        line_id = plant.name
        shard_errors.setdefault(line_id, [])
        for r in results:
            status_counts[r.status] += 1
            if r.status == "ok":
                fresh[sharding.shard_custom_id(line_id, r.shard_id)] = r.obj
                completions.append(r.completion_tokens)
                if first_raw is None:
                    first_raw = (line_id, r.shard_id, r.raw)
            elif r.status != "skip (cached)":
                shard_errors[line_id].append(f"{r.shard_id}: {r.error}")

        if cold is not None and cold.hit_pct is not None:
            cold_hits.append(cold.hit_pct)
        warm_pct = [r.hit_pct for r in warm if r.hit_pct is not None]
        warm_hits.extend(warm_pct)
        warm_mean = sum(warm_pct) / len(warm_pct) if warm_pct else None

        ok = sum(1 for r in results if r.status == "ok")
        cached_n = sum(1 for r in results if r.status == "skip (cached)")
        plants_done += 1
        # The breaker fires on "nothing is working", so a plant counts as
        # productive if it yielded ANY usable shard -- freshly fetched or already
        # on disk.  Counting only fresh successes would make a fully resumed run
        # (every shard cached, zero requests sent) look like a total failure and
        # abort at the third plant.
        plants_with_no_success = 0 if (ok or cached_n) else plants_with_no_success + 1

        mem = mem_available_gib()
        if mem is not None:
            mem_low = mem if mem_low is None else min(mem_low, mem)

        if not single:
            rate = plants_done / max(1e-9, time.time() - started)
            eta = (len(plants) - plants_done) / rate if rate else 0
            if cold is None:
                cold_s = "cold -"
            elif cold.hit_pct is None:
                cold_s = f"cold {cold.wall_seconds:.1f}s (hit n/a)"
            else:
                cold_s = f"cold {cold.wall_seconds:.1f}s (hit {cold.hit_pct:.1f}%)"
            # warm_mean is None when every warm shard failed -- there is no
            # usage to average, but the group still took wall-clock time.
            if not warm:
                warm_s = "warm -"
            elif warm_mean is None:
                warm_s = f"warm {len(warm)}x {warm_wall:.1f}s (hit n/a)"
            else:
                warm_s = (f"warm {len(warm)}x {warm_wall:.1f}s "
                          f"(hit {warm_mean:.1f}%)")
            print(f"[{idx:>4}/{len(plants)}] {line_id}  {ok}/{len(results)} ok"
                  f"{f'  {cached_n} cached' if cached_n else ''}"
                  f"  {cold_s}  {warm_s}  total {plant_wall:.1f}s  depth {depth_now}"
                  f"  ETA {_fmt_eta(eta)}"
                  f"  MemAvail {f'{mem:.1f}G' if mem is not None else 'n/a'}",
                  flush=True)
        else:
            _print_shard_table([(r.shard_id, r.status, r.completion_tokens,
                                 r.cached_tokens) for r in results])

        # The cache-hit canary.  This is the only immediate signal that the
        # prefix cache has stopped working, and it has to be seen at plant 3 and
        # not at plant 267.
        if warm_mean is not None and warm_mean < WARM_HIT_WARN_PCT:
            print(f"  WARNING: {line_id} warm shards averaged only "
                  f"{warm_mean:.1f}% prefix-cache hit (expected >95%).  Check "
                  f"the image order and that no mm_processor_kwargs is being "
                  f"sent.", flush=True)

    def _one_plant(plant):
        """Worker for a whole plant.  Returns everything the main thread folds in."""
        t0 = time.time()
        try:
            blocks = create_image_content_list(str(plant), args.image_transport)
        except Exception as e:  # noqa: BLE001
            return None, None, [], 0.0, time.time() - t0, str(e)
        results, cold, warm, warm_wall = _run_plant_shards(ctx, plant.name, blocks)
        return results, cold, warm, warm_wall, time.time() - t0, None

    queue = list(enumerate(plants, 1))
    inflight = {}
    # Deliberately NOT a `with` block.  ThreadPoolExecutor.__exit__ calls
    # shutdown(wait=True), which on Ctrl-C blocks until every queued plant has
    # also run -- so the interrupt would either be ignored for minutes or kill
    # the process before the merge.  Owning shutdown here lets us cancel what
    # has not started, wait only for what is in flight, and still merge.
    pool = ThreadPoolExecutor(max_workers=max(1, depth), thread_name_prefix="plant")
    try:
        try:
            while queue or inflight:
                    # Launch up to `depth`, but never open a NEW plant while the
                    # host is under the floor: fall back to finishing what is in
                    # flight and recover automatically once memory frees up.
                    while queue and len(inflight) < depth:
                        mem = mem_available_gib()
                        if (mem is not None and inflight
                                and mem < args.mem_floor_gib):
                            guard_trips += 1
                            print(f"  WARNING: MemAvailable {mem:.1f} GiB is "
                                  f"below the {args.mem_floor_gib} GiB floor — "
                                  f"not starting another plant until the "
                                  f"current one finishes.", flush=True)
                            break
                        idx, plant = queue.pop(0)
                        inflight[pool.submit(_one_plant, plant)] = (idx, plant)

                    if not inflight:
                        break
                    done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                    depth_now = len(inflight)
                    for fut in done:
                        idx, plant = inflight.pop(fut)
                        results, cold, warm, warm_wall, plant_wall, img_err = fut.result()
                        if img_err is not None:
                            print(f"[{idx:>4}/{len(plants)}] {plant.name}  "
                                  f"image error: {img_err}", flush=True)
                            shard_errors.setdefault(plant.name, []).append(
                                f"(all shards): {img_err}")
                            status_counts["image error"] += len(shards)
                            plants_done += 1
                            plants_with_no_success += 1
                        else:
                            _finish(idx, plant, results, cold, warm, warm_wall,
                                    plant_wall, depth_now)

                        if plants_with_no_success >= CIRCUIT_BREAKER_PLANTS:
                            aborted = (f"{CIRCUIT_BREAKER_PLANTS} consecutive "
                                       f"plant(s) produced no successful shard")
                            queue.clear()
                    if aborted:
                        break
        except KeyboardInterrupt:
            aborted = "interrupted by Ctrl-C"
            queue.clear()
            print("\n*** Ctrl-C — finishing the plant(s) already in flight, "
                  "then merging. Press Ctrl-C again to drop them. ***",
                  flush=True)
            for fut, (idx, plant) in list(inflight.items()):
                try:
                    r = fut.result()
                except (KeyboardInterrupt, Exception):  # noqa: BLE001
                    continue
                if r[5] is None:
                    _finish(idx, plant, r[0], r[1], r[2], r[3], r[4], len(inflight))
            inflight.clear()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    if aborted:
        # Merge anyway.  Losing 200 plants' merged records to an abort would mean
        # re-running the command just to read work already paid for.
        print(f"\n*** ABORTED: {aborted} — merging the {plants_done} plant(s) "
              f"already finished ***", flush=True)
        last = next((errs[-1] for errs in reversed(list(shard_errors.values())) if errs),
                    None)
        if last:
            print(f"    last error: {last}", flush=True)

    done_plants = [p.name for p in plants[:plants_done]] if aborted else [p.name for p in plants]

    if first_raw is not None and single:
        line_id, shard_id, raw = first_raw
        print(f"\n--- Raw response text, {line_id} {shard_id} "
              f"(first {RAW_PREVIEW_CHARS} chars) ---")
        print("Check by eye for: a ``` code fence, prose around the JSON, or a "
              "cut-off tail.  Leaked reasoning is asserted on, not eyeballed.")
        print(raw[:RAW_PREVIEW_CHARS])
        if len(raw) > RAW_PREVIEW_CHARS:
            print(f"... [{len(raw) - RAW_PREVIEW_CHARS} more chars]")

    print("\n--- Merging ---")
    merge_ids = sorted(set(list(shard_errors) + [sharding.split_custom_id(c)[0]
                                                 for c in fresh])) or done_plants
    stats = merge_sharded_results(
        fresh, shard_errors, merge_ids, master_index, str(out),
        provider.provider_name, model,
    )

    elapsed = time.time() - started
    failed_total = sum(n for s, n in status_counts.items()
                       if s not in ("ok", "skip (cached)"))
    print(f"\nPartial store:  {partial_dir}")
    print(f"Merged records: {out}/<line_id>.json")
    _print_run_summary(plants, plants_done, stats, status_counts, completions,
                       cold_hits, warm_hits, elapsed, mem_low, guard_trips,
                       gate, len(shards), failed_total)
    return 1 if (failed_total or aborted) else 0


def _pct(values, q):
    if not values:
        return "-"
    v = sorted(values)
    i = max(0, min(len(v) - 1, int(round((len(v) - 1) * q))))
    return v[i]


def _print_run_summary(plants, plants_done, stats, status_counts, completions,
                       cold_hits, warm_hits, elapsed, mem_low, guard_trips,
                       gate, n_shards, failed_total):
    """Numbers that decide whether the defaults are right for the next machine."""
    print("\n--- Run summary ---")
    print(f"Plants:            {plants_done} of {len(plants)} attempted; "
          f"{stats.get('written', 0)} merged record(s); "
          f"{stats.get('plants_with_gaps', 0)} with gaps "
          f"({stats.get('total_gaps', 0)} missing trait(s))")
    total_shards = sum(status_counts.values())
    print(f"Shards:            {total_shards} total"
          + (f"   ({', '.join(f'{s}={n}' for s, n in sorted(status_counts.items()))})"
             if status_counts else ""))
    if completions:
        print(f"completion_tokens: p50={_pct(completions, 0.5)}  "
              f"p90={_pct(completions, 0.9)}  max={max(completions)}  "
              f"n={len(completions)}   (vs SHARD_MAX_TOKENS={SHARD_MAX_TOKENS})")
    cold_mean = sum(cold_hits) / len(cold_hits) if cold_hits else None
    warm_mean = sum(warm_hits) / len(warm_hits) if warm_hits else None
    print(f"Prefix-cache hit:  cold shards "
          f"{f'{cold_mean:.1f}%' if cold_mean is not None else '-'}  "
          f"(n={len(cold_hits)})   warm shards "
          f"{f'{warm_mean:.1f}%' if warm_mean is not None else '-'}  "
          f"(n={len(warm_hits)})")
    per_plant = elapsed / plants_done if plants_done else 0
    # Only quote a speed-up when requests were actually issued.  A fully resumed
    # run finishes in milliseconds and would otherwise report a meaningless
    # thousand-fold "speed-up" against the serial reference.
    speedup = ""
    if per_plant > 1.0 and completions:
        speedup = (f"   ({SERIAL_SECONDS_PER_PLANT / per_plant:.2f}x the "
                   f"{SERIAL_SECONDS_PER_PLANT:.1f}s all-serial reference "
                   f"measured on this box)")
    print(f"Wall clock:        {elapsed:.0f}s total, {per_plant:.1f}s per plant{speedup}")
    print(f"Memory:            MemAvailable low-water "
          f"{f'{mem_low:.1f} GiB' if mem_low is not None else 'n/a'}; "
          f"guard withheld a plant {guard_trips} time(s)")
    print(f"Peak in-flight:    {gate.peak} request(s) (global cap {gate.limit})")
    # Ticket #6 is not implemented, so there is no value_status field to count.
    print("value_status:      not available — ticket #6 (value_raw / "
          "value_status) is not implemented, so merged records carry no "
          "per-value status to summarise.")
    if failed_total:
        print(f"\n{failed_total} shard(s) failed.  Re-run the same command; "
              f"--resume skips everything already on disk.")


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
        help="Skip work already on disk without re-billing it (default).  With "
             "--shard-dir this is per SHARD: any shard whose "
             "_partial/<line_id>__<shard_id>.json parses is skipped, and a plant "
             "with some shards missing still re-runs just those.  With --schema "
             "it is per PLANT: a plant whose output file exists is skipped.",
    )
    parser.add_argument(
        "--no-resume", dest="resume", action="store_false",
        help="Re-run everything even if results already exist",
    )
    parser.add_argument(
        "--concurrency", type=_positive_int, default=8,
        help="Upper bound on concurrent requests WITHIN one plant, after its "
             "first (cold) shard has been sent alone.  A hardware-pressure cap "
             "measured on this box, not a function of the shard count: the "
             "effective width is min(--concurrency, n_shards - 1), so a "
             "30-shard set still fans out at most this many.  1 reproduces the "
             "old fully serial behaviour.  (default: 8)",
    )
    parser.add_argument(
        "--pipeline-depth", type=_pipeline_depth, default=2,
        help="How many plants may be in flight at once, so one plant's cold "
             "prefill overlaps another's warm fan-out.  1 disables overlap.  "
             "Capped at 2: depth 3 drove host MemAvailable to 7.37 GiB against "
             "an 8 GiB stop line for 5%% more speed, and exhausting the unified "
             "pool hard locks the machine.  (default: 2)",
    )
    parser.add_argument(
        "--mem-floor-gib", type=float, default=5.0,
        help="Do not START another plant while host MemAvailable is below this "
             "(GiB); finish what is in flight first, then recover "
             "automatically.  Specific to this machine's 128 GB unified "
             "memory — reset it on other hardware.  Ignored where "
             "/proc/meminfo does not exist.  (default: 5)",
    )
    parser.add_argument(
        "--limit", type=_positive_int, default=None,
        help="Process only the first N plants.  For timing runs.",
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
