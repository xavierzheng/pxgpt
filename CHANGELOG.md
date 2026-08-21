# Changelog

## New features

- **Local Stage 3: `pxgpt schema --shard-dir --input-dir` runs a whole dataset.**
  The batch stages need the providers' Batch APIs, which no local server offers, so
  until now a self-hosted Stage 3 was not possible at all. `schema` gained a second
  axis: `--schema` (one schema) or `--shard-dir` (a whole shard set, one request per
  shard, merged into one record per plant), crossed with `--input-folder` (one plant)
  or `--input-dir` (a tree, one subfolder per plant — the same meaning the batch
  stages give it). `--output` becomes a directory whenever more than one file can
  come out, and the sharded path reuses `merge_sharded_results` and the
  `_partial/<line_id>__<shard_id>.json` store, so its merge rule, gap rule and
  recovery behaviour cannot drift from the cloud paths.
  - **Resume covers plants as well as shards**: a 277-plant run killed at plant 200
    restarts without re-issuing the first 199, and a plant missing three shards
    re-runs only those three.
  - Single-plant mode prints the full per-shard table (used as a cheap rehearsal
    before committing GPU hours); multi-plant mode prints one line per plant.
- **Native structured output on every OpenAI-wire backend.**
  `OpenAICompatProvider` previously had one schema path: paste the schema into the
  system prompt as prose and hope. Pointed at vLLM that means the model reads a
  *description* of a schema and then generates freely — survivable for one request,
  not for thousands. It now sends `response_format {"type": "json_schema", …,
  "strict": true}`, i.e. real constrained decoding, with the schema passed verbatim
  (the frozen shard schemas already carry `additionalProperties: false` and full
  `required` lists, which grammar backends take as-is). The legacy path still exists
  in the provider but no command selects it, and the two never combine — with a
  schema constraint in force the system prompt is left byte-identical to what the
  other providers see, so runs stay comparable. **A backend that rejects
  `response_format` raises**; there is deliberately no fallback, because a silent
  downgrade yields output that looks fine and is completely unconstrained.
- **`--image-transport {base64,file}` on `schema` and `analyze`.** `base64` (default,
  unchanged) embeds the bytes in every request. `file` sends `file://` URIs so a
  local server reads the images off its own mount — a plant line costs a few hundred
  bytes of paths instead of megabytes. This is also the only transport that exercises
  the failure modes that matter locally (mount path differing inside and outside the
  container, `--allowed-local-media-path` not covering the tree), so a base64 smoke
  test proves nothing about whether the real run will start.
- **Concurrent shard dispatch, measured rather than guessed.** A plant's shards share
  a prefix of system prompt plus every image, and only the first request to arrive
  pays to build it. Each plant therefore sends **one shard alone**, waits, then fans
  the remainder onto the warm prefix; with two plants in flight one's cold prefill
  overlaps another's warm group. Measured on 4 plants of `03_mature_v2`, same plants
  both ways on a restarted container: **111.8 s/plant serial → 60.6 s/plant, 1.85x**,
  warm shards at 96.8 % prefix-cache hit.
  - `--concurrency` (default 8) caps requests *within* one plant. It is a
    hardware-pressure limit, **not** `n_shards - 1`: effective width is
    `min(--concurrency, n_shards - 1)`, so a 30-shard set still fans out 8, never 29.
  - `--pipeline-depth` (default 2, refused above 2) caps plants in flight. A
    **separate** global ceiling of `--concurrency + 1` requests applies across all of
    them; depth alone would not bound this, since depth 2 × width 8 could put two
    plants in their warm phase at once — 16 concurrent requests, exactly the server's
    `MAX_NUM_SEQS`. The gate reports its own observed peak.
  - `--mem-floor-gib` (default 5) refuses to *start* a new plant below that host
    `MemAvailable` and recovers automatically, because exhausting a unified memory
    pool hard-locks the machine rather than raising an error.
  - `--limit N` runs the first N plants, for timing on new hardware.
  - The head is *the first pending shard*, not `shard_01`: a partial on disk says
    nothing about the server's KV cache, and a restarted container has an empty one.
    A failed head still lets the rest fan out, since `length`, `reasoning leak` and
    `parse error` all happen after the prefill.
  - **Circuit breaker**: three consecutive plants with no usable shard aborts the
    run. A constant, deliberately not a flag — a tunable safety net gets tuned off.
  - **Merge on abort**: the breaker and `Ctrl-C` both finish the plants in flight,
    merge what completed, and exit. Losing 200 plants' merged records to an
    interrupt would mean re-running a command just to read work already paid for.
  - **Run summary** reports status counts, the real `completion_tokens` distribution,
    cold vs warm cache-hit means, `MemAvailable` low-water, guard trips and peak
    in-flight requests — the numbers needed to decide whether the defaults suit the
    next machine.
- **Cache-hit canary.** Warm shards averaging under 50 % prefix-cache hit print a
  warning naming the likely causes. This is the only immediate signal the prefix
  cache has stopped working, and it needs to be seen at plant 3 rather than plant 277.
  Its one known blind spot is documented next to the threshold: a client that sends
  `mm_processor_kwargs` on *every* request is internally consistent, so its warm
  shards still report 97-100 % while each plant's first shard silently re-pays a full
  prefill.
- **`analyze --effort` now works on the local backends.** Any level switches the chat
  template's `enable_thinking` on (those models have no reasoning *levels*, so any
  level means on). Only the final answer is written to `--output`: the server's
  reasoning parser keeps the thinking in its own response field, which pxGPT never
  saves. Stage 3 stays pinned to thinking off — the shard schemas already require a
  `rationale`, so reasoning would restate it at several times the cost, and the
  setting has to be identical across a whole run for the results to compare.
- **`TOP_P` / `TOP_K` config knobs** (defaults 0.95 / 64, the Gemma 4 checkpoint's
  own values), sent on every OpenAI-wire request alongside `temperature`. The local
  server pins the same values itself; the duplication is deliberate, because only
  what the client sends appears in the request record a paper has to cite.

- **`phenotype-batch --dispatch sequential` is now crash-safe, resumable and
  live-logging.** A sharded sequential run is ~`plants × shards` synchronous
  calls (hundreds to thousands, many hours); previously every result lived only
  in RAM until the whole loop finished, so a SLURM wall-time kill / node crash /
  OOM lost *all* completed work, and stdout stayed empty until exit.
  - **Incremental persistence**: each shard's parsed JSON is written to
    `<output>/_partial/<line_id>__<shard_id>.json` the moment it returns, and
    because requests are plant-contiguous a plant's final merged
    `<line_id>.json` is written as soon as its last shard is attempted. A
    `<output>/_partial/progress.jsonl` logs one line per completed call.
  - **Resume** (`--resume` / `--no-resume`, default on): on startup any shard
    with a valid partial on disk is skipped rather than re-billed; the run
    continues where it stopped. The number of skipped calls is reported, and
    token totals count only the calls actually made this run. A failed or
    unparseable call writes no partial, so it is retried on the next run.
  - **Bounded in-run retry**: transient errors (429 / 5xx / Anthropic's 529
    "Overloaded" / connection blips) are retried up to 3× with exponential
    backoff before a shard is given up — previously the client's `max_retries=0`
    meant a single overload blip permanently dropped a shard. A `400` such as
    "Grammar compilation timed out" is *not* retried in-run (it is a schema-size
    error, deferred to resume / re-sharding).
  - **Live stdout**: the CLI now line-buffers stdout (and progress lines flush),
    so SLURM logs update as the run proceeds without needing `PYTHONUNBUFFERED`.
  - A clean uninterrupted run produces the same `<line_id>.json` /
    `<line_id>.gaps.json` files as before (plus the `_partial/` dir alongside).
    Batch mode and single-schema mode are unchanged.
- **`json-to-table` command**: flattens Stage 3 per-plant
  `Result_Stage3/<cultivar_id>.json` files into one row-per-plant, analysis-ready
  table (`pxgpt/core/json2table.py`). Trait metadata (`scale_type`/`unit`/ordinal
  level labels) is read from the master schema (authoritative), falling back to
  the shard schemas for any trait master doesn't cover (logged as a warning).
  Nominal traits stay plain strings in both outputs (never a category/factor);
  quantitative traits become numeric `<trait>_<unit>` columns (unit sanitized,
  e.g. `m²` → `m2`); ordinal traits are reconstructed from their integer level
  code into the schema label — a plain string in the CSV, an **ordered**
  `pandas.Categorical` over the full schema-defined level set in the feather
  file, so R's `arrow::read_feather()` reads them as ordered factors. Missing
  traits and the `not_assessable` sentinel become real NA in every column
  (never a spurious category level). The column set is the union of every
  trait seen across all files, in a deterministic order (master schema order,
  then shard-fallback traits, then any unknown traits). Writes both
  `<prefix>.csv` and `<prefix>.feather` (Arrow IPC v2). Adds `pandas` and
  `pyarrow` to `requirements.txt`.
- **`json-to-table` column-name collision handling.** A column name is just
  the trait's leaf key (plus `_<unit>`), which silently overwrote data if the
  master schema ever assessed the same leaf key under two organ groups (e.g.
  `length` under both `leaf` and `petal`). Each trait's full dotted source
  path (`group.trait`, or deeper) is now tracked so collisions on the *final*
  name (post unit-suffix) can be detected and resolved instead of one column
  silently clobbering the other. New `--on-collision {error,prefix_collided,
  prefix_all}` (default `error`): `error` writes no files and prints a
  ready-to-fill `--rename-map` template listing every clash; `prefix_collided`
  auto-prefixes only the clashing columns with the minimal group-path prefix
  needed to disambiguate (auto-deepening past one level if still ambiguous);
  `prefix_all` prefixes every column with its full path regardless of
  collisions. New `--rename-map FILE` (JSON, keyed by dotted path, applied
  before `--on-collision`) lets the user hand-pick names for specific clashing
  columns verbatim (no unit re-appended). Traits sharing a leaf key but with
  different units (e.g. `stem.length` cm vs `hair.length` mm) are correctly
  treated as distinct and never flagged. A final global-uniqueness check
  always runs regardless of mode, so a `--rename-map` that itself introduces a
  duplicate is caught rather than reaching the CSV/feather output.
- **Stage 3 schema sharding (fixes "compiled grammar is too large").** The full
  Stage 3 structured-output schema (13 organ groups / 46 traits) exceeds the
  Anthropic structured-outputs internal grammar-size limit and every request
  errors with `invalid_request_error: The compiled grammar is too large`. The
  schema is now **sharded by organ group**, bin-packed to a configurable
  grammar-cost budget, so each call carries a small, compilable schema; the
  per-shard `{rationale, value}` outputs are **merged back into one record per
  plant**.
  - **`shard-schema` command** (`pxgpt/core/shard_builder.py`): generates the
    shard set from a master schema — under `<shard-dir>/`, one
    `shard_NN.schema.json` + `shard_NN.prompt.md` per shard, a shared
    `shards_system.md` (the invariant preamble → cached system block), and
    `shards_manifest.json`. Quantitative `value` is `{"type":"string"}` (parsed
    downstream) **not** `anyOf` — union types blow up the grammar. Args:
    `--master`, `--shard-dir`, `--shard-budget` (default 40), `--combined`. The
    standalone `build_stage3.py` in the analysis tree is now a thin shim over
    this module (single source of truth), and the auto-reshard runs it
    **in-process** (no subprocess).
  - `phenotype-batch` gains a **sharded mode** (`--shard-dir`): builds one
    request per *(plant × shard)* with a cached shared system block. Images remain
    ordinary input and stay before the per-shard text prompt, avoiding repeated
    image cache writes when each shard changes the Structured Outputs schema.
    `--dispatch {batch,sequential}` (default `batch`) selects one Message Batch
    for everything vs. near-synchronous per-plant calls. A **pre-flight live
    compile check** verifies each shard schema compiles and **auto-reshards** at a
    smaller budget (re-running `build_stage3.py`) if one still trips the limit. In
    sharded mode `--schema`/`--system-prompt`/`--prompt` are optional (taken from
    the shard set). `--master-schema` overrides the manifest's master path used
    for merge validation.
  - `fetch-results` handles the new `phenotype_sharded` checkpoint stage:
    demultiplexes `custom_id = "<line>__<shard>"`, merges, parses quantitative
    strings → numbers, validates coverage against the master schema, and writes
    one `{line_id}.json` per plant plus `{line_id}.gaps.json` for any missing
    traits / shard errors. Cache-creation vs cache-read tokens are logged.

- **`extract-report` command**: backward-compatible extractor for the legacy
  `<think>...</think><report>...</report>` chain-of-thought prompt convention.
  Keeps only the `<report>` body (discards `<think>`); auto-closes truncated
  tags. Handles a single-response file **and** the grouped multi-cultivar
  `describe-batch` output (one `<report>` per `### <id>` section) via
  `--mode {auto,grouped,single}`. The standalone `extract_report_tags.py` stays
  available for the simple single-file case. Use this only with the
  chain-of-thought prompt path; native reasoning (`--effort`) needs no extraction.
- **`DESCRIBE_EFFORT` reasoning knob for Stage 1**: `describe-batch` now accepts
  `--effort {off,low,medium,high,xhigh,max}` (and the `DESCRIBE_EFFORT` env),
  enabling Anthropic adaptive thinking for the description stage. **Default off**
  — Stage 1 keeps its original behavior (no reasoning, temperature sent). When
  effort is set, the temperature guard omits temperature and native thinking
  blocks are stripped from the saved description.
- **`analyze` / `schema` support more backends**: `lmstudio` and `vllm` are now
  first-class `--provider` values alongside `openai`, `ollama`, `google`. LM
  Studio and vLLM route through LiteLLM's OpenAI-compatible path
  (`openai/<model>` + their own base URL), each with dedicated env vars
  (`LMSTUDIO_BASE_URL`/`LMSTUDIO_MODEL`/`LMSTUDIO_API_KEY`,
  `VLLM_BASE_URL`/`VLLM_MODEL`/`VLLM_API_KEY`). `api_base`/`api_key` are now
  passed per request instead of via LiteLLM globals (no cross-provider clash),
  `drop_params=True` is set for cross-backend robustness, OpenAI reasoning
  models (gpt-5/o-series) omit a custom temperature, and Google routes via
  `gemini/<model>`. vLLM requires `VLLM_MODEL` (clear error otherwise).
- **`--effort` reasoning control for sync commands**: `analyze` and `schema`
  accept `--effort {off,low,medium,high,xhigh,max}` (Anthropic adaptive thinking).
  `analyze` gains optional reasoning (new `ANALYZE_EFFORT` env, default off);
  `schema`'s flag overrides `STAGE3_EFFORT`. Non-anthropic providers ignore it.
  Config gains `Config.build_output_config(effort, schema)`.
- **OpenAI Batch API stages**: new `describe-batch-openai` (Stage 1) and
  `phenotype-batch-openai` (Stage 3) commands, mirroring the Anthropic batch
  commands on the OpenAI Batch API using the **Responses** endpoint
  (`/v1/responses` JSONL). The Responses API is required because images can only
  be referenced by Files-API `file_id` there (Chat Completions cannot reference
- **OpenAI Batch API stages**: new `describe-batch-openai` (Stage 1) and
  `phenotype-batch-openai` (Stage 3) commands, mirroring the Anthropic batch
  commands on the OpenAI Batch API using the **Responses** endpoint
  (`/v1/responses` JSONL). The Responses API is required because images can only
  be referenced by Files-API `file_id` there (Chat Completions cannot reference
  uploaded images). Images are uploaded once via the OpenAI Files API
  (`purpose="vision"`) and reused by `file_id` through a separate manifest
  (`openai_file_manifest.json`); the same `--no-files-api` / `USE_FILES_API=false`
  toggle embeds them inline as base64. Stage 3 uses OpenAI strict structured
  outputs (`text.format` json_schema with `strict: true`, all properties
  required). New env vars:
  `OPENAI_REASONING_EFFORT` (gpt-5/o-series only) and
  `OPENAI_BATCH_COMPLETION_WINDOW` (default `24h`).
- **`fetch-results` is provider-aware**: dispatches on the checkpoint
  `provider` field (`anthropic` or `openai`); pre-existing checkpoints without
  the field default to `anthropic`.
- **`cleanup-files` command**: delete Files-API uploads recorded in a manifest
  (provider auto-detected) and, for OpenAI, the batch input/output/error files
  referenced by `--checkpoint`. Supports `--dry-run`; already-deleted files
  (404) count as deleted; the manifest is pruned as files are removed. OpenAI
  bills for stored files, so clean up after fetching results. Backed by a new
  `delete_all()` method on both `FilesManager` and `OpenAIFilesManager`.
- **Optional Files API for batch stages**: `describe-batch` and `phenotype-batch`
  no longer *require* the Files API. The Files API is still **on by default**
  (upload once, reuse `file_id`s). Pass `--no-files-api` (or set
  `USE_FILES_API=false` in `.env`) to embed each image inline as base64 in the
  request instead; the `files-api-2025-04-14` beta header and the manifest are
  skipped in that mode.

## Changed

- **`schema` no longer uses the legacy system-prompt path for any provider**, and the
  user prompt no longer needs to ask for JSON-only output. `README.md`,
  `user_manual.md` and `ops/local-vllm/README_vllm.md` all said otherwise in several
  places and have been corrected.
- **`--max-tokens` defaults to 2048 in `--shard-dir` mode** (`MAX_TOKENS` otherwise).
  A grammar constrains the *shape* of the output, not its length, so a runaway
  `rationale` can hit the cap mid-object; that is not a partial result. Validated
  against the complete Sonnet-5 reference run — all 1420 shard answers of
  `02_mature_v1` re-tokenized with the served model's own tokenizer: p50 419, p90 596,
  max **832**, **0 over 2048**. Sonnet-5 is 1.36x more verbose than Gemma on identical
  shards, so that is a conservative bound.
- **An image folder containing no images is now an error** (see Fixed).
- **`--provider` accepts `ollama`, `lmstudio` and `vllm`**; the long-removed `google`
  choice is gone from the docs that still advertised it.
- `CLAUDE.md` and `HANDOFF.md` are no longer tracked. Both are session hand-off notes
  for this checkout, not part of the package.
- The run summary no longer prints a speed-up ratio against a fixed constant. Per-plant
  cost scales with photo count and shard count, so a baseline from another dataset is
  not a valid divisor — it is a source of wrong numbers, not a convenience.

- **Stage 1 (`describe-batch`) prompts split by growth stage.** `prompts/describe_plant.txt`
  is replaced by `prompts/describe_plant_mature.txt` (10×10×6.5 cm rockwool cube) and
  `prompts/describe_plant_seedling.txt` (2.5 cm cube) — same morphology-description
  instructions, growth-stage-specific rockwool dimensions. `prompts/phenotyping_system.txt`
  is renamed `prompts/describe_plant_system.txt` (content unchanged).
- **Stage 3 system prompt rewritten for native structured output.**
  `prompts/phenotyping_system_schema.txt` (a legacy "return this JSON schema verbatim"
  instruction from the pre-structured-output era) is replaced by two purpose-built
  prompts: `prompts/phenotype_schema_system_template.txt` (per-plant scoring, mature
  growth stage — rockwool dimensions left as placeholders to fill in) and
  `prompts/phenotype_schema_system_seedling.txt` (per-*cultivar* scoring across a group
  of individuals, fixed 2.5 cm seedling cube). Both specify the `rationale`-then-`value`
  output order, require citing which image(s) support a judgment, and add an explicit
  absence-vs-`not_assessable` rule (a well-supported "no such structure present" is a
  valid value, distinct from "cannot be scored from these images").
- **`generate_master_schema_v2.txt` promoted to `generate_master_schema.txt`** (drops
  the `_v2` suffix; content otherwise unchanged apart from a generic placeholder for
  the describe-output file name).
- Legacy prompt/schema versions (`extract_traits.txt`, `phenotype_schema.json`,
  `phenotyping_system_schema.txt`) archived under `prompts/old_v0.1.0/` instead of
  being deleted outright.
- **`user_manual.md` — master-schema generation prompt hardened**:
  - Documents the mandatory top-level container shape: `trait_groups` must be a JSON
    *object* keyed by group name (not an array, not named `groups`), each value
    `{"description", "traits"}`.
  - Nominal trait `values` are now an array of `{"value", "definition"}` objects (a
    purely visual, self-contained definition shown verbatim to the downstream scorer)
    instead of a bare array of category strings; population/frequency language
    (`"most"`, `"rare"`, cultivar ids, support counts) is banned from these
    definitions and must go in `design_note` instead.
  - Updates the format anchor example accordingly and fixes a couple of typos/spacing.

- **`phenotype-batch --input-dir` is now optional with the Files API.** Stage 3
  can reuse the images already uploaded by `describe-batch` directly from
  `--manifest`: when `--input-dir` is omitted, the plant lines and their
  `file_id`s are reconstructed from the manifest (grouping each uploaded image
  path by its parent-directory name, the Stage 1 `custom_id`), so the original
  image tree need not be present on disk and nothing is re-uploaded. Pass
  `--input-dir` to additionally upload images added since Stage 1.
  `--input-dir` is still **required** with `--no-files-api`, since inline base64
  mode must read the image bytes from disk.

- **Effort env vars accept `off`/`none`** (in addition to blank) as the
  "no reasoning" value, so they match the `--effort off` flag. Across
  `STAGE3_EFFORT`, `DESCRIBE_EFFORT`, `ANALYZE_EFFORT`, `OPENAI_REASONING_EFFORT`:
  **default = off = none = no reasoning + temperature is sent**; a level
  (`low`…`max`) enables reasoning.
- **Reasoning is now OFF by default everywhere.** `STAGE3_EFFORT` default changed
  from `medium` → `""` (empty). Stage 3 (`phenotype-batch`) and the `schema`
  command now run **without reasoning and send `temperature`** by default;
  structured output (`output_config.format`) is unaffected. Set `STAGE3_EFFORT`
  (or pass `--effort`) to opt back into adaptive thinking.

## Fixed

- **An image folder with no images produced a text-only request instead of an error.**
  Pointing `--input-folder` at the *tree* of plant folders rather than at one plant
  sent zero image blocks, no warning: the model answered the prompt from nothing,
  every shard still validated against its schema, and the run looked completely
  healthy while the trait values were invented. `list_images` now raises, and names
  the subdirectories it found when the folder looks like a plant tree — the mistake
  that actually happens, given `--input-folder` takes one plant and `--input-dir`
  takes the tree.
- **A truncated response is now a failed shard, not a partial result.**
  `finish_reason == "length"` raises instead of returning a string that cannot be
  parsed.
- **Leaked reasoning now fails the shard.** `enable_thinking: false` is sent
  explicitly on every request rather than left to the chat template's default (a
  default does not appear in the request record and is owned by somebody else), and a
  non-empty reasoning field on the way back raises, naming the field and its first 200
  characters. Both spellings are checked — `reasoning` and `reasoning_content`;
  neither is modelled by the OpenAI SDK. Nothing is stripped: silently cleaning it up
  would turn a configuration fault into an invisible data-cleaning step. Confirmed
  live that vLLM 0.24 emits `reasoning`, so the assertion cannot sit silent.
- **`mm_processor_kwargs` and `seed` can no longer be sent.** `extra_body` is
  assembled in exactly one function, which is what makes the prohibition checkable
  (with tests asserting the keys appear in no executable line). Neither would error:
  the first puts the images in a different prefix-cache namespace — re-confirmed on
  vLLM 0.24, a byte-identical cached prompt re-paid a 50.7 s prefill at 2.5 % hit —
  and the second collapses to zero the run-to-run variance the consistency study
  exists to measure.
- **`Ctrl-C` did not work during a sharded run.** `ThreadPoolExecutor.__exit__` calls
  `shutdown(wait=True)`, so a real `SIGINT` either stalled for minutes or killed the
  process before the merge. The executor's lifetime is now owned explicitly: cancel
  what has not started, finish what is in flight, merge, exit.
- **A fully resumed run tripped the circuit breaker.** Every shard cached meant zero
  *fresh* successes per plant, which read as three consecutive dead plants. A plant
  now counts as productive on any usable shard, cached or fresh.
- **The progress line crashed on a plant whose warm shards all failed** — no usage to
  average, and `None` reached a format string.
- Image discovery is shared by both transports through one sorted,
  `IMAGE_EXTENSIONS`-filtered helper, so the order a plant's photos are sent in cannot
  differ between shards. A differing order misses the prefix cache from the first
  changed block onward.


- **`phenotype-batch --dispatch batch` gaps are now recoverable.** A batch
  request that errored (typically a transient
  `overloaded_error: File storage is temporarily unavailable`) is terminal inside
  the Batch API, and the old `fetch-results` overwrote `<line_id>.json` from
  scratch on every fetch without persisting per-shard results — so a momentary
  Files-API blip became a permanent gap that `--resume` (sequential-only) could
  not touch. `write_phenotype_sharded_results` now shares the sequential
  dispatch's `<output>/_partial/<line_id>__<shard_id>.json` store: it adopts
  partials already on disk, persists each freshly-succeeded shard, merges the
  **union** of prior partials + this batch, and only writes `<line_id>.gaps.json`
  for traits still missing (removing a stale gaps file once filled). This makes
  `fetch-results` idempotent and lets a batch's failed shards be recovered with a
  short `--dispatch sequential` resume to the same `--output` — re-issuing only
  the missing shards with in-run transient retry. See
  `dispatch_batch_vs_sequential.md` → "Recovering failed shards from a batch".
  (Shared atomic writer `batch_utils.write_json_atomic`; the sequential path's
  private copy was removed.)

- **Image uploads now retry transient gateway errors.** A Cloudflare `502 Bad
  Gateway` (or `503`/`504`/`429`/connection/timeout) during a Files-API upload
  no longer aborts the run — `FilesManager` / `OpenAIFilesManager` retry up to
  5 times with exponential backoff + jitter, reopening the file each attempt.
  Non-transient errors (e.g. `400`) still fail fast. Already-uploaded images are
  skipped via the manifest, so reruns were always safe; this avoids needing one.

## v0.3.0 — 2026-06-04

### Breaking changes
- **Model**: default changed from `claude-3-7-sonnet-20250219` → `claude-sonnet-4-6`
  (set `ANTHROPIC_MODEL` in `.env` to override).
- **`schema` command**: Anthropic path now uses native structured output
  (`output_config.format`) instead of embedding the schema in the system
  prompt.  Output is raw JSON; `extract_report_tags.py` is no longer needed
  for Anthropic runs.

### New features
- **Stage 1 — `describe-batch`**: upload images once via the Files API
  (persistent manifest), submit a Message Batch for rich descriptions grouped
  one section per plant line/cultivar, save a checkpoint for later retrieval.
- **Stage 3 — `phenotype-batch`**: reuse the Stage 1 manifest, submit a
  Message Batch with `output_config.format` (native structured output) and
  `output_config.effort` (adaptive thinking), write one `.json` per plant line.
- **`fetch-results`**: retrieve and write batch results from a checkpoint
  file (works for both stages).
- **`normalize-schema`**: add `additionalProperties: false` and an empty
  `required` array to every object node; strip unsupported `format` and
  `$schema` keywords; write the result back to disk.
- **Files API (`core/files_manager.py`)**: concurrent uploads
  (ThreadPoolExecutor), crash-safe manifest (written after every upload).
- **`core/batch_utils.py`**: centralised temperature guard, text-content
  extractor (skips thinking blocks), shared poll + result-writer helpers.

### Migration from v0.2

1. Update `.env`:
   ```
   ANTHROPIC_MODEL=claude-sonnet-4-6
   STAGE1_MAX_TOKENS=16384
   STAGE3_MAX_TOKENS=16384
   STAGE3_EFFORT=medium
   BATCH_300K_OUTPUT=false
   UPLOAD_CONCURRENCY=10
   ```

2. Normalize your schema before the first Stage 3 run:
   ```bash
   pxgpt normalize-schema --schema prompts/phenotype_schema.json
   ```

3. Stage 1 run:
   ```bash
   pxgpt describe-batch \
     --input-dir ./images \
     --output descriptions.txt \
     --system-prompt prompts/phenotyping_system.txt \
     --prompt prompts/describe_plant.txt
   # → prints batch ID and saves checkpoint_<id>.json
   ```

4. Stage 3 run (after Stage 1 images are uploaded):
   ```bash
   pxgpt phenotype-batch \
     --input-dir ./images \
     --schema prompts/phenotype_schema.json \
     --output phenotypes/ \
     --system-prompt prompts/phenotyping_system_schema.txt \
     --prompt prompts/extract_traits.txt
   ```

5. Retrieve results when the batch completes:
   ```bash
   pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json
   ```

---

## v0.2.2

- Add `Example_master_schema.tsv`

## v0.2.1

- Add `.gitignore`

## v0.2.0

- Add `user_manual.md`

## v0.1.0

- Initial release: `analyze` and `schema` commands with multi-provider support
