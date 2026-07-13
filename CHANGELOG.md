# Changelog

## Unreleased

### Changed
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

### New features
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
    request per *(plant × shard)* with a byte-identical, **cached** system+image
    prefix (`cache_control` on the last image block) and only the small per-shard
    prompt + schema as the uncached suffix, so a plant's first shard pays the
    image cost and the rest hit the cache. `--dispatch {batch,sequential}`
    (default `batch`) selects one Message Batch for everything vs. near-synchronous
    per-plant calls (reliable 5-min image cache). A **pre-flight live compile
    check** verifies each shard schema compiles and **auto-reshards** at a smaller
    budget (re-running `build_stage3.py`) if one still trips the limit. In sharded
    mode `--schema`/`--system-prompt`/`--prompt` are optional (taken from the
    shard set). `--master-schema` overrides the manifest's master path used for
    merge validation.
  - `fetch-results` handles the new `phenotype_sharded` checkpoint stage:
    demultiplexes `custom_id = "<line>__<shard>"`, merges, parses quantitative
    strings → numbers, validates coverage against the master schema, and writes
    one `{line_id}.json` per plant plus `{line_id}.gaps.json` for any missing
    traits / shard errors. Cache-creation vs cache-read tokens are logged.

### Changed
- **`phenotype-batch --input-dir` is now optional with the Files API.** Stage 3
  can reuse the images already uploaded by `describe-batch` directly from
  `--manifest`: when `--input-dir` is omitted, the plant lines and their
  `file_id`s are reconstructed from the manifest (grouping each uploaded image
  path by its parent-directory name, the Stage 1 `custom_id`), so the original
  image tree need not be present on disk and nothing is re-uploaded. Pass
  `--input-dir` to additionally upload images added since Stage 1.
  `--input-dir` is still **required** with `--no-files-api`, since inline base64
  mode must read the image bytes from disk.

### Fixed
- **Image uploads now retry transient gateway errors.** A Cloudflare `502 Bad
  Gateway` (or `503`/`504`/`429`/connection/timeout) during a Files-API upload
  no longer aborts the run — `FilesManager` / `OpenAIFilesManager` retry up to
  5 times with exponential backoff + jitter, reopening the file each attempt.
  Non-transient errors (e.g. `400`) still fail fast. Already-uploaded images are
  skipped via the manifest, so reruns were always safe; this avoids needing one.

### Changed
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

### New features
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
