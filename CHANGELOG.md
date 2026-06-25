# Changelog

## Unreleased

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
