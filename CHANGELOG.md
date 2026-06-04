# Changelog

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
  (persistent manifest), submit a Message Batch for rich per-line
  descriptions, save a checkpoint for later retrieval.
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
