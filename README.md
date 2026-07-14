# pxGPT - Plant Analysis Tool

**pxGPT** (Phenotype eXplorer GPT) is a command-line tool for large-scale plant phenotyping using multiple LLM providers (Anthropic Claude, OpenAI, Google, Ollama, LM Studio, vLLM).

## Features

- **Batch API** (Stage 1 & 3): submit hundreds of plant lines in a single API call; fire-and-forget with checkpoint-based result retrieval
- **Files API**: upload each image once, reuse the same `file_id` across Stage 1 and Stage 3 — no re-uploading 10 k images
- **Adaptive thinking** (Stage 3): native `output_config.effort` on claude-sonnet-4-6; temperature guard enforced automatically
- **Native structured output** (Stage 3): schema passed directly as `output_config.format`; no regex or tag parsing
- **Schema normalizer**: one command adds `additionalProperties: false` and `required` arrays to every object in your schema
- **JSON-to-table flattening**: one command turns the per-plant Stage 3 JSON output into a wide, typed CSV + feather table (ordinal traits reconstructed from level code to label, as an ordered factor in R)
- **Multiple providers**: Anthropic, OpenAI, Google, Ollama, LM Studio, vLLM
- **Prompt caching**: automatic for Anthropic (reduces costs on repeated system prompts)
- **Robust error handling**: exponential backoff, per-request failure isolation, crash-safe manifest
- **Crash-safe sequential dispatch** (Stage 3 sharded): `--dispatch sequential` persists each shard to disk as it returns and **resumes** after a kill/crash — skipping already-completed calls (no re-billing) and retrying transient overloads in-run
- **Recoverable batch gaps** (Stage 3 sharded): `fetch-results` saves every succeeded shard to `<output>/_partial/`, so a batch that errored some shards (e.g. a transient `overloaded_error`) is fixed by a short `--dispatch sequential` resume that re-issues **only** the failed shards
- **Example master schema**: see [Example_master_schema.tsv](Example_master_schema.tsv) for the flattened field reference

## Pipeline overview

| Stage | Automated? | Command |
|-------|-----------|---------|
| 1 — plant line/cultivar descriptions | ✅ | `pxgpt describe-batch` |
| 2 — schema synthesis | Manual (human-in-the-loop) | GUI session with an LLM |
| 3 — structured phenotyping | ✅ | `pxgpt phenotype-batch` |

## 📖 User Manual

**For complete workflows, advanced usage, and troubleshooting see the [User Manual](user_manual.md).**

---

## Installation

```bash
git clone https://github.com/xavierzheng/pxgpt.git
cd pxgpt
pip install -r requirements.txt
pip install -e .
cp .env.example .env   # then fill in your API keys
```

## Configuration

Key variables in `.env`:

```bash
ANTHROPIC_API_KEY=your_key_here
DEFAULT_PROVIDER=anthropic

# Model (default already set to the current recommended model)
ANTHROPIC_MODEL=claude-sonnet-4-6

# Batch token budgets
STAGE1_MAX_TOKENS=16384   # raise to 65536 for long descriptions
STAGE3_MAX_TOKENS=16384

# Adaptive thinking effort. default = off = none = NO reasoning + temperature is sent.
# (blank, "off", and "none" are equivalent.) Set a level low/medium/high/xhigh/max to
# enable reasoning; the --effort flag overrides per run.
STAGE3_EFFORT=     # Stage 3 / schema command   (off | low | medium | high | xhigh | max)
DESCRIBE_EFFORT=   # Stage 1 describe-batch
ANALYZE_EFFORT=    # sync analyze command

# Set true to allow up to 300 k output tokens per response in Stage 1 batches
BATCH_300K_OUTPUT=false

# Parallel image upload threads
UPLOAD_CONCURRENCY=10

# Use the Files API (default true). Set false — or pass --no-files-api — to
# embed images inline as base64 instead of uploading once and reusing file_ids.
USE_FILES_API=true
```

### Local / self-hosted providers (analyze + schema)

Each is a first-class `--provider` value with its own env vars (no need to overload the OpenAI ones). Use a **vision-capable** model since both commands send images.

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b

# LM Studio (OpenAI-compatible)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=gemma4:12b
LMSTUDIO_API_KEY=lm-studio          # any non-empty placeholder

# vLLM (OpenAI-compatible) — VLLM_MODEL is required (the served model name)
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=gemma4:12b
VLLM_API_KEY=EMPTY                  # match --api-key if the server sets one
```

Then, e.g.: `pxgpt analyze --provider vllm ...` or `pxgpt schema --provider lmstudio ...`.

> Batch stages (`describe-batch*`, `phenotype-batch*`) are Anthropic/OpenAI-only; the local providers apply to the sync `analyze` and `schema` commands.

---

## Usage

### Batch workflow (recommended for large collections)

**Image layout**: one subdirectory per plant line inside a root folder; the subdir name is used as the line ID.

```
images/
├── s0001/
│   ├── angle1.jpg
│   └── angle2.jpg
├── s0002/
│   └── ...
```

**Step 1 — normalize your schema** (one-time, in-place):
```bash
pxgpt normalize-schema --schema prompts/phenotype_schema.json
```

**Step 2 — Stage 1 descriptions**:
```bash
pxgpt describe-batch \
  --input-dir ./images \
  --output descriptions.txt \
  --system-prompt prompts/phenotyping_system.txt \
  --prompt prompts/describe_plant.txt
# Prints batch ID and saves checkpoint_<batch_id>.json
```
When results are fetched, `descriptions.txt` contains grouped descriptions, one section per plant line/cultivar.

**Step 3 — Stage 3 structured phenotyping** (can run concurrently with Stage 1; images are already uploaded). With the Files API, `--input-dir` is optional — the plant lines and their `file_id`s are reused straight from `--manifest`:
```bash
pxgpt phenotype-batch \
  --schema prompts/phenotype_schema.json \
  --output phenotypes/ \
  --system-prompt prompts/phenotyping_system_schema.txt \
  --prompt prompts/extract_traits.txt \
  --manifest file_manifest.json
```

**Step 4 — retrieve results** (once the Anthropic batch finishes, usually within a few hours):
```bash
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json
```

#### Large schemas: sharded Stage 3 (fixes "compiled grammar is too large")

A big master schema (many traits/enums/nested groups) can exceed Anthropic's
structured-outputs **internal grammar-size limit** — every request fails with
`invalid_request_error: The compiled grammar is too large`. Shard the schema by
organ group so each call carries a small, compilable schema, then merge:

```bash
# 1. Generate shards from the master schema (writes <shard-dir>/ + shards_manifest.json)
pxgpt shard-schema --master master_schema.json --shard-budget 40
#    -> shard_NN.schema.json, shard_NN.prompt.md, shards_system.md, shards_manifest.json

# 2. Run Stage 3 in sharded mode (one small schema per shard; images are ordinary input)
pxgpt phenotype-batch \
  --shard-dir path/to/shards \
  --output phenotypes/ \
  --manifest file_manifest.json
#    --dispatch batch (default) | sequential   (sequential = crash-safe + resumable;
#                                                see below)

# 3. Fetch + merge: one {line_id}.json per plant, {line_id}.gaps.json for any gaps
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json

# 4. (ONLY if step 3 left {line_id}.gaps.json files) recover the failed shards.
#    A batch request that errored (e.g. a transient "overloaded_error") is stuck
#    inside the Batch API; re-fetching just reproduces the same gap. fetch-results
#    saves every SUCCEEDED shard to <output>/_partial/, so a short sequential
#    resume to the SAME --output re-issues only the failed shards (with in-run
#    retry) and clears the gaps. Match the original run's model/effort.
export ANTHROPIC_MODEL=claude-sonnet-5 STAGE3_EFFORT=medium
pxgpt phenotype-batch \
  --shard-dir path/to/shards \
  --manifest file_manifest.json \
  --master-schema master_schema.json \
  --output phenotypes/ \
  --dispatch sequential
#    -> "1410 skip, 10 to run ... 0 plant(s) with gaps"; gaps.json deleted once filled
```

In sharded mode `--schema`/`--system-prompt`/`--prompt` are optional (the per-shard
schemas and the shared system preamble come from the shard set). A pre-flight live
compile check verifies each shard and auto-reshards at a smaller budget if one
still trips the limit. Only the shared system block is marked for prompt caching.
Images remain ordinary input and stay before the per-shard text prompt.

`--dispatch sequential` is **crash-safe and resumable**: each shard is written to
`<output>/_partial/` as it returns, so a SLURM kill / crash loses nothing. Just
re-run the same command — completed `(plant, shard)` calls are skipped (not
re-billed) and only the missing ones run (`--no-resume` forces a fresh run).
Transient overloads (429 / 5xx / 529) are retried in-run with backoff, and
progress prints live to the SLURM log.

**Recovering `batch` gaps.** A batch request that errors (typically a transient
`overloaded_error: File storage is temporarily unavailable`) is terminal inside
the Batch API — `--resume` can't touch a batch, and re-fetching reproduces the same
`{line_id}.gaps.json`. Since `fetch-results` now persists every succeeded shard to
`<output>/_partial/`, you recover by running step 4 above: `--dispatch sequential`
to the same `--output` re-issues only the still-missing shards. **Match the
original run's settings exactly** — copy `--system-prompt`, `STAGE3_EFFORT` and
`ANTHROPIC_MODEL` from that batch's `step_04_phenotyping.sh` (omitting a
`--system-prompt` override silently falls back to a different prompt). See the full
worked example (with a sample `gaps.json` and expected output) in
[`user_manual.md`](user_manual.md) → *Stage 3 (sharded) → Step 4*, and
[`dispatch_batch_vs_sequential.md`](dispatch_batch_vs_sequential.md).

#### Downstream analysis: flatten results into a table

The per-plant `{line_id}.json` files aren't analysis-ready as-is (ordinal
traits store an integer level code, not a label; quantitative traits carry no
unit). Flatten the whole result directory into one row-per-plant table:

```bash
pxgpt json-to-table \
  --result-dir phenotypes/ \
  --master-schema master_schema.json \
  --out-prefix analysis/stage3_table
# Writes analysis/stage3_table.csv and analysis/stage3_table.feather
```

Nominal traits stay plain strings, quantitative traits become numeric
`<trait>_<unit>` columns, and ordinal traits are reconstructed into their
schema label — a plain string in the CSV, an **ordered** `pandas.Categorical`
in the feather file so `arrow::read_feather()` reads them as ordered factors
in R. Missing traits and `not_assessable` become real NA everywhere.

If two traits ever compute the same column name (e.g. the same leaf key
assessed under two organ groups), `json-to-table` refuses to silently drop
one — it writes no files and prints a `--rename-map` fill-in template by
default. Pass `--on-collision prefix_collided` to auto-disambiguate just the
clashing columns instead, or `--rename-map FILE` to hand-pick names. See the
**Downstream analysis** section of the [User Manual](user_manual.md) for the
full column-typing rules and a worked collision-resolution example.

### Single-sample commands (for testing / small runs)

```bash
# Plain text description
pxgpt analyze \
  --input-folder images/s0001 \
  --output s0001_desc.txt \
  --system-prompt prompts/phenotyping_system.txt \
  --prompt prompts/describe_plant.txt

# Structured JSON (uses native structured output for Anthropic)
pxgpt schema \
  --input-folder images/s0001 \
  --output s0001.json \
  --system-prompt prompts/phenotyping_system_schema.txt \
  --schema prompts/phenotype_schema.json \
  --prompt prompts/extract_traits.txt
```

---

## Commands

| Command | Purpose |
|---------|---------|
| `pxgpt describe-batch` | Stage 1 (Anthropic): upload images via Files API, submit batch for descriptions |
| `pxgpt phenotype-batch` | Stage 3 (Anthropic): reuse file_ids, submit batch with structured output |
| `pxgpt describe-batch-openai` | Stage 1 (OpenAI): same as describe-batch on the OpenAI Batch API |
| `pxgpt phenotype-batch-openai` | Stage 3 (OpenAI): strict structured output on the OpenAI Batch API |
| `pxgpt fetch-results` | Retrieve results for any pending batch (Anthropic or OpenAI) from a checkpoint |
| `pxgpt cleanup-files` | Delete Files-API uploads from a manifest (both providers); OpenAI bills for storage |
| `pxgpt extract-report` | Extract `<report>` from `<think>`/`<report>` output (single or grouped); back-compat for non-native reasoning |
| `pxgpt normalize-schema` | Add `additionalProperties: false` + `required` to all objects in a schema |
| `pxgpt shard-schema` | Split a master schema into compilable Stage 3 shards (+ per-shard prompts) for `phenotype-batch --shard-dir` |
| `pxgpt json-to-table` | Flatten Stage 3 per-plant JSON results into a wide, typed CSV + feather table (with column-name collision detection/resolution) |
| `pxgpt analyze` | Single-folder text description (sync, all providers) |
| `pxgpt schema` | Single-folder structured JSON (sync, all providers) |

Run `pxgpt <command> --help` for full argument details.

---

## Providers

| Provider | Caching | Batch API | analyze / schema | Notes |
|----------|---------|-----------|------------------|-------|
| **Anthropic** (default) | ✅ | ✅ | ✅ | Native thinking, structured output, Files API |
| **OpenAI** | — | ✅ | ✅ | Batch API stages + Files API (`vision`); sync via LiteLLM |
| **Ollama** | — | — | ✅ | Local; `ollama/` route; use a vision model |
| **LM Studio** | — | — | ✅ | OpenAI-compatible (`LMSTUDIO_*`); use a vision model |
| **vLLM** | — | — | ✅ | OpenAI-compatible (`VLLM_*`, model required); use a vision model |
| **Google Gemini** | — | — | ✅ | Via LiteLLM (`gemini/` route) |

For `analyze` / `schema`, structured output on non-Anthropic providers is delivered by appending the schema to the system prompt — make the prompt request JSON-only output (the bundled `prompts/extract_traits.txt` does this).

---

## Project structure

```
pxgpt/
├── core/
│   ├── config.py          # All config with env-var overrides
│   ├── batch_utils.py     # Anthropic: temperature guard, poll, result writers
│   ├── openai_batch_utils.py  # OpenAI: JSONL build, strict schema, result writers
│   ├── files_manager.py   # Anthropic Files API upload + manifest
│   ├── openai_files_manager.py  # OpenAI Files API upload + manifest
│   ├── schema_utils.py    # JSON schema normalizer
│   ├── shard_builder.py   # Stage 3 shard generation from a master schema
│   ├── sharding.py        # Stage 3 shard loading, compile-check, merge/validate
│   ├── json2table.py      # Flatten per-plant Stage 3 JSON into a wide table
│   ├── image_utils.py     # Base64 + file_id content builders
│   └── file_utils.py      # File I/O helpers
├── providers/
│   ├── anthropic_provider.py
│   ├── litellm_provider.py
│   └── base.py
├── commands/
│   ├── describe.py        # describe-batch (Anthropic)
│   ├── phenotype.py       # phenotype-batch (Anthropic)
│   ├── openai_batch.py    # describe-batch-openai / phenotype-batch-openai
│   ├── fetch_results.py   # fetch-results (provider-aware)
│   ├── cleanup_files.py   # cleanup-files (delete Files-API uploads)
│   ├── extract_report.py  # extract-report (<think>/<report> back-compat)
│   ├── normalize_schema.py
│   ├── shard_schema.py     # shard-schema (build Stage 3 shards from a master)
│   ├── json2table.py       # json-to-table (flatten Stage 3 JSON -> CSV/feather)
│   ├── analyze.py
│   └── schema.py
└── main.py
```

## License

MIT License
