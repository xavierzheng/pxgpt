# pxGPT User Manual — Plant Phenotyping with Large Language Models

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Complete Workflow](#complete-workflow)
4. [Command Reference](#command-reference)
5. [Provider Configuration](#provider-configuration)
6. [Schema Design](#schema-design)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

---

## Introduction

**pxGPT** (Phenotype eXplorer GPT) is a command-line tool for large-scale plant phenotyping using Large Language Models. It processes germplasm collections of hundreds of plant lines with thousands of images through a two-stage automated pipeline, with a deliberate human-in-the-loop step between stages for schema design.

### Pipeline overview

| Stage | Automated? | Description |
|-------|-----------|-------------|
| **1 — Descriptions** | ✅ `describe-batch` | Feed multi-angle images per plant line/cultivar → rich descriptive text |
| **2 — Schema synthesis** | Manual | Paste Stage 1 output into a conversational LLM session; design a JSON schema that captures the observed variation |
| **3 — Structured phenotyping** | ✅ `phenotype-batch` | Feed the same images + your schema → validated JSON per plant line/cultivar |

Stages 1 and 3 reference the **same uploaded images**: each image is uploaded once via the Files API and its `file_id` is stored in a manifest, so re-running or adding Stage 3 after Stage 1 never re-uploads anything.

### Why pxGPT?

- **Scale**: process hundreds of lines / ~10 000 images in a single Batch API submission
- **Cost**: images uploaded once; prompt caching on repeated system prompts; 50–90 % cache savings typical
- **Accuracy**: adaptive thinking (`output_config.effort`) on Stage 3 improves structured extraction quality
- **Reliability**: fire-and-forget batches with checkpoint files; per-request failure isolation; crash-safe manifest

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/xavierzheng/pxgpt.git
cd pxgpt
pip install -r requirements.txt
pip install -e .
```

### 2. Configuration

```bash
cp .env.example .env
vim .env          # add ANTHROPIC_API_KEY at minimum
```

### 3. Image layout

Create one subdirectory per plant line inside a root folder. The subdirectory name becomes the line ID (used as `custom_id` in the batch and as the output filename in Stage 3):

```
images/
├── s0001/
│   ├── top_view.jpg
│   ├── side_left.jpg
│   └── roots.jpg
├── s0002/
│   └── ...
└── s0N/
    └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`

### 4. Normalize your schema (one-time)

Before the first Stage 3 run, normalize your JSON schema to meet the Anthropic structured-output requirements:

```bash
pxgpt normalize-schema --schema prompts/phenotype_schema.json
```

This adds `additionalProperties: false` and an empty `required` array to every object node, and strips the `format` keyword (e.g. `"format": "date"`) which is not supported by the API.

---

## Complete Workflow

### Stage 1 — Batch descriptions

Submit all plant lines in one batch call. Images are uploaded to the Files API first (skipping any already in the manifest):

```bash
pxgpt describe-batch \
  --input-dir ./images \
  --output descriptions.txt \
  --system-prompt prompts/phenotyping_system.txt \
  --prompt prompts/describe_plant.txt \
  --manifest file_manifest.json
```

**What happens:**
1. Discovers all subdirectories in `--input-dir`
2. Uploads new images via `client.beta.files` (parallel, up to `UPLOAD_CONCURRENCY` threads); skips already-uploaded files found in `--manifest`
3. Submits a Message Batch (one request per plant line)
4. Saves `checkpoint_<batch_id>.json` and exits immediately (fire-and-forget)

**Output format** (`descriptions.txt`):
Grouped descriptions, one section per plant line/cultivar:

```
### s0001

[Rich morphological description of the plant...]

---

### s0002

[...]
```

This format is designed to be copy-pasted directly into a conversational LLM session for Stage 2.

**Retrieve results when the batch finishes** (check status on the Anthropic console or just run):
```bash
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json
```

**Optional: block and poll** (for small test batches):
```bash
pxgpt describe-batch ... --wait
```

---

### Stage 2 — Schema synthesis (manual, human-in-the-loop)

This stage is intentionally kept manual. Open a conversational LLM session (Claude.ai with extended thinking recommended) and paste the contents of `descriptions.txt`.

**Prompt template:**

```
You are a professional botanist and data scientist. Based on the phenotyping
reports below, generate a comprehensive JSON schema covering all possible
phenotypic descriptions for this Brassica collection.

Requirements:
- Use ontology-like terms
- Transform qualitative traits to enum with all observed values
- Standardize units for quantitative traits
- Include every trait observed in at least one cultivar — nothing is too rare
- For every object node: include additionalProperties: false and a required array

[Paste descriptions.txt content here]
```

**Iterate** until the schema covers all observed variation. Save the final schema to a file (e.g. `prompts/phenotype_schema.json`), then normalize it:

```bash
pxgpt normalize-schema --schema prompts/phenotype_schema.json
```

---

### Stage 3 — Batch structured phenotyping

Submit all plant lines for structured extraction. The same manifest from Stage 1 is reused, so no images are re-uploaded. With the Files API (default), `--input-dir` is **optional**: omit it and the plant lines plus their `file_id`s are reconstructed straight from `--manifest`, so the image tree need not even be present on disk:

```bash
pxgpt phenotype-batch \
  --schema prompts/phenotype_schema.json \
  --output phenotypes/ \
  --system-prompt prompts/phenotyping_system_schema.txt \
  --prompt prompts/extract_traits.txt \
  --manifest file_manifest.json
```

Pass `--input-dir ./images` as well if you want Stage 3 to also pick up (and upload) any images added since Stage 1. `--input-dir` **is required** with `--no-files-api`, because inline base64 mode must read the image bytes from disk.

**API features used:**
- `output_config.format = {"type": "json_schema", "schema": …}` — native structured output; the schema grammar is compiled once and cached across all requests in the batch
- `output_config.effort` — adaptive thinking, **off by default**. Stage 3 runs without reasoning and sends `temperature`. Enable reasoning by setting `STAGE3_EFFORT` (e.g. `medium`).
- When `STAGE3_EFFORT` is set, `temperature` is **not sent** (the API enforces this; the guard is automatic)

**Retrieve results:**
```bash
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json
```

**Output**: one `{line_id}.json` file per plant line in the `--output` directory. If JSON parsing fails for a line, a `{line_id}.err.txt` file is written instead for manual inspection.

---

### Stage 3 (sharded) — for schemas too large to compile

Structured outputs compile your JSON schema into a constrained-decoding grammar, and there is an **internal limit on the compiled grammar size**. A large master schema (many traits, enums and nested organ groups) trips it, and *every* request fails with:

```
invalid_request_error: The compiled grammar is too large, which would cause
performance issues. Simplify your tool schemas or reduce the number of strict tools.
```

This is not a bug and not the published 24-optional-parameter / 16-union limit — it's the internal grammar-size ceiling. The fix is to **shard** the schema by organ group so each request carries a small, compilable schema, then **merge** the per-shard results back into one record per plant.

**Step 1 — generate the shard set from your master schema:**

```bash
pxgpt shard-schema --master master_schema.json --shard-budget 40
# Writes <master dir>/shards/:
#   shard_NN.schema.json   one small structured-output schema per shard
#   shard_NN.prompt.md     the organ-specific prompt text for that shard
#   shards_system.md       the shared invariant preamble (cached system block)
#   shards_manifest.json   shard list + trait inventory (drives the merge)
```

The master schema is the Stage 2 format (`trait_groups → traits` with `scale_type`/`values`/`unit`). Groups are bin-packed up to `--shard-budget` (a grammar-cost proxy; default 40). Lower the budget if a shard is still too large — a group that alone exceeds the budget is automatically sub-sharded across its traits. Quantitative `value`s are emitted as strings (parsed back to numbers at merge time), never `anyOf`, since union types inflate the grammar.

**Step 2 — run Stage 3 in sharded mode:**

```bash
pxgpt phenotype-batch \
  --shard-dir master_schema_generation/shards \
  --output phenotypes/ \
  --manifest file_manifest.json
#   --dispatch batch (default) | sequential
```

In sharded mode `--schema`, `--system-prompt` and `--prompt` are **optional** — the per-shard schemas and the shared system preamble come from the shard set (pass `--system-prompt` only to override the preamble). `--master-schema` overrides the master path recorded in the manifest (used to validate the merged record).

- **Pre-flight compile check**: each distinct shard schema is test-compiled with a tiny live request before the run. If one still trips the limit, pxGPT **auto-reshards** at a smaller budget (re-running the generator in-process so schema, prompt and manifest stay in sync) and re-checks.
- **Prompt caching**: the system prompt + the plant's images form a byte-identical, cached prefix shared across that plant's shards (`cache_control` on the last image block); only the small per-shard prompt and schema are re-sent. The first shard of a plant pays the image cost; the rest hit the cache. Cache-creation vs cache-read tokens are logged so you can confirm caching empirically.
- **Dispatch**: `batch` (default) submits one Message Batch for all *(plant × shard)* requests — cheapest, but the 5-minute prompt cache may expire before async execution. `sequential` runs each plant's shards as near-synchronous calls, which reliably keeps the image prefix in the cache window. Make the call once and compare the logged cache-read rate.

**Step 3 — retrieve + merge:**

```bash
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json
```

For a sharded run, `fetch-results` demultiplexes the `custom_id = "<line_id>__<shard_id>"` results, merges each plant's shards into one record keyed by the master organ structure, parses quantitative strings to numbers, and validates coverage against the master schema.

**Output**: one merged `{line_id}.json` per plant. If any trait is missing or a shard errored, a `{line_id}.gaps.json` is written alongside it listing the missing traits and shard errors.

---

### Downstream analysis

**Load all JSON files into a single DataFrame (Python):**

```python
import json
import pandas as pd
from pathlib import Path

records = []
for f in sorted(Path("phenotypes/").glob("*.json")):
    data = json.loads(f.read_text())
    data["_line_id"] = f.stem
    records.append(pd.json_normalize(data))

df = pd.concat(records, ignore_index=True)
```

**R:**
```r
library(jsonlite)
library(dplyr)

files <- list.files("phenotypes/", pattern = "\\.json$", full.names = TRUE)
df <- bind_rows(lapply(files, function(f) {
  d <- fromJSON(f, flatten = TRUE)
  d$line_id <- tools::file_path_sans_ext(basename(f))
  d
}))
```

---

## Command Reference

### `pxgpt describe-batch`

Stage 1 batch description.

```
pxgpt describe-batch \
  --input-dir PATH \
  --output FILE \
  --system-prompt FILE \
  --prompt FILE \
  [--manifest FILE]      # default: file_manifest.json (ignored with --no-files-api)
  [--no-files-api]       # embed images inline as base64 instead of uploading
  [--effort {off,low,medium,high,xhigh,max}]   # overrides DESCRIBE_EFFORT; default off
  [--wait]               # poll until done; default is fire-and-forget
```

Output file: grouped descriptions, one section per plant line/cultivar.

By default Stage 1 runs **without reasoning** and sends `temperature` (the model's whole response is the description — no `<think>`/`<report>` tags needed). Enable Anthropic adaptive thinking with `--effort` (e.g. `--effort medium`) or by setting `DESCRIBE_EFFORT`; thinking blocks are produced natively and stripped from the saved description.

> Using a legacy `<think>`/`<report>` prompt instead of native reasoning? The tags are saved verbatim — post-process the output with [`pxgpt extract-report`](#pxgpt-extract-report) to keep only the `<report>` body.

---

### `pxgpt phenotype-batch`

Stage 3 batch structured phenotyping.

```
pxgpt phenotype-batch \
  --output DIR \
  --schema FILE \        # required in single-schema mode; ignored with --shard-dir
  --system-prompt FILE \ # required in single-schema mode; optional override with --shard-dir
  --prompt FILE \        # required in single-schema mode; ignored with --shard-dir
  [--input-dir PATH]     # optional with the Files API: lines + file_ids are
                         # reconstructed from --manifest. Required with --no-files-api.
  [--manifest FILE]      # default: file_manifest.json (ignored with --no-files-api)
  [--no-files-api]       # embed images inline as base64 instead of uploading
  [--shard-dir DIR]      # SHARDED mode: per-shard schemas+prompts from shard-schema
  [--master-schema FILE] # sharded: master used to validate the merged record
  [--dispatch {batch,sequential}]   # sharded dispatch strategy (default: batch)
  [--wait]
```

Output directory: one `{line_id}.json` per plant line; `{line_id}.err.txt` for parse failures (single-schema mode) or `{line_id}.gaps.json` for missing traits (sharded mode). See **Stage 3 (sharded)** above for when and how to use `--shard-dir`.

---

### `pxgpt describe-batch-openai` / `pxgpt phenotype-batch-openai`

OpenAI equivalents of the two Anthropic batch stages, running on the **OpenAI Batch API** via the **Responses** endpoint (`/v1/responses` JSONL). The Responses API is required because an uploaded image can only be referenced by `file_id` there — the Chat Completions API cannot reference uploaded images. Same input layout (one subdirectory per plant line) and the same fire-and-forget / `--wait` workflow.

```
pxgpt describe-batch-openai \
  --input-dir PATH \
  --output FILE \
  --system-prompt FILE \
  --prompt FILE \
  [--manifest FILE]      # default: openai_file_manifest.json (ignored with --no-files-api)
  [--no-files-api]       # embed images inline as base64 instead of uploading
  [--wait]

pxgpt phenotype-batch-openai \
  --input-dir PATH \
  --schema FILE \
  --output DIR \
  --system-prompt FILE \
  --prompt FILE \
  [--manifest FILE]
  [--no-files-api]
  [--wait]
```

Key differences from the Anthropic stages:

- **Model**: uses `OPENAI_MODEL` (default `gpt-5-2025-08-07`).
- **Files API**: images are uploaded with OpenAI's `purpose="vision"` and referenced by `file_id`. Because OpenAI and Anthropic file_ids are different namespaces, the OpenAI manifest defaults to a **separate file** (`openai_file_manifest.json`) — do not point it at the Anthropic `file_manifest.json`.
- **Structured output** (`phenotype-batch-openai`): the schema is normalized in memory for OpenAI **strict** mode — every property is forced into `required` and `additionalProperties: false` is set on every object (stricter than `pxgpt normalize-schema`, which targets Anthropic). The file on disk is not modified.
- **Reasoning effort**: set `OPENAI_REASONING_EFFORT` (`minimal`/`low`/`medium`/`high`, or empty to disable) — applied only to reasoning models (gpt-5, o-series). For those models a custom `temperature` is omitted automatically.
- **Completion window**: `OPENAI_BATCH_COMPLETION_WINDOW` (default `24h`).

Checkpoints are tagged with `"provider": "openai"`, so `pxgpt fetch-results` retrieves them the same way as Anthropic batches.

> **Cost reminder:** OpenAI bills for stored files. After fetching results, delete the uploaded images (and the batch's input/output/error files) with `pxgpt cleanup-files --manifest openai_file_manifest.json --checkpoint checkpoint_<batch_id>.json`. See [`pxgpt cleanup-files`](#pxgpt-cleanup-files).

---

### `pxgpt fetch-results`

Retrieve results for any pending or completed batch — Anthropic **or** OpenAI. The backend is selected from the checkpoint's `provider` field automatically.

```
pxgpt fetch-results \
  --checkpoint FILE \    # checkpoint_<batch_id>.json written at submit time
  [--output PATH]        # override the output path stored in the checkpoint
```

Prints batch status. If the batch is still processing, exits with a message. If ended, writes results and prints a token-usage summary.

---

### `pxgpt cleanup-files`

Delete files uploaded via the Files API once you no longer need them. **OpenAI bills for stored files**, so always clean up OpenAI uploads after fetching results (Anthropic uploads are also removable). The command deletes every `file_id` recorded in a manifest, auto-detecting the provider, and prunes the manifest as it goes. A file that is already gone (HTTP 404) counts as deleted.

```
pxgpt cleanup-files \
  --manifest FILE \         # manifest of uploaded images to delete (file_ids)
  [--provider auto|anthropic|openai]   # default: auto-detect from the manifest
  [--checkpoint FILE]       # (repeatable) also delete OpenAI batch input/output/error files
  [--dry-run]               # show what would be deleted, delete nothing
```

Examples:

```bash
# Anthropic: delete all uploaded images (provider auto-detected)
pxgpt cleanup-files --manifest file_manifest.json

# OpenAI: delete uploaded images AND the batch input/output/error files
pxgpt cleanup-files --manifest openai_file_manifest.json \
  --checkpoint checkpoint_batch_xxxxxxxx.json

# Preview only — nothing is deleted
pxgpt cleanup-files --manifest openai_file_manifest.json --dry-run
```

Notes:

- The **manifest** is the same `--manifest` file you passed to the batch commands (`file_manifest.json` for Anthropic, `openai_file_manifest.json` for OpenAI). After cleanup the deleted entries are removed, so the manifest is safe to keep or discard.
- `--checkpoint` is **OpenAI-only** and removes the batch's `input`, `output`, and `error` files (extra stored files OpenAI creates per batch). Anthropic batch results are not stored as separate Files-API objects, so no checkpoint cleanup is needed there.
- Once files are deleted, any manifest still pointing at them is stale — do not reuse it for a new batch, or re-upload first.

Manual alternative (single file, no manifest):

```python
# OpenAI
from openai import OpenAI; OpenAI().files.delete("file-...")
# Anthropic
from anthropic import Anthropic; Anthropic().beta.files.delete("file_...")
```

---

### `pxgpt extract-report`

Two ways to get the model to reason before answering:

1. **Native reasoning** (recommended) — enable adaptive thinking with `--effort` / `*_EFFORT`. The reasoning happens in a separate channel and is stripped automatically; the saved output is already clean. **No extraction needed.**
2. **Chain-of-thought tags** (backward-compatible) — prompt the model to emit `<think>...</think><report>...</report>`. The whole response (tags and all) is saved verbatim, so you post-process it with `extract-report` to keep only the `<report>` body.

`extract-report` supports both a **single-response file** and the **grouped** multi-cultivar file from `describe-batch` / `describe-batch-openai` (one `<report>` per `### <id>` section).

```
pxgpt extract-report \
  --input FILE \                      # a single response, or a grouped describe output
  [--output FILE] \                   # default: print to stdout
  [--mode {auto,grouped,single}]      # default auto (detects '### ' section headers)
```

- **grouped** mode extracts the `<report>` from every `### <id>` section and preserves the section structure (`### <id>` + `---` separators).
- **single** mode treats the whole file as one response.
- `<think>` reasoning is **discarded**; only `<report>` is kept. Unclosed tags (e.g. a response truncated by the token limit) are auto-closed before extraction.

Examples:

```bash
# Batch: clean a grouped describe output (all cultivars at once)
pxgpt extract-report --input descriptions.txt --output descriptions.clean.txt

# Single file -> stdout
pxgpt extract-report --input one_plant.txt > one_plant.clean.txt
```

> The standalone `extract_report_tags.py` script is still available for the simple single-file case (`python extract_report_tags.py FILE`), but `pxgpt extract-report` is preferred — it also handles the grouped batch output.

---

### `pxgpt normalize-schema`

Prepare a JSON schema for Anthropic structured outputs.

```
pxgpt normalize-schema \
  --schema FILE \
  [--output FILE]        # default: overwrite in-place
```

Changes applied:
- Adds `additionalProperties: false` to every `object` node
- Adds `required: []` to every object that has a `properties` dict but no `required` array
- Strips `"format"` keyword (e.g. `"format": "date"`) — not supported by the API
- Strips the root `$schema` meta-key

---

### `pxgpt shard-schema`

Split a master phenotype schema into compilable Stage 3 shards (used when the full schema trips the structured-outputs grammar-size limit). See **Stage 3 (sharded)** for the full workflow.

```
pxgpt shard-schema \
  --master FILE \           # master schema (Stage 2 format: trait_groups -> traits)
  [--shard-dir DIR] \       # default: <master dir>/shards
  [--shard-budget N] \      # grammar-cost budget per shard (default: 40)
  [--combined] \            # also write combined stage3_schema.json + stage3_prompt.md
  [--combined-dir DIR]      # default: parent of --shard-dir
```

Writes per shard `shard_NN.schema.json` + `shard_NN.prompt.md`, the shared `shards_system.md`, and `shards_manifest.json`. Whole organ groups are bin-packed up to `--shard-budget`; a group exceeding it alone is sub-sharded across its traits. The resulting shard directory is consumed by `pxgpt phenotype-batch --shard-dir`.

> The standalone `build_stage3.py` in the analysis tree is a thin wrapper over this command (same output), kept for the existing local workflow.

---

### `pxgpt analyze`

Single-folder text description (sync, all providers). Useful for testing prompts on one plant line.

```
pxgpt analyze \
  --input-folder PATH \
  --output FILE \
  --system-prompt FILE \
  --prompt FILE \
  [--provider {anthropic,openai,google,ollama}]
  [--effort {off,low,medium,high,xhigh,max}]   # Anthropic adaptive thinking; default off
```

`--effort` enables Anthropic adaptive thinking (overrides `ANALYZE_EFFORT`; default **off**, preserving the original non-thinking behavior). When thinking is active, `temperature` is omitted automatically and the thinking blocks are stripped from the output. Ignored for non-anthropic providers.

#### Recipe: gather descriptions from `analyze` (single-folder mode)

`analyze` processes **one folder at a time**, so the classic workflow is to loop over cultivars and then merge the per-cultivar descriptions into one document — e.g. to feed Stage 2 (schema synthesis). This is the single-file counterpart to `describe-batch`.

**1. Run `analyze` per cultivar:**

```bash
for i in $(ls germplasm_images/); do
  pxgpt analyze \
    --input-folder germplasm_images/${i} \
    --output results/${i}_description.txt \
    --system-prompt prompts/phenotyping_system.txt \
    --prompt prompts/describe_plant.txt \
    --provider anthropic
done
```

**2. Merge the descriptions into one document:**

- **If your prompt uses `<think>`/`<report>` tags** — extract the `<report>` from each file and concatenate with a header per cultivar:

  ```bash
  : > combined_phenotypes.txt   # start fresh
  for i in $(ls germplasm_images/); do
    echo "# This is cultivar ${i}"              >> combined_phenotypes.txt
    pxgpt extract-report --input results/${i}_description.txt >> combined_phenotypes.txt
    printf '\n\n'                               >> combined_phenotypes.txt
  done
  ```

  (Each file is a single response, so `extract-report` runs in `single` mode automatically. The legacy `python extract_report_tags.py results/${i}_description.txt` works identically here.)

- **If you use native reasoning (`--effort`) or a plain prompt** — there are no tags, so the whole file *is* the description. Just concatenate, no extraction:

  ```bash
  : > combined_phenotypes.txt
  for i in $(ls germplasm_images/); do
    echo "# This is cultivar ${i}" >> combined_phenotypes.txt
    cat results/${i}_description.txt >> combined_phenotypes.txt
    printf '\n\n' >> combined_phenotypes.txt
  done
  ```

**3.** Feed `combined_phenotypes.txt` into Stage 2 schema synthesis (see the pipeline overview).

> Tip: for many cultivars, `describe-batch` (one batch call, grouped output) is cheaper and simpler than looping `analyze`. If you already have a grouped describe output with tags, run `pxgpt extract-report --input descriptions.txt --output descriptions.clean.txt` once instead of the loop.

---

### `pxgpt schema`

Single-folder structured JSON output (sync, all providers). For Anthropic, uses native `output_config.format`; for other providers, appends the schema to the system prompt.

```
pxgpt schema \
  --input-folder PATH \
  --output FILE \
  --system-prompt FILE \
  --schema FILE \
  --prompt FILE \
  [--provider {anthropic,openai,google,ollama}]
  [--effort {off,low,medium,high,xhigh,max}]   # overrides STAGE3_EFFORT
```

For Anthropic, `schema` runs **without reasoning by default** (and sends `temperature`). Enable adaptive thinking with `--effort` (e.g. `--effort medium`) or by setting `STAGE3_EFFORT`.

---

## Provider Configuration

### Anthropic Claude (recommended)

```bash
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6

# Adaptive thinking effort (Anthropic). For every knob below:
#   default = off = none = NO reasoning + temperature IS sent
#   (blank, "off", "none" are all equivalent). Set a level to enable reasoning.
STAGE3_EFFORT=          # off/none (default) | low | medium | high | xhigh | max  — Stage 3 / schema
DESCRIBE_EFFORT=        # off/none (default) | low | medium | high | xhigh | max  — Stage 1 describe-batch
ANALYZE_EFFORT=         # off/none (default) | low | medium | high | xhigh | max  — sync analyze

# Token budgets
STAGE1_MAX_TOKENS=16384   # up to 65536 on sync; up to 300000 with BATCH_300K_OUTPUT=true
STAGE3_MAX_TOKENS=16384

# Files API (default true). Set false to embed images inline as base64 in every
# batch request instead of uploading once and reusing file_ids. The
# --no-files-api flag on describe-batch / phenotype-batch overrides this.
USE_FILES_API=true
```

**Files API vs. inline base64**: with the Files API (default) each image is uploaded once and referenced by `file_id` across Stage 1 and Stage 3 — best for large collections re-used across stages. With `USE_FILES_API=false` (or `--no-files-api`) images are embedded as base64 in each request: no upload step or manifest, useful when the Files API beta is unavailable or for one-off runs, at the cost of re-sending image bytes on every request.

**Prompt caching**: the system prompt is cached with `cache_control: ephemeral` on every request. Repeated Stage 3 runs over the same collection see 50–90 % cache hit rates on the (large) system prompt.

**300 k output tokens** (for very verbose Stage 1 descriptions):
```bash
BATCH_300K_OUTPUT=true
STAGE1_MAX_TOKENS=65536   # or higher, up to 300000
```

### OpenAI

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-2025-08-07
OPENAI_BASE_URL=                  # optional: point the openai provider at a proxy

# OpenAI Batch API stages (describe-batch-openai / phenotype-batch-openai)
OPENAI_REASONING_EFFORT=          # minimal | low | medium | high | "" (gpt-5/o-series only)
OPENAI_BATCH_COMPLETION_WINDOW=24h
```

Note: GPT-5 / o-series reasoning models only accept the default `temperature`; pxGPT omits a custom temperature for them automatically.

### Local / self-hosted providers (`analyze` + `schema` only)

`analyze` and `schema` run on Ollama, LM Studio, and vLLM in addition to the cloud providers. Each is a first-class `--provider` value with its own env vars. **Use a vision-capable model** — both commands send images. (The batch stages are Anthropic/OpenAI-only.)

```bash
# Ollama — pxgpt analyze --provider ollama ...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:12b           # a vision model, e.g. gemma4 / gemma3 / llava

# LM Studio (OpenAI-compatible) — pxgpt analyze --provider lmstudio ...
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=gemma4:12b         # name as shown in LM Studio
LMSTUDIO_API_KEY=lm-studio        # any non-empty placeholder

# vLLM (OpenAI-compatible) — pxgpt schema --provider vllm ...
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=gemma4:12b            # REQUIRED: the served model name
VLLM_API_KEY=EMPTY                # match the server's --api-key if set
```

How each is routed through LiteLLM: `ollama/<model>` @ `OLLAMA_BASE_URL`; `openai/<model>` @ `LMSTUDIO_BASE_URL` / `VLLM_BASE_URL` (both expose an OpenAI-compatible API). `api_base`/`api_key` are passed per request, so several providers can be configured at once without clashing. Unsupported parameters are dropped automatically per backend (`drop_params`).

For `schema` on these providers, the JSON schema is appended to the system prompt (Anthropic-style native structured output is not used). Make the user prompt request **JSON-only** output — the bundled `prompts/extract_traits.txt` already does this.

- Ollama: ensure `ollama serve` is running and the model is pulled (`ollama pull gemma3:12b`).
- vLLM: start with e.g. `vllm serve google/gemma-4-12b-it --port 8000`; set `VLLM_MODEL` to the same served name.

### Google Gemini

```bash
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-2.5-pro
```

---

## Schema Design

### Design principles

1. **Use enum for all qualitative traits** — the model selects from your list exactly; never invents new values.
2. **Standardize units** — pick one unit per measurement type (e.g. always `cm`).
3. **Cover rare phenotypes** — a trait seen in only one cultivar matters; include it.
4. **Flat is fine** — nested objects work, but deeply nested schemas increase the chance of the model losing track.

### Required structure for Anthropic structured output

Every `object` node must have:
```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["field1", "field2"],
  "properties": { ... }
}
```

Run `pxgpt normalize-schema` to add these automatically. Review the `required` arrays afterward — you may want to add all property names to `required` so the model always fills them in (use `"NA"` for unknown string fields, and your best estimate for numeric/boolean fields).

### Example schema fragment

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["growth_stage", "leaf_morphology"],
  "properties": {
    "growth_stage": {
      "type": "object",
      "additionalProperties": false,
      "required": ["stage", "true_leaf_count"],
      "properties": {
        "stage": {
          "type": "string",
          "enum": ["cotyledon","early_vegetative","mid_vegetative","bolting","flowering"]
        },
        "true_leaf_count": {"type": "integer"}
      }
    }
  }
}
```

See [Example_master_schema.tsv](Example_master_schema.tsv) for the flattened field reference of the included Brassica schema.

---

## Best Practices

### Prompt engineering

**Stage 1 system prompt** (`prompts/phenotyping_system.txt`): brief role definition. Keep it stable across runs — it is cached.

**Stage 1 user prompt** (`prompts/describe_plant.txt`): ask for all morphological traits you care about. The richer the descriptions, the better the Stage 2 schema synthesis.

**Stage 3 system prompt** (`prompts/phenotyping_system_schema.txt`): define output requirements. Include "Use NA only for string fields when a value cannot be confidently determined."

**Stage 3 user prompt** (`prompts/extract_traits.txt`): instruct the model to fill in every field, use scale references (rockwool cube, ColorChecker, ruler), and return a single valid JSON object.

### Upload concurrency

10 parallel uploads is a safe default. Raise to 20–30 if your network is fast and you have many small images:
```bash
UPLOAD_CONCURRENCY=20
```

### Manifest reuse

Always pass `--manifest file_manifest.json` to both `describe-batch` and `phenotype-batch`. If you accidentally omit it on a Stage 3 run, all 10 k images are re-uploaded. The manifest path defaults to `file_manifest.json` in the current directory.

### Running Stage 3 without Stage 1

If you already have a manifest from a previous run (or built it with `describe-batch`), `phenotype-batch` will reuse all cached `file_id`s. Only genuinely new images are uploaded.

In fact you can drop `--input-dir` entirely: with the Files API, `phenotype-batch` reconstructs every plant line and its `file_id`s from `--manifest` alone, so Stage 3 runs without the original image tree on disk. Provide `--input-dir` only when you want to upload images added since Stage 1, or when running with `--no-files-api` (which must read image bytes from disk).

### Cost optimization

- Prompt caching is automatic for Anthropic: the system prompt is marked `cache_control: ephemeral`. Each Stage 3 batch call uses the cached prompt for all requests after the first.
- Keep system prompts identical across runs (no date stamps, no plant-line/cultivar-specific insertions) so the cache key matches.
- For Stage 1, moderate `STAGE1_MAX_TOKENS` (16 384) is usually enough for descriptive text. Enable `BATCH_300K_OUTPUT=true` only if descriptions are being truncated.

---

## Troubleshooting

### "ANTHROPIC_API_KEY is not set"

Add the key to `.env` and confirm it is in the same directory where you run `pxgpt`:
```bash
ANTHROPIC_API_KEY=sk-ant-...
```

### No images found / wrong image format

- Supported extensions: `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`
- Images must be inside subdirectories of `--input-dir`, not directly in the root folder
- Use absolute paths if relative paths are ambiguous

### Batch status shows `errored` requests

`fetch-results` writes `.err.txt` files for failed lines. Common causes:
- Image too large or corrupted → check the original file
- Schema validation error → run `pxgpt normalize-schema` and verify the schema is valid JSON
- Model overloaded → re-submit just the failed lines

### JSON parse failures in Stage 3 output

A `{line_id}.err.txt` file contains the raw response. Usually caused by:
- Schema contains unsupported keywords → re-run `pxgpt normalize-schema`
- `output_config.format` schema is invalid → validate with `python -m json.tool schema.json`
- The model ran out of `max_tokens` mid-JSON → raise `STAGE3_MAX_TOKENS`

### Temperature error (400 Bad Request)

This should never happen with pxGPT ≥ 0.3.0 — the temperature guard is enforced centrally. If you see it after manual changes to the code, check that `build_request_params` from `core/batch_utils.py` is being used consistently.

### Batch takes too long / need partial results

Batches can take up to 24 hours for large jobs. `fetch-results` is idempotent — run it as many times as you like; it only writes when `processing_status == "ended"`. You can check status on the Anthropic console at any time using the batch ID printed at submit time.

### Rate limit on image uploads

Reduce `UPLOAD_CONCURRENCY` (e.g. to 5) and re-run. The manifest ensures already-uploaded images are skipped automatically.

### Verbose error output

```bash
pxgpt --verbose describe-batch ...
```

---

## Advanced Usage

### Running stages concurrently

Because both stages use the same manifest and the batch API is asynchronous, you can submit both stages back-to-back immediately after Stage 1 completes the upload phase (the batch itself does not need to finish before Stage 3 uploads):

```bash
# Submit Stage 1
pxgpt describe-batch --input-dir ./images --output descriptions.txt \
  --system-prompt prompts/phenotyping_system.txt \
  --prompt prompts/describe_plant.txt
# → all images are now uploaded; checkpoint_S1.json saved

# Submit Stage 3 immediately (reuses file_ids from manifest)
pxgpt phenotype-batch --input-dir ./images \
  --schema prompts/phenotype_schema.json \
  --output phenotypes/ \
  --system-prompt prompts/phenotyping_system_schema.txt \
  --prompt prompts/extract_traits.txt
# → no uploads; checkpoint_S3.json saved

# Retrieve results for both when done
pxgpt fetch-results --checkpoint checkpoint_<S1_id>.json
pxgpt fetch-results --checkpoint checkpoint_<S3_id>.json
```

### Per-project .env files

```bash
# project_A.env
DEFAULT_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-4-6
STAGE3_EFFORT=high
STAGE1_MAX_TOKENS=32768

source project_A.env && pxgpt describe-batch ...
```

### Integration with HPC job schedulers

For SLURM: wrap each `pxgpt` command in a job script. The batch submission itself is fast (seconds). The long wait is on Anthropic's side, so the SLURM job can exit immediately after `describe-batch` / `phenotype-batch` prints the checkpoint path. Submit a second short job (with a dependency or a manual trigger) to run `fetch-results`.

Example:
```bash
#!/bin/bash
#SBATCH --job-name=pxgpt_submit
#SBATCH --time=00:30:00

module load miniconda3/3.12.4
source activate pxgpt

pxgpt describe-batch \
  --input-dir /data/images \
  --output /results/descriptions.txt \
  --system-prompt prompts/phenotyping_system.txt \
  --prompt prompts/describe_plant.txt

# Checkpoint file is now in the working directory
echo "Batch submitted. Checkpoint: checkpoint_*.json"
```

### Custom configuration per stage

Override any `Config` field via environment variables in the same shell:

```bash
STAGE3_EFFORT=high STAGE3_MAX_TOKENS=32768 \
  pxgpt phenotype-batch --input-dir ./images ...
```

---

## Support and Contributing

### Getting help

1. Check this user manual
2. Enable `--verbose` for full tracebacks
3. Check the [CHANGELOG.md](CHANGELOG.md) for recent breaking changes
4. Open an issue at https://github.com/xavierzheng/pxgpt/issues

### Reporting issues

Include: full error message, pxGPT version (`pxgpt --version`), provider and model, anonymized `.env` (no API keys), checkpoint file if the error is batch-related.

### Citation

```
[Your citation format here]
```

---

**pxGPT** — Empowering plant research through automated phenotyping with Large Language Models.
