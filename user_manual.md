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
| **1 — Descriptions** | ✅ `describe-batch` | Feed multi-angle images per line → rich descriptive text |
| **2 — Schema synthesis** | Manual | Paste Stage 1 output into a conversational LLM session; design a JSON schema that captures the observed variation |
| **3 — Structured phenotyping** | ✅ `phenotype-batch` | Feed the same images + your schema → validated JSON per line |

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

Submit all plant lines for structured extraction. The same manifest from Stage 1 is reused, so no images are re-uploaded:

```bash
pxgpt phenotype-batch \
  --input-dir ./images \
  --schema prompts/phenotype_schema.json \
  --output phenotypes/ \
  --system-prompt prompts/phenotyping_system_schema.txt \
  --prompt prompts/extract_traits.txt \
  --manifest file_manifest.json
```

**API features used:**
- `output_config.effort = "medium"` (configurable via `STAGE3_EFFORT`) — adaptive thinking
- `output_config.format = {"type": "json_schema", "schema": …}` — native structured output; the schema grammar is compiled once and cached across all requests in the batch
- Temperature is **not sent** when effort is set (the API enforces this; the guard is automatic)

**Retrieve results:**
```bash
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json
```

**Output**: one `{line_id}.json` file per plant line in the `--output` directory. If JSON parsing fails for a line, a `{line_id}.err.txt` file is written instead for manual inspection.

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
  [--manifest FILE]      # default: file_manifest.json
  [--wait]               # poll until done; default is fire-and-forget
```

Output file: grouped plain text, one `### {line_id}` section per plant line.

---

### `pxgpt phenotype-batch`

Stage 3 batch structured phenotyping.

```
pxgpt phenotype-batch \
  --input-dir PATH \
  --schema FILE \
  --output DIR \
  --system-prompt FILE \
  --prompt FILE \
  [--manifest FILE]      # default: file_manifest.json
  [--wait]
```

Output directory: one `{line_id}.json` per plant line; `{line_id}.err.txt` for parse failures.

---

### `pxgpt fetch-results`

Retrieve results for any pending or completed batch.

```
pxgpt fetch-results \
  --checkpoint FILE \    # checkpoint_<batch_id>.json written at submit time
  [--output PATH]        # override the output path stored in the checkpoint
```

Prints batch status. If the batch is still processing, exits with a message. If ended, writes results and prints a token-usage summary.

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

### `pxgpt analyze`

Single-folder text description (sync, all providers). Useful for testing prompts on one plant line.

```
pxgpt analyze \
  --input-folder PATH \
  --output FILE \
  --system-prompt FILE \
  --prompt FILE \
  [--provider {anthropic,openai,google,ollama}]
```

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
```

---

## Provider Configuration

### Anthropic Claude (recommended)

```bash
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-4-6

# Thinking effort for Stage 3 and the schema command
STAGE3_EFFORT=medium   # low | medium | high | xhigh | max | ""

# Token budgets
STAGE1_MAX_TOKENS=16384   # up to 65536 on sync; up to 300000 with BATCH_300K_OUTPUT=true
STAGE3_MAX_TOKENS=16384
```

**Prompt caching**: the system prompt is cached with `cache_control: ephemeral` on every request. Repeated Stage 3 runs over the same collection see 50–90 % cache hit rates on the (large) system prompt.

**300 k output tokens** (for very verbose Stage 1 descriptions):
```bash
BATCH_300K_OUTPUT=true
STAGE1_MAX_TOKENS=65536   # or higher, up to 300000
```

### OpenAI / LM Studio

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5-2025-08-07

# LM Studio (OpenAI-compatible local server)
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=gemma3:12b
```

Note: GPT-5 models only accept `temperature=1`; pxGPT handles this automatically.

### Google Gemini

```bash
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-2.5-pro
```

### Ollama (local)

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b
```

Ensure the service is running (`ollama serve`) and the model is downloaded (`ollama pull gemma3:12b`).

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

### Cost optimization

- Prompt caching is automatic for Anthropic: the system prompt is marked `cache_control: ephemeral`. Each Stage 3 batch call uses the cached prompt for all requests after the first.
- Keep system prompts identical across runs (no date stamps, no per-line insertions) so the cache key matches.
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
