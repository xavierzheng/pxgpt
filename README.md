# pxGPT - Plant Analysis Tool

**pxGPT** (Phenotype eXplorer GPT) is a command-line tool for large-scale plant phenotyping using multiple LLM providers (Anthropic Claude, OpenAI, Google, Ollama, LM Studio).

## Features

- **Batch API** (Stage 1 & 3): submit hundreds of plant lines in a single API call; fire-and-forget with checkpoint-based result retrieval
- **Files API**: upload each image once, reuse the same `file_id` across Stage 1 and Stage 3 — no re-uploading 10 k images
- **Adaptive thinking** (Stage 3): native `output_config.effort` on claude-sonnet-4-6; temperature guard enforced automatically
- **Native structured output** (Stage 3): schema passed directly as `output_config.format`; no regex or tag parsing
- **Schema normalizer**: one command adds `additionalProperties: false` and `required` arrays to every object in your schema
- **Multiple providers**: Anthropic, OpenAI, Google, Ollama, LM Studio
- **Prompt caching**: automatic for Anthropic (reduces costs on repeated system prompts)
- **Robust error handling**: exponential backoff, per-request failure isolation, crash-safe manifest
- **Example master schema**: see [Example_master_schema.tsv](Example_master_schema.tsv) for the flattened field reference

## Pipeline overview

| Stage | Automated? | Command |
|-------|-----------|---------|
| 1 — per-line descriptions | ✅ | `pxgpt describe-batch` |
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

# Adaptive thinking effort for Stage 3 (low/medium/high/xhigh/max, or "" to disable)
STAGE3_EFFORT=medium

# Set true to allow up to 300 k output tokens per response in Stage 1 batches
BATCH_300K_OUTPUT=false

# Parallel image upload threads
UPLOAD_CONCURRENCY=10
```

### Local providers (LM Studio / Ollama)

```bash
# LM Studio (OpenAI-compatible)
OPENAI_BASE_URL=http://localhost:1234/v1
OPENAI_API_KEY=lm-studio
OPENAI_MODEL=gemma3:12b

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b
```

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

**Step 3 — Stage 3 structured phenotyping** (can run concurrently with Stage 1; images are already uploaded):
```bash
pxgpt phenotype-batch \
  --input-dir ./images \
  --schema prompts/phenotype_schema.json \
  --output phenotypes/ \
  --system-prompt prompts/phenotyping_system_schema.txt \
  --prompt prompts/extract_traits.txt
```

**Step 4 — retrieve results** (once the Anthropic batch finishes, usually within a few hours):
```bash
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json
```

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
| `pxgpt describe-batch` | Stage 1: upload images via Files API, submit batch for descriptions |
| `pxgpt phenotype-batch` | Stage 3: reuse file_ids, submit batch with structured output |
| `pxgpt fetch-results` | Retrieve results for any pending batch from a checkpoint file |
| `pxgpt normalize-schema` | Add `additionalProperties: false` + `required` to all objects in a schema |
| `pxgpt analyze` | Single-folder text description (sync, all providers) |
| `pxgpt schema` | Single-folder structured JSON (sync, all providers) |

Run `pxgpt <command> --help` for full argument details.

---

## Providers

| Provider | Caching | Batch API | Notes |
|----------|---------|-----------|-------|
| **Anthropic** (default) | ✅ | ✅ | Native thinking, structured output, Files API |
| **OpenAI** | — | — | Via LiteLLM; also handles LM Studio |
| **Google Gemini** | — | — | Via LiteLLM |
| **Ollama** | — | — | Local; no API cost |

---

## Project structure

```
pxgpt/
├── core/
│   ├── config.py          # All config with env-var overrides
│   ├── batch_utils.py     # Temperature guard, poll, result writers
│   ├── files_manager.py   # Files API upload + manifest
│   ├── schema_utils.py    # JSON schema normalizer
│   ├── image_utils.py     # Base64 + file_id content builders
│   └── file_utils.py      # File I/O helpers
├── providers/
│   ├── anthropic_provider.py
│   ├── litellm_provider.py
│   └── base.py
├── commands/
│   ├── describe.py        # describe-batch
│   ├── phenotype.py       # phenotype-batch
│   ├── fetch_results.py   # fetch-results
│   ├── normalize_schema.py
│   ├── analyze.py
│   └── schema.py
└── main.py
```

## License

MIT License
