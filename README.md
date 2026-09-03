# pxGPT - Plant Analysis Tool

[![CI](https://github.com/xavierzheng/pxgpt/actions/workflows/ci.yml/badge.svg)](https://github.com/xavierzheng/pxgpt/actions/workflows/ci.yml)

**pxGPT** (Phenotype eXplorer GPT) is a command-line tool for large-scale plant phenotyping using multiple LLM providers (Anthropic Claude, OpenAI, Ollama, LM Studio, vLLM).

> **Running locally? Use vLLM.** Ollama and LM Studio work, but neither lets you
> control the per-image visual token budget — and for phenotyping, that budget is
> the measurement. See [Which local backend](#which-local-backend-use-vllm).

## Features

- **Batch API** (Stage 1 & 3): submit hundreds of plant lines in a single API call; fire-and-forget with checkpoint-based result retrieval
- **Files API**: upload each image once, reuse the same `file_id` across Stage 1 and Stage 3 — no re-uploading 10 k images
- **Adaptive thinking** (Stage 3): native `output_config.effort` on claude-sonnet-5; temperature guard enforced automatically
- **Native structured output** (Stage 3): schema passed directly as `output_config.format`; no regex or tag parsing
- **Schema normalizer**: one command adds `additionalProperties: false` and `required` arrays to every object in your schema
- **JSON-to-table flattening**: one command turns the per-plant Stage 3 JSON output into a wide, typed CSV + feather table (ordinal traits reconstructed from level code to label, as an ordered factor in R)
- **Record-level provenance**: every merged Stage 3 record carries a `_provenance` block (provider, model, schema name/version, pxGPT version, run id, timestamp), and `json-to-table` writes `provider`/`model`/`schema_version` columns per row plus the whole block into the feather file's Arrow metadata. A result directory mixing providers, models or schema versions is refused unless you pass `--allow-mixed-provenance`
- **Multiple providers**: Anthropic, OpenAI, Ollama, LM Studio, vLLM — for local inference **vLLM is the recommended backend**; Ollama and LM Studio are supported but not recommended for phenotyping, and are slated for removal in a future major release ([why](#which-local-backend-use-vllm))
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

**To self-host a local model** see [Hosting Gemma 4 on vLLM](ops/local-vllm/README_vllm.md) (operating guide) and [ops/local-vllm/RUNBOOK.md](ops/local-vllm/RUNBOOK.md) (the measurements behind it).

---

## Installation

```bash
git clone https://github.com/xavierzheng/pxgpt.git
cd pxgpt
pip install -r requirements.txt
pip install -e .
cp .env.example project_A.env       # then fill in your API keys
set -a && source project_A.env && set +a
```

### Running the tests

The suite is fast (a few seconds) and needs no API key, no network and no GPU.
`pytest` is **not** a runtime dependency, so it lives in a `dev` extra rather
than in `requirements.txt` — that file becomes `install_requires`, and anything
added to it is forced on every user of the package:

```bash
pip install -e ".[dev]"     # adds pytest; the base install does not
pytest tests/ -q            # 248 tests
```

Two groups are worth knowing about:

```bash
# the local vLLM setup path, without a GPU or a 40 GB download: it copies
# env.example, runs pull.sh, and asserts every gate refuses with an explanation
pytest tests/test_ops_local_vllm_setup.py -q

# schema sharding, provenance and json-to-table
pytest tests/ -q -k "shard or provenance or json2table"
```

There is no CI, so the suite runs when you run it. Worth doing after touching
`ops/local-vllm/` or anything under `pxgpt/core/`.

## Configuration

pxGPT reads its settings from the **process environment** — it does not load a
`.env` file on its own. The file only takes effect once its variables are
**exported**, so wrap the `source` in `set -a` / `set +a`:

```bash
set -a && source project_A.env && set +a   # now exported
pxgpt describe-batch ...
```

Plain `source project_A.env` is not enough: it sets shell variables that
`pxgpt` — a child process — never sees. (Prefixing every line with `export`
works too.)

Variables that never change, typically the API keys, can instead be exported
from your `~/.bashrc` / `~/.zshrc`, leaving only the per-project settings in the
file. Anything left unset falls back to the default compiled into
`pxgpt/core/config.py`; the run banner echoes the values actually in effect, so
check it before a long batch.

Key variables:

```bash
ANTHROPIC_API_KEY=your_key_here
DEFAULT_PROVIDER=anthropic

# Model (default already set to the current recommended model)
ANTHROPIC_MODEL=claude-sonnet-5

# Batch token budgets
STAGE1_MAX_TOKENS=16384   # raise to 65536 for long descriptions
STAGE3_MAX_TOKENS=16384

# Reasoning effort, shared by BOTH providers. default = off = none = NO reasoning.
# (blank, "off", and "none" are equivalent.) Set a level low/medium/high/xhigh/max to
# enable reasoning; the --effort flag overrides per run.
# TEMPERATURE only goes out with reasoning off, and only where the model accepts a
# custom value: Anthropic Sonnet 4.6 and earlier, or OpenAI at effort "none".
STAGE3_EFFORT=     # Stage 3: phenotype-batch + phenotype-batch-openai + schema
DESCRIBE_EFFORT=   # Stage 1: describe-batch + describe-batch-openai
ANALYZE_EFFORT=    # sync analyze command (Anthropic + OpenAI)

# Set true to allow up to 300 k output tokens per response in Stage 1 batches
BATCH_300K_OUTPUT=false

# Parallel image upload threads
UPLOAD_CONCURRENCY=10

# Use the Files API (default true). Set false — or pass --no-files-api — to
# embed images inline as base64 instead of uploading once and reusing file_ids.
USE_FILES_API=true

# Retry / backoff knobs behind "robust error handling" above
MAX_RETRIES=3         # retries after the first attempt (so 4 tries in total)
RATE_LIMIT_SLEEP=60   # flat seconds to wait after a rate-limit error
```

### Local / self-hosted providers (analyze + schema)

Each is a first-class `--provider` value with its own env vars (no need to overload the OpenAI ones). Use a **vision-capable** model since both commands send images.

#### Which local backend: use vLLM

All three work. Only **vLLM** is recommended for phenotyping, and the reason is
visual tokenization.

This tool scores fine-grained traits — petiole cross-section shape, leaf margin
type, colour hue. Whether the model can see those depends on how many visual
tokens each photo is turned into. vLLM makes that a setting you choose:
`--mm-processor-kwargs '{"max_soft_tokens": N}'`, on the ladder
`70 / 140 / 280 / 560 / 1120`. The deployment in
[`ops/local-vllm/`](ops/local-vllm/README_vllm.md) pins **1120**, the top of the
ladder, because it lands close to Anthropic Sonnet 5's per-image tokenization —
which is what makes a local run and a cloud run comparable at all. Left unset,
that checkpoint would default to 280, a quarter of the detail.

**Ollama and LM Studio expose no equivalent control.** Whatever downsampling
they apply is not something you can set, read back, or hold constant across a
backend or model update. That is the objection: not a benchmarked loss — it has
not been measured on either backend — but an uncontrolled variable sitting
underneath every trait you record.

Both remain fully functional and nothing warns at runtime. Both are **slated
for removal in a future major release**. If you measure their tokenization and
it holds up, that decision should be revisited.

```bash
# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b

# LM Studio (OpenAI-compatible)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model          # exactly the name LM Studio shows
LMSTUDIO_API_KEY=lm-studio          # any non-empty placeholder

# vLLM (OpenAI-compatible) — VLLM_MODEL is required
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=gemma4-26b-a4b-nvfp4     # the SERVED name, not the HF repo
VLLM_API_KEY=EMPTY                  # match --api-key if the server sets one
```

`VLLM_MODEL` must equal the server's `--served-model-name`, which is **not** the
checkpoint path (`unsloth/gemma-4-26B-A4B-it-NVFP4`) and has no Ollama-style tag
syntax. Check what is actually served:
`curl -s localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -1`.

Then, e.g.: `pxgpt analyze --provider vllm ...` or `pxgpt schema --provider lmstudio ...`.

> Batch stages (`describe-batch*`, `phenotype-batch*`) are Anthropic/OpenAI-only; the local providers apply to the sync `analyze` and `schema` commands.

### Two `.env` files, and they are not interchangeable

There are two of them, they use **different variable names**, and neither one
feeds the other:

| | `ops/local-vllm/.env` | your shell (or `project_A.env`) |
|---|---|---|
| configures | the **server** | **pxGPT**, the client |
| read by | `up.sh` | pxGPT |
| model name | `SERVED_MODEL_NAME` | `VLLM_MODEL` |
| endpoint | `PORT` | `VLLM_BASE_URL` |
| image root | `MEDIA_ROOT` | *(not read at all)* |

Starting the server does **not** configure pxGPT. `up.sh` sources its `.env`
inside its own process, and a child process cannot export variables back to the
shell that launched it. pxGPT also never loads a `.env` of its own — it reads the
process environment only. So `VLLM_MODEL`, `VLLM_BASE_URL` and `VLLM_API_KEY`
must always be set by you.

**Best practice: derive the client values from the server file, so they cannot
drift apart.**

```bash
# One source of truth: the file that started the server.
set -a; source ops/local-vllm/.env; set +a

export VLLM_MODEL="$SERVED_MODEL_NAME"            # guaranteed to match
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_API_KEY=EMPTY                         # any non-empty string; the
                                                  # server does not check it
export TIMEOUT=1800                               # local runs need this, the
                                                  # 300 s default is too tight

pxgpt schema --provider vllm ...
```

That also carries `TEMPERATURE`, `TOP_P` and `TOP_K` across, which is the one
overlap between the two files — so the client sends exactly the sampling the
server was started with, instead of a second copy you have to remember to update.

Typing the model name by hand works too, but then two files hold the same string
and nothing checks them. If they disagree, the server returns a 404 for a model
it is not serving.

### Quick start: a whole dataset on local vLLM

No local server offers a Batch API, so `schema --shard-dir --input-dir` is the
local Stage 3 runner. Four steps:

```bash
# 1. Start the server (once). MEDIA_ROOT is the ONE line you edit, and it must
#    be a PARENT of your images. Check your hardware first -- see the guide.
cd ops/local-vllm && cp env.example .env && $EDITOR .env
export HF_TOKEN=hf_...                      # https://huggingface.co/settings/tokens
./pull.sh && ./up.sh                        # pull.sh needs HF_TOKEN; ~40 GB

# 2. Point pxGPT at it. TIMEOUT matters: a cold prefill takes 75-95 s.
export VLLM_MODEL=gemma4-26b-a4b-nvfp4 TIMEOUT=1800

# 3. Rehearse on ONE plant before committing hours of GPU time.
pxgpt schema --provider vllm --image-transport file \
  --shard-dir shard_master_schema --input-folder images/s0019 --output /tmp/smoke

# 4. Run the dataset. Re-run the same command to resume; Ctrl-C still merges.
pxgpt schema --provider vllm --image-transport file \
  --shard-dir shard_master_schema --input-dir images --output results
```

`--image-transport file` sends `file://` URIs instead of base64, so the server
reads the images off its own mount. That needs `MEDIA_ROOT` in
`ops/local-vllm/.env` to be a **parent directory** of the images: `up.sh` uses it
both as the bind mount and as `--allowed-local-media-path`. `--input-dir` is an
absolute host path *under* that root — not a path relative to it — and it is
identical inside and outside the container. Outside the tree you get a `400`;
inside it but missing, a `500`. Plain `base64` needs no mount at all.

Dispatch defaults (`--concurrency 8`, `--pipeline-depth 2`, `--mem-floor-gib 5`)
were measured on one GB10 box and are **not** portable — run `--limit 4` first on
new hardware and watch the reported `MemAvail`. Full explanation of every flag:
[user_manual.md](user_manual.md#local--self-hosted-providers-analyze--schema-only).

**Self-hosting a local model?** See [ops/local-vllm/README_vllm.md](ops/local-vllm/README_vllm.md) for a tested, reproducible vLLM deployment of Gemma 4 26B A4B (NVFP4) on a DGX Spark. Start at its **Step 0**, which checks whether your machine qualifies before you download ~40 GB — the pins there are measured on one machine class and do not transfer unchanged.

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

**Step 1 — normalize your schema** (one-time, in-place). This is **your** master
schema, synthesised in Stage 2 with [`prompts/generate_master_schema.txt`](prompts/generate_master_schema.txt).
No master or shard schema ships with pxGPT: yours depends on your traits, your
growth stage and your imaging setup, so a shipped one would be misleading.
```bash
pxgpt normalize-schema --schema master_schema.json
```

**Step 2 — Stage 1 descriptions**:
```bash
pxgpt describe-batch \
  --input-dir ./images \
  --output descriptions.txt \
  --system-prompt prompts/describe_plant_system.txt \
  --prompt prompts/describe_plant_mature.txt     # or _seedling
# Prints batch ID and saves checkpoint_<batch_id>.json
```
When results are fetched, `descriptions.txt` contains grouped descriptions, one section per plant line/cultivar.

**Step 3 — Stage 3 structured phenotyping** (can run concurrently with Stage 1; images are already uploaded). With the Files API, `--input-dir` is optional — the plant lines and their `file_id`s are reused straight from `--manifest`:
```bash
pxgpt shard-schema --master master_schema.json --shard-dir shard_master_schema

pxgpt phenotype-batch \
  --shard-dir shard_master_schema \
  --output phenotypes/ \
  --system-prompt prompts/phenotype_schema_system_mature.txt \
  --manifest file_manifest.json
```

**Why `--system-prompt` here, when `shard-schema` already writes one.**
`shard-schema` emits `shards_system.md` and the run uses it unless you override
it. Override it: the shipped
`prompts/phenotype_schema_system_{mature,seedling}.txt` carry two things the
generated preamble cannot know — the **scale reference for your growth stage**
(a 10 x 10 x 6.5 cm rockwool cube for mature plants, 2.5 cm for seedlings) and
the emphasis on when to answer `not_assessable`. Both change what the model
reports. No `--prompt` is needed with `--shard-dir`: each shard carries its own.

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

**OpenAI.** `pxgpt phenotype-batch-openai` takes the same `--shard-dir`,
`--master-schema`, `--allow-reshard`, `--dispatch` and `--resume/--no-resume`
flags, writes the same `<output>/_partial/` store and produces the same
`{line_id}.json` / `{line_id}.gaps.json`, so one shard set can be scored by both
providers and compared trait-for-trait. Three things differ in practice:

- **The shard *count* is an Anthropic constraint, not an OpenAI one.** `gpt-5.6-luna`
  accepts the whole 49-trait `02_mature_v1` schema in a *single* shard, so on OpenAI
  `--shard-budget` is the main cost lever — every extra shard repeats the whole
  ~30 k-token image payload. `pxgpt shard-schema` is still required either way: it
  is what injects `not_assessable` into every nominal/ordinal enum.
- **Prompt caching recovers only the ~1 k-token system prompt** across a plant's
  shards, not the ~30 k of images, so a fresh sharded OpenAI run pays close to full
  input price per shard.
- **`--no-files-api` does not scale with shards.** 1 plant × 15 images × 10 shards is
  a 200 MB batch input file against OpenAI's 200 MB cap, versus 94 KB through the
  Files API; pxGPT estimates this before encoding and refuses to upload over 190 MB.

Sizing, costs and the measurements behind all three are in
[`dispatch_batch_vs_sequential.md`](dispatch_batch_vs_sequential.md) → *OpenAI*, and
the flags in [`user_manual.md`](user_manual.md) →
*`pxgpt describe-batch-openai` / `pxgpt phenotype-batch-openai` → Sharded mode*.

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

Right after `cultivar_id` every row also carries `provider`, `model` and
`schema_version`, read from that record's `_provenance` block, and the feather
file repeats the full block as Arrow schema metadata under `pxgpt_provenance`.
If the `--result-dir` holds records that disagree on `(provider, model,
schema_version)`, `json-to-table` writes nothing and tells you which records
differ — split the directory, or pass `--allow-mixed-provenance` when the
mixture is deliberate.

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
  --system-prompt prompts/describe_plant_system.txt \
  --prompt prompts/describe_plant_mature.txt

# Structured JSON (uses native structured output for Anthropic)
pxgpt schema \
  --input-folder images/s0001 \
  --output s0001.json \
  --shard-dir shard_master_schema \
  --system-prompt prompts/phenotype_schema_system_mature.txt
```

---

## Commands

| Command | Purpose |
|---------|---------|
| `pxgpt describe-batch` | Stage 1 (Anthropic): upload images via Files API, submit batch for descriptions |
| `pxgpt phenotype-batch` | Stage 3 (Anthropic): reuse file_ids, submit batch with structured output |
| `pxgpt describe-batch-openai` | Stage 1 (OpenAI): same as describe-batch on the OpenAI Batch API |
| `pxgpt phenotype-batch-openai` | Stage 3 (OpenAI): strict structured output on the OpenAI Batch API (single `--schema` or a `--shard-dir` shard set) |
| `pxgpt fetch-results` | Retrieve results for any pending batch (Anthropic or OpenAI) from a checkpoint |
| `pxgpt cleanup-files` | Delete Files-API uploads from a manifest (both providers); OpenAI bills for storage |
| `pxgpt extract-report` | Extract `<report>` from `<think>`/`<report>` output (single or grouped); back-compat for non-native reasoning |
| `pxgpt normalize-schema` | Add `additionalProperties: false` + `required` to all objects in a schema |
| `pxgpt shard-schema` | Split a master schema into compilable Stage 3 shards (+ per-shard prompts) for `phenotype-batch --shard-dir` / `phenotype-batch-openai --shard-dir` |
| `pxgpt json-to-table` | Flatten Stage 3 per-plant JSON results into a wide, typed CSV + feather table, with per-row provenance columns (column-name collision detection/resolution; `--allow-mixed-provenance` to combine mismatched runs) |
| `pxgpt analyze` | Single-folder text description (sync, all providers) |
| `pxgpt schema` | Single-folder structured JSON (sync, all providers) |

Run `pxgpt <command> --help` for full argument details.

---

## Providers

| Provider | Caching | Batch API | analyze / schema | Notes |
|----------|---------|-----------|------------------|-------|
| **Anthropic** (default) | ✅ | ✅ | ✅ | Native thinking, structured output, Files API |
| **OpenAI** | — | ✅ | ✅ | Batch API stages + Files API (`vision`); sync via the OpenAI SDK |
| **Ollama** | — | — | ⚠️ | Works, **not recommended** — no visual-token control ([why](#which-local-backend-use-vllm)); removal planned. OpenAI-compatible (`OLLAMA_BASE_URL` + `/v1`); use a vision model |
| **LM Studio** | — | — | ⚠️ | Works, **not recommended** — same reason; removal planned. OpenAI-compatible (`LMSTUDIO_*`); use a vision model |
| **vLLM** | — | — | ✅ | **Recommended local backend** — pins the per-image token budget. OpenAI-compatible (`VLLM_*`, model required); use a vision model — [hosting guide](ops/local-vllm/README_vllm.md) |

**pxGPT does not support Apple Silicon.** The visual token budget cannot be
pinned there, and an unpinnable tokenizer makes the measurement irreproducible —
the same reason Ollama and LM Studio are not supported. macOS remains a
supported platform for the **test suite**, which needs no GPU and no weights;
CI runs it there on every push.

For `schema`, the JSON schema always reaches the model as a real decoding constraint, on every provider: Anthropic gets native `output_config.format`, the OpenAI-wire providers (OpenAI, Ollama, LM Studio, vLLM) get `response_format` `json_schema` with `strict: true`. It is **not** pasted into the system prompt, so the user prompt does **not** need to ask for JSON-only output. If a backend rejects `response_format` the command fails rather than silently falling back to prompt text.

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
│   ├── openai_compat_provider.py
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
