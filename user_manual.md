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
- **Reliability**: fire-and-forget batches with checkpoint files; per-request failure isolation; crash-safe manifest; **crash-safe, resumable sequential dispatch** for Stage 3 sharded runs (partials on disk, skip-completed resume, in-run transient retry)

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

pxGPT reads its settings from the process environment; it does not load a `.env`
file on its own. Copy the template, fill it in, then **export** it:

```bash
cp .env.example project_A.env
vim project_A.env                       # add ANTHROPIC_API_KEY at minimum
set -a; source project_A.env; set +a    # plain `source` is not enough — see below
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

Supported image formats: `.jpg`, `.jpeg`, `.png` (upper-case extensions are fine too). `.gif` and `.webp` are not supported.

### 4. Normalize your schema (one-time)

Before the first Stage 3 run, normalize your JSON schema to meet the Anthropic structured-output requirements:

```bash
pxgpt normalize-schema --schema master_schema.json
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
  --system-prompt prompts/describe_plant_system.txt \
  --prompt prompts/describe_plant_mature.txt \
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

This stage is intentionally kept manual. Open a conversational LLM session (Claude.ai with extended thinking recommended or claude code) and paste the contents of `descriptions.txt`.

* If use claude code, I highly suggest to install the skill superpowers:dispatching-parallel-agents to fan out subagent to perform this task.
* superpowers: https://github.com/obra/Superpowers

**Prompt template:**

```
# CONTEXT (read first — this drives every rule below)
You are generating a MASTER PHENOTYPE SCHEMA for vision-based plant phenotyping.
The downstream consumer is a MULTIMODAL LLM that will look at plant PHOTOGRAPHS and assign a value for each trait. Therefore every trait, every category, and every level must be something that LLM can reliably DISCRIMINATE FROM IMAGES. A category that is botanically real but not visually distinguishable in a photo is useless here and must be merged or dropped. Optimize for: canonical reusable trait names, small mutually-exclusive
visually-distinct value sets, and biologically coherent organization. Exhaustiveness means "cover every real trait", NOT "list every synonym as a separate category".

You are a professional botanist and data scientist. From the provided document (phenotyping reports of individual cultivars), generate a SINGLE master JSON schema.

OUTPUT: a JSON schema DEFINITION (a template), not a filled-in example for any one plant.
Output only the JSON, no prose preamble.

## TOP-LEVEL CONTAINER (exact shape — mandatory, do not vary)
The downstream sharding tool keys groups by NAME, so groups MUST be a JSON OBJECT
named "trait_groups" (NOT an array, and NOT named "groups"). Emit exactly this
top-level shape:
```
{
  "schema_name": "<short identifier>",
  "schema_version": "<version string>",
  "description": "<one-line description of the schema>",
  "trait_groups": {
    "<group_name_snake_case>": {
      "description": "<what this organ/functional group covers>",
      "traits": [ <per-trait object>, <per-trait object>, ... ]
    },
    "<next_group_name>": { "description": "...", "traits": [ ... ] }
  }
}
```
- "trait_groups" is an OBJECT whose KEYS are the group names (snake_case). Do NOT
  emit a list of `{"group": name, "traits": [...]}` objects, and do NOT put the
  group name inside the group object.
- Each group value is an object with EXACTLY two keys: "description" (string) and
  "traits" (array of per-trait objects defined below).
- Any extra top-level metadata keys you add are ignored downstream, but the
  "trait_groups" object and its structure above are required verbatim.

## TRAIT GROUPING (biological coherence — mandatory)
- Organize traits into top-level groups that each correspond to ONE coherent anatomical structure or functional/developmental unit. Examples of valid groups (use only those the data warrants): whole_plant_architecture, root_system, stem, leaf_blade, leaf_margin, leaf_apex_base, leaf_surface, petiole, venation, inflorescence, flower, fruit, seed, phenology.
- A trait must sit in the group matching the organ/structure it describes.
- FORBIDDEN: catch-all groups that mix unrelated organs or functions. Do NOT create a group like "health_medium_taxonomy_reproductive". If a trait fits no organ group, create a small well-defined functional group (e.g. "phenology"), never a mixed bucket.

## TRAIT NAMING (ontology-style, canonical)
- Use canonical, reusable, ontology-style names (e.g. leaf_blade_shape, leaf_margin_type, flower_color_hue, petiole_length). Lowercase snake_case, organ_attribute pattern.
- The name must be reusable across cultivars and species — never encode one cultivar's specific value into the trait name.
- GLOBALLY UNIQUE across the whole schema: no two traits may share the same trait_name, even under different groups. Rationale: the downstream pipeline flattens this schema into atable using trait_name as the column name, so any repeated name would collide and silently overwrite data. The organ_attribute pattern above already guarantees this when applied infull (leaf_blade_length vs petiole_length never clash); the failure mode to avoid is a bare attribute name ("length", "width", "color", "shape") recurring under several organs. Always keep an organ prefix specific enough that the name self-disambiguates.


## TRAIT CLASSIFICATION — assign each trait a "scale_type":
- "nominal": discrete categories with NO inherent order (e.g. leaf shape, color hue).
- "ordinal": a feature with a GENUINE inherent order (e.g. intensity, relative size, area-proportion bands).
- "quantitative": a directly measurable continuous value (counts, lengths). Give a standardized "unit"; do NOT bin into an enum.

## NOMINAL VALUE RULES (this is where over-enumeration must be stopped)
- Consolidate synonyms and near-duplicates onto ONE canonical controlled-vocabulary term. Map observed surface variants (e.g. "violet", "purplish", "magenta-purple") onto the nearest canonical category; do NOT create a separate category per wording.
- Every category must be (a) mutually exclusive, (b) visually discriminable from a photo by a non-expert, and (c) biologically meaningful.
- Every canonical category must be grounded in >=1 actual observation in the document. Consolidating variants is required; inventing unobserved categories is forbidden.
- If a nominal trait exceeds ~7 categories, treat it as a red flag: you are almost certainly splitting synonyms or encoding non-visual distinctions. Consolidate unless each category is genuinely visually distinct AND biologically warranted.

## COLOR HANDLING (decompose — do not let color explode)
- NEVER create composite color categories (e.g. "light-purplish-green"). Composites are the main cause of enum explosion.
- Decompose any color trait into separate sub-traits, each small:
    *_hue       -> nominal, a BASIC canonical palette only (e.g. white, yellow, orange, red, pink, purple, blue, green, brown). Map shades onto the nearest               basic hue; do not enumerate shade words.
    *_intensity -> ordinal, <=5 levels (this version uses LLM perception, no instrument).
    *_coverage  -> ordinal area-proportion bands, <=5 levels, only if the document describes how much area the color occupies.

## ORDINAL LEVEL RULES
- Default max 5 levels; hard max 7. Prefer an ODD number (3, 5, or 7) so midpoint and both extremes are semantically anchored.
- Every level needs an explicit semantic definition, not a bare number.
- If a trait truly needs >5 levels, justify it in "design_note"; if you cannot justify it, reduce the levels.

## PER-TRAIT STRUCTURE — each trait is an object with:
- "trait_name": canonical ontology-style name
- "description": what the trait captures
- "scale_type": "nominal" | "ordinal" | "quantitative"
- "values":
    nominal      -> array of objects, each { "value": canonical category string,
                    "definition": ONE clause defining that category VISUALLY —
                    what the scorer literally sees in a photo. No population/
                    frequency words (see rule below). }
    ordinal      -> ordered array of { "level": int, "label": str, "definition": str }
    quantitative -> null
- "unit": standardized unit string for quantitative traits; otherwise null
- "support": integer count of cultivars exhibiting this feature
- "design_note": brief rationale for the chosen scale_type, categories, and levels; mandatory justification if a nominal trait exceeds 7 categories or an ordinal exceeds 5.
- The per-category "definition" text (and ordinal level "definition"s) are shown
  VERBATIM to a downstream multimodal LLM that has NO other context. They must be
  self-contained visual definitions. They MUST NOT contain population or dataset
  frequency information — no "most", "rare", "near-universal", "overwhelmingly",
  "N of cases", cultivar ids, or support counts. Put any such rationale ONLY in
  "design_note", which is author-only and is never shown to the scorer.

## COVERAGE
- Include every trait, even one appearing in a single cultivar. Record its "support".
- Coverage applies to TRAITS, not to synonym-level categories. Cover every real trait; consolidate categories.

## SCHEMA HYGIENE
- Use only basic JSON Schema constructs (object, array, string, number, integer, boolean, enum). Nesting (organ -> sub-traits) is allowed and encouraged. Do NOT use recursive or self-referential schemas, external $ref, minLength/maxLength, or minimum/maximum, so the schema stays compatible with grammar-constrained structured output downstream.

## FORMAT ANCHOR (illustrative only — derive ALL real content from the document)
A good leaf color treatment looks like:
  leaf_color_hue (nominal), with values:
    [ { "value": "green",  "definition": "lamina predominantly green" },
      { "value": "purple", "definition": "lamina predominantly purple/anthocyanic" } ]
  leaf_green_intensity (ordinal, 3 levels: pale / medium / deep, each defined)
NOT a single nominal trait "leaf_color" with values
  [light green, medium green, dark green, purplish green, green-purple, ...],
and NOT a bare nominal value list like ["green", "purple"] with no definitions.

All the phenotyping reports are combined into [paste your describing output file name here]. Use this file as your phenotyping reports source.
```

**Iterate** until the schema covers all observed variation. Save the final schema to a file (e.g. `master_schema.json` — this is yours, and nothing equivalent ships with pxGPT), then normalize it:

```bash
pxgpt normalize-schema --schema master_schema.json
```

---

### Stage 3 — Batch structured phenotyping

Submit all plant lines for structured extraction. The same manifest from Stage 1 is reused, so no images are re-uploaded. With the Files API (default), `--input-dir` is **optional**: omit it and the plant lines plus their `file_id`s are reconstructed straight from `--manifest`, so the image tree need not even be present on disk:

```bash
pxgpt shard-schema --master master_schema.json --shard-dir shard_master_schema

pxgpt phenotype-batch \
  --shard-dir shard_master_schema \
  --output phenotypes/ \
  --system-prompt prompts/phenotype_schema_system_mature.txt \
  --manifest file_manifest.json
```

Pass `--input-dir ./images` as well if you want Stage 3 to also pick up (and upload) any images added since Stage 1. `--input-dir` **is required** with `--no-files-api`, because inline base64 mode must read the image bytes from disk.

**API features used:**
- `output_config.format = {"type": "json_schema", "schema": …}` — native structured output; the schema grammar is compiled once and cached across all requests in the batch
- `output_config.effort` — adaptive thinking, **off by default**. Stage 3 then runs without reasoning; whether a custom `temperature` goes out depends on the model tier — sent on Sonnet 4.6 and earlier, omitted on `claude-sonnet-5` and newer (which instead receive an explicit `thinking: {"type": "disabled"}`). Enable reasoning by setting `STAGE3_EFFORT` (e.g. `medium`).
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
- **Prompt caching**: pxGPT marks only the shared system block as a cache breakpoint. Images remain ordinary input and stay before the per-shard text prompt. Every shard uses a different `output_config.format` schema, so a smaller same-schema system/format prefix may still be reused across plants without repeatedly cache-writing the much larger image input. Image tokens are therefore reported under ordinary `input_tokens`, not `cache_creation_input_tokens`. See [`dispatch_batch_vs_sequential.md`](dispatch_batch_vs_sequential.md) for details.
- **Dispatch**: `batch` (default) submits one asynchronous Message Batch for all *(plant × shard)* requests. `sequential` sends plant-contiguous synchronous calls and adds incremental output writes, automatic resume, bounded retry and live progress logging. Dispatch mode does not change the cache layout: only the system block is cached, while images remain ordinary input. See [`dispatch_batch_vs_sequential.md`](dispatch_batch_vs_sequential.md) for the transport and cache differences.
- **Sequential dispatch is crash-safe and resumable.** A real sequential run is `plants × shards` synchronous calls (easily thousands, many hours), so a SLURM wall-time kill, node crash or OOM must not throw the work away:
  - Each shard's parsed JSON is written to `<output>/_partial/<line_id>__<shard_id>.json` the instant it returns, and — because requests are plant-contiguous — a plant's final merged `<line_id>.json` is written as soon as its last shard is attempted. `<output>/_partial/progress.jsonl` records one line per completed call. The `_partial/` directory is left in place after a successful run (inspect or delete it yourself).
  - **Resume is automatic** (`--resume`, default on): re-run the *same* command and any shard whose partial already exists and parses is skipped — you are **not re-billed** for completed calls — while failed/unparseable shards (which wrote no partial) are retried. The run logs how many calls it skipped, and the closing token summary counts only the calls made in *this* run. Pass `--no-resume` to force a clean run that ignores existing partials.
  - **Transient errors are retried in-run**: `429` / `5xx` / Anthropic's `529` "Overloaded" / connection blips get up to 3 attempts with exponential backoff before a shard is dropped. A `400` such as *"Grammar compilation timed out"* is a schema-size error and is **not** retried in-run — reduce `--shard-budget` and re-shard, or rely on resume.
  - **Live logging**: progress lines appear in the SLURM/stdout log as the run proceeds (stdout is line-buffered by the CLI; no `PYTHONUNBUFFERED` needed).
  - A clean, uninterrupted sequential run produces exactly the same `<line_id>.json` / `<line_id>.gaps.json` files as before — only the `_partial/` directory is new.
  - **The `_partial/` store guards its own identity in `<output>/_partial/.run.json`**, checked before every merge (sequential dispatch, batch `fetch-results`, and the local `pxgpt schema --shard-dir` path share the same guard). It refuses to reuse a store that a different provider or model created, and now also records `schema_name`/`schema_version` and refuses a store whose `schema_version` differs from this run's — same model, different schema means the traits no longer line up, and merging those partials would build a record that never came from one schema. Either refusal names what the store was created with vs. what this run uses and gives the same two remedies: point `--output` at a different directory, or delete `<output>/_partial/.run.json` if you are certain the existing partials belong to this run. Two things are tolerated rather than refused: a legacy stamp with no `schema_version` key (one warning; the field is filled in from this run) and a run that cannot name its own schema version (`schema_version` is `null` — e.g. local `--shard-dir` mode, whose merge index comes from the manifest and never opens a master schema) — nothing to compare, so the stamp is left alone.

**Step 3 — retrieve + merge:**

```bash
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json
```

For a sharded run, `fetch-results` demultiplexes the `custom_id = "<line_id>__<shard_id>"` results, merges each plant's shards into one record keyed by the master organ structure, parses quantitative strings to numbers, and validates coverage against the master schema. Each succeeded shard is also written to `<output>/_partial/<line_id>__<shard_id>.json` (the same store the sequential resume reads), and the merge uses the union of any partials already there plus this batch — so re-running `fetch-results` is idempotent.

**Output**: one merged `{line_id}.json` per plant. If any trait is missing or a shard errored, a `{line_id}.gaps.json` is written alongside it listing the missing traits and shard errors.

**Step 4 (only if the batch left gaps) — recover the failed shards.**

#### What a gap looks like

After Step 3 you may see `*.gaps.json` files next to some records:

```
Result_Stage3/
├── s0004.json          ← merged record (missing one shard's traits)
├── s0004.gaps.json     ← ← the gap report
├── s0097.json
├── s0097.gaps.json
└── ...
```

Open one — the `shard_errors` field names exactly which shard(s) failed and why:

```json
{
  "line_id": "s0097",
  "missing_traits": [
    { "group": "stem",      "trait": "stem_elongation" },
    { "group": "phenology", "trait": "plant_developmental_stage" }
  ],
  "shard_errors": [
    "shard_09: overloaded_error: File storage is temporarily unavailable. Please retry.",
    "shard_02: overloaded_error: File storage is temporarily unavailable. Please retry."
  ]
}
```

That `overloaded_error` is a **transient** Files-API blip. In `batch` dispatch the
Batch API cannot re-run one request, so the shard is stuck errored: re-fetching the
same batch just reproduces the identical `*.gaps.json`, and `--resume` has no effect
on a batch. To fill the gaps you must issue *new* API calls for the failed shards —
which is what `--dispatch sequential` does (it retries transient errors in-run).

#### The two commands

You need the same shard set, image manifest and master schema the batch used — they
are all recorded in the `checkpoint_<batch_id>.json` from Step 2/3. Run **both**
commands from the directory that holds `Result_Stage3` (so the relative paths
resolve), and re-use the **same** `--output` both times.

```bash
# 4a. FREE — re-download the completed batch so every SUCCEEDED shard is saved
#     into Result_Stage3/_partial/. (Needed only for batches fetched before this
#     partial-persistence behavior existed; a fresh Step-3 fetch already did this.)
pxgpt fetch-results --checkpoint checkpoint_msgbatch_01N37xDTe4Tz8GkWUVSCmFrY.json

# 4b. Re-issue ONLY the still-missing shards. Resume skips everything already in
#     _partial/, so this bills just the handful of failed shards, not the whole set.
#     Set the SAME model/effort the original batch used so recovered shards match.
export ANTHROPIC_MODEL=claude-sonnet-5 STAGE3_EFFORT=medium
pxgpt phenotype-batch \
    --shard-dir shard_master_schema \
    --manifest  file_manifest.json \
    --master-schema master_schema_v2.json \
    --output    Result_Stage3 \
    --dispatch  sequential
```

Step 4b prints how many calls it skipped vs. ran, and ends with a gap count:

```
--- Resume: 1410 of 1420 shard(s) already on disk; skipping those calls ---
--- Sequential dispatch: 1420 call(s) (1410 skip, 10 to run) ---
  [34/1420] s0004__shard_04  cache_read=0 cache_creation=1285
  ...
  [1153/1420] s0150__shard_03 cache_read=0 cache_creation=2216

  Wrote 142 merged JSON files; 0 call error(s) this run; 1410 shard(s) skipped;
  0 plant(s) with gaps (0 missing traits)
```

Each recovered shard is written into `_partial/`, its plant is re-merged, and the
plant's `*.gaps.json` is **deleted** once its traits are filled.

> **⚠ Match the original run's request settings exactly.** The recovered shards
> must be produced under the *same* conditions as the other shards, or your dataset
> becomes internally inconsistent. Copy the settings from the batch's original
> `step_04_phenotyping.sh` (and its environment), not from a different pipeline's:
> - **`--system-prompt`**: if the original passed a `--system-prompt` override, pass
>   the *same* file. Omitting it silently falls back to the shard set's
>   `shards_system.md`, which is usually different content.
> - **`STAGE3_EFFORT` / `ANTHROPIC_MODEL`**: match them. If you are unsure whether the
>   original ran with reasoning on, check a succeeded request in the batch — a
>   `thinking` content block (or notably larger `output_tokens`) means effort was on;
>   only `text` blocks means effort was off. A run with no `.env` and no exported
>   `STAGE3_EFFORT` used the default (**off**).
>
> The run banner echoes what it will use — verify `system prompt: ...` and
> `output_config.effort: ...` before letting it issue calls. If you already recovered
> with the wrong settings, delete just those `_partial/<line_id>__<shard_id>.json`
> files and re-run with the correct ones (resume re-issues only the deleted shards).

#### Confirm it worked

```bash
ls Result_Stage3/*.gaps.json 2>/dev/null | wc -l   # expect 0
```

If a shard is still failing (e.g. a genuine schema error, not a transient overload),
its `*.gaps.json` remains with the reason in `shard_errors` — fix that cause (for
grammar-size errors, lower `--shard-budget` and re-shard) and re-run Step 4b.

> **Note — future batches recover in one step.** From now on, the Step-3
> `fetch-results` already writes every succeeded shard into `_partial/`, so you can
> skip Step 4a and just run Step 4b to fill any gaps.

---

### Downstream analysis

Each per-plant JSON is `{group: {trait: {rationale, value}}}`, and `value` is
**not** analysis-ready as-is: ordinal traits store the integer level *code*
(not the label), quantitative traits carry no unit in the column, and a naive
`json_normalize` bakes in the `rationale`/`value` nesting and the `not_assessable`
sentinel as a literal string rather than a real NA.

**Every per-plant JSON also carries a `_provenance` block**, written first in the
file: `provider`, `model`, `schema_name`, `schema_version`, `pxgpt_version`,
`created` (UTC `YYYY-MM-DDTHH:MM:SSZ`), `run_id`. `schema_name`/`schema_version`
come from the master schema's own two top-level fields (`null` wherever the run
never opens one — `shards_manifest.json` is *not* a source; its `version` field is
the manifest format version, not a schema identity), and `run_id` is the batch id
on batch paths, `null` on sequential/local paths. It is recomputed at write time,
so re-merging (an idempotent `fetch-results`, a sequential resume) always leaves
exactly one, current block behind. `_provenance` is metadata, not a trait group —
skip keys starting with `_` if you iterate a record's top-level keys yourself.
Every writer stamps it: the shared sharded merge (both providers, batch and
sequential dispatch), the unsharded per-plant batch writers, and the local `pxgpt
schema --shard-dir` / `--schema` paths (for `--schema`, only a response that
parses as a JSON object gets the block; anything else is written verbatim).

**Recommended: `pxgpt json-to-table`** — flattens the whole result directory into
a wide, typed, analysis-ready table in one command, using the master (+ shard)
schema to reconstruct ordinal labels, unit-suffix quantitative columns, and
encode missing/`not_assessable` values as real NA:

```bash
pxgpt json-to-table \
  --result-dir phenotypes/ \
  --master-schema master_schema.json \
  [--shard-dir master_schema_generation/shards] \  # fallback for traits absent from master
  --out-prefix analysis/stage3_table
# Writes analysis/stage3_table.csv and analysis/stage3_table.feather
```

- One row per plant line/cultivar; `cultivar_id` (the filename stem) is the first column.
- Immediately after `cultivar_id`, three provenance columns — `provider`, `model`,
  `schema_version` — read per row from that record's `_provenance` block. Records
  from an older pxGPT with no `_provenance` fall back to
  `<result-dir>/_partial/.run.json` when one is present, else the three columns
  are NA; either way the flattener still writes the table, with one warning.
- **nominal** → column = trait name; value = plain string (character in both outputs — never a factor/category).
- **quantitative** → column = `<trait>_<unit>` (e.g. `plant_height_cm`; unit sanitized — `m²` → `m2`); numeric.
- **ordinal** → column = trait name; the integer level code is reconstructed into its schema label (e.g. `1` → `"mild"`). In the CSV this is a plain label string; in the feather file it's an **ordered** `pandas.Categorical` over the full schema-defined level set, so R's `arrow::read_feather()` reads it as an ordered factor.
- Missing traits and the `not_assessable` sentinel become real NA in every column type; for ordinal columns, NA is never added as a spurious category level.
- The column set is the union of every trait seen across all files, in a deterministic order (master schema order, then any shard-only fallback traits, then unknown traits — each logged as a warning).

**Column name collisions.** A column name is normally just the trait's leaf
key (plus `_<unit>` for quantitative traits). If the master schema ever
assesses the *same* leaf key under two different organ groups (e.g. `length`
under both `leaf` and `petal`), both would compute the same final name —
by default `json-to-table` refuses to silently let one overwrite the other.
`cultivar_id`, `provider`, `model` and `schema_version` are also reserved: a
trait that resolves onto one of them is a collision too, in *every*
`--on-collision` mode (including the prefix ones) — those columns carry the
row's own identity and are never silently renamed out from under the reader.
Use `--rename-map` to give the trait a different name instead. A plain leaf-key
clash between two traits looks like this:

```
$ pxgpt json-to-table --result-dir phenotypes/ --master-schema master_schema.json --out-prefix analysis/stage3_table
Column name collision(s). Unresolved:
 'length_cm' <- leaf.length (quantitative, unit=cm), petal.length (quantitative, unit=cm)

Fill in this rename map (path -> column name) and re-run with --rename-map:
 {
 "leaf.length": "",
 "petal.length": ""
 }
```

No files are written when this happens. Fix it one of two ways:

- Fill in the printed template and re-run with `--rename-map` (worked example below).
- Or skip hand-naming and pass `--on-collision prefix_collided` to auto-prefix *only* the clashing columns with the minimal group-path prefix needed to disambiguate (`leaf_length_cm`, `petal_length_cm`); every other column keeps its short name. `--on-collision prefix_all` is a blunter escape hatch that prefixes *every* column with its full path, colliding or not.

**Example: resolving a collision with `--rename-map`.** Take the printed
template from the error above, save it as a file, fill in the column names
you want for each clashing *path* (the map is keyed by the dotted
`group.trait` path, not the colliding name itself — that's what makes the
two `length` traits distinguishable):

```bash
cat > rename_map.json <<'JSON'
{
  "leaf.length": "leaf_length_cm",
  "petal.length": "petal_length_cm"
}
JSON

pxgpt json-to-table \
  --result-dir phenotypes/ \
  --master-schema master_schema.json \
  --out-prefix analysis/stage3_table \
  --rename-map rename_map.json
```

```
--- Flattening phenotypes/ -> analysis/stage3_table.{csv,feather} ---
  Rows: 42   Columns: 4
  Wrote analysis/stage3_table.csv
  Wrote analysis/stage3_table.feather
```

The resulting table has `leaf_length_cm` and `petal_length_cm` as separate
columns; `flower.color`, which never collided, keeps its short name `color`
untouched. A value you supply in `--rename-map` is used **verbatim** — for a
quantitative trait, include your own unit suffix (`_cm`, `_mm`, ...); pxGPT
does not re-append one. `--rename-map` entries take priority over
`--on-collision`, so you can rename just the columns you care about and let
`--on-collision prefix_collided` handle any others left clashing (pxGPT
still refuses to write files if the map itself introduces a new duplicate,
e.g. mapping two different paths to the same name).

Traits with the same leaf key but genuinely different units (e.g. `stem.length` in cm vs `hair.length` in mm) are never flagged — they already compute to distinct final names. Whichever mode you use, a final uniqueness check always runs before any file is written, so a bad `--rename-map` that itself introduces a duplicate is caught too.

See `pxgpt/core/json2table.py` for the flattening logic if you need to call it as a library from a notebook instead of the CLI.

**Mixed provenance is refused by default.** If `--result-dir` holds records from
more than one run — different provider, different model, or the same model
scored against a different `schema_version` — `json-to-table` stops before
writing anything and lists each distinct `(provider, model, schema_version)`
tuple with the cultivar ids that carry it:

```
Refusing to flatten records from more than one run:
  provider='anthropic' model='claude-sonnet-5' schema_version='2.1'  <- 38 record(s): s0001, s0002, s0003, s0004 ...
  provider='openai' model='gpt-5.6-luna' schema_version='2.1'  <- 4 record(s): s0039, s0040, s0041, s0042
```

One table whose rows come from different models reads as one experiment and is
not one. Pass **`--allow-mixed-provenance`** if the mixture is deliberate — the
table is written anyway, and the per-row `provider`/`model`/`schema_version`
columns carry the truth.

**The feather file also repeats the whole provenance block** as Arrow schema
metadata, under the key `pxgpt_provenance` (JSON-encoded bytes) — so it survives
even if the three columns are later dropped. It is added to, never in place of,
the `pandas` metadata key pandas itself writes (that key is what makes an ordinal
column come back as an ORDERED Categorical / R ordered factor):

```python
import json, pyarrow.feather as pf
md = pf.read_table("analysis/stage3_table.feather").schema.metadata
print(json.loads(md[b"pxgpt_provenance"]))
```

The value is the single provenance block when every record agrees; with
`--allow-mixed-provenance` and a genuine mixture, it is instead
`{"mixed": true, "values": [<distinct blocks>]}`.

**Reading the outputs downstream (Python):**
```python
import pandas as pd
df = pd.read_csv("analysis/stage3_table.csv")            # ordinal cols are plain label strings
df = pd.read_feather("analysis/stage3_table.feather")     # ordinal cols are ordered pd.Categorical
```

**R:**
```r
library(arrow)
df <- read_feather("analysis/stage3_table.feather")
# ordinal columns come back as ordered factors; nominal columns as character
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

By default Stage 1 runs **without reasoning** (the model's whole response is the description — no `<think>`/`<report>` tags needed); a custom `temperature` is sent only on Sonnet 4.6 and earlier. Enable Anthropic adaptive thinking with `--effort` (e.g. `--effort medium`) or by setting `DESCRIBE_EFFORT`; thinking blocks are produced natively and stripped from the saved description.

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
  [--resume | --no-resume]          # sequential dispatch: resume from <output>/_partial/
                                    # instead of re-running completed calls (default: --resume)
  [--wait]
```

Output directory: one `{line_id}.json` per plant line; `{line_id}.err.txt` for parse failures (single-schema mode) or `{line_id}.gaps.json` for missing traits (sharded mode). See **Stage 3 (sharded)** above for when and how to use `--shard-dir`.

`--dispatch batch` (default) is a single async Message Batch at 50% off standard pricing but with unreliable cross-shard prompt caching; `--dispatch sequential` runs full-price synchronous calls that reliably hit the cache. Which is cheaper depends on your shard count and batch's actual cache-hit rate — see [`dispatch_batch_vs_sequential.md`](dispatch_batch_vs_sequential.md) for the functional difference and a cost-crossover model.

`--dispatch sequential` is **crash-safe and resumable**: shards are written to `<output>/_partial/` as they complete, so re-running the same command after a kill/crash skips completed calls (no re-billing) and retries only what's missing (`--no-resume` forces a fresh run). Transient overloads are retried in-run and progress prints live. See **Stage 3 (sharded)** above for details. `--resume` has no effect in `batch` dispatch (a batch is retrieved via `fetch-results` + the checkpoint). If a batch *errors* some shards (e.g. a transient `overloaded_error`), those requests are terminal inside the Batch API; recover them by running `--dispatch sequential` to the same `--output` — `fetch-results` leaves the succeeded shards in `<output>/_partial/`, so the sequential resume re-issues only the failed shards. See **Stage 3 (sharded) → Step 4** above.

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
  [--effort {off,low,medium,high,xhigh,max}]   # overrides DESCRIBE_EFFORT
  [--wait]

pxgpt phenotype-batch-openai \
  --input-dir PATH \
  --output DIR \
  (--schema FILE | --shard-dir DIR)   # mutually exclusive; exactly one is required
  [--system-prompt FILE]   # required with --schema; with --shard-dir it OVERRIDES
                           #   the shard set's shared system preamble
  [--prompt FILE]          # required with --schema; ignored with --shard-dir
                           #   (per-shard prompts come from the shard set)
  [--master-schema FILE]   # sharded: defaults to the path in shards_manifest.json
  [--allow-reshard]        # sharded: OVERWRITES the files in --shard-dir (see below)
  [--dispatch {batch,sequential}]   # sharded; default: batch
  [--resume | --no-resume]          # sequential dispatch only; default: --resume
  [--manifest FILE]
  [--no-files-api]
  [--wait]
```

Key differences from the Anthropic stages:

- **Model**: uses `OPENAI_MODEL` (default `gpt-5.6-luna`).
- **Files API**: images are uploaded with OpenAI's `purpose="vision"` and referenced by `file_id`. Because OpenAI and Anthropic file_ids are different namespaces, the OpenAI manifest defaults to a **separate file** (`openai_file_manifest.json`) — do not point it at the Anthropic `file_manifest.json`.
- **Structured output** (`phenotype-batch-openai`): the schema is normalized in memory for OpenAI **strict** mode — every property is forced into `required`, `additionalProperties: false` is set on every object, and an all-string `enum` with no `"type"` gets `"type": "string"` (stricter than `pxgpt normalize-schema`, which targets Anthropic). The file on disk is not modified. With `--shard-dir` the same normalization is applied to each shard schema — see **Sharded mode** below.
- **Reasoning effort**: the same knobs as the Anthropic stages — `DESCRIBE_EFFORT` for Stage 1 (with `--effort` to override per run) and `STAGE3_EFFORT` for Stage 3 (no flag; the variable is the only control). Levels `low`/`medium`/`high`/`xhigh`/`max`; empty/`off`/`none` means reasoning **off**. There is no OpenAI-specific effort variable.
- **Reasoning off is explicit**: "off" is sent as `reasoning: {"effort": "none"}`. Omitting the parameter would *not* disable reasoning — the model falls back to its own default (`medium` on gpt-5.6) — so pxGPT always sends a level. Verified: `effort=none` returns `reasoning_tokens=0`.
- **Temperature**: `TEMPERATURE` is sent **only** when effort is `none`; any other level rejects it with `400 Unsupported parameter: 'temperature' is not supported with this model.`, so pxGPT omits it there. The run banner prints the effort and what happened to temperature.
- **Completion window**: `OPENAI_BATCH_COMPLETION_WINDOW` (default `24h`).

Checkpoints are tagged with `"provider": "openai"`, so `pxgpt fetch-results` retrieves them the same way as Anthropic batches — including the sharded stage, which merges the shards into one record per plant.

#### Sharded mode (`--shard-dir`)

Same shard set, same flags and same output layout as `phenotype-batch`: one request per (plant × shard), merged into one `{line_id}.json` per plant with `{line_id}.gaps.json` for anything still missing. So a single shard set can be scored by both providers and compared trait-for-trait — which is the point of running it this way.

`--dispatch batch` and `--dispatch sequential` send **identical request bodies** (the batch JSONL only adds the `custom_id` / `method` / `url` envelope), and both write the same `<output>/_partial/<line_id>__<shard_id>.json` store. So a batch that left gaps is recovered by a sequential resume to the same `--output`, exactly as on the Anthropic side, and a resume re-bills only the missing shards.

Four things are OpenAI-specific:

- **The shard *count* is not forced here, but `pxgpt shard-schema` still is.** The grammar-size limit that forces sharding is Anthropic's: `gpt-5.6-luna` accepts all 49 traits in a *single* shard (measured at 159 object properties, 165 enum values and depth 4, against OpenAI's limits of 5,000 / 1,000 / 10). What you cannot skip is the generator, because it is what *injects* `not_assessable` into every nominal and ordinal enum — in the 02_mature_v1 master, 45 of 45 such traits get it added and none list it themselves. So raising `--shard-budget` to cut cost is safe, but feeding a hand-written master schema to `--schema` is not. Sizing and cost are worked through in [`dispatch_batch_vs_sequential.md`](dispatch_batch_vs_sequential.md) → *OpenAI → `--shard-budget` is the real cost lever*.
- **The pre-flight is per shard and named.** Every shard schema is checked against the live API *before any image is uploaded*, and the schema's `title` becomes the response-format name (`stage3_shard_04`), so a rejection names the shard that caused it. A schema error is per-request on the Batch API, so without this one bad shard would bill 9/10 of a run to deliver 9/10 of the traits.
- **`--allow-reshard` only fires on a size limit.** A shard rejected with `... exceeds limit of ...` (nesting depth or parameter count) can be resharded at a halved budget — that is the one failure a smaller budget fixes. Any *other* schema rejection (an unresolved `$ref`, an invalid type, an array without `items`) aborts the run and leaves `--shard-dir` byte-for-byte untouched **even when `--allow-reshard` is given**, because resharding would not fix it.
- **`--no-files-api` does not scale with shards.** Inline base64 repeats every image once per shard: 1 plant × 15 images × 10 shards is a **200 MB** batch input file (209,301,506 bytes) against OpenAI's 200 MB cap — the same run is **94 KB** (93,686 bytes) through the Files API. pxGPT estimates the size from the images on disk *before* encoding anything, warns, and then refuses to upload a JSONL over 190 MB rather than failing after the transfer. That cap applies to the batch input file (`purpose="batch"`) only; it has nothing to do with image uploads (`purpose="vision"`).

**Prompt caching differs, and it costs money.** The Anthropic path marks the shared system block with `cache_control` explicitly. OpenAI's caching is automatic, and measured against `gpt-5.6-luna` it recovers only the **shared system prompt** across a plant's shards — not the images:

| request | input tokens | `cache_read` |
|---|---|---|
| 10 consecutive shards of one plant | ~31 k each | 1,055–1,281 (one 0) |
| the *same* shard body re-sent later | ~31 k | 30,672–31,196 |

The system prompt alone is ~1 k tokens, which is the whole of that first row. Because the images sit *before* the per-shard text prompt in the request, a differing text tail alone could not have excluded them — the per-shard response-format schema evidently breaks the cacheable prefix ahead of the image blocks.

Practical consequence: **a sharded OpenAI run pays close to full input price on every shard** (~30 k tokens × shards × plants), and the cache only pays off when an *identical* shard request is re-sent — a resume or a gap recovery. Do not assume the two providers' per-plant input costs scale the same way; the Anthropic path's explicit `cache_control` on the system block behaves differently.

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

### `pxgpt json-to-table`

Flatten Stage 3 per-cultivar JSON results into a single wide, analysis-ready table. See **Downstream analysis** for the full column-typing rules.

```
pxgpt json-to-table \
  --result-dir DIR \        # Stage 3 output: one <cultivar_id>.json per plant
  --master-schema FILE \    # master schema (authoritative trait metadata)
  [--shard-dir DIR] \       # fallback trait metadata for traits absent from master
  --out-prefix PREFIX \     # writes <prefix>.csv and <prefix>.feather
  [--on-collision {error,prefix_collided,prefix_all}] \  # default: error
  [--rename-map FILE] \     # JSON: {"group.trait": "desired_column_name", ...}
  [--allow-mixed-provenance]  # write anyway when records disagree on provider/model/schema_version
```

Writes `<prefix>.csv` (ordinal traits as label strings) and `<prefix>.feather` (Arrow IPC v2; identical except ordinal traits are ordered `pandas.Categorical`, so R's `arrow::read_feather()` reads them as ordered factors). Nominal columns are plain strings in both — never a category/factor. Any trait found in results but not in the master or shard schemas is logged as a warning and included as a best-effort string column. Every row also carries `provider`/`model`/`schema_version` from that record's `_provenance` block (see **Downstream analysis** above); by default, a `--result-dir` holding more than one distinct `(provider, model, schema_version)` is refused before anything is written — pass `--allow-mixed-provenance` to write anyway.

`--on-collision` controls what happens when two traits compute the same final column name (see **Column name collisions** above): `error` (default) writes no files and prints a `--rename-map` fill-in template; `prefix_collided` auto-prefixes only the clashing columns with the minimal group-path prefix needed; `prefix_all` prefixes every column with its full path regardless of collisions. `--rename-map` is applied first and takes a JSON file keyed by dotted source path (`group.trait`, not the possibly-colliding column name), values used verbatim with no unit re-appended. A global uniqueness check runs last regardless of mode — it raises rather than ever letting one column silently overwrite another.

---

### `pxgpt analyze`

Text description (sync, all providers). One plant line, or a whole tree of them.

```
pxgpt analyze \
  (--input-folder PATH | --input-dir PATH)     # one plant, or a tree of plants
  --output PATH                                # FILE for one plant; DIRECTORY with --input-dir
  --system-prompt FILE \
  --prompt FILE \
  [--provider {anthropic,openai,ollama,lmstudio,vllm}]
  [--image-transport {base64,file}]            # default base64; 'file' = file:// URIs
  [--resume | --no-resume]                     # default --resume (skips plants already written)
  [--effort {off,low,medium,high,xhigh,max}]   # default off
```

`--effort` enables reasoning (overrides `ANALYZE_EFFORT`; default **off**, preserving the original non-thinking behavior). On Anthropic it becomes adaptive thinking and the thinking blocks are stripped from the output; on OpenAI it becomes `reasoning_effort`, with "off" sent as the explicit level `none`. Either way `temperature` is omitted whenever the model will not accept a custom value.

**On the local backends (Ollama / LM Studio / vLLM) `--effort` is NOT ignored** — it switches the chat template's `enable_thinking` on. Those models have no reasoning *levels*, so any level simply means on and `off` means off. Only the final answer is written to `--output`: the server's reasoning parser keeps the thinking in its own response field, which pxGPT never saves. Expect it to be several times slower — measured on one plant line, 2506 completion tokens / 94 s with thinking on against 36 tokens / 1.7 s off.

#### Recipe: gather descriptions from `analyze` (single-folder mode)

`analyze` processes **one folder at a time**, so the classic workflow is to loop over cultivars and then merge the per-cultivar descriptions into one document — e.g. to feed Stage 2 (schema synthesis). This is the single-file counterpart to `describe-batch`.

**1. Run `analyze` per cultivar:**

```bash
for i in $(ls germplasm_images/); do
  pxgpt analyze \
    --input-folder germplasm_images/${i} \
    --output results/${i}_description.txt \
    --system-prompt prompts/describe_plant_system.txt \
    --prompt prompts/describe_plant_mature.txt \
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

Structured JSON output (sync, all providers). The schema always reaches the model as a real decoding constraint: Anthropic gets native `output_config.format`, the OpenAI-wire providers get `response_format` `json_schema` (strict). It is never pasted into the system prompt.

Two independent choices — one schema or a whole shard set, and one plant or a whole tree:

```
pxgpt schema \
  (--schema FILE | --shard-dir PATH)           # one schema, or a shard set
  (--input-folder PATH | --input-dir PATH)     # one plant, or a tree of plants
  --output PATH                                # FILE only for --schema + --input-folder;
                                               #   DIRECTORY otherwise
  [--system-prompt FILE]                       # required with --schema; overrides the
                                               #   shard set's own with --shard-dir
  [--prompt FILE]                              # required with --schema; ignored with
                                               #   --shard-dir (each shard carries its own)
  [--provider {anthropic,openai,ollama,lmstudio,vllm}]
  [--image-transport {base64,file}]            # default base64; 'file' = file:// URIs
  [--max-tokens N]                             # default 2048 with --shard-dir, else MAX_TOKENS
  [--resume | --no-resume]                     # default --resume
  [--effort {off,low,medium,high,xhigh,max}]   # overrides STAGE3_EFFORT
  # --shard-dir dispatch (local runs; see "Local / self-hosted providers"):
  [--concurrency N]                            # default 8   requests within one plant
  [--pipeline-depth {1,2}]                     # default 2   plants in flight
  [--mem-floor-gib X]                          # default 5   pause overlap below this
  [--limit N]                                  # first N plants only, for timing
```

`--resume` means different things per mode: with `--shard-dir` it is per **shard** (anything in `<output>/_partial/` is skipped, so a plant missing three shards re-runs only those); with `--schema` it is per **plant** (a plant whose output file exists is skipped).

For Anthropic, `schema` runs **without reasoning by default**; a custom `temperature` is sent only on Sonnet 4.6 and earlier. Enable adaptive thinking with `--effort` (e.g. `--effort medium`) or by setting `STAGE3_EFFORT`.

---

## Provider Configuration

### Anthropic Claude (recommended)

```bash
ANTHROPIC_API_KEY=your_key_here
ANTHROPIC_MODEL=claude-sonnet-5

# Adaptive thinking effort (Anthropic). For every knob below:
#   default = off = none = NO reasoning
#   (blank, "off", "none" are all equivalent). Set a level to enable reasoning.
# TEMPERATURE is sent only with reasoning off AND a Sonnet 4.6-or-earlier model;
# claude-sonnet-5 and newer reject a custom value, so pxGPT omits it.
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

# Retry / backoff (sync + sequential paths, both providers). A connection error
# or a 502/503/504 is retried with exponential backoff (2^attempt + jitter); a
# rate-limit error instead waits a flat RATE_LIMIT_SLEEP seconds. Anything else
# (a 400, for instance) is raised immediately, never retried.
MAX_RETRIES=3         # retries after the first attempt -> 4 tries in total
RATE_LIMIT_SLEEP=60
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
OPENAI_MODEL=gpt-5.6-luna
OPENAI_BASE_URL=                  # optional: point the openai provider at a proxy

# OpenAI Batch API stages (describe-batch-openai / phenotype-batch-openai)
                                  # Reasoning effort is NOT set here — the OpenAI stages read the
                                  # shared DESCRIBE_EFFORT / STAGE3_EFFORT knobs (see above).
OPENAI_BATCH_COMPLETION_WINDOW=24h
```

Note: GPT-5 / o-series reasoning models only accept the default `temperature`; pxGPT omits a custom temperature for them automatically.

### Local / self-hosted providers (`analyze` + `schema` only)

`analyze` and `schema` run on Ollama, LM Studio, and vLLM in addition to the cloud providers. Each is a first-class `--provider` value with its own env vars. **Use a vision-capable model** — both commands send images. (The batch stages are Anthropic/OpenAI-only.)

#### Use vLLM. The other two are supported, not recommended.

All three work today, and nothing warns you at runtime. For phenotyping, only
**vLLM** is recommended, and the reason is visual tokenization rather than speed
or convenience.

pxGPT scores fine-grained traits — petiole cross-section shape, leaf margin type,
colour hue. Whether the model can resolve those at all depends on how many visual
tokens each photo becomes. vLLM makes that an explicit, recorded setting:

```
--mm-processor-kwargs '{"max_soft_tokens": N}'      # ladder: 70 140 280 560 1120
```

The deployment in `ops/local-vllm/` pins **`IMAGE_TOKEN_BUDGET=1120`**, the top of
the ladder, for two reasons: the traits are fine-grained, and 1120 lands close to
Anthropic Sonnet 5's per-image tokenization, which is the only thing that makes a
local run and a cloud run comparable. Measured, one photo: 70 → 84 prompt tokens,
280 → 284, 1120 → 1131. Left unset, that checkpoint defaults to **280** — a
quarter of the detail, silently. See
[`ops/local-vllm/README_vllm.md`](ops/local-vllm/README_vllm.md) §2 for the full
cost table.

**Ollama and LM Studio expose no equivalent knob.** Whatever downsampling each
applies is not settable, not reportable, and not guaranteed to survive a backend
or model update. That is the objection, stated precisely: **this has not been
benchmarked on either backend** — the problem is not a measured loss of quality,
it is an uncontrolled variable sitting underneath every trait value you record.
For a consistency study that compares backends, an unpinnable tokenizer is
disqualifying on its own.

> **Planned removal.** `ollama` and `lmstudio` are slated for removal in a future
> major release; `vllm` is the local backend that will be kept. They are not
> deprecated in code — no warning, no gate, no behaviour change. If someone
> measures their visual tokenization and it proves controllable and stable, that
> decision should be revisited.

```bash
# Ollama — pxgpt analyze --provider ollama ...
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:12b           # an Ollama tag; must be a VISION model

# LM Studio (OpenAI-compatible) — pxgpt analyze --provider lmstudio ...
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model        # exactly the name LM Studio shows
LMSTUDIO_API_KEY=lm-studio        # any non-empty placeholder

# vLLM (OpenAI-compatible) — pxgpt schema --provider vllm ...
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL=gemma4-26b-a4b-nvfp4   # REQUIRED: the SERVED name, not the HF repo
VLLM_API_KEY=EMPTY                # match the server's --api-key if set
```

> **`VLLM_MODEL` is the served name, not the checkpoint name.** These are two
> different strings and mixing them up is the most common local-setup failure:
>
> | | value |
> |---|---|
> | HF repo (what the server downloads) | `unsloth/gemma-4-26B-A4B-it-NVFP4` |
> | served name (what requests must say) | `gemma4-26b-a4b-nvfp4` |
>
> `up.sh` sets the served name from `SERVED_MODEL_NAME` in
> `ops/local-vllm/.env`; `VLLM_MODEL` must equal it exactly. Ollama-style tags
> such as `gemma4:12b` are **not** valid here — vLLM has no tag syntax. Confirm
> what is actually served with:
>
> ```bash
> curl -s http://localhost:8000/v1/models | python -m json.tool | grep '"id"'
> ```

How each is routed: all three speak the OpenAI wire protocol, so pxGPT sends them through one `openai.OpenAI` client with the appropriate `base_url` (`OpenAICompatProvider`). Model names are sent verbatim — there are no route prefixes. `base_url`/`api_key` are per client, so several providers can be configured at once without clashing. Ollama's `/v1` suffix is appended automatically if `OLLAMA_BASE_URL` omits it. A parameter a backend rejects now raises an error rather than being silently dropped.

For `schema` on these providers the JSON schema is sent as **native structured output** — `response_format {"type": "json_schema", …, "strict": true}`, i.e. real constrained decoding via the server's grammar backend. The schema therefore appears in exactly one place, so the prompt stays byte-identical to what Anthropic and OpenAI receive and the runs remain comparable. The user prompt does **not** need to ask for JSON. If the backend rejects `response_format` the command fails rather than falling back to prompt text, because a silent downgrade produces output that looks fine and is in fact completely unconstrained.

- vLLM (recommended): start it with the scripted deployment below, then set `VLLM_MODEL` to the served name.
- Ollama: ensure `ollama serve` is running and the model is pulled (`ollama pull gemma3:12b`). Two strikes against it for this workload — its grammar support is weaker than vLLM's, so prefer vLLM for `schema`; and it gives you no control over the per-image visual token budget (above).
- LM Studio: same visual-token objection as Ollama. Useful for a quick interactive check, not for a run whose numbers you intend to report.

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
#### The trap

`ops/local-vllm/README_vllm.md` tells you to run this before `smoke.py`:

```bash
set -a; source .env; set +a
python smoke.py
```

That genuinely does load the server file into your shell — but it gives you
`SERVED_MODEL_NAME`, never `VLLM_MODEL`. `smoke.py` reads the server names;
pxGPT reads the client names. Running that line and then expecting
`pxgpt --provider vllm` to work is the most natural wrong assumption here, and it
fails with:

```
Provider 'vllm' is not properly configured.  Check your API keys.
```

which is a confusing message for a missing model name. The two-line `export`
above is what fixes it.

#### Put it in the run script, not your interactive shell

A long unattended run should not depend on what happened to be exported in the
terminal that launched it. Set the variables inside the script that runs the job,
so the run is reproducible from the script alone:

```bash
#!/bin/bash
set -uo pipefail
source /home/xavier/miniconda3/etc/profile.d/conda.sh   # conda activate is a
conda activate pxgpt                                    # shell function

set -a; source /path/to/pxgpt/ops/local-vllm/.env; set +a
export VLLM_MODEL="$SERVED_MODEL_NAME"
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_API_KEY=EMPTY
export TIMEOUT=1800

# Fail now, not in three hours, if the server is not serving what we expect.
served=$(curl -s -m 5 "localhost:${PORT}/v1/models" \
         | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
[[ "$served" == "$VLLM_MODEL" ]] || { echo "FATAL: server has '$served'" >&2; exit 1; }

pxgpt schema --provider vllm --shard-dir ... --input-dir ... --output ...
```

#### Running a whole dataset locally (the production path)

The batch stages need the providers' Batch APIs, which no local server offers, so `schema --shard-dir --input-dir` **is** the local Stage 3 runner:

```bash
export VLLM_MODEL=gemma4-26b-a4b-nvfp4
export TIMEOUT=1800                      # a cold prefill measures 75-95 s

pxgpt schema --provider vllm \
  --shard-dir  /abs/path/to/shard_master_schema \
  --input-dir  /abs/path/to/images \
  --output     /abs/path/to/writable/results \
  --image-transport file
```

**Image transport, and how `MEDIA_ROOT` gates it.** `--image-transport base64`
(the default) embeds the bytes in every request and needs no server
configuration at all. `--image-transport file` sends `file://` URIs instead —
the server reads the files off its own mount, so a plant line costs a few
hundred bytes of paths instead of megabytes of base64. That is the recommended
local path, and it is the one `MEDIA_ROOT` controls.

`MEDIA_ROOT` is set in `ops/local-vllm/.env` and used **twice** by `up.sh`, both
of which are required:

| use | effect if missing |
|---|---|
| `-v "$MEDIA_ROOT:$MEDIA_ROOT:ro"` | the container cannot see the file at all |
| `--allowed-local-media-path "$MEDIA_ROOT"` | vLLM refuses the path even when mounted |

Three consequences worth knowing:

- **`--input-dir` is an absolute host path, not a path relative to
  `MEDIA_ROOT`.** It must *sit under* `MEDIA_ROOT`, and because the mount uses
  the same path on both sides, the string is identical inside and outside the
  container. pxGPT itself never reads `MEDIA_ROOT`.
- **`MEDIA_ROOT` is a prefix**, so point it at a parent directory (e.g. the
  project root) and every dataset beneath it works. It is baked into the running
  container: changing it means `./down.sh && ./up.sh`.
- **The two failure modes are distinguishable.** `400 … must be a subpath of
  --allowed-local-media-path` means the path is outside the tree; `500 … No such
  file or directory` means it is inside the tree but the file is not there.

**Dispatch flags.** Each plant sends one shard alone — all its shards share a
prefix of system prompt plus every image, and only the first arrival pays to
build it — then fans the rest onto the warm prefix.

| flag | default | what it does |
|---|---|---|
| `--concurrency` | 8 | cap on concurrent requests *within* one plant, after its cold shard. A hardware-pressure limit, **not** `n_shards - 1`: effective width is `min(--concurrency, n_shards - 1)`, so a 30-shard set still fans out 8. `1` = fully serial. |
| `--pipeline-depth` | 2 | plants in flight, so one's cold prefill overlaps another's warm group. Refused above 2. |
| `--mem-floor-gib` | 5 | do not *start* another plant below this host `MemAvailable`; recovers automatically. **Linux only** — see below. |
| `--limit` | — | run only the first N plants, for timing. |
| `--max-tokens` | 2048 | per-shard output cap. `finish_reason == "length"` fails that shard rather than storing a truncated result. |

> **The memory guard is Linux-only, and it fails open.** `MemAvailable` is read
> from `/proc/meminfo`, which does not exist on macOS. There the reading is
> `None`, `--mem-floor-gib` has no effect whatever you set it to, and the run
> prints `Note: /proc/meminfo is unavailable, so the memory guard is off`. It
> does not stop, and it does not fall back to another source.
>
> This matters most exactly where the guard would help most. The defaults
> (`--concurrency 8`, `--pipeline-depth 2`, `--mem-floor-gib 5`) were measured on
> a 128 GB GB10 box; a laptop has far less headroom and no guard. On macOS,
> control the pressure yourself: `--pipeline-depth 1` to stop overlapping plants,
> and a lower `--concurrency`. Run `--limit 4` first and watch Activity Monitor,
> the way the `MemAvail` column would be watched on Linux.

A separate global ceiling of `--concurrency + 1` requests applies across all
in-flight plants. Depth alone would not bound this: depth 2 with width 8 could
put two plants in their warm phase at once, which is 16 concurrent requests and
exactly the server's `MAX_NUM_SEQS`.

Every default above was measured on one machine (GB10, 128 GB unified memory) and
none is portable. Before raising anything on new hardware, run `--limit 4` and
watch the `MemAvailable` column.

Measured on 4 plants of `03_mature_v2` (9 shards, 14-20 photos each), restarted
container, same plants both ways:

| dispatch | per plant | warm hit |
|---|---|---|
| `--concurrency 1 --pipeline-depth 1` (serial) | 111.8 s | 96.8 % |
| defaults (`8` / `2`) | **60.6 s** | 96.8 % |

**1.85x**, measured against a serial baseline on the *same plants*. Beware
comparing across datasets: the 161.6 s/plant serial figure in
[RUNBOOK.md](ops/local-vllm/RUNBOOK.md) was measured on `02_mature_v1`, which has
10 shards and 26-32 photos per plant, so dividing by it would overstate the gain.
Get your own baseline with `--concurrency 1 --pipeline-depth 1 --limit 4`.

**Reading the progress line.**

```
[ 12/277] s0019  9/9 ok  cold 87.0s (hit 0.0%)  warm 8x 21.4s (hit 98.4%)  total 108.4s  depth 2  ETA 5h52m  MemAvail 11.2G
```

`hit` is `cached_tokens / prompt_tokens`. The cold shard should be near 0 % and
the warm group 97-99 %. If warm hits fall below 50 % pxGPT prints a WARNING —
that is the only immediate signal the prefix cache has stopped working, and you
want it at plant 3, not plant 277.

**Resuming.** Every shard's JSON is written to `<output>/_partial/` the moment it
succeeds, so re-running the identical command skips whatever is already there
without re-billing it; `--no-resume` forces everything. Ctrl-C finishes the
plants in flight, merges what completed, and exits — it does not discard the run.
`--output` must be a writable directory, and never inside a frozen dataset tree.

> **Self-hosting Gemma 4 26B A4B (NVFP4) on a DGX Spark?** Use the tested,
> reproducible deployment in **[ops/local-vllm/README_vllm.md](ops/local-vllm/README_vllm.md)**
> — scripted start/stop, a pinned image digest and model revision, an acceptance
> suite that runs against your real shard schemas, and the vLLM **version
> constraint** that decides whether the checkpoint loads at all. The supporting
> measurements are in [ops/local-vllm/RUNBOOK.md](ops/local-vllm/RUNBOOK.md).

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

What ships in `prompts/`, and what each file is for:

| File | Stage | Role |
|---|---|---|
| `describe_plant_system.txt` | 1 | System prompt: role definition. Keep it stable across runs — it is cached. |
| `describe_plant_mature.txt`, `describe_plant_seedling.txt` | 1 | User prompt, **split by growth stage**. Ask for every morphological trait you care about; the richer the descriptions, the better the Stage 2 synthesis. |
| `generate_master_schema.txt` | 2 | The prompt you paste into an LLM session to synthesise your master schema from the Stage 1 descriptions. |
| `phenotype_schema_system_mature.txt`, `phenotype_schema_system_seedling.txt` | 3 | System prompt, passed as `--system-prompt`. See below — this one matters more than it looks. |
| `phenotype_schema_system_template.txt` | 3 | Starting point for a growth stage not covered above. |

**No master schema and no shard schema ship with pxGPT.** Yours depends on your
traits, your growth stage and your imaging rig, so a shipped one would be
actively misleading — every user's differs. Stage 2 is where you produce it.

**Always pass `--system-prompt` at Stage 3, even though `shard-schema` writes
one.** The generator emits `shards_system.md` and a run falls back to it
(`pxgpt/core/sharding.py`, `load_system_prompt` — the CLI override wins).
Override it anyway, because the shipped Stage 3 system prompts carry two things
the generated preamble cannot know:

- **The scale reference for your growth stage.** `_mature` states a
  10 x 10 x 6.5 cm rockwool cube; `_seedling` states 2.5 cm. A quantitative trait
  is an eyeball estimate against that reference, so the wrong one shifts every
  measurement.
- **When to answer `not_assessable`.** Emphasised four times in the mature
  prompt, and absent from the generated preamble. `shard-schema` injects the
  *value* into every nominal and ordinal enum, but nothing tells the model when
  it is the honest answer.

Stage 3 needs no user prompt with `--shard-dir`: each shard carries its own.

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

Export the key into the environment `pxgpt` runs in. A `.env` file sitting in the
working directory has no effect on its own — pxGPT never reads one:
```bash
export ANTHROPIC_API_KEY=sk-ant-...     # or from ~/.bashrc / ~/.zshrc
set -a; source project_A.env; set +a    # or export a whole project file at once
```
Check with `echo ${ANTHROPIC_API_KEY:0:7}` in the same shell that runs `pxgpt`.

### No images found / wrong image format

- Supported extensions: `.jpg`, `.jpeg`, `.png` — matched case-insensitively, so `.JPG` works. `.gif` and `.webp` are **not** supported and are skipped silently.
- The batch commands (`--input-dir`) need images inside subdirectories, not directly in the root folder; the sync commands (`--input-folder`) read images directly from the folder given.
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
  --system-prompt prompts/describe_plant_system.txt \
  --prompt prompts/describe_plant_mature.txt
# → all images are now uploaded; checkpoint_S1.json saved

# Submit Stage 3 immediately (reuses file_ids from manifest)
pxgpt phenotype-batch --input-dir ./images \
  --shard-dir shard_master_schema \
  --output phenotypes/ \
  --system-prompt prompts/phenotype_schema_system_mature.txt
# → no uploads; checkpoint_S3.json saved

# Retrieve results for both when done
pxgpt fetch-results --checkpoint checkpoint_<S1_id>.json
pxgpt fetch-results --checkpoint checkpoint_<S3_id>.json
```

### Per-project .env files

```bash
# project_A.env
DEFAULT_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-5
STAGE3_EFFORT=high
STAGE1_MAX_TOKENS=32768

set -a; source project_A.env; set +a
pxgpt describe-batch ...
```

`set -a` is what makes the assignments **exported**. Plain
`source project_A.env` only sets shell variables, which `pxgpt` — a child
process — never sees, so the run would silently use the defaults instead.
(Prefixing every line in the file with `export` has the same effect.)

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
  --system-prompt prompts/describe_plant_system.txt \
  --prompt prompts/describe_plant_mature.txt

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
