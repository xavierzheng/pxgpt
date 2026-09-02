# Stage 3 sharded dispatch: `batch` vs `sequential`

`pxgpt phenotype-batch --shard-dir ... --dispatch {batch,sequential}` builds the
same *(plant × shard)* requests in both modes. The mode changes how requests are
sent. It does not change how Anthropic Structured Outputs affect prompt caching.

Both providers offer the same two dispatch modes. Everything up to
**"Practical mode selection"** describes the Anthropic path
(`pxgpt phenotype-batch`); the OpenAI path (`pxgpt phenotype-batch-openai`) is
covered in **"OpenAI"** at the end, including measured caching numbers and the
`--shard-budget` cost lever that only applies there.

## Bottom line

- `batch` sends one asynchronous Message Batch. It has the Batch API discount
  and higher throughput. Its `fetch-results` now persists each succeeded shard
  to `<output>/_partial/`, so a batch that leaves gaps can be recovered with a
  short sequential resume to the same `--output` (see "Recovering failed shards").
- `sequential` sends one synchronous call at a time, with all shards for one
  plant kept together. It has no batch discount, but it provides incremental
  output writes, resume, bounded retry and live progress.
- Images are intentionally outside the cached prefix. They are ordinary input in
  both dispatch modes and stay before the per-shard text prompt.

Choose the mode for transport, throughput and recovery behavior. Prompt caching
applies only to the smaller system/format prefix.

## Request structure

`build_sharded_requests` creates requests in plant-contiguous order:

```text
plant A, shard 01
plant A, shard 02
...
plant B, shard 01
plant B, shard 02
...
```

Each request has one explicit cache breakpoint on the shared system block.

The visible request content is:

```text
system: [shared system | cache breakpoint]
user:   [plant images] [per-shard text prompt]
```

Images do not carry `cache_control`. They remain at the start of the user content,
before the text prompt, following Anthropic's recommended image-then-text layout.
The request also contains a different `output_config.format` schema for every
shard.

## Structured Outputs changes the effective cache identity

Anthropic Structured Outputs add format-specific system instructions to the
effective prompt. Changing `output_config.format` invalidates the related prompt
cache. A schema being a separate top-level API parameter does not keep it outside
the effective cache identity.

This has two important effects in sharded Stage 3:

- Across shards of the **same plant**, the schema changes, so those shards do not
  share one system/format cache identity.
- Across plants using the **same shard**, the schema stays the same, so the
  smaller system/format prefix may be read from cache.

Images remain ordinary input in both cases. This avoids paying the cache-write
premium repeatedly for image tokens that different shard schemas cannot reuse.

Structured Outputs also cache the compiled grammar separately. That grammar
cache reduces later schema-compilation latency. It is not prompt caching and
does not reduce image input tokens.

See Anthropic's official documentation for
[Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
and [prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

## `--dispatch sequential`

Sequential dispatch uses a plain synchronous loop. One request must return before
the next request is sent.

It provides:

- plant-contiguous request order;
- immediate writes to `<output>/_partial/`;
- automatic resume from valid partial files;
- bounded retry for transient API errors;
- live per-request cache usage in stdout.

Serial order removes the concurrent cold-start race between adjacent requests.
It does not bypass cache invalidation caused by changing
`output_config.format`. The default five-minute cache lifetime can also matter
after long delays, errors or resumed runs.

Sequential calls use normal Messages API pricing. They do not receive the Batch
API discount.

## `--dispatch batch`

Batch dispatch submits all *(plant × shard)* requests to one asynchronous Message
Batch.

It provides:

- the Batch API discount;
- higher throughput;
- fire-and-forget submission with a checkpoint;
- later retrieval and merge through `fetch-results`.

Batch requests may run concurrently or far apart in time, so system/format cache
hits are best-effort. Structured Outputs cache invalidation still applies when
the shard schema changes. Images are ordinary input regardless of scheduling.

`fetch-results` writes each succeeded shard to `<output>/_partial/` — the **same**
store the sequential resume reads — and merges the union of prior partials plus
this batch. Re-running `fetch-results` is therefore idempotent, and this shared
store is what makes batch gaps recoverable (below).

## Recovering failed shards from a batch (`overloaded_error`, etc.)

A batch request that errors — e.g. a transient
`overloaded_error: File storage is temporarily unavailable` — is **terminal
inside that batch**. The Batch API cannot re-run one request, so re-fetching the
same batch returns the same error and regenerates the same `<line_id>.gaps.json`.
`--resume` does not apply to a batch: there is nothing in the batch to resume.

Recover the failed shards with the sequential path, which re-issues real API
calls and retries transient errors in-run. Two steps, to the **same `--output`**:

```bash
# 1. FREE: re-download the batch so every succeeded shard lands in _partial/.
#    (Needed only for batches fetched before partial persistence existed; new
#    runs already populate _partial/ on the first fetch.)
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json

# 2. Re-issue ONLY the still-missing shards (resume skips everything in _partial/).
pxgpt phenotype-batch \
    --shard-dir <shard_dir> --manifest <file_manifest.json> \
    --master-schema <master_schema.json> --output <same output dir> \
    --dispatch sequential
```

Step 2 rebuilds all *(plant × shard)* requests, skips the ones already in
`_partial/`, and calls only the failed shards (bounded transient retry). Each
recovered shard is written to `_partial/` and its plant is re-merged; a
`*.gaps.json` whose traits are now filled is deleted. Use the **same** model and
`STAGE3_EFFORT` as the original run so the recovered shards match the rest.

## Reading token usage

With images outside the cache breakpoint, expect this general pattern:

```text
input_tokens   include the plant images and per-shard text prompt
cache_read     may be stable for the same shard across different plants
cache_creation represents a cold system/format prefix, not the plant images
```

The CLI prints token counts, not the number of cache operations:

```text
input=<input_tokens>
cache_read=<cache_read_input_tokens>
cache_creation=<cache_creation_input_tokens>
```

## Practical mode selection

Use `sequential` when recovery and observability matter most. It is the safer
choice for long HPC jobs because completed shards are persisted immediately and
can be resumed without re-billing.

Use `batch` when throughput and the Batch API discount matter most, and delayed
result retrieval is acceptable.

For cost decisions, run a small representative pilot and compare actual input,
output, cache-read and cache-creation tokens. Image tokens should now appear as
ordinary input instead of a large cache creation on every shard.

---

# OpenAI

`pxgpt phenotype-batch-openai --shard-dir ... --dispatch {batch,sequential}` is
the OpenAI counterpart. It consumes the **same shard set**, writes the same
`<output>/_partial/` store and produces the same `{line_id}.json` /
`{line_id}.gaps.json`, so one shard set can be scored by both providers and the
results compared trait-for-trait.

Every number in this section was measured against `gpt-5.6-luna` through the Files
API. The dispatch and caching measurements use one real plant (s0001, 15 images)
with the frozen 10-shard set; the shard-budget cost table below adds the measured
10-plant means, which are the ones to extrapolate from.

## Bottom line (OpenAI)

- The two dispatches send **identical request bodies**. Measured: the same plant
  cost **308,397 input tokens** through `--dispatch batch` and **308,397**
  through `--dispatch sequential`. The batch JSONL only adds the
  `custom_id` / `method` / `url` envelope around that body.
- `batch` gets the Batch API's documented 50% discount and a 24 h completion
  window; the 10-request job above actually finished in about 5 minutes.
- `sequential` gets no discount but gives incremental writes, resume, bounded
  retry and live progress — the same properties as the Anthropic path.
- **Neither dispatch gets image tokens out of cache.** See below; this is the
  dominant cost fact and it is not a mode choice.
- A batch that leaves gaps is recovered by a sequential resume to the same
  `--output`, measured end to end (below).

## Request structure (Responses API)

Requests go to `/v1/responses`, which is required so images can be referenced by
Files-API `file_id`. There is no explicit cache breakpoint to place — OpenAI's
prompt caching is automatic.

```text
instructions: [shared system preamble]
input:        [plant images] [per-shard text prompt]
text.format:  [per-shard json_schema, strict]   <- top-level parameter
```

Requests are built plant-major, shard-minor, matching the Anthropic builder.

## Prompt caching: only the system prompt is reused across shards

The Anthropic section above explains that changing `output_config.format`
invalidates the prompt cache, so a plant's shards cannot share one cache
identity. **OpenAI behaves the same way**, measured independently:

| request | input tokens | `cache_read` |
|---|---|---|
| 10 consecutive shards of one plant | ~31 k each | 1,055–1,281 (one 0) |
| the *same* shard body re-sent later | ~31 k | 30,672–31,196 |

The shared system prompt is ~1 k tokens on its own, which accounts for the whole
of that first row. So across a plant's shards the ~30 k of image tokens are **not**
reused.

The images cannot be what breaks it: they sit *before* the per-shard text prompt
in `input`, so a differing text tail alone would still leave them inside a common
prefix. The per-shard `text.format.schema` evidently breaks the cacheable prefix
ahead of the image blocks — the same effect Anthropic Structured Outputs have, for
the same reason.

Two consequences:

- Referencing images by `file_id` does **not** prevent caching. The second row
  above is a `file_id` request hitting 31,117 of 31,120 tokens, so the image
  content is cacheable when the *whole* prefix matches.
- The cache pays off only when an **identical** *(plant, shard)* request is
  re-sent — a resume or a gap recovery. Measured: a resume that re-sent 2 shards
  read 31,196 and 30,672 tokens from cache.

Budget a **fresh** sharded OpenAI run at close to full input price per shard. Do
not assume the two providers' per-plant input costs scale alike.

## `--shard-budget` is the real cost lever on OpenAI

Sharding exists because a large schema exceeds **Anthropic's** grammar-size
limit. OpenAI has no such limit at this scale: its strict-mode caps are 5,000
object properties, 1,000 enum values and depth 10, and the whole 49-trait schema
in one shard measures 159 / 165 / 4. Every shard set below was accepted by the
live API, including the single-shard one:

| `--shard-budget` | shards | largest shard (props / enum values / depth) | OpenAI accepts | input tok/plant, `s0001` (15 img) | input tok/plant, 10-plant mean (19.9 img) |
|---|---|---|---|---|---|
| 40 (current) | 10 | 25 / 26 / 4 | yes | **308,397** (measured) | **403,506** (measured) |
| 80 | 4 | 46 / 52 / 4 | yes | **127,691** (measured) | **165,735** (measured) |
| 160 | 2 | 93 / 93 / 4 | yes | ~65 k | ~85 k |
| 320 | 1 | 159 / 165 / 4 | yes | **37,355** (measured) | ~49 k |

Budgets 40 and 80 are measured in both columns; 320 was scored on `s0001` only and
160 was never scored, so their right-hand entries scale the `s0001` measurement by
the image-count ratio. Since the ~30 k-token image payload is what every request
repeats, **halving the shard count nearly halves the input cost**.

**Quote the 10-plant column for any collection estimate.** `s0001` carries 15
images against a 19.6-image mean over the whole 142-plant collection
(2,784 images / 142 plants), so its tokens under-state a full run by about a
quarter. Measured over 10 plants at `gpt-5.6-luna` pricing, 142 plants
sequentially cost **$11.91 at budget 40** versus **$5.10 at budget 80** — $5.96 and
$2.55 respectively through `--dispatch batch`. Budget 320 extrapolates to ~$1.7
(n = 1 plant), but see the quality verdict below: do not use it. Output tokens
(~2,600 per plant) barely move, because the same 49 rationales get written either
way.

**All four shard sets keep every trait and every `not_assessable`.** This matters:
`pxgpt shard-schema` *adds* `not_assessable` to every nominal and ordinal enum —
in the 02_mature_v1 master, 45 of 45 nominal/ordinal traits get it injected, none
list it themselves. So the shard set is not merely a split of the master, and you
cannot skip `pxgpt shard-schema` on the grounds that OpenAI would accept a bigger
schema. Raising the budget is safe; bypassing the generator is not.

Before raising it, weigh three things:

- **Comparability.** Anthropic still needs the small budget. If the two providers
  run different shard counts, the per-request context differs and the comparison
  is no longer like-for-like. Keep separate `--shard-dir` and `--output` trees if
  you do this.
- **Blast radius.** With 10 shards a failed request costs ~5 traits for that
  plant; with 1 shard it costs all 49.
- **Quality does change, but less than the run-to-run noise.** Measured over 10
  plants, two runs per configuration
  ([`experiment_2026-08-14_shard_budget_openai.md`](experiment_2026-08-14_shard_budget_openai.md)):
  a configuration disagrees with **itself** on 7.0 of 49 traits between runs, and
  budget 80 disagrees with budget 40 on 8.2 — an excess of **+1.68 traits over all
  49 pooled** (paired t(9) = 3.03, p = 0.014, 95% CI +0.42 … +2.93). Over the 45
  **categorical** traits alone — the subset that speaks to scoring, since the 4
  quantitative traits are ruler estimates — the excess is **+1.88** (t(9) = 3.23,
  p = 0.010, 95% CI +0.56 … +3.19), and categorical run-to-run reproducibility is
  **89.8%, Gwet's AC1 0.88**. Real, and smaller than a re-run. Per-trait
  rationale length drops 15% (180.6 → 152.9 chars), which matters because
  `trait_object()` puts `rationale` before `value` precisely to force
  chain-of-thought. Budget 80 is a fair trade for 2.43× less input cost; budget
  320 is not — on one plant it diverged 10.2 traits, as much as changing provider,
  and cut rationale length by a third.

## `--dispatch sequential` (OpenAI)

Uses `client.responses.create()` with the same body the batch path writes to its
JSONL — deliberately not the sync chat-completions provider. If the two
dispatches sent different request formats their results would not be comparable
and their `_partial/` stores could not be mixed, which is the whole point of the
shared store.

Same properties as the Anthropic sequential path: plant-contiguous order,
immediate `_partial/` writes, resume from valid partials, bounded retry for
429/5xx/connection errors, and a 400 surfaced without retry (it writes no partial,
so the next resume retries it).

## `--dispatch batch` (OpenAI)

Submits all *(plant × shard)* requests as one JSONL job and writes a checkpoint
with `"stage": "phenotype_sharded"` plus `shard_dir`, `master_schema` and
`shard_ids` — the same field names the Anthropic sharded checkpoint uses, so
`fetch-results` serves both.

One OpenAI-specific limit: **the batch input file cannot exceed 200 MB.** With
the Files API this is a non-issue — the 10-shard job above was a 93,686-byte
JSONL, because each request carries only `file_id` strings. With
`--no-files-api`, every image is embedded once per shard: the *same one-plant job*
becomes 209,301,506 bytes, already over the cap. pxGPT estimates the size from
the images on disk before encoding anything, warns, and refuses to upload a JSONL
over 190 MB rather than failing after the transfer. That cap applies to the batch
input file (`purpose="batch"`) only and has nothing to do with image uploads
(`purpose="vision"`).

## Recovering failed shards from an OpenAI batch

Identical to the Anthropic recovery, and verified live. Deleting files from
`_partial/` does **not** create a gap — the batch results are still on the server,
so a re-fetch restores them (that is idempotency, not recovery). A real gap is a
shard the batch never produced. Measured with one:

```bash
# 1. fetch: the plant with no shards in the batch gets a gaps file
pxgpt fetch-results --checkpoint checkpoint_<batch_id>.json
#    -> "s0002: 49 missing trait(s)"; s0002.gaps.json written

# 2. re-issue ONLY the missing shards, to the SAME --output
pxgpt phenotype-batch-openai \
    --input-dir <images> --shard-dir <shard_dir> \
    --system-prompt <system.txt> --output <same output dir> \
    --dispatch sequential
#    -> "Resume: 10 of 20 shard(s) already on disk"
#    -> "20 call(s) (10 skip, 10 to run)"; gaps file deleted, both plants 49/49
```

The batch's 10 succeeded shards were adopted and **not** re-billed. Use the same
model and `STAGE3_EFFORT` as the original run so the recovered shards match.

A `_partial/.run.json` stamp records which provider and model created the store
and refuses a run that does not match, so an OpenAI run cannot silently merge its
shards into an Anthropic run's output directory.

## Reading token usage (OpenAI)

`cache_creation` is always 0: the Responses usage has no counterpart to
Anthropic's cache-write counter, because OpenAI's caching is automatic and the
write is not separately billed. `cache_read` comes from
`usage.input_tokens_details.cached_tokens`.

```text
input=<input_tokens>
cache_read=<cached_tokens>     ~1 k across a plant's shards; ~30 k on a re-send
```

## Practical mode selection (OpenAI)

The same rule as Anthropic — `sequential` for recovery and observability on long
HPC jobs, `batch` for throughput and the discount — but on OpenAI the mode is the
*smaller* of the two cost decisions. Shard count moves the bill by up to ~10×;
the batch discount moves it by 2×. Decide `--shard-budget` first.
