# Stage 3 sharded dispatch: `batch` vs `sequential`

`pxgpt phenotype-batch --shard-dir ... --dispatch {batch,sequential}` sends the
same set of *(plant × shard)* requests through two different transports. This
note explains what each one actually does, and how to work out which is
cheaper for your dataset.

## Functional difference

Both modes build requests the same way (`pxgpt/core/sharding.py:246`
`build_sharded_requests`): for each plant, the system prompt + that plant's
images form a byte-identical prefix with an `ephemeral` cache breakpoint on
the last image block (`sharding.py:222` `_images_with_cache_breakpoint`).
Only the per-shard prompt text and `output_config.format` schema differ
between shards of the same plant. Where the two modes diverge is *how* those
requests are sent — which determines whether that shared prefix actually
hits the prompt cache.

### `--dispatch batch` (default)

`_dispatch_batch` (`pxgpt/commands/phenotype.py:327`) hands every
plant×shard request to `client.beta.messages.batches.create(...)` — the
async **Message Batches API** — in one call.

- Anthropic's batch backend processes the queued requests independently,
  with no guarantee of ordering or of how close in time two requests for the
  same plant actually run.
- The prompt cache is ephemeral with a ~5‑minute TTL. If the batch backend
  happens to schedule a plant's shards far apart (different workers, queued
  behind other jobs), the cache entry can expire or simply miss before the
  next shard runs.
- Batch mode is billed at Anthropic's standard **50% Batch API discount** on
  all token types, but caching across a plant's shards is opportunistic, not
  guaranteed.

### `--dispatch sequential`

`_dispatch_sequential` (`pxgpt/commands/phenotype.py:372`) does **not** use
the Batch API. It's a plain Python loop calling the regular synchronous
**Messages API** one request at a time — `client.messages.create(...)` /
`client.beta.messages.create(...)` — grouped by plant (all of plant A's
shards, then plant B's, ...).

- Because each shard's call for a given plant fires immediately after the
  previous one returns, they land well inside the 5‑minute cache TTL almost
  every time, so `cache_read_input_tokens` on shards 2..n of a plant should
  be consistently non‑zero.
- There is no batch discount — every call is billed at full, non-batch
  rates.
- `cache_read`/`cache_creation`/`input`/`output` totals are printed per call
  (`phenotype.py:405-407`) so hits vs. misses are directly observable.

**Bottom line:** batch = cheaper base price, unreliable caching. Sequential
= full price, reliable caching. Which wins depends on how much caching batch
actually achieves in practice — see the cost model below.

## Cost model

Pricing facts (verified, not recalled): the model configured for this
project is `claude-sonnet-4-6` (`pxgpt/core/config.py:46`) at **$3/MTok
input, $15/MTok output**. The Anthropic Batch API gives a flat **50%
discount on every token type** — input, output, cache write, and cache
read. Prompt cache pricing is unchanged by batch/sequential: **cache write
≈1.25×** base price for the default 5‑minute ephemeral TTL used here (no
`ttl` override in `sharding.py`), **cache read ≈0.1×** base price.

Split each plant's cost into:

- **F** — shared prefix tokens (system prompt + images), repeated
  identically across all `n` shards of a plant
- **M** — per-shard marginal tokens (shard prompt + output), small relative
  to F in a vision pipeline where images dominate

**Sequential** (shard 1 writes the cache, shards 2..n read it, no batch
discount):

```
Cost_seq = P·F·[1.25 + 0.1(n-1)] + P·n·M
```

**Batch** (50% off everything, but an unknown cross-shard cache-hit rate
`h` for shards 2..n of a plant — `h=1` means every later shard hits cache,
`h=0` means every later shard is a full miss):

```
Cost_batch = 0.5·P·F·[1.25 + (n-1)(1 - 0.9h)] + 0.5·P·n·M
```

The `M` term is always cheaper under batch (flat 50% off), so it never
favors sequential — it only nudges the crossover slightly toward batch.
Solving `Cost_seq = Cost_batch` for the `F` term alone, with `x = n-1`:

```
x = 0.625 / (0.4 - 0.45h)
```

### This project's numbers

The Stage 3 shard set here has **n = 7 shards/plant**
(`shard_master_schema/shards_manifest.json`, `shard_count: 7`), so `x = 6`.
Solving for the break-even hit rate `h*`:

```
6 = 0.625 / (0.4 - 0.45h*)   →   h* ≈ 0.657
```

**At 7 shards/plant, sequential is cheaper whenever batch's cross-shard
cache-hit rate is below ~66%. Above ~66%, batch wins even with materially
worse caching**, because a flat 50%-off-everything eventually beats caching
gains. Including the `M` term (always favors batch) pushes the true `h*` a
little above 66%, by an amount that depends on how big your shard
prompts/outputs are relative to the image tokens.

| Batch hit rate `h` | Break-even `n` (shards/plant) |
|---|---|
| 0% (never hits) | ~2.6 |
| 25% | ~3.1 |
| 50% | ~4.9 |
| 66% (this project's `h*`) | 7 |
| 75% | ~11 |
| 89%+ | never — batch always wins |

## Measuring `h` for your run

Nobody has measured `h` for this pipeline yet — `print_token_summary`
(`pxgpt/core/batch_utils.py:291`) prints totals but doesn't persist them,
and the only completed batch checkpoint on disk errored on every plant with
"compiled grammar is too large" (the bug this shard-set is fixing), so its
numbers aren't usable.

To get a real number: run the same 2–3 plants through both
`--dispatch batch --wait` and `--dispatch sequential`, and compare the
printed `Cache read tokens` / `Cache creation tokens` / `Input tokens` in
each summary. For the batch run, `h` is roughly the fraction of shards
after the first (per plant) whose prefix tokens land in `cache_read` rather
than `input`/`cache_creation`. Plug that into `x = 0.625/(0.4-0.45h)` and
compare against your actual shard count to decide which mode is cheaper.
