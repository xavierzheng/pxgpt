# Stage 3 sharded dispatch: `batch` vs `sequential`

`pxgpt phenotype-batch --shard-dir ... --dispatch {batch,sequential}` sends the
same set of *(plant × shard)* requests through two different transports. This
note explains what each one does, why batch's cost is a gamble and sequential's
is fixed, and how to decide between them with a single measurement.

**Bottom line:** sequential has a fixed, guaranteed cost because its caching is
structural. Batch is cheaper per token but its caching is best-effort, so its
cost is a range that straddles the break-even point — you cannot know which side
you land on without measuring. At this project's shard count (n = 7), sequential
is the safe default. As n grows, sequential's advantage widens. To settle it for
real, run a small pilot and read one number: cache *creations* per plant.

## Functional difference

Both modes build requests the same way (`pxgpt/core/sharding.py:246`
`build_sharded_requests`): for each plant, the system prompt + that plant's
images form a byte-identical prefix with an `ephemeral` cache breakpoint on the
last image block (`sharding.py:222` `_images_with_cache_breakpoint`). Only the
per-shard prompt text and `output_config.format` schema differ between shards of
the same plant. What differs between the two modes is *how* those requests are
sent — which decides whether the shared prefix actually hits the prompt cache.

### `--dispatch sequential`

`_dispatch_sequential` (`pxgpt/commands/phenotype.py:372`) is a plain Python loop
on the synchronous Messages API — `client.messages.create(...)` — grouped by
plant (all of plant A's shards, then plant B's, ...).

- **Caching is guaranteed, not best-effort.** A cache entry only becomes
  readable *after the request that wrote it has started responding.* In a serial
  loop, shard 1's call fully returns — so its write has definitely landed —
  before shard 2 fires. Shards 2..n of a plant therefore read the prefix every
  time. This is a structural guarantee of the transport, not a hope.
- Each shard fires immediately after the previous one returns, so they land well
  inside the 5-minute cache TTL. `cache_read_input_tokens` on shards 2..n is
  reliably non-zero.
- No batch discount — every call is billed at full, non-batch rates.
- `cache_read`/`cache_creation`/`input`/`output` totals are printed per call
  (`phenotype.py:405-407`), so hits vs. misses are directly observable.

### `--dispatch batch` (default)

`_dispatch_batch` (`pxgpt/commands/phenotype.py:327`) hands every plant×shard
request to `client.beta.messages.batches.create(...)` — the async **Message
Batches API** — in one call.

- Billed at Anthropic's flat **50% Batch API discount** on all token types
  (input, output, cache write, cache read).
- **Caching is best-effort, because batch requests run asynchronously and
  concurrently.** Anthropic's own guidance states that cache hit rates in batch
  workloads range from roughly **30% to 98%**, depending on traffic pattern.
  That wide band is the whole problem: it straddles the break-even (below).

**So the trade is:** sequential = full price, guaranteed caching. Batch =
half price, unpredictable caching. Which wins depends entirely on the batch's
actual cross-shard hit rate `h`.

## Why batch caching is unreliable: two failure modes

The original version of this note blamed only TTL expiry. There are actually
**two independent ways** a batch loses cache hits, and they behave differently.

1. **Cold-start race (the dominant risk).** A cache entry is readable only once
   the first writer has begun responding. If the batch backend picks up several
   of a plant's shards at the same time while the prefix is still cold, they all
   miss and all write. You pay the write premium `n` times instead of once. TTL
   length does nothing here — this is about *ordering*, not *lifetime*.

2. **TTL expiry over a long batch.** A batch can take longer than 5 minutes to
   work through. Even if shard 1 writes the prefix cleanly, the entry can expire
   before a plant's later shards are scheduled, forcing a re-write.

These two failure modes matter because **the 1-hour TTL only fixes the second
one.** See below.

## The 1-hour cache TTL: what it fixes and what it doesn't

Pricing (verified against Anthropic's current pricing docs, not recalled):
- Base `claude-sonnet-4-6` (`config.py:46`): **$3/MTok input, $15/MTok output**.
- Cache **read** ≈ **0.1×** base.
- Cache **write, 5-minute TTL** ≈ **1.25×** base (project default — no `ttl`
  override in `sharding.py`).
- Cache **write, 1-hour TTL** ≈ **2.0×** base, set via
  `"cache_control": {"type": "ephemeral", "ttl": "1h"}`.
- Batch's 50% discount **stacks** on top of all cache multipliers.

What the 1-hour TTL does:

- **Fixes TTL expiry.** For a batch that shares context and runs longer than 5
  minutes, a 1-hour entry survives the gap, so a plant's later shards can still
  read what shard 1 wrote. This is exactly the case Anthropic recommends it for.
- **Does NOT fix the cold-start race.** If shards fan out concurrently before any
  write lands, they still all miss.
- **Makes the cold-start race *worse* when it happens.** Under a full stampede,
  every shard writes at the 2.0× premium instead of 1.25×. More shards fanning
  out at once → higher stampede odds → 1-hour amplifies the damage.

So switching to 1-hour is not a free win. It helps only if your batch's misses
come from expiry, and hurts if they come from concurrency.

## Why pre-warming doesn't rescue batch

The obvious fix — fire one cheap request per plant to write the cache, then
submit the batch — does not work cleanly:

- Anthropic states pre-warming targets time-to-first-token, which does not apply
  to batch processing, and a warmed entry can expire before the batch runs.
- Economically it's worse than sequential anyway. A pre-warm write is billed at
  full (non-batch) rate. At 1-hour TTL that's `2.0` units for the prefix, plus
  `n` batch reads at `0.5 × 0.1` each → **≈ 2.35 units/plant**, above
  sequential's **1.85**.

Pre-warming buys you nothing here. Skip it.

## Cost model

Split each plant's cost into:

- **F** — shared prefix tokens (system prompt + images), repeated identically
  across all `n` shards. In a vision pipeline, images dominate, so F dominates.
- **M** — per-shard marginal tokens (shard prompt + output), small relative to F.

Let P be the base per-token price. Costs below are in units of `P·F` for the
prefix term.

**Sequential** (shard 1 writes at 1.25×, shards 2..n read at 0.1×, full price):

```
Cost_seq = P·F·[1.25 + 0.1(n-1)] + P·n·M
```

**Batch** (50% off everything; unknown cross-shard hit rate `h` for shards
2..n, where `h=1` means every later shard hits cache, `h=0` means every later
shard is a full miss):

```
Cost_batch = 0.5·P·F·[1.25 + (n-1)(1 - 0.9h)] + 0.5·P·n·M
```

Setting the F terms equal and solving for the break-even, with `x = n-1`:

```
x = 0.625 / (0.4 - 0.45h)
```

### This project's numbers

The Stage 3 shard set has **n = 7 shards/plant** (`shards_manifest.json`,
`shard_count: 7`), so `x = 6`. Solving for the break-even hit rate:

```
6 = 0.625 / (0.4 - 0.45 h*)   →   h* ≈ 0.657
```

**At 7 shards/plant, sequential is cheaper whenever batch's cross-shard hit rate
is below ~66%.** Anthropic's stated batch range is 30–98%, so the real `h` sits
on *both sides* of this line — that is precisely why batch's cost is a gamble.

## Four options, side by side (per plant, prefix term, n = 7)

| Mode | Per-plant prefix cost (× P·F) | Caching reliability |
|---|---|---|
| sequential, 5-min TTL | **1.85** (fixed) | Guaranteed by serial ordering |
| sequential, 1-hour TTL | 2.60 (fixed) | Guaranteed, but 1h write premium is wasted — **never use** |
| batch, 5-min TTL | **0.98 – 2.82** (depends on `h`) | Best-effort; `h` ∈ 30–98% |
| batch, 1-hour TTL | 1.30 (only if `h`→1) … up to 7.0 (full stampede) | Best-effort; fixes expiry, not the cold-start race |

Batch's range comes straight from Anthropic's 30–98% band applied to
`Cost_batch`. At `h ≈ 0.66` it equals sequential; below that batch is worse.
A full stampede costs `0.5 × n × 1.25 = 4.4` at 5-min TTL and
`0.5 × n × 2.0 = 7.0` at 1-hour — both far worse than sequential's fixed 1.85.

## How the break-even scales with shard count

The batch discount is a fixed one-time 50%. Sequential's cache reads compound:
every extra shard adds one more guaranteed 0.1× read — the cheapest token there
is. So **more shards tilt the balance toward sequential.** Solving the
break-even for `h` gives a closed form:

```
h* = 8/9 - (25/18)/(n-1)   ≈   0.889 - 1.389/(n-1)
```

`h*` is the hit rate batch must exceed to win. Sequential wins when `h < h*`.

| n (shards/plant) | break-even `h*` | sequential wins when |
|---|---|---|
| 2 | < 0 | batch always wins |
| 3 | 0.19 | h < 19% (batch usually wins) |
| 5 | 0.54 | h < 54% |
| 7 | 0.66 | h < 66% |
| 8 | 0.69 | h < 69% |
| 10 | 0.73 | h < 74% |
| 12 | 0.76 | h < 76% |
| n → ∞ | 0.889 | h < 89% |

`h*` increases monotonically in `n` (`dh*/dn = 1.389/(n-1)² > 0`) and is capped
at `8/9 ≈ 0.889`. Reading the table: at large n, batch wins only if it sustains
~89%+ cache hits — which best-effort batch scheduling rarely does. So a pipeline
whose shard count only ever grows should treat sequential as the default.

Two things make this even stronger, one weakens it slightly:

- **(Strengthens sequential)** You add shards by sharding the *schema* finer to
  dodge the grammar-size limit. That leaves the images untouched, so **F is
  unchanged** — you're only stacking cheap read-only shards onto each plant.
- **(Strengthens sequential)** Fanning out more identical-prefix requests at once
  raises cold-start collision odds, pushing batch's real `h` down just as `h*`
  rises. The two effects compound.
- **(Weakens it, second order)** The `n·M` term always favors batch (flat 50%
  off, no caching). It lowers `h*` by roughly `1.11 × (M/F)`. In an
  image-dominated pipeline `M/F` is tiny, so this is a minor correction — unless
  finer sharding inflates per-shard output (long `rationale`/`design_note`
  scaffolding), in which case watch it.

## How to decide in practice: one diagnostic

You do not need to estimate `h` from a formula. Run the pilot the manifest
already suggests (2–3 plants) through `--dispatch batch --wait`, and read the
per-plant token summary. The deciding number is **cache creations per plant**:

- **≈ 1 creation + (n-1) reads per plant** → the backend is staggering shards,
  write-before-read holds, `h` is high → **use batch**, and add `ttl: "1h"` on
  the full run to protect against expiry.
- **≈ n creations per plant** → cold-start stampede, `h` is on the floor →
  **use sequential**, and do **not** switch to 1-hour (it only raises the write
  premium on every redundant write).

`cache_creation_input_tokens` vs `cache_read_input_tokens` per plant collapses
the entire question into one observation — far more direct than back-solving
`h` through `x = 0.625/(0.4 - 0.45h)`.

## Practical caveats beyond cost-per-token

- **Throughput / wall-clock.** Sequential is a serial loop of synchronous calls.
  At this project's scale (~284 materials × 7 shards ≈ 2,000 requests) it runs
  in roughly an hour and won't hit Sonnet rate limits, so it's operationally
  fine. But as `n` grows, wall-clock grows with it — the cost math and the
  latency math pull in opposite directions.
- **Output tokens never cache.** Caching only ever discounts the input prefix.
  Per-shard output is discounted only by batch's 50%. The larger your structured
  output per shard, the more the output term alone tilts toward batch — a factor
  independent of the caching story.

---

*Pricing and batch cache behavior verified against Anthropic's current pricing
and batch-processing / prompt-caching documentation. Cost figures are relative
(units of `P·F`); confirm absolute rates at claude.com/pricing before budgeting.*

