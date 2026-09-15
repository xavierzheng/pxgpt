# Experiment: does raising `--shard-budget` on OpenAI cost quality?

**Date:** 2026-08-14
**Conclusion updated:** 2026-09-15
**Model:** `gpt-5.6-luna` (`STAGE3_EFFORT` off → `reasoning.effort: "none"`, `TEMPERATURE=0.5`)
**Plants:** experiment A — 1 (`s0001`); experiment B — 10 (`s0001`–`s0011`)

> **Status:** the original quality conclusion from this pilot is superseded. These
> runs compared model outputs with one another but not with human reference labels,
> so they measured disagreement rather than accuracy. A later human-referenced
> ablation tested 119 `mature_v2` plants, with two replicates per condition. Mean
> categorical accuracy was 0.7066 with the frozen nine-shard set (budget 40) and
> 0.7150 without sharding (one request, budget 320). The difference was +0.0084
> (95% CI −0.0006 to +0.0175; paired t-test p = 0.0712), so no accuracy loss was
> detected. This document remains useful for its historical cost, output-difference
> and rationale-length measurements, but it does not show that a larger shard
> budget reduces scoring quality.
>
> The raw outputs and study-specific analysis artifacts are not distributed with
> the pxGPT software repository. If released, they should be deposited as a
> versioned research-data archive and cited here with a persistent identifier.

## Question

Sharding exists because a large Stage 3 schema exceeds *Anthropic's* grammar-size
limit. OpenAI has no such limit at this scale, so a bigger `--shard-budget` means
fewer requests per plant and, because every request repeats the same ~30 k-token
image payload, a nearly proportional cut in input cost. The open question was
whether packing more traits into one request degrades the scoring.

## Updated answer

**Increasing the shard budget reduced input cost and changed some outputs in this
10-plant pilot, but the pilot cannot identify those differences as errors because
it has no human reference. The later 119-plant ablation found no detectable
accuracy loss when sharding was removed (budget 320, one request).**

Two experiments, in this order. The second supersedes the first wherever they
overlap:

| experiment | plants | runs per config | scope |
|---|---|---|---|
| **A** (below) | 1 (`s0001`) | 2 | budgets 40 / 80 / 320 + Anthropic reference |
| **B** ([below](#experiment-b-budget-80-on-10-plants)) | 10 | 2 | budgets 40 / 80 only |

Headline numbers, disagreeing traits out of 49:

| comparison | experiment A (n=1) | experiment B (n=10) |
|---|---|---|
| run-to-run noise, budget 40 | 5.0 | **7.00** (sd 1.49) |
| run-to-run noise, budget 80 | 5.0 | **6.10** (sd 1.60) |
| budget 40 → budget 80 | 6.5 | **8.22** (sd 2.30) |
| budget 40 → budget 320 | 10.2 | *not re-tested* |
| *switching provider* (Anthropic b40 → OpenAI b40) | 10.0 | *not re-tested* |

**The single-plant run underestimated the noise floor.** `s0001` happened to be a
quiet plant; across 10 plants a configuration disagrees with *itself* on 7 of 49
traits on average, not 5. That reframes everything: budget 80's excess over noise
is **+1.68 traits of the pooled 49** (paired t(9) = 3.03, p = 0.014,
95% CI +0.42 … +2.93), which is statistically detectable but modest —
**re-running budget 40 already changes ~7 traits; switching to budget 80 changes
~8.2.** This is excess disagreement, not evidence of lower accuracy. Over the 45
categorical traits alone the same test gives **+1.88** (t(9) = 3.23, p = 0.010,
95% CI +0.56 … +3.19) — see the subset table below, and quote the subset with the
number.

**Split the 4 quantitative traits out and the picture is cleaner.** They are
eyeballed against a nearby ruler / colorchecker over several camera angles, so
their inconsistency is expected and accepted; pooling them into an agreement rate
muddies what that rate means:

| trait subset | b40 noise | b80 noise | b40 vs b80 | excess | 95% CI | t(9) | p | plants > own noise |
|---|---|---|---|---|---|---|---|---|
| all 49 (pooled) | 7.00 | 6.10 | 8.22 | +1.68 | +0.42 … +2.93 | 3.03 | 0.0143 | 7 of 10 |
| **categorical (45)** | **4.60** (10.2%) | **3.50** (7.8%) | **5.92** (13.2%) | **+1.88** | **+0.56 … +3.19** | **3.23** | **0.0104** | **8 of 10** |
| quantitative (4) | 2.40 | 2.60 | 2.30 | −0.20 | −0.51 … +0.11 | −1.44 | 0.18 (ns) | 2 of 10 |

**Every excess figure in this document belongs to one of these three rows.** The
pooled row and the categorical row are both correct and they are not
interchangeable — always state which subset a number comes from.

Quantitative traits carry **no budget signal at all**, and budget 80 is no worse
than a re-run. Their exact-match rate (2.40 of 4 changing between two runs of the
same configuration) is an **artifact of counting a continuous measurement as a
category**, not a scoring failure — read their repeatability as a CV instead
([below](#quantitative-traits-in-their-own-units)): 3% on leaf count, 8% on height, 9% on blade length,
28% on canopy spread.

Excluding them makes the categorical excess disagreement slightly *clearer*
(+1.88 of 45, p = 0.010, 8 of 10 plants above their own noise).
Categorical reproducibility is **89.8% raw agreement, Gwet's AC1 0.88** — better
than the 86% the pooled 49-trait count implies.

Per-trait rationale length — the chain-of-thought the schema deliberately forces
before each value — shortens robustly: 180.6 → 152.9 characters (−15%, n = 490
each). Completeness and the `not_assessable` rate are identical between budgets.

Neither variant is *wrong*: there is no ground truth in either experiment, so a
divergence is a difference, not an error. Which is more accurate needs the manual
corrections.

# Experiment A — one plant, three budgets

## Experimental setup

Experiment A used 15 images of one `mature_v1` plant and the same 49-trait master
schema at every budget. Fresh shard sets were generated at budgets 40, 80 and 320.
The frozen budget-40 Anthropic production result was included only as a
cross-provider comparison; it was not treated as ground truth. The frozen inputs
and shard sets were verified unchanged across runs.

Record labels used throughout:

| label | provider / model | shards | dispatch |
|---|---|---|---|
| `A-b40` | anthropic `claude-sonnet-5` | 10 | batch |
| `O-b40-s` | openai `gpt-5.6-luna` | 10 | sequential |
| `O-b40-b` | openai `gpt-5.6-luna` | 10 | batch |
| `O-b80-1` | openai `gpt-5.6-luna` | 4 | sequential |
| `O-b80-2` | openai `gpt-5.6-luna` | 4 | sequential |
| `O-b320-1` | openai `gpt-5.6-luna` | 1 | sequential |
| `O-b320-2` | openai `gpt-5.6-luna` | 1 | sequential |

Two runs per OpenAI configuration, so each configuration supplies its own
run-to-run noise floor rather than being compared against an assumed one. For
budget 40 the two runs are `--dispatch sequential` and `--dispatch batch`, which
send byte-identical request bodies (verified: both cost exactly 308,397 input
tokens), so they function as two samples of one configuration.

## Shard-set generation

`pxgpt shard-schema` generated a fresh shard set at each budget. All sets retain
**49/49 traits** and `not_assessable` on
**every** nominal/ordinal enum — the generator injects that value (45 of 45 such
traits in this master, none of which list it themselves), which is why raising the
budget is acceptable but hand-writing a big schema is not.

| `--shard-budget` | shards | largest shard: props / enum values / depth | live OpenAI compile-check |
|---|---|---|---|
| 40 (current) | 10 | 25 / 26 / 4 | accepted |
| 80 | 4 | 46 / 52 / 4 | accepted |
| 160 | 2 | 93 / 93 / 4 | accepted |
| 320 | 1 | 159 / 165 / 4 | accepted |

OpenAI strict-mode caps for reference: 5,000 object properties, 1,000 enum values,
depth 10. Even the single-shard schema is an order of magnitude inside them.

## Cost — `s0001` only (15 images)

⚠ **These are one plant's tokens. Do not extrapolate them to the collection** —
see [experiment B's cost table](#result-cost-measured-over-10-plants), which is the figure to quote.

| config | input tokens / plant | output tokens | $/plant |
|---|---|---|---|
| budget 40 (10 shards) | **308,397** | 2,583 / 2,671 | $0.065 |
| budget 80 (4 shards) | **127,691** | 2,311 / 2,337 | $0.028 |
| budget 320 (1 shard) | **37,355** | 1,881 / 2,001 | $0.010 |

All input/output figures are measured. Pricing at `gpt-5.6-luna` $0.20/M input,
$1.20/M output.

`s0001` carries **15 images**, against a 19.9-image mean over experiment B's ten
plants and a **19.6-image mean over the full 142-plant collection**
(2,784 images / 142 plants). Because input
scales with images, these per-plant figures under-state a whole-collection cost by
about a quarter: budget 40 measures **403,506** input tokens per plant over ten
plants, not 308,397.

Input scales almost exactly with shard count because the ~30 k-token image payload
is repeated per request — that repetition, not the schema, is the bill.

## Result 1 — completeness is unaffected

| record | traits | `not_assessable` |
|---|---|---|
| `A-b40` | 49 / 49 | 2 |
| `O-b40-s` / `O-b40-b` | 49 / 49 | 2 / 1 |
| `O-b80-1` / `O-b80-2` | 49 / 49 | 1 / 1 |
| `O-b320-1` / `O-b320-2` | 49 / 49 | 1 / 1 |

No configuration dropped a trait, and none showed the "gives up under load"
failure mode of a rising `not_assessable` rate. No truncation either: the
single-shard response used 1,881 output tokens against a 16,384 cap.

## Result 2 — rationale length shortens monotonically

Characters per trait rationale:

| record | mean | median | min | max |
|---|---|---|---|---|
| `A-b40` (anthropic) | **261** | 250 | 113 | 607 |
| `O-b40-s` | 178 | 172 | 90 | 290 |
| `O-b40-b` | 173 | 169 | 99 | 305 |
| `O-b80-1` | 152 | 155 | 61 | 238 |
| `O-b80-2` | 154 | 151 | 72 | 241 |
| `O-b320-1` | **113** | 111 | 52 | 174 |
| `O-b320-2` | 124 | 125 | 52 | 178 |

The shortening is uniform, not concentrated on the traits that end up
disagreeing (−32% on agreeing traits, −38% on disagreeing ones).
`shard_builder.trait_object()` declares `rationale` *before* `value` so that the
rationale can condition the categorical value under autoregressive decoding.
Rationale length alone is not a measure of reasoning or accuracy: the later
119-plant ablation also found shorter rationales without a detectable loss in
accuracy.

A concrete pair, from a trait where the two configurations disagree:

```
leaf_blade.leaf_heterophylly_presence

  budget 40  ->  value = "present"   (279 chars)
    "Across the whole-plant and top-down views, the plant shows smooth, rounded/
     glaucous older leaves together with distinctly narrower, more upright,
     strongly undulate/savoyed younger leaves in the central crown. This is a
     clear difference in blade form beyond simple size variation."

  budget 320 ->  value = "absent"    (116 chars)
    "Leaves vary mainly in size and maturity; the apparent differences do not
     constitute two clearly distinct leaf forms."
```

The budget-40 rationale cites specific structures and reaches `present`; the
budget-320 one is generic and reaches the opposite conclusion. Note this is an
illustration of an output difference, not evidence about which answer is correct
or why the values differ.

## Result 3 — pairwise agreement

Agreeing traits out of 49:

```
          A-b40   O-b40-s O-b40-b O-b80-1 O-b80-2 O-b320-1 O-b320-2
A-b40        49       39      39      39      38       38       36
O-b40-s      39       49      44      43      44       38       39
O-b40-b      39       44      49      42      41       39       39
O-b80-1      39       43      42      49      44       40       38
O-b80-2      38       44      41      44      49       39       41
O-b320-1     38       38      39      40      39       49       44
O-b320-2     36       39      39      38      41       44       49
```

As disagreements (mean over all cross pairs in each class):

| comparison | disagreeing traits / 49 | individual pairs |
|---|---|---|
| within budget 40 (seq vs batch) | 5.0 | 5 |
| within budget 80 (run1 vs run2) | 5.0 | 5 |
| within budget 320 (run1 vs run2) | 5.0 | 5 |
| budget 40 ↔ budget 80 | 6.5 | 6, 5, 7, 8 |
| budget 40 ↔ budget 320 | 10.2 | 11, 10, 10, 10 |
| budget 80 ↔ budget 320 | 9.5 | 9, 11, 10, 8 |
| anthropic b40 ↔ openai b40 | 10.0 | 10, 10 |
| anthropic b40 ↔ openai b320 | 12.0 | 11, 13 |

Two readings matter:

1. **The observed noise floor is 5/49 for every configuration.** In these two
   runs, budget 320 was as self-consistent as budget 40 but produced different
   values on some fields. With one plant and no human reference, this cannot be
   interpreted as an accuracy difference.
2. **The budget-320 difference (10.2) was similar in size to the provider
   difference (10.0) on this plant.** These are output-disagreement counts, not
   measures of which configuration is more accurate. Budget 80's difference
   (6.5) was 1.5 traits above the observed within-configuration disagreement.

## Result 4 — the systematically shifted traits

Traits where *both* runs of a configuration disagree with *both* budget-40 runs
(so the divergence cannot be a single unlucky sample):

**Budget 80 — 4 of 49**

| trait | budget 40 | budget 80 |
|---|---|---|
| `whole_plant_architecture.plant_canopy_spread` | 12.0, 14.0 | 10.0, 11.0 |
| `leaf_blade.leaf_blade_length` | 5.0 | 6.0, 7.0 |
| `leaf_surface.leaf_surface_texture` | 1 | 2 |
| `petiole.petiole_cross_section_shape` | flattened_channeled, not_assessable | slender_ungrooved |

**Budget 320 — 7 of 49**

| trait | budget 40 | budget 320 |
|---|---|---|
| `whole_plant_architecture.plant_canopy_spread` | 12.0, 14.0 | 12.5, 13.0 |
| `leaf_blade.leaf_blade_length` | 5.0 | 8.0 |
| `leaf_blade.leaf_heterophylly_presence` | present | absent |
| `leaf_surface.leaf_surface_texture` | 1 | 2 |
| `petiole.petiole_relative_length` | 2 | 1 |
| `root_system.root_hair_visibility` | sparse_or_absent | present |
| `root_system.root_colonization_extent` | 1 | 2 |

Three traits shift at *both* budgets (`plant_canopy_spread`,
`leaf_blade_length`, `leaf_surface_texture`), which suggests those are simply the
least stable traits rather than a budget effect. The budget-320-only shifts add
categorical calls (`heterophylly present → absent`, `root_hair sparse_or_absent →
present`) — changes of kind, not degree.

## Limitations of experiment A

- **n = 1 plant, 2 runs per configuration.** Superseded by experiment B for
  budgets 40 and 80: B measures a higher noise floor (7.0 vs 5.0) and shows the
  "systematically shifted traits" of Result 4 do not concentrate across plants.
  The budget-320 numbers above remain n = 1.
- Only `s0001` was used (15 images, a complete set). Plants with fewer or poorer
  images may behave differently.

# Experiment B: budget 80 on 10 plants

Experiment A's single plant could not separate a budget effect from run-to-run
variation with any confidence, and its noise estimate turned out to be low. B
repeats the budget 40 vs budget 80 comparison on 10 plants, two runs per
configuration, so each plant contributes its own noise floor.

## Experimental setup

The 10 plants are the first ten by ID (note `s0005` does not exist in the image
tree). Image counts vary 15–26 per plant, 199 images total — deliberately not
uniform, so the result is not specific to one image count. The same master schema,
system prompt and generated field definitions were used at both budgets.

Four `--dispatch batch` submissions, 280 requests total, **0 failures**:

| run | shards | requests | JSONL | input tokens | output tokens |
|---|---|---|---|---|---|
| `b40-r1` | 10 | 100 | 969,690 B | 4,035,060 | 26,641 |
| `b40-r2` | 10 | 100 | 969,690 B | 4,035,060 | 26,380 |
| `b80-r1` | 4 | 40 | 597,932 B | 1,657,346 | 23,084 |
| `b80-r2` | 4 | 40 | 597,932 B | 1,657,346 | 23,034 |

The two runs of each configuration consumed **exactly** the same input tokens,
which confirms they are two samples of one configuration rather than two different
requests. All 199 images were uploaded once into a shared manifest and reused by
`file_id` across all four runs.

## Result: cost measured over 10 plants

**This is the cost table to quote.** It replaces experiment A's single-plant
figures for any collection-level estimate.

| config | input tok / plant | $/plant | 142 plants | 142 plants, `--dispatch batch` |
|---|---|---|---|---|
| budget 40 | **403,506** | $0.0839 | **$11.91** | $5.96 |
| budget 80 | **165,735** | $0.0359 | **$5.10** | $2.55 |

Measured over the 10 plants above at `gpt-5.6-luna` $0.20/M input, $1.20/M output;
the batch column applies the Batch API's documented 50% discount. Input ratio
b40 / b80 = **2.43×**.

The extrapolation to 142 plants is sound because input scales with images per
plant, and these 10 plants average **19.9 images** against **19.6 for the whole
142-plant collection** (2,784 images / 142 plants). Experiment A's `s0001` has only
15, which is why its 308,397 tokens per plant under-state the collection by ~24%.

## Validity check: the two shard sets differ ONLY in partitioning

The frozen budget-40 set was generated on 2026-07-30. A changed master schema
would have confounded the comparison, so the two generated shard sets were checked
directly:

- Regenerating budget 40 from today's master reproduces the frozen shard schemas
  **byte-identically** (all 10 files, sha256).
- The budget-80 set carries the same 49 traits with **0** differing value
  constraints (enum member lists and types compared leaf by leaf).

So the only variable between the two configurations is how the traits are packed
into requests.

## Result — per-plant disagreement (of 49 traits)

| plant | images | b40 noise (r1 vs r2) | b80 noise (r1 vs r2) | b40 vs b80 (mean of 4 cross pairs) |
|---|---|---|---|---|
| `s0001` | 15 | 8 | 4 | 5.75 |
| `s0002` | 16 | 5 | 8 | 6.00 |
| `s0003` | 18 | 6 | 8 | 11.00 |
| `s0004` | 17 | 7 | 6 | 10.50 |
| `s0006` | 17 | 6 | 5 | 6.75 |
| `s0007` | 21 | 6 | 7 | 9.75 |
| `s0008` | 24 | 8 | 5 | 8.25 |
| `s0009` | 22 | 10 | 8 | 11.50 |
| `s0010` | 23 | 6 | 6 | 7.25 |
| `s0011` | 26 | 8 | 4 | 5.50 |
| **mean** | | **7.00** | **6.10** | **8.22** |
| sd | | 1.49 | 1.60 | 2.30 |

Pooled within-configuration noise floor: **6.55 / 49**. Budget 80 divergence:
**8.22 / 49**.

### Is that excess real?

Paired per plant (each plant's cross-config divergence minus its own noise floor),
so plants that are simply unstable cannot create the effect:

```
ALL 49 TRAITS (pooled)
per-plant excess: -0.25, -0.50, +4.00, +4.00, +1.25, +3.25, +1.75, +2.50, +1.25, -0.50
n = 10   mean = +1.68 traits   sd = 1.75   se = 0.55
t(9) = 3.03   two-sided p = 0.0143   95% CI  +0.42 … +2.93
7 of 10 plants diverged more than their own noise

CATEGORICAL ONLY (45 traits) -- the comparison that speaks to scoring
per-plant excess: +0.50, -0.50, +4.50, +4.50, +1.50, +3.50, +1.50, +2.75, +0.50,  0.00
n = 10   mean = +1.88 traits   sd = 1.84   se = 0.58
t(9) = 3.23   two-sided p = 0.0104   95% CI  +0.56 … +3.19
8 of 10 plants diverged more than their own noise
```

The pooled +1.68 / CI +0.42 … +2.93 / "7 of 10" and the categorical
+1.88 / CI +0.56 … +3.19 / "8 of 10" are different measurements, not a
disagreement.

The excess disagreement is detectable and small. Its confidence interval spans
"half a trait" to "three traits". Image count does not predict divergence
(Pearson r = +0.03, n = 10).

### The most important number is the noise floor itself

A configuration disagrees with **itself** on 7 of 49 traits between two runs at
`TEMPERATURE=0.5`. Stated as a rate that is 86%, but **the pooled figure is the
wrong one to quote**: it drags the 4 ruler-eyeballed quantitative traits into a
count of categorical agreement. On the 45 categorical traits the same two runs give
**89.8% raw agreement and Gwet's AC1 0.88**. The appropriate headline is therefore
**~90% categorical run-to-run reproducibility, AC1 0.88**,
with the quantitative traits reported separately as CVs.

Either way this is a property of the current production setup, not of budget 80,
and it is larger than the measured excess disagreement between budgets 40 and 80.
Any downstream analysis that treats a single Stage 3 run as a fixed measurement
is absorbing ~10% per-trait categorical instability already.

## Result — rationale length

Pooled over 10 plants × 49 traits (n = 490 per run):

| run | mean | median | min | max |
|---|---|---|---|---|
| `b40-r1` | 180.6 | 174 | 88 | 337 |
| `b40-r2` | 177.8 | 173 | 90 | 350 |
| `b80-r1` | **152.9** | 153 | 61 | 241 |
| `b80-r2` | **153.2** | 151 | 71 | 268 |

−15% at budget 80, reproducing experiment A's single-plant figure (178 → 152)
almost exactly. This is the most robust effect in either experiment.

## Result — completeness and `not_assessable`

| run | traits | `not_assessable` |
|---|---|---|
| `b40-r1` / `b40-r2` | 490 / 490 | 10 (2.0%) / 11 (2.2%) |
| `b80-r1` / `b80-r2` | 490 / 490 | 10 (2.0%) / 11 (2.2%) |

Identical. No trait dropped, no gaps file written in any of the four runs, and no
sign of the "gives up under load" failure mode.

## Result — the shifts are diffuse, not trait-specific

Cells where *both* budget-80 runs disagree with *both* budget-40 runs:
**33 of 490 (6.7%)**, spread across **23 distinct traits**. Only one trait is hit
in as many as 4 of 10 plants (`whole_plant_architecture.plant_canopy_spread`) —
and that one is **quantitative, with ±4.4 cm of spread between two runs of the same
configuration**, so it is measurement noise rather than a budget effect. Retracted
accordingly.

This corrects experiment A twice over. From one plant it looked as though budget 80
shifted a specific handful of traits; across ten it does not concentrate anywhere,
and the single apparent concentration is a ruler-eyeballed measurement. Budget 80
produces diffuse differences rather than a directional shift in particular traits.

## Quantitative traits, in their own units

Their run-to-run inconsistency is expected and accepted (rough measurement against
a nearby ruler / colorchecker, from several camera angles), so they are reported as
magnitudes rather than folded into an agreement count.

**Use the CV. The exact-match rate is an artifact.** Counting a continuous
measurement as "agreeing" only when two runs print the identical number makes 10.0
vs 10.5 cm a full disagreement, which is why these four traits show a ~60%
"disagreement" rate *within a single configuration*. That number describes the
metric, not the model. The repeatability figure that means something is the
coefficient of variation — mean |run 1 − run 2| over the trait mean, budget 40,
10 plants:

| trait | unit | mean | mean \|b40r1 − b40r2\| | **CV** | mean \|b40 − b80\| |
|---|---|---|---|---|---|
| `plant_true_leaf_number` | count | 10.1 | 0.30 | **3%** | 0.80 |
| `plant_height` | cm | 12.2 | 1.00 | **8%** | 0.70 |
| `leaf_blade_length` | cm | 9.5 | 0.85 | **9%** | 0.75 |
| `plant_canopy_spread` | cm | 15.7 | **4.40** | **28%** | 2.05 |

Read this way three of the four are repeatable to within 10%, and only
`plant_canopy_spread` is genuinely unusable at ±28%. The 60% exact-match figure
would have condemned all four equally.

For three of the four, the *within*-configuration spread is as large as or larger
than the between-budget difference. Budget choice is not what determines these
values.

---

# Recommendation

- **Keep budget 40 for anything compared against the human evaluation.** That
  evaluation runs against the frozen budget-40 shard set; re-sharding makes the
  model configuration different, regardless of whether accuracy changes.
- **A larger budget can reduce OpenAI input cost for a new collection.** In this
  pilot, budget 80 gave a 2.43× reduction
  ($11.91 → $5.10 per 142 plants sequentially, or $5.96 → $2.55 via
  `--dispatch batch`). It produced +1.68 traits of disagreement beyond the
  within-configuration rate, but this pilot could not determine whether those
  changes were more or less accurate.
- **Budget 320 is technically viable on OpenAI.** The later 119-plant ablation
  completed with one request per plant and found no detectable accuracy loss
  relative to budget 40. This result supports its use on that tested dataset; it
  does not guarantee the same result for every schema, model or collection.
- **Fix the provider, model, frozen schema and shard configuration within a formal
  comparison.** This keeps the model inputs and output contract reproducible and
  avoids attributing configuration differences to biological variation.
- **Consider whether one run per plant is enough at all.** The 7/49 noise floor is
  the largest effect measured here. If per-trait reliability matters, two runs and
  a disagreement flag would buy more than any budget change — and at budget 80,
  two runs cost less than one run at budget 40.

# Limitations

- **No ground truth in either experiment.** Every comparison is between model
  configurations, so "more divergent" is not "less accurate". The later
  human-referenced 119-plant ablation supplies the accuracy comparison that this
  pilot lacked and supersedes its original quality conclusion.
- **10 plants, 2 runs each.** Enough to establish the noise floor and to detect
  excess disagreement at budget 80 (p = 0.014), not enough to characterise which
  *kinds* of traits are affected — 23 traits with 1–4 hits each is too sparse for
  that.
- **Budgets 160 and 320 were only compile-checked on 10 plants, not scored.** The
  budget-320 result reported in this historical pilot is n = 1; the later
  119-plant ablation used a different dataset and frozen schema.
- Cost figures scale with images per plant (15–26 here, mean 19.9; the full
  142-plant collection means 19.6, so these 10 plants extrapolate cleanly). A
  collection with more images per plant pays proportionally more, and the budget
  saving grows in absolute terms.
- All runs used `STAGE3_EFFORT` off (`reasoning.effort: "none"`) and
  `TEMPERATURE=0.5`. Reasoning-enabled runs may behave differently — untested.
