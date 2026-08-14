# Experiment: does raising `--shard-budget` on OpenAI cost quality?

**Date:** 2026-08-14
**Model:** `gpt-5.6-luna` (`STAGE3_EFFORT` off → `reasoning.effort: "none"`, `TEMPERATURE=0.5`)
**Plant:** one — `s0001`

## Question

Sharding exists because a large Stage 3 schema exceeds *Anthropic's* grammar-size
limit. OpenAI has no such limit at this scale, so a bigger `--shard-budget` means
fewer requests per plant and, because every request repeats the same ~30 k-token
image payload, a nearly proportional cut in input cost. The open question was
whether packing more traits into one request degrades the scoring.

## Answer

**Budget 80 (4 shards) is defensible; budget 320 (1 shard) is not.**

Divergence from the current budget-40 configuration, measured against the
run-to-run noise floor of the same configuration:

| comparison | disagreeing traits (of 49) |
|---|---|
| run-to-run noise, *any* single configuration | **5.0** |
| budget 40 → budget 80 | 6.5 |
| budget 40 → budget 320 | **10.2** |
| *switching provider entirely* (Anthropic b40 → OpenAI b40) | **10.0** |

Budget 320 changes the phenotype about as much as **changing model provider**.
Budget 80 sits close to the noise floor. Per-trait rationale length — the
chain-of-thought the schema deliberately forces before each value — shortens
monotonically as shards grow: 178 → 152 → 113 characters.

Neither variant is *wrong*: there is no ground truth in this experiment, so a
divergence is a difference, not an error. What the data does establish is that
budget 320 systematically shifts values while visibly gathering less evidence per
trait, which is the wrong direction for a study whose validity rests on the
rationale.

## Data provenance

Everything below is reproducible from these exact paths. Nothing in
`shard_master_schema/` or `Result_Stage3/` was modified (all 22 shard-set files
verified byte-identical by sha256 after every run).

| what | path |
|---|---|
| images (15 × `.jpg`) | `/home/xavier/project/pxgpt/02_mature_v1/images/s0001` |
| master schema | `/home/xavier/project/pxgpt/02_mature_v1/master_schema_v2.json` |
| system prompt (override) | `/home/xavier/project/pxgpt/02_mature_v1/system_2_schema.txt` |
| shard set, budget 40 (frozen, under human evaluation) | `/home/xavier/project/pxgpt/02_mature_v1/shard_master_schema` |
| shard set, budget 80 (generated for this experiment) | `/home/xavier/project/pxgpt/02_mature_v1/Result_openai_shard_budget_pilot/shardset_budget_80` |
| shard set, budget 320 (generated for this experiment) | `/home/xavier/project/pxgpt/02_mature_v1/Result_openai_shard_budget_pilot/shardset_budget_320` |
| **Anthropic reference result** (`claude-sonnet-5`, budget 40) | `/home/xavier/project/pxgpt/02_mature_v1/Result_Stage3/s0001.json` |
| all raw OpenAI results from this experiment | `/home/xavier/project/pxgpt/02_mature_v1/Result_openai_shard_budget_pilot/` |

The Anthropic reference is **your existing production run**, not something
generated here — it is included only as a cross-provider yardstick for how large a
"real" difference looks.

Record labels used throughout:

| label | provider / model | shards | dispatch | raw file (under `Result_openai_shard_budget_pilot/`) |
|---|---|---|---|---|
| `A-b40` | anthropic `claude-sonnet-5` | 10 | batch | *(your `02_mature_v1/Result_Stage3/s0001.json`)* |
| `O-b40-s` | openai `gpt-5.6-luna` | 10 | sequential | `Result_Stage3_openai/s0001.json` |
| `O-b40-b` | openai `gpt-5.6-luna` | 10 | batch | `Result_Stage3_openai_batch/s0001.json` |
| `O-b80-1` | openai `gpt-5.6-luna` | 4 | sequential | `Result_b80_run1/s0001.json` |
| `O-b80-2` | openai `gpt-5.6-luna` | 4 | sequential | `Result_b80_run2/s0001.json` |
| `O-b320-1` | openai `gpt-5.6-luna` | 1 | sequential | `Result_b320_run1/s0001.json` |
| `O-b320-2` | openai `gpt-5.6-luna` | 1 | sequential | `Result_b320_run2/s0001.json` |

Two runs per OpenAI configuration, so each configuration supplies its own
run-to-run noise floor rather than being compared against an assumed one. For
budget 40 the two runs are `--dispatch sequential` and `--dispatch batch`, which
send byte-identical request bodies (verified: both cost exactly 308,397 input
tokens), so they function as two samples of one configuration.

## Shard-set generation

`pxgpt shard-schema --master master_schema_v2.json --shard-budget N`, into a fresh
directory each time. All sets retain **49/49 traits** and `not_assessable` on
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

## Cost

| config | input tokens / plant | output tokens | $/plant | 142 plants | 142 plants via `--dispatch batch` |
|---|---|---|---|---|---|
| budget 40 (10 shards) | **308,397** | 2,583 / 2,671 | $0.065 | **$9.21** | $4.61 |
| budget 80 (4 shards) | **127,691** | 2,311 / 2,337 | $0.028 | **$4.02** | $2.01 |
| budget 320 (1 shard) | **37,355** | 1,881 / 2,001 | $0.010 | **$1.39** | $0.70 |

All input/output figures are measured. Pricing at `gpt-5.6-luna` $0.20/M input,
$1.20/M output; the batch column applies the Batch API's documented 50% discount.

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
disagreeing (−32% on agreeing traits, −38% on disagreeing ones). Because
`shard_builder.trait_object()` declares `rationale` *before* `value` specifically
to force chain-of-thought under autoregressive decoding, a 32% shorter rationale
means measurably less reasoning per trait. That is the design intent being
partially defeated, and it is the most likely mechanism behind Result 3.

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
illustration of the mechanism, not evidence about which answer is correct.

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

1. **The noise floor is 5/49 for every configuration.** Budget 320 is not
   *noisier* than budget 40 — it is equally self-consistent and systematically
   different. That rules out "the single shard is just less reliable" and points
   at a shift in behaviour instead.
2. **Budget 320's shift (10.2) equals a provider change (10.0).** That is the
   scale of the decision being made. Budget 80's shift (6.5) is 1.5 traits above
   noise.

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

## Recommendation

- **Keep budget 40 for anything compared against the human evaluation.** That
  evaluation is running against the frozen budget-40 shard set; re-sharding makes
  the results incomparable regardless of quality.
- **Budget 80 is a reasonable cost/quality trade** if a new collection needs to be
  cheaper: 2.3× cheaper, 4 systematic trait shifts, 1.5 traits above noise. Worth
  confirming on more plants before committing.
- **Do not use budget 320.** A 6.6× saving that moves the phenotype as much as
  changing provider, while cutting per-trait reasoning by a third.
- Keep provider and shard budget **fixed within a collection**. Both move results
  by ~10/49 traits, so mixing configurations inside one dataset would be
  indistinguishable from real biological variation.

## Limitations

- **n = 1 plant, 2 runs per configuration.** The rationale-length effect is robust
  (uniform across 4 runs, monotonic in budget). The value-shift magnitudes are a
  clear signal but not statistically established; 5–10 plants would be needed to
  put an interval on them.
- **No ground truth.** All comparisons are between model configurations. Which
  configuration is more *accurate* cannot be answered here — that needs the human
  corrections. The full per-trait table below has an empty `human` column for
  exactly that purpose.
- Only `s0001` was used, and it is one of the plants with a complete image set (15
  images). Plants with fewer or poorer images may behave differently.
- Cost figures assume the measured ~30 k-token image payload, which scales with
  image count per plant.

## `pxgpt json-to-table` verification on OpenAI output

Separately requested: confirm `json-to-table` handles OpenAI-produced Stage 3
JSON. It does — the merged records are provider-agnostic by construction, since
both providers go through the same `merge_sharded_results`.

```bash
pxgpt json-to-table \
  --result-dir Result_Stage3_openai_batch \
  --master-schema 02_mature_v1/master_schema_v2.json \
  --shard-dir 02_mature_v1/shard_master_schema \
  --out-prefix table_openai_b40
# -> Rows: 2   Columns: 50
```

Verified against the source JSON for all three scale types:

| scale type | JSON value | table column | table value | notes |
|---|---|---|---|---|
| quantitative | `9.0` | `plant_height_cm` | `9.0` (float64) | unit suffix appended, numeric dtype |
| ordinal | `2` | `stem_elongation` | `slightly_elongated` | integer level reconstructed to the schema label |
| ordinal | `0` | `leaf_blade_anthocyanin_coverage` | `none` | level 0 handled (not treated as missing) |
| nominal | `open_spreading` | `plant_growth_habit` | `open_spreading` | plain string passthrough |

The feather file carries ordinal columns as **ordered** pandas Categoricals
(`ordered=True` confirmed), so `arrow::read_feather()` reads them as ordered
factors in R. 2 rows for 2 plants, 50 columns = 49 traits + `cultivar_id`.

Output archived at
`Result_openai_shard_budget_pilot/json_to_table_verification_b40.csv`.

Run `json-to-table` only on a **completed** result set: recover any shards left
missing (via `--dispatch sequential` to the same `--output`, which reads
`_partial/`) before tabulating.

## Reproducing this

```bash
set -a && source project_A.env && set +a     # OPENAI_API_KEY; STAGE3_EFFORT unset (off)
cd /home/xavier/project/pxgpt/02_mature_v1

# 1. generate a shard set at the budget under test (writes a NEW directory)
pxgpt shard-schema --master master_schema_v2.json \
    --shard-dir /tmp/b80 --shard-budget 80

# 2. one plant, two independent runs -> separate --output dirs
mkdir -p /tmp/one/s0001 && cp images/s0001/*.jpg /tmp/one/s0001/
for R in 1 2; do
  pxgpt phenotype-batch-openai \
    --input-dir /tmp/one \
    --shard-dir /tmp/b80 \
    --system-prompt system_2_schema.txt \
    --dispatch sequential \
    --output /tmp/Result_b80_run$R \
    --manifest /tmp/manifest_pilot.json
done

# 3. clean up the OpenAI uploads (OpenAI bills for stored files)
pxgpt cleanup-files --manifest /tmp/manifest_pilot.json
```

Separate `--output` directories are required: a second run pointed at the first
one would adopt its `_partial/` shards and make no API calls at all.

## Full per-trait results

⚠ marks a trait where both runs of that budget diverge from both budget-40 runs.
The `human` column is intentionally empty — fill it from the manual corrections to
turn this table into an accuracy comparison rather than a consistency one.

| group | trait | scale | A-b40 | O-b40-s | O-b40-b | O-b80-1 | O-b80-2 | O-b320-1 | O-b320-2 | human |
|---|---|---|---|---|---|---|---|---|---|---|
| whole_plant_architecture | plant_growth_habit | nom | `open_spreading` | `open_spreading` | `open_spreading` | `open_spreading` | `open_spreading` | `open_spreading` | `open_spreading` |  |
| whole_plant_architecture | plant_branching_habit | nom | `unbranched_single_axis` | `unbranched_single_axis` | `unbranched_single_axis` | `unbranched_single_axis` | `unbranched_single_axis` | `unbranched_single_axis` | `unbranched_single_axis` |  |
| whole_plant_architecture | plant_height | qty | `10.5` | `10.0` | `9.0` | `10.0` | `10.0` | `10.0` | `10.0` |  |
| whole_plant_architecture | plant_canopy_spread ⚠b80 ⚠b320 | qty | `19.0` | `12.0` | `14.0` | `10.0` | `11.0` | `12.5` | `13.0` |  |
| whole_plant_architecture | plant_true_leaf_number | qty | `5.0` | `8.0` | `8.0` | `8.0` | `8.0` | `8.0` | `8.0` |  |
| whole_plant_architecture | plant_axillary_bud_development | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| whole_plant_architecture | plant_head_formation | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| whole_plant_architecture | leaf_phyllotaxy | nom | `spiral_alternate` | `spiral_alternate` | `spiral_alternate` | `spiral_alternate` | `spiral_alternate` | `spiral_alternate` | `spiral_alternate` |  |
| stem | stem_elongation | ord | `2` | `2` | `2` | `2` | `2` | `2` | `2` |  |
| stem | stem_base_anthocyanin | nom | `present` | `present` | `present` | `present` | `present` | `present` | `present` |  |
| stem | stem_surface_texture | nom | `smooth` | `smooth` | `smooth` | `smooth` | `smooth` | `smooth` | `smooth` |  |
| stem | stem_thickness | ord | `1` | `2` | `2` | `2` | `2` | `2` | `2` |  |
| stem | stem_leaf_scars | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| leaf_blade | leaf_blade_shape | nom | `obovate_spatulate` | `obovate_spatulate` | `obovate_spatulate` | `obovate_spatulate` | `obovate_spatulate` | `obovate_spatulate` | `obovate_spatulate` |  |
| leaf_blade | leaf_blade_apex_shape | nom | `rounded` | `rounded` | `rounded` | `rounded` | `rounded` | `rounded` | `rounded` |  |
| leaf_blade | leaf_blade_base_shape | nom | `tapering_decurrent` | `tapering_decurrent` | `tapering_decurrent` | `tapering_decurrent` | `tapering_decurrent` | `tapering_decurrent` | `tapering_decurrent` |  |
| leaf_blade | leaf_blade_length ⚠b80 ⚠b320 | qty | `10.5` | `5.0` | `5.0` | `7.0` | `6.0` | `8.0` | `8.0` |  |
| leaf_blade | leaf_blade_curvature | nom | `flat` | `flat` | `flat` | `flat` | `flat` | `concave_cupped` | `flat` |  |
| leaf_blade | leaf_blade_green_intensity | ord | `2` | `2` | `2` | `2` | `2` | `2` | `2` |  |
| leaf_blade | leaf_blade_anthocyanin_coverage | ord | `0` | `0` | `0` | `0` | `0` | `0` | `0` |  |
| leaf_blade | leaf_heterophylly_presence ⚠b320 | nom | `absent` | `present` | `present` | `present` | `present` | `absent` | `absent` |  |
| leaf_margin | leaf_margin_type | nom | `wavy_undulate` | `toothed` | `wavy_undulate` | `wavy_undulate` | `toothed` | `wavy_undulate` | `toothed` |  |
| leaf_margin | leaf_margin_anthocyanin | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| leaf_surface | leaf_surface_texture ⚠b80 ⚠b320 | ord | `1` | `1` | `1` | `2` | `2` | `2` | `2` |  |
| leaf_surface | leaf_surface_glaucousness | ord | `1` | `1` | `1` | `1` | `1` | `1` | `1` |  |
| leaf_surface | leaf_surface_pubescence | nom | `glabrous` | `glabrous` | `glabrous` | `glabrous` | `glabrous` | `glabrous` | `glabrous` |  |
| leaf_surface | leaf_abaxial_anthocyanin | nom | `not_assessable` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| venation | leaf_venation_pattern | nom | `pinnate_reticulate` | `pinnate_reticulate` | `pinnate_reticulate` | `pinnate_reticulate` | `pinnate_reticulate` | `pinnate_reticulate` | `pinnate_reticulate` |  |
| venation | leaf_vein_anthocyanin | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| petiole | petiole_thickness | ord | `2` | `2` | `2` | `1` | `2` | `2` | `2` |  |
| petiole | petiole_relative_length ⚠b320 | ord | `1` | `2` | `2` | `2` | `2` | `1` | `1` |  |
| petiole | petiole_anthocyanin | nom | `absent` | `absent` | `absent` | `absent` | `present` | `absent` | `present` |  |
| petiole | petiole_cross_section_shape ⚠b80 | nom | `slender_ungrooved` | `not_assessable` | `flattened_channeled` | `slender_ungrooved` | `slender_ungrooved` | `slender_ungrooved` | `flattened_channeled` |  |
| inflorescence | inflorescence_stage | ord | `0` | `0` | `0` | `0` | `0` | `0` | `0` |  |
| inflorescence | flower_petal_color_hue | nom | `not_assessable` | `not_assessable` | `not_assessable` | `not_assessable` | `not_assessable` | `not_assessable` | `not_assessable` |  |
| inflorescence | inflorescence_curd_formation | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| inflorescence | fruit_silique_presence | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| root_system | root_density | ord | `1` | `1` | `2` | `1` | `1` | `2` | `2` |  |
| root_system | root_color | nom | `white_cream` | `white_cream` | `white_cream` | `white_cream` | `white_cream` | `white_cream` | `white_cream` |  |
| root_system | root_hair_visibility ⚠b320 | nom | `sparse_or_absent` | `sparse_or_absent` | `sparse_or_absent` | `sparse_or_absent` | `sparse_or_absent` | `present` | `present` |  |
| root_system | root_colonization_extent ⚠b320 | ord | `1` | `1` | `1` | `1` | `1` | `2` | `2` |  |
| phenology | plant_developmental_stage | ord | `2` | `2` | `2` | `2` | `2` | `2` | `2` |  |
| phenology | cotyledon_persistence | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| phenology | foliar_senescence | ord | `0` | `0` | `0` | `0` | `0` | `0` | `0` |  |
| foliar_condition | leaf_interveinal_chlorosis | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| foliar_condition | leaf_necrotic_lesions | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| foliar_condition | leaf_variegation | nom | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` | `absent` |  |
| leaf_damage | leaf_damage_type | nom | `none` | `none` | `none` | `none` | `none` | `none` | `none` |  |
| leaf_damage | leaf_damage_extent | ord | `0` | `0` | `0` | `0` | `0` | `0` | `0` |  |
