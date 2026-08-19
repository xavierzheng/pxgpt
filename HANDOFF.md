# Handoff — OpenAI Stage 3 support and the shard-budget benchmarks

Written 2026-08-14 for whoever picks this up next; updated 2026-08-19. Read this
before opening the long documents; it is meant to save you from re-deriving
anything.

---

## What exists now

`pxgpt phenotype-batch-openai` reached feature parity with the Anthropic
`phenotype-batch`: `--shard-dir`, `--master-schema`, `--allow-reshard`,
`--dispatch {batch,sequential}`, `--resume/--no-resume`, and `fetch-results`
handles the OpenAI sharded stage. Both providers share one merge core
(`batch_utils.merge_sharded_results`) so their gap rule and recovery behaviour
cannot drift apart.

Since then (2026-08-19) the local vLLM path got real constrained decoding:
`OpenAICompatProvider` sends `response_format` `json_schema` (strict) instead of
pasting the schema into the system prompt, with no fallback if the backend
refuses — a silent downgrade there would produce output that looks fine and is
completely unconstrained. `pxgpt schema --shard-dir` runs one plant through a
whole shard set as the cheap rehearsal before a full run, printing each shard's
`cached_tokens` so the prefix-cache premise is visible rather than assumed
(measured on `02_mature_v1`/s0019: shard_01 at 0, shards 2-10 at 20096). Both
`schema` and `analyze` take `--image-transport file` for `file://` delivery.
`analyze --effort <level>` turns thinking on for the local backends and still
saves only the final text; Stage 3 stays pinned to thinking off. 146 tests pass.

Then two benchmarks answered "can we cut OpenAI cost by using fewer, bigger
shards?" — `--shard-budget 40` (10 shards, current) vs `80` (4) vs `320` (1).

---

## Conclusions — do not re-derive these

| | verdict |
|---|---|
| **budget 80** | Defensible. **2.43× less input cost**, at the price of +1.88 categorical traits of 45 diverging beyond run-to-run noise (paired t(9)=3.23, **p=0.010**, 10 plants). |
| **budget 320** | **No.** Diverged 10.2 of 49 — as much as changing provider — and cut rationale length by a third. n=1 plant, already disqualifying. |
| **budget 160** | Untested. The unexplored middle. |

Facts worth keeping in mind:

- **Reproducibility of the method: 89.8% raw agreement, Gwet's AC1 0.88** between
  two runs of the production config, over ALL 45 categorical traits (10 plants).
  Ordinal disagreements are ±1 level 89% of the time. Quantitative CVs: leaf count
  3%, height 8%, blade length 9%, canopy spread 28%.
- **Do NOT compute discrimination statistics on `02_mature_v1`.** `s0001`-`s0011`
  are all Chinese kale (芥藍) and were photographed *before* jie lan varieties become
  distinguishable, so near-zero between-plant variance is the correct biological
  answer. Cohen's kappa drives correct "no difference" reporting toward 0, which
  scores the method being right as a defect. The kappa numbers are kept in
  `analysis/reproducibility.py` output as a baseline to recompute on a panel with
  real between-line variation — not as trait quality. A discrimination study needs
  lines that genuinely differ at a stage where the traits are expressed.
- The least *stable* traits (AC1 0.49-0.66) are shape/margin/curvature judgements
  plus one disease call: `leaf_blade_curvature`, `petiole_relative_length`,
  `leaf_margin_type`, `leaf_necrotic_lesions`, `foliar_senescence`,
  `leaf_blade_shape`. Still n=10, so indicative.
- **`TEMPERATURE=0` is untested and is the obvious lever** — every run so far used
  0.5. It is settable on the OpenAI path (effort `none` accepts a temperature) but
  **not on Anthropic `claude-sonnet-5`, which rejects a custom temperature**, so the
  Anthropic path's reproducibility cannot be tuned at all. Worth knowing before
  claiming reproducibility in a paper.
- **Quantitative traits (4 of 49) are exempt.** They are eyeballed against a nearby
  ruler / colorchecker from several camera angles; the user **accepts** their
  inconsistency (~60% disagreement *within* one config, ±4.4 cm on
  `plant_canopy_spread`). They carry **no budget signal**. Never pool them into an
  agreement rate — doing so once produced a false "systematically shifted trait"
  finding that had to be retracted.
- **Sharding is an *Anthropic* constraint.** `gpt-5.6-luna` accepts all 49 traits in
  a single shard (159 props / 165 enum values / depth 4 vs caps of 5,000/1,000/10).
  But `pxgpt shard-schema` is still mandatory: it **injects `not_assessable`** into
  every nominal/ordinal enum (45 of 45 traits in this master; none list it
  themselves). Never hand-write a Stage 3 schema to bypass it.
- **OpenAI prompt caching barely helps a sharded run.** Across one plant's shards
  only the ~1 k-token system prompt is reused, not the ~30 k of images, because the
  per-shard schema breaks the cacheable prefix. An *identical* re-sent request hits
  ~31 k. So budget a fresh run at full input price per shard; the cache pays off
  only on resumes and gap recovery.
- **`--no-files-api` does not scale with shards.** 1 plant × 15 images × 10 shards is
  a 209 MB batch input file against OpenAI's 200 MB cap; the same run is 94 KB via
  the Files API. There is a pre-encode estimate and a hard fail above 190 MB.

---

## Hard rules

1. **Never modify `02_mature_v1/shard_master_schema/`** — frozen, `chmod -w`,
   sha256-recorded, a manual evaluation is running against it. Same for
   `02_mature_v1/Result_Stage3/`. Verify with sha256 after anything risky.
2. **Keep `--shard-budget 40` for anything compared against the human evaluation.**
   Re-sharding makes results incomparable regardless of quality.
3. **Always finish an OpenAI run with `pxgpt cleanup-files`** — the Files API is
   billed for storage. Verify with `client.files.list()` returning 0, not just by
   trusting the command.
4. **Run `json-to-table` only on a complete result set.** Recover missing shards
   first via `--dispatch sequential` to the same `--output` (it reads `_partial/`).
   A `.gaps.json` present means the run is not finished — `_partial/` exists
   precisely so broken inference is repaired rather than tabulated around.
5. **No drive-by changes.** This user scopes work tightly; fix what was asked, then
   report anything else you noticed.

---

## Where things are

| | |
|---|---|
| narrative write-up, both experiments | `experiment_2026-08-14_shard_budget_openai.md` |
| **data + re-runnable analysis** | `../02_mature_v1/Result_openai_shard_budget_pilot/LAB_NOTEBOOK.md` |
| dispatch / cost / caching guide | `dispatch_batch_vs_sequential.md` → *OpenAI* |
| CLI reference | `user_manual.md` → *phenotype-batch-openai → Sharded mode* |

In the pilot directory: raw records for all 16 runs, batch checkpoints and logs,
the generated shard sets, and **`analysis/compare_configs.py`** — re-derives every
statistic from the archived JSON with **no API calls and no cost**. There is also
`tables/all_configs_long.csv` (1,960 rows) built to join against the manual
annotations; the recipe is in LAB_NOTEBOOK §5.

---

## What NOT to spend money on again

- **Don't re-run inference to redo statistics.** The pilot directory's
  `analysis/compare_configs.py` does it offline.
- **Don't re-measure OpenAI API behaviour.** The probes that established the caching
  behaviour, the strict-mode size limits (`"exceeds limit of"`), and which schemas
  are accepted are archived in `provenance/openai_api_probes/`.
- One earlier mistake worth not repeating: a full CLI run was spent to test whether
  `minLength` is rejected by strict mode. It is **accepted**. Probe a single schema
  before spending a whole run.

---

## Next study: discrimination, on `03_mature_v2` — UNBLOCKED

`02_mature_v1` cannot answer whether the method separates lines (all Chinese kale,
photographed too early). `03_mature_v2` is the intended dataset for that: a
**later growth stage**, where variety differences are expressed.

### ✅ As of 2026-08-19 the data is correct — both gates passed

The earlier blocker (the user had scp'd a byte-identical copy of
`02_mature_v1/images`, and `master_schema.json` was absent) is resolved. Both
checks the previous handoff demanded were run, not assumed:

1. **Images are their own dataset.** `03_mature_v2/images` now holds **277 lines /
   5208 files** dated `2025-04-01`, against v1's 142 / 2784 dated `2025-01-26`.
   The two trees no longer share a filename set.
2. **The master reproduces the frozen shard set byte-identically.**
   `master_schema.json` (a symlink to `master_schema_opus4-8_v2.json`) re-sharded
   at `--shard-budget 40` gives all **9 of 9** `shard_*.schema.json` files
   `cmp`-identical to the frozen ones. This is the same check that validated v1.

The shard set also loads clean end to end: 9 shards, 50 traits / 13 groups, and
`master_index_from_manifest()` agrees with `load_master_index()` field for field.

`03_mature_v2/images/` and `03_mature_v2/shard_master_schema/` are `dr-xr-x---`.
Treat them exactly like the v1 frozen set: **never write there.**

### Re-running the gates

Only needed if the files are re-scp'd again. Both take seconds:

```bash
cd /home/xavier/project/pxgpt
# 1. are the images actually the later timepoint, not the 02 copies?
diff <(cd 02_mature_v1/images && find . -type f -printf '%p %s\n' | sort) \
     <(cd 03_mature_v2/images && find . -type f -printf '%p %s\n' | sort) >/dev/null \
  && echo "WRONG IMAGES — stop" || echo "trees differ, proceed to check 2"

# 2. does the master reproduce the frozen 03 shard set byte-identically?
pxgpt shard-schema --master 03_mature_v2/master_schema.json \
    --shard-dir /tmp/v2check --shard-budget 40
for f in 03_mature_v2/shard_master_schema/shard_*.schema.json; do
  cmp -s "$f" "/tmp/v2check/$(basename $f)" && echo "$(basename $f) OK" \
    || echo "$(basename $f) DIFFERS -> wrong master, stop"
done
```

### Reaching the images from the local vLLM server

`file://` transport is gated by `MEDIA_ROOT` in `ops/local-vllm/.env`, which
`up.sh` uses **twice**: as the bind mount (`$MEDIA_ROOT:$MEDIA_ROOT:ro`) and as
`--allowed-local-media-path`. Both are needed, which is why one variable feeds
both. pxGPT itself never reads it.

It is now `/home/xavier/project/pxgpt` — the project root, not one dataset's
images — so **both** `02_mature_v1` and `03_mature_v2` resolve without a restart.
It is a prefix, baked into the running container; changing it means `down.sh` +
`up.sh`. Two distinct failures tell you which gate bit: **400** "must be a subpath
of `--allowed-local-media-path`" means outside the tree, **500** "No such file or
directory" means inside the tree but absent.

### What does NOT carry over from `02_mature_v1`

The v2 schema is a redesign, not a revision: **only 4 trait names are shared** with
v1. 50 traits / 13 groups / **7 quantitative** (v1: 49 / 12 / 4), with renamed and
new traits (`leaf_count`, `canopy_spread`, `bolting_status`, `storage_organ`,
`growing_medium`, …). So:

- every stability number in this handoff (AC1 0.88, the per-trait AC1 ranking, the
  4-trait quantitative split) is **specific to the v1 schema** and must be
  re-measured on v2 before it means anything there;
- `analysis/compare_configs.py` and `reproducibility.py` take the master path as
  input and will work unchanged — but the quantitative/categorical split becomes
  43 + 7, not 45 + 4.

`03_mature_v2` has never been run **on this machine**: no `file_manifest.json`,
no `openai_file_manifest.json`, no `Result_Stage3/` — so there is nothing here to
recover or reconcile, and a first local run starts clean.

`03_mature_v2/step_04_phenotyping.sh` is **a reference copy, not a runnable
script here.** The Anthropic `describe-batch` / `phenotype-batch` for v2 were
already run on a different HPC; the file was scp'd over only to record which
system prompt and shard schema that run used. It will not execute here (it points
at a `file_manifest.json` that does not exist on this machine) and it is not
meant to — do not "fix" it, and do not re-run Stage 3 through Anthropic to make
it work. That would re-buy results the user already owns.

### Study design, once the data is right

Two runs per configuration on ~20–30 lines gives both the v2 stability baseline and
the discrimination estimate from the same data: Cohen's kappa becomes interpretable
because between-line variance is real. Worth pairing with `TEMPERATURE=0` vs `0.5`,
since that lever is still untested and is the cheapest way to raise reproducibility
on the OpenAI path.

---

## Open questions

1. **Accuracy.** Everything measured so far is model-vs-model *consistency*. Which
   configuration is closer to truth needs the manual annotations (LAB_NOTEBOOK §5).
   The user intends to do this comparison themselves.
2. **Two runs instead of one?** The noise floor exceeds the whole budget effect, so
   replicate-and-vote would buy more per-trait reliability than any budget change.
   Two budget-80 runs cost less than one budget-40 run ($0.072 vs $0.084 per plant);
   three cost ~30% more.
3. **Reproducibility properly measured** would need ~20–30 plants x 3 runs, at
   `TEMPERATURE` 0 and 0.5, and ideally a comparison against human inter-rater
   agreement on the same plants — human phenotypers disagree too, and that is the
   benchmark that matters, not 100%.
4. Budget 160 unscored; budget 320 only n=1; reasoning-on (`STAGE3_EFFORT` set)
   entirely untested — every benchmark run had reasoning off.
5. Deferred by scope, not by judgement: `--dispatch batch` only checks `_partial/`
   provenance at fetch time (both providers), and `phenotype-batch-openai` has no
   manifest-only mode (would need `group_by_line()` on `OpenAIFilesManager`).
