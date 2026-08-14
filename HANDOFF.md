# Handoff — OpenAI Stage 3 support and the shard-budget benchmarks

Written 2026-08-14 for whoever picks this up next. Read this before opening the
long documents; it is meant to save you from re-deriving anything.

---

## What exists now

`pxgpt phenotype-batch-openai` reached feature parity with the Anthropic
`phenotype-batch`: `--shard-dir`, `--master-schema`, `--allow-reshard`,
`--dispatch {batch,sequential}`, `--resume/--no-resume`, and `fetch-results`
handles the OpenAI sharded stage. Both providers share one merge core
(`batch_utils.merge_sharded_results`) so their gap rule and recovery behaviour
cannot drift apart. 93 tests pass.

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

- **"Reproducibility" is two questions; do not answer them with one number.**
  *Stability* (run it twice, same output?) counts EVERY trait: the production config
  scores **89.8% raw agreement, Gwet's AC1 0.88** over all 45 categorical traits.
  *Discrimination* (can a trait separate genotypes?) needs Cohen's kappa on the
  varying traits only: **0.59 mean / 0.64 median**, with 23 of 37 above 0.6 and a
  tail of 8 at kappa <= 0.20 — but those 8 have AC1 0.79-0.90, so they are stable
  yet non-discriminating in this sample, not unstable. Kappa alone is a trap here
  (kappa paradox on skewed marginals). n=10 cannot identify *which* traits are
  weak: the b40 and b80 worst-8 lists overlap on only 3. Script and numbers:
  `analysis/reproducibility.py` in the pilot dir.
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
