# Local vLLM on DGX Spark — `unsloth/gemma-4-26B-A4B-it-NVFP4`

Everything below was measured on this machine, not copied from a playbook. There
is no vendor-published recipe for this checkpoint on GB10: NVIDIA's
`dgx-spark-playbooks` support matrix lists Gemma 4 26B A4B only as BF16 Base (the
31B is the one with an NVFP4 entry), and vLLM's own DGX Spark blog post is a
Nemotron-3-Super-120B recipe. Half of this ticket was finding out which image and
flags actually work.

- **Host**: `spark-1306`, NVIDIA GB10, sm_121 (compute cap 12.1), aarch64,
  121 GiB unified memory, Docker 29.1.3
- **Date measured**: 2026-08-12

> This file is the **evidence**. To actually run the server, follow
> [README_vllm.md](README_vllm.md) — it carries the quick start, the `.env`
> reference, request examples and troubleshooting, and cites the numbers below.

## Pinned configuration

| Setting | Value |
|---|---|
| Image | `nvcr.io/nvidia/vllm@sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268` (tag `26.07-py3`, vLLM `0.24.0+092c4842.dev`) |
| `MODEL_REPO` | `unsloth/gemma-4-26B-A4B-it-NVFP4` |
| `MODEL_REVISION` | `20df0542b1a86ce19f495ac2eca2c7c12bce82f9` |
| Weights on disk | 16.9 GB (single `model.safetensors`) |
| `MAX_MODEL_LEN` | 65536 |
| `MAX_NUM_SEQS` | 16 (see Concurrency) |
| `GPU_MEM_UTIL` | 0.80 |
| **Thinking** | **off** (`enable_thinking: false` per request) |
| **Visual token budget** | **1120 tokens/image** (`--mm-processor-kwargs '{"max_soft_tokens": 1120}'`) |
| **MoE backend** | **pinned** — `MOE_BACKEND=cutlass` → `--moe-backend cutlass`, which the log calls `VLLM_CUTLASS` |
| **Sampling** | **temperature 1.0 / top_p 0.95 / top_k 64**, set on both the server and every request. **No seed.** |

## Image selection

Tried in the ticket's order; the first one that loaded the model and answered
`/health` wins.

### 1. `vllm/vllm-openai:gemma4-cu130` — REJECTED

Ships vLLM `0.19.1.dev6+g6d4a8e6d2`. Registers `Gemma4ForConditionalGeneration`
and correctly auto-detects `compressed-tensors` / NVFP4, then dies loading the
per-expert NVFP4 activation scales:

```
INFO  [nvfp4_utils.py:85] Using NvFp4LinearBackend.FLASHINFER_CUTLASS for NVFP4 GEMM
INFO  [nvfp4.py:256] Using 'VLLM_CUTLASS' NvFp4 MoE backend
ERROR   File ".../vllm/model_executor/models/gemma4.py", line 1359, in load_weights
ERROR KeyError: 'layers.0.experts.0.down_proj.input_global_scale'
```

Its Gemma4 MoE weight loader has no mapping for this checkpoint's
`input_global_scale` tensors. Consistent with the model card asking for
`vllm>=0.25.0`. Not worked around — patching the model executor is out of scope.

### 2. `nvcr.io/nvidia/vllm:26.07-py3` — **SELECTED**

`26.07-py3` was the newest tag in the NGC registry when checked (queried
`nvcr.io/v2/nvidia/vllm/tags/list`, not guessed). vLLM `0.24.0+092c4842.dev`.
Loads and serves correctly. Backend auto-selection:

```
Selected CutlassFP8ScaledMMLinearKernel for CompressedTensorsW8A8Fp8
Using FlashInferCutlassNvFp4LinearKernel for NVFP4 GEMM
Using 'FLASHINFER_CUTLASS' NvFp4 MoE backend out of potential backends:
  ['FLASHINFER_TRTLLM','FLASHINFER_CUTEDSL','FLASHINFER_CUTEDSL_BATCHED',
   'FLASHINFER_CUTLASS','VLLM_CUTLASS','MARLIN','EMULATION']
Using AttentionBackendEnum.TRITON_ATTN backend
```

The model card says "do not use the Marlin backend (around 2x slower); let vLLM
auto-select the NVFP4 kernel" — auto-selection picks a FlashInfer/CUTLASS path,
never Marlin.

> **Superseded.** This section originally read that the MoE choice is "not
> stable across boots" because an earlier boot selected `VLLM_CUTLASS`, and
> concluded that pinning a backend by hand would be wrong. Five consecutive
> boots later failed to reproduce any drift, and the earlier `VLLM_CUTLASS`
> sighting matches the log of the **rejected** candidate 1 quoted above, not
> this image. `--moe-backend` is now pinned to `cutlass` (= `VLLM_CUTLASS`) —
> see "Pinned runtime selections" for the boot-diff table, the reasoning and the
> measured cost.

### 3. `vllm/vllm-openai:cu130-nightly` — works, but not needed

Tested anyway because candidate 2 initially looked unreachable (see the registry
note below). vLLM `0.19.2rc1.dev107+g4eafc7292` — it *does* load this checkpoint
(so the fix for candidate 1's `KeyError` landed between `0.19.1` and `0.19.2rc1`)
and served a text completion. Left unselected because candidate 2 ranks higher
in the ticket's list. Kept here as a fallback: it is a *floating nightly* tag, so
pin it by digest if you ever switch to it.

Comparison on the same text prompt ("Name three plant organs."):
candidate 3 took 6.2 s / 200 completion tokens; candidate 2 took 1.6 s /
28 tokens with a clean `stop`. Candidate 2 is the better pick on merit as well.

### NGC pull needed a credential workaround

`docker pull nvcr.io/nvidia/vllm:...` failed before contacting the registry:

```
error getting credentials - err: exit status 1, out: `exit status 2:
gpg: public key decryption failed: No such file or directory`
```

`~/.docker/config.json` contains `{"credsStore":""}`; with no usable
`credsStore`, Docker falls back to a gpg/`pass`-backed helper that is not
installed on this host (no `docker-credential-*` binary on `PATH`). Docker Hub
pulls are unaffected because they never trigger a credential lookup. NGC's
`nvidia/vllm` repo *is* anonymously pullable, so the fix is to stop Docker
consulting the broken helper:

```bash
mkdir -p /tmp/ngccfg && echo '{"auths":{"nvcr.io":{}}}' > /tmp/ngccfg/config.json
DOCKER_CONFIG=/tmp/ngccfg docker pull nvcr.io/nvidia/vllm:26.07-py3
```

**This is a host configuration bug worth fixing properly** (repair or remove the
credential helper); `pull.sh` does not paper over it, so a fresh run of `pull.sh`
on this host will still fail at candidate 2 until it is fixed.

## Startup timings

| Phase | Cold HF cache page-in | Warm |
|---|---|---|
| Weight load | 108.9 s | 98.1 s |
| Model loading total (16.35 GiB) | 111.4 s | 100.5 s |
| `torch.compile` | — | 41.1 s |
| Engine init (profile + KV cache + warmup) | — | 74.3 s |
| **`up.sh` start → `/health` 200** | **260 s** | **242 s** |
| First-request JIT + cold prefill (10245 tok, `max_tokens=3`) | **11.2 s** | |

KV cache after startup: **74.75 GiB, 5,090,918 tokens**. `--trust-remote-code`
was **not** required — the checkpoint uses the stock `Gemma4ForConditionalGeneration`
architecture.

## Decision (a): thinking off

The chat template settles this. `chat_template.jinja` has
`{%- set enable_thinking = enable_thinking | default(false) -%}`, injects
`<|think|>` at the top of the first system turn only when thinking is on, and
when it is **off** pre-fills the generation prompt with an already-closed thought
block:

```jinja
{%- if not enable_thinking -%}{{- '<|channel>thought\n<channel|>' -}}{%- endif -%}
```

So with thinking off the model's first generated token is already the answer, and
a `response_format` grammar binds cleanly from token 0. There is no conflict to
work around — the premise that the grammar would strangle the reasoning block
only applies if thinking is left on.

Both paths were measured against the **largest** real shard schema (`shard_02`,
4395 bytes, the one most likely to trip a grammar limit) with one real photo:

| | latency | completion tokens | JSON valid | `reasoning` populated |
|---|---|---|---|---|
| **D1** thinking off | **9.4–13.3 s** | 427–553 | yes | n/a |
| **D2** thinking on + `--reasoning-parser gemma4` | **73–85 s** | 3312–3862 | yes | yes (8.7–11.4 k chars) |

**Both pass.** Chose **D1 (thinking off)** because:

1. ~7–9x faster for an equally schema-valid result. Over 2400 requests that is
   the difference between ~9 h and ~3 days.
2. The reasoning text is thrown away. The Stage 3 schema already carries a
   per-trait `rationale` string, so the model states its justification *inside*
   the validated output; `reasoning` would be a second, unvalidated copy that
   nothing consumes.

`--reasoning-parser gemma4` is still passed in `up.sh` even though thinking is
off: it is inert on the D1 path and keeps D2 reproducible against the same
server. The parser name is exactly `gemma4` (registered in
`vllm/reasoning/__init__.py` → `gemma4_engine_reasoning_parser`).

> **Field-name trap for the #1 provider work:** vLLM 0.24 returns the separated
> reasoning as `message.reasoning`, **not** `message.reasoning_content`. Neither
> is modelled by the OpenAI SDK, so both arrive in `model_extra`. `smoke.py`
> checks both names. An early D2 run looked like a failure purely because only
> `reasoning_content` was read.

## Decision (b): 1120 visual tokens per image

The knob is `max_soft_tokens`, **not** `image_seq_length` — from
`vllm/model_executor/models/gemma4_mm.py::_get_max_soft_tokens`, which reads it
either top-level or from `images_kwargs`. It works both as a server flag
(`--mm-processor-kwargs '{"max_soft_tokens": N}'`) and per request
(`mm_processor_kwargs`). Verified against the ladder on one photo:

| `max_soft_tokens` | prompt tokens (1 photo + short text) |
|---|---|
| unset (default) | 284 |
| 70 | 84 |
| 140 | 150 |
| **280** | **284** |
| 560 | 575 |
| 1120 | 1131 |

The checkpoint's own `processor_config.json` sets `image_seq_length` and
`max_soft_tokens` to **280**, so unset == 280. The ticket's worry about
25 x 1120 ≈ 28 k tokens is real at the top of the ladder but affordable: the
largest line in the dataset is 32 photos, so 32 x 1120 + prompt ≈ **37 k**, still
inside `MAX_MODEL_LEN=65536`.

Acceptance E on the full 32-photo set for `s0019` + `shard_02`:

| budget | prompt tokens | latency | JSON valid |
|---|---|---|---|
| 140 | 5 957 | 15.1 s | yes |
| **280** | **10 245** | **20.8 s** | yes |
| 560 | 19 557 | 42.1 s | yes |

**Chose 1120**, the top of the ladder, on accuracy and comparability grounds
rather than latency:

1. The Stage 3 traits are fine-grained — petiole cross-section, colour hue, leaf
   margin — and the model card explicitly recommends *higher* budgets for
   fine-detail work (OCR, document parsing, small text) and lower ones only for
   classification and captioning. This task is the former.
2. 1120 lands closest to what Anthropic Sonnet 5 spends per image, so the local
   and cloud backends see comparable visual information. Since the point of this
   deployment is to compare them, that parity is worth more than throughput.
   Concretely: the dataset's photos are 1568 x 1043 (already sized to Anthropic's
   long-edge limit), and Anthropic's `(w x h) / 750` puts them at ~2180 tokens
   each before its own megapixel downscaling — order 1.5-2 k in practice. 1120 is
   the nearest rung below that; 560 would be 3-4x off.

All budgets tested produce schema-valid JSON, so this is not a validity choice —
it is a detail-versus-cost choice, resolved in favour of detail.

**Cost of the decision — measured at 1120, not extrapolated.** Both budgets were
benchmarked on a fresh container; see the Benchmark section for the raw runs.

| metric (32-photo line, `shard_02`, thinking off) | 280 | **1120** |
|---|---|---|
| prompt tokens | 10 245 | **37 349** |
| cold prefill TTFT (26-photo probe) | 7.83 s | **58.17 s** |
| cold request, total | 24.4 s | **72.7 s** |
| prefix-cached request, total | 12.2 s | **14.6 s** |
| decode throughput | 44.2 tok/s | **40.8 tok/s** |
| 2400-request run, measured per-shard | 9.0 h* | **12.0 h** |

The shape of the cost is the interesting part, and it is *not* the 4x that the
token count suggests:

- **Cold prefill gets much worse — 7.4x.** Effective cold prefill throughput
  drops from ~1 110 tok/s at 280 to ~528 tok/s at 1120, because the vision tower
  does far more work per image: patches scale as
  `max_soft_tokens x pooling_kernel_size^2`, i.e. 2 520 patches per image at 280
  against 10 080 at 1120. Much of that 58 s is the encoder, not LLM prefill.
- **Cached requests barely move — 12.2 s to 14.6 s.** Once the image prefix is
  cached, decode dominates, and decode only slows ~8 %.
- **So the end-to-end cost rises ~33 %, not 4x**, because pxGPT sends 9 shard
  requests per plant off one shared image prefix: only the first pays the cold
  prefill, the other eight are cache hits. Verified directly by sending all 9
  shards of one plant and reading `/metrics` per request — see "Per-shard
  measurement of the real workload". (*the 9.0 h at 280 is the modelled figure;
  only 1120 was measured shard-by-shard.)

An earlier note in this file extrapolated "~80-85 s per full-line request" from
the 280 → 560 step. That number is about right *for a cold request* (measured
72.7 s at 26 photos, 84.8 s at 32) but it is the wrong figure to plan with, since
8 of every 9 requests are cache hits. The measured figure is **12.0 h**, not
days.

**This value is now pinned and must not drift** — it sets the visual information
available to the model, so changing it invalidates comparisons against the
Anthropic and OpenAI backends.

## Acceptance results

`smoke.py` — all green, exit 0. Re-run at the pinned budget of **1120** (182 s);
the 280 column is the original run, kept for comparison. Check H was added later
and re-run against the fully pinned configuration (186 s, all of A–H green).

| | check | result @1120 | @280 |
|---|---|---|---|
| A | `/v1/models` `data[0].id` == `gemma4-26b-a4b-nvfp4` | PASS | PASS |
| B | text-only completion non-empty | PASS | PASS |
| C | one real photo via `file:///abs/path.jpg` | PASS | PASS, 282 prompt tokens |
| D1 | thinking off + strict `json_schema` on real shard | PASS | PASS, 13.0 s |
| D2 | thinking on + `gemma4` parser, same schema | PASS, 82.5 s | PASS, 73.0 s |
| E | all 32 photos of `s0019` + schema | PASS, **37 349** ≤ 65 536, 84.8 s | PASS, 10 245 ≤ 65 536 |
| F | `up.sh` twice → one container; `down.sh` then `up.sh` → back up | PASS | PASS |
| G | `bench.sh` reports the five numbers | PASS | PASS |
| H | same plant + shard twice at temperature 0.7 → outputs differ | PASS (426 vs 528 completion tokens) | — |

C deliberately asserts only that the pipeline works (HTTP 200, non-empty
content), not that the model sees correctly. For the record it did describe the
photo accurately ("a small green kale plant ... next to a vertical wooden ruler
for scale").

## Benchmark (`bench.sh`)

One real plant line's full photo set + one real shard schema → structured JSON.
Not `vllm bench serve`: its random dataset is short-prompt/long-output, the
inverse of this workload. `s0019`, 32 photos, `shard_02`, thinking off, fresh
container each time.

At the pinned **budget 1120**:

```
warm-up (max_tokens=3):  77.8 s   [JIT codegen + cold prefill of 37349 tokens]

run 1: prompt=37349 completion=455 ttft=3.37s total=14.5s decode=40.9 tok/s
run 2: prompt=37349 completion=619 ttft=0.21s total=15.4s decode=40.8 tok/s
run 3: prompt=37349 completion=588 ttft=0.21s total=14.6s decode=40.8 tok/s

cold-prefill probe (s0044, 26 photos, unseen): prompt=30695 ttft=58.17s total=72.7s
```

The original **budget 280** run, for comparison:

```
warm-up (max_tokens=3):  11.2 s   [JIT codegen + cold prefill of 10245 tokens]

run 1: prompt=10245 completion=453 ttft=0.92s total=11.1s decode=44.2 tok/s
run 2: prompt=10245 completion=552 ttft=0.18s total=12.6s decode=44.3 tok/s
run 3: prompt=10245 completion=530 ttft=0.23s total=12.2s decode=44.2 tok/s

cold-prefill probe (s0044, 26 photos, unseen): prompt=8673 ttft=7.83s total=24.4s
```

| metric (median of 3) | **@1120 (pinned)** | @280 |
|---|---|---|
| prompt tokens (incl. images) | **37 349** | 10 245 |
| TTFT (prefix-cache hit) | **0.21 s** | 0.23 s |
| decode throughput | **40.8 tok/s** | 44.2 tok/s |
| total latency | **14.6 s** | 12.2 s |
| completion tokens | **588** | 530 |
| cold-prefill probe (`s0044`, 26 photos) | TTFT **58.17 s**, total **72.7 s** | TTFT 7.83 s, total 24.4 s |

### Per-shard measurement of the real workload

The projection below models the run as "1 cold + 8 cached" per plant. That model
was then **verified directly**: all 9 shards of one plant (`s0019`, 32 photos, the
largest line in the dataset) sent consecutively on a fresh container, with
`/metrics` read around each request. No averaging.

| shard | prompt tok | TTFT | total | completion | prefix cache hit | mm cache |
|---|---|---|---|---|---|---|
| `shard_01` | 37 259 | **76.20 s** | **86.17 s** | 408 | 0 / 37 259 — **0.0 %** | 0/32 |
| `shard_02` | 37 349 | 2.04 s | 14.73 s | 517 | 36 448 / 37 349 — 97.6 % | 32/32 |
| `shard_03` | 37 012 | 0.73 s | 8.09 s | 298 | 36 448 / 37 012 — 98.5 % | 32/32 |
| `shard_04` | 37 322 | 1.04 s | 12.56 s | 470 | 36 448 / 37 322 — 97.7 % | 32/32 |
| `shard_05` | 36 904 | 0.63 s | 5.39 s | 195 | 36 448 / 36 904 — 98.8 % | 32/32 |
| `shard_06` | 37 249 | 0.92 s | 8.79 s | 321 | 36 448 / 37 249 — 97.8 % | 32/32 |
| `shard_07` | 37 194 | 0.89 s | 9.17 s | 338 | 36 448 / 37 194 — 98.0 % | 32/32 |
| `shard_08` | 36 729 | 0.47 s | 4.39 s | 161 | 36 448 / 36 729 — 99.2 % | 32/32 |
| `shard_09` | 37 220 | 0.92 s | 12.92 s | 488 | 36 448 / 37 220 — 97.9 % | 32/32 |

**One plant = 162.2 s** (2.7 min) for all 9 shards.

What the numbers confirm:

- **The cached prefix is 36 448 tokens** — the system block plus all 32 images —
  reused identically by shards 2-9. The 0.8-2.4 % miss is each shard's own prompt
  text at the tail, which is exactly the part that should differ.
- **`mm_cache` goes 0/32 → 32/32.** The vision encoder's output for all 32 images
  is reused, which is why TTFT collapses from 76.20 s to 0.47-2.04 s. The encoder,
  not LLM prefill, is the bulk of the cold cost.
- **Cached shards are decode-bound.** Their latency tracks completion tokens
  almost exactly (161 tok → 4.39 s, 517 tok → 14.73 s) at ~40 tok/s.
- The spread across shards 2-9 is 4.39-14.73 s, so quoting a single mean would
  hide a 3.4x range. Median 8.98 s, mean 9.5 s.

**Measured 2400-request estimate: 267 plants x 162.2 s = 12.0 h**, serial. This
supersedes the 14.0 h modelled figure below, which was slightly pessimistic
because it applied `bench.sh`'s 14.6 s median to all eight cached shards; the real
cached shards average 9.5 s. It is also mildly conservative in the other
direction, since `s0019` is the largest line and most plants carry 24-30 photos.

### 2400-request projection (modelled)

Prefix caching is on by default (V1), and pxGPT sends 9 shard requests per plant
that all share the same image prefix — so per plant one request pays a cold
prefill and eight hit the cache. That mix, not the raw median, is the right basis:

| scenario | **@1120 (pinned)** | @280 |
|---|---|---|
| all-cold upper bound | 48.5 h (72.7 s x 2400) | 16.3 h (24.4 s x 2400) |
| all-cached lower bound | 9.7 h (14.6 s x 2400) | 8.1 h (12.2 s x 2400) |
| **realistic (267 plants x [1 cold + 8 cached])** | **14.0 h** | **9.0 h** |

Serial only; concurrency above 1 was not measured. The 280 figures were
cross-checked independently on three other lines (`s0016`/`s0017`/`s0014`): cold
TTFT median 9.26 s / total 19.5 s, cached TTFT 0.24 s / total 12.4 s → ~8.8 h,
agreeing with the 9.0 h above.

The realistic row assumes all 9 shards of a plant are sent while that plant's
image prefix is still resident in the KV cache. Dispatching shard-major (all
plants for shard 1, then all plants for shard 2) would pay the cold prefill
**every time** and land near the 48.5 h upper bound — a 3.5x difference from
ordering alone. `build_sharded_requests()` already emits plant-major
(`for line_id ... for s in shards`), and `--dispatch sequential` preserves that
order; a future vLLM Stage 3 path must keep it.

## Concurrency

`MAX_NUM_SEQS` was raised from 2 to 16. It is **free for KV cache**: the pool is
preallocated from `gpu-memory-utilization`, not from the sequence count —
5 074 332 tokens at 16 against 5 081 419 at 2. At ~37 k tokens per request even
16 concurrent requests need only ~600 k tokens.

> **Narrowed.** This paragraph used to say the setting was free full stop, on
> the evidence that the KV token count barely moved. That evidence cannot
> support the broader claim: the KV pool is preallocated, so its size is
> **constitutionally incapable** of reflecting what concurrency costs — the
> vision encoder's activations and the multimodal processor cache are not in it,
> and the failure mode this deployment actually risks is exhaustion of the
> unified host pool, not a GPU-side OOM.
>
> What is now measured: **free when the multimodal cache is fully hit** — a
> plant's warm shards run 8-wide at no memory cost, because no new images are
> processed. **Not free for cold concurrency** — running several plants' cold
> prefills at once does move host memory, and at three plants in flight
> `MemAvailable` fell to 7.37 GiB, past the 8 GiB stop line. See "Pipeline
> depth, memory, and runaway generation" for the numbers; the recommended depth
> is **2 plants**, not more.

### Concurrency on an already-cached prefix

`s0019` warmed with `shard_01`, then shards 2-9 replayed at increasing client
concurrency against that same cached prefix. Two repeats each, no averaging:

| conc | wall (rep 1 / rep 2) | completion tok | aggregate tok/s | speedup |
|---|---|---|---|---|
| 1 | 74.8 s / 74.9 s | 2799 / 2988 | 37.4 / 39.9 | 1.0x |
| 2 | 42.1 s / 43.2 s | 2803 / 2957 | 66.7 / 68.4 | 1.75x |
| 4 | 27.0 s / 29.7 s | 2788 / 2809 | 103.3 / 94.7 | 2.6x |
| 8 | **21.4 s / 21.9 s** | 2912 / 2922 | **136.0 / 133.5** | **3.4x** |

Scaling is sub-linear but strong. Individual requests get slower (`shard_02`
14.9 s at conc 1 to 21.4 s at conc 8) — the usual batching trade — but wall clock
for the set falls 3.4x, which is what a 2400-request run cares about.

### Whole-plant patterns, measured cold

Four dispatch patterns on **cold** 25-26 photo plants, fresh container each time
(prefix cache verified empty at 0 queries / 0 hits before starting):

| pattern | plant | per-plant wall | prefix hit | 2400 requests | vs serial |
|---|---|---|---|---|---|
| **A** all 9 serial | `s0044` (26 ph) | 161.6 s | 86.9 % | 12.0 h | 1.0x |
| **C** all 9 concurrent from cold | `s0011` (26 ph) | **402.1 s** | **9.5 %** | 29.8 h | **0.40x** |
| **B** cold `shard_01`, then 2-9 @conc 8 | `s0035` (26 ph) | 101.6 s | 97.8 % | 7.5 h | 1.59x |
| **D** 2 plants, colds overlapped, warm @conc 16 | `s0022`+`s0013` (25 ph) | **83.6 s** | — | **6.2 h** | **1.93x** |

Per-request latencies:

```
A  77.1  18.8 9.4 12.8 7.6 9.4 10.7 5.8 10.0      (cold first, then warm)
C  397.8 400.1 386.6 402.0 386.0 387.7 390.7 385.0 395.6
B  cold 72.5 | 29.2 18.2 24.7 13.3 23.2 18.6 13.0 21.6
```

Pattern A independently reproduces the earlier per-shard measurement — 161.6 s on
`s0044` against 162.2 s on `s0019` — so the 12.0 h serial baseline is confirmed on
a second plant.

**Pattern C is the trap.** Firing all 9 shards of a cold plant at once collapses
the prefix-cache hit rate to **9.5 %**: none of them finds the prefix because none
has finished writing it, so all nine independently prefill the same ~31 k tokens
and re-run the vision encoder over the same 26 images nine times, while competing
for the GPU. It is **2.5x slower than plain serial** and would turn a 12 h run
into 30 h. The shared prefix must be established by **one** request before the
rest are released.

**Pattern D is the best measured.** Overlapping two plants' cold prefills, then
running both warm sets at conc 16, reaches 83.6 s per plant — 1.93x better than
serial — because one plant's cold prefill overlaps the other's decode.

> This row was n = 1 (two plants, one run) with no prefix-hit or memory numbers
> — the `—` in the table. It has since been re-measured properly at depths 1, 2
> and 3 with per-plant prefix hits and a memory trace: **75.1 s per plant at two
> in flight (5.57 h)**, both plants hitting 97.5 %. Deeper pipelining was also
> tested and is **not** recommended: three in flight buys 5 % and spends the
> whole memory margin. See "Pipeline depth, memory, and runaway generation".

**Recommendation for a vLLM Stage 3 path:** per plant, send `shard_01` alone,
wait for it, then release shards 2-9 at concurrency 8; overlap the next plant's
cold prefill with the current plant's warm set, **two plants in flight, not
more**. Never fan out a cold plant.

## Pinned runtime selections

Pinning the image by digest pins the *software*. It does not pin what that
software chooses at runtime, and this deployment's third goal is consistency:
the same plant will be inferred repeatedly at temperature > 0 to see how much
the model's reading of it moves. Anything else that moves between runs
contaminates that measurement, so this section establishes what actually moves.

### What floats between boots, measured

Five consecutive `./down.sh && ./up.sh` cycles on the pinned digest, with
`--moe-backend` still unset, every auto-selection line diffed:

| selection | log line it comes from | 5 boots | verdict |
|---|---|---|---|
| NVFP4 MoE backend | `nvfp4.py` `Using 'X' NvFp4 MoE backend out of potential backends: [...]` | `FLASHINFER_CUTLASS` x5 | **stable** |
| NVFP4 GEMM (linear) kernel | `__init__.py:937` `Using X for NVFP4 GEMM` | `FlashInferCutlassNvFp4LinearKernel` x5 | **stable** |
| FP8 scaled-mm kernel | `__init__.py:594` `Selected X for CompressedTensorsW8A8Fp8` | `CutlassFP8ScaledMMLinearKernel` x5 | **stable** |
| attention backend | `cuda.py:420` `Using AttentionBackendEnum.X backend` | `TRITON_ATTN` x5 | **stable, and forced** |
| MoE prepare/finalize | `nvfp4.py:482` `Using X` | `MoEPrepareAndFinalizeNoDPEPModular` x5 | **stable** |
| top-p/top-k sampler | `topk_topp_sampler.py:55` `Using X for top-p & top-k sampling` | `FlashInfer` x5 | **stable** |
| KV cache size | `kv_cache_utils.py` `GPU KV cache size: N tokens` | 5 052 777 / 5 065 352 / 5 070 923 / 5 071 006 / 5 089 983 | **floats, +-0.4 %** |

Three findings worth stating plainly:

1. **The MoE drift this ticket was written to stop did not reproduce.** Five
   boots, same backend every time. The selection is not a race or a
   timing-based probe either: `select_nvfp4_moe_backend()` walks a fixed
   priority list (`FLASHINFER_TRTLLM`, `FLASHINFER_CUTEDSL`,
   `FLASHINFER_CUTEDSL_BATCHED`, `FLASHINFER_CUTLASS`, `VLLM_CUTLASS`,
   `MARLIN`, `EMULATION`) and returns the first entry whose
   `is_supported_config()` passes. Same inputs, same answer.

   The earlier `VLLM_CUTLASS` sighting is almost certainly the **rejected**
   candidate-1 image, not this one: its log is quoted verbatim under "Image
   selection" above and reads `[nvfp4.py:256] Using 'VLLM_CUTLASS' NvFp4 MoE
   backend`. That is vLLM `0.19.1`, a different priority list. Two images, one
   note, read as one image drifting.

   Pinning is still right — the choice is now stated instead of inherited from
   a list any image bump can reorder — but it is insurance, not a fix for an
   observed fault.

2. **The attention backend cannot float on this model.** Before any
   auto-selection runs, `config.py:99` prints `Gemma4 model has heterogeneous
   head dimensions (head_dim=256, global_head_dim=512). FA4 not available,
   forcing TRITON_ATTN backend.` It is forced by the architecture, not chosen.

3. **The one thing that does move is the KV cache size**, by up to 37 k tokens
   (0.4 %) between boots. That is memory-profiling noise —
   `--gpu-memory-utilization` is applied to whatever is free at profiling time —
   and it is harmless here: at ~37 k prompt tokens per request even the smallest
   observed pool holds ~135 full requests. It is recorded because it is the only
   measured boot-to-boot variance, so if a run ever behaves oddly this is the
   number to compare first.

### What is still unpinned, and why it is only recorded

`enable_flashinfer_autotune=True` runs a **timing-based** tactic search at every
startup. It profiles tactics and keeps the fastest, and the winner is never
logged — so it cannot be diffed across boots the way the table above was, and
there is no CLI flag that pins the outcome. Per the ticket, recorded rather than
fought with environment variables or patches.

Pinning the MoE backend does cut it down measurably. Counting autotuner blocks
in the startup logs:

| boot | tuned |
|---|---|
| unpinned (`FLASHINFER_CUTLASS` MoE) | `fp4_gemm`, `trtllm::fused_moe::gemm1`, `trtllm::fused_moe::gemm2` |
| pinned (`cutlass` → `VLLM_CUTLASS` MoE) | `fp4_gemm` only |

So the pin removes the FlashInfer MoE path and with it two of the three
autotuned kernels. What remains is `fp4_gemm`, belonging to the NVFP4 *linear*
kernel, which is still `FlashInferCutlassNvFp4LinearKernel`. If that ever needs
pinning too the flag is `--linear-backend` (same lowercase spelling rule); it is
deliberately left on `auto` here because it never floated in the log.

### The `--moe-backend` spelling trap

The CLI and the log use **different names for the same kernel**. Passing the
name from the log fails outright:

```
vllm serve: error: argument --moe-backend: invalid choice: 'vllm_cutlass'
(choose from 'aiter', 'auto', 'cutlass', 'deep_gemm', 'deep_gemm_mega_moe',
 'emulation', 'flashinfer_b12x', 'flashinfer_cutedsl', 'flashinfer_cutlass',
 'flashinfer_trtllm', 'flydsl', 'humming', 'marlin', 'triton', 'triton_unfused')
```

The CLI takes lowercase `MoEBackend` values; the log prints `NvFp4MoeBackend`
values. `map_nvfp4_backend()` in
`vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` is the mapping, and it
sends `"cutlass"` → `VLLM_CUTLASS`. So **`MOE_BACKEND=cutlass` in `.env` is what
produces `Using 'VLLM_CUTLASS'` in the log.** Verified: three consecutive
`down.sh && up.sh` cycles, `VLLM_CUTLASS` all three times.

### Cost of the pin

Auto-selection currently picks `FLASHINFER_CUTLASS`, so pinning to
`VLLM_CUTLASS` *changes* the serving kernel — every measurement elsewhere in
this file predates the pin and was taken on `FLASHINFER_CUTLASS`. Minimum
comparison, one cold + three warm on the same unseen plant (`s0176`, 23 photos,
`shard_02`, 26 774 prompt tokens), fresh container each side, with a JIT warm-up
on a different plant first so the ~11 s of first-request codegen does not land
inside the cold TTFT:

| | cold TTFT | cold total | warm total (3) | warm decode |
|---|---|---|---|---|
| `FLASHINFER_CUTLASS` (auto picks this) | 52.34 s | 65.5 s | 14.9 / 12.8 / 14.6 s | 41.4 / 41.3 / 41.4 tok/s |
| **`VLLM_CUTLASS` (pinned)** | **50.51 s** | **63.6 s** | **13.5 / 15.5 / 13.6 s** | **39.5 / 40.0 / 39.8 tok/s** |
| difference | **-3.5 %** (faster) | -2.9 % | comparable | **-3.6 %** (slower) |

Decode is ~3.6 % slower and cold prefill ~3.5 % faster — both far inside the
20 % limit that would have required stopping, and both close enough to the noise
between repeats (the warm totals overlap) that neither is worth acting on. The
pin is effectively free.

`bench.sh` re-run on a fresh container with everything pinned, against the same
`s0019` / `shard_02` / `s0044` setup as the Benchmark section above, so the
headline figures can be read at the configuration that will actually serve the
run:

| metric (median of 3) | before the pin (`FLASHINFER_CUTLASS`) | after the pin (`VLLM_CUTLASS`) |
|---|---|---|
| prompt tokens | 37 349 | 37 349 |
| TTFT (prefix-cache hit) | 0.21 s | 0.22 s |
| decode throughput | 40.8 tok/s | **39.2 tok/s** (-3.9 %) |
| total latency | 14.6 s | 12.8 s |
| completion tokens | 588 | 462 |
| cold probe (`s0044`, 26 photos) | TTFT 58.17 s, total 72.7 s | TTFT 60.41 s, total 75.4 s |
| serial 2400-request projection | 12.0 h | 13.2 h |

Decode throughput is the only figure that moves consistently, by the same ~4 %
the head-to-head found. Total latency and the projection move mostly because
sampling produced shorter completions this time (462 against 588 tokens) — at
temperature 1.0 the cached shards are decode-bound, so completion length drives
their latency more than the kernel does. Do not read the 12.8 s as a speed-up.

### Sampling parameters were never recorded until now

Nothing was passing sampling parameters, so every measurement in this file up to
this point ran on the checkpoint's own defaults. Those defaults, read from
`generation_config.json` inside the container
(`/root/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-NVFP4/snapshots/20df0542.../`):

```json
{
  "bos_token_id": 2,
  "do_sample": true,
  "eos_token_id": [1, 106, 50],
  "pad_token_id": 0,
  "temperature": 1.0,
  "top_k": 64,
  "top_p": 0.95,
  "transformers_version": "5.13.0"
}
```

**So every latency and token count above was measured at temperature 1.0 / top_p
0.95 / top_k 64 — sampled, not greedy.** vLLM says so at startup, and the line
is worth grepping for after any config change:

```
WARNING [model.py:1477] Default vLLM sampling parameters have been overridden by
the model's `generation_config.json`: `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}`.
```

`TEMPERATURE` / `TOP_P` / `TOP_K` in `.env` now carry exactly these values, so
behaviour is unchanged — the difference is that they are stated. They are
consumed in **two** places from that one source: `up.sh` passes
`--override-generation-config`, and `smoke.py` / `bench.sh` send them on every
request. Server-side alone would keep them out of the request record, so writing
the paper's methods section would mean archaeology; client-side alone would let
any client that forgets them fall back to Google's checkpoint defaults silently.
Sharing one set of variables is what stops the two layers disagreeing.

**No `seed` is sent anywhere** (`grep -rn seed *.sh *.py` finds only comments).
vLLM accepts a per-request seed, and a fixed one would make repeated inference
of the same plant return identical output with zero variance and no error —
which would quietly destroy the consistency study rather than break it. Note the
engine config dump prints `seed=0`: that is this build's default engine seed and
it is *not* a per-request seed. Sequential identical requests still differ,
which acceptance check H asserts on every run.

### `--override-generation-config` merges, it does not replace

Verified in this build rather than assumed, by constructing `ModelConfig` with
different overrides and reading back `get_diff_sampling_param()`:

| override passed | effective sampling defaults |
|---|---|
| `{}` (what every earlier measurement ran on) | `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}` |
| `{"temperature": 0.5}` | `{'temperature': 0.5, 'top_k': 64, 'top_p': 0.95}` |
| `{"temperature": 1.0, "top_p": 0.95, "top_k": 64}` (up.sh) | `{'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}` |
| `{"repetition_penalty": 1.05}` | `{'repetition_penalty': 1.05, 'temperature': 1.0, 'top_k': 64, 'top_p': 0.95}` |

Row 2 settles it: `top_k` and `top_p` survive an override that never mentions
them. The mechanism is `config.update(self.override_generation_config)` in
`ModelConfig.get_diff_sampling_param()` — a dict merge over the loaded
`generation_config.json`.

**Nothing therefore regresses to a vLLM default.** The effective sampling
configuration is byte-identical before and after the flag, so the pin does not
invalidate any earlier measurement. Only six keys are ever forwarded as
server-side defaults (`repetition_penalty`, `temperature`, `top_k`, `top_p`,
`min_p`, `max_new_tokens`); this checkpoint's file sets three of them, and
`up.sh` overrides those same three. `repetition_penalty` was never present and
still is not.

### Departure: `--enable-prompt-tokens-details`

`up.sh` also passes `--enable-prompt-tokens-details`, which is not on the
ticket's flag list. It is a **reporting** flag only — no kernel, no sampling, no
memory — and it makes the server fill in
`usage.prompt_tokens_details.cached_tokens` per response. Without it that field
is `null`, and a per-request prefix-cache hit count is unobtainable: `/metrics`
counters are cumulative and cannot be attributed to a request once requests
overlap. It is added because the pipeline-depth measurement below has to answer
"when two plants are in flight, did **both** warm sets hit the cache, or did one
evict the other" with per-plant numbers.

## Pipeline depth, memory, and runaway generation

Everything above this point measured latency. Nothing measured **memory**, which
is the resource whose exhaustion hard-locks this machine, and the recommended
dispatch pattern (D) was the least examined of the four. This section closes
both gaps and puts a number on runaway generation.

### Two samplers, and how to read them

`sample_mem.sh` (0.2 s) and `sample_metrics.sh` (1 s) write TSV files. Both
intervals are deliberate: host memory moves on a sub-second scale because the
vision encoder's activations do, while `/metrics` is a ~100-series Prometheus
dump that would load the server being measured if scraped five times a second.
Neither uses `watch` — the number that matters is the *lowest point*
`MemAvailable` ever reached, and a repainted terminal cannot be queried
afterwards.

**Metric names were read off this build's live `/metrics`, not from docs**, and
the ticket's warning was justified: the cache-usage series is
`vllm:kv_cache_usage_perc` here, **not** the `vllm:gpu_cache_usage_perc` older
vLLM used. A stale name greps to an empty column rather than an error, so
`sample_metrics.sh` verifies every name against `/metrics` before it starts and
refuses to run if one is missing. Verified names:

```
vllm:num_requests_running      vllm:num_requests_waiting
vllm:num_preemptions_total     vllm:kv_cache_usage_perc
vllm:prefix_cache_queries_total  vllm:prefix_cache_hits_total
vllm:mm_cache_queries_total      vllm:mm_cache_hits_total
```

Reading them:

- **`MemAvailable` is the safety metric.** The GB10's memory is one unified pool
  shared with the OS and the page cache, and exhausting it is the failure mode
  that needs a power cycle.
- **KV cache usage is not.** The pool is preallocated from
  `--gpu-memory-utilization`, so the fraction cannot exceed 1.0 and cannot
  exhaust anything; saturating it produces *preemption*, not a crash.
- **Preemption count is the scheduling-pressure metric.** Monotonic; any
  increase means the scheduler discarded work to free KV blocks.

### Incremental pipeline depth: N = 1, 2, 3

N plants in flight, each dispatched by Pattern D's rule (`shard_01` alone, wait,
then shards 2–9 at concurrency 8). N lanes run concurrently and each lane
processes **two** plants back to back, so every level contains the within-lane
cold/warm overlap as well as the across-lane one. Container restarted between
levels, `vllm:prefix_cache_queries_total` asserted at 0 before each, and plants
the container had never served. All plants 22–23 photos, so the levels are
comparable.

Stop rule fixed in advance: `MemAvailable` low-water below 8 GiB, or any
preemption. **Read the next subsection before treating these floors as the
operating margin** — a restarted container also has an empty multimodal cache,
which flatters every one of them by roughly 2.7 GiB.

| N | plants | wall | per-plant | 267 plants | `MemAvailable` low | margin to 8 GiB | preemptions | per-plant prefix hit |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 178.4 s | 89.2 s | **6.62 h** | 8.74 GiB | +0.74 | 0 | 97.5 %, 97.5 % |
| **2** | 4 | 300.6 s | **75.1 s** | **5.57 h** | **9.04 GiB** | **+1.04** | 0 | 97.5 % x4 |
| 3 | 6 | 426.3 s | 71.0 s | 5.27 h | **7.37 GiB** | **-0.63** | 0 | 97.4–97.5 % x6 |
| 4 | — | **not run** — N = 3 tripped the stop rule | | | | | | |

**N = 3 triggered the stop condition, so N = 4 was not attempted and the
recommendation falls back to N = 2.** Per the ticket this is a pass, not a
failure. The excursion is not one stray sample: 22 of 2 092 samples (1.1 %,
4.4 s of wall clock) sat below 8 GiB, p1 was 7.85 GiB, and the *median* fell
from 10.06 GiB at N = 2 to 8.56 GiB at N = 3. The machine survived and was
healthy afterwards — but a single survival is not evidence about a failure mode
that hard-locks, which is exactly why the low-water mark is reported instead of
"it did not crash".

**Backing off costs almost nothing.** N = 2 → N = 3 buys 5.4 % (5.57 h → 5.27 h)
while spending the entire memory margin. N = 1 → N = 2 buys 16 %. The knee is at
N = 2.

**Both plants' warm sets hit. Neither evicts the other.** This is the question
Pattern D's `—` left open, and it is answered per plant rather than in
aggregate: every plant at every depth reused 97.4–97.5 % of its prompt, i.e.
the whole shared system+images prefix, with the 2.5 % miss being each shard's
own trailing prompt text — exactly the part that should differ. Concretely at
N = 2, all four plants read `207104/212379 = 97.5 %`; at N = 3, all six read
97.4–97.5 %. No plant was starved.

Per-request totals (`shard_01` first, then shards 02–09):

```
N=1  s0167: 65.9  28.7 19.6 24.9 15.6 17.6 22.1 13.0 18.7
     s0162: 58.3  22.9 20.3 25.5 13.3 15.4 20.6 13.0 21.0
N=2  s0147: 113.4 40.9 23.5 35.2 19.9 27.4 27.4 18.0 35.1
     s0146: 113.2 39.6 26.9 40.7 20.9 32.5 29.5 20.5 27.7
     s0141: 109.6 34.1 25.3 36.8 19.2 23.8 29.7 19.8 28.8
     s0046: 109.6 34.3 22.5 32.9 18.3 27.4 30.9 19.7 26.1
N=3  s0018: 181.5 31.6 18.1 99.1 23.1 25.3 25.5 29.2 127.2
     s0010: 163.3 45.3 27.7 47.7 18.3 23.8 38.5 16.2 39.0
     s0195: 167.8 43.9 29.1 39.7 17.7 30.5 35.3 16.9 40.9
     s0188:  86.5 31.0 22.6 28.8 14.4 21.8 21.6 15.3 25.3
     s0183: 162.3 41.7 19.3 38.1 18.3 35.3 21.9 32.1 27.5
     s0178: 157.4 41.8 24.0 37.3 18.7 22.9 24.6 19.8 37.3
```

Individual requests get much slower as N rises — cold `shard_01` goes 58–66 s at
N = 1 to 110–113 s at N = 2 to 87–182 s at N = 3, and the N = 3 tail is ragged
(`s0018` has warm shards at 99 s and 127 s). Throughput still improves because
the machine is never idle. Anything that cares about per-request latency rather
than total wall clock should read this table before choosing a depth.

**What the memory trace actually shows.** The drop is a **step, not a spike**.
At N = 1, `MemAvailable` sits at 12.9 GiB idle, falls to ~9.3 GiB within four
seconds of the first cold prefill and stays flat there for its whole 66 s, rises
to ~10.6 GiB during the decode-bound warm set, and reaches its 8.7 GiB minimum
at the instant one plant's warm tail overlaps the next plant's cold prefill —
the Pattern D handover itself. So the cost is a resident working set plus a
handover overlap, not a sub-second activation peak. The 0.2 s sampling was still
the right choice: it is what establishes that there is no spike.

Note also that N = 2's low-water mark (9.04 GiB) is *higher* than N = 1's
(8.74 GiB). Memory does not scale with the number of plants in flight the way
the ticket's premise assumed — the KV pool is preallocated and the multimodal
cache is capped, so what grows is largely bounded. It is only at N = 3 that the
floor moves decisively.

### The fresh-container floors are optimistic, and this matters more than depth

Every row of the table above was measured on a container restarted moments
earlier, because that is the only way to get an honest *cold* prefix cache. But
a restarted container also has an **empty multimodal processor cache**, and that
cache grows to its 4 GiB cap and stays there. A 267-plant production run spends
essentially all of its time on the far side of that fill.

Measured directly. Idle `MemAvailable`, same server, same settings:

| | idle `MemAvailable` |
|---|---|
| container just booted, mm cache empty | **12.9–13.7 GiB** |
| after ~1 h of serving, mm cache at its 4 GiB cap | **7.76 GiB** |

So N = 2 was re-run **without restarting**, on four plants the container had
never served, with the multimodal cache already saturated — the steady state of
a long run:

| N = 2 | `MemAvailable` low | median | samples < 8 GiB | preemptions | per-plant prefix hit |
|---|---|---|---|---|---|
| fresh container | 9.04 GiB | 10.06 GiB | 0 / 1 482 | 0 | 97.5 % x4 |
| **mm cache saturated** | **6.32 GiB** | **7.71 GiB** | **1 175 / 1 337 (88 %)** | **0** | 97.2 % x4 |

**In steady state, two plants in flight sits below the 8 GiB stop line for most
of the run** — 2.7 GiB lower than the fresh-container measurement suggested. The
run completed cleanly, all 36 requests succeeded, no preemptions, and every
plant still kept 97.2 % of its prefix, but the headroom is not what the table
above implies. (Its 68.1 s per plant is *not* comparable with the 75.1 s above:
those plants carry 21 photos rather than 23, and the system-prompt prefix was
already resident. Read this run for its memory numbers, not its speed.)

**The lever is the cache, not the depth.** What consumes the margin is a
standing 4 GiB allocation, so backing off from two plants to one does not
recover it — one plant in flight would sit on the same saturated cache. The
knobs that do:

- **`--mm-processor-cache-gb`.** At the pinned 1120-token budget a plant of
  22–23 photos costs ~0.69 GB, so two in flight need ~1.4 GB. The default 4 GiB
  is holding roughly four plants that have already finished. Lowering it to
  **2 GiB should return ~2 GiB of headroom** and cost only re-processing of
  plants that are no longer in flight — nothing a live plant needs. Deliberately
  **not changed here**: adding server flags is outside this ticket, and the
  claim above is a prediction from the measured per-plant cost, not a
  measurement. Worth one experiment before the production run.
- `--gpu-memory-utilization`, in the other direction, is the known route to a
  hard lock and should not be touched.

Stated plainly, because it is the number a reader will want: the recommendation
is still **two plants in flight**, but on the understanding that the 8 GiB stop
line is crossed in steady state at *any* depth with the default multimodal cache
size, and that the fix is to shrink that cache rather than to serialise the
pipeline.

### Multimodal processor cache: `--mm-processor-cache-gb`, default 4 GiB

The flag is `--mm-processor-cache-gb` (`float`, **default 4**, `ge=0`), with
`--mm-processor-cache-type` defaulting to `lru`. Neither is set by `up.sh`, so
**4 GiB LRU is what is in force.**

What it holds is the **processor** output — the pre-encoder tensors — not the
vision encoder's result. Measured directly by running this checkpoint's
`AutoProcessor` over a real dataset photo (1568 x 1043):

| visual budget | `pixel_values` | per image | per 32-photo plant | plants per 4 GiB |
|---|---|---|---|---|
| 280 | `(1, 2520, 768)` float32 | 7.78 MB | 0.25 GB | ~17 |
| **1120 (pinned)** | `(1, 10080, 768)` float32 | **31.13 MB** | **1.00 GB** | **~4.3** |

At 22–23 photos — the common case in this dataset — a plant costs ~0.69 GB, so
the cache holds about **6 plants**. That is a real constraint on pipelining, and
it is independent of the memory stop rule: at N = 3, six plants totalled ~4.1 GB
against a 4 GiB budget, so the level finished right at the eviction boundary
(harmlessly — by then the evicted plant was done). Beyond ~4 plants in flight
with 32-photo lines, a plant's images could be evicted *before its own warm set
finishes*, which is a different failure from running out of memory and has a
different fix: raise `--mm-processor-cache-gb`, do not reduce depth.

**This is how to tell the two apart.** LRU eviction shows up as a per-plant
prefix-hit rate collapsing for one plant while `MemAvailable` stays flat;
memory pressure shows up as `MemAvailable` falling with hit rates intact.
Neither happened at N ≤ 3 — every plant kept 97.2–97.5 %.

**But the cache is also what eats the memory margin.** Its 4 GiB is a standing
allocation that does not shrink when the pipeline does, and it is what takes
steady-state `MemAvailable` from ~12.9 GiB to ~7.8 GiB idle. See "The
fresh-container floors are optimistic" above: at two plants in flight only
~1.4 GB of that 4 GiB is doing live work, so this is the first knob to try if
headroom is needed — not a shallower pipeline.

Two footnotes worth keeping. The visual-budget decision cost 4x here too: 1120
shrank effective cache capacity from ~17 plants to ~4.3. And the documented
accounting `mm_processor_cache_gb * (api_server_count + data_parallel_size)`
overstates this deployment — the API process stores only
`MultiModalProcessorCacheItemMetadata` (an `item_size` plus prompt updates)
because, in vLLM's own words, "P1 already stores the tensor data". The tensor
data exists once.

### Runaway generation: 0 in 30, which bounds it rather than measures it

30 structured-output requests at the pinned temperature of 1.0, spread over
**10 different plants**, 3 shards each with the shard rotated so all 9 are
covered — not 30 repeats of one plant, whose rationale text would be too
self-similar to show the real spread.

| | value |
|---|---|
| `max_tokens` actually in force | **8192** (what `smoke.py` / `bench.sh` send; unset, vLLM would allow `MAX_MODEL_LEN` minus the prompt, ~28 k) |
| completion tokens: min / median / p90 / max | **176 / 362.5 / 607 / 681** |
| mean | 374.2 |
| `finish_reason == "length"` | **0 / 30 (0 %)** |
| median latency | 12.6 s |
| slowest request | 63.3 s (a cold prefill, not a runaway) |

**Zero occurrences does not mean zero rate.** With 0 events in 30 trials the
rule of three puts the 95 % upper bound at **~10 %**, which is far too loose to
plan with — at 10 % a 2400-request run would contain 240 runaways. Folding in
the one runaway previously observed (8192 tokens, 190.6 s) over roughly 150
earlier requests gives a point estimate nearer **0.5 %**. The honest statement
is: rare enough not to appear in 30 requests, not rare enough to ignore.

**Suggested cap: `max_tokens = 2048`** — about 3.4x the observed p90 (607) and
3x the observed max (681), so it cannot truncate a legitimate answer, while
bounding a runaway at ~50 s instead of ~190 s. This is a recommendation only;
the ticket puts the retry/timeout decision at the provider layer.

What runaway does to the estimate, using N = 2's measured 5.57 h as the base:

| assumption | added time | 267-plant total |
|---|---|---|
| **no runaway** (what every figure in this file assumes) | — | **5.57 h** |
| 0.5 % at `max_tokens=8192` | 12 requests x ~178 s | 6.16 h |
| 0.5 % at `max_tokens=2048` | 12 requests x ~39 s | 5.70 h |
| 10 % (the 95 % upper bound) at 8192 | 240 x ~178 s | 17.4 h |
| 10 % at 2048 | 240 x ~39 s | 8.2 h |

The last two rows are the argument for the cap: at the pessimistic end of what
n = 30 can rule out, capping `max_tokens` is the difference between 8 h and 17 h.
Concurrency makes it worse than the arithmetic suggests, because a runaway
occupies a scheduler slot for its whole duration and holds up the batch it is
in.

## Deviations from the ticket's flag list

The "deliberately not added" list was honoured in full: no `--kv-cache-dtype`,
no `--quantization`, no `--linear-backend`, no `--enable-prefix-caching`, no
`--tensor-parallel-size`, no `--trust-remote-code`. (`--moe-backend` was on that
list too and is now set — see "Pinned runtime selections" for why, and for the
measured cost.) Four unavoidable departures:

1. **`vllm serve` is not always the command.** `vllm/vllm-openai` images have
   `ENTRYPOINT ["vllm","serve"]`, so passing `vllm serve <model>` yields
   `vllm: error: unrecognized arguments: serve unsloth/...`. The NGC image's
   entrypoint (`/opt/nvidia/nvidia_entrypoint.sh`) execs its args, so it *does*
   need the subcommand. `up.sh`/`pull.sh` inspect `.Config.Entrypoint` and
   prepend `vllm serve` only when absent.
2. **`--mm-processor-kwargs` added** — required to pin decision (b).
3. **`--reasoning-parser gemma4` added** — required for D2; inert for D1.
4. **NGC credential workaround** (above) to pull candidate 2 at all.

## Known failure modes

- **Runaway generation to `max_tokens`.** Observed once: a structured-output
  request on `s0016` produced 8192 completion tokens and took **190.6 s** instead
  of the usual ~500 tokens / ~12 s. The grammar keeps output well-formed, but
  nothing bounds *length* — the model can pad `rationale` strings indefinitely.
  A 30-request probe over 10 plants at temperature 1.0 produced **none**, which
  bounds the rate at ~10 % (95 %, rule of three) rather than measuring it; the
  point estimate including the earlier sighting is ~0.5 %. Suggested cap
  `max_tokens=2048` — 3.4x the observed p90 of 607. The #1 provider still needs
  a per-request timeout and should treat `finish_reason == "length"` as a failed
  shard rather than a partial result. Not worked around here: that is a
  provider-layer decision. Numbers in "Pipeline depth, memory, and runaway
  generation".
- **Memory is genuinely tight at `GPU_MEM_UTIL=0.80`.** With the server up,
  `free -g` shows 109 GiB of 121 GiB used and only ~12 GiB available. Under load
  that shrinks further: a single plant's cold prefill holds it at ~9.3 GiB, and
  three plants in flight took the low-water mark to **7.37 GiB**. vLLM also
  warns that 0.80 is effectively 0.7958 once CUDA-graph profiling is counted. Do
  not raise this casually — the unified pool is shared with the OS and page
  cache, and raising it is the reported route to a hard lock needing a power
  cycle. If you must, go up by 0.05 and re-run acceptance E each time.
- **Fanning out a cold plant's shards destroys the prefix cache.** All 9 shards
  issued concurrently before any of them has written the shared prefix gives a
  9.5 % hit rate and 402.1 s per plant — 2.5x slower than serial (Pattern C
  above). Serialize the first shard, then parallelize the rest.

- **A killed client does not stop in-flight work, and the cache remembers.**
  While measuring this, a test run was cut off by a harness timeout; the Python
  process died but vLLM had already processed its requests, so the next run found
  those plants' prefixes cached and reported a 100 % hit rate and a 9.3 s "cold"
  prefill. Any cold measurement must start from a restarted container with
  `vllm:prefix_cache_queries_total` verified at 0, on plants that container has
  never served.

- **`mm_processor_kwargs` per request fragments the prefix cache.** Sending
  `mm_processor_kwargs: {"max_soft_tokens": 1120}` on a request puts its
  multimodal inputs in a *different* cache namespace from an otherwise identical
  request that relies on the server-side `--mm-processor-kwargs`, even though the
  tokenization is byte-identical. Measured on an unseen line (`s0035`,
  26 photos, 30 695 prompt tokens both ways):

  | request | total |
  |---|---|
  | 1. with kwargs, cold | 71.3 s |
  | 2. with kwargs, repeat | 14.8 s (cache hit) |
  | 3. **without** kwargs, same images already prefilled twice | 71.6 s (**miss**) |
  | 4. without kwargs, repeat | 16.5 s (cache hit) |

  So clients must be *consistent*. The budget is pinned server-side, so the rule
  is: **never send it per request.** `bench.sh` was changed to stop sending it,
  and `smoke.py` never did. Mixing the two forms costs ~57 s on every request
  that lands on the wrong side — silently, with no error.

- **Prefix cache lives as long as the container.** A second `bench.sh` against a
  running server measures cache hits, not prefills (its "cold" probe reported
  TTFT 0.20 s and the projection collapsed to a meaningless 8.0 h lower bound).
  Always `./down.sh && ./up.sh` before benchmarking, and point
  `BENCH_COLD_LINE` at a line that server has not served.
- **First request after startup costs ~11 s** of JIT codegen on top of the
  prefill. `bench.sh` absorbs this in a `max_tokens=3` warm-up; production code
  should not read the first response's latency as representative.
- ~~**MoE backend auto-selection varies between boots**~~ — **not reproduced,
  and now moot.** Five consecutive boots of the pinned digest chose
  `FLASHINFER_CUTLASS` every time, and the selection is a deterministic walk
  down a fixed priority list. The backend is pinned anyway; see "Pinned runtime
  selections". If a boot ever behaves oddly this log line is still the first
  one to check — it now has to read `VLLM_CUTLASS`.
- Image ships `flashinfer-python 0.6.6` / `nvidia-cutlass-dsl 4.4.2`, both
  *below* the model card's recommended `>=0.6.13` / `>=4.5.2`. It works anyway;
  noted in case a future symptom traces back here.

## Frozen data was never written

`--shard-dir` and the photo tree were opened read-only throughout. After all
runs, `03_mature_v2/shard_master_schema` is still `dr-xr-x---` with every file's
mtime at `2026-07-30 11:32`, and `03_mature_v2/images` is unchanged at
`2026-07-13 18:17`.
