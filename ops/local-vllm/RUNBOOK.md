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
| `MAX_NUM_SEQS` | 2 |
| `GPU_MEM_UTIL` | 0.80 |
| **Thinking** | **off** (`enable_thinking: false` per request) |
| **Visual token budget** | **1120 tokens/image** (`--mm-processor-kwargs '{"max_soft_tokens": 1120}'`) |

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
never Marlin, so `--moe-backend` stays unset as instructed. Note the choice is
**not stable across boots**: an earlier boot of the same image selected
`VLLM_CUTLASS` rather than `FLASHINFER_CUTLASS` for the MoE. Both served
correctly; this is why pinning a backend by hand would be wrong.

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

**Cost of the decision, stated plainly.** Every latency and throughput figure in
the "Benchmark" section below was measured at **280** and is therefore an
optimistic lower bound for the pinned configuration. Prompt tokens rise ~3.6x
(10 245 → ~37 k for a 32-photo line) and this workload is prefill-dominated.
Extrapolating the 280 → 560 step (20.8 s → 42.1 s), expect roughly **~80-85 s**
per full-line request at 1120, and re-derive the 2400-request projection from a
fresh `./down.sh && ./up.sh && ./bench.sh` before committing to a full run.
The 140/280/560 measurements are left intact below as the evidence they are.

**This value is now pinned and must not drift** — it sets the visual information
available to the model, so changing it invalidates comparisons against the
Anthropic and OpenAI backends.

## Acceptance results

`smoke.py` — all green, exit 0 (111 s):

| | check | result |
|---|---|---|
| A | `/v1/models` `data[0].id` == `gemma4-26b-a4b-nvfp4` | PASS |
| B | text-only completion non-empty | PASS |
| C | one real photo via `file:///abs/path.jpg` | PASS, 282 prompt tokens |
| D1 | thinking off + strict `json_schema` on real shard | PASS, 13.0 s |
| D2 | thinking on + `gemma4` parser, same schema | PASS, 73.0 s |
| E | all 32 photos of `s0019` + schema | PASS, 10 245 ≤ 65 536 |
| F | `up.sh` twice → one container; `down.sh` then `up.sh` → back up | PASS |
| G | `bench.sh` reports the five numbers | PASS |

C deliberately asserts only that the pipeline works (HTTP 200, non-empty
content), not that the model sees correctly. For the record it did describe the
photo accurately ("a small green kale plant ... next to a vertical wooden ruler
for scale").

## Benchmark (`bench.sh`)

One real plant line's full photo set + one real shard schema → structured JSON.
Not `vllm bench serve`: its random dataset is short-prompt/long-output, the
inverse of this workload. `s0019`, 32 photos, `shard_02`, **budget 280**,
thinking off, fresh container.

> Measured before the budget was pinned to 1120. Kept as recorded; see the cost
> note in Decision (b) and re-run `bench.sh` for figures at 1120.

```
warm-up (max_tokens=3):  11.2 s   [JIT codegen + cold prefill of 10245 tokens]

run 1: prompt=10245 completion=453 ttft=0.92s total=11.1s decode=44.2 tok/s
run 2: prompt=10245 completion=552 ttft=0.18s total=12.6s decode=44.3 tok/s
run 3: prompt=10245 completion=530 ttft=0.23s total=12.2s decode=44.2 tok/s
```

| metric | median |
|---|---|
| prompt tokens (incl. images) | 10 245 |
| TTFT (prefix-cache hit) | 0.23 s |
| decode throughput | 44.2 tok/s |
| total latency | 12.2 s |
| completion tokens | 530 |
| cold-prefill probe (`s0044`, 26 photos, unseen) | TTFT **7.83 s**, total **24.4 s** |

### 2400-request projection

Prefix caching is on by default (V1), and pxGPT sends 9 shard requests per plant
that all share the same image prefix — so per plant one request pays a cold
prefill and eight hit the cache. That mix, not the raw median, is the right basis:

| scenario | estimate |
|---|---|
| all-cold upper bound | 16.3 h (24.4 s x 2400) |
| all-cached lower bound | 8.1 h (12.2 s x 2400) |
| **realistic (267 plants x [1 cold + 8 cached])** | **9.0 h** |

Serial only; concurrency above 1 was not measured. Cross-checked independently on
three other lines (`s0016`/`s0017`/`s0014`): cold TTFT median 9.26 s / total
19.5 s, cached TTFT 0.24 s / total 12.4 s → ~8.8 h, agreeing with the 9.0 h above.

## Deviations from the ticket's flag list

The "deliberately not added" list was honoured in full: no `--kv-cache-dtype`,
no `--quantization`, no `--moe-backend`, no `--linear-backend`, no
`--enable-prefix-caching`, no `--tensor-parallel-size`, no `--trust-remote-code`.
Four unavoidable departures:

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
  At 9 shards x 267 plants a few such requests are near-certain, so the #1
  provider needs a per-request timeout and should treat
  `finish_reason == "length"` as a failed shard rather than a partial result.
  Not worked around here: that is a provider-layer decision.
- **Memory is genuinely tight at `GPU_MEM_UTIL=0.80`.** With the server up,
  `free -g` shows 109 GiB of 121 GiB used and only ~12 GiB available. vLLM also
  warns that 0.80 is effectively 0.7958 once CUDA-graph profiling is counted. Do
  not raise this casually — the unified pool is shared with the OS and page
  cache, and raising it is the reported route to a hard lock needing a power
  cycle. If you must, go up by 0.05 and re-run acceptance E each time.
- **Prefix cache lives as long as the container.** A second `bench.sh` against a
  running server measures cache hits, not prefills (its "cold" probe reported
  TTFT 0.20 s and the projection collapsed to a meaningless 8.0 h lower bound).
  Always `./down.sh && ./up.sh` before benchmarking, and point
  `BENCH_COLD_LINE` at a line that server has not served.
- **First request after startup costs ~11 s** of JIT codegen on top of the
  prefill. `bench.sh` absorbs this in a `max_tokens=3` warm-up; production code
  should not read the first response's latency as representative.
- **MoE backend auto-selection varies between boots** of the same image
  (`FLASHINFER_CUTLASS` vs `VLLM_CUTLASS`). Both worked. If a boot ever behaves
  oddly, check this log line before anything else.
- Image ships `flashinfer-python 0.6.6` / `nvidia-cutlass-dsl 4.4.2`, both
  *below* the model card's recommended `>=0.6.13` / `>=4.5.2`. It works anyway;
  noted in case a future symptom traces back here.

## Frozen data was never written

`--shard-dir` and the photo tree were opened read-only throughout. After all
runs, `03_mature_v2/shard_master_schema` is still `dr-xr-x---` with every file's
mtime at `2026-07-30 11:32`, and `03_mature_v2/images` is unchanged at
`2026-07-13 18:17`.
