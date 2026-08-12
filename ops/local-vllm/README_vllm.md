# Hosting Gemma 4 26B A4B (NVFP4) on vLLM for pxGPT

How to stand up a local, OpenAI-compatible vLLM server for
`unsloth/gemma-4-26B-A4B-it-NVFP4` on a DGX Spark (GB10), and point pxGPT's
`analyze` / `schema` commands at it.

This is the **operating guide**. The measurements behind every choice — image
comparison, latency tables, rejected candidates — are in
[RUNBOOK.md](RUNBOOK.md). Read this file to run the server; read the RUNBOOK to
understand why it is configured this way or to re-derive it on new hardware.

- Back to the project overview: [../../README.md](../../README.md)
- Full pxGPT workflows and provider reference: [../../user_manual.md](../../user_manual.md)

---

## The version constraint — read this first

**Not every vLLM build can load this checkpoint.** This is the single thing most
likely to waste your afternoon, so it comes before everything else.

The NVFP4 weights are `compressed-tensors` with per-expert activation scales.
Older Gemma 4 MoE weight loaders have no mapping for them and die during startup:

```
File ".../vllm/model_executor/models/gemma4.py", line 1359, in load_weights
KeyError: 'layers.0.experts.0.down_proj.input_global_scale'
```

If you see that `KeyError`, **the image is too old — nothing else is wrong.** Do
not add flags, do not re-quantize, do not patch the model executor. Change image.

| Image | vLLM | Loads this checkpoint? |
|---|---|---|
| `vllm/vllm-openai:gemma4-cu130` | `0.19.1.dev6` | **No** — the `KeyError` above |
| `vllm/vllm-openai:cu130-nightly` | `0.19.2rc1.dev107` | Yes (floating nightly tag) |
| **`nvcr.io/nvidia/vllm:26.07-py3`** | **`0.24.0+092c4842.dev`** | **Yes — the pinned choice** |

The fix landed between `0.19.1` and `0.19.2rc1`. Note the image named for Gemma 4
is the one that *cannot* serve it — pick by tested behaviour, not by tag name.

The model card asks for `vllm>=0.25.0`; `0.24.0` works in practice and is what is
pinned here. Treat "≥ 0.19.2rc1, and verify" as the real rule.

**Pinned image** (never use a floating tag for a run you intend to cite):

```
nvcr.io/nvidia/vllm@sha256:95c498a475142c20c989c65e5d223348c09fed83ba17ddf44f117610c0bd3268
```

---

## Prerequisites

- **Hardware**: DGX Spark / GB10 (sm_121), 128 GB unified memory. Verified on
  `aarch64`, Docker 29.1.3, driver exposing compute capability 12.1.
- **Disk**: ~17 GB for the weights plus ~22 GB for the image (the other two
  candidates are 20–23 GB each if `pull.sh` has to try them).
- **`HF_TOKEN`** exported in your shell (`pull.sh` refuses to run without it).
- **Memory headroom**: the server holds ~109 GB of the 121 GB pool while up. Stop
  other GPU work first.

---

## Quick start

```bash
cd ops/local-vllm
cp env.example .env

# Set MEDIA_ROOT to the absolute path of your photo root, e.g.
#   MEDIA_ROOT=/home/xavier/project/pxgpt/03_mature_v2/images
$EDITOR .env

export HF_TOKEN=hf_...
./pull.sh          # downloads weights, finds a working image, pins both into .env
./up.sh            # starts the server; waits for /health (~4 min)
```

Then verify against your real data:

```bash
pip install -r requirements.txt
set -a; source .env; set +a
python smoke.py                       # acceptance A-E, exits non-zero on failure
```

Day-to-day:

```bash
./up.sh      # idempotent: a running server is left alone
./logs.sh    # docker logs -f
./down.sh    # stop + remove (the HF weight cache is kept)
```

`pull.sh` only needs re-running to change image or model revision. It walks the
candidate images in order, keeps the first that loads the model *and* answers
`/health`, records each rejection (with 50 log lines) into `RUNBOOK.md`, resolves
the HF revision to an immutable commit sha, and writes both pins into `.env`.

### If `pull.sh` fails at the NGC image

On this host, `docker pull nvcr.io/nvidia/...` fails before reaching the registry:

```
error getting credentials - err: exit status 1, out: `exit status 2:
gpg: public key decryption failed: No such file or directory`
```

`~/.docker/config.json` has `{"credsStore":""}`, so Docker falls back to a
gpg/`pass` helper that is not installed. It is a **host configuration bug**, not a
pxGPT one, and it blocks only NGC (Docker Hub never triggers a credential
lookup). The repo does not paper over it. Either repair the helper, or pull once
by hand with a config that stops Docker consulting it — the repo *is* anonymously
pullable:

```bash
mkdir -p /tmp/ngccfg && echo '{"auths":{"nvcr.io":{}}}' > /tmp/ngccfg/config.json
DOCKER_CONFIG=/tmp/ngccfg docker pull nvcr.io/nvidia/vllm:26.07-py3
./pull.sh          # now finds the image already present
```

---

## Configuration (`.env`)

`env.example` is version-controlled; `.env` is gitignored. `<FILL>` values are
written by `pull.sh` or by you.

| Variable | Meaning |
|---|---|
| `VLLM_IMAGE` | Image **pinned by digest**. `up.sh` refuses a floating tag. |
| `MODEL_REPO` | `unsloth/gemma-4-26B-A4B-it-NVFP4` |
| `MODEL_REVISION` | HF commit sha, e.g. `20df0542b1a86ce19f495ac2eca2c7c12bce82f9` |
| `SERVED_MODEL_NAME` | `gemma4-26b-a4b-nvfp4` — what clients pass as `model` |
| `MEDIA_ROOT` | Absolute photo root. Mounted read-only at the **same path** inside the container. |
| `MAX_MODEL_LEN` | 65536 |
| `MAX_NUM_SEQS` | 2 |
| `GPU_MEM_UTIL` | 0.80 — see the warning below |
| `ENABLE_THINKING` | `false` (pinned decision) |
| `IMAGE_TOKEN_BUDGET` | `1120` (pinned decision) |
| `SHARD_DIR`, `SYSTEM_PROMPT`, `BENCH_LINE`, `BENCH_COLD_LINE` | Real data used by `smoke.py` / `bench.sh`, **read-only** |

> **Do not raise `GPU_MEM_UTIL` casually.** The unified pool is shared with the OS
> and page cache; with the server up only ~12 GB of 121 GB remains free. Raising
> it is the reported route to a hard lock needing a power cycle. If you must, go
> up by 0.05 at a time and re-run `smoke.py` after each step.

`MEDIA_ROOT` is bind-mounted as `$MEDIA_ROOT:$MEDIA_ROOT:ro` deliberately: the
path must be **identical inside and outside** the container, or `file://` URLs
sent by a client will not resolve on the server side.

---

## The two settings that matter for correctness

These are pinned in `.env` and passed by `up.sh`. Both were chosen from
measurements ([RUNBOOK.md](RUNBOOK.md)); changing either invalidates comparisons
against the Anthropic / OpenAI backends.

### 1. Thinking is off

Gemma 4 is a reasoning model, but its chat template defaults
`enable_thinking=false` and, when off, pre-fills an already-closed
`<|channel>thought\n<channel|>` block. The model's first generated token is
therefore the answer itself, and a `response_format` grammar binds cleanly from
token 0. The feared grammar-versus-reasoning conflict simply does not arise.

Thinking *on* also produces schema-valid JSON (with
`--reasoning-parser gemma4`, which `up.sh` passes so the path stays available),
but takes **73–85 s** against **9–13 s** for the same result, and its reasoning
text duplicates the `rationale` field the Stage 3 schema already requires. Off is
the default for good reason.

### 2. Visual token budget: 1120 tokens per image

The knob is **`max_soft_tokens`**, not `image_seq_length`. Supported ladder:
`70 · 140 · 280 · 560 · 1120`. The checkpoint's own `processor_config.json`
defaults to 280, so unset == 280 — this deployment overrides it to the top of the
ladder. Measured prompt-token cost for one photo:

| `max_soft_tokens` | 70 | 140 | 280 | 560 | 1120 |
|---|---|---|---|---|---|
| prompt tokens | 84 | 150 | 284 | 575 | 1131 |

**Pinned at 1120**, the top of the ladder. The Stage 3 traits are fine-grained
(petiole cross-section, colour hue, leaf margin), and 1120 sits close to Anthropic
Sonnet 5's per-image tokenization, so local and cloud runs stay comparable — which
matters more here than throughput, because the whole point is comparing backends.

Cost at 1120, measured (not extrapolated): a 32-photo line is **37 349** prompt
tokens, comfortably inside `MAX_MODEL_LEN=65536` — the largest line in the dataset
is 32 photos, and even 49 would fit.

| | 280 | **1120** |
|---|---|---|
| prompt tokens, 32-photo line | 10 245 | **37 349** |
| cold request, total | 24.4 s | **72.7 s** |
| prefix-cached request, total | 12.2 s | **14.6 s** |
| realistic 2400-request run | 9.0 h | **14.0 h** |

The 4x token increase does **not** cost 4x wall-clock. Cold prefill does get much
worse (7.4x: the vision tower processes 10 080 patches per image instead of
2 520), but prefix-cached requests barely move, and pxGPT sends 9 shard requests
per plant off one shared image prefix — so only 1 in 9 pays the cold prefill. The
end-to-end cost of choosing 1120 over 280 is about **+56 %**. Full tables in
[RUNBOOK.md](RUNBOOK.md).

---

## Sending requests

Any OpenAI client works. `api_key` must be non-empty but is not checked.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local",
                max_retries=0, timeout=1800)

resp = client.chat.completions.create(
    model="gemma4-26b-a4b-nvfp4",
    max_tokens=8192,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            # images BEFORE text, as the Gemma 4 card requires
            {"type": "image_url", "image_url": {"url": "file:///abs/path/photo.jpg"}},
            {"type": "text", "text": shard_prompt},
        ]},
    ],
    response_format={"type": "json_schema",
                     "json_schema": {"name": "shard_02", "schema": schema, "strict": True}},
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        # NOTE: do NOT send mm_processor_kwargs -- see the warning below
    },
)
```

Four things to get right:

1. **Images before text.** Both the model card and pxGPT's own request layout put
   image blocks first.
2. **`file://` needs the matching mount.** The path must be under `MEDIA_ROOT`
   and identical on both sides. Base64 `data:` URLs work with no mount at all —
   that is what `pxgpt analyze` sends.
3. **Reasoning arrives as `message.reasoning`**, not `reasoning_content`, on
   vLLM 0.24. Neither is modelled by the OpenAI SDK, so both land in
   `model_extra`. Check both names.
4. **Bound the output and check `finish_reason`.** The grammar constrains shape
   but not length; see Troubleshooting.

> **Never send `mm_processor_kwargs` per request.** The budget is pinned
> server-side by `up.sh`, and passing it on the request puts the images in a
> *different* prefix-cache namespace — identical tokenization, but the cached
> entry is not reused. Measured on the same 26-photo line: 14.8 s when the key
> matched, 71.6 s when it did not. There is no error, just a silent ~57 s penalty
> on every mismatched request. Rely on the server default and stay consistent.

---

## Pointing pxGPT at the server

pxGPT reaches vLLM through `OpenAICompatProvider`. `VLLM_MODEL` must equal
`SERVED_MODEL_NAME`:

```bash
VLLM_BASE_URL=http://localhost:8000/v1    # already the default
VLLM_MODEL=gemma4-26b-a4b-nvfp4           # REQUIRED — the served name
VLLM_API_KEY=EMPTY                        # any non-empty placeholder
```

```bash
pxgpt analyze --provider vllm \
  --input-folder path/to/images --output out.txt \
  --system-prompt system.txt --prompt prompt.txt
```

`analyze` and `schema` support vLLM. The batch stages (`describe-batch*`,
`phenotype-batch*`) are Anthropic/OpenAI-only — they depend on the providers'
Batch and Files APIs, which a local server does not offer. For `schema`, the JSON
schema is appended to the system prompt rather than sent as native structured
output, so the user prompt must ask for JSON-only output (the bundled
`prompts/extract_traits.txt` does).

See [../../user_manual.md](../../user_manual.md) for the full command reference.

---

## Verifying and benchmarking

`smoke.py` runs the acceptance suite against **real** shard schemas, photos and
system prompt — all opened read-only — and exits non-zero on any failure, printing
the offending response and the `jsonschema` error path.

```bash
set -a; source .env; set +a
python smoke.py                 # A B C D1 D2 E
python smoke.py --tests D1,E    # a subset
python smoke.py --shard shard_04 --line s0016
```

| | Check |
|---|---|
| A | `/v1/models` reports `SERVED_MODEL_NAME` |
| B | text-only completion returns content |
| C | one real photo over `file://` (pipeline only — no semantic assertion) |
| D1 | thinking off + strict `json_schema` → parses and validates |
| D2 | thinking on + `gemma4` parser → content validates, `reasoning` non-empty |
| E | a whole plant line's photos + schema, within `MAX_MODEL_LEN` |

By default `smoke.py` picks the **largest** shard schema, the one most likely to
trip a grammar limit.

`bench.sh` measures the shape pxGPT actually sends — a full photo set plus one
real shard schema — rather than `vllm bench serve`'s synthetic
short-prompt/long-output dataset:

```bash
./down.sh && ./up.sh && ./bench.sh     # restart first, see below
```

Reference figures at the pinned budget of **1120** (32 photos, thinking off,
fresh container): **37 349** prompt tokens, **40.8 tok/s** decode, **14.6 s** with
a warm prefix cache, **72.7 s** cold, projecting **~14.0 h** for 2400 requests
(267 plants x [1 cold + 8 cached]).

> Ordering matters as much as the budget: 9 shards per plant share one image
> prefix, so dispatching **plant-major** keeps 8 of 9 requests on the cache. Going
> shard-major would pay the cold prefill every time and push the same run toward
> **48.5 h**.

> **Always restart before benchmarking.** Prefix caching is on by default and its
> cache lives as long as the container, so a second `bench.sh` against the same
> server measures cache hits and reports a meaninglessly low TTFT.
> `BENCH_COLD_LINE` must also name a line that server has not served yet.

---

## Troubleshooting

**`KeyError: '...input_global_scale'` at startup** — image too old. See
[the version constraint](#the-version-constraint--read-this-first).

**`vllm: error: unrecognized arguments: serve unsloth/...`** — the
`vllm/vllm-openai` images already have `ENTRYPOINT ["vllm","serve"]`, so the
subcommand must not be repeated; the NGC image needs it spelled out. `up.sh` and
`pull.sh` inspect `.Config.Entrypoint` and adapt, so this only bites hand-rolled
`docker run` commands.

**A request takes ~190 s and returns 8192 completion tokens** — runaway
generation. The grammar keeps output well-formed but nothing bounds length, so the
model can pad `rationale` strings indefinitely. Set a per-request timeout and
treat `finish_reason == "length"` as a **failed** shard, not a partial result.

**First request is ~11 s slower than the rest** — JIT codegen. Send a
`max_tokens=3` warm-up before timing anything.

**`file://` URL rejected or not found** — the path must sit under `MEDIA_ROOT`
*and* be identical inside the container. Check `--allowed-local-media-path` in
`docker inspect --format '{{join .Args " "}}' pxgpt-vllm`.

**Server never becomes healthy** — `up.sh` prints the last 50 log lines on
timeout or exit. Cold start is ~4 minutes (~100 s weights, ~41 s
`torch.compile`, ~74 s engine init); allow for it before assuming a hang.

**MoE backend differs between boots** (`FLASHINFER_CUTLASS` vs `VLLM_CUTLASS`) —
expected. Auto-selection is not stable across boots and both serve correctly,
which is exactly why `--moe-backend` is left unset. Never force Marlin: the model
card measures it ~2x slower.

---

## What is deliberately not configured

`--kv-cache-dtype`, `--quantization`, `--moe-backend`, `--linear-backend`,
`--enable-prefix-caching`, `--tensor-parallel-size`, `--trust-remote-code`. Each
is either auto-detected, already the default, meaningless on a single GB10, or a
known performance trap on Spark. [RUNBOOK.md](RUNBOOK.md) records the reasoning
per flag. If you find yourself needing one, record why there too.

The scripts also never write to `SHARD_DIR` or `MEDIA_ROOT`. The Stage 3 shard
sets are frozen, `chmod -w`, checksummed and under human evaluation; if any step
appears to need write access there, stop and report it rather than working around
it.
