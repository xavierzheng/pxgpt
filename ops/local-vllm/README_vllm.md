# Hosting Gemma 4 26B A4B (NVFP4) on vLLM for pxGPT

How to stand up a local, OpenAI-compatible vLLM server for
`unsloth/gemma-4-26B-A4B-it-NVFP4` on a DGX Spark (GB10), and point pxGPT's
`analyze` / `schema` commands at it.

This is the **operating guide**. The measurements behind every choice — image
comparison, latency tables, rejected candidates — are in
[RUNBOOK.md](RUNBOOK.md). Read this file to run the server; read the RUNBOOK to
understand why it is configured this way or to re-derive it on new hardware.

**Just want it running?** [Step 0](#step-0-will-this-run-on-my-machine) checks
your hardware, [Quick start](#quick-start) is four commands, then
[Pointing pxGPT at the server](#pointing-pxgpt-at-the-server) connects pxGPT to
it. Everything after that is reference — read it when you need it.

- Back to the project overview: [../../README.md](../../README.md)
- Full pxGPT workflows and provider reference: [../../user_manual.md](../../user_manual.md)

---

## Step 0: will this run on my machine?

**Run these four checks first.** This deployment is pinned to one machine class.
If a check fails, stop here: you will otherwise find out ~40 minutes and ~40 GB
into the download.

```bash
uname -m                    # want: aarch64
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
                            # want: compute_cap 12.1, memory.total ~128 GB (GB10)
docker --version            # want: any recent Docker; verified on 29.1.3
df -h .                     # want: >= 40 GB free (17 GB weights + 22 GB image)
```

| Check | Needed | If it does not match |
|---|---|---|
| architecture | `aarch64` | This is a DGX Spark / GB10 deployment. x86 hosts need their own image and a re-run of [RUNBOOK.md](RUNBOOK.md); the pins here do not transfer. |
| compute capability | `12.1` (sm_121) | The pinned image is built for sm_121. Another GPU needs a different vLLM build. |
| GPU memory | ~128 GB unified | The server holds **~109 GB of the 121 GB pool** while up. A smaller card cannot load this model at `MAX_MODEL_LEN=65536`. Stop other GPU work before starting. |
| free disk | ≥ 40 GB | 17 GB weights + 22 GB image. Add ~45 GB more if `pull.sh` has to fall through to the other candidates. |

You also need a **Hugging Face token**, because the weights repo may be gated:
create one at <https://huggingface.co/settings/tokens> (read access is enough).
It is a credential — `export` it in your shell, never put it in `.env`.
`pull.sh` reads it from the process environment, not from `.env`, and validates it
against the API before it downloads anything.

> **Not a DGX Spark?** Nothing above is a soft warning. The image digest, the MoE
> backend and the memory numbers were all measured on this one machine class, and
> [RUNBOOK.md](RUNBOOK.md) is the guide for re-deriving them elsewhere. pxGPT
> itself talks plain OpenAI HTTP, so any vLLM server you can start will work —
> see [Pointing pxGPT at the server](#pointing-pxgpt-at-the-server).

---

## Quick start

> **Do [Step 0](#step-0-will-this-run-on-my-machine) first.** It is four read-only commands and takes under a
> minute. `pull.sh` downloads ~40 GB; Step 0 is how you find out beforehand that
> this machine can run the result.

Copy the block below and **edit one line of `.env`**. Everything else is written
for you by `pull.sh` (`VLLM_IMAGE`, `MODEL_REVISION`) or ships already tuned. You
are already inside the repo if you are reading this file — no clone needed.

```bash
cd ops/local-vllm                      # from the repo root

cp env.example .env
${EDITOR:-nano} .env                   # set MEDIA_ROOT. Type a real path:
                                       #   MEDIA_ROOT=/home/you/plants
                                       # NOT MEDIA_ROOT=<your path> -- angle
                                       # brackets are a bash syntax error.

export HF_TOKEN=hf_...                 # from https://huggingface.co/settings/tokens
curl -sf -H "Authorization: Bearer $HF_TOKEN" \
     https://huggingface.co/api/whoami-v2 >/dev/null \
  && echo "token OK" || echo "TOKEN REJECTED -- fix it before pull.sh"

./pull.sh                              # ~40 GB: weights + a working image,
                                       # both pinned into .env for you
./up.sh                                # starts the server and waits for /health:
                                       # ~4 min typical, 20 min cap on a cold
                                       # weight load + JIT codegen
```

**What to put in `MEDIA_ROOT`**: the absolute path to the folder that holds your
photos — your own path, not one from this repo. It is a **prefix**, so point it at
a parent directory rather than one dataset's images; then every dataset under it
resolves without restarting the server. Two rules, because `.env` is loaded by
`source`:

- An absolute path. No `~` — it is not expanded here.
- **No angle brackets.** `MEDIA_ROOT=<your path>` is a bash *syntax error*, not a
  placeholder: `<` reads as a redirection. Type the real path.

`pull.sh` checks the token itself before it downloads anything, so the `curl`
line above is belt-and-braces — useful if you want to confirm the token on its
own first.

If either script stops, read what it printed. `pull.sh` and `up.sh` both check
`.env` before doing anything and name exactly what is wrong — a missing file, a
line that is not `NAME=value`, an empty or non-existent `MEDIA_ROOT`, an
unexported `HF_TOKEN`, an image that is not digest-pinned. That message is the
whole fix; you should not need to come back here.

### Then check it actually works

`up.sh` exits once the server answers `/health`, so the server is up when it
returns. `smoke.py` is the separate question of whether it serves **your** model
and resolves **your** image paths. Run it once after the first `up.sh`.

```bash
# still in ops/local-vllm/. These packages are needed ONLY by smoke.py and
# bench.sh -- the server itself runs in Docker and needs nothing from pip.
# This is the ops requirements file, separate from the repo root's.
pip install -r requirements.txt

set -a; source .env; set +a   # loads the SERVER names, for smoke.py only.
                              # This does NOT configure pxgpt -- pxgpt reads a
                              # different set of variables, see "Pointing pxGPT
                              # at the server" below.
python smoke.py               # acceptance checks A-H; non-zero exit means failure
```

`smoke.py` passing means the server is up, serves the right model, and resolves
your `file://` image paths.

### Last step — point pxGPT at it

**`up.sh` cannot do this for you.** It sources `.env` inside its own process, and
a child cannot export back into your shell. pxGPT also never reads a `.env` — only
the process environment. So set these yourself, in the shell you run `pxgpt` from:

```bash
cd ../..                                  # repo root
set -a; source ops/local-vllm/.env; set +a   # gets SERVED_MODEL_NAME and PORT
export VLLM_MODEL="$SERVED_MODEL_NAME"    # server name and client name must match
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_API_KEY=EMPTY                 # any non-empty string; nothing checks it
export TIMEOUT=1800                       # a cold prefill takes 75-95 s

pxgpt analyze --provider vllm --image-transport file \
  --input-folder "$MEDIA_ROOT"/<your plant folder> \
  --output /tmp/one.txt \
  --system-prompt prompts/phenotyping_system.txt \
  --prompt prompts/describe_plant.txt
```

Skip the `set -a; source` line and you must type the literals instead — the
server's names and pxGPT's names are deliberately different, so nothing catches a
mismatch except a 404. Full mapping and the reasoning:
[Pointing pxGPT at the server](#pointing-pxgpt-at-the-server).

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

`env.example` is version-controlled; `.env` is gitignored. Values ship **empty**
when something has to fill them in: `MEDIA_ROOT` is yours to set, `VLLM_IMAGE`
and `MODEL_REVISION` are written by `pull.sh`. Everything else ships with a
working value.

`.env` is `source`d by both scripts, so it has to stay valid shell — every line
`NAME=value`, and any value with a space or one of `< > | & ( ) ; # *` quoted.
Empty is how "not set yet" is spelled; a placeholder like `<FILL>` is a **bash
syntax error**, because `<` reads as a redirection.

| Variable | Meaning |
|---|---|
| `VLLM_IMAGE` | Image **pinned by digest**. `up.sh` refuses a floating tag. |
| `MODEL_REPO` | `unsloth/gemma-4-26B-A4B-it-NVFP4` |
| `MODEL_REVISION` | HF commit sha, e.g. `20df0542b1a86ce19f495ac2eca2c7c12bce82f9` |
| `SERVED_MODEL_NAME` | `gemma4-26b-a4b-nvfp4` — what clients pass as `model` |
| `PORT` | `8000`, shipped filled in. `up.sh` binds it on the host; pxGPT reaches it as `VLLM_BASE_URL=http://localhost:8000/v1`. Change it here, not in the client. |
| `MEDIA_ROOT` | Absolute photo root. Mounted read-only at the **same path** inside the container, and passed as `--allowed-local-media-path`. A **prefix**, so any subfolder resolves — point it at the project root to cover every dataset. |
| `MAX_MODEL_LEN` | 65536 |
| `MAX_NUM_SEQS` | 16 — free *for KV cache* (that pool is sized by `GPU_MEM_UTIL`, not by sequence count). It is not a licence to run many plants' cold prefills at once: see the depth limit under [Concurrency](#concurrency-how-to-dispatch). |
| `GPU_MEM_UTIL` | 0.80 — see the warning below |
| `ENABLE_THINKING` | `false` (pinned decision) |
| `IMAGE_TOKEN_BUDGET` | `1120` (pinned decision) |
| `MOE_BACKEND` | `cutlass` — the **CLI** spelling of the backend the log calls `VLLM_CUTLASS`. Passing `VLLM_CUTLASS` here makes vLLM refuse to start. |
| `TEMPERATURE`, `TOP_P`, `TOP_K` | `1.0` / `0.95` / `64` — the checkpoint's own defaults, now stated. Read by **both** `up.sh` and the clients. |
| `SHARD_DIR`, `SYSTEM_PROMPT`, `BENCH_LINE`, `BENCH_COLD_LINE` | Real data used by `smoke.py` / `bench.sh`, **read-only** |

> **Do not raise `GPU_MEM_UTIL` casually.** The unified pool is shared with the OS
> and page cache; with the server up only ~12 GB of 121 GB remains free. Raising
> it is the reported route to a hard lock needing a power cycle. If you must, go
> up by 0.05 at a time and re-run `smoke.py` after each step.

`MEDIA_ROOT` is bind-mounted as `$MEDIA_ROOT:$MEDIA_ROOT:ro` deliberately: the
path must be **identical inside and outside** the container, or `file://` URLs
sent by a client will not resolve on the server side.

> **Sampling is set in two places from one source, and there is no seed.**
> `up.sh` passes `--override-generation-config` (which *merges* into the
> checkpoint's `generation_config.json` — verified, nothing else regresses to a
> vLLM default), and `smoke.py` / `bench.sh` send the same three values on every
> request. Server-side alone would keep them out of the request record;
> client-side alone would let a client that forgets them silently inherit
> Google's checkpoint defaults. Never add a `seed`: repeated inference of one
> plant is how consistency gets measured, and a fixed seed would return
> identical output with zero variance and no error. `smoke.py` check H guards
> this.

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
the default for good reason, and `schema` pins it off: a stray `STAGE3_EFFORT`
prints a note and is ignored, because the setting has to be identical across all
267 plants for the results to be comparable.

`analyze` can turn it on with `--effort <level>` (the local models have no
reasoning *levels*, so any level simply means on; no server restart or `.env`
change is needed — `--reasoning-parser gemma4` is already running). The parser
keeps the thinking in the response's own `reasoning` field and leaves `content`
holding the final answer alone, so **only the answer is written to `--output`**;
the reasoning is never saved. Measured on one plant line, one-sentence prompt:
2506 completion tokens / 94 s with thinking on, against 36 tokens / 1.7 s off.

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
| 2400-request run, serial | 9.0 h (modelled) | **12.0 h (measured)** |
| 2400-request run, best dispatch | — | **5.6 h (measured, n=4 plants)** |

> **What these hours do and do not include.** Both are wall-clock extrapolations
> from plants that completed normally: **runaway generation is not in them.** It
> was probed at 30 requests over 10 plants and did not occur, which bounds its
> rate at ~10 % rather than measuring it (see [RUNBOOK.md](RUNBOOK.md)); at the
> ~0.5 % point estimate the 5.6 h becomes ~6.2 h, and capping `max_tokens` at
> 2048 brings it back to ~5.7 h. The 5.6 h supersedes an earlier 6.2 h figure
> that came from a single run of two plants with no prefix-hit or memory data.

The 4x token increase does **not** cost 4x wall-clock. Cold prefill does get much
worse (7.4x: the vision tower processes 10 080 patches per image instead of
2 520), but prefix-cached requests barely move, and pxGPT sends 9 shard requests
per plant off one shared image prefix — so only 1 in 9 pays the cold prefill.
End-to-end cost of 1120 over 280 is about **+33 %**.

Measured directly, all 9 shards of `s0019` (32 photos) back to back on a fresh
container: `shard_01` cost **86.2 s** at a **0 %** prefix-cache hit, then shards
2-9 cost **4.4-14.7 s** each at **97.6-99.2 %** hit (36 448 tokens of system +
images reused; `mm_cache` 32/32). **One plant = 162.2 s**, so 267 plants =
**12.0 h**. Per-shard table in [RUNBOOK.md](RUNBOOK.md).

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
    temperature=0.5, top_p=0.95,        # from .env; top_k rides in extra_body
    # NO seed -- see the note under Configuration
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
        "top_k": 64,          # no OpenAI field for it, so it goes here
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
>
> Re-confirmed on vLLM 0.24 (2026-08-20): a byte-identical prompt already in the
> cache re-paid a full 50.7 s prefill at 2.5 % hit the first time the kwargs were
> attached. Note that a client sending them on *every* request looks healthy —
> its warm requests hit 100 % — so the hit-rate warning in `schema --shard-dir`
> does not detect this. See RUNBOOK.md for the table.

---

## Pointing pxGPT at the server

**This `.env` configures the server. It does not configure pxGPT.** The two use
different variable names and neither feeds the other:

| | here (`ops/local-vllm/.env`) | pxGPT's environment |
|---|---|---|
| model name | `SERVED_MODEL_NAME` | `VLLM_MODEL` |
| endpoint | `PORT` | `VLLM_BASE_URL` |
| image root | `MEDIA_ROOT` | *(never read by pxGPT)* |

`up.sh` sources this file inside its own process, so nothing it sets survives
into the shell you later run `pxgpt` from — a child process cannot export back to
its parent. pxGPT also never loads a `.env` itself; it reads the process
environment only. Both facts point the same way: **you always set the client
variables yourself.**

The safest way is to derive them from this file, so the two cannot drift:

```bash
set -a; source ops/local-vllm/.env; set +a        # one source of truth

export VLLM_MODEL="$SERVED_MODEL_NAME"            # cannot disagree
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_API_KEY=EMPTY                         # any non-empty string; up.sh
                                                  # passes no --api-key, so the
                                                  # server never checks it
export TIMEOUT=1800                               # 300 s default is too tight
```

This also carries `TEMPERATURE` / `TOP_P` / `TOP_K` over — the only three names
the two files share — so the client sends exactly the sampling the server was
started with.

Writing the literals out works too, and is what the rest of this guide shows.
**Keep the `export`.** pxGPT never loads a `.env` of its own — it reads the
process environment only, so a bare `VLLM_MODEL=...` stays local to your shell
and pxGPT will not see it. The symptom is a 404 for a model the server *is*
serving:

```bash
export VLLM_BASE_URL=http://localhost:8000/v1    # 8000 is PORT's shipped value
export VLLM_MODEL=gemma4-26b-a4b-nvfp4           # REQUIRED — the served name
export VLLM_API_KEY=EMPTY                        # any non-empty placeholder
```

If `VLLM_MODEL` and `SERVED_MODEL_NAME` disagree the server 404s on a model it is
not serving, so check with
`curl -s localhost:8000/v1/models | grep -o '"id":"[^"]*"' | head -1`.

```bash
pxgpt analyze --provider vllm \
  --input-folder path/to/images --output out.txt \
  --system-prompt system.txt --prompt prompt.txt
```

`analyze` and `schema` support vLLM. The batch stages (`describe-batch*`,
`phenotype-batch*`) are Anthropic/OpenAI-only — they depend on the providers'
Batch and Files APIs, which a local server does not offer.

`schema` sends the JSON schema as **native structured output** —
`response_format` `{"type": "json_schema", …, "strict": true}`, i.e. real
constrained decoding through xgrammar — not as prose in the system prompt. The
schema therefore appears in exactly one place, so the prompt stays byte-identical
to what the other providers see and the runs remain comparable. If the server
rejects `response_format` the command fails; it never quietly falls back to the
prompt-text path, because that produces output that looks fine and is completely
unconstrained. The user prompt does not need to ask for JSON.

`schema --shard-dir` runs one plant through a whole shard set — the cheap
rehearsal before committing hours of GPU time to all 267:

```bash
pxgpt schema --provider vllm \
  --shard-dir  ../02_mature_v1/shard_master_schema \
  --input-folder path/to/images/s0019 \
  --output      /tmp/smoke \
  --image-transport file
```

`--image-transport file` sends `file://` URIs instead of base64 and is the
production path here; the server must have that directory mounted at the very
same path (`MEDIA_ROOT`, `--allowed-local-media-path`). Test with `file`, not
base64: base64 passes with no mount at all, so a base64 smoke test tells you
nothing about whether the real run will start.

Shards run one at a time and each one prints its `cached_tokens`. `shard_01`
should be ~0 and every shard after it 97-99 %; if they are all ~0 the prefix
cache is not being reused. Set `TIMEOUT=1800` for local runs — a cold prefill
measures 75-86 s and the 300 s default leaves little headroom on a busy box.
Each shard's result lands in `<output>/_partial/` as it succeeds, so a re-run
skips what already worked (`--no-resume` to force everything).

See [../../user_manual.md](../../user_manual.md) for the full command reference.

---

## Concurrency: how to dispatch

`MAX_NUM_SEQS=16` costs no KV cache — that pool is preallocated from
`GPU_MEM_UTIL`, not from the sequence count (5 074 332 tokens at 16 vs 5 081 419
at 2). It does **not** follow that concurrency is free: the KV pool is
preallocated, so its size cannot reflect what the vision encoder and the
multimodal cache spend, and those come out of the same unified host pool that
hard-locks the machine when exhausted. What is free is running one plant's *warm*
shards 8-wide, where no new images are processed.

**How** you dispatch a plant's 9 shards matters more than the setting, and the
obvious approach is the worst one. Measured on cold plants, fresh container each
time (25–26 photos for the first three rows, 22–23 for the last two):

| pattern | per-plant | 2400 requests |
|---|---|---|
| all 9 serial | 161.6 s | 12.0 h |
| **all 9 concurrent from cold** | **402.1 s** | **29.8 h** |
| cold `shard_01` alone, then 2-9 at conc 8 | 101.6 s | 7.5 h |
| **+ overlap the next plant's cold prefill (2 plants in flight)** | **75.1 s** | **5.6 h** |
| 3 plants in flight | 71.0 s | 5.3 h — **do not use**, see below |

> **Two plants in flight, not three.** At three, host `MemAvailable` bottoms out
> at **7.37 GiB** against an 8 GiB stop line — and exhausting the unified pool is
> the failure that hard-locks this machine. The third plant buys 5 %. All plants
> keep their cached prefix at both depths (97.2–97.5 % per plant, measured per
> plant rather than in aggregate), so depth is limited by memory, not by cache
> eviction. The 75.1 s / 5.6 h row replaces an earlier 83.6 s / 6.2 h figure
> that came from a single run with no memory or prefix-hit data.

> **Treat the memory floors as soft.** They were measured on freshly restarted
> containers, which also start with an empty `--mm-processor-cache-gb` (default
> **4 GiB**); once that fills, idle `MemAvailable` falls from ~12.9 GiB to
> ~7.8 GiB and two plants in flight bottom out at **6.32 GiB** — still with no
> preemptions and prefix hits intact. And `MemAvailable` is a whole-machine
> number that included the tooling doing the measuring, so it understates the
> real headroom of an unattended run. **The settled configuration is two plants
> in flight with the cache left at 4 GiB**; that combination completes cleanly
> both cold and saturated, which is the property that matters.
> `--mm-processor-cache-gb 2` is the lever if headroom is ever needed (two
> plants need only ~1.4 GB of the 4 GiB) — untested. Never raise `GPU_MEM_UTIL`
> to buy headroom: that is the known route to a hard lock.

> **Never fan out a cold plant.** Releasing all 9 shards at once drops the
> prefix-cache hit rate to **9.5 %** — none of them finds the shared prefix
> because none has finished writing it, so all nine re-prefill the same ~31 k
> tokens and re-run the vision encoder over the same images nine times. That is
> **2.5x slower than plain serial**.

The rule, for any client driving this server:

1. Send `shard_01` **alone** and wait for it — this writes the shared
   system+images prefix.
2. Release shards 2-9 at **concurrency 8**. They hit ~98 % of the prefix and the
   set finishes 3.4x faster than serial (74.8 s → 21.4 s measured on a warm
   prefix; aggregate decode 37 → 136 tok/s).
3. Overlap the *next* plant's cold `shard_01` with the current plant's warm set,
   and keep **two plants in flight**. Measured 75.1 s per plant; three in flight
   is 5 % faster and lands past the memory stop line.
4. Bound the output. `max_tokens=2048` is 5.4x the p90 of 381 tokens measured
   at the shipped temperature 0.5 (3.4x the 607 measured at temperature 1.0),
   so it cannot truncate a real answer, and it caps a runaway at ~50 s instead
   of ~190 s.

Individual requests do get slower under concurrency (`shard_02`: 14.9 s at conc 1,
21.4 s at conc 8). That is the expected batching trade — throughput is what a
2400-request run cares about.

## Verifying and benchmarking

`smoke.py` runs the acceptance suite against **real** shard schemas, photos and
system prompt — all opened read-only — and exits non-zero on any failure, printing
the offending response and the `jsonschema` error path.

```bash
set -a; source .env; set +a
python smoke.py                 # A B C D1 D2 E H
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
| H | same plant + shard sent twice at temperature 0.7 → the two outputs must differ. Guards against a seed being pinned somewhere or sampling collapsing to greedy, either of which would silently zero the variance the consistency study measures. Never auto-retried. |

By default `smoke.py` picks the **largest** shard schema, the one most likely to
trip a grammar limit.

`bench.sh` measures the shape pxGPT actually sends — a full photo set plus one
real shard schema — rather than `vllm bench serve`'s synthetic
short-prompt/long-output dataset:

```bash
./down.sh && ./up.sh && ./bench.sh     # restart first, see below
```

Reference figures at the pinned budget of **1120** and the pinned MoE backend
(32 photos, thinking off, fresh container): **37 349** prompt tokens,
**39.2 tok/s** decode, **12.8 s** with a warm prefix cache, **75.4 s** cold.
Sending a plant's full 9-shard set serially measures **162.2 s per plant**, i.e.
**~12.0 h** for 2400 requests — falling to **~5.6 h** with the dispatch pattern
in [Concurrency](#concurrency-how-to-dispatch) below. `bench.sh` itself is serial
by design, so its numbers are the conservative baseline, and none of these
figures includes runaway generation.

> Ordering matters as much as the budget: 9 shards per plant share one image
> prefix, so dispatching **plant-major** keeps 8 of 9 requests on the cache. Going
> shard-major would pay the cold prefill every time and push the same run toward
> **48.5 h**.

Two samplers exist for watching a run rather than timing it. Both write TSV, so
the low-water mark survives the run:

```bash
./sample_mem.sh mem.tsv &          # /proc/meminfo every 0.2 s
./sample_metrics.sh metrics.tsv &  # selected /metrics series every 1 s
```

`MemAvailable` is the safety number — the unified pool is shared with the OS and
exhausting it hard-locks the box. `vllm:kv_cache_usage_perc` is **not**: the KV
pool is preallocated, so it cannot exceed 100 % and saturating it causes
preemption, not a crash. Watch `vllm:num_preemptions_total` for that.
`sample_metrics.sh` checks every metric name against `/metrics` before it starts,
because vLLM renames series between versions and a stale name would silently
produce an empty column.

> `MemAvailable` counts the **whole machine**, including whatever is driving the
> workload. Run these samplers from as quiet a host as you can, and read the
> numbers as lower bounds on the server's headroom rather than as measurements
> of it. The RUNBOOK's figures were collected from an interactive session on the
> same box and are caveated accordingly.

> **Always restart before benchmarking.** Prefix caching is on by default and its
> cache lives as long as the container, so a second `bench.sh` against the same
> server measures cache hits and reports a meaninglessly low TTFT.
> `BENCH_COLD_LINE` must also name a line that server has not served yet.

---

## The vLLM version constraint (background)

**Not every vLLM build can load this checkpoint.** `pull.sh` handles this for
you — it tries the tested images best-first and keeps the first that actually
serves the model, so you only need this section if it fails or if you are
choosing an image by hand on other hardware.

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

## Troubleshooting

**`.env: line N: syntax error near unexpected token 'newline'`** — a value in
`.env` is not valid shell. Almost always an unquoted `<` or `>`, which bash reads
as a redirection: `MEDIA_ROOT=<FILL>` or `MEDIA_ROOT=<your path>`. Put the real
path in, unquoted if it has no spaces, quoted if it does. Current `pull.sh` and
`up.sh` catch this before sourcing and print the offending line; if you see the
raw bash error instead, your scripts predate that check.

**`line N: <something>: command not found` while loading `.env`** — an unquoted
space, e.g. `MEDIA_ROOT=/home/me/my photos`. Quote it:
`MEDIA_ROOT="/home/me/my photos"`.

**`KeyError: '...input_global_scale'` at startup** — image too old. See
[the vLLM version constraint](#the-vllm-version-constraint-background).

**`vllm: error: unrecognized arguments: serve unsloth/...`** — the
`vllm/vllm-openai` images already have `ENTRYPOINT ["vllm","serve"]`, so the
subcommand must not be repeated; the NGC image needs it spelled out. `up.sh` and
`pull.sh` inspect `.Config.Entrypoint` and adapt, so this only bites hand-rolled
`docker run` commands.

**A request takes ~190 s and returns 8192 completion tokens** — runaway
generation. The grammar keeps output well-formed but nothing bounds length, so the
model can pad `rationale` strings indefinitely. Set a per-request timeout, cap
`max_tokens` at 2048 (5.4x the p90 of 381 measured at temperature 0.5; the
607 figure was taken at temperature 1.0 — see the sampling note below), and treat
`finish_reason == "length"` as a **failed** shard, not a partial result. Rate:
0 in 30 requests across 10 plants, which bounds it at ~10 % rather than
measuring it — do not assume it will not happen.

**First request is ~11 s slower than the rest** — JIT codegen. Send a
`max_tokens=3` warm-up before timing anything.

**`file://` URL rejected or not found** — the path must sit under `MEDIA_ROOT`
*and* be identical inside the container. Check `--allowed-local-media-path` in
`docker inspect --format '{{join .Args " "}}' pxgpt-vllm`.

**Server never becomes healthy** — `up.sh` prints the last 50 log lines on
timeout or exit. Cold start is ~4 minutes (~100 s weights, ~41 s
`torch.compile`, ~74 s engine init); allow for it before assuming a hang.

**`--moe-backend: invalid choice: 'VLLM_CUTLASS'`** — the CLI and the startup log
use different names for the same kernel. The CLI takes lowercase
(`auto`, `cutlass`, `flashinfer_cutlass`, `marlin`, …); `cutlass` is the one the
log prints as `VLLM_CUTLASS`. That is what `MOE_BACKEND` in `.env` must hold.

**Startup log shows a MoE backend you did not expect** — check `MOE_BACKEND` is
set and reaching the container (`docker inspect --format '{{join .Args " "}}'
pxgpt-vllm`). Unpinned, vLLM walks a priority list and takes the first supported
entry; five consecutive boots all picked `FLASHINFER_CUTLASS`, so an unexplained
change means something else moved. Never force Marlin: the model card measures it
~2x slower.

---

## What is deliberately not configured

`--kv-cache-dtype`, `--quantization`, `--linear-backend`,
`--enable-prefix-caching`, `--tensor-parallel-size`, `--trust-remote-code`. Each
is either auto-detected, already the default, meaningless on a single GB10, or a
known performance trap on Spark. [RUNBOOK.md](RUNBOOK.md) records the reasoning
per flag. If you find yourself needing one, record why there too.

The scripts also never write to `SHARD_DIR` or `MEDIA_ROOT`. The Stage 3 shard
sets are frozen, `chmod -w`, checksummed and under human evaluation; if any step
appears to need write access there, stop and report it rather than working around
it.
