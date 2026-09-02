#!/usr/bin/env bash
# Start the pinned vLLM server. Idempotent: a running container is left alone.
set -euo pipefail
cd "$(dirname "$0")"

# --- external tools --------------------------------------------------------
missing=()
command -v docker >/dev/null 2>&1 || missing+=("docker")
command -v curl   >/dev/null 2>&1 || missing+=("curl")
if (( ${#missing[@]} )); then
  echo "Missing required command(s): ${missing[*]}" >&2
  echo "Install them and re-run. docker must work without sudo:" >&2
  echo "    docker run --rm hello-world" >&2
  exit 1
fi

# `command -v docker` only proves the binary exists. It says nothing about
# whether this user can reach the daemon, and that is a separate, very common
# failure: the socket is root-owned. Without this check pull.sh sails past the
# gate, downloads 17 GB of weights, and only then dies at `docker pull`.
if ! docker info >/dev/null 2>&1; then
  echo "docker is installed, but this user cannot talk to the daemon:" >&2
  # `|| true`: the pipeline is expected to fail, and set -e + pipefail would
  # otherwise kill the script before the explanation below is printed.
  { docker info 2>&1 | head -3 | sed 's/^/    /' >&2; } || true
  # Quoted heredoc: the text contains backticks and $VAR, none of which should
  # be expanded or executed here.
  cat >&2 <<'MSG'

Three ways out. This is a host decision, not a pxGPT one, so pick deliberately
-- the first is not merely a convenience:

  1. Add yourself to the "docker" group. Easiest, and it grants this account
     what amounts to root on the host: group members can mount any path into a
     container. Do not do this on a shared machine without saying so.

         sudo usermod -aG docker "$USER"

     Then LOG OUT AND BACK IN. A running shell keeps its old group list, so
     without that it looks as though the change did nothing. Running
     "newgrp docker" fixes the current shell only.

  2. Rootless Docker. Keeps the daemon away from root; needs its own setup for
     GPU passthrough. https://docs.docker.com/engine/security/rootless/

  3. Run these scripts under sudo. No permission change, but mind the trap:
     sudo resets HOME, so HF_HOME lands in root's home, the weights download
     somewhere else, and up.sh will not find them. Pass it through:

         sudo -E HF_HOME="$HF_HOME" ./pull.sh

Verify whichever you chose with:  docker run --rm hello-world

MSG
  exit 1
fi

# --- .env: load it, and say something useful when it is wrong ---------------
if [[ ! -f .env ]]; then
  cat >&2 <<'MSG'
No .env in ops/local-vllm/. Set it up with:

    cp env.example .env
    $EDITOR .env        # set MEDIA_ROOT -- the only line you must edit
    export HF_TOKEN=hf_...

MSG
  exit 1
fi

env_help() {
  cat >&2 <<'MSG'

Every line in .env must be NAME=value. Quote any value containing a space or
any of  < > | & ( ) ; # *  -- a leftover placeholder is the usual cause, since
bash reads < as a redirection:

    MEDIA_ROOT=<FILL>              wrong -- syntax error
    MEDIA_ROOT=/home/me/my pics    wrong -- unquoted space
    MEDIA_ROOT=/home/me/pxgpt      right
    MEDIA_ROOT="/home/me/my pics"  right

Compare yours against env.example.
MSG
}

# 1. Shape: anything that is not a comment, blank, or NAME=value.
env_shape="$(grep -nvE '^[[:space:]]*(#|$)|^[A-Za-z_][A-Za-z0-9_]*=' .env || true)"
if [[ -n "$env_shape" ]]; then
  echo "These lines in .env are not NAME=value:" >&2
  printf '    %s\n' "$env_shape" >&2
  env_help
  exit 1
fi

# 2. Load it in a subshell first, so a bad value reports itself here instead of
#    killing this script with a bare bash error and a line number. Catches both
#    syntax errors and the unquoted-space case, which is syntactically valid and
#    only fails when it runs.
if ! env_err="$(bash -c 'set -e; source .env' 2>&1 >/dev/null)"; then
  echo ".env could not be loaded. bash reports:" >&2
  printf '    %s\n' "$env_err" >&2
  env_help
  exit 1
fi
# shellcheck disable=SC1091
source .env

# MEDIA_ROOT is the only value a user has to supply by hand.
if [[ -z "${MEDIA_ROOT:-}" ]]; then
  cat >&2 <<'MSG'
MEDIA_ROOT is empty in .env. It is the one value you must set.

It is the absolute path to your photo root -- bind-mounted read-only at the
SAME path inside the container and passed as --allowed-local-media-path, so the
file:// URLs pxgpt sends resolve on both sides. It is a PREFIX, so point it at
the project root and every dataset under it resolves:

    MEDIA_ROOT=/home/xavier/project/pxgpt

MSG
  exit 1
fi
if [[ ! -d "$MEDIA_ROOT" ]]; then
  echo "MEDIA_ROOT is not a directory: $MEDIA_ROOT" >&2
  echo "It must be an absolute path that exists on this machine (no ~)." >&2
  exit 1
fi

# Written by pull.sh. Empty means pull.sh has not run (or did not finish).
for v in VLLM_IMAGE MODEL_REVISION; do
  [[ -n "${!v-}" ]] || { echo "$v is empty in .env -- run ./pull.sh first." >&2; exit 1; }
done
# Pinned settings that ship filled in; empty means .env was edited or truncated.
for v in IMAGE_TOKEN_BUDGET MOE_BACKEND TEMPERATURE TOP_P TOP_K; do
  [[ -n "${!v-}" ]] || { echo "$v is empty in .env -- restore it from env.example." >&2; exit 1; }
done
[[ "$VLLM_IMAGE" == *@sha256:* ]] || { echo "VLLM_IMAGE must be pinned by digest, got: $VLLM_IMAGE" >&2; exit 1; }

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "$CONTAINER_NAME is already running -- nothing to do."
  exit 0
fi
# A stopped container of the same name would block docker run.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

echo "==> Starting $CONTAINER_NAME"
echo "    image    : $VLLM_IMAGE"
echo "    model    : $MODEL_REPO @ $MODEL_REVISION"
echo "    media    : $MEDIA_ROOT (ro, same path inside)"
echo "    img budget: $IMAGE_TOKEN_BUDGET tokens/image"
echo "    moe      : $MOE_BACKEND (pinned; CLI spelling, log says VLLM_CUTLASS)"
echo "    sampling : temperature=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K (no seed)"

# Pinned decisions (RUNBOOK.md has the evidence):
#   - visual token budget goes on --mm-processor-kwargs so every request shares
#     one budget; leaving it per-request would break comparability with the
#     Anthropic / OpenAI backends.
#   - thinking is OFF at the request layer (ENABLE_THINKING=false). The chat
#     template already defaults enable_thinking=false and pre-fills an empty
#     <|channel>thought\n<channel|> block, so a response_format grammar binds to
#     the answer from its first token. --reasoning-parser gemma4 is still passed
#     so the thinking-on path (acceptance D2) stays reproducible against this
#     same server; it is inert while thinking is off.
#   - --moe-backend is PINNED so the NVFP4 expert kernel is stated rather than
#     inherited from vLLM's priority list, which can be reordered by any image
#     bump. A different MoE kernel means a different floating-point accumulation
#     order, and the consistency study runs the same plant repeatedly at
#     temperature > 0 -- kernel drift there would be indistinguishable from
#     model instability. NOTE the CLI takes lowercase names: `cutlass` is what
#     the startup log calls VLLM_CUTLASS.
#   - --override-generation-config states the sampling parameters that were
#     previously inherited, unrecorded, from the checkpoint's
#     generation_config.json. It MERGES into that file rather than replacing it
#     (verified, see RUNBOOK). smoke.py and bench.sh send the same three values
#     per request from the same .env, so the two layers cannot disagree.
#     No seed is passed at either layer -- see env.example.
#   - --enable-prompt-tokens-details makes the server report
#     usage.prompt_tokens_details.cached_tokens per response. Reporting only:
#     it changes no kernel, no sampling and no memory. It is the only way to
#     get a PER-REQUEST prefix-cache hit count, and therefore the only way to
#     ask "did both plants' warm sets hit, or did one evict the other" while
#     several plants are in flight -- a /metrics delta cannot be attributed to
#     a request once requests overlap.
# Deliberately NOT set: --kv-cache-dtype, --quantization, --linear-backend,
# --enable-prefix-caching, --tensor-parallel-size, --trust-remote-code.
# See RUNBOOK.md for why each is omitted.
#
# vllm/vllm-openai images have ENTRYPOINT ["vllm","serve"], so the model must be
# the first argument; the NGC image execs its args verbatim and needs the
# subcommand spelled out. Inspect rather than assume.
ENTRYPOINT="$(docker inspect --format '{{json .Config.Entrypoint}}' "$VLLM_IMAGE")"
SUBCMD=()
if [[ "$ENTRYPOINT" != *'"serve"'* ]]; then
  SUBCMD=(vllm serve)
fi

docker run -d --name "$CONTAINER_NAME" --ipc=host --restart unless-stopped \
  --gpus all -p "$PORT:8000" \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$MEDIA_ROOT:$MEDIA_ROOT:ro" \
  "$VLLM_IMAGE" \
  "${SUBCMD[@]}" "$MODEL_REPO" \
    --revision "$MODEL_REVISION" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --allowed-local-media-path "$MEDIA_ROOT" \
    --mm-processor-kwargs "{\"max_soft_tokens\": $IMAGE_TOKEN_BUDGET}" \
    --moe-backend "$MOE_BACKEND" \
    --override-generation-config \
      "{\"temperature\": $TEMPERATURE, \"top_p\": $TOP_P, \"top_k\": $TOP_K}" \
    --enable-prompt-tokens-details \
    --reasoning-parser gemma4 >/dev/null

echo "    waiting for /health (weight load + JIT codegen, up to 20 min)..."
for _ in $(seq 1 240); do
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container exited. Last 50 lines:" >&2
    docker logs "$CONTAINER_NAME" 2>&1 | tail -50 >&2
    exit 1
  fi
  if curl -fsS -m 5 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "==> healthy on http://localhost:$PORT/v1"
    exit 0
  fi
  sleep 5
done

echo "Timed out waiting for /health. Last 50 lines:" >&2
docker logs "$CONTAINER_NAME" 2>&1 | tail -50 >&2
exit 1
