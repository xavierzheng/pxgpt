#!/usr/bin/env bash
# Start the pinned vLLM server. Idempotent: a running container is left alone.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] || { echo "No .env -- run: cp env.example .env && ./pull.sh" >&2; exit 1; }
# shellcheck disable=SC1091
source .env

for v in VLLM_IMAGE MODEL_REVISION MEDIA_ROOT IMAGE_TOKEN_BUDGET; do
  [[ "${!v}" == "<FILL>" || -z "${!v}" ]] && { echo "$v is unset in .env (run ./pull.sh)" >&2; exit 1; }
done
[[ "$VLLM_IMAGE" == *@sha256:* ]] || { echo "VLLM_IMAGE must be pinned by digest, got: $VLLM_IMAGE" >&2; exit 1; }
[[ -d "$MEDIA_ROOT" ]] || { echo "MEDIA_ROOT does not exist: $MEDIA_ROOT" >&2; exit 1; }

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
# Deliberately NOT set: --kv-cache-dtype, --quantization, --moe-backend,
# --linear-backend, --enable-prefix-caching, --tensor-parallel-size,
# --trust-remote-code. See RUNBOOK.md for why each is omitted.
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
