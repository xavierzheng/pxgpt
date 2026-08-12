#!/usr/bin/env bash
# Find an image that actually serves this checkpoint, then pin it by digest.
#
# Tries the candidate images in order and keeps the FIRST one that loads the
# model and answers /health. Every failure is appended to RUNBOOK.md with the
# last 50 log lines, so the rejected candidates stay on the record.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] || { echo "No .env -- run: cp env.example .env" >&2; exit 1; }
# shellcheck disable=SC1091
source .env

RUNBOOK="$PWD/RUNBOOK.md"
PROBE="${CONTAINER_NAME}-probe"

CANDIDATES=(
  "vllm/vllm-openai:gemma4-cu130"
  "nvcr.io/nvidia/vllm:26.07-py3"   # latest NGC tag as of 2026-08-12
  "vllm/vllm-openai:cu130-nightly"
)

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set. Export it before running (the repo may be gated)." >&2
  exit 1
fi

# --- weights ---------------------------------------------------------------
echo "==> Downloading $MODEL_REPO"
hf download "$MODEL_REPO" --revision main
# Resolve 'main' to the immutable commit sha and pin it in .env.
SNAP_DIR="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${MODEL_REPO//\//--}/snapshots"
REVISION="$(ls -1 "$SNAP_DIR" | head -1)"
[[ -n "$REVISION" ]] || { echo "Could not resolve a commit sha under $SNAP_DIR" >&2; exit 1; }
echo "==> MODEL_REVISION=$REVISION"
sed -i "s|^MODEL_REVISION=.*|MODEL_REVISION=$REVISION|" .env

cleanup_probe() { docker rm -f "$PROBE" >/dev/null 2>&1 || true; }
trap cleanup_probe EXIT

record_failure() {  # $1=image  $2=reason
  { echo ""
    echo "### Rejected: \`$1\`"
    echo ""
    echo "Reason: $2"
    echo ""
    echo '```'
    docker logs "$PROBE" 2>&1 | tail -50 || echo "(no logs -- container never started)"
    echo '```'
  } >> "$RUNBOOK"
}

# --- candidates ------------------------------------------------------------
for IMAGE in "${CANDIDATES[@]}"; do
  echo ""
  echo "==> Trying $IMAGE"
  cleanup_probe

  if ! docker pull "$IMAGE"; then
    record_failure "$IMAGE" "docker pull failed"
    continue
  fi

  # vllm/vllm-openai images already have ENTRYPOINT ["vllm","serve"]; the NGC
  # image needs the subcommand spelled out. Inspect rather than assume.
  ENTRYPOINT="$(docker inspect --format '{{json .Config.Entrypoint}}' "$IMAGE")"
  SUBCMD=()
  [[ "$ENTRYPOINT" != *'"serve"'* ]] && SUBCMD=(vllm serve)

  docker run -d --name "$PROBE" --ipc=host --gpus all -p "$PORT:8000" \
    -e HF_TOKEN="$HF_TOKEN" \
    -v "$HF_HOME:/root/.cache/huggingface" \
    -v "$MEDIA_ROOT:$MEDIA_ROOT:ro" \
    "$IMAGE" \
    "${SUBCMD[@]}" "$MODEL_REPO" \
      --revision "$REVISION" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --max-model-len "$MAX_MODEL_LEN" \
      --max-num-seqs "$MAX_NUM_SEQS" \
      --gpu-memory-utilization "$GPU_MEM_UTIL" \
      --allowed-local-media-path "$MEDIA_ROOT" >/dev/null

  # First load includes weight load + JIT codegen; allow 20 minutes.
  echo "    waiting for /health (up to 20 min)..."
  OK=0
  for _ in $(seq 1 240); do
    if ! docker ps --format '{{.Names}}' | grep -qx "$PROBE"; then
      break  # container died; fall through to the failure path
    fi
    if curl -fsS -m 5 "http://localhost:$PORT/health" >/dev/null 2>&1; then
      OK=1; break
    fi
    sleep 5
  done

  if [[ $OK -eq 1 ]]; then
    DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "$IMAGE")"
    echo "==> $IMAGE serves the model. Pinning $DIGEST"
    sed -i "s|^VLLM_IMAGE=.*|VLLM_IMAGE=$DIGEST|" .env
    cleanup_probe
    echo "==> Done. Start the real server with ./up.sh"
    exit 0
  fi

  record_failure "$IMAGE" "model never became healthy within 20 min"
  echo "    FAILED -- see RUNBOOK.md"
done

echo "All candidate images failed. See RUNBOOK.md for the logs." >&2
exit 1
