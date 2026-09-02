#!/usr/bin/env bash
# Find an image that actually serves this checkpoint, then pin it by digest.
#
# Tries the candidate images in order and keeps the FIRST one that loads the
# model and answers /health. Every failure is appended to RUNBOOK.md with the
# last 50 log lines, so the rejected candidates stay on the record.
set -euo pipefail
cd "$(dirname "$0")"

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

RUNBOOK="$PWD/RUNBOOK.md"
PROBE="${CONTAINER_NAME}-probe"

# Ordered best-tested first, so a fresh setup does not spend a 20+ GB download
# on a candidate this repo already knows fails. See README_vllm.md
# "The version constraint" for the tested verdict on each.
CANDIDATES=(
  "nvcr.io/nvidia/vllm:26.07-py3"    # vLLM 0.24.0 -- the pinned choice; loads it
  "vllm/vllm-openai:cu130-nightly"   # 0.19.2rc1  -- loads it, but a floating tag
  "vllm/vllm-openai:gemma4-cu130"    # 0.19.1     -- KNOWN BAD: KeyError on the
                                     # per-expert NVFP4 scales. Kept last only
                                     # because it is a floating tag that may be
                                     # rebuilt; do not expect it to work.
)

if [[ -z "${HF_TOKEN:-}" ]]; then
  cat >&2 <<'MSG'
HF_TOKEN is not set. Export it in your shell before running this:

    export HF_TOKEN=hf_...

Get one at https://huggingface.co/settings/tokens (read access is enough).
It is needed because the weights repo may be gated. Do NOT put it in .env --
that file is for server settings and this is a credential.
MSG
  exit 1
fi

# Validate the token before spending 17 GB of download on it. A rejected token
# otherwise surfaces partway through `hf download`, which looks like a network
# fault rather than a credential one. Offline is not an error here -- only an
# explicit rejection is.
if command -v curl >/dev/null 2>&1; then
  http="$(curl -s -o /dev/null -w '%{http_code}' -m 15 \
          -H "Authorization: Bearer $HF_TOKEN" \
          https://huggingface.co/api/whoami-v2 2>/dev/null || echo 000)"
  case "$http" in
    200) echo "==> HF_TOKEN accepted" ;;
    401|403)
      cat >&2 <<MSG
HF_TOKEN was rejected by Hugging Face (HTTP $http).

The token is set but not valid -- expired, revoked, or mistyped. Create a fresh
one at https://huggingface.co/settings/tokens (read access is enough) and
re-export it:

    export HF_TOKEN=hf_...

Stopping now rather than failing partway through a 17 GB download.
MSG
      exit 1 ;;
    000) echo "==> Could not reach huggingface.co to check the token; continuing." ;;
    *)   echo "==> Unexpected HTTP $http checking the token; continuing." ;;
  esac
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
