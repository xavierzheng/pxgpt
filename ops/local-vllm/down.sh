#!/usr/bin/env bash
# Stop and remove the container. The HF weight cache is left alone.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] || { echo "No .env -- nothing to do." >&2; exit 0; }
# shellcheck disable=SC1091
source .env

if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "$CONTAINER_NAME does not exist -- nothing to do."
  exit 0
fi

# --restart unless-stopped means stop must come before rm.
docker stop "$CONTAINER_NAME" >/dev/null
docker rm "$CONTAINER_NAME" >/dev/null
echo "Removed $CONTAINER_NAME (HF cache kept)."
