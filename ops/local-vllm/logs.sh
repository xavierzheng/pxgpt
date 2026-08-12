#!/usr/bin/env bash
# Follow the server log.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] || { echo "No .env -- nothing to do." >&2; exit 0; }
# shellcheck disable=SC1091
source .env

if ! docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "$CONTAINER_NAME does not exist -- nothing to follow."
  exit 0
fi

exec docker logs -f "$CONTAINER_NAME"
