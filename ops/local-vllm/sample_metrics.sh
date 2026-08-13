#!/usr/bin/env bash
# Sample the server's own scheduler counters into a TSV while a workload runs.
#
#     ./sample_metrics.sh out.tsv &
#     ...run the workload...
#     kill %1
#
# 1 s, not 0.2 s. /metrics is a full Prometheus dump (~100 series); scraping it
# five times a second would put measurable load on the very server being
# measured. Scheduler pressure is a per-second phenomenon anyway. Host memory is
# the sub-second one -- that is sample_mem.sh's job.
#
# How to read the output:
#   preemptions      the real scheduling-pressure signal. A monotonic counter;
#                    any increase means the scheduler ran out of KV blocks and
#                    threw work away to make room. > 0 is the stop condition.
#   kv_cache_usage   NOT a memory-safety signal. The KV pool is preallocated
#                    from --gpu-memory-utilization, so this fraction cannot
#                    exceed 1.0 and cannot exhaust the machine; it saturates
#                    into preemption instead. Read MemAvailable for safety.
#   prefix_cache_*   cumulative queries/hits over the container's whole life.
#                    Useful for "is the cache empty before I start" (queries
#                    must read 0) and for a run-level hit rate -- but under
#                    concurrency a delta cannot be attributed to one request,
#                    so per-request hits come from
#                    usage.prompt_tokens_details.cached_tokens instead.
#   mm_cache_*       the same, for the vision encoder's output cache.
#
# The metric names below were read off this build's live /metrics, not copied
# from vLLM docs: the names do move between versions (cache usage used to be
# vllm:gpu_cache_usage_perc; here it is vllm:kv_cache_usage_perc). A stale name
# would silently produce an empty column rather than an error, so the script
# refuses to start if any of them is missing.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:?usage: sample_metrics.sh OUT.tsv [INTERVAL_SECONDS]}"
INTERVAL="${2:-1}"
PORT="${PORT:-8000}"
URL="http://localhost:$PORT/metrics"

METRICS=(
  vllm:num_requests_running
  vllm:num_requests_waiting
  vllm:num_preemptions_total
  vllm:prefix_cache_queries_total
  vllm:prefix_cache_hits_total
  vllm:mm_cache_queries_total
  vllm:mm_cache_hits_total
  vllm:kv_cache_usage_perc
)

SNAPSHOT="$(curl -fsS -m 10 "$URL")" || { echo "cannot reach $URL" >&2; exit 1; }
missing=()
for m in "${METRICS[@]}"; do
  grep -q "^${m}[ {]" <<<"$SNAPSHOT" || missing+=("$m")
done
if (( ${#missing[@]} )); then
  echo "these metric names are not in $URL: ${missing[*]}" >&2
  echo "vLLM renames metrics between versions -- run" >&2
  echo "  curl -s $URL | grep -iE 'cache|preempt|running|waiting'" >&2
  echo "and update METRICS in this script rather than letting the column go blank." >&2
  exit 1
fi

printf 'epoch\t%s\n' "$(IFS=$'\t'; echo "${METRICS[*]}")" > "$OUT"
echo "sampling $URL every ${INTERVAL}s -> $OUT (ctrl-c or kill to stop)" >&2

while :; do
  now="$(date +%s.%N)"
  body="$(curl -fsS -m 10 "$URL" 2>/dev/null || true)"
  row="$now"
  for m in "${METRICS[@]}"; do
    # Take the last field of the first matching sample line; empty scrape or a
    # vanished series writes NA rather than shifting every later column left.
    v="$(grep -m1 "^${m}[ {]" <<<"$body" | awk '{print $NF}')"
    row+=$'\t'"${v:-NA}"
  done
  printf '%s\n' "$row" >> "$OUT"
  sleep "$INTERVAL"
done
