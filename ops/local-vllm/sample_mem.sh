#!/usr/bin/env bash
# Sample host memory into a TSV while something else is running.
#
#     ./sample_mem.sh out.tsv &        # or: ./sample_mem.sh out.tsv 0.2
#     ...run the workload...
#     kill %1
#
# Writes, not draws. `watch free -g` only repaints a screen: the interesting
# number here is the LOWEST point MemAvailable ever reached, and a repainted
# screen cannot be queried afterwards.
#
# 0.2 s because the vision encoder's activation peak is sub-second. At 1 s a
# 32-image cold prefill can rise and fall between two samples, and the file
# would then claim there was plenty of headroom when there was not.
#
# MemAvailable is the real safety indicator on this box: the GB10's memory is
# one unified pool shared by the OS, the page cache and the GPU, and the known
# failure mode is exhausting *that*, which hard-locks the machine. It is not the
# same thing as vllm:kv_cache_usage_perc, which cannot exceed 100% and cannot
# hard-lock anything -- see sample_metrics.sh.
#
# Committed_AS rides along as the leading indicator: it counts what has been
# promised rather than what has been touched, so it moves before MemAvailable.
set -euo pipefail

OUT="${1:?usage: sample_mem.sh OUT.tsv [INTERVAL_SECONDS]}"
INTERVAL="${2:-0.2}"

printf 'epoch\tmem_available_kb\tcommitted_as_kb\n' > "$OUT"
echo "sampling /proc/meminfo every ${INTERVAL}s -> $OUT (ctrl-c or kill to stop)" >&2

while :; do
  # One read of /proc/meminfo, both fields out of it, so the two numbers
  # describe the same instant.
  awk -v now="$(date +%s.%N)" '
    /^MemAvailable:/ {avail = $2}
    /^Committed_AS:/ {committed = $2}
    END {printf "%s\t%s\t%s\n", now, avail, committed}
  ' /proc/meminfo >> "$OUT"
  sleep "$INTERVAL"
done
