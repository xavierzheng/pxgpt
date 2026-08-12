#!/usr/bin/env bash
# Benchmark the request shape pxGPT actually sends: one real plant line's full
# photo set + one real shard schema -> short structured JSON.
#
# Deliberately NOT `vllm bench serve`: its random dataset is a synthetic
# short-prompt/long-output workload, the opposite of ours (very long multimodal
# prefill, short constrained output), so its numbers would not transfer.
#
# 3 timed runs, median reported. The first request after startup triggers JIT
# codegen (tens of seconds), so a max_tokens=3 warm-up runs first and is
# reported separately rather than polluting the median.
#
# IMPORTANT: prefix caching is on by default and its cache lives as long as the
# container, so a second bench.sh run against the same server measures cache
# hits, not prefills. For honest cold numbers restart first:
#     ./down.sh && ./up.sh && ./bench.sh
# BENCH_COLD_LINE must also name a line this server has not seen yet.
set -euo pipefail
cd "$(dirname "$0")"

[[ -f .env ]] || { echo "No .env -- run: cp env.example .env && ./pull.sh" >&2; exit 1; }
# shellcheck disable=SC1091
set -a; source .env; set +a

: "${RUNS:=3}"
: "${TOTAL_REQUESTS:=2400}"
PY="${PYTHON:-python3}"

exec "$PY" - "$@" <<'PYEOF'
import json
import os
import statistics
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, os.getcwd())
from smoke import load_shard, line_photos, image_parts, response_format  # noqa: E402

BASE = f"http://localhost:{os.getenv('PORT', '8000')}/v1"
MODEL = os.getenv("SERVED_MODEL_NAME", "gemma4-26b-a4b-nvfp4")
RUNS = int(os.getenv("RUNS", "3"))
TOTAL = int(os.getenv("TOTAL_REQUESTS", "2400"))
BUDGET = int(os.getenv("IMAGE_TOKEN_BUDGET", "280"))
THINKING = os.getenv("ENABLE_THINKING", "false").lower() in ("1", "true", "yes")

client = OpenAI(base_url=BASE, api_key="local", max_retries=0, timeout=3600)

shard_id, schema, shard_prompt = load_shard(os.environ["SHARD_DIR"], os.getenv("SHARD_ID"))
system_prompt = Path(os.environ["SYSTEM_PROMPT"]).read_text(encoding="utf-8")
line = os.getenv("BENCH_LINE", "s0019")
photos = line_photos(os.environ["MEDIA_ROOT"], line)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": image_parts(photos) + [{"type": "text", "text": shard_prompt}]},
]
extra = {
    "chat_template_kwargs": {"enable_thinking": THINKING},
    "mm_processor_kwargs": {"max_soft_tokens": BUDGET},
}

print(f"model        : {MODEL}")
print(f"line         : {line} ({len(photos)} photos)")
print(f"shard        : {shard_id}")
print(f"img budget   : {BUDGET} tokens/image")
print(f"thinking     : {THINKING}")
print(f"runs         : {RUNS}")
print("")


def timed_run(max_tokens, structured):
    """One streamed request. Reads the module-level `messages`. Returns (ttft, total, prompt_tok, completion_tok)."""
    kwargs = dict(model=MODEL, messages=messages, max_tokens=max_tokens,
                  stream=True, stream_options={"include_usage": True},
                  extra_body=extra)
    if structured:
        kwargs["response_format"] = response_format(schema, shard_id)
    t0 = time.time()
    ttft = None
    usage = None
    for chunk in client.chat.completions.create(**kwargs):
        if ttft is None and chunk.choices and chunk.choices[0].delta.content:
            ttft = time.time() - t0
        if getattr(chunk, "usage", None):
            usage = chunk.usage        # final usage-only chunk
    total = time.time() - t0
    if usage is None:
        sys.exit("Server returned no usage chunk; stream_options.include_usage unsupported?")
    return ttft, total, usage.prompt_tokens, usage.completion_tokens


# --- warm-up: absorbs JIT codegen, reported separately ---------------------
t0 = time.time()
_, warm_total, warm_prompt, _ = timed_run(max_tokens=3, structured=False)
print(f"warm-up (max_tokens=3): {warm_total:.1f}s  [JIT codegen + prefill of {warm_prompt} tokens]")
print("")

rows = []
for i in range(RUNS):
    ttft, total, pt, ct = timed_run(max_tokens=8192, structured=True)
    decode_tps = (ct - 1) / (total - ttft) if ttft and total > ttft and ct > 1 else float("nan")
    rows.append((ttft, total, pt, ct, decode_tps))
    print(f"run {i + 1}: prompt={pt} completion={ct} ttft={ttft:.2f}s "
          f"total={total:.1f}s decode={decode_tps:.1f} tok/s")

med = lambda idx: statistics.median(r[idx] for r in rows)  # noqa: E731
m_ttft, m_total, m_pt, m_ct, m_dec = (med(0), med(1), med(2), med(3), med(4))

print("")
print(f"===== median of {RUNS} =====")
print(f"  prompt tokens (incl. images) : {m_pt:.0f}")
print(f"  TTFT                         : {m_ttft:.2f} s")
print(f"  decode throughput            : {m_dec:.1f} tok/s")
print(f"  total latency                : {m_total:.1f} s")
print(f"  completion tokens            : {m_ct:.0f}")
print("")

# Runs 2..N above reuse run 1's image prefix, so their TTFT is a prefix-cache
# hit (V1 enables prefix caching by default) and understates a cold prefill.
# Measure one genuinely cold line so the projection is not built on cache hits.
cold_line = os.getenv("BENCH_COLD_LINE", "s0044")
if cold_line != line:
    cold_photos = line_photos(os.environ["MEDIA_ROOT"], cold_line)
    cold_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_parts(cold_photos) +
         [{"type": "text", "text": shard_prompt}]},
    ]
    saved, messages = messages, cold_messages
    c_ttft, c_total, c_pt, c_ct = timed_run(max_tokens=8192, structured=True)
    messages = saved
    print(f"cold-prefill probe ({cold_line}, {len(cold_photos)} photos, unseen prefix):")
    print(f"  prompt={c_pt} ttft={c_ttft:.2f}s total={c_total:.1f}s completion={c_ct}")
    print("")
else:
    c_total = m_total
    print(f"(BENCH_COLD_LINE == BENCH_LINE; no separate cold probe)\n")

# pxGPT sends SHARDS_PER_PLANT requests per plant, all sharing the same image
# prefix: the first pays a cold prefill, the rest hit the prefix cache.
shards_per_plant = int(os.getenv("SHARDS_PER_PLANT", "9"))
plants = TOTAL / shards_per_plant
per_plant = c_total + (shards_per_plant - 1) * m_total
mixed_h = plants * per_plant / 3600

print(f"projected {TOTAL} requests (serial, max-num-seqs={os.getenv('MAX_NUM_SEQS', '2')}):")
print(f"  all-cold upper bound : {c_total * TOTAL / 3600:.1f} h  ({c_total:.1f} s x {TOTAL})")
print(f"  all-cached lower bnd : {m_total * TOTAL / 3600:.1f} h  ({m_total:.1f} s x {TOTAL})")
print(f"  realistic            : {mixed_h:.1f} h  "
      f"({plants:.0f} plants x [1 cold {c_total:.1f}s + {shards_per_plant - 1} cached {m_total:.1f}s])")
print("Serial estimates only -- concurrency above 1 is not measured here.")
PYEOF
