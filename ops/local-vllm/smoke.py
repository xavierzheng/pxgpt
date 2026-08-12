#!/usr/bin/env python3
"""Acceptance checks A-E for the local vLLM server, against real pxGPT data.

Everything under --shard-dir and --media-root is opened read-only; the frozen
shard set is never written to. Run after ./up.sh:

    pip install -r requirements.txt
    set -a; source .env; set +a
    python smoke.py                # A B C D1 D2 E
    python smoke.py --tests D2     # D2 alone (server needs --reasoning-parser)

Exit 1 if any selected check fails, with the offending response and the
jsonschema error path printed.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jsonschema
from openai import OpenAI

ALL_TESTS = ["A", "B", "C", "D1", "D2", "E"]


def log(msg=""):
    print(msg, flush=True)


def preview(text, n=500):
    if text is None:
        return "<None>"
    text = str(text)
    return text[:n] + (f"... [{len(text)} chars total]" if len(text) > n else "")


def load_shard(shard_dir, shard_id):
    """Read one shard's schema + prompt. Read-only; picks the largest shard by
    default because that is the one most likely to trip a grammar limit."""
    manifest = json.loads((Path(shard_dir) / "shards_manifest.json").read_text(encoding="utf-8"))
    entries = manifest["shards"]
    if shard_id:
        entries = [e for e in entries if e["shard_id"] == shard_id]
        if not entries:
            sys.exit(f"No shard {shard_id!r} in {shard_dir}")
        entry = entries[0]
    else:
        entry = max(entries, key=lambda e: (Path(shard_dir) / e["schema_file"]).stat().st_size)
    schema = json.loads((Path(shard_dir) / entry["schema_file"]).read_text(encoding="utf-8"))
    prompt = (Path(shard_dir) / entry["prompt_file"]).read_text(encoding="utf-8")
    return entry["shard_id"], schema, prompt


def line_photos(media_root, line):
    d = Path(media_root) / line
    if not d.is_dir():
        sys.exit(f"No photo directory {d}")
    exts = {".jpg", ".jpeg", ".png"}
    photos = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    if not photos:
        sys.exit(f"No photos under {d}")
    return photos


def image_parts(paths):
    # Images before text, as the Gemma 4 card and pxgpt's own layout require.
    return [{"type": "image_url", "image_url": {"url": f"file://{p}"}} for p in paths]


def response_format(schema, name):
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


def thinking_body(enabled):
    return {"chat_template_kwargs": {"enable_thinking": bool(enabled)}}


def reasoning_of(message):
    """Pull the separated reasoning text off a response message.

    vLLM 0.24's OpenAI layer calls the field ``reasoning``; older builds and
    some other servers use ``reasoning_content``. The SDK does not model either
    one, so they arrive in model_extra.
    """
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning", "reasoning_content"):
        val = getattr(message, key, None) or extra.get(key)
        if val:
            return val
    return None


def check_json(content, schema, label):
    """json.loads + jsonschema.validate, reporting where it broke."""
    if not content:
        log(f"    FAIL: {label} returned empty content")
        return False
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        log(f"    FAIL: {label} content is not JSON: {e}")
        log(f"    content: {preview(content)}")
        return False
    try:
        jsonschema.validate(obj, schema)
    except jsonschema.ValidationError as e:
        log(f"    FAIL: {label} JSON violates the schema")
        log(f"      path   : {'/'.join(str(p) for p in e.absolute_path) or '<root>'}")
        log(f"      message: {e.message}")
        log(f"      content: {preview(content)}")
        return False
    log(f"    OK: {label} parsed and validated ({len(content)} chars)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=f"http://localhost:{os.getenv('PORT', '8000')}/v1")
    ap.add_argument("--model", default=os.getenv("SERVED_MODEL_NAME", "gemma4-26b-a4b-nvfp4"))
    ap.add_argument("--shard-dir", default=os.getenv("SHARD_DIR"), help="READ-ONLY")
    ap.add_argument("--shard", default=os.getenv("SHARD_ID"), help="default: largest schema")
    ap.add_argument("--system-prompt", default=os.getenv("SYSTEM_PROMPT"))
    ap.add_argument("--media-root", default=os.getenv("MEDIA_ROOT"), help="READ-ONLY")
    ap.add_argument("--line", default=os.getenv("BENCH_LINE", "s0019"))
    ap.add_argument("--max-model-len", type=int, default=int(os.getenv("MAX_MODEL_LEN", "65536")))
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--tests", default=",".join(ALL_TESTS))
    args = ap.parse_args()

    for req in ("shard_dir", "system_prompt", "media_root"):
        if not getattr(args, req):
            sys.exit(f"--{req.replace('_', '-')} is required (source .env first)")

    tests = [t.strip().upper().replace("D1", "D1").replace("D2", "D2")
             for t in args.tests.split(",") if t.strip()]
    unknown = set(tests) - set(ALL_TESTS)
    if unknown:
        sys.exit(f"Unknown tests: {sorted(unknown)}; valid: {ALL_TESTS}")

    client = OpenAI(base_url=args.base_url, api_key="local", max_retries=0, timeout=1800)

    shard_id, schema, shard_prompt = load_shard(args.shard_dir, args.shard)
    system_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    photos = line_photos(args.media_root, args.line)

    log(f"base_url  : {args.base_url}")
    log(f"model     : {args.model}")
    log(f"shard     : {shard_id} (from {args.shard_dir}, read-only)")
    log(f"line      : {args.line} ({len(photos)} photos)")
    log("")

    results = {}
    t0_all = time.time()

    # --- A: served model name -------------------------------------------
    if "A" in tests:
        log("[A] GET /v1/models")
        try:
            models = client.models.list()
            got = models.data[0].id
            ok = got == args.model
            log(f"    data[0].id = {got!r} (expected {args.model!r}) -> {'OK' if ok else 'FAIL'}")
            results["A"] = ok
        except Exception as e:
            log(f"    FAIL: {type(e).__name__}: {preview(e)}")
            results["A"] = False

    # --- B: plain text --------------------------------------------------
    if "B" in tests:
        log("[B] text-only chat completion")
        try:
            r = client.chat.completions.create(
                model=args.model, max_tokens=args.max_tokens,
                messages=[{"role": "user", "content": "Name three plant organs."}],
                extra_body=thinking_body(False))
            c = r.choices[0].message.content
            ok = bool(c and c.strip())
            log(f"    content: {preview(c, 200)}")
            log(f"    tokens: prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens} -> {'OK' if ok else 'FAIL'}")
            results["B"] = ok
        except Exception as e:
            log(f"    FAIL: {type(e).__name__}: {preview(e)}")
            results["B"] = False

    # --- C: one real photo over file:// ---------------------------------
    if "C" in tests:
        log(f"[C] single real photo via file:// ({photos[0].name})")
        try:
            r = client.chat.completions.create(
                model=args.model, max_tokens=args.max_tokens,
                messages=[{"role": "user", "content": image_parts(photos[:1]) +
                           [{"type": "text", "text": "Describe this photograph in one sentence."}]}],
                extra_body=thinking_body(False))
            c = r.choices[0].message.content
            ok = bool(c and c.strip())   # pipeline check only: no semantic assertion
            log(f"    content: {preview(c, 300)}")
            log(f"    tokens: prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens} -> {'OK' if ok else 'FAIL'}")
            results["C"] = ok
        except Exception as e:
            log(f"    FAIL: {type(e).__name__}: {preview(e)}")
            results["C"] = False

    # --- D1 / D2: guided decoding against the real shard schema ---------
    schema_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": image_parts(photos[:1]) + [{"type": "text", "text": shard_prompt}]},
    ]

    if "D1" in tests:
        log(f"[D1] thinking OFF + response_format strict json_schema ({shard_id})")
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=args.model, max_tokens=args.max_tokens, messages=schema_msgs,
                response_format=response_format(schema, shard_id),
                extra_body=thinking_body(False))
            dt = time.time() - t0
            m = r.choices[0].message
            log(f"    latency {dt:.1f}s | prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens}")
            log(f"    reasoning: {preview(reasoning_of(m), 120)}")
            results["D1"] = check_json(m.content, schema, "D1")
        except Exception as e:
            log(f"    FAIL: {type(e).__name__}: {preview(e)}")
            results["D1"] = False

    if "D2" in tests:
        log(f"[D2] thinking ON + reasoning parser + same schema ({shard_id})")
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=args.model, max_tokens=args.max_tokens, messages=schema_msgs,
                response_format=response_format(schema, shard_id),
                extra_body=thinking_body(True))
            dt = time.time() - t0
            m = r.choices[0].message
            rc = reasoning_of(m)
            log(f"    latency {dt:.1f}s | prompt={r.usage.prompt_tokens} completion={r.usage.completion_tokens}")
            log(f"    reasoning: {preview(rc, 200)}")
            ok = check_json(m.content, schema, "D2")
            if not (rc and str(rc).strip()):
                log("    FAIL: reasoning is empty (needs --reasoning-parser gemma4 on the server)")
                ok = False
            results["D2"] = ok
        except Exception as e:
            log(f"    FAIL: {type(e).__name__}: {preview(e)}")
            results["D2"] = False

    # --- E: the full photo set for one real line ------------------------
    if "E" in tests:
        log(f"[E] ALL {len(photos)} photos of {args.line} + {shard_id} schema")
        try:
            t0 = time.time()
            r = client.chat.completions.create(
                model=args.model, max_tokens=args.max_tokens,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": image_parts(photos) +
                           [{"type": "text", "text": shard_prompt}]}],
                response_format=response_format(schema, shard_id),
                extra_body=thinking_body(False))
            dt = time.time() - t0
            pt = r.usage.prompt_tokens
            log(f"    latency {dt:.1f}s | prompt={pt} completion={r.usage.completion_tokens}")
            fits = pt <= args.max_model_len
            log(f"    prompt_tokens {pt} <= MAX_MODEL_LEN {args.max_model_len} -> {'OK' if fits else 'FAIL'}")
            log(f"    ~{pt / len(photos):.0f} prompt tokens per photo")
            results["E"] = check_json(r.choices[0].message.content, schema, "E") and fits
        except Exception as e:
            log(f"    FAIL: {type(e).__name__}: {preview(e)}")
            results["E"] = False

    # --- verdict --------------------------------------------------------
    log("")
    log(f"===== summary ({time.time() - t0_all:.0f}s) =====")
    for t in ALL_TESTS:
        if t in results:
            log(f"  {t:2} : {'PASS' if results[t] else 'FAIL'}")

    hard = [t for t in ("A", "B", "C", "E") if t in results and not results[t]]
    # The ticket's rule: at least one of D1/D2 must work. Both failing is a
    # real finding about this checkpoint's guided decoding -- report, don't patch.
    d_run = [t for t in ("D1", "D2") if t in results]
    d_ok = [t for t in d_run if results[t]]
    if d_run and not d_ok:
        log("")
        log("  Neither D1 nor D2 produced schema-valid JSON. Not adding retries or")
        log("  JSON repair here: that is a provider-layer design decision, and if")
        log("  vLLM guided decoding is unreliable for this checkpoint it must be")
        log("  visible rather than hidden behind a fallback.")

    if hard or (d_run and not d_ok):
        log("")
        log("RESULT: FAIL")
        return 1
    log("")
    log("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
