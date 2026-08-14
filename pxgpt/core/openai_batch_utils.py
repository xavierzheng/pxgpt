"""Helpers for the OpenAI Batch API (describe-batch-openai / phenotype-batch-openai).

Uses the **Responses API** endpoint (``/v1/responses``).  This is required for
the OpenAI Files API: an uploaded image can only be referenced by ``file_id``
through the Responses API (``{"type": "input_image", "file_id": ...}``).  The
Chat Completions API cannot reference uploaded images at all — its ``image_url``
needs a real URL / data URL and its ``file`` type accepts PDFs only.

The OpenAI Batch API consumes a JSONL file where each line is a standalone
``/v1/responses`` request:

    {"custom_id": "...", "method": "POST", "url": "/v1/responses",
     "body": { ...responses params... }}

Images are referenced either by an uploaded Files-API id
(``{"type": "input_image", "file_id": "<id>"}``) or embedded inline as a base64
data URL (``{"type": "input_image", "image_url": "data:...;base64,..."}``).
Results come back as a JSONL file (one line per request) downloaded via
``client.files.content(output_file_id)``.
"""

import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import openai

from .image_utils import get_base64_encoded_image, _MEDIA_TYPES
from .batch_utils import strip_code_fence  # shared code-fence stripper


# ---------------------------------------------------------------------------
# Image content blocks (Responses API format)
# ---------------------------------------------------------------------------

def build_openai_file_id_blocks(file_ids: Dict[str, str]) -> List[Dict[str, Any]]:
    """Return input_image blocks that reference OpenAI Files-API ids.

    Insertion order of *file_ids* (filename → file_id) is preserved.
    """
    return [
        {"type": "input_image", "file_id": fid}
        for fid in file_ids.values()
    ]


def build_openai_base64_blocks(image_paths) -> List[Dict[str, Any]]:
    """Return input_image blocks with inline base64 data URLs for *image_paths*."""
    blocks: List[Dict[str, Any]] = []
    for p in image_paths:
        p = Path(p)
        media_type = _MEDIA_TYPES.get(p.suffix.lower(), "image/jpeg")
        data = get_base64_encoded_image(str(p))
        blocks.append(
            {"type": "input_image", "image_url": f"data:{media_type};base64,{data}"}
        )
    return blocks


# ---------------------------------------------------------------------------
# Schema normalization for OpenAI strict structured outputs
# ---------------------------------------------------------------------------

# Keywords OpenAI strict structured outputs reject or ignore at the node level.
_OPENAI_STRIP_NODE_KEYS = {"format", "default", "title", "examples", "$schema"}


def openai_normalize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of *schema* adjusted for OpenAI strict json_schema.

    OpenAI's ``strict: true`` mode requires that every object set
    ``additionalProperties: false`` and list **all** of its properties in
    ``required``.  This differs from the Anthropic normalizer (which allows an
    empty ``required``), so OpenAI gets its own pass.
    """
    schema = copy.deepcopy(schema)
    schema.pop("$schema", None)
    _walk_openai(schema)
    return schema


def _walk_openai(node: Any) -> None:
    if not isinstance(node, dict):
        return

    for k in _OPENAI_STRIP_NODE_KEYS:
        node.pop(k, None)

    # OpenAI strict mode wants each node to declare a type.  The frozen shard
    # schemas write enum leaves as {"enum": [...]} with no "type", which
    # Anthropic and xgrammar accept.  Adding "type": "string" to an all-string
    # enum does not change which strings are accepted, so the effective
    # constraint — and the constrained-decoding mask — stays identical across
    # backends.  A mixed-type enum is left alone rather than guessed at, and so
    # is a degenerate empty one; OpenAI can report those itself.
    enum_members = node.get("enum")
    if (
        "type" not in node
        and isinstance(enum_members, list)
        and enum_members
        and all(isinstance(m, str) for m in enum_members)
    ):
        node["type"] = "string"

    if node.get("type") == "object" or "properties" in node:
        node["additionalProperties"] = False
        if "properties" in node:
            # strict mode: every property must be required
            node["required"] = list(node["properties"].keys())
        for child in node.get("properties", {}).values():
            _walk_openai(child)

    if isinstance(node.get("items"), dict):
        _walk_openai(node["items"])

    for key in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(key, []):
            _walk_openai(sub)


# ---------------------------------------------------------------------------
# Request body / JSONL building (Responses API)
# ---------------------------------------------------------------------------

def _is_reasoning_model(model: str) -> bool:
    """True for OpenAI reasoning models that reject a custom temperature."""
    m = model.lower()
    return "gpt-5" in m or m.startswith(("o1", "o3", "o4"))


def openai_effort_status(model: str, effort: str) -> str:
    """Human-readable summary of what ``build_responses_request_body`` will send.

    For the run banner — keeps it in sync with the logic below instead of
    assuming a model family.
    """
    if not _is_reasoning_model(model):
        return "not a reasoning model; temperature sent"
    if effort:
        return f"{effort} (temperature omitted — only effort 'none' accepts it)"
    return "none — reasoning explicitly off; temperature sent"


def schema_format_name(schema: Dict[str, Any], fallback: str = "structured_output") -> str:
    """Return a Responses-API ``format.name`` derived from the schema's ``title``.

    Shard schemas carry titles like ``stage3_shard_01``, which makes an API error
    traceable to a shard.  The name is sanitised to ``[a-zA-Z0-9_-]``.

    Call this BEFORE :func:`openai_normalize_schema`, which strips ``title``.
    """
    title = schema.get("title") if isinstance(schema, dict) else None
    if not isinstance(title, str):
        return fallback
    return re.sub(r"[^a-zA-Z0-9_-]", "_", title) or fallback


def build_text_format(schema: Dict[str, Any], name: str = "structured_output") -> Dict[str, Any]:
    """Return a Responses-API ``text`` value for strict structured output."""
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": True,
        }
    }


def openai_compile_check_schema(
    client,
    model: str,
    schema: Dict[str, Any],
    name: str = "structured_output",
) -> Tuple[bool, Optional[str]]:
    """Pre-flight one schema against the Responses API.

    Returns ``(ok, error_message)``, matching
    :func:`pxgpt.core.sharding.compile_check_schema`'s contract.

    Schema errors are per-request on the Batch API: one bad shard errors its own
    requests while the rest of the batch succeeds and bills.  Without a
    pre-flight that means paying for 7/8 of a run to discover 1/8 is missing.

    Any 400 is read as a schema problem.  Nothing else in this request can be
    invalid — a nonexistent model is a 404, the input is one short sentence, and
    no optional parameters are sent — which is far sturdier than matching on
    error text.  Every other exception propagates unchanged.
    """
    try:
        client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "Reply with any JSON object that satisfies the schema.",
                }],
            }],
            max_output_tokens=16,
            text=build_text_format(schema, name=name),
        )
        return True, None
    except openai.BadRequestError as e:
        return False, str(e)
    except openai.APIStatusError as e:
        if getattr(e, "status_code", None) == 400:
            return False, str(e)
        raise


# Wording OpenAI uses when a schema busts one of its strict-mode SIZE limits,
# measured against gpt-5.6-luna:
#
#   nesting depth 15  -> "14 levels of nesting exceeds limit of 10."
#   6000 properties   -> "6000 parameters exceeds limit of 5000."
#
# Both are 400 invalid_json_schema, and both carry "exceeds limit of".  A schema
# that is merely malformed for strict mode does not, e.g.
#   "'additionalProperties' is required to be supplied and to be false."
# That distinction is what keeps ``--allow-reshard`` from overwriting a frozen
# shard set over an error resharding cannot fix.  The extra phrases are cheap
# insurance against future wording; only "exceeds limit of" is measured.
_OPENAI_SIZE_LIMIT_MARKERS = ("exceeds limit of", "too complex", "too large",
                              "too many")


def is_openai_size_limit_error(message: str) -> bool:
    """True if *message* is OpenAI rejecting a schema for exceeding a size limit."""
    msg = (message or "").lower()
    return any(marker in msg for marker in _OPENAI_SIZE_LIMIT_MARKERS)


def openai_compile_probe(client, model: str, schema: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """``ensure_compilable``-compatible probe for the OpenAI path.

    Takes the RAW shard schema, names the response format from its ``title``
    before :func:`openai_normalize_schema` strips it, and probes the normalized
    copy.

    Returns ``False`` only for a genuine size-limit rejection — the one class of
    failure a smaller shard budget can fix.  Any other schema 400 (strict-mode
    malformation, an unsupported keyword) raises ``RuntimeError`` instead:
    resharding would not fix it, and a ``False`` return is what authorises
    ``--allow-reshard`` to overwrite the shard set on disk.
    """
    name = schema_format_name(schema)
    ok, err = openai_compile_check_schema(
        client, model, openai_normalize_schema(schema), name=name
    )
    if ok:
        return True, None
    if is_openai_size_limit_error(err):
        return False, err
    raise RuntimeError(
        f"OpenAI rejected the schema for response_format {name!r}, and not for a "
        f"size limit — resharding at a smaller budget would not fix this, so the "
        f"shard set was NOT modified:\n  {err}"
    )


def build_responses_request_body(
    model: str,
    system_prompt: str,
    image_blocks: List[Dict[str, Any]],
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    text_format: Optional[Dict[str, Any]] = None,
    reasoning_effort: str = "",
) -> Dict[str, Any]:
    """Assemble one ``/v1/responses`` request body.

    The system prompt is passed via ``instructions``.  Reasoning models always
    receive an explicit ``reasoning.effort`` (default ``"none"``); temperature
    rides along only when that effort is ``"none"``, the one level that accepts
    it.
    """
    user_content = image_blocks + [{"type": "input_text", "text": user_prompt}]
    body: Dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": user_content}],
        "max_output_tokens": max_tokens,
    }
    if _is_reasoning_model(model):
        # Always send an explicit effort.  Omitting ``reasoning`` does NOT turn
        # reasoning off — the model falls back to its own default (medium on
        # gpt-5.6), so "off" has to be spelled out as "none".
        effort = reasoning_effort or "none"
        body["reasoning"] = {"effort": effort}
        # A custom temperature is accepted only with reasoning fully off; any
        # other effort level rejects it with a 400 (verified against gpt-5.6-luna).
        if effort == "none":
            body["temperature"] = temperature
    else:
        body["temperature"] = temperature
    if text_format is not None:
        body["text"] = text_format
    return body


def write_jsonl_requests(requests: List[Dict[str, Any]], path: str) -> None:
    """Write one batch request per line to *path* as JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Batch polling
# ---------------------------------------------------------------------------

# Terminal states for an OpenAI batch.
_TERMINAL_STATES = {"completed", "failed", "expired", "cancelled"}


def poll_openai_batch(client, batch_id: str, interval: int = 30):
    """Block until the OpenAI batch reaches a terminal state.  Prints progress."""
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = getattr(counts, "total", 0)
        completed = getattr(counts, "completed", 0)
        failed = getattr(counts, "failed", 0)
        print(
            f"  [{batch_id}] {batch.status} | "
            f"completed={completed}  failed={failed}  total={total}"
        )
        if batch.status in _TERMINAL_STATES:
            return batch
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Result extraction
# ---------------------------------------------------------------------------

def _download_jsonl(client, file_id: Optional[str]) -> List[Dict[str, Any]]:
    if not file_id:
        return []
    text = client.files.content(file_id).text
    return [json.loads(line) for line in text.strip().split("\n") if line.strip()]


def collect_openai_results(client, batch) -> Dict[str, Dict[str, Any]]:
    """Return ``{custom_id: parsed-output-line}`` from a finished batch.

    Merges the success output file and the error file; a given custom_id
    appears in exactly one of them.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for line in _download_jsonl(client, getattr(batch, "output_file_id", None)):
        results[line["custom_id"]] = line
    for line in _download_jsonl(client, getattr(batch, "error_file_id", None)):
        results.setdefault(line["custom_id"], line)
    return results


def _extract_text_and_usage(line: Dict[str, Any]):
    """Return ``(content, error_message, usage_dict)`` for one Responses output line.

    *content* is None when the request errored or the model refused.
    """
    err = line.get("error")
    resp = line.get("response")
    if err:
        return None, f"{err.get('code', 'error')}: {err.get('message', err)}", {}
    if not resp or resp.get("status_code") != 200:
        body = (resp or {}).get("body") or {}
        be = body.get("error")
        if be:
            msg = f"{be.get('code', '')}: {be.get('message', be)}".strip(": ")
            return None, msg, body.get("usage", {}) or {}
        code = resp.get("status_code") if resp else "no-response"
        return None, f"HTTP {code}", {}

    body = resp.get("body", {})
    usage = body.get("usage", {}) or {}

    # The Responses body holds an `output` array; assistant text lives in
    # message items as `output_text` parts (reasoning items are skipped).
    texts: List[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            ptype = part.get("type")
            if ptype == "output_text":
                texts.append(part.get("text", ""))
            elif ptype == "refusal":
                return None, f"refusal: {part.get('refusal')}", usage
    if not texts:
        return None, "no output_text in response", usage
    return "".join(texts), None, usage


def _accumulate_usage(totals: Dict[str, int], usage: Dict[str, Any]) -> None:
    # Responses API usage uses input_tokens / output_tokens.
    totals["input"] += usage.get("input_tokens", 0)
    totals["output"] += usage.get("output_tokens", 0)
    details = usage.get("input_tokens_details") or {}
    totals["cache_read"] += details.get("cached_tokens", 0)


# ---------------------------------------------------------------------------
# Result writers (mirror core.batch_utils for the Anthropic path)
# ---------------------------------------------------------------------------

def write_openai_describe_results(
    client, batch, line_ids: List[str], output_path: str
) -> Dict[str, int]:
    """Write grouped description text from a finished OpenAI batch."""
    results = collect_openai_results(client, batch)
    totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}
    raw: Dict[str, str] = {}

    for cid, line in results.items():
        content, err, usage = _extract_text_and_usage(line)
        _accumulate_usage(totals, usage)
        if err is None:
            raw[cid] = content
        else:
            raw[cid] = f"[ERROR {err}]"
            print(f"  WARNING: {cid} failed — {err}")

    sections = []
    for lid in line_ids:
        text = raw.get(lid, "[NOT FOUND IN RESULTS]")
        sections.append(f"### {lid}\n\n{text}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(sections) + "\n")

    return totals


def write_openai_phenotype_results(
    client, batch, line_ids: List[str], output_dir: str
) -> Dict[str, int]:
    """Write one JSON file per plant line from a finished OpenAI batch."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    totals = {"input": 0, "output": 0, "cache_creation": 0, "cache_read": 0}

    results = collect_openai_results(client, batch)
    written = 0
    errored = 0

    for cid, line in results.items():
        content, err, usage = _extract_text_and_usage(line)
        _accumulate_usage(totals, usage)
        if err is not None:
            dest = out / f"{cid}.err.txt"
            with open(dest, "w", encoding="utf-8") as f:
                f.write(f"[ERROR {err}]\n")
            print(f"  WARNING: {cid} failed — {err}")
            errored += 1
            continue
        try:
            parsed = json.loads(strip_code_fence(content))
            dest = out / f"{cid}.json"
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2)
                f.write("\n")
            written += 1
        except json.JSONDecodeError:
            dest = out / f"{cid}.err.txt"
            with open(dest, "w", encoding="utf-8") as f:
                f.write(content or "")
            print(f"  WARNING: {cid} — JSON parse failed; raw text saved to {dest}")
            errored += 1

    print(f"  Wrote {written} JSON files; {errored} errors")
    return totals
