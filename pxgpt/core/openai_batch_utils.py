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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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

    The system prompt is passed via ``instructions``.  Temperature is omitted
    for reasoning models (which only accept the default); for those models a
    ``reasoning.effort`` is added when configured.
    """
    user_content = image_blocks + [{"type": "input_text", "text": user_prompt}]
    body: Dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": [{"role": "user", "content": user_content}],
        "max_output_tokens": max_tokens,
    }
    if _is_reasoning_model(model):
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
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
