"""Batch fetch of sharded results must be partial-aware and cumulative.

Regression tests for the bug where ``phenotype-batch --dispatch batch`` gaps
(a shard that errored, e.g. a transient ``overloaded_error``) could not be
recovered: the batch path overwrote ``<lid>.json`` from scratch each fetch and
never persisted per-shard partials, so there was nothing for a later resume to
build on.  ``write_phenotype_sharded_results`` is now expected to share the
sequential dispatch's ``<output>/_partial/<line_id>__<shard_id>.json`` store:

  * adopt any per-shard partials already on disk,
  * persist each freshly-succeeded shard as a partial,
  * merge the UNION (prior partials + this batch) into ``<lid>.json``, and
  * only write ``<lid>.gaps.json`` for traits still missing after the union,
    removing a stale gaps file once its traits are filled.
"""

import json
from pathlib import Path

from pxgpt.core.batch_utils import write_phenotype_sharded_results


# --- Minimal fake Anthropic batch client -----------------------------------

class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Usage:
    input_tokens = 1
    output_tokens = 1
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _Message:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.usage = _Usage()


class _Inner:
    """Mimics result.result — either succeeded (message) or errored (error)."""

    def __init__(self, *, message=None, error=None):
        if message is not None:
            self.type = "succeeded"
            self.message = message
        else:
            self.type = "errored"
            self.error = error


class _Result:
    def __init__(self, custom_id, inner):
        self.custom_id = custom_id
        self.result = inner


class _Error:
    """Mimics BetaErrorResponse: type='error' with a nested .error object."""

    class _E:
        type = "overloaded_error"
        message = "File storage is temporarily unavailable. Please retry."

    type = "error"
    error = _E()


class _Batches:
    def __init__(self, results):
        self._results = results

    def results(self, batch_id):  # noqa: ARG002
        return iter(self._results)


class _Messages:
    def __init__(self, results):
        self.batches = _Batches(results)


class _Beta:
    def __init__(self, results):
        self.messages = _Messages(results)


class _Client:
    def __init__(self, results):
        self.beta = _Beta(results)


# --- Fixtures ----------------------------------------------------------------

# Master index: two groups, shard_a covers g1, shard_b covers g2.
GROUP_ORDER = ["g1", "g2"]
GROUP_TRAITS = {"g1": ["t1"], "g2": ["t2"]}
TRAIT_META = {("g1", "t1"): {"scale_type": "nominal"},
              ("g2", "t2"): {"scale_type": "nominal"}}
MASTER_INDEX = (GROUP_ORDER, GROUP_TRAITS, TRAIT_META)

SHARD_A = json.dumps({"g1": {"t1": {"rationale": "r1", "value": "A"}}})
SHARD_B = json.dumps({"g2": {"t2": {"rationale": "r2", "value": "B"}}})


def _succeeded(custom_id, text):
    return _Result(custom_id, _Inner(message=_Message(text)))


def _errored(custom_id):
    return _Result(custom_id, _Inner(error=_Error()))


def _partial(out, custom_id, text):
    pd = out / "_partial"
    pd.mkdir(parents=True, exist_ok=True)
    (pd / f"{custom_id}.json").write_text(text, encoding="utf-8")


def test_succeeded_shard_is_persisted_as_partial(tmp_path):
    client = _Client([_succeeded("p1__shard_a", SHARD_A),
                      _succeeded("p1__shard_b", SHARD_B)])
    write_phenotype_sharded_results(client, "b", ["p1"], MASTER_INDEX, str(tmp_path))

    assert (tmp_path / "_partial" / "p1__shard_a.json").exists()
    assert (tmp_path / "_partial" / "p1__shard_b.json").exists()
    record = json.loads((tmp_path / "p1.json").read_text())
    assert record["g1"]["t1"]["value"] == "A"
    assert record["g2"]["t2"]["value"] == "B"
    assert not (tmp_path / "p1.gaps.json").exists()


def test_prior_partial_fills_gap_from_failed_shard(tmp_path):
    # shard_a already recovered on disk; the batch only re-ran shard_b OK.
    _partial(tmp_path, "p1__shard_a", SHARD_A)
    client = _Client([_succeeded("p1__shard_b", SHARD_B)])

    write_phenotype_sharded_results(client, "b", ["p1"], MASTER_INDEX, str(tmp_path))

    record = json.loads((tmp_path / "p1.json").read_text())
    assert record["g1"]["t1"]["value"] == "A"   # from the adopted partial
    assert record["g2"]["t2"]["value"] == "B"   # from this batch
    assert not (tmp_path / "p1.gaps.json").exists()


def test_still_missing_shard_reports_gap(tmp_path):
    client = _Client([_succeeded("p1__shard_a", SHARD_A),
                      _errored("p1__shard_b")])

    write_phenotype_sharded_results(client, "b", ["p1"], MASTER_INDEX, str(tmp_path))

    gaps = json.loads((tmp_path / "p1.gaps.json").read_text())
    assert {"group": "g2", "trait": "t2"} in gaps["missing_traits"]
    assert any("shard_b" in e for e in gaps["shard_errors"])


def test_stale_gaps_removed_once_filled(tmp_path):
    # A prior run left a gaps file and one shard partial; this fetch fills the rest.
    _partial(tmp_path, "p1__shard_a", SHARD_A)
    (tmp_path / "p1.gaps.json").write_text(
        json.dumps({"line_id": "p1",
                    "missing_traits": [{"group": "g2", "trait": "t2"}],
                    "shard_errors": ["shard_b: overloaded_error"]}),
        encoding="utf-8")

    client = _Client([_succeeded("p1__shard_b", SHARD_B)])
    write_phenotype_sharded_results(client, "b", ["p1"], MASTER_INDEX, str(tmp_path))

    assert not (tmp_path / "p1.gaps.json").exists()
