"""Concurrency, limits and the abort paths for ``schema --shard-dir``.

The load-bearing test here is `test_global_inflight_never_exceeds_the_cap`.
Pipeline depth alone does not bound concurrent requests: depth 2 with a
within-plant width of 8 could put two plants in their warm phase at once, which
is 16 requests -- never measured, and exactly the server's MAX_NUM_SEQS. The
global semaphore is what prevents that, so it is asserted rather than reviewed.
"""

import argparse
import json
import threading
import time
from pathlib import Path

import pytest

from pxgpt.commands import schema as sc
from pxgpt.core.batch_utils import _RUN_META_NAME
from pxgpt.providers.base import APIResponse, TokenUsage


# ---------------------------------------------------------------------- fixtures

def _parser():
    p = argparse.ArgumentParser(prog="pxgpt")
    sc.setup_schema_parser(p.add_subparsers(dest="command", required=True))
    return p


def _make_shard_set(root, n_shards, n_traits=None):
    n_traits = n_traits or n_shards
    d = root / "shards"
    d.mkdir(parents=True)
    (d / "system.txt").write_text("system", encoding="utf-8")
    shards, traits = [], []
    for i in range(1, n_shards + 1):
        sid = f"shard_{i:02d}"
        t = f"t{i:02d}"
        traits.append({"group": "g", "trait": t, "scale_type": "nominal", "unit": None})
        (d / f"{sid}.schema.json").write_text(json.dumps(
            {"title": sid, "type": "object", "additionalProperties": False,
             "required": ["g"], "properties": {"g": {"type": "object"}}}),
            encoding="utf-8")
        (d / f"{sid}.prompt.txt").write_text(f"score {t}", encoding="utf-8")
        shards.append({"shard_id": sid, "schema_file": f"{sid}.schema.json",
                       "prompt_file": f"{sid}.prompt.txt", "groups": ["g"],
                       "traits": [t]})
    (d / "shards_manifest.json").write_text(json.dumps(
        {"system_file": "system.txt", "shards": shards, "all_traits": traits}),
        encoding="utf-8")
    return d


def _make_tree(root, n_plants):
    tree = root / "tree"
    for i in range(1, n_plants + 1):
        p = tree / f"s{i:04d}"
        p.mkdir(parents=True)
        (p / "a.jpg").write_bytes(b"\xff\xd8\xff")
    return tree


class RecordingProvider:
    """Counts concurrent in-flight calls and records dispatch order."""

    provider_name = "openai-compat-vllm"

    def __init__(self, delay=0.02, fail_first_pending=False, hold=None):
        self.delay = delay
        self.fail_first_pending = fail_first_pending
        self.hold = hold
        self.order = []
        self.concurrent = 0
        self.peak = 0
        self._lock = threading.Lock()
        self._seen = set()

    def send_request_with_retry(self, **kw):
        prompt = kw["messages"][0]["content"][-1]["text"]
        with self._lock:
            self.order.append(prompt)
            self.concurrent += 1
            self.peak = max(self.peak, self.concurrent)
            first = len(self._seen) == 0
            self._seen.add(prompt)
        try:
            time.sleep(self.delay)
            if self.fail_first_pending and first:
                raise RuntimeError("head failed on purpose")
            return APIResponse(
                content=json.dumps({"g": {prompt.split()[-1]: {"value": "v",
                                                               "rationale": "r"}}}),
                usage=TokenUsage(input_tokens=1000, output_tokens=10,
                                 cache_read_tokens=990),
                request_id="r", model="m", finish_reason="stop")
        finally:
            with self._lock:
                self.concurrent -= 1


def _run(monkeypatch, shard_dir, out, provider, argv_extra=(), tree=None, plant=None):
    monkeypatch.setenv("VLLM_MODEL", "m")
    monkeypatch.setattr(sc, "create_provider", lambda n, c: provider)
    argv = ["schema", "--provider", "vllm", "--shard-dir", str(shard_dir),
            "--output", str(out)]
    argv += ["--input-dir", str(tree)] if tree is not None else ["--input-folder", str(plant)]
    argv += list(argv_extra)
    return sc.schema_command(_parser().parse_args(argv))


# ------------------------------------------------------- 2/3/4: width is a cap

@pytest.mark.parametrize("n_shards,expected_width", [(7, 6), (10, 8), (9, 8), (30, 8)])
def test_effective_width_is_capped_never_derived_from_shard_count(
        tmp_path, monkeypatch, capsys, n_shards, expected_width):
    """01_seedling=7, 02_mature_v1=10, 03_mature_v2=9, plus a 30-shard set.

    30 shards must still fan out 8, not 29 -- the cap is hardware pressure
    measured on this box, and it is only a coincidence that a 9-shard set has 8
    in its remainder.
    """
    sd = _make_shard_set(tmp_path, n_shards)
    prov = RecordingProvider()
    _run(monkeypatch, sd, tmp_path / "o", prov,
         plant=_make_tree(tmp_path, 1) / "s0001")

    line = capsys.readouterr().out
    assert f"up to {expected_width} concurrent" in line
    assert "global cap 9" in line


def test_concurrency_four_gives_width_four_and_cap_five(tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 10)
    _run(monkeypatch, sd, tmp_path / "o", RecordingProvider(),
         argv_extra=["--concurrency", "4"],
         plant=_make_tree(tmp_path, 1) / "s0001")

    out = capsys.readouterr().out
    assert "up to 4 concurrent" in out
    assert "global cap 5" in out


def test_pipeline_depth_three_is_refused_by_argparse():
    with pytest.raises(SystemExit):
        _parser().parse_args(["schema", "--output", "o", "--shard-dir", "d",
                              "--input-dir", "t", "--pipeline-depth", "3"])


def test_concurrency_zero_is_refused():
    with pytest.raises(SystemExit):
        _parser().parse_args(["schema", "--output", "o", "--shard-dir", "d",
                              "--input-dir", "t", "--concurrency", "0"])


# ------------------------------------------------------ 6: exactly one cold shard

def test_exactly_one_shard_is_sent_alone_before_the_rest(tmp_path, monkeypatch):
    """The cold shard must complete before any other starts.

    All of a plant's shards share a prefix of system prompt + every image, and
    only the first arrival pays to build it.  Firing them together would have
    each miss and pay its own prefill.
    """
    sd = _make_shard_set(tmp_path, 5)
    prov = RecordingProvider(delay=0.05)
    _run(monkeypatch, sd, tmp_path / "o", prov,
         plant=_make_tree(tmp_path, 1) / "s0001")

    assert prov.order[0] == "score t01"          # manifest order, sent first
    assert prov.peak == 4                        # the remaining 4, together
    assert len(prov.order) == 5


def test_concurrency_one_is_fully_serial(tmp_path, monkeypatch):
    """The regression safety net: --concurrency 1 never overlaps anything."""
    sd = _make_shard_set(tmp_path, 5)
    prov = RecordingProvider(delay=0.01)
    code = _run(monkeypatch, sd, tmp_path / "o", prov,
                argv_extra=["--concurrency", "1", "--pipeline-depth", "1"],
                plant=_make_tree(tmp_path, 1) / "s0001")

    assert code == 0
    assert prov.peak == 1
    assert prov.order == [f"score t{i:02d}" for i in range(1, 6)]


# ------------------------------------------- 7: the global in-flight ceiling

def test_global_inflight_never_exceeds_the_cap(tmp_path, monkeypatch):
    """depth 2 x width 8 must NOT become 16 concurrent requests."""
    sd = _make_shard_set(tmp_path, 9)
    tree = _make_tree(tmp_path, 6)
    prov = RecordingProvider(delay=0.03)

    code = _run(monkeypatch, sd, tmp_path / "o", prov, tree=tree,
                argv_extra=["--concurrency", "8", "--pipeline-depth", "2"])

    assert code == 0
    assert prov.peak <= 9, f"saw {prov.peak} concurrent requests, cap is 9"
    assert prov.peak > 1, "expected real overlap, got none"


def test_reported_peak_matches_the_gate(tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 9)
    _run(monkeypatch, sd, tmp_path / "o", RecordingProvider(delay=0.02),
         tree=_make_tree(tmp_path, 4))

    out = capsys.readouterr().out
    assert "Peak in-flight:" in out
    peak = int(out.split("Peak in-flight:")[1].split("request")[0].strip())
    assert 1 < peak <= 9


# --------------------------------------------------------------- 8: resume head

def test_resume_sends_only_the_missing_shards_and_colds_the_first(tmp_path, monkeypatch):
    """After deleting shard_05 and shard_07, shard_05 is the new cold shard.

    A partial on disk says nothing about the server's KV cache -- a restarted
    container has an empty one -- so the first shard actually SENT is the one
    that pays the prefill, whatever its number.
    """
    sd = _make_shard_set(tmp_path, 9)
    plant = _make_tree(tmp_path, 1) / "s0001"
    out = tmp_path / "o"
    _run(monkeypatch, sd, out, RecordingProvider(delay=0.005), plant=plant)

    for sid in ("shard_05", "shard_07"):
        (out / "_partial" / f"s0001__{sid}.json").unlink()

    prov = RecordingProvider(delay=0.02)
    _run(monkeypatch, sd, out, prov, plant=plant)

    assert prov.order[0] == "score t05"                 # cold, alone
    assert sorted(prov.order) == ["score t05", "score t07"]
    assert prov.peak == 1                               # only one left to fan out


# ------------------------------------------------------------- 9: head failure

def test_a_failed_head_still_lets_the_rest_fan_out(tmp_path, monkeypatch):
    """length / reasoning leak / parse error all happen AFTER the prefill."""
    sd = _make_shard_set(tmp_path, 6)
    prov = RecordingProvider(delay=0.02, fail_first_pending=True)

    code = _run(monkeypatch, sd, tmp_path / "o", prov,
                plant=_make_tree(tmp_path, 1) / "s0001")

    assert code == 1                       # the head failure is reported
    assert len(prov.order) == 6            # ...and the other 5 still ran
    assert prov.peak == 5


# ------------------------------------------------------- 10/11: memory guard

def test_an_impossible_memory_floor_prevents_all_overlap(tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 4)
    prov = RecordingProvider(delay=0.02)

    # Pin the reading instead of taking the host's. `mem_available_gib` reads
    # /proc/meminfo, which does not exist on macOS, where it returns None and
    # the guard turns itself off -- so this test used to fail there for a
    # reason that has nothing to do with the guard, and nothing to do with how
    # much memory the machine has. The sibling test below covers the None path
    # deliberately.
    monkeypatch.setattr(sc, "mem_available_gib", lambda: 8.0)

    code = _run(monkeypatch, sd, tmp_path / "o", prov, tree=_make_tree(tmp_path, 3),
                argv_extra=["--mem-floor-gib", "9999", "--pipeline-depth", "2"])

    out = capsys.readouterr().out
    assert code == 0
    assert "below the 9999.0 GiB floor" in out
    assert "guard withheld a plant" in out
    assert "guard withheld a plant 0 time(s)" not in out


def test_a_normal_floor_never_trips_the_guard(tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 4)

    # Pinned for the same reason as above: assert on the guard, not on the host.
    monkeypatch.setattr(sc, "mem_available_gib", lambda: 8.0)

    _run(monkeypatch, sd, tmp_path / "o", RecordingProvider(delay=0.01),
         tree=_make_tree(tmp_path, 3), argv_extra=["--mem-floor-gib", "0.001"])

    out = capsys.readouterr().out
    assert "below the" not in out
    assert "guard withheld a plant 0 time(s)" in out


def test_missing_proc_meminfo_disables_the_guard_without_erroring(
        tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 4)
    monkeypatch.setattr(sc, "mem_available_gib", lambda: None)

    code = _run(monkeypatch, sd, tmp_path / "o", RecordingProvider(delay=0.005),
                tree=_make_tree(tmp_path, 2))

    assert code == 0
    assert "memory guard is off" in capsys.readouterr().out


# ------------------------------------------------------------ 12/13: aborting

def test_circuit_breaker_aborts_after_three_barren_plants_and_still_merges(
        tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 3)
    tree = _make_tree(tmp_path, 10)
    out = tmp_path / "o"

    class Dead(RecordingProvider):
        def send_request_with_retry(self, **kw):
            raise RuntimeError("connection refused")

    code = _run(monkeypatch, sd, out, Dead(), tree=tree,
                argv_extra=["--pipeline-depth", "1"])

    printed = capsys.readouterr().out
    assert code == 1
    assert "ABORTED" in printed
    assert "consecutive plant(s) produced no successful shard" in printed
    assert "connection refused" in printed          # the last error is shown
    # Merge still ran, so the finished plants have records rather than nothing.
    assert "--- Merging ---" in printed
    assert sc.CIRCUIT_BREAKER_PLANTS == 3


def test_keyboard_interrupt_still_merges_what_finished(tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 2)
    tree = _make_tree(tmp_path, 8)
    out = tmp_path / "o"

    class Interrupting(RecordingProvider):
        def send_request_with_retry(self, **kw):
            if len(self.order) >= 6:
                raise KeyboardInterrupt
            return super().send_request_with_retry(**kw)

    code = _run(monkeypatch, sd, out, Interrupting(delay=0.005), tree=tree,
                argv_extra=["--pipeline-depth", "1"])

    printed = capsys.readouterr().out
    assert "--- Merging ---" in printed
    assert list(out.glob("s*.json")), "an interrupted run must still write records"
    assert (out / "_partial").is_dir()
    assert code == 1


# --------------------------------------------------------------- --limit, misc

def test_limit_runs_only_the_first_n_plants(tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 2)
    prov = RecordingProvider(delay=0.005)

    _run(monkeypatch, sd, tmp_path / "o", prov, tree=_make_tree(tmp_path, 9),
         argv_extra=["--limit", "3"])

    assert "running the first 3 of 9" in capsys.readouterr().out
    assert len(prov.order) == 6          # 3 plants x 2 shards


def test_rows_are_reported_in_manifest_order_despite_completion_order(
        tmp_path, monkeypatch, capsys):
    """Completion order under fan-out is nondeterministic; the table must not be."""
    sd = _make_shard_set(tmp_path, 6)

    class Jittered(RecordingProvider):
        def send_request_with_retry(self, **kw):
            n = int(kw["messages"][0]["content"][-1]["text"][-2:])
            time.sleep(0.05 / n)          # later shards finish FIRST
            return APIResponse(
                content=json.dumps({"g": {f"t{n:02d}": {"value": "v", "rationale": "r"}}}),
                usage=TokenUsage(input_tokens=100, output_tokens=5, cache_read_tokens=90),
                request_id="r", model="m", finish_reason="stop")

    _run(monkeypatch, sd, tmp_path / "o", Jittered(), plant=_make_tree(tmp_path, 1) / "s0001")

    table = capsys.readouterr().out.split("shard_id")[-1]
    seen = [ln.split()[0] for ln in table.splitlines()
            if ln.startswith("shard_")]
    assert seen == [f"shard_{i:02d}" for i in range(1, 7)]


def test_low_cache_hit_rate_raises_the_canary_warning(tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 5)

    class ColdAlways(RecordingProvider):
        def send_request_with_retry(self, **kw):
            n = int(kw["messages"][0]["content"][-1]["text"][-2:])
            return APIResponse(
                content=json.dumps({"g": {f"t{n:02d}": {"value": "v", "rationale": "r"}}}),
                usage=TokenUsage(input_tokens=1000, output_tokens=5, cache_read_tokens=0),
                request_id="r", model="m", finish_reason="stop")

    _run(monkeypatch, sd, tmp_path / "o", ColdAlways(), tree=_make_tree(tmp_path, 1))

    out = capsys.readouterr().out
    assert "WARNING" in out and "prefix-cache hit" in out
    assert "mm_processor_kwargs" in out          # points at the likely cause


def test_provenance_guard_blocks_before_any_request(tmp_path, monkeypatch, capsys):
    sd = _make_shard_set(tmp_path, 4)
    out = tmp_path / "o"
    (out / "_partial").mkdir(parents=True)
    (out / "_partial" / _RUN_META_NAME).write_text(
        json.dumps({"provider": "anthropic", "model": "claude-sonnet-5"}),
        encoding="utf-8")
    prov = RecordingProvider()

    code = _run(monkeypatch, sd, out, prov, tree=_make_tree(tmp_path, 2))

    assert code == 1
    assert prov.order == []
    assert not list((out / "_partial").glob("s*.json"))


# ------------------------------------- 2 (real datasets), 17 (data unchanged)

REAL = Path("/home/xavier/project/pxgpt")
DATASETS = [("01_seedling", 7, 6), ("02_mature_v1", 10, 8), ("03_mature_v2", 9, 8)]


@pytest.mark.parametrize("name,n_shards,width", DATASETS)
def test_the_three_real_shard_sets_load_and_cap_correctly(name, n_shards, width):
    """No per-dataset configuration: 7/10/9 shards all cap at the same ceiling."""
    sd = REAL / name / "shard_master_schema"
    if not (sd / "shards_manifest.json").exists():
        pytest.skip(f"{name} not present on this machine")

    from pxgpt.core import sharding
    manifest, shards = sharding.load_shard_set(str(sd))

    assert len(shards) == n_shards
    assert min(8, max(1, len(shards) - 1)) == width      # effective width
    assert 8 + 1 == 9                                    # global ceiling


@pytest.mark.parametrize("name", [d[0] for d in DATASETS])
def test_frozen_data_checksums_verify(name):
    """The 鐵則: shard set and images are read-only and must still match."""
    import subprocess
    root = REAL / name
    for f in ("shard_schema.sha256", "images.sha256"):
        if not (root / f).exists():
            pytest.skip(f"{name}/{f} not present")
        r = subprocess.run(["sha256sum", "-c", "--quiet", f],
                           cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, f"{name}/{f} FAILED:\n{r.stdout}{r.stderr}"
