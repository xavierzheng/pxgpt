"""json-to-table: provenance columns, feather metadata, and the mixed-run refusal.

A table whose rows came from two models reads as one experiment and is not one,
so the default is refusal.  These tests pin that, the three reserved columns in
BOTH outputs (a CSV reader is not second-class), the Arrow metadata living
alongside — never instead of — the ``pandas`` key that makes ordinals ordered,
and the legacy directories that must still flatten.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.feather as pa_feather
import pytest

from pxgpt.core import json2table
from pxgpt.commands.json2table import setup_json2table_parser

PROV_COLS = ["provider", "model", "schema_version"]

MASTER = {
    "schema_name": "probe_master",
    "schema_version": "3.1",
    "trait_groups": {
        "leaf": {"traits": [
            {"trait_name": "color", "scale_type": "nominal", "unit": None,
             "values": ["green", "red"]},
            {"trait_name": "length", "scale_type": "quantitative", "unit": "cm",
             "values": None},
            {"trait_name": "lobing", "scale_type": "ordinal", "unit": None,
             "values": [{"level": 1, "label": "none"},
                        {"level": 2, "label": "shallow"},
                        {"level": 3, "label": "deep"}]},
        ]},
    },
}

LOBING_LEVELS = ["none", "shallow", "deep"]


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _prov(model="claude-sonnet-5", provider="anthropic", schema_version="3.1",
          run_id="msgbatch_abc"):
    return {
        "provider": provider,
        "model": model,
        "schema_name": "probe_master",
        "schema_version": schema_version,
        "pxgpt_version": "0.4.0",
        "created": "2026-08-21T09:00:00Z",
        "run_id": run_id,
    }


def _record(color="green", length=12.5, lobing=3, prov=None):
    record = {}
    if prov is not None:
        record["_provenance"] = prov
    record["leaf"] = {
        "color": {"rationale": "r", "value": color},
        "length": {"rationale": "r", "value": length},
        "lobing": {"rationale": "r", "value": lobing},
    }
    return record


@pytest.fixture
def master_path(tmp_path):
    p = tmp_path / "master_schema.json"
    _write_json(p, MASTER)
    return str(p)


@pytest.fixture
def uniform_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    _write_json(d / "c1.json", _record(prov=_prov()))
    _write_json(d / "c2.json", _record(color="red", length=9.0, lobing=1,
                                       prov=_prov()))
    return str(d)


# ---------------------------------------------------------------------------
# columns in both outputs
# ---------------------------------------------------------------------------

def test_uniform_provenance_becomes_three_columns_after_cultivar_id(
        uniform_dir, master_path, tmp_path):
    csv_df, feather_df, warnings, prov = json2table.build_table(
        uniform_dir, master_path)

    expected = ["cultivar_id"] + PROV_COLS + ["color", "length_cm", "lobing"]
    assert list(csv_df.columns) == expected
    assert list(feather_df.columns) == expected
    assert warnings == []
    assert list(csv_df["provider"]) == ["anthropic", "anthropic"]
    assert list(csv_df["model"]) == ["claude-sonnet-5", "claude-sonnet-5"]
    assert list(csv_df["schema_version"]) == ["3.1", "3.1"]

    # and the same values survive the round trip through both files
    prefix = str(tmp_path / "out")
    json2table.write_table(csv_df, feather_df, prefix, provenance=prov)
    from_csv = pd.read_csv(f"{prefix}.csv")
    from_feather = pd.read_feather(f"{prefix}.feather")
    assert list(from_csv.columns) == expected
    assert list(from_feather.columns) == expected
    for col in PROV_COLS:
        assert list(from_csv[col].astype(str)) == list(from_feather[col].astype(str))


def test_feather_metadata_carries_the_block_without_dropping_pandas(
        uniform_dir, master_path, tmp_path):
    csv_df, feather_df, _warnings, prov = json2table.build_table(
        uniform_dir, master_path)
    prefix = str(tmp_path / "out")
    json2table.write_table(csv_df, feather_df, prefix, provenance=prov)

    metadata = pa_feather.read_table(f"{prefix}.feather").schema.metadata
    assert b"pandas" in metadata            # ordinals depend on this one
    block = json.loads(metadata[b"pxgpt_provenance"])
    assert (block["provider"], block["model"], block["schema_version"]) == (
        "anthropic", "claude-sonnet-5", "3.1")
    assert block["run_id"] == "msgbatch_abc"

    # the behaviour that b"pandas" pays for: an ORDERED categorical, full levels
    lobing = pd.read_feather(f"{prefix}.feather")["lobing"]
    assert isinstance(lobing.dtype, pd.CategoricalDtype)
    assert lobing.dtype.ordered is True
    assert list(lobing.cat.categories) == LOBING_LEVELS


# ---------------------------------------------------------------------------
# reserved column names
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reserved", ["provider", "model", "schema_version",
                                      "cultivar_id"])
def test_a_trait_named_like_a_reserved_column_is_a_collision(tmp_path, reserved):
    master = {
        "schema_name": "probe_master", "schema_version": "3.1",
        "trait_groups": {"leaf": {"traits": [
            {"trait_name": reserved, "scale_type": "nominal", "unit": None,
             "values": ["a", "b"]},
        ]}},
    }
    master_p = tmp_path / "master.json"
    _write_json(master_p, master)
    results = tmp_path / "results"
    results.mkdir()
    _write_json(results / "c1.json", {
        "_provenance": _prov(),
        "leaf": {reserved: {"rationale": "r", "value": "a"}},
    })
    prefix = tmp_path / "out"

    with pytest.raises(json2table.ColumnCollisionError) as excinfo:
        csv_df, feather_df, _w, prov = json2table.build_table(
            str(results), str(master_p))
        json2table.write_table(csv_df, feather_df, str(prefix), provenance=prov)

    message = str(excinfo.value)
    assert "--rename-map" in message
    assert f"'{reserved}'" in message
    assert "never" in message and "renamed" in message
    assert not (tmp_path / "out.csv").exists()
    assert not (tmp_path / "out.feather").exists()


def test_reserved_collision_is_refused_in_the_prefix_modes_too(tmp_path):
    """Auto-prefixing a reserved name away would hide it from the reader."""
    master = {
        "trait_groups": {"leaf": {"traits": [
            {"trait_name": "model", "scale_type": "nominal", "unit": None,
             "values": ["a"]},
        ]}},
    }
    master_p = tmp_path / "master.json"
    _write_json(master_p, master)
    results = tmp_path / "results"
    results.mkdir()
    _write_json(results / "c1.json", {
        "_provenance": _prov(),
        "leaf": {"model": {"rationale": "r", "value": "a"}},
    })

    with pytest.raises(json2table.ColumnCollisionError):
        json2table.build_table(str(results), str(master_p),
                               on_collision="prefix_collided")

    # prefix_all names every column by its full path, so nothing lands on it
    csv_df, _f, _w, _p = json2table.build_table(
        str(results), str(master_p), on_collision="prefix_all")
    assert "leaf_model" in csv_df.columns


# ---------------------------------------------------------------------------
# mixed provenance
# ---------------------------------------------------------------------------

@pytest.fixture
def mixed_dir(tmp_path):
    d = tmp_path / "mixed"
    d.mkdir()
    _write_json(d / "c1.json", _record(prov=_prov()))
    _write_json(d / "c2.json", _record(color="red",
                                       prov=_prov(provider="openai",
                                                  model="gpt-5.6-luna")))
    return str(d)


def test_two_models_in_one_dir_are_refused_by_default(mixed_dir, master_path):
    with pytest.raises(json2table.MixedProvenanceError) as excinfo:
        json2table.build_table(mixed_dir, master_path)

    message = str(excinfo.value)
    assert "claude-sonnet-5" in message and "gpt-5.6-luna" in message
    assert "anthropic" in message and "openai" in message
    assert "c1" in message and "c2" in message
    assert "--allow-mixed-provenance" in message


def test_allow_mixed_writes_per_row_truth_and_flags_the_metadata(
        mixed_dir, master_path, tmp_path):
    csv_df, feather_df, _warnings, prov = json2table.build_table(
        mixed_dir, master_path, allow_mixed_provenance=True)

    assert list(csv_df["model"]) == ["claude-sonnet-5", "gpt-5.6-luna"]
    assert list(csv_df["provider"]) == ["anthropic", "openai"]

    prefix = str(tmp_path / "out")
    json2table.write_table(csv_df, feather_df, prefix, provenance=prov)
    block = json.loads(
        pa_feather.read_table(f"{prefix}.feather").schema.metadata[b"pxgpt_provenance"])
    assert block["mixed"] is True
    assert {v["model"] for v in block["values"]} == {"claude-sonnet-5", "gpt-5.6-luna"}
    assert list(pd.read_csv(f"{prefix}.csv")["model"]) == ["claude-sonnet-5",
                                                           "gpt-5.6-luna"]


def test_a_schema_version_change_alone_counts_as_mixed(tmp_path, master_path):
    d = tmp_path / "two_versions"
    d.mkdir()
    _write_json(d / "c1.json", _record(prov=_prov(schema_version="3.1")))
    _write_json(d / "c2.json", _record(prov=_prov(schema_version="2.0")))

    with pytest.raises(json2table.MixedProvenanceError) as excinfo:
        json2table.build_table(str(d), master_path)
    assert "'2.0'" in str(excinfo.value) and "'3.1'" in str(excinfo.value)


def test_two_merges_of_one_run_are_not_mixed_but_are_named_in_the_metadata(
        tmp_path, master_path):
    """Same identity, different ``created`` — allowed, and the metadata says so."""
    d = tmp_path / "two_merges"
    d.mkdir()
    _write_json(d / "c1.json", _record(prov=_prov(run_id="msgbatch_abc")))
    _write_json(d / "c2.json", _record(prov=_prov(run_id="msgbatch_def")))

    csv_df, _f, _w, prov = json2table.build_table(str(d), master_path)

    assert list(csv_df["model"]) == ["claude-sonnet-5", "claude-sonnet-5"]
    assert prov["mixed"] is True
    assert {v["run_id"] for v in prov["values"]} == {"msgbatch_abc", "msgbatch_def"}


# ---------------------------------------------------------------------------
# legacy directories
# ---------------------------------------------------------------------------

def test_legacy_records_fall_back_to_the_run_stamp(tmp_path, master_path):
    d = tmp_path / "legacy_stamped"
    (d / "_partial").mkdir(parents=True)
    _write_json(d / "c1.json", _record())          # no _provenance
    _write_json(d / "_partial" / ".run.json", {
        "provider": "openai", "model": "gpt-5.6-luna",
        "schema_name": "probe_master", "schema_version": "3.1",
        "created": "2026-08-01T00:00:00Z",
    })

    csv_df, feather_df, warnings, prov = json2table.build_table(str(d), master_path)

    assert len(warnings) == 1
    assert ".run.json" in warnings[0]
    assert list(csv_df["provider"]) == ["openai"]
    assert list(csv_df["model"]) == ["gpt-5.6-luna"]
    assert list(csv_df["schema_version"]) == ["3.1"]
    assert prov["model"] == "gpt-5.6-luna"

    prefix = str(tmp_path / "out")
    json2table.write_table(csv_df, feather_df, prefix, provenance=prov)
    assert list(pd.read_csv(f"{prefix}.csv")["model"]) == ["gpt-5.6-luna"]


def test_legacy_records_with_no_stamp_are_na_with_one_warning(tmp_path, master_path):
    d = tmp_path / "legacy_bare"
    d.mkdir()
    _write_json(d / "c1.json", _record())
    _write_json(d / "c2.json", _record(color="red"))

    csv_df, feather_df, warnings, prov = json2table.build_table(str(d), master_path)

    assert len(warnings) == 1
    assert "NA" in warnings[0]
    assert csv_df["provider"].isna().all()
    assert csv_df["model"].isna().all()
    assert csv_df["schema_version"].isna().all()
    assert len(csv_df) == 2                      # the table still writes

    prefix = str(tmp_path / "out")
    json2table.write_table(csv_df, feather_df, prefix, provenance=prov)
    back = pd.read_feather(f"{prefix}.feather")
    assert back["model"].isna().all()
    assert list(back["color"]) == ["green", "red"]


def test_a_mix_of_stamped_and_legacy_records_is_still_mixed(tmp_path, master_path):
    d = tmp_path / "half_legacy"
    d.mkdir()
    _write_json(d / "c1.json", _record(prov=_prov()))
    _write_json(d / "c2.json", _record())

    with pytest.raises(json2table.MixedProvenanceError):
        json2table.build_table(str(d), master_path)


# ---------------------------------------------------------------------------
# sidecars
# ---------------------------------------------------------------------------

def test_gaps_sidecars_are_not_read_as_records(tmp_path, master_path):
    d = tmp_path / "with_gaps"
    d.mkdir()
    _write_json(d / "c1.json", _record(prov=_prov()))
    _write_json(d / "c1.gaps.json", {
        "line_id": "c1",
        "missing_traits": [{"group": "leaf", "trait": "petiole"}],
        "shard_errors": [],
    })

    csv_df, _f, warnings, _p = json2table.build_table(str(d), master_path)

    assert list(csv_df["cultivar_id"]) == ["c1"]
    assert warnings == []                         # no phantom legacy row either


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_cli(argv):
    parser = argparse.ArgumentParser()
    setup_json2table_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(argv)
    return args.func(args)


def test_cli_refuses_mixed_then_accepts_the_flag(mixed_dir, master_path, tmp_path,
                                                  capsys):
    prefix = tmp_path / "cli_out"
    argv = ["json-to-table", "--result-dir", mixed_dir,
            "--master-schema", master_path, "--out-prefix", str(prefix)]

    assert _run_cli(argv) == 1
    assert "--allow-mixed-provenance" in capsys.readouterr().out
    assert not Path(f"{prefix}.csv").exists()

    assert _run_cli(argv + ["--allow-mixed-provenance"]) == 0
    out = capsys.readouterr().out
    assert "Provenance: MIXED" in out
    df = pd.read_csv(f"{prefix}.csv")
    assert list(df.columns)[:4] == ["cultivar_id"] + PROV_COLS


def test_cli_reports_the_single_provenance_it_wrote(uniform_dir, master_path,
                                                    tmp_path, capsys):
    prefix = tmp_path / "cli_uniform"
    rc = _run_cli(["json-to-table", "--result-dir", uniform_dir,
                   "--master-schema", master_path, "--out-prefix", str(prefix)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "model='claude-sonnet-5'" in out
    assert "schema_version='3.1'" in out
