"""Tests for json2table column-name collision handling (on_collision / rename_map).

Fixtures mirror the worked examples in the design spec (Examples A-E): same
leaf key appearing under two organ groups, a three-level nested path that
needs to auto-deepen past one prefix level, a same-name-different-unit
non-collision, and a rename_map that itself collides (safety net).
"""

import argparse
import json

import pytest

from pxgpt.core import json2table
from pxgpt.commands.json2table import setup_json2table_parser


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _trait(name, scale_type, unit=None, values=None):
    return {"trait_name": name, "scale_type": scale_type, "unit": unit, "values": values}


# ---------------------------------------------------------------------------
# Example A: leaf.length / petal.length both -> length_cm
# ---------------------------------------------------------------------------

MASTER_A = {
    "trait_groups": {
        "leaf": {"traits": [_trait("length", "quantitative", unit="cm")]},
        "petal": {"traits": [_trait("length", "quantitative", unit="cm")]},
        "flower": {"traits": [_trait("color", "nominal", values=["purple", "white"])]},
    }
}


def _write_result_a(d, name, height=12.5, petal_len=3.1, color="purple"):
    _write_json(d / name, {
        "leaf": {"length": {"rationale": "r", "value": height}},
        "petal": {"length": {"rationale": "r", "value": petal_len}},
        "flower": {"color": {"rationale": "r", "value": color}},
    })


@pytest.fixture
def master_a(tmp_path):
    p = tmp_path / "master_a.json"
    _write_json(p, MASTER_A)
    return str(p)


@pytest.fixture
def results_a(tmp_path):
    d = tmp_path / "results_a"
    d.mkdir()
    _write_result_a(d, "c1.json")
    return str(d)


def test_example_a1_error_default_raises_with_fill_in_template(master_a, results_a):
    with pytest.raises(json2table.ColumnCollisionError) as excinfo:
        json2table.build_table(results_a, master_a)

    message = str(excinfo.value)
    expected = (
        "Column name collision(s). Unresolved:\n"
        " 'length_cm' <- leaf.length (quantitative, unit=cm), "
        "petal.length (quantitative, unit=cm)\n"
        "\n"
        "Fill in this rename map (path -> column name) and re-run with --rename-map:\n"
        " {\n"
        ' "leaf.length": "",\n'
        ' "petal.length": ""\n'
        " }"
    )
    assert message == expected


def test_example_a2_prefix_collided_only_touches_clashing_columns(master_a, results_a):
    csv_df, feather_df, warnings = json2table.build_table(
        results_a, master_a, on_collision="prefix_collided"
    )
    assert list(csv_df.columns) == [
        "cultivar_id", "leaf_length_cm", "petal_length_cm", "color",
    ]
    assert list(feather_df.columns) == list(csv_df.columns)


def test_example_a3_prefix_all_uses_full_path_for_every_column(master_a, results_a):
    csv_df, feather_df, warnings = json2table.build_table(
        results_a, master_a, on_collision="prefix_all"
    )
    assert list(csv_df.columns) == [
        "cultivar_id", "leaf_length_cm", "petal_length_cm", "flower_color",
    ]
    assert list(feather_df.columns) == list(csv_df.columns)


def test_example_a4_rename_map_used_verbatim(master_a, results_a):
    rename_map = {"leaf.length": "leaf_len_cm", "petal.length": "petal_len_cm"}
    csv_df, feather_df, warnings = json2table.build_table(
        results_a, master_a, rename_map=rename_map
    )
    assert list(csv_df.columns) == [
        "cultivar_id", "leaf_len_cm", "petal_len_cm", "color",
    ]
    assert list(feather_df.columns) == list(csv_df.columns)


# ---------------------------------------------------------------------------
# Example B: sepal.tip.angle / bract.tip.angle -> one-level prefix insufficient
# ---------------------------------------------------------------------------

MASTER_B = {
    "trait_groups": {
        "sepal": {"traits": [_trait("angle", "quantitative", unit="deg")]},
        "bract": {"traits": [_trait("angle", "quantitative", unit="deg")]},
    }
}


@pytest.fixture
def master_b(tmp_path):
    p = tmp_path / "master_b.json"
    _write_json(p, MASTER_B)
    return str(p)


@pytest.fixture
def results_b(tmp_path):
    d = tmp_path / "results_b"
    d.mkdir()
    _write_json(d / "c1.json", {
        "sepal": {"tip": {"angle": {"rationale": "r", "value": 30}}},
        "bract": {"tip": {"angle": {"rationale": "r", "value": 45}}},
    })
    return str(d)


def test_example_b_prefix_collided_auto_deepens_past_one_level(master_b, results_b):
    csv_df, feather_df, warnings = json2table.build_table(
        results_b, master_b, on_collision="prefix_collided"
    )
    assert list(csv_df.columns) == [
        "cultivar_id", "sepal_tip_angle_deg", "bract_tip_angle_deg",
    ]
    assert list(feather_df.columns) == list(csv_df.columns)


# ---------------------------------------------------------------------------
# Example C: regression, no collision, default error mode is a no-op
# ---------------------------------------------------------------------------

MASTER_C = {
    "trait_groups": {
        "whole_plant": {"traits": [_trait("plant_height", "quantitative", unit="cm")]},
        "leaf": {"traits": [_trait("leaf_color", "nominal", values=["green", "red"])]},
        "vigor_group": {"traits": [_trait("vigor", "ordinal", values=[
            {"level": 0, "label": "low", "definition": "d0"},
            {"level": 1, "label": "medium", "definition": "d1"},
            {"level": 2, "label": "high", "definition": "d2"},
        ])]},
    }
}


@pytest.fixture
def master_c(tmp_path):
    p = tmp_path / "master_c.json"
    _write_json(p, MASTER_C)
    return str(p)


@pytest.fixture
def results_c(tmp_path):
    d = tmp_path / "results_c"
    d.mkdir()
    _write_json(d / "c1.json", {
        "whole_plant": {"plant_height": {"rationale": "r", "value": 40}},
        "leaf": {"leaf_color": {"rationale": "r", "value": "green"}},
        "vigor_group": {"vigor": {"rationale": "r", "value": 1}},
    })
    return str(d)


def test_example_c_regression_no_collision_default_error_is_noop(master_c, results_c):
    csv_df, feather_df, warnings = json2table.build_table(results_c, master_c)
    assert list(csv_df.columns) == [
        "cultivar_id", "plant_height_cm", "leaf_color", "vigor",
    ]
    assert list(feather_df.columns) == list(csv_df.columns)
    assert warnings == []


# ---------------------------------------------------------------------------
# Example D: same trait_name, different unit -> must NOT be flagged
# ---------------------------------------------------------------------------

MASTER_D = {
    "trait_groups": {
        "stem": {"traits": [_trait("length", "quantitative", unit="cm")]},
        "hair": {"traits": [_trait("length", "quantitative", unit="mm")]},
    }
}


@pytest.fixture
def master_d(tmp_path):
    p = tmp_path / "master_d.json"
    _write_json(p, MASTER_D)
    return str(p)


@pytest.fixture
def results_d(tmp_path):
    d = tmp_path / "results_d"
    d.mkdir()
    _write_json(d / "c1.json", {
        "stem": {"length": {"rationale": "r", "value": 40}},
        "hair": {"length": {"rationale": "r", "value": 2}},
    })
    return str(d)


def test_example_d_same_name_different_unit_not_flagged(master_d, results_d):
    csv_df, feather_df, warnings = json2table.build_table(results_d, master_d)
    assert list(csv_df.columns) == ["cultivar_id", "length_cm", "length_mm"]
    assert list(feather_df.columns) == list(csv_df.columns)


# ---------------------------------------------------------------------------
# Example E: rename_map itself duplicates -> final safety net raises
# ---------------------------------------------------------------------------

def test_example_e_rename_map_self_collision_raises_safety_net(master_a, results_a):
    rename_map = {"leaf.length": "len", "petal.length": "len"}
    with pytest.raises(json2table.DuplicateColumnError):
        json2table.build_table(results_a, master_a, rename_map=rename_map)


def test_unknown_on_collision_mode_rejected(master_a, results_a):
    with pytest.raises(ValueError):
        json2table.build_table(results_a, master_a, on_collision="bogus")


# ---------------------------------------------------------------------------
# CLI wiring: --on-collision / --rename-map, and "error" mode writes no files
# ---------------------------------------------------------------------------

def _parse_cli_args(argv):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    setup_json2table_parser(subparsers)
    return parser.parse_args(["json-to-table"] + argv)


def test_cli_error_mode_prints_template_and_writes_no_files(master_a, results_a, tmp_path, capsys):
    out_prefix = str(tmp_path / "out" / "table")
    args = _parse_cli_args([
        "--result-dir", results_a, "--master-schema", master_a, "--out-prefix", out_prefix,
    ])
    rc = args.func(args)
    assert rc == 1

    captured = capsys.readouterr()
    assert "Column name collision(s). Unresolved:" in captured.out
    assert "Fill in this rename map" in captured.out
    assert not (tmp_path / "out" / "table.csv").exists()
    assert not (tmp_path / "out" / "table.feather").exists()


def test_cli_prefix_collided_writes_files(master_a, results_a, tmp_path):
    out_prefix = str(tmp_path / "out" / "table")
    args = _parse_cli_args([
        "--result-dir", results_a, "--master-schema", master_a, "--out-prefix", out_prefix,
        "--on-collision", "prefix_collided",
    ])
    rc = args.func(args)
    assert rc == 0
    assert (tmp_path / "out" / "table.csv").exists()
    assert (tmp_path / "out" / "table.feather").exists()


def test_cli_rename_map_from_file(master_a, results_a, tmp_path):
    rename_map_path = tmp_path / "rename_map.json"
    rename_map_path.write_text(
        json.dumps({"leaf.length": "leaf_len_cm", "petal.length": "petal_len_cm"})
    )
    out_prefix = str(tmp_path / "out" / "table")
    args = _parse_cli_args([
        "--result-dir", results_a, "--master-schema", master_a, "--out-prefix", out_prefix,
        "--rename-map", str(rename_map_path),
    ])
    rc = args.func(args)
    assert rc == 0

    import pandas as pd
    df = pd.read_csv(out_prefix + ".csv")
    assert list(df.columns) == ["cultivar_id", "leaf_len_cm", "petal_len_cm", "color"]
