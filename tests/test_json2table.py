"""Tests for pxgpt.core.json2table (Stage 3 per-cultivar JSON -> wide table)."""

import json

import pandas as pd
import pytest

from pxgpt.core import json2table


MASTER_SCHEMA = {
    "trait_groups": {
        "g1": {
            "traits": [
                {
                    "trait_name": "leaf_color",
                    "scale_type": "nominal",
                    "values": ["green", "red"],
                    "unit": None,
                },
                {
                    "trait_name": "plant_height",
                    "scale_type": "quantitative",
                    "values": None,
                    "unit": "cm",
                },
                {
                    "trait_name": "canopy_area",
                    "scale_type": "quantitative",
                    "values": None,
                    "unit": "m²",
                },
                {
                    "trait_name": "senescence",
                    "scale_type": "ordinal",
                    "values": [
                        {"level": 0, "label": "none", "definition": "d0"},
                        {"level": 1, "label": "mild", "definition": "d1"},
                        {"level": 2, "label": "extensive", "definition": "d2"},
                    ],
                    "unit": None,
                },
            ]
        }
    }
}

SHARD_SCHEMA = {
    "properties": {
        "g2": {
            "properties": {
                "root_vigor": {
                    "properties": {
                        "value": {"enum": ["weak", "strong", "not_assessable"]}
                    }
                }
            }
        }
    }
}


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


@pytest.fixture
def master_path(tmp_path):
    p = tmp_path / "master_schema.json"
    _write_json(p, MASTER_SCHEMA)
    return str(p)


@pytest.fixture
def shard_dir(tmp_path):
    d = tmp_path / "shards"
    d.mkdir()
    _write_json(d / "shard_01.schema.json", SHARD_SCHEMA)
    return str(d)


@pytest.fixture
def result_dir(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    _write_json(d / "c1.json", {
        "g1": {
            "leaf_color": {"rationale": "r", "value": "green"},
            "plant_height": {"rationale": "r", "value": 19.0},
            "canopy_area": {"rationale": "r", "value": 21.0},
            "senescence": {"rationale": "r", "value": 1},
        },
        "g2": {
            "root_vigor": {"rationale": "r", "value": "strong"},
        },
        "g3": {
            "mystery_trait": {"rationale": "r", "value": "foo"},
        },
    })
    _write_json(d / "c2.json", {
        "g1": {
            "leaf_color": {"rationale": "r", "value": "not_assessable"},
            # plant_height entirely absent -> NA
            "canopy_area": {"rationale": "r", "value": 15.5},
            "senescence": {"rationale": "r", "value": "not_assessable"},
        },
    })
    _write_json(d / "c3.json", {
        "g1": {
            "leaf_color": {"rationale": "r", "value": "red"},
            "plant_height": {"rationale": "r", "value": 30.0},
            "canopy_area": {"rationale": "r", "value": 40.0},
            "senescence": {"rationale": "r", "value": 2},
        },
        "g2": {
            "root_vigor": {"rationale": "r", "value": "weak"},
        },
    })
    return str(d)


@pytest.fixture
def built(master_path, shard_dir, result_dir):
    return json2table.build_table(result_dir, master_path, shard_dir=shard_dir)


def test_row_count_and_cultivar_id_column(built):
    csv_df, feather_df, warnings = built
    assert len(csv_df) == 3
    assert len(feather_df) == 3
    assert list(csv_df.columns)[0] == "cultivar_id"
    assert list(feather_df.columns)[0] == "cultivar_id"
    assert sorted(csv_df["cultivar_id"]) == ["c1", "c2", "c3"]


def test_nominal_column_stays_string_in_both_outputs(built):
    csv_df, feather_df, warnings = built
    assert "leaf_color" in csv_df.columns
    row = csv_df.set_index("cultivar_id")
    assert row.loc["c1", "leaf_color"] == "green"
    assert pd.isna(row.loc["c2", "leaf_color"])  # not_assessable -> NA

    frow = feather_df.set_index("cultivar_id")
    assert not isinstance(feather_df["leaf_color"].dtype, pd.CategoricalDtype)
    assert frow.loc["c1", "leaf_color"] == "green"


def test_quantitative_column_uses_sanitized_unit_and_is_numeric(built):
    csv_df, feather_df, warnings = built
    assert "plant_height_cm" in csv_df.columns
    assert "canopy_area_m2" in csv_df.columns  # superscript sanitized
    row = csv_df.set_index("cultivar_id")
    assert row.loc["c1", "plant_height_cm"] == pytest.approx(19.0)
    assert pd.isna(row.loc["c2", "plant_height_cm"])  # trait absent for c2
    assert pd.api.types.is_numeric_dtype(csv_df["plant_height_cm"])
    assert pd.api.types.is_numeric_dtype(feather_df["canopy_area_m2"])


def test_ordinal_reconstructs_label_and_feather_is_ordered_categorical(built):
    csv_df, feather_df, warnings = built
    row = csv_df.set_index("cultivar_id")
    assert row.loc["c1", "senescence"] == "mild"
    assert row.loc["c3", "senescence"] == "extensive"

    col = feather_df.set_index("cultivar_id")["senescence"]
    assert isinstance(feather_df["senescence"].dtype, pd.CategoricalDtype)
    assert feather_df["senescence"].cat.ordered is True
    assert list(feather_df["senescence"].cat.categories) == ["none", "mild", "extensive"]
    assert col.loc["c1"] == "mild"


def test_ordinal_not_assessable_becomes_na_and_not_a_category(built):
    csv_df, feather_df, warnings = built
    row = csv_df.set_index("cultivar_id")
    assert pd.isna(row.loc["c2", "senescence"])
    assert "not_assessable" not in list(feather_df["senescence"].cat.categories)
    frow = feather_df.set_index("cultivar_id")
    assert pd.isna(frow.loc["c2", "senescence"])


def test_shard_fallback_trait_included_as_nominal(built):
    csv_df, feather_df, warnings = built
    assert "root_vigor" in csv_df.columns
    row = csv_df.set_index("cultivar_id")
    assert row.loc["c1", "root_vigor"] == "strong"
    assert row.loc["c3", "root_vigor"] == "weak"
    assert pd.isna(row.loc["c2", "root_vigor"])  # absent for c2


def test_unknown_trait_logged_and_included_as_string_fallback(built):
    csv_df, feather_df, warnings = built
    assert "mystery_trait" in csv_df.columns
    row = csv_df.set_index("cultivar_id")
    assert row.loc["c1", "mystery_trait"] == "foo"
    assert pd.isna(row.loc["c2", "mystery_trait"])
    assert any("mystery_trait" in w for w in warnings)


def test_deterministic_column_order(built):
    csv_df, feather_df, warnings = built
    cols = list(csv_df.columns)
    # master-schema traits come first (in schema order), shard fallback next,
    # unknown traits last; identical between csv and feather outputs.
    assert cols == list(feather_df.columns)
    assert cols.index("leaf_color") < cols.index("plant_height_cm")
    assert cols.index("plant_height_cm") < cols.index("canopy_area_m2")
    assert cols.index("canopy_area_m2") < cols.index("senescence")
    assert cols.index("senescence") < cols.index("root_vigor")
    assert cols.index("root_vigor") < cols.index("mystery_trait")


def test_write_table_round_trip(built, tmp_path):
    csv_df, feather_df, warnings = built
    out_prefix = str(tmp_path / "out" / "table")
    json2table.write_table(csv_df, feather_df, out_prefix)

    reread_csv = pd.read_csv(out_prefix + ".csv")
    assert len(reread_csv) == 3

    reread_feather = pd.read_feather(out_prefix + ".feather")
    assert len(reread_feather) == 3
    assert isinstance(reread_feather["senescence"].dtype, pd.CategoricalDtype)
    assert reread_feather["senescence"].cat.ordered is True
