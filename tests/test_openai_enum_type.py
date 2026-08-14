"""OpenAI shard-schema primitives: enum ``type`` injection and format naming.

The frozen shard schemas write enum leaves as ``{"enum": [...]}`` with no
``"type"``.  Anthropic and xgrammar accept that; OpenAI strict mode does not.
The fix has to happen inside the OpenAI normalizer — the shard set on disk is
frozen — and it must not change which values are accepted.

Fixture below mirrors the shard shape (group -> trait -> {rationale, value}).
It does not read the real shard directory.
"""

import copy

from pxgpt.core.openai_batch_utils import (
    openai_normalize_schema,
    schema_format_name,
)


def _fixture_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "stage3_shard_01",
        "type": "object",
        "properties": {
            "whole_plant_architecture": {
                "type": "object",
                "properties": {
                    # all-string enum, no "type" — the shard shape to be fixed
                    "plant_growth_habit": {
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string"},
                            "value": {"enum": ["compact_rosette",
                                               "upright_erect",
                                               "not_assessable"]},
                        },
                        "required": ["rationale", "value"],
                        "additionalProperties": False,
                    },
                    # plain typed string, no enum — must be left as-is
                    "plant_note": {
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["rationale", "value"],
                        "additionalProperties": False,
                    },
                    # mixed-type enum — must NOT be guessed at
                    "plant_count": {
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string"},
                            "value": {"enum": ["many", 3, None]},
                        },
                        "required": ["rationale", "value"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    }


def _value_node(normalized, trait):
    return (normalized["properties"]["whole_plant_architecture"]
            ["properties"][trait]["properties"]["value"])


def test_string_enum_gains_type_string():
    out = openai_normalize_schema(_fixture_schema())
    assert _value_node(out, "plant_growth_habit")["type"] == "string"


def test_enum_members_and_order_unchanged():
    original = _fixture_schema()
    expected = (original["properties"]["whole_plant_architecture"]["properties"]
                ["plant_growth_habit"]["properties"]["value"]["enum"])
    out = openai_normalize_schema(original)
    assert _value_node(out, "plant_growth_habit")["enum"] == expected


def test_mixed_type_enum_is_left_alone():
    out = openai_normalize_schema(_fixture_schema())
    node = _value_node(out, "plant_count")
    assert "type" not in node
    assert node["enum"] == ["many", 3, None]


def test_plain_typed_string_untouched():
    out = openai_normalize_schema(_fixture_schema())
    assert _value_node(out, "plant_note") == {"type": "string"}


def test_strict_object_behaviour_unchanged():
    out = openai_normalize_schema(_fixture_schema())
    trait = (out["properties"]["whole_plant_architecture"]["properties"]
             ["plant_growth_habit"])
    assert trait["additionalProperties"] is False
    assert trait["required"] == ["rationale", "value"]
    group = out["properties"]["whole_plant_architecture"]
    assert group["additionalProperties"] is False
    assert group["required"] == ["plant_growth_habit", "plant_note", "plant_count"]
    assert out["additionalProperties"] is False


def test_input_schema_not_mutated_in_place():
    original = _fixture_schema()
    before = copy.deepcopy(original)
    openai_normalize_schema(original)
    assert original == before


def test_schema_format_name_from_title():
    assert schema_format_name({"title": "stage3_shard_01"}) == "stage3_shard_01"


def test_schema_format_name_sanitises():
    assert schema_format_name({"title": "stage3 shard/01!"}) == "stage3_shard_01_"


def test_schema_format_name_falls_back():
    assert schema_format_name({}) == "structured_output"
    assert schema_format_name({"title": ""}) == "structured_output"
    assert schema_format_name({"title": None}) == "structured_output"
    assert schema_format_name({}, fallback="shard") == "shard"


def test_schema_format_name_reads_title_before_normalize_strips_it():
    """Ordering guard: the name must be taken from the raw schema."""
    raw = _fixture_schema()
    name = schema_format_name(raw)
    normalized = openai_normalize_schema(raw)
    assert name == "stage3_shard_01"
    assert "title" not in normalized
    assert schema_format_name(normalized) == "structured_output"
