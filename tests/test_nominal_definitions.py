from pxgpt.core import shard_builder as sb


def _nominal(values):
    return {"trait_name": "t", "scale_type": "nominal", "values": values}


def test_nominal_categories_accepts_bare_strings():
    pairs = sb.nominal_categories(_nominal(["a", "b"]))
    assert pairs == [("a", None), ("b", None)]


def test_nominal_categories_accepts_objects():
    pairs = sb.nominal_categories(_nominal(
        [{"value": "a", "definition": "def a"}, {"value": "b"}]))
    assert pairs == [("a", "def a"), ("b", None)]


def test_value_schema_enum_is_bare_tokens_for_object_values():
    schema = sb.value_schema(_nominal(
        [{"value": "a", "definition": "def a"}, {"value": "b", "definition": "def b"}]))
    assert schema == {"enum": ["a", "b", sb.NA]}


def test_value_schema_enum_bare_strings_unchanged():
    assert sb.value_schema(_nominal(["a", "b"])) == {"enum": ["a", "b", sb.NA]}
