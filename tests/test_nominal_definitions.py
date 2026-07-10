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


def test_group_section_renders_per_category_definitions():
    tr = {"trait_name": "plant_growth_habit", "scale_type": "nominal",
          "description": "Overall silhouette.",
          "values": [{"value": "compact_rosette", "definition": "flat/tight rosette"},
                     {"value": "upright_erect", "definition": "leaves held vertically"}]}
    out = sb.group_section("whole_plant_architecture", "", [tr])
    assert "- `compact_rosette`: flat/tight rosette" in out
    assert "- `upright_erect`: leaves held vertically" in out
    assert "not_assessable" in out
    # bare comma-joined form must NOT be used when definitions exist
    assert "Allowed categories: `compact_rosette`," not in out


def test_group_section_bare_nominal_uses_oneline_form():
    tr = {"trait_name": "t", "scale_type": "nominal", "description": "d",
          "values": ["a", "b"]}
    out = sb.group_section("g", "", [tr])
    assert "Allowed categories: `a`, `b`, `not_assessable`." in out


def test_extract_defs_parses_semicolon_clauses():
    from tools import migrate_nominal_definitions as mig
    note = ("Overall silhouette. compact_rosette = flat/tight rosette; "
            "upright_erect = leaves held vertically; "
            "open_spreading = long petioles splaying outward. Score the dominant form.")
    got = mig.extract_defs(note, ["compact_rosette", "upright_erect", "open_spreading"])
    assert got["compact_rosette"] == "flat/tight rosette"
    assert got["upright_erect"] == "leaves held vertically"
    assert got["open_spreading"] == "long petioles splaying outward"


def test_migrate_master_reports_unresolved():
    from tools import migrate_nominal_definitions as mig
    master = {"trait_groups": {"g": {"description": "", "traits": [
        {"trait_name": "t", "scale_type": "nominal",
         "design_note": "x = defx",
         "values": ["x", "y"]}]}}}
    new, unresolved = mig.migrate_master(master)
    vals = new["trait_groups"]["g"]["traits"][0]["values"]
    assert {"value": "x", "definition": "defx"} in vals
    assert "g/t/y" in unresolved


def test_extract_defs_keeps_internal_periods_and_trims_trailing_sentence():
    from tools import migrate_nominal_definitions as mig
    note = ("Overall silhouette. compact_rosette = flat/tight rosette; "
            "open_spreading = long petioles splaying outward. Score the dominant form.")
    got = mig.extract_defs(note, ["compact_rosette", "open_spreading"])
    assert got["compact_rosette"] == "flat/tight rosette"
    assert got["open_spreading"] == "long petioles splaying outward"

def test_extract_defs_preserves_abbreviations_and_decimals():
    from tools import migrate_nominal_definitions as mig
    note = "toothed = has teeth (e.g. serrate) along margin; tall = grows 5.5 cm."
    got = mig.extract_defs(note, ["toothed", "tall"])
    assert got["toothed"] == "has teeth (e.g. serrate) along margin"
    assert got["tall"] == "grows 5.5 cm"

def test_extract_defs_word_boundary_no_suffix_bleed():
    from tools import migrate_nominal_definitions as mig
    note = "semi_erect = leaning; erect = fully upright."
    got = mig.extract_defs(note, ["erect", "semi_erect"])
    assert got["erect"] == "fully upright"
    assert got["semi_erect"] == "leaning"
