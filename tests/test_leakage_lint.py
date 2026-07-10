# tests/test_leakage_lint.py
import json, os, tempfile
from collections import OrderedDict
from pxgpt.core import shard_builder as sb

def test_find_leakage_flags_population_phrases():
    assert sb.find_leakage("spiral_alternate (most rosettes), the rare pairing") == ["most", "rare"]
    assert "near-universal" in sb.find_leakage("Near-universal rosette here.")
    assert sb.find_leakage("12 of the described cases are abaxial-dominant")

def test_find_leakage_clean_text_is_empty():
    assert sb.find_leakage("flat/tight rosette; leaves held vertically") == []
    # botanical 'typical' must not false-positive
    assert sb.find_leakage("narrow, several times longer than wide") == []

def _master(defs_note):
    return {
        "trait_groups": OrderedDict([
            ("g", {"description": "grp",
                   "traits": [{"trait_name": "t", "scale_type": "nominal",
                               "description": defs_note,
                               "values": [{"value": "a", "definition": defs_note},
                                          {"value": "b", "definition": "clean def"}]}]}),
        ])
    }

def test_generate_shards_reports_leakage_and_strict_fails(tmp_path=None):
    d = tempfile.mkdtemp()
    mp = os.path.join(d, "m.json")
    with open(mp, "w") as f:
        json.dump(_master("mostly rosette, rare otherwise"), f)
    # non-strict: reported but not a hard failure
    s = sb.generate_shards(mp, os.path.join(d, "s1"), budget=40)
    flat = [p for _, ph in s["leakage"] for p in ph]
    assert "most" in flat or "mostly" in flat
    assert not any("leakage" in p.lower() for p in s["problems"])
    # strict: leakage becomes a problem
    s2 = sb.generate_shards(mp, os.path.join(d, "s2"), budget=40, strict_leakage=True)
    assert any("leakage" in p.lower() for p in s2["problems"])
