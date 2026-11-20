"""The names of the profiled diseases (20 November 2026), without network."""

import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "disease_names", Path(__file__).resolve().parents[1] / "scripts" / "disease_names.py")
names = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(names)

MONDO = """format-version: 1.2

[Term]
id: MONDO:0004974
name: acute appendicitis
synonym: "appendicitis, acute" EXACT []
"""


def test_the_query_asks_both_ways_of_writing_the_id():
    q = names.query(["MONDO:0004974"])
    assert '"MONDO:0004974" "0004974"' in q and "wdt:P5270" in q


def test_wikidata_answers_are_read_with_the_prefix():
    payload = {"results": {"bindings": [{"mondo": {"value": "0004974"}, "name": {"value": "appendicite acuta"}}]}}
    assert names.read(payload) == {"MONDO:0004974": {"appendicite acuta"}}


def test_the_file_joins_mondo_and_wikidata(tmp_path, monkeypatch):
    (tmp_path / "mondo.obo").write_text(MONDO)
    (tmp_path / "profiles.json").write_text(json.dumps({"profiles": [{"disease_id": "MONDO:0004974"}]}))
    monkeypatch.setattr(names, "ask", lambda q: {"results": {"bindings": [
        {"mondo": {"value": "MONDO:0004974"}, "name": {"value": "appendicite acuta"}}]}})
    assert names.main(["--profiles", str(tmp_path / "profiles.json"), "--mondo", str(tmp_path / "mondo.obo"),
                       "--out", str(tmp_path / "names.json")]) == 0
    out = json.loads((tmp_path / "names.json").read_text())
    assert out["MONDO:0004974"] == ["acute appendicitis", "appendicite acuta", "appendicitis, acute"]
