"""The systematic review of a disease (22 November 2026), without network: layers, screening, saturation as a
parameter, the profile with its weights."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "disease_review", Path(__file__).resolve().parents[1] / "scripts" / "disease_review.py")
dr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dr)

MONDO = """format-version: 1.2

[Term]
id: MONDO:0004974
name: acute appendicitis
synonym: "appendicitis, acute" EXACT []
"""
HPO = """format-version: 1.2

[Term]
id: HP:0001945
name: Fever

[Term]
id: HP:0002017
name: Nausea and vomiting

[Term]
id: HP:0100281
name: Right lower quadrant pain
"""


def test_saturation_is_a_parameter_at_least_five_and_zero_is_no_limit():
    assert dr.saturation_of(5) == 5 and dr.saturation_of(12) == 12
    assert dr.saturation_of(2) == 5                       # raised to the least
    assert dr.saturation_of(0) == 0                       # no limit: to the end of the sources


def test_every_page_of_a_layer_is_read_to_the_end():
    pages = {"*": {"resultList": {"result": [{"pmid": "1"}, {"pmid": "2"}]}, "nextCursorMark": "A"},
             "A": {"resultList": {"result": [{"pmid": "3"}]}, "nextCursorMark": "B"},
             "B": {"resultList": {"result": []}, "nextCursorMark": "B"}}

    def get(url, accept=""):
        cursor = url.split("cursorMark=")[1].split("&")[0]
        return json.dumps(pages[{"%2A": "*"}.get(cursor, cursor)])
    assert [r["pmid"] for r in dr.records(["acute appendicitis"], "guideline", get)] == ["1", "2", "3"]


def _files(tmp_path):
    for name, body in (("mondo.obo", MONDO), ("hp.obo", HPO), ("phenotype.hpoa", "#\n"), ("doid.owl", ""),
                       ("symp.obo", "format-version: 1.2\n")):
        (tmp_path / name).write_text(body)
    return ["--mondo", str(tmp_path / "mondo.obo"), "--hpo", str(tmp_path / "hp.obo"), "--hpoa",
            str(tmp_path / "phenotype.hpoa"), "--doid-owl", str(tmp_path / "doid.owl"), "--symp",
            str(tmp_path / "symp.obo")]


@pytest.fixture
def world(monkeypatch):
    """Twelve abstracts of acute appendicitis: the first states two manifestations, the next ones none new."""
    from melampo.memory import literature_connectors as lc
    from melampo.memory import symptom_sources

    monkeypatch.setattr(symptom_sources, "parse_doid_owl", lambda path: [])
    monkeypatch.setattr(lc, "read_sources", lambda *a, **k: lc.Reading())
    first = "In acute appendicitis, right lower quadrant pain was present in 80 of 100 patients (80%), fever often."
    later = "Acute appendicitis is diagnosed clinically."
    abstracts = [{"pmid": str(i), "title": f"Clinical presentation of acute appendicitis {i}",
                  "abstractText": first if i == 0 else later} for i in range(12)]

    def records(names, layer, get=None):
        yield from (abstracts if layer == "guideline" else [])
    monkeypatch.setattr(dr, "records", records)
    monkeypatch.setattr(dr.rs, "text_of", lambda r: {"id": r["pmid"], "read": "abstract",
                                                    "source": {"pmid": r["pmid"], "title": r["title"]},
                                                    "sections": [("Abstract", r["abstractText"])]})
    monkeypatch.setattr(dr.rs, "epmc", lambda q, size=15: [])
    calls = {"read": 0, "screen": 0, "judge": 0}

    def chat(self, model, prompt, *, reasoning, max_tokens):
        if prompt.startswith("Disease:") and "Is this publication" in prompt:
            calls["screen"] += 1
            return '{"about": "yes"}', "stop"
        if prompt.startswith("Disease:"):
            calls["judge"] += 1
            return '{"states": "yes", "category": "manifestation", "same_as": "none"}', "stop"
        calls["read"] += 1
        if first in prompt:
            return json.dumps([
                {"manifestation": "right lower quadrant pain", "with_disease": {"count": 80, "total": 100,
                                                                             "percent": "80%"},
                 "quote": "In acute appendicitis, right lower quadrant pain was present in 80 of 100 patients (80%), "
                          "fever often."},
                {"manifestation": "fever", "verbal": "often",
                 "quote": "In acute appendicitis, right lower quadrant pain was present in 80 of 100 patients (80%), "
                          "fever often."}]), "stop"
        return "[]", "stop"
    monkeypatch.setattr(dr.rs.Router, "chat", chat)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    return calls


def test_the_review_stops_after_saturation_texts_that_add_nothing(tmp_path, world):
    out = tmp_path / "out" / "profiles.json"
    assert dr.main(_files(tmp_path) + ["--diagnoses", "acute appendicitis", "--saturation", "5",
                                       "--out", str(out)]) == 0
    summary = json.loads(out.with_name("summary.json").read_text())
    row = summary["rows"][0]
    assert row["texts_read"] == 6 and row["stopped_by"] == "saturation"     # 1 that adds, then 5 that do not
    profile = json.loads(out.read_text())["profiles"][0]
    by = {m["concept"]["code"]: m for m in profile["manifestations"]}     # the source's words, the HPO code beside
    assert by["HP:0100281"]["frequency"] == 0.8 and by["HP:0100281"]["words"] == "right lower quadrant pain"
    assert by["HP:0001945"]["frequency_basis"] == "verbal"
    assert by["HP:0100281"]["citations"][0]["quote"].startswith("In acute appendicitis")
    curve = json.loads(out.with_name("curves.json").read_text())["acute appendicitis"]
    assert [c["new"] for c in curve] == [2, 0, 0, 0, 0, 0]


def test_with_no_limit_every_text_is_read_to_the_end_of_the_sources(tmp_path, world):
    out = tmp_path / "out" / "profiles.json"
    assert dr.main(_files(tmp_path) + ["--diagnoses", "acute appendicitis", "--saturation", "0",
                                       "--out", str(out)]) == 0
    row = json.loads(out.with_name("summary.json").read_text())["rows"][0]
    assert row["texts_read"] == 12 and row["stopped_by"] == "end_of_sources"


def test_a_resumed_review_asks_nothing_again(tmp_path, world):
    out = tmp_path / "out" / "profiles.json"
    dr.main(_files(tmp_path) + ["--diagnoses", "acute appendicitis", "--out", str(out)])
    before = dict(world)
    again = tmp_path / "again" / "profiles.json"
    dr.main(_files(tmp_path) + ["--diagnoses", "acute appendicitis", "--out", str(again),
                                "--resume", str(out.parent)])
    assert world == before
