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

    def records(names, layer, get=None, since=""):
        world_since.append(since)
        later_ones = [a for a in abstracts if int(a["pmid"]) >= 100]
        yield from (later_ones if since else abstracts) if layer == "guideline" else []
    world_since: list = []
    monkeypatch.setattr(dr, "records", records)
    monkeypatch.setattr(dr.rs, "text_of", lambda r: {"id": r["pmid"], "read": "abstract",
                                                    "source": {"pmid": r["pmid"], "title": r["title"]},
                                                    "sections": [("Abstract", r["abstractText"])]})
    monkeypatch.setattr(dr.rs, "epmc", lambda q, size=15: [])
    calls = {"read": 0, "screen": 0, "judge": 0}
    monkeypatch.setattr(dr, "_world", {"abstracts": abstracts, "since": world_since}, raising=False)

    def chat(self, model, prompt, *, reasoning, max_tokens):
        if prompt.startswith("Disease:") and "Is this publication" in prompt:
            calls["screen"] += 1
            return '{"about": "yes"}', "stop"
        if prompt.startswith("Disease:"):
            calls["judge"] += 1
            return '{"states": "yes"}', "stop"
        calls["read"] += 1
        if first in prompt:
            return json.dumps([
                {"words": "right lower quadrant pain", "name": "right lower quadrant pain",
                 "category": "manifestation", "specific": True, "observed_by": "history", "nature": "pathological",
                 "hpo_term": "Right lower quadrant pain", "same_as": "",
                 "with_disease": {"count": 80, "total": 100, "percent": "80%"},
                 "quote": "In acute appendicitis, right lower quadrant pain was present in 80 of 100 patients (80%), "
                          "fever often."},
                {"words": "fever", "name": "fever", "category": "manifestation", "specific": True,
                 "observed_by": "examination", "nature": "pathological", "hpo_term": "Fever", "same_as": "",
                 "verbal": "often",
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


def test_the_weekly_update_reads_only_what_came_after_and_keeps_every_measure(tmp_path, world):
    """The first review goes to the knowledge folder; the update asks Europe PMC only for what was published from the
    day of that review, and computes the weights again on the old and the new measures together."""
    first = tmp_path / "first" / "profiles.json"
    assert dr.main(_files(tmp_path) + ["--diagnoses", "acute appendicitis", "--out", str(first)]) == 0
    publish = importlib.util.spec_from_file_location(
        "knowledge_publish", Path(__file__).resolve().parents[1] / "scripts" / "knowledge_publish.py")
    kp = importlib.util.module_from_spec(publish)
    publish.loader.exec_module(kp)
    store = tmp_path / "knowledge"
    assert kp.main(["--run", str(first.parent), "--knowledge", str(store)]) == 0
    reviews = json.loads((store / "reviews.json").read_text())
    assert reviews["acute appendicitis"]["stopped_by"] == "saturation"
    measures = [json.loads(x) for x in (store / "measures.jsonl").read_text().splitlines()]
    assert len(measures) == 2
    # a week later: one new abstract states vomiting with a count
    dr._world["abstracts"].append({"pmid": "100", "title": "Presentation of acute appendicitis in 2026",
                                   "abstractText": "first"})
    update = tmp_path / "update" / "profiles.json"
    assert dr.main(_files(tmp_path) + ["--update", "--knowledge", str(store), "--saturation", "0",
                                       "--targeted", "0", "--resume", str(store / "logs"), "--out", str(update)]) == 0
    assert dr._world["since"][-1] == reviews["acute appendicitis"]["date"]
    row = json.loads(update.with_name("summary.json").read_text())["rows"][0]
    assert row["texts_read"] == 1 and row["since"] == reviews["acute appendicitis"]["date"]
    assert kp.main(["--run", str(update.parent), "--knowledge", str(store)]) == 0
    profile = json.loads((store / "profiles.json").read_text())["profiles"][0]
    assert {m["concept"]["code"] for m in profile["manifestations"]} >= {"HP:0100281", "HP:0001945"}


def test_saturation_is_counted_layer_by_layer(tmp_path, world, monkeypatch):
    """Five guidelines in a row that add nothing close the guidelines, not the review: the next layer is read."""
    abstracts = world_abstracts = dr._world["abstracts"]
    accuracy = [{"pmid": "500", "title": "Diagnostic accuracy of signs in acute appendicitis",
                 "abstractText": "Acute appendicitis is diagnosed clinically."}]

    def records(names, layer, get=None, since=""):
        yield from abstracts if layer == "guideline" else accuracy if layer == "diagnostic_accuracy" else []
    monkeypatch.setattr(dr, "records", records)
    out = tmp_path / "out" / "profiles.json"
    dr.main(_files(tmp_path) + ["--diagnoses", "acute appendicitis", "--saturation", "5", "--out", str(out)])
    row = json.loads(out.with_name("summary.json").read_text())["rows"][0]
    assert row["layers"] == {"guideline": 6, "diagnostic_accuracy": 1} and row["saturated_layers"] == ["guideline"]


def test_the_interpreters_interpret_and_what_they_read_differently_stays_a_doubt():
    a = dr.interpreted({"name": "fever", "category": "manifestation", "specific": True, "observed_by": "examination",
                        "nature": "pathological", "hpo_term": "Fever", "same_as": ""})
    b = dr.interpreted({**a, "hpo_term": "Pyrexia of unknown origin"})
    view, doubts = dr.agreed([a, b])
    assert view["category"] == "manifestation" and view["hpo_term"] is None and doubts == ["hpo_term"]
    assert dr.interpreted({"name": "typical symptoms", "specific": "false"})["specific"] is False
    assert dr.interpreted({"name": "x", "category": "diagnosis"})["category"] == ""      # not one of the categories


def test_the_judges_only_verify():
    prompt = dr.judge_prompt("pneumonia", {"manifestation": "cough", "category": "manifestation",
                                           "quote": "Cough was present."})
    assert "hpo" not in prompt.lower() and "category:" not in prompt and '{"states": "yes"}' in prompt


def test_mondo_is_found_by_another_synonym_when_no_exact_one_names_the_disease():
    lines = ["[Term]", "id: MONDO:0004974", "name: appendicitis",
             'synonym: "acute appendicitis" RELATED []', "[Term]", "id: MONDO:1", "name: x",
             'synonym: "shared" RELATED []', "[Term]", "id: MONDO:2", "name: y", 'synonym: "shared" RELATED []']
    found = dr.any_synonym(lines)
    assert found["acute appendicitis"] == "MONDO:0004974" and "shared" not in found
