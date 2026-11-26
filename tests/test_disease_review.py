"""The systematic review of a disease (22 November 2026), without network: layers, screening, saturation as a
parameter, the profile with its weights."""

import importlib.util
import json
import math
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

    def records(names, layer, get=None, since="", **_):
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
        if prompt.startswith("You and another reader"):   # the same finding in other words: both say so
            calls["same"] = calls.get("same", 0) + 1
            return '[{"item": 1, "same_as": "a"}]', "stop"
        if prompt.startswith("Findings agreed"):
            return "[]", "stop"
        calls["read"] += 1
        if first in prompt and model == dr.rs.READERS["muse"]:   # Muse names the pain in other words
            return json.dumps([
                {"words": "right lower quadrant pain", "name": "pain in the right lower quadrant",
                 "what": "abdominal pain", "category": "manifestation", "specific": True, "observed_by": "history",
                 "nature": "pathological", "site": "right lower quadrant",
                 "measures": {"with_disease": {"count": 80, "total": 100, "percent": "80%"}},
                 "quote": "In acute appendicitis, right lower quadrant pain was present in 80 of 100 patients (80%), "
                          "fever often."},
                {"words": "fever", "name": "fever", "category": "manifestation", "specific": True,
                 "observed_by": "examination", "nature": "pathological", "measures": {"verbal": "often"},
                 "quote": "In acute appendicitis, right lower quadrant pain was present in 80 of 100 patients (80%), "
                          "fever often."}]), "stop"
        if first in prompt:
            return json.dumps([
                {"words": "right lower quadrant pain", "name": "right lower quadrant pain",
                 "what": "abdominal pain", "category": "manifestation", "specific": True, "observed_by": "history",
                 "nature": "pathological", "site": "right lower quadrant",
                 "measures": {"with_disease": {"count": 80, "total": 100, "percent": "80%"}},
                 "quote": "In acute appendicitis, right lower quadrant pain was present in 80 of 100 patients (80%), "
                          "fever often."},
                {"words": "fever", "name": "fever", "category": "manifestation", "specific": True,
                 "observed_by": "examination", "nature": "pathological",
                 "measures": {"verbal": "often"},
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
    assert len(measures) == 4 and {m["reader"] for m in measures} == {"nemotron", "muse"}   # each interpreter's own
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

    def records(names, layer, get=None, since="", **_):
        yield from abstracts if layer == "guideline" else accuracy if layer == "diagnostic_accuracy" else []
    monkeypatch.setattr(dr, "records", records)
    out = tmp_path / "out" / "profiles.json"
    dr.main(_files(tmp_path) + ["--diagnoses", "acute appendicitis", "--saturation", "5", "--out", str(out)])
    row = json.loads(out.with_name("summary.json").read_text())["rows"][0]
    assert row["layers"] == {"guideline": 6, "diagnostic_accuracy": 1} and row["saturated_layers"] == ["guideline"]


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


def test_a_result_stated_twice_in_one_article_counts_once():
    src = {"pmcid": "PMC9"}
    same = [{"manifestation": n, "quote": q, "sensitivity": 0.94, "specificity": 0.87, "source": src}
            for n, q in (("PSV", "PSV: 94%, 87%."), ("peak systolic velocity", "Peak systolic velocity 94% / 87%."))]
    other = {"manifestation": "PSV", "quote": "x", "sensitivity": 0.94, "specificity": 0.87, "source": {"pmcid": "PMC8"}}
    kept = dr.dedupe(same + [other])
    assert len(kept) == 2 and kept[0]["also_stated"] == ["Peak systolic velocity 94% / 87%."]


def test_a_comparison_between_forms_of_the_disease_does_not_weigh_the_diagnosis():
    ms = [{"compares": "forms_of_the_disease", "sensitivity": 0.8},
          {"compares": "", "comparator": "patients with uncomplicated acute appendicitis", "sensitivity": 0.7},
          {"compares": "disease_vs_without", "comparator": "patients without appendicitis", "sensitivity": 0.6}]
    assert [m["sensitivity"] for m in dr.for_weights(ms, ["acute appendicitis", "appendicitis"])] == [0.6]


def test_a_direction_without_a_number_is_a_provisional_weight():
    from melampo.clinical import case_graph as cg
    from melampo.reasoning import likelihood_ranking as lr

    summary = dr.direction_summary([{"when": "present", "effect": "more_likely", "words": ""},
                                    {"when": "absent", "effect": "less_likely", "words": "strongly"},
                                    {"when": "absent", "effect": "less_likely", "words": ""}])
    assert summary == {"present": {"effect": "more_likely", "sources": 1, "strong": False},
                       "absent": {"effect": "less_likely", "sources": 2, "strong": True}}
    m = cg.Manifestation("m", "migratory pain", cg.Concept("HPO", "HP:1"), direction=summary,
                         citations=[{"id": "x"}])
    d = cg.Disease("d", "hypothesized_diseases", "appendicitis", "declared", manifestations=[m],
                   states=[cg.ManifestationState("m", cg.ABSENT, "stated")])
    r = lr.score(d, {})
    assert abs(r.log10_odds - math.log10(0.25)) < 1e-9 and "to be calibrated" in r.contributions[0].note
    assert dr.direction_summary([{"when": "present", "effect": "more_likely", "words": ""},
                                 {"when": "present", "effect": "less_likely", "words": ""}])["present"]["effect"] \
        == "doubt"


# -- corrections of 24 November 2026, after the second run --------------------------------------------------------

def test_a_measure_is_one_value_the_text_states_not_a_bound_or_a_range():
    df = dr.df
    assert df.point_problem("98%", "a non-compressible appendix (up to 98 % sensitive)") \
        == "a_bound_or_range_in_the_text"
    assert df.point_problem("30", "fever ranges from 30 to 80% [2, 5, 8]") == "a_bound_or_range_in_the_text"
    assert df.point_problem("80%", "fever ranges from 30 to 80% [2, 5, 8]") == "a_bound_or_range_in_the_text"
    assert df.point_problem("79% accurate diagnosis", "79% accurate") == "not_one_value"
    assert df.point_problem("0.85", "sensitivity 0.85 (95% CI 0.80 to 0.89)") == ""
    assert df.point_problem("87% [71-95]", "specificity of 87 % [71-95]") == ""
    assert df.point_problem("75.3%", "sensitivity and specificity were 77.6% and 75.3% for US") == ""
    assert df.point_problem(9083, "data from 9083 participants") == ""


def test_the_measures_are_read_inside_measures_and_a_refused_one_leaves_the_manifestation():
    text = [("Abstract", "The most sensitive sign is a non-compressible appendix (up to 98 % sensitive). "
                         "Fever was present in 30 of 100 patients.")]
    nested = {"name": "fever", "quote": "Fever was present in 30 of 100 patients.",
              "measures": {"with_disease": {"count": 30, "total": 100, "percent": None}, "design": "case_series"}}
    m, why = dr.item_checked(nested, text)
    assert why == "" and m["with_disease"] == {"count": 30.0, "total": 100.0}
    bound = {"name": "non-compressible appendix", "quote": "The most sensitive sign is a non-compressible appendix "
             "(up to 98 % sensitive).", "measures": {"sensitivity": "98%"}}
    m, why = dr.item_checked(bound, text)
    assert why == "a_bound_or_range_in_the_text" and "sensitivity" not in m \
        and m["measure_refused"] == why and m["manifestation"] == "non-compressible appendix"


def test_a_direction_is_verified_as_a_claim_of_its_own():
    item = {"manifestation": "vomiting before pain", "quote": "The presence of vomiting before pain makes "
            "appendicitis unlikely.", "direction": {"when": "present", "effect": "less_likely", "words": ""}}
    assert dr.claim_of("acute appendicitis", item) == \
        "vomiting before pain, when present, makes acute appendicitis less likely"
    assert dr.claim_of("acute appendicitis", {"manifestation": "fever"}) == "in acute appendicitis, fever"


def test_a_page_that_does_not_answer_is_tried_again_and_reported():
    answers = iter(["", "not json", json.dumps({"hitCount": 3, "resultList": {"result": [{"pmid": "1"}]},
                                                "nextCursorMark": "*"})])
    report: dict = {}
    got = list(dr.records(["x"], "guideline", lambda url, accept="": next(answers), report=report, wait=lambda s: 0))
    assert [r["pmid"] for r in got] == ["1"] and report == {"hits": 3, "pages": 1, "unreachable": False}
    report = {}
    assert list(dr.records(["x"], "guideline", lambda url, accept="": "", report=report, wait=lambda s: 0)) == []
    assert report["unreachable"] is True and report["hits"] is None


def test_a_measure_weighs_only_when_both_interpreters_read_it_and_a_source_gives_one_result():
    src = {"pmid": "1"}
    psv = [{"reader": r, "sensitivity": 0.94, "specificity": 0.87, "source": src, "quote": "q"}
           for r in ("nemotron", "muse")]
    alone = {"reader": "muse", "sensitivity": 0.81, "specificity": 0.88, "source": {"pmid": "2"}, "quote": "q2"}
    assert [m["sensitivity"] for m in dr.agreed_measures(psv + [alone])] == [0.94]
    conflict = psv + [{**m, "sensitivity": 0.82, "specificity": 0.97} for m in psv]
    assert dr.agreed_measures(conflict) == []           # two results of one source for one manifestation: a doubt
    partial = psv + [{"reader": r, "sensitivity": 0.94, "source": src, "quote": "abstract"} for r in ("nemotron", "muse")]
    assert [m.get("specificity") for m in dr.agreed_measures(partial)] == [0.87]   # one result, stated twice
    directions, alone_ = dr.agreed_directions(
        [{"reader": r, "source": src, "direction": {"when": "present", "effect": "more_likely", "words": ""}}
         for r in ("nemotron", "muse")] +
        [{"reader": "muse", "source": src, "direction": {"when": "absent", "effect": "less_likely", "words": ""}}])
    assert directions == [{"when": "present", "effect": "more_likely", "words": ""}] and alone_ == 1


def test_a_server_that_does_not_answer_is_asked_again_and_a_page_given_up_is_counted(monkeypatch):
    import io
    import urllib.error
    answers = [TimeoutError(), urllib.error.HTTPError("u", 503, "busy", {}, None), io.BytesIO(b'{"ok": 1}')]

    class Page:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self.body

        def __exit__(self, *a):
            return False

    def urlopen(req, timeout=60):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return Page(a)

    monkeypatch.setattr(dr.rs.urllib.request, "urlopen", urlopen)
    dr.rs.FAILED.clear()
    assert dr.rs.get("https://www.ebi.ac.uk/x", wait=lambda s: 0) == '{"ok": 1}' and dr.rs.FAILED == {}
    answers[:] = [urllib.error.HTTPError("u", 404, "not found", {}, None)]
    assert dr.rs.get("https://www.ebi.ac.uk/y", wait=lambda s: 0) == ""
    assert dr.rs.FAILED == {"www.ebi.ac.uk http_404": 1} and answers == []   # a page not there is asked once


def test_a_publication_is_read_with_the_one_procedure_other_words_paired_in_the_sentence(tmp_path, world):
    """Muse names the pain in other words: both interpreters, asked in the sentence, say it is the same finding,
    and the measure both read weighs (25 November 2026)."""
    out = tmp_path / "out" / "profiles.json"
    assert dr.main(_files(tmp_path) + ["--diagnoses", "acute appendicitis", "--out", str(out)]) == 0
    row = json.loads(out.with_name("summary.json").read_text())["rows"][0]
    assert world["same"] == 2 and row["measures_agreed"] == 2 and row["measures_one_interpreter"] == 0
    measures = [json.loads(x) for x in out.with_name("measures.jsonl").read_text().splitlines()]
    pain = [m for m in measures if m["key"] == "HP:0100281"]
    assert {m["reader"] for m in pain} == {"nemotron", "muse"} and pain[0]["paired_by"] == "same_in_context"
    thesaurus = dr.Thesaurus.load(out.with_name("thesaurus.jsonl"))
    labels = {x["label"] for x in thesaurus.concepts["HP:0100281"].labels}
    assert labels == {"right lower quadrant pain", "pain in the right lower quadrant"}


def test_an_obsolete_mondo_term_is_followed_to_the_term_that_replaces_it():
    lines = ['[Term]', 'id: MONDO:0004974', 'name: obsolete acute appendicitis', 'synonym: "appendicitis acute" EXACT []',
             'is_obsolete: true', 'replaced_by: MONDO:0005649', '', '[Term]', 'id: MONDO:0005649', 'name: appendicitis',
             'synonym: "acute appendicitis" RELATED []']
    got = dr.replaced(lines)
    assert got[dr.dp.norm("acute appendicitis")] == ("MONDO:0004974", "MONDO:0005649")
    assert dr.dp.norm("appendicitis") not in got
