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
        calls["read"] += 1
        if first in prompt:
            return json.dumps([
                {"words": "right lower quadrant pain", "name": "right lower quadrant pain",
                 "category": "manifestation", "specific": True, "observed_by": "history", "nature": "pathological",
                 "hpo_term": "Right lower quadrant pain", "same_as": "",
                 "measures": {"with_disease": {"count": 80, "total": 100, "percent": "80%"}},
                 "quote": "In acute appendicitis, right lower quadrant pain was present in 80 of 100 patients (80%), "
                          "fever often."},
                {"words": "fever", "name": "fever", "category": "manifestation", "specific": True,
                 "observed_by": "examination", "nature": "pathological", "hpo_term": "Fever", "same_as": "",
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


def test_one_format_for_every_model_measures_only_inside_measures():
    text = [("Abstract", "Fever was present in 30 of 100 patients.")]
    outside = {"name": "fever", "quote": "Fever was present in 30 of 100 patients.", "category": "manifestation",
               "observed_by": "history, examination", "with_disease": {"count": 30, "total": 100}}
    m, why = dr.item_checked(outside, text)
    assert "with_disease" not in m                         # another form is not read
    problems = dr.format_problems(outside)
    assert "measures_outside" in problems and "value_not_in_list:observed_by" in problems
    assert dr.format_problems(dr.FORMAT_EXAMPLE) == []
    assert json.dumps(dr.FORMAT_EXAMPLE, ensure_ascii=False) in dr.read_prompt("x", [("A", "t")])


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


def test_a_link_read_by_one_interpreter_alone_is_not_made():
    one = dr.interpreted({"name": "vomiting before pain", "category": "manifestation", "specific": True,
                          "hpo_term": "Vomiting", "same_as": "vomiting"})
    view, doubts = dr.agreed([one])
    assert view["hpo_term"] is None and view["same_as"] is None and view["category"] == "manifestation"
    view, _ = dr.agreed([one, dict(one)])
    assert view["hpo_term"] == "Vomiting"                  # both read it: the link is made


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


def test_the_names_of_one_manifestation_are_joined_only_where_both_interpreters_join_them():
    entries = [{"key": "positive psoas sign", "name": "positive psoas sign", "records": 1},
               {"key": "psoas sign", "name": "psoas sign", "records": 3},
               {"key": "HP:0001945", "name": "fever", "records": 4},
               {"key": "absence of fever", "name": "absence of fever", "records": 1},
               {"key": "appendix wider than 6 mm", "name": "appendix wider than 6 mm", "records": 1},
               {"key": "appendix wider than 7 mm", "name": "appendix wider than 7 mm", "records": 1}]
    one = dr.read_groups('[{"members": [0, 1]}, {"members": [2, 3], "absent": [3]}, {"members": [4, 5]}]', 6)
    two = dr.read_groups('[{"members": [0, 1]}, {"members": [2, 3], "absent": [3]}, {"members": [4]}, '
                         '{"members": [5]}]', 6)
    groups = dr.meet([one, two], 6)
    row: dict = {}
    remap, absent = dr.joined_keys(entries, groups, set(), row)
    assert remap == {"positive psoas sign": "psoas sign", "absence of fever": "HP:0001945"}
    assert absent == {"absence of fever"}                  # 6 mm and 7 mm: one interpreter alone joined them
    two_codes = [{"key": "HP:1", "name": "a", "records": 1}, {"key": "HP:2", "name": "b", "records": 1}]
    both = dr.read_groups('[{"members": [0, 1]}]', 2)
    assert dr.joined_keys(two_codes, dr.meet([both, both], 2), set(), row) == ({}, set())
    assert row["groups_not_joined"] == 1


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
