"""How much each manifestation weighs (21 November 2026), without network: rules, pooling, the ranking's use."""

import importlib.util
import json
import math
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "disease_frequencies", Path(__file__).resolve().parents[1] / "scripts" / "disease_frequencies.py")
df = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(df)

from melampo.clinical import case_graph as cg  # noqa: E402
from melampo.reasoning import likelihood_ranking as lr  # noqa: E402

TEXT = [("Results", "Migration of pain to the right lower quadrant had a sensitivity of 64% and a specificity of 82% "
                    "(positive likelihood ratio 3.6). Fever was present in 41 of 100 patients (41%). "
                    "Vomiting is often present.")]


def item(**kw):
    base = {"manifestation": "migratory pain", "with_disease": {}, "sensitivity": "64%", "specificity": "82%",
            "lr_positive": "3.6", "setting": "emergency department", "patients": None, "design": "prospective_cohort",
            "comparator": "patients with suspected appendicitis",
            "quote": "Migration of pain to the right lower quadrant had a sensitivity of 64% and a specificity of 82% "
                     "(positive likelihood ratio 3.6)."}
    base.update(kw)
    return base


def test_the_rules_fixed_before_the_run():
    m, why = df.checked(item(), TEXT)
    assert why == "" and (m["sensitivity"], m["specificity"], m["lr_positive"]) == (0.64, 0.82, 3.6)
    assert df.checked(item(lr_positive="5.0"), TEXT)[1] == "number_not_in_quote"
    assert df.checked(item(quote="Pain migrates."), TEXT)[1] == "quote_not_in_text"
    fever = {"manifestation": "fever", "with_disease": {"count": 41, "total": 100, "percent": "41%"},
             "quote": "Fever was present in 41 of 100 patients (41%).", "setting": "hospital"}
    m, why = df.checked(fever, TEXT)
    assert why == "" and m["with_disease"] == {"count": 41, "total": 100}
    assert df.checked({**fever, "with_disease": {"count": 41, "total": 100, "percent": "100"}}, TEXT)[1] in (
        "count_and_percent_disagree", "number_not_in_quote")
    words = {"manifestation": "vomiting", "verbal": "often", "quote": "Vomiting is often present."}
    m, why = df.checked(words, TEXT)
    assert why == "" and m["verbal"] == {"word": "often", "class": "frequent", "frequency": 0.545}


def test_a_likelihood_ratio_that_disagrees_with_its_sensitivity_and_specificity_is_refused():
    text = [("R", "Sensitivity 64%, specificity 82%, positive likelihood ratio 9.0.")]
    bad = item(lr_positive="9.0", quote="Sensitivity 64%, specificity 82%, positive likelihood ratio 9.0.")
    assert df.checked(bad, text)[1] == "lr_disagrees_with_sensitivity_and_specificity"


def test_the_random_effects_pool_on_the_logit_scale():
    one = df.pool([(41, 100)])
    assert one["studies"] == 1 and abs(one["value"] - 0.41) < 1e-3
    two = df.pool([(40, 100), (60, 100)])
    assert 0.45 < two["value"] < 0.55 and two["ci"][0] < two["value"] < two["ci"][1] and two["patients"] == 200
    same = df.pool([(50, 100), (50, 100)])
    assert same["i2"] == 0.0


def test_settings_are_kept_apart_and_a_meta_analysis_is_taken_as_it_is():
    ed = {"manifestation": "m", "setting": "emergency department", "design": "meta_analysis", "pooled": True,
          "patients": 2000, "sensitivity": 0.7, "specificity": 0.8, "comparator": "", "source": {"id": "MA"}}
    pc = {**ed, "setting": "primary care", "patients": 300, "sensitivity": 0.5, "specificity": 0.9,
          "source": {"id": "PC"}}
    w = df.weigh([ed, pc])
    assert w["accuracy"]["setting"] == "emergency department" and w["accuracy"]["sensitivity"] == 0.7
    assert w["accuracy"]["other_settings"] == ["primary care"]


def test_the_ranking_uses_the_measured_ratio_with_its_background():
    m = cg.Manifestation("m1", "migratory pain", cg.Concept("HPO", "HP:0000001"), typicality="common",
                         accuracy={"sensitivity": 0.64, "specificity": 0.82, "setting": "emergency department",
                                   "design": "meta_analysis"})
    d = cg.Disease("d", "hypothesized_diseases", "appendicitis", "declared", manifestations=[m],
                   states=[cg.ManifestationState("m1", cg.PRESENT, "stated")])
    r = lr.score(d, {})
    assert abs(r.log10_odds - math.log10(0.64 / 0.18)) < 1e-6 and "LR+" in r.contributions[0].note
    d.states = [cg.ManifestationState("m1", cg.ABSENT, "stated")]
    assert abs(lr.score(d, {}).log10_odds - math.log10(0.36 / 0.82)) < 1e-6


def test_a_run_weighs_the_profile_and_resumes(tmp_path, monkeypatch):
    hpo = tmp_path / "hp.obo"
    hpo.write_text("format-version: 1.2\n\n[Term]\nid: HP:0001945\nname: Fever\n")
    profile = cg.Disease("MONDO:0004974", "hypothesized_diseases", "acute appendicitis", "profile", manifestations=[
        cg.Manifestation("m1", "Fever", cg.Concept("HPO", "HP:0001945"), citations=[{"id": "x"}])])
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"profiles": [profile.as_dict()]}))
    monkeypatch.setattr(df, "texts_for", lambda d, *a: [{"id": "PMC1", "read": "full_text", "kind": "descriptive",
                                                     "sections": TEXT, "source": {"title": "t", "pmcid": "PMC1"}}])
    fever = {"manifestation": "fever", "with_disease": {"count": 41, "total": 100, "percent": "41%"},
             "quote": "Fever was present in 41 of 100 patients (41%).", "setting": "hospital",
             "design": "retrospective_cohort"}
    calls = []

    def chat(self, model, prompt, *, reasoning, max_tokens):
        calls.append(model)
        if prompt.startswith("Disease:"):
            return '{"states": "yes", "category": "manifestation", "same_as": "Fever"}', "stop"
        return json.dumps([fever]), "stop"
    monkeypatch.setattr(df.rs.Router, "chat", chat)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    out = tmp_path / "out" / "frequencies.json"
    assert df.main(["--profiles", str(profiles), "--hpo", str(hpo), "--out", str(out)]) == 0
    weighted = json.loads((tmp_path / "out" / "profiles_weighted.json").read_text())["profiles"][0]
    m = weighted["manifestations"][0]
    assert m["frequency_basis"] == "one_study" and abs(m["frequency"] - 0.41) < 1e-3
    calls.clear()
    again = tmp_path / "again" / "frequencies.json"
    assert df.main(["--profiles", str(profiles), "--hpo", str(hpo), "--out", str(again),
                    "--resume", str(tmp_path / "out")]) == 0
    assert calls == []


def test_the_corrections_declared_after_the_first_run():
    text = [("R", "Common symptoms were fever (40 [98%] of 41 patients), headache (three [8%] of 38)."),
            ("Table: accuracy", "Emergency physician | 14 | 0.71 (0.67–0.74) | 0.92 (0.90–0.93) | 10.40 (4.80–20.3)")]
    headache = {"manifestation": "headache", "with_disease": {"count": 3, "total": 38, "percent": 8},
                "quote": "Common symptoms were fever (40 [98%] of 41 patients), headache (three [8%] of 38)."}
    assert df.checked(headache, text)[1] == ""                        # "three" is a number
    row = {"manifestation": "ultrasound", "sensitivity": "0.71 (0.67–0.74)", "specificity": "0.92 (0.90–0.93)",
           "lr_positive": "10.40 (4.80–20.3)", "pooled": True, "design": "meta_analysis",
           "quote": "| Emergency physician | 14 | 0.71 (0.67–0.74) | 0.92 (0.90–0.93) | 10.40 (4.80–20.3) |"}
    assert df.checked(row, text)[1] == ""                             # a table row with bars, pooled ratios
    assert df.checked({**row, "sensitivity": "71-94%"}, text)[1] == "a_range_not_a_value"


def test_tests_scores_and_outcomes_are_kept_apart_from_the_manifestations(tmp_path, monkeypatch):
    hpo = tmp_path / "hp.obo"
    hpo.write_text("format-version: 1.2\n")
    profile = cg.Disease("MONDO:1", "hypothesized_diseases", "acute appendicitis", "profile", manifestations=[
        cg.Manifestation("m1", "Right lower quadrant pain", cg.Concept("HPO", "HP:0100281"), citations=[{"id": "x"}])])
    (tmp_path / "profiles.json").write_text(json.dumps({"profiles": [profile.as_dict()]}))
    text = [("R", "Migration of pain to the right lower quadrant: sensitivity 64%, specificity 82%. "
                  "CT: sensitivity 99%, specificity 84%. Mortality was 1 of 100 (1%).")]
    monkeypatch.setattr(df, "texts_for", lambda d, *a: [{"id": "PMC1", "read": "full_text", "kind": "accuracy",
                                                     "sections": text, "source": {"title": "t", "pmcid": "PMC1"}}])
    items = [{"manifestation": "migratory right lower quadrant pain", "sensitivity": "64%", "specificity": "82%",
              "patients": 100, "setting": "emergency department", "design": "prospective_cohort",
              "quote": "Migration of pain to the right lower quadrant: sensitivity 64%, specificity 82%."},
             {"manifestation": "CT", "sensitivity": "99%", "specificity": "84%", "patients": 100,
              "setting": "emergency department", "design": "prospective_cohort",
              "quote": "CT: sensitivity 99%, specificity 84%."},
             {"manifestation": "mortality", "with_disease": {"count": 1, "total": 100, "percent": "1%"},
              "quote": "Mortality was 1 of 100 (1%)."}]
    verdict = {"migratory": ("manifestation", "Right lower quadrant pain"), "CT": ("test", "none"),
               "mortality": ("outcome", "none")}

    def chat(self, model, prompt, *, reasoning, max_tokens):
        if prompt.startswith("Disease:"):
            what = prompt.split("Measured: ", 1)[1].split("\n", 1)[0]
            cat, same = next(v for k, v in verdict.items() if what.startswith(k))
            return json.dumps({"states": "yes", "category": cat, "same_as": same}), "stop"
        return json.dumps(items), "stop"
    monkeypatch.setattr(df.rs.Router, "chat", chat)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    out = tmp_path / "out" / "frequencies.json"
    assert df.main(["--profiles", str(tmp_path / "profiles.json"), "--hpo", str(hpo), "--out", str(out)]) == 0
    weighted = json.loads((tmp_path / "out" / "profiles_weighted.json").read_text())["profiles"][0]
    assert [m["words"] for m in weighted["manifestations"]] == ["Right lower quadrant pain"]
    assert weighted["manifestations"][0]["accuracy"]["sensitivity"] == 0.64
    tests = json.loads((tmp_path / "out" / "tests.json").read_text())
    assert tests["acute appendicitis"][0]["name"] == "CT"
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert (summary["outcomes_left_out"], summary["matched_to_profile"], summary["new_manifestations"]) == (1, 1, 0)


def test_the_search_leaves_off_topic_titles_and_prefers_the_rational_clinical_examination(monkeypatch):
    found = [{"pmid": "1", "title": "Clinical features of patients with COVID-19 pneumonia", "abstractText": "fever 98%"},
             {"pmid": "2", "title": "Signs and symptoms of community-acquired pneumonia",
              "abstractText": "Fever, cough and crackles: sensitivity 80%."},
             {"pmid": "3", "title": "Does this patient have community-acquired pneumonia?", "abstractText": "LR"}]
    monkeypatch.setattr(df, "epmc_relevance", lambda q, size: found)
    monkeypatch.setattr(df.rs, "text_of", lambda r: {"id": r["pmid"], "read": "abstract", "source": {"title": r["title"]},
                                                     "sections": [("Abstract", r["abstractText"])]})
    got = df.texts_for("pneumonia", ["Fever", "Cough"])
    assert [t["id"] for t in got] == ["3", "2"]                      # COVID-19 left out, the RCE first
