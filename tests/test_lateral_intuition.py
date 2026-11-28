"""The lateral part of the intuition engine (28 November 2026): salience, waves, generation, evaluation, guard."""

import json
import math

from melampo.clinical import case_graph as cg
from melampo.memory.profile_library import ProfileLibrary
from melampo.reasoning import likelihood_ranking as lr
from melampo.reasoning.case_diagnosis import diagnose
from melampo.reasoning.lateral_intuition import lateral_cycle, observations, waves

SOURCE = [{"source": "review", "id": "PMC1"}]


def m(mid, words, code, typicality="common", **kw):
    return cg.Manifestation(mid, words, cg.Concept("HPO", code, words), typicality=typicality, citations=SOURCE,
                            evidence="established", **kw)


def profile(did, words, *ms_):
    return cg.Disease(did, "hypothesized_diseases", words, "profile", concept=cg.Concept("MONDO", did, words),
                      manifestations=list(ms_))


# two diseases, each with what the other has not, and one that shares a little of both
HEART = profile("MONDO:1", "heart failure", m("a", "dyspnea", "HP:A", "hallmark"), m("b", "leg edema", "HP:B"),
                m("c", "orthopnea", "HP:C"))
THYROID = profile("MONDO:2", "hyperthyroidism", m("x", "tremor", "HP:X", "hallmark"), m("y", "weight loss", "HP:Y"),
                  m("z", "heat intolerance", "HP:Z"))
DECOY = profile("MONDO:3", "anxiety", m("x", "tremor", "HP:X", "rare"), m("q", "panic", "HP:Q", "hallmark"))
FEVER = profile("MONDO:4", "infection", m("f", "fever", "HP:F", "hallmark"), m("a", "dyspnea", "HP:A", "rare"))


def lib(tmp_path, *diseases):
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps({"profiles": [d.as_dict() for d in diseases]}))
    return ProfileLibrary.load(path)


def case(*facts, negated=(), uncertain=()):
    graph = cg.CaseGraph("c")
    graph.documents.append(cg.Document("note", "note"))
    for fid, words, code in facts:
        graph.findings.append(cg.Finding(fid, "note", words, cg.Provenance("note", words),
                                         polarity=cg.NEGATED if fid in negated else cg.AFFIRMED,
                                         certainty="uncertain" if fid in uncertain else "",
                                         concept=cg.Concept("HPO", code, words)))
    return graph


DUAL = (("f1", "dispnea", "HP:A"), ("f2", "edemi declivi", "HP:B"), ("f3", "tremore", "HP:X"),
        ("f4", "calo ponderale", "HP:Y"))


def test_a_dual_diagnosis_comes_as_a_pair_that_earns_its_prior(tmp_path):
    library = lib(tmp_path, HEART, THYROID, DECOY, FEVER)
    graph = case(*DUAL)
    out = diagnose(graph, library)
    lateral = out["lateral"]
    assert lateral["unclear"] is True and lateral["waves"] == 64
    pairs = [a for a in lateral["accepted"] if a["kind"] == "pair"]
    assert [p["diseases"] for p in pairs] == [["MONDO:1", "MONDO:2"]]
    pair = pairs[0]
    assert pair["insight"] and pair["explains_salient"] == ["f1", "f2", "f3", "f4"]
    # the pair is better than each disease alone: the second diagnosis paid its prior and still won
    singles = {a["hypothesis"]: a["energy"] for a in lateral["accepted"] if a["kind"] == "single"}
    assert all(pair["energy"] < e for e in singles.values())
    # both enter the case as candidates, each a hypothesis to exclude with what proposed it
    cands = {c["words"]: c["proposed_by"][0] for c in out["candidates"]}
    assert set(cands) == {"heart failure", "hyperthyroidism"}
    assert cands["heart failure"]["together_with"] == ["MONDO:2"]
    assert cands["heart failure"]["status"] == "hypothesis_to_exclude"
    # the ranking weighs what each one leaves: LIRICAL's no-match ratio for each observation it does not explain
    first = out["ranking"][0]
    assert sorted(first["unexplained"]) in (["f3", "f4"], ["f1", "f2"])
    # the decoy explains the tremor less well than hyperthyroidism and nothing else: it does not enter
    assert "anxiety" not in cands


def test_one_disease_that_explains_everything_needs_no_pair(tmp_path):
    library = lib(tmp_path, HEART, THYROID, DECOY)
    out = diagnose(case(("f1", "dispnea", "HP:A"), ("f2", "edemi", "HP:B"), ("f3", "ortopnea", "HP:C")), library)
    lateral = out["lateral"]
    assert lateral["unclear"] is False and lateral["waves"] == 0 and lateral["evaluated"]["pairs"] == 0
    assert [a["diseases"] for a in lateral["accepted"]] == [["MONDO:1"]]


def test_a_drug_that_explains_an_observation_takes_it_out_of_the_salient_ones(tmp_path):
    library = lib(tmp_path, HEART, THYROID)
    graph = case(*DUAL)
    graph.medications.append(cg.Medication("drug1", "levotiroxina", cg.Provenance("note", "levotiroxina"),
                                           name="levothyroxine"))
    graph.drug_effects.append(cg.DrugEffect("drug1", "f3", "adverse_reactions", "tremors have been reported",
                                            timing="after_start"))
    graph.drug_effects.append(cg.DrugEffect("drug1", "f4", "adverse_reactions", "weight loss", timing="unknown"))
    lateral = lateral_cycle(graph, library)
    by_drug = [e for e in lateral["explained_beforehand"] if e["by"] == "drug"]
    assert [(e["observation"], e["medication"]) for e in by_drug] == [("f3", "levothyroxine")]
    assert "f3" not in lateral["observed"] and "f4" in lateral["observed"]     # timing unknown: a doubt, not a cause
    assert "f3" not in {s["observation"] for s in lateral["salient"]}


def test_what_a_drug_can_hide_is_not_weighed_when_absent(tmp_path):
    library = lib(tmp_path, FEVER, HEART)
    graph = case(("f1", "dispnea", "HP:A"), ("f2", "apiretico", "HP:F"), negated={"f2"})
    plain = lateral_cycle(graph, library)
    graph.drug_profiles.append(cg.DrugProfile("drug1", "paracetamolo", inn="paracetamol", masks=[
        {"concept": "HP:F", "words": "fever", "section": "warnings", "sentence": "may mask fever"}]))
    hidden = lateral_cycle(graph, library)
    assert hidden["masked"] == [{"manifestation": "HP:F",
                                 "why": "paracetamol can hide it (warnings: may mask fever)"}]
    bg = lr.background(library.profiles.values())
    from melampo.reasoning.lateral_intuition import _States
    view = _States(graph, None).view(library.profiles["MONDO:4"])
    with_mask = lr.score(view, bg, masked=hidden["masked_keys"])
    without = lr.score(view, bg)
    assert with_mask.masked == ["f"] and with_mask.log10_odds > without.log10_odds
    assert plain["masked"] == []


def test_a_present_disease_explains_its_own_findings(tmp_path):
    library = lib(tmp_path, HEART, THYROID)
    graph = case(*DUAL)
    graph.add_disease(cg.Disease("dis:1", cg.PRESENT_DISEASES, "ipertiroidismo noto", "case",
                                 manifestations=list(THYROID.manifestations)))
    lateral = lateral_cycle(graph, library)
    known = {e["observation"] for e in lateral["explained_beforehand"] if e["by"] == "present disease"}
    assert known == {"f3", "f4"}
    assert set(lateral["observed"]) == {"f1", "f2"}
    assert [a["diseases"] for a in lateral["accepted"]] == [["MONDO:1"]] and lateral["unclear"] is False


def test_a_doubtful_observation_is_salient_and_never_a_penalty(tmp_path):
    library = lib(tmp_path, HEART, THYROID)
    graph = case(("f1", "dispnea", "HP:A"), ("f3", "forse tremore", "HP:X"), uncertain={"f3"})
    assert [o.doubtful for o in observations(graph)] == [False, True]
    lateral = lateral_cycle(graph, library)
    assert {"observation": "f3", "words": "forse tremore", "reasons": ["unexplained", "doubtful"]} not in \
        lateral["salient"]
    assert any(s["observation"] == "f3" and s["reasons"] == ["doubtful"] for s in lateral["salient"])
    assert "f3" not in lateral["observed"]


def test_the_guard_lets_nothing_through_that_is_worse_than_no_disease(tmp_path):
    # a profile whose hallmarks are all stated absent explains one finding and loses against no disease at all
    # (three would not be enough: explaining an observation, against LIRICAL's no-match 1/100, is worth a lot)
    weak = profile("MONDO:9", "weak", m("a", "dyspnea", "HP:A", "rare"),
                   *(m(f"h{i}", f"h{i}", f"HP:H{i}", "hallmark") for i in range(1, 5)))
    library = lib(tmp_path, weak, THYROID)
    graph = case(("f1", "dispnea", "HP:A"), *((f"n{i}", f"no h{i}", f"HP:H{i}") for i in range(1, 5)),
                 negated={f"n{i}" for i in range(1, 5)})
    lateral = lateral_cycle(graph, library)
    assert lateral["accepted"] == []
    assert lateral["salient"] == [{"observation": "f1", "words": "dispnea", "reasons": ["unexplained"]}]


def test_the_waves_are_deterministic_periodic_and_bind_every_pair_of_observations():
    ids = [f"f{i}" for i in range(6)]
    one, two = waves(ids, 64), waves(list(reversed(ids)), 64)
    assert one == two                                       # the same case gives the same waves, in any order
    assert len(set(one)) > 20                               # not the same pattern over and over
    for i in ids:                                           # each observation is active about half of the time
        share = sum(i in w for w in one) / len(one)
        assert 0.3 < share < 0.7
    for i in ids:                                           # every two observations are bound together sometimes
        for j in ids:
            assert any(i in w and j in w for w in one)


def test_a_shared_manifestation_is_expected_with_the_noisy_or_of_its_frequencies(tmp_path):
    a = profile("MONDO:7", "a", m("s", "shared", "HP:S", frequency=0.5))
    b = profile("MONDO:8", "b", m("s", "shared", "HP:S", frequency=0.5))
    graph = case(("f1", "shared", "HP:S"))
    from melampo.reasoning.lateral_intuition import _States
    states = _States(graph, None)
    bg = {"HP:S": 0.25}
    alone = lr.score(states.view(a), bg)
    together = lr.score_set([states.view(a), states.view(b)], bg)
    assert math.isclose(alone.log10_odds, math.log10(0.5 / 0.25))
    expected = math.log10(0.75 / 0.25) + math.log10(lr.SECOND_DIAGNOSIS)
    assert math.isclose(together.log10_odds, expected)
    assert together.explains == ["f1"] and together.words == "a + b"


def test_without_observations_to_account_for_nothing_is_penalised(tmp_path):
    graph = case(("f1", "dispnea", "HP:A"))
    from melampo.reasoning.lateral_intuition import _States
    view = _States(graph, None).view(HEART)
    bg = lr.background([HEART])
    assert lr.score(view, bg).unexplained == []
    scored = lr.score(view, bg, observed={"f1": "dispnea", "f9": "something else"})
    assert scored.unexplained == ["f9"]
    assert any(c.log10_lr == math.log10(lr.NO_MATCH) for c in scored.contributions)
