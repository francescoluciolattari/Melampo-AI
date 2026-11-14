"""The vetting bench, version 3 (14 November 2026): mechanisms judged by the sources and by two judges that must
agree; the link stated with its certainty."""

import json

from melampo.evaluation import mechanism_bench as mb

OBO = """format-version: 1.2
data-version: test

[Term]
id: GO:1
name: renal water absorption
synonym: "renal water retention" EXACT []

[Term]
id: GO:2
name: collecting duct water absorption
is_a: GO:1 ! renal water absorption

[Term]
id: GO:3
name: gluconeogenesis

[Term]
id: HP:9
name: Edema
synonym: "water retention" EXACT []
"""


def _sources():
    return mb.Sources.from_terms({"go": mb.parse_obo(OBO.splitlines())})


SIADH = next(c for c in mb.MECHANISM_CASES if c.case_id == "siadh_hyponatraemia")
NO_LINK = next(c for c in mb.MECHANISM_CASES if c.case_id == "no_link_myeloma_thyroid")


def _answer(link, mechanism, chain=()):
    return json.dumps({"link": link, "mechanism": mechanism, "chain": list(chain)})


def _judge(**verdict):
    base = {"term": "none", "matches": "none", "relation": "different", "established": "uncertain", "why": ""}
    base.update(verdict)
    return lambda prompt: json.dumps(base)


def test_the_accepted_mechanisms_resolve_in_the_sources_and_a_named_one_is_right_without_a_judge():
    sources = _sources()
    resolved = mb.resolve_case(SIADH, sources)
    water = next(r for r in resolved if r.accepted.label == "water retention")
    assert "GO:1" in water.ids                                  # "renal water absorption", a name it carries
    assert "HP:9" not in water.ids          # a broader term holding the label only as a synonym is not the concept
    asked = []
    judges = {"j1": lambda p: asked.append(p) or "", "j2": lambda p: asked.append(p) or ""}
    got = mb.score_answer(SIADH, mb.read_candidate(_answer("known", "renal water retention diluting sodium")),
                          resolved, sources, judges)
    assert got["outcome"] == mb.RIGHT and got["by"] == "names" and not asked
    # the same mechanism, stated only as possible: right but hedged
    got = mb.score_answer(SIADH, mb.read_candidate(_answer("possible", "free water retention")), resolved, sources,
                          judges)
    assert got["outcome"] == mb.RIGHT_HEDGED


def test_the_judges_read_the_meaning_and_the_ontology_decides_when_they_map_to_a_term_below():
    sources = _sources()
    resolved = mb.resolve_case(SIADH, sources)
    answer = mb.read_candidate(_answer("known", "aquaporin-2 insertion in the collecting duct"))
    # both judges map the claim to a term the ontology places under an accepted one
    got = mb.score_answer(SIADH, answer, resolved, sources, {"a": _judge(term="GO:2"), "b": _judge(term="GO:2")})
    assert got["outcome"] == mb.RIGHT
    # both read it as a step of the accepted chain
    both = {"a": _judge(matches="A", relation="step_of_chain"), "b": _judge(matches="A", relation="same")}
    assert mb.score_answer(SIADH, answer, resolved, sources, both)["outcome"] == mb.RIGHT
    # both read another mechanism: a confident error
    wrong = {"a": _judge(matches="none"), "b": _judge(matches="A", relation="different")}
    assert mb.score_answer(SIADH, answer, resolved, sources, wrong)["outcome"] == mb.WRONG_CONFIDENT
    # they disagree: a doubt, never counted right or wrong
    split = {"a": _judge(matches="A", relation="same"), "b": _judge(matches="none")}
    assert mb.score_answer(SIADH, answer, resolved, sources, split)["outcome"] == mb.DOUBT
    # a judge that does not answer: a doubt
    assert mb.score_answer(SIADH, answer, resolved, sources, {"a": _judge(), "b": lambda p: ""})["outcome"] == mb.DOUBT
    # "none" on an established link: a miss
    assert mb.score_answer(SIADH, mb.read_candidate(_answer("none", "")), resolved, sources, {})["outcome"] == mb.MISS


def test_where_no_link_is_established_a_prudent_answer_is_right_and_a_known_one_is_judged():
    resolved = mb.resolve_case(NO_LINK, None)
    assert mb.score_answer(NO_LINK, mb.read_candidate(_answer("none", "")), resolved, None, {})["outcome"] == mb.RIGHT
    got = mb.score_answer(NO_LINK, mb.read_candidate(_answer("possible", "paraprotein assay interference")),
                          resolved, None, {})
    assert got["outcome"] == mb.RIGHT
    known = mb.read_candidate(_answer("known", "amyloid goitre"))
    invented = {"a": _judge(established="no"), "b": _judge(established="no")}
    assert mb.score_answer(NO_LINK, known, resolved, None, invented)["outcome"] == mb.INVENTED
    documented = {"a": _judge(established="yes"), "b": _judge(established="yes")}
    assert mb.score_answer(NO_LINK, known, resolved, None, documented)["outcome"] == mb.CASE_HAS_A_LINK
    unsure = {"a": _judge(established="yes"), "b": _judge(established="uncertain")}
    assert mb.score_answer(NO_LINK, known, resolved, None, unsure)["outcome"] == mb.DOUBT


def test_the_answer_must_state_its_certainty_and_the_bench_counts_the_errors_apart():
    assert mb.read_candidate("bears on: yes") is None
    assert mb.read_candidate('```json\n{"link": "Known", "mechanism": "x", "chain": "a"}\n```')["link"] == "known"
    model = lambda prompt: _answer("known", "water retention") if "SIADH" in prompt else _answer("none", "")  # noqa
    result = mb.bench_mechanisms("m", model, [SIADH, NO_LINK], _sources(), {}, trials_per_case=2)
    assert result.runs == 4 and result.count(mb.RIGHT) == 4 and result.errors == 0
    payload = result.as_dict()
    assert payload["right_rate"] == 1.0 and payload["per_case"][0]["by"] == "names"


def test_every_case_is_written_with_its_accepted_mechanisms_and_the_judge_sees_the_sources():
    cases = mb.MECHANISM_CASES
    assert len(cases) == 16 and sum(not c.has_link for c in cases) == 3
    assert all(len(c.accepted) >= 2 for c in cases if c.has_link)
    sources = _sources()
    resolved = mb.resolve_case(SIADH, sources)
    prompt = mb.judge_prompt(SIADH, {"link": "known", "mechanism": "aquaporin", "chain": []}, resolved,
                             sources.nearest("collecting duct water"))
    assert "GO:1" in prompt and "renal water absorption" in prompt and "GO:2" in prompt
    report = mb.resolution_report([SIADH], sources)
    assert any("GO:1" in line for line in report["siadh_hyponatraemia"])


def test_a_judge_that_misreads_a_known_answer_judges_nothing():
    def careful(prompt):
        wrong = any(w in prompt for w in ("aldosterone causing renal sodium loss", "viral infection",
                                          "spinal cord ischaemia"))
        return json.dumps({"term": "none", "matches": "none" if wrong else "A",
                           "relation": "different" if wrong else "same", "established": "yes"})

    assert mb.calibrate(careful, None) == []
    yes_to_all = _judge(matches="A", relation="same")
    assert len(mb.calibrate(yes_to_all, None)) == 3          # the three plausible but wrong mechanisms
    assert len(mb.calibrate(lambda p: "", None)) == len(mb.CALIBRATION)
    # an unreadable answer is asked again once, for its form only
    asked = []

    def late(prompt):
        asked.append(prompt)
        return careful(prompt) if mb.ONLY_THE_JSON in prompt else "Let me think about it."

    assert mb.calibrate(late, None) == [] and len(asked) == 2 * len(mb.CALIBRATION)
    failures = mb.calibrate(lambda p: "I believe it is right.", None)
    assert "I believe it is right." in failures[0]
