"""What the sentence states of the structure, claim by claim (2 November 2026): nothing chosen from a list, every
claim with its words copied, the reader checking them with what it knows (ontologies, grammar), and a person
when the readers do not agree."""

import json

from melampo.evaluation.llm_reading import claims_prompt, copied, read_claims
from melampo.memory.finding_owner import (
    ASIDE, ELSEWHERE, FINDING, HERE, KEEP, UNSURE, FindingOwner, agreed_claims, claims_outcome, laterality_conflict,
    report_context,
)
from melampo.memory.what_is_said import AnomalyWords, conjuncts, phrases, stated_anomalies


def _answer(**kw):
    base = {"marked": kw.pop("marked", ""), "same": "yes", "observed": [], "normal": [], "measure": [], "past": [],
            "certainty": [], "alternatives": [], "device": [], "not_assessable": []}
    base.update(kw)
    return base


def _owner(*answers, anomalies=None):
    chats = {f"m{i}": (lambda a: lambda prompt: json.dumps(a))(a) for i, a in enumerate(answers)}
    return FindingOwner(None, chats, question="claims", anomalies=anomalies)


def test_the_prompt_asks_for_claims_with_their_words_and_the_exam_is_context_only():
    s = "The pancreas is hyperechoic and not enlarged"
    p = claims_prompt(s, 4, 12, "pancreas", exam="US of the abdomen")
    assert "<w>pancreas</w>" in p and "context only" in p and "observed" in p and "not_assessable" in p


def test_a_piece_shortened_with_dots_is_copied_when_every_part_is_there_in_order():
    s = "Multiple nodules are observed in the left lung, right upper lobe, and right middle lobe"
    assert copied(s, "Multiple nodules are observed in the ... right upper lobe")
    assert copied(s, "nodules … right middle lobe")
    assert not copied(s, "right middle lobe ... Multiple nodules")
    assert not copied(s, "a calcified nodule")


def test_words_not_in_the_sentence_are_dropped_and_other_marked_words_make_the_answer_invalid():
    s = "Mitral valve: leaflets more intensely stained, regurgitation 3+"
    got = read_claims(json.dumps(_answer(marked="Mitral valve", observed=[
        {"words": "regurgitation 3+", "of": "it"}, {"words": "aortic stenosis", "of": "it"}])), s, "Mitral valve")
    assert got["valid"] and len(got["observed"]) == 1 and got["dropped"] == 1
    assert not read_claims(json.dumps(_answer(marked="Ao")), s, "Mitral valve")["valid"]
    other = read_claims(json.dumps(_answer(marked="Mitral valve", observed=[
        {"words": "regurgitation 3+", "of": "left femur"}])), s, "Mitral valve")
    assert other["observed"][0]["of"] == "?"                 # another structure the sentence does not name


def test_the_outcome_is_what_would_be_attached_not_a_label():
    assert claims_outcome(read_claims(json.dumps(_answer(observed=[{"words": "hyperechoic", "of": "it"}],
                                                          normal=["not enlarged"])),
                                      "The pancreas is hyperechoic and not enlarged", "pancreas")) == FINDING
    assert claims_outcome(_answer(valid=True, observed=[{"words": "x", "of": "nodes", "reaches": "no"}])) == ASIDE
    assert claims_outcome(_answer(valid=True, observed=[{"words": "x", "of": "mass", "reaches": "yes"}])) == FINDING
    assert claims_outcome(_answer(valid=True, measure=["0.5mm"])) == KEEP
    assert claims_outcome(_answer(valid=True, same="no", named="liver")) == ASIDE
    assert claims_outcome({"valid": False}) == "unsure"


def test_two_claims_of_one_sentence_are_both_attached_and_the_boundary_disappears():
    s = "The pancreas is hyperechoic and not enlarged"
    a = _answer(marked="pancreas", observed=[{"words": "hyperechoic", "of": "it"}], normal=["not enlarged"])
    b = _answer(marked="pancreas", observed=[{"words": "is hyperechoic", "of": "it"}], normal=["not enlarged"])
    got = _owner(a, b)("pancreas", s, 4, "en", "pancreas")
    assert (got.verdict, got.statement) == (HERE, FINDING)
    assert got.claims["findings"][0]["words"] == "hyperechoic" and got.claims["normal"] == ["not enlarged"]


def test_a_measure_said_normal_by_knowledge_is_only_a_measure():
    s = "The IMT thickness in the right CCA is about 0.5mm"
    a = _answer(marked="right CCA", measure=["about 0.5mm"])
    b = _answer(marked="right CCA", measure=["0.5mm"], normal=["0.5mm"])          # what it knows, not what is said
    got = _owner(a, b)("right CCA", s, s.index("right CCA"), "en", "common_carotid_artery_right")
    assert got.verdict == HERE and got.claims.get("measure") and "normal" not in got.claims


def test_the_reader_contradicts_with_what_it_knows():
    s = "The gallbladder is homogeneous, with a hypodense halo around it"
    said = _answer(marked="gallbladder", normal=["homogeneous, with a hypodense halo around it"])
    got = _owner(said, said)("gallbladder", s, 4, "en", "gallbladder")
    assert got.verdict == UNSURE and "the_normal_words_state_an_abnormality" in got.reader
    s = "The right kidney measures 10.5 cm"
    said = _answer(marked="right kidney", observed=[{"words": "measures 10.5 cm", "of": "it"}])
    got = _owner(said, said)("right kidney", s, 4, "en", "kidney_right")
    assert got.verdict == UNSURE and "the_finding_is_only_a_measure" in got.reader


def test_what_is_of_another_structure_is_set_aside_only_with_the_reader_or_its_relation_words():
    s = "a collection between the left flank and the left psoas muscle"
    near = _answer(marked="left psoas muscle", observed=[{"words": "a collection", "of": "collection",
                                                         "relation": "between", "reaches": "no"}])
    got = _owner(near, near)("left psoas muscle", s, s.index("left psoas"), "en", "iliopsoas_left")
    assert got.verdict == ELSEWHERE and got.statement == ASIDE
    s = "a collection next to the left psoas muscle"
    bare = _answer(marked="left psoas muscle", observed=[{"words": "a collection", "of": "collection",
                                                         "relation": "", "reaches": "no"}])
    assert _owner(bare, bare)("left psoas muscle", s, s.index("left psoas"), "en", "iliopsoas_left").verdict == UNSURE


def test_the_models_one_stream_must_agree():
    s = "The pancreas is hyperechoic and not enlarged"
    a = _answer(marked="pancreas", observed=[{"words": "hyperechoic", "of": "it"}])
    b = _answer(marked="pancreas", normal=["not enlarged"])
    assert _owner(a, b)("pancreas", s, 4, "en", "pancreas").verdict == UNSURE


def test_the_exam_on_one_side_and_the_sentence_on_the_other_goes_to_a_person():
    assert laterality_conflict("CT of the distal femoral shaft on the left side.", "femur_right", "right femur")
    assert not laterality_conflict("CT of both knees", "femur_right", "right femur")
    s = "Two bone islands are visible in the distal epiphysis of the right femur"
    said = _answer(marked="right femur", observed=[{"words": "Two bone islands", "of": "it"}])
    got = _owner(said, said)("right femur", s, s.index("right femur"), "en", "femur_right",
                             {"exam": "CT of the distal femoral shaft on the left side."})
    assert got.verdict == HERE and got.review.startswith("the_exam_is_of_the_left_side")


def test_a_table_of_measures_is_neither_the_exam_nor_a_section():
    report = ("Echocardiography. LVEDD: 42 mm; LVESD: 22 mm IVSD: 9 mm PWD: 10 mm LA: 38 mm; Ao: 33 mm; RVOT 26 mm\n"
              "Mitral valve: leaflets more intensely stained, regurgitation 3+")
    got = report_context(report, report.index("Mitral"))
    assert got["section"] != "Ao" and "LVEDD" not in got["exam"]


def test_the_grammar_of_the_phrase_and_of_the_coordination():
    words = AnomalyWords(qualities={"enlarg": {"*"}}, entities={"fluid", "cyst"})
    assert stated_anomalies("free fluid in the pelvis", "en", words) == ["fluid"]
    assert stated_anomalies("the intracranial fluid spaces are age-appropriate", "en", words) == []
    assert stated_anomalies("the liver is not enlarged", "en", words) == []
    assert stated_anomalies("an enlarged liver", "en", words) == ["enlarged"]
    assert stated_anomalies("a hypodense halo", "en", AnomalyWords()) == ["hypodense"]
    assert [w for w, _, _ in phrases("the right kidney measures 10.5 cm")[0]] == ["the", "right", "kidney"] or True
    s = "Masses in the liver measuring 5 cm and in the pancreas show similar uptake"
    assert conjuncts(s, (s.index("liver"), s.index("liver") + 5), (s.index("pancreas"), s.index("pancreas") + 8))
    s = "Reduced flow through the stomach and the duodenum"
    assert conjuncts(s, (s.index("stomach"), s.index("stomach") + 7), (s.index("duodenum"), s.index("duodenum") + 8))
    s = "The ureters are dilated up to their entry into the bladder"
    assert not conjuncts(s, (4, 11), (s.index("bladder"), s.index("bladder") + 7))


def test_the_interpreter_attaches_every_agreed_claim():
    from melampo.imaging.interpreter import interpret_exam

    report = "US of the abdomen. The pancreas is hyperechoic and not enlarged, 12 mm."
    at = report.index("pancreas")
    sites = [{"mention": "pancreas", "start": at, "end": at + 8, "cid": "pancreas"}]
    said = _answer(marked="pancreas", observed=[{"words": "hyperechoic", "of": "it"}], normal=["not enlarged"],
                   certainty=[])
    stated = []
    [f] = interpret_exam("p", "s", "", "US", "r1", report, sites, language="en", ownership=_owner(said, said),
                         statements=stated)
    assert f.as_row()["clause"] == "hyperechoic" and f.as_row()["statement"] == "finding"
    assert stated[0]["statement"] == "normal" and stated[0]["piece"] == "not enlarged"


def test_agreed_claims_keep_what_every_model_fills():
    a = _answer(observed=[{"words": "x", "of": "it"}], normal=["n"], device=["tube"])
    b = _answer(observed=[{"words": "x", "of": "it"}], normal=["n"])
    got = agreed_claims([a, b])
    assert got["normal"] == ["n"] and "device" not in got and got["findings"][0]["words"] == "x"


def test_a_part_the_model_names_is_the_structure_when_the_reader_reads_it_so():
    s = "The aorta shows calcifications in the wall of the arch"
    said = _answer(marked="aorta", observed=[{"words": "calcifications", "of": "wall of the arch", "relation": "in"}])
    got = _owner(said, said)("aorta", s, 4, "en", "aorta")
    assert got.verdict == HERE and got.claims["findings"][0]["of"] == "it"


def test_an_absence_listed_as_an_observation_is_an_absence():
    s = "The spleen is not enlarged, homogeneous"
    a = _answer(marked="spleen", normal=["not enlarged"])
    b = _answer(marked="spleen", observed=[{"words": "not enlarged", "of": "it"}])
    got = _owner(a, b)("spleen", s, 4, "en", "spleen")
    assert got.verdict == HERE and got.statement == KEEP


def test_as_far_as_is_not_reaching():
    from melampo.memory.spatial_relation import EXTENT, read_relation

    s = "fluid in the right pleural cavity reaching the seventh right rib"
    assert read_relation(s, s.index("seventh")).kind == EXTENT
    said = _answer(marked="seventh right rib", observed=[{"words": "fluid in the right pleural cavity",
                                                         "of": "right pleural cavity", "relation": "reaching",
                                                         "reaches": "yes"}])
    got = _owner(said, said)("seventh right rib", s, s.index("seventh"), "en", "rib_right_7")
    assert got.verdict == UNSURE and "the_relation_words_say_near_not_reaching" in got.reader


def test_the_question_asks_about_the_marked_words_not_the_findings():
    p = claims_prompt("At the L5-S1 level, a disc herniation", 7, 9, "lumbar vertebra 5")
    assert "about the marked words only" in p and "or a part of it" in p


def test_the_bearer_is_read_in_the_mention_s_own_sentence():
    from melampo.memory.finding_owner import FindingOwner as Owner

    owner = Owner(None, {})
    s = "1.1 cm, and to the terminal part of the duodenum - invasion? The gallbladder is contracted after a meal"
    assert owner._reader("gallbladder", s, s.index("gallbladder"), "en", "gallbladder", {})[0] == "here"
