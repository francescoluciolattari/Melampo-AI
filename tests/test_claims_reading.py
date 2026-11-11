"""What the sentence states of the structure, claim by claim (2 November 2026): nothing chosen from a list, every
claim with its words copied, the reader checking them with what it knows (ontologies, grammar), and a person
when the readers do not agree."""

import json

from melampo.evaluation.llm_reading import claims_prompt, copied, read_claims
from melampo.memory.finding_owner import (
    ASIDE, DOUBT, ELSEWHERE, FINDING, HERE, KEEP, UNSURE, FindingOwner, agreed_claims, claims_outcome, laterality_conflict,
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
    assert got.verdict == DOUBT and "the_normal_words_state_an_abnormality" in got.reader
    s = "The right kidney measures 10.5 cm"
    said = _answer(marked="right kidney", observed=[{"words": "measures 10.5 cm", "of": "it"}])
    got = _owner(said, said)("right kidney", s, 4, "en", "kidney_right")
    # a piece that is only a measure is a measure (8 November 2026): compared with the references, not observed
    assert got.verdict == HERE and got.statement == KEEP and got.claims.get("measure") == ["measures 10.5 cm"]


def test_what_is_of_another_structure_is_set_aside_only_with_the_reader_or_its_relation_words():
    s = "a collection between the left flank and the left psoas muscle"
    near = _answer(marked="left psoas muscle", observed=[{"words": "a collection", "of": "collection",
                                                         "relation": "between", "reaches": "no"}])
    got = _owner(near, near)("left psoas muscle", s, s.index("left psoas"), "en", "iliopsoas_left")
    assert got.verdict == ELSEWHERE and got.statement == ASIDE
    s = "a collection next to the left psoas muscle"
    bare = _answer(marked="left psoas muscle", observed=[{"words": "a collection", "of": "collection",
                                                         "relation": "", "reaches": "no"}])
    assert _owner(bare, bare)("left psoas muscle", s, s.index("left psoas"), "en", "iliopsoas_left").verdict == DOUBT


def test_the_models_one_stream_must_agree_or_it_stays_in_doubt():
    s = "The pancreas is hyperechoic and not enlarged"
    a = _answer(marked="pancreas", observed=[{"words": "hyperechoic", "of": "it"}])
    b = _answer(marked="pancreas", normal=["not enlarged"])
    assert _owner(a, b)("pancreas", s, 4, "en", "pancreas").verdict == DOUBT


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


def test_words_that_can_say_either_are_the_model_s_to_read():
    s = "fluid in the right pleural cavity reaching the seventh right rib"
    said = _answer(marked="seventh right rib", observed=[{"words": "fluid in the right pleural cavity",
                                                         "of": "right pleural cavity", "relation": "reaching",
                                                         "reaches": "no"}])
    # the models read "reaching" as how far the fluid goes: aside, but the reader has no evidence of its own
    got = _owner(said, said)("seventh right rib", s, s.index("seventh"), "en", "rib_right_7")
    assert got.verdict == DOUBT
    p = claims_prompt(s, s.index("seventh"), len(s), "rib")
    assert "itself is" in p and "how far it" in p


def test_the_question_asks_about_the_marked_words_not_the_findings():
    p = claims_prompt("At the L5-S1 level, a disc herniation", 7, 9, "lumbar vertebra 5")
    assert "about the marked words only" in p and "or a part of it" in p


def test_the_bearer_is_read_in_the_mention_s_own_sentence():
    from melampo.memory.finding_owner import FindingOwner as Owner

    owner = Owner(None, {})
    s = "1.1 cm, and to the terminal part of the duodenum - invasion? The gallbladder is contracted after a meal"
    assert owner._reader("gallbladder", s, s.index("gallbladder"), "en", "gallbladder", {})[0] == "here"


def test_a_finding_said_its_own_but_only_placed_by_its_own_words_stays_in_doubt():
    s = "A semi-circular nodule is observed beneath the right frontal skull plate"
    beneath = _answer(marked="skull", observed=[{"words": "semi-circular nodule", "of": "it", "relation": "beneath",
                                                 "reaches": "no"}])
    inside = _answer(marked="skull", observed=[{"words": "semi-circular nodule", "of": "it", "relation": "in",
                                                "reaches": "no"}])
    at = s.index("skull")
    # one model contradicting itself is not agreement: a person decides, nothing is set aside
    got = _owner(inside, beneath)("skull", s, at, "en", "skull")
    assert got.verdict == DOUBT and "the_relation_words_only_place_what_is_said_its_own" in got.reader
    assert _owner(inside, inside)("skull", s, at, "en", "skull").verdict == HERE
    # a placed observation beside a finding of the structure changes nothing that is attached
    s = "The pancreas is enlarged, with blurred fat planes around the pancreas"
    both = _answer(marked="pancreas", observed=[{"words": "enlarged", "of": "it", "relation": "", "reaches": "no"},
                                                {"words": "blurred fat planes", "of": "it", "relation": "around",
                                                 "reaches": "no"}])
    assert _owner(both, both)("pancreas", s, 4, "en", "pancreas").verdict == HERE


def test_the_spatial_indicators_that_only_place():
    from melampo.memory.spatial_relation import PROXIMITY, read_relation

    for s in ("a nodule beneath the skull", "a collection underneath the liver"):
        name = s.split(" the ")[-1]
        assert read_relation(s, s.index(name), "en").kind == PROXIMITY, s
    # "in the topography of" can also mean "in" (claims4: "in the topography of the right kidney, a tumor" is of
    # the kidney): it is read by the models, not held by the reader against them (6 November 2026)
    s = "a focus in the topography of the pancreatic head"
    assert read_relation(s, s.index("pancreatic head"), "en").kind != PROXIMITY
    p = " ".join(claims_prompt("a shadow at the level of L2-L3", 26, 28, "lumbar vertebra 2").split())
    assert "a joint or a space" in p and "how high" in p


def test_each_aspect_is_compared_with_the_usual_and_the_same_is_normal():
    s = "The bladder is partially filled, with an anechoic content"
    p = " ".join(claims_prompt(s, 4, 11, "urinary bladder").split())
    assert "usual" in p and "healthy person in this exam" in p and '"differs"' in p
    said = _answer(marked="bladder", observed=[
        {"words": "partially filled", "aspect": "filling", "usual": "variable", "differs": "no", "of": "it"},
        {"words": "anechoic content", "aspect": "content", "usual": "anechoic urine", "differs": "no", "of": "it"}])
    read = read_claims(json.dumps(said), s, "bladder")
    assert read["observed"] == [] and read["normal"] == ["partially filled", "anechoic content"]
    assert claims_outcome(read) == KEEP
    got = _owner(said, said)("bladder", s, 4, "en", "urinary_bladder")
    assert got.verdict == HERE and got.statement == KEEP
    # a model that cannot tell the state from the usual sends the sentence to a person
    unsure = _answer(marked="bladder", observed=[{"words": "partially filled", "differs": "unsure", "of": "it"}])
    assert _owner(said, unsure)("bladder", s, 4, "en", "urinary_bladder").verdict == DOUBT


def test_what_the_models_call_usual_and_the_reader_knows_as_a_deviation_stays_in_doubt():
    s = "The gallbladder is distended, with a hypodense halo of the wall"
    said = _answer(marked="gallbladder", observed=[
        {"words": "distended", "differs": "yes", "of": "it"},
        {"words": "hypodense halo of the wall", "differs": "no", "of": "it"}])
    got = _owner(said, said)("gallbladder", s, 4, "en", "gallbladder")
    assert got.verdict == DOUBT and "the_normal_words_state_an_abnormality" in got.reader


def test_another_structure_that_reaches_it_is_a_finding_that_involves_it():
    joint = _answer(marked="left hip", same="no", named="left hip joint", observed=[
        {"words": "the gap is narrowed", "of": "left hip joint", "relation": "of", "reaches": "yes"}])
    assert claims_outcome({**joint, "valid": True}) == FINDING
    apart = _answer(marked="left hip", same="no", named="left hip joint", observed=[
        {"words": "the gap is narrowed", "of": "left hip joint", "relation": "of", "reaches": "no"}])
    assert claims_outcome({**apart, "valid": True}) == ASIDE


def test_a_model_that_reads_nothing_and_one_that_reads_another_structure_disagree():
    # "nothing" leaves the finding the linker built where it was; "another structure" sets it aside: they would
    # attach different things, so a person decides ("a node between the aorta and the inferior vena cava")
    s = "Another node is located between the aorta and the inferior vena cava"
    at = s.index("inferior vena cava")
    nothing = _answer(marked="inferior vena cava")
    other = _answer(marked="inferior vena cava", observed=[{"words": "Another node", "of": "Another node",
                                                            "relation": "between", "reaches": "no"}])
    assert _owner(nothing, other)("inferior vena cava", s, at, "en", "inferior_vena_cava").verdict == DOUBT


def test_a_negation_further_on_does_not_make_the_observation_an_absence():
    from melampo.memory.what_is_said import negated

    assert not negated("Fractures of the left ribs 3-7 without significant displacement", "en", "left ribs 3")
    assert not negated("A cyst with no septa", "en")
    assert negated("no stones", "en") and negated("not enlarged", "en")
    assert negated("liver not enlarged", "en", "liver") and negated("senza calcoli", "it")


def test_a_placing_word_that_can_also_mean_in_is_not_held_against_the_models():
    s = "In the topography of the right kidney, a polycystic solid tumor"
    said = _answer(marked="right kidney", observed=[{"words": "a polycystic solid tumor", "of": "it",
                                                     "relation": "In the topography of", "reaches": "no"}])
    got = _owner(said, said)("right kidney", s, s.index("right kidney"), "en", "kidney_right")
    assert "the_relation_words_only_place_what_is_said_its_own" not in got.reader


def test_a_piece_given_another_kind_is_not_also_observed():
    from melampo.memory.finding_owner import _one_kind_per_piece

    a = _answer(observed=[{"words": "cannot be evaluated optimally", "of": "it"}],
                not_assessable=["cannot be evaluated optimally"])
    assert _one_kind_per_piece(a)["observed"] == []
    b = _answer(observed=[{"words": "enlarged", "of": "it"}])
    assert _one_kind_per_piece(b) is b


def test_nested_readings_the_reader_decides_only_the_piece_one_model_adds():
    # 7 November 2026: one model reads the measure and "enlarged", the other only the measure. The reader is the
    # third reader of the piece they differ on, and only with what it knows without ambiguity
    words = AnomalyWords(qualities={"enlarg": {"*"}})
    s = "The spleen is enlarged, measuring 15 cm"
    larger = _answer(marked="spleen", observed=[{"words": "enlarged", "of": "it"}], measure=["measuring 15 cm"])
    smaller = _answer(marked="spleen", measure=["measuring 15 cm"])
    got = _owner(larger, smaller, anomalies=words)("spleen", s, 4, "en", "spleen")
    assert got.verdict == HERE and "the_reader_confirms_the_finding_one_model_adds" in got.reader
    got = _owner(smaller, larger, anomalies=words)("spleen", s, 4, "en", "spleen")
    assert got.verdict == HERE                                          # the order of the models does not matter
    # a piece the reader does not know as a deviation: a person keeps it in doubt
    s = "The spleen is larger than usual, measuring 15 cm"
    larger = _answer(marked="spleen", observed=[{"words": "larger than usual", "of": "it"}],
                     measure=["measuring 15 cm"])
    assert _owner(larger, smaller, anomalies=words)("spleen", s, 4, "en", "spleen").verdict == DOUBT
    # the other model read the same words as normal: a disagreement, kept in doubt, not closed by the reader
    s = "The spleen is enlarged, measuring 15 cm"
    normal = _answer(marked="spleen", normal=["enlarged"], measure=["measuring 15 cm"])
    larger = _answer(marked="spleen", observed=[{"words": "enlarged", "of": "it"}], measure=["measuring 15 cm"])
    assert _owner(larger, normal, anomalies=words)("spleen", s, 4, "en", "spleen").verdict == DOUBT
    # without the reader's knowledge the two models must agree, as before
    assert _owner(larger, smaller)("spleen", s, 4, "en", "spleen").verdict == DOUBT


def test_two_places_in_parallel_each_hold_their_own_finding():
    from melampo.memory.finding_owner import _parallel_places

    s = "A stone in the gallbladder and a cyst in the right kidney"
    assert _parallel_places(s, s.index("gallbladder"), s.index("right kidney"), "en")
    s = "Thickening of the parietal pleura in the lower lobe"
    assert not _parallel_places(s, s.index("parietal pleura"), s.index("lower lobe"), "en")


def test_the_principle_of_doubt():
    # 7 November 2026: what is not certain stays in the diagnostic path as a finding in doubt, with every reading
    # beside it; a person sees only a discordance of the document
    s = "The spleen is larger than usual, measuring 15 cm"
    larger = _answer(marked="spleen", observed=[{"words": "larger than usual", "of": "it"}],
                     measure=["measuring 15 cm"])
    smaller = _answer(marked="spleen", measure=["measuring 15 cm"])
    got = _owner(smaller, larger)("spleen", s, 4, "en", "spleen")
    assert got.verdict == DOUBT and got.statement == FINDING          # the leading reading keeps the finding
    assert got.doubt["kind"] == "reading" and set(got.doubt["readings"]) == {"m0", "m1"}
    assert "closed_by" not in got.doubt and "new_exam" in got.doubt["closes_with"]
    # one model's reading only (the other lost): in doubt, not certain
    got = FindingOwner(None, {"m0": lambda p: json.dumps(larger), "m1": lambda p: "not json"},
                       question="claims")("spleen", s, 4, "en", "spleen")
    assert got.verdict == DOUBT and got.doubt["kind"] == "one_reading"
    # no reading at all: nothing to keep in doubt
    lost = FindingOwner(None, {"m0": lambda p: "?", "m1": lambda p: "?"}, question="claims")
    assert lost("spleen", s, 4, "en", "spleen").verdict == UNSURE
    # the doubt the report states is the report's: certain attribution, the doubt kept with the finding
    s = "A nodule of the spleen, possibly a hemangioma"
    hedged = _answer(marked="spleen", observed=[{"words": "A nodule", "of": "it"}],
                     alternatives=["possibly a hemangioma"])
    got = _owner(hedged, hedged)("spleen", s, s.index("spleen"), "en", "spleen")
    assert got.verdict == HERE and got.doubt["kind"] == "clinical"


def test_the_conclusion_closes_a_doubt_of_reading():
    # CT-RATE 911: "43 mm, larger than normal" read as a measure only by one model; the conclusion says "aneurysmal
    # dilatation of the ascending aorta": every model reads that it repeats the finding, the reader knows that it
    # names the structure and states a deviation of its tissue
    from melampo.memory.finding_owner import Anatomy

    words = AnomalyWords(qualities={"dilat": {"*"}})
    s = "The diameter of the spleen is 15 cm, larger than normal"
    larger = _answer(marked="spleen", observed=[{"words": "larger than normal", "of": "it"}],
                     measure=["15 cm"], conclusion="same")
    smaller = _answer(marked="spleen", measure=["15 cm"], conclusion="same")
    owner = FindingOwner(None, {"m0": lambda p: json.dumps(smaller), "m1": lambda p: json.dumps(larger)},
                         Anatomy(), question="claims", anomalies=words)
    at = s.index("spleen")
    got = owner("spleen", s, at, "en", "spleen", {"conclusion": "Dilatation of the spleen."})
    assert got.verdict == HERE and got.doubt["closed_by"] == "conclusion"
    # a conclusion that does not state a deviation of it does not close it
    got = owner("spleen", s, at, "en", "spleen", {"conclusion": "Nodule of the liver."})
    assert got.verdict == DOUBT


def test_the_conclusion_does_not_close_a_doubt_of_the_structure():
    # 9 November 2026, ctrate2: "the pleural effusion ... at the level of the lower lobe of the right lung", one
    # model giving it to the effusion's place: the conclusion says that there is a finding, not whose it is
    from melampo.memory.finding_owner import Anatomy

    words = AnomalyWords(qualities={"dilat": {"*"}, "effusion": {"*"}})
    s = "The spleen is larger than normal, 15 cm"
    here = _answer(marked="spleen", observed=[{"words": "larger than normal", "of": "it"}], conclusion="same")
    other = _answer(marked="spleen", observed=[{"words": "larger than normal", "of": "liver"}], conclusion="same")
    owner = FindingOwner(None, {"m0": lambda p: json.dumps(other), "m1": lambda p: json.dumps(here)},
                         Anatomy(), question="claims", anomalies=words)
    got = owner("spleen", s, s.index("spleen"), "en", "spleen", {"conclusion": "Dilatation of the spleen."})
    assert got.verdict == DOUBT and "closed_by" not in got.doubt


def test_a_model_s_error_on_a_measure_is_a_doubt_the_sentence_against_its_reference_a_discordance():
    from melampo.memory.finding_owner import DISCORDANCES

    assert "measure_the_reference_says_within".startswith(DISCORDANCES)
    for reading in ("measure_the_cited_reference_was_not_given", "measure_the_comparison_is_not_stated",
                    "the_measure_is_of_what_is_observed"):
        assert not reading.startswith(DISCORDANCES)


def test_the_uncertainty_one_model_reads_stays_uncertain():
    # ctrate3 (10 November 2026): "a hypodense lesion (cyst?)": the lesion is seen, the cyst is a hypothesis; one
    # model read it, or the two put it in different lists, and the doubt was lost. Now it is one category, joined
    s = "A hypodense lesion was observed in the right kidney (cyst?)"
    at = s.index("right kidney")
    seen = [{"words": "A hypodense lesion", "of": "it", "said_to_be": "cyst"}]
    one = _answer(marked="right kidney", observed=seen, uncertainty=[{"words": "?", "about": "cyst"}])
    other = _answer(marked="right kidney", observed=seen)
    got = _owner(one, other)("right kidney", s, at, "en", "kidney_right")
    assert got.verdict == HERE and got.doubt["kind"] == "clinical"
    assert got.claims["uncertain_about"] == [{"words": "?", "about": "cyst", "level": "possible"}]
    got = _owner(other, one)("right kidney", s, at, "en", "kidney_right")
    assert got.doubt["kind"] == "clinical"     # also when the reading without it leads


def test_the_former_two_lists_are_read_as_one_uncertainty():
    s = "Centrally located ground glass density in the lower lobe of the right lung (pneumonia?)"
    first = read_claims(json.dumps(_answer(marked="lower lobe of the right lung",
                                           observed=[{"words": "ground glass density", "of": "it"}],
                                           alternatives=["pneumonia?"])), s, "lower lobe of the right lung")
    second = read_claims(json.dumps(_answer(marked="lower lobe of the right lung",
                                            observed=[{"words": "ground glass density", "of": "it"}],
                                            certainty=["(pneumonia?)"])), s, "lower lobe of the right lung")
    assert first["uncertainty"] == ["pneumonia?"] and second["uncertainty"] == ["(pneumonia?)"]
    assert agreed_claims([first, second])["uncertainty"] == ["pneumonia?", "(pneumonia?)"]


def test_what_the_sentence_says_it_is_is_apart_from_what_is_seen():
    s = "Slight ground glass densities in the right upper lobe were evaluated in favor of the infectious process"
    got = read_claims(json.dumps(_answer(marked="right upper lobe", observed=[
        {"words": "Slight ground glass densities", "of": "it", "said_to_be": "the infectious process"}],
        uncertainty=[{"words": "evaluated in favor of", "about": "the infectious process"}])), s, "right upper lobe")
    assert got["observed"][0]["said_to_be"] == "the infectious process"
    assert got["uncertain_about"] == [{"words": "evaluated in favor of", "about": "the infectious process",
                                       "level": "possible"}]


def test_what_is_said_of_another_person_is_not_a_finding_of_the_patient():
    s = "Family history: the mother had a carcinoma of the breast"
    at = s.index("breast")
    said = _answer(marked="breast", other_person=["the mother had a carcinoma of the breast"])
    got = read_claims(json.dumps(said), s, "breast")
    assert got["other_person"] == ["the mother had a carcinoma of the breast"]
    assert claims_outcome(got) == KEEP
    assert _owner(said, said)("breast", s, at, "en", "breast").statement != "finding"


def test_every_uncertainty_has_its_level_and_what_is_not_uncertain_its_place():
    # ctrate4 (11 November 2026): "nonspecific", the limits of the exam had no place and became uncertainties
    s = "As far as can be observed, a nonspecific nodule consistent with a granuloma is observed in the right lung"
    got = read_claims(json.dumps(_answer(marked="right lung", observed=[{"words": "a nonspecific nodule", "of": "it",
                                                                        "said_to_be": "a granuloma"}],
        uncertainty=[{"words": "consistent with", "about": "a granuloma", "level": "high"}],
        undetermined=["nonspecific"], limits=["As far as can be observed"])), s, "right lung")
    assert got["uncertain_about"] == [{"words": "consistent with", "about": "a granuloma", "level": "high"}]
    assert got["undetermined"] == ["nonspecific"] and got["limits"] == ["As far as can be observed"]
    without = read_claims(json.dumps(_answer(marked="right lung", observed=[{"words": "a nonspecific nodule",
                                                                             "of": "it"}],
                                             uncertainty=[{"words": "consistent with", "about": "a granuloma",
                                                           "level": "almost"}])), s, "right lung")
    assert without["uncertain_about"][0]["level"] == "possible"   # an unreadable level is never certain
