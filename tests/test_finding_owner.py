"""Of what is the finding said (31 October 2026): the relation read before the mention, the reader's second stage
(structure or landmark), the agreement of the reader and the model on the outcome, the interpreter that sets a
clause aside, and the gold set that takes the new role."""

import json

import pytest

from melampo.memory import spatial_relation as sr
from melampo.memory.finding_owner import ELSEWHERE, HERE, UNSURE, FindingOwner


def _relation(sentence, mention, language="en"):
    return sr.read_relation(sentence, sentence.index(mention), language)


@pytest.mark.parametrize("sentence,mention,kind", [
    ("Some enlarged nodes are evident at the hila, along the aortic arch", "aortic arch", sr.PROXIMITY),
    ("Calcification of the aortic arch", "aortic arch", sr.CONTAINMENT),
    ("two nodular densities near the anterior pole of the spleen and the splenic flexure", "splenic flexure",
     sr.PROXIMITY),
    ("fluid accumulation between the loops of the small intestine", "small intestine", sr.PROXIMITY),
    ("Dilatation of small bowel loops", "small bowel", sr.CONTAINMENT),
    ("In the dorsal part of the right kidney, a cortical cyst", "right kidney", sr.CONTAINMENT),
    ("The appendix, located internally and laterally to the cecum, was enlarged", "cecum", sr.PROXIMITY),
    ("In the area of the gallbladder, a tubular structure", "gallbladder", sr.PROXIMITY),
    ("hypertrophy of the zone protruding into the bladder", "bladder", sr.DIRECTION),
])
def test_the_relation_before_the_mention(sentence, mention, kind):
    assert _relation(sentence, mention).kind == kind


def test_the_italian_relations_and_a_junction():
    assert _relation("Linfonodi vicino al surrene sinistro", "surrene", "it").indicator == "vicino al"
    assert _relation("versamento fluido tra le anse del piccolo intestino", "piccolo", "it").kind == sr.PROXIMITY
    assert _relation("Nel lobo superiore destro nodulo", "lobo", "it").kind == sr.CONTAINMENT
    # a junction between two structures is part of both
    got = _relation("stenosi alla confluenza tra vena cefalica e succlavia", "vena cefalica", "it")
    assert got.kind == sr.CONTAINMENT


@pytest.fixture(scope="module")
def reader():
    from melampo.memory import chunk_lattice as cl
    from melampo.memory.ci_reader import CIReader
    from melampo.memory.longer_names import LongerNames

    lattice = cl.ChunkLattice(cl.BlockMemory.load(cl.DEFAULT_PATH, LongerNames.load().names, cl.load_anatomy_names()))
    return CIReader(lattice)


def test_the_reader_reads_a_landmark_in_its_second_stage_only(reader):
    s = "Some enlarged nodes are evident at the hila, along the aortic arch"
    assert reader.read("aortic arch", s, s.index("aortic arch")).decision == "landmark"
    s = "Calcification of the aortic arch"
    assert reader.read("aortic arch", s, s.index("aortic arch")).decision == "structure"
    # the readings of the first stage are never moved by the relation
    s = "Ultrasound: liver biopsy near the hilum"
    assert reader.read("liver", s, s.index("liver")).decision == "procedure_site"
    from melampo.memory.ci_reader import CIReading, LANDMARK

    assert CIReading(LANDMARK, LANDMARK, 1.0, {LANDMARK: 1.0}, 1).p_not_structure == 0.0   # still a body site


def _chat(answer, linked="same", finding=""):
    """A model answering the owner question (``llm_reading.owner_prompt``)."""
    def chat(prompt):
        return json.dumps({"finding": finding, "bearer": "", "linked": linked, "answer": answer, "why": ""})
    return chat


def test_the_reader_and_the_model_decide_together(reader):
    s = "Some enlarged nodes are evident at the hila, along the aortic arch"
    at = s.index("aortic arch")
    # the reader alone never removes a finding
    assert FindingOwner(reader)("aortic arch", s, at, cid="aorta").verdict == UNSURE
    got = FindingOwner(reader, {"m": _chat("no", "reference", "enlarged nodes")})("aortic arch", s, at, cid="aorta")
    assert got.verdict == ELSEWHERE and got.placed == "enlarged nodes"
    # the model reads it as the arch's: they contradict each other, a person decides
    assert FindingOwner(reader, {"m": _chat("yes")})("aortic arch", s, at, cid="aorta").verdict == UNSURE
    # two models are one stream: they must agree with each other first
    two = {"a": _chat("no", "reference"), "b": _chat("uncertain", "uncertain")}
    assert FindingOwner(reader, two)("aortic arch", s, at, cid="aorta").verdict == UNSURE
    s = "Calcification of the aortic arch"
    got = FindingOwner(reader, {"m": _chat("yes")})("aortic arch", s, s.index("aortic arch"), cid="aorta")
    assert got.verdict == HERE


def test_the_owner_answer_is_read_and_its_steps_must_agree():
    from melampo.evaluation.llm_reading import owner_prompt, read_owner

    p = owner_prompt("Minimal hypertrophy of the middle lobe", 27, 38, "middle lobe of right lung",
                     exam="MRI of the prostate", section="TZ", language="English")
    assert "middle lobe of right lung" in p and "MRI of the prostate" in p and "<w>middle lobe</w>" in p
    assert read_owner('{"linked": "wrong", "answer": "no"}')["answer"] == "no"
    assert read_owner('{"linked": "reference", "answer": "yes"}')["answer"] == "uncertain"   # steps contradict
    assert read_owner("no JSON")["answer"] == "uncertain"


def test_what_a_radiologist_knows_besides_the_words(reader):
    from melampo.memory.finding_owner import Anatomy, report_context

    owner = FindingOwner(reader, {}, Anatomy())
    # the exam: a lobe of the lung in a prostate MRI is outside what the exam sees
    report = "RM prostata multiparametrica. TZ: Apprezzabile minima ipertrofia del lobo medio che impronta il pavimento."
    clause = "Apprezzabile minima ipertrofia del lobo medio che impronta il pavimento"
    context = report_context(report, report.index(clause))
    assert "pelvis" in context["areas"] and context["section"] == "TZ"
    got = owner("lobo medio", clause, clause.index("lobo medio"), "it", "lung_middle_lobe_right", context)
    assert got.verdict == UNSURE and "outside_the_exam" in got.reader
    # a differential names a structure only as a possible explanation
    clause = "Partial opacity of the left apex (differential diagnosis: inflammatory changes, shadow of the left subclavian artery)"
    got = owner("left subclavian artery", clause, clause.index("left subclavian"), "en", "subclavian_artery_left")
    assert "named_only_in_a_differential" in got.reader
    # a hedge on the finding itself still speaks of the structure
    clause = "Clinical question: suspected liver lesion"
    assert owner("liver", clause, clause.index("liver"), "en", "liver").verdict == HERE


def test_another_organ_described_first():
    from melampo.memory import hpo_candidates as hc
    from melampo.memory.finding_owner import Anatomy

    obo = """[Term]
id: UBERON:1
name: ureter
subset: organ_slim

[Term]
id: UBERON:2
name: urinary bladder
subset: organ_slim

[Term]
id: UBERON:3
name: wall of urinary bladder
relationship: part_of UBERON:2 ! urinary bladder

[Term]
id: UBERON:4
name: compound organ
subset: organ_slim
"""
    uberon = hc.Anatomy.from_obo(obo.splitlines())
    anatomy = Anatomy(uberon, uberon_names={"UBERON:1": ["ureter"], "UBERON:2": ["urinary bladder"],
                                            "UBERON:3": ["wall of urinary bladder"]})
    assert anatomy.other_organ({"UBERON:1"}, {"UBERON:2"})
    assert not anatomy.other_organ({"UBERON:3"}, {"UBERON:2"})          # a part of the same organ
    assert "UBERON:4" not in anatomy.organs                             # a kind of organ is not an organ
    assert anatomy.find("The ureters are dilated up to their entry into the")[0][0] == "ureter"


def test_the_interpreter_sets_aside_a_clause_about_something_else(reader):
    from melampo.imaging.interpreter import interpret_exam

    report = "Some enlarged nodes of 12 mm are evident along the aortic arch. Calcification of the aortic arch."
    first, second = report.index("aortic arch"), report.rindex("aortic arch")
    sites = [{"mention": "aortic arch", "start": a, "end": a + 11, "cid": "aorta"} for a in (first, second)]
    aside = []
    got = interpret_exam("p", "s", "", "CT", "r", report, sites, language="en",
                         ownership=FindingOwner(reader, {"m": _chat("no", "reference", "nodes")}), set_aside=aside)
    assert all(f.text.start == second for f in got)
    assert aside and aside[0]["structure"] == "aorta" and aside[0]["placed"] == "nodes"
    # with the reader alone the finding is made and goes to a person
    got = interpret_exam("p", "s", "", "CT", "r", report, sites[:1], language="en",
                         ownership=FindingOwner(reader), set_aside=[])
    assert got and got[0].image.needs_review
    assert any(r.startswith("is_the_finding_of_this_structure") for r in got[0].image.review)


def test_the_gold_set_takes_the_landmark_and_several_roles():
    from melampo.evaluation.gold_set import validate_sheet

    rows = {"1": {"item_id": "1", "structure": "aorta", "relation": "equal", "role": "landmark", "mention": "aorta"},
            "2": {"item_id": "2", "structure": "colon", "relation": "part_of", "role": "structure+landmark",
                  "mention": "flessura"},
            "3": {"item_id": "3", "structure": "colon", "relation": "equal", "role": "structure+inside",
                  "mention": "colon"}}
    problems = validate_sheet(rows, ["aorta", "colon"])
    assert len(problems) == 1 and problems[0].startswith("3:")


def test_only_the_senses_of_a_person_s_body_are_recalled():
    # 7 November 2026: "corpus" was read as the septomaxilla of amphibians (UBERON:3000645) and "aortic arch" as
    # the embryonic pharyngeal arch arteries ("aortic arch 1" with its digit dropped)
    from melampo.memory import hpo_candidates as hc
    from melampo.memory.finding_owner import EMBRYONIC, Anatomy

    obo = f"""[Term]
id: {EMBRYONIC}
name: embryonic structure

[Term]
id: UBERON:0003118
name: pharyngeal arch artery 1
is_a: {EMBRYONIC} ! embryonic structure

[Term]
id: UBERON:0001508
name: arch of aorta
subset: organ_slim
"""
    uberon = hc.Anatomy.from_obo(obo.splitlines())
    anatomy = Anatomy(uberon, uberon_names={"UBERON:0003118": ["aortic arch 1", "first arch artery"],
                                            "UBERON:3000645": ["corpus"],
                                            "UBERON:0001508": ["arch of aorta"]})
    assert not anatomy.find("the first arch artery")
    assert not anatomy.find("the corpus of the stomach", short=True) or all(
        "UBERON:3000645" not in ids for _, ids in anatomy.find("the corpus of the stomach", short=True))
    assert any("UBERON:0001508" in ids for _, ids in anatomy.find("the arch of aorta is calcified"))
