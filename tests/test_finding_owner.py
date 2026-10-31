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


def _chat(roles, placed=""):
    def chat(prompt):
        sentence = prompt.rsplit("Sentence: ", 1)[1].split("\n")[0]
        phrase = sentence.split("<w>")[1].split("</w>")[0]
        return json.dumps({"phrase": phrase, "head": phrase.split()[-1], "type": "anatomical structure",
                           "governor": {"word": "", "names": ""}, "predicate": "", "placed": placed,
                           "readings": [], "roles": roles})
    return chat


def test_the_reader_and_the_model_decide_together(reader):
    s = "Some enlarged nodes are evident at the hila, along the aortic arch"
    at = s.index("aortic arch")
    # the reader alone never removes a finding
    assert FindingOwner(reader)("aortic arch", s, at).verdict == UNSURE
    got = FindingOwner(reader, {"m": _chat(["landmark"], "enlarged nodes")})("aortic arch", s, at)
    assert got.verdict == ELSEWHERE and got.placed == "enlarged nodes"
    # the model reads it as the arch's: they contradict each other, a person decides
    assert FindingOwner(reader, {"m": _chat(["structure"])})("aortic arch", s, at).verdict == UNSURE
    # two models are one stream: they must agree with each other first
    two = {"a": _chat(["landmark"]), "b": _chat(["structure", "landmark"])}
    assert FindingOwner(reader, two)("aortic arch", s, at).verdict == UNSURE
    s = "Calcification of the aortic arch"
    assert FindingOwner(reader, {"m": _chat(["structure"])})("aortic arch", s, s.index("aortic arch")).verdict == HERE


def test_the_interpreter_sets_aside_a_clause_about_something_else(reader):
    from melampo.imaging.interpreter import interpret_exam

    report = "Some enlarged nodes of 12 mm are evident along the aortic arch. Calcification of the aortic arch."
    first, second = report.index("aortic arch"), report.rindex("aortic arch")
    sites = [{"mention": "aortic arch", "start": a, "end": a + 11, "cid": "aorta"} for a in (first, second)]
    aside = []
    got = interpret_exam("p", "s", "", "CT", "r", report, sites, language="en",
                         ownership=FindingOwner(reader, {"m": _chat(["landmark"], "nodes")}), set_aside=aside)
    assert [f.text.start for f in got] == [] or all(f.text.start == second for f in got)
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
