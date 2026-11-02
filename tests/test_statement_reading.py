"""What the sentence says of the structure (1 November 2026): the model copies the piece and gives its type, the
reader checks the piece and the type, the interpreter attaches only that statement, and a person decides when the
readers do not agree."""

import json

import pytest

from melampo.evaluation.llm_reading import STATEMENT_TYPES, in_sentence, read_statement, statement_prompt
from melampo.memory.finding_owner import (
    ASIDE, ELSEWHERE, FINDING, HERE, HISTORY, MEASURE, NOTHING, UNSURE, FindingOwner, type_check, window,
)
from melampo.memory.spatial_relation import CONTAINMENT, PROJECTION, PROXIMITY, read_relation

THYROID = "History of thyroid cancer surgery, now spinal pain radiating to the left leg"
KIDNEY = "The left kidney is longer than the right kidney 10.5 cm"


def _says(**answer):
    return lambda prompt: json.dumps(answer)  # noqa: E731


def _owner(*answers, reader=None):
    return FindingOwner(reader, {f"m{i}": _says(**a) for i, a in enumerate(answers)}, question="statement")


def test_the_prompt_asks_what_the_sentence_says_of_the_structure_as_the_atlas_means_it():
    p = statement_prompt(THYROID, 11, 18, "left hip bone", linked_names=["left hip bone", "left innominate bone"],
                         exam="MRI of the lumbar spine")
    assert "<w>thyroid</w>" in p and "left hip bone (also called: left innominate bone)" in p
    assert all(t in p for t in STATEMENT_TYPES)


def test_the_piece_must_be_in_the_sentence():
    assert in_sentence(THYROID, "thyroid  cancer SURGERY")
    got = read_statement('{"piece": "thyroid nodule", "type": "finding"}', THYROID)
    assert got["type"] == "uncertain" and not got["found"]          # not read from the sentence
    got = read_statement('text {"piece": "", "type": "nothing"}', "BRAIN MRI")
    assert got["type"] == "nothing" and got["valid"]
    assert read_statement("no json", THYROID)["valid"] is False


def test_the_reader_checks_the_type_with_what_it_knows():
    assert type_check(FINDING, "right kidney 10.5 cm", "", "right kidney") == "the_piece_is_only_a_measure"
    assert type_check(MEASURE, "right kidney 10.5 cm", "", "right kidney") == ""
    assert type_check(MEASURE, "portal vein dilated to 15.5 mm", "", "portal vein")
    assert type_check(FINDING, "thyroid cancer surgery", "history of", "thyroid") == "the_piece_states_a_past_fact"
    assert type_check("normal", "liver not enlarged", "", "liver") == ""
    assert type_check(HISTORY, "anything", "", "x") == ""            # it cannot say a past fact is not one


def test_a_long_table_is_read_around_the_words():
    clause = "IVC 21 mm; " * 40 + "LA / LAA 12 mm; " + "mitral valve normal; " * 30
    text, at = window(clause, clause.index("LA /"), 2)
    assert len(text) <= 330 and text[at:at + 2] == "LA" and "mitral" not in text[:at]


def test_the_thyroid_gets_its_history_and_the_kidney_its_measure():
    owner = _owner({"piece": "thyroid cancer surgery", "type": "history"},
                   {"piece": "History of thyroid cancer surgery", "type": "history"})
    got = owner("thyroid", THYROID, THYROID.index("thyroid"), "en", "thyroid_gland")
    assert (got.verdict, got.statement, got.piece) == (HERE, HISTORY, "thyroid cancer surgery")
    owner = _owner({"piece": "right kidney 10.5 cm", "type": "measure"}, {"piece": "10.5 cm", "type": "measure"})
    got = owner("right kidney", KIDNEY, KIDNEY.index("right kidney"), "en", "kidney_right")
    assert (got.verdict, got.statement, got.piece) == (HERE, MEASURE, "10.5 cm")


def test_the_models_are_one_stream_and_nothing_gives_way():
    s = "At L4-L5 level, mild central disc bulge"
    at = s.index("L4")
    both = _owner({"piece": "mild central disc bulge", "type": "involved"},
                  {"piece": "At L4-L5 level, mild central disc bulge", "type": "finding"})
    assert both("L4", s, at, "en", "vertebrae_L4").statement == FINDING     # one group: the same attachment
    split = _owner({"piece": "disc bulge", "type": "involved"}, {"piece": "L4-L5 level", "type": "reference"})
    assert split("L4", s, at, "en", "vertebrae_L4").verdict == UNSURE
    header = _owner({"piece": "", "type": "nothing"}, {"piece": "mild central disc bulge", "type": "involved"})
    assert header("L4", s, at, "en", "vertebrae_L4").statement == FINDING
    lost = FindingOwner(None, {"m": lambda p: (_ for _ in ()).throw(TimeoutError())}, question="statement")
    assert lost("L4", s, at, "en", "vertebrae_L4").verdict == UNSURE


def test_a_type_the_reader_contradicts_goes_to_a_person_and_the_models_alone_set_nothing_aside():
    owner = _owner({"piece": "right kidney 10.5 cm", "type": "finding"}, {"piece": "10.5 cm", "type": "finding"})
    got = owner("right kidney", KIDNEY, KIDNEY.index("right kidney"), "en", "kidney_right")
    assert got.verdict == UNSURE and "the_piece_is_only_a_measure" in got.reader
    s = "Some enlarged nodes are evident along the aortic arch"
    aside = _owner({"piece": "enlarged nodes along the aortic arch", "type": "reference", "of": "lymph nodes"},
                   {"piece": "along the aortic arch", "type": "reference"})
    got = aside("aortic arch", s, s.index("aortic arch"), "en", "aorta")
    assert got.verdict == UNSURE                     # the reader (absent here) saw no reference


def test_the_interpreter_attaches_only_the_statement():
    from melampo.imaging.interpreter import interpret_exam

    report = "MRI of the lumbar spine. History of thyroid cancer surgery, now 12 mm disc protrusion at L5-S1."
    at = report.index("thyroid")
    sites = [{"mention": "thyroid", "start": at, "end": at + 7, "cid": "thyroid_gland"}]
    owner = _owner({"piece": "thyroid cancer surgery", "type": "history"},
                   {"piece": "thyroid cancer surgery", "type": "history"})
    stated, asked = [], []
    found = interpret_exam("p", "s", "", "MR", "r1", report, sites, language="en", ownership=owner,
                           questions=asked, statements=stated)
    assert found == [] and asked == []
    assert stated[0]["statement"] == HISTORY and stated[0]["piece"] == "thyroid cancer surgery"
    report = "CT of the abdomen. The right kidney measures 10.5 cm and the left 12 cm."
    at = report.index("right kidney")
    sites = [{"mention": "right kidney", "start": at, "end": at + 12, "cid": "kidney_right"}]
    owner = _owner({"piece": "right kidney measures 10.5 cm", "type": "measure"},
                   {"piece": "right kidney measures 10.5 cm", "type": "measure"})
    [f] = interpret_exam("p", "s", "", "CT", "r2", report, sites, language="en", ownership=owner)
    row = f.as_row()
    assert row["statement"] == MEASURE and row["clause"] == "right kidney measures 10.5 cm"
    assert row["text_sizes_mm"] == [105.0]


def test_the_question_may_be_only_owner_or_statement():
    with pytest.raises(ValueError):
        FindingOwner(None, {}, question="maybe")
    assert ASIDE != ELSEWHERE and NOTHING == "nothing"


def test_a_part_of_the_structure_is_placed_in_it_and_a_projection_is_weak():
    s = "mild wall thickening in the area of the descending colon"
    assert read_relation(s, s.index("descending")).kind == CONTAINMENT
    s = "enlarged nodes in the area of the aortic arch"
    assert read_relation(s, s.index("aortic")).kind == PROXIMITY
    s = "In the projection of the head of the right femur, osteolytic foci"
    assert read_relation(s, s.index("head")).kind == PROJECTION
