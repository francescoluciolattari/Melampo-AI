"""The question a person is asked, complete in itself (1 November 2026): its kind, the sentence and the structure,
the closed answers, the system's readings kept for after the answer, what an answer means and how it is stored."""

import pytest

from melampo.memory.finding_owner import Ownership
from melampo.memory.review_question import (
    ASIDE, KEEP, OPEN, STATEMENTS, build_question, decide, kind_of, review_record,
)


def _question(reader="landmark", models=None, placed="enlarged nodes", context=None):
    sentence = "Some enlarged nodes are evident along the aortic arch"
    owner = Ownership("unsure", reader, models or {"m": {"linked": "reference", "answer": "no"}}, placed,
                      "reader_and_model_disagree")
    return build_question(owner, report="r1", language="en", sentence=sentence, mention="aortic arch",
                          start=sentence.index("aortic arch"), structure="aorta", structure_name="aorta",
                          context=context or {"exam": "CT of the chest", "section": "Findings"})


def test_the_question_is_complete_in_itself():
    q = _question()
    assert q.kind == "reference_or_finding" and "aorta" in q.question_it and q.exam == "CT of the chest"
    assert q.sentence[q.start:q.start + len(q.mention)] == "aortic arch"
    assert [o["code"] for o in q.options] == list(STATEMENTS)
    assert all(o["label_it"] and o["label_en"] for o in q.options)
    # the system's readings are kept, and not shown before the answer
    assert q.system["reader"] == "landmark" and "system" not in q.for_the_person()


def test_the_kind_follows_what_the_readers_saw():
    assert kind_of("outside_the_exam:pelvis", {}) == "outside_the_exam"
    assert kind_of("named_only_in_a_differential", {}) == "differential"
    assert kind_of("describes:ureter", {}) == "other_organ"
    assert kind_of("", {"a": {"linked": "reference"}}) == "reference_or_finding"
    assert kind_of("", {}) == "what_does_it_say"
    q = _question(reader="describes:ureter", placed="")
    assert "ureter" in q.question_it


def test_what_an_answer_means():
    q = _question()
    assert decide(q, "involved")["outcome"] == KEEP
    assert decide(q, "history")["statement"] == "history"
    got = decide(q, "reference", "lymph nodes")
    assert got["outcome"] == ASIDE and got["of"] == "lymph nodes"
    assert decide(q, "cannot_say")["outcome"] == OPEN
    with pytest.raises(ValueError):
        decide(q, "reference")                      # of what, then?
    with pytest.raises(ValueError):
        decide(q, "maybe")


def test_the_answer_is_a_review_record_that_can_teach_once_confirmed():
    q = _question()
    record = review_record(q, "reference", "dr-a", "lymph nodes", pseudonymised=True)
    assert record.problems() == [] and record.roles == ["landmark"] and record.site
    assert record.system["answer"] == "reference" and record.system["of"] == "lymph nodes"
    assert not record.confirmed
    record.confirmed_by = "dr-b"
    assert record.confirmed
    other = review_record(q, "other_structure", "dr-a", "subclavian artery", pseudonymised=True)
    assert other.structure == "subclavian artery"
    assert review_record(q, "cannot_say", "dr-a", pseudonymised=True).problems() == []


def test_the_interpreter_writes_the_question_for_each_reading_sent_to_a_person():
    import json

    from melampo.imaging.interpreter import interpret_exam
    from melampo.memory import chunk_lattice as cl
    from melampo.memory.ci_reader import CIReader
    from melampo.memory.finding_owner import FindingOwner
    from melampo.memory.longer_names import LongerNames

    reader = CIReader(cl.ChunkLattice(cl.BlockMemory.load(cl.DEFAULT_PATH, LongerNames.load().names,
                                                          cl.load_anatomy_names())))
    report = "CT of the chest. Some enlarged nodes of 12 mm are evident along the aortic arch."
    at = report.index("aortic arch")
    sites = [{"mention": "aortic arch", "start": at, "end": at + 11, "cid": "aorta"}]
    chat = lambda prompt: json.dumps({"finding": "nodes", "linked": "reference", "answer": "no"})  # noqa: E731
    aside, asked = [], []
    interpret_exam("p", "s", "", "CT", "r1", report, sites, language="en",
                   ownership=FindingOwner(reader, {"m": chat}), set_aside=aside, questions=asked)
    [q] = asked
    assert q["kind"] == "reference_or_finding" and q["structure"] == "aorta" and aside[0]["question"] == q["id"]
    assert q["sentence"][q["start"]:q["start"] + 11] == "aortic arch" and q["exam"].startswith("CT of the chest")
