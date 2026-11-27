"""The confirmations of the readings (27 November 2026): the doubt of a card asked with closed answers, confirmed
by a second person and adjudicated by a third when no interpreter was right; the sentence de-identified and read
again to the same card before it may train; one case in five only measures."""

import datetime as dt
import json
import sys
from pathlib import Path

import pytest

from melampo.data.case_review import CaseFolder
from melampo.data.deidentify import deidentify, shift_of
from melampo.evaluation.review_feedback import (
    ConfirmationStore,
    ReadingConfirmation,
    case_key,
    check_deidentified,
    part_of,
)

ROOT = Path(__file__).resolve().parents[1]
ROSSI = {"patient_fiscal_code": "RSSMRA80A01H501U", "patient_name": "Mario", "patient_surname": "Rossi"}


def test_the_sentence_is_kept_as_written_without_the_person():
    text = ("Paziente: Rossi Mario, nato a Bologna il 01/02/1940, CF RSSMRA40B01A944X, residente in Via Roma 12, "
            "tel. 333 1234567. N. cartella 2026/4581. Esame del 15/03/2026 rispetto al precedente del 01/12/2025: "
            "nodulo di 12 mm, PCR 52 mg/L. Paziente di 92 anni. Il sig. Rossi riferisce dolore dal 2019. "
            "Refertato dal Dott. Bianchi.")
    done = deidentify(text, ["Mario", "Rossi"], 40)
    for gone in ("Rossi", "Mario", "Bologna", "RSSMRA40B01A944X", "Via Roma", "333 1234567", "4581", "Bianchi",
                 "92 anni", "15/03/2026"):
        assert gone not in done.text
    for kept in ("nodulo di 12 mm", "PCR 52 mg/L", "riferisce dolore dal 2019"):   # what the sentence says
        assert kept in done.text
    # the dates move together: the interval between the two exams is the same
    assert "24/04/2026" in done.text and "10/01/2026" in done.text
    assert (dt.date(2026, 4, 24) - dt.date(2026, 1, 10)) == (dt.date(2026, 3, 15) - dt.date(2025, 12, 1))
    assert done.kinds["date"] == 3 and done.kinds["name"] >= 3 and "Rossi" not in json.dumps(done.kinds)
    assert shift_of("caso-1") == shift_of("caso-1") and 30 <= abs(shift_of("caso-2")) <= 364


READING = {"texts": [{"report": "note", "cards": [
    {"words": "ectasia del rene", "name": "renal pelvis dilatation", "category": "manifestation", "specific": True,
     "what": "renal pelvis", "interpretation": "increased", "value": None, "threshold": None, "site": "rene",
     "side": None, "observed_by": "imaging", "time": "", "nature": "pathological", "part": "findings",
     "state": "present", "temporality": "current", "subject": "patient", "doubts": ["side"],
     "alternatives": {"side": {"a": "left", "b": "right"}},
     "quote": "Il sig. Rossi presenta ectasia del rene sinistro."}]}]}


@pytest.fixture()
def case(tmp_path):
    folder = CaseFolder.create("caso-1", ROSSI, tmp_path)
    folder.save_reading(READING)
    return folder


def test_a_doubt_of_a_card_is_a_closed_question_with_each_interpreters_reading(case):
    [q] = case.card_questions()
    assert q["kind"] == "card:field" and q["field"] == "side"
    assert [o["code"] for o in q["options"]] == ["value:1", "value:2", "other", "not_stated"]
    assert {o.get("value") for o in q["options"][:2]} == {"left", "right"}


def test_an_answer_trains_only_once_a_second_person_confirms_it_de_identified(case, tmp_path):
    [q] = case.card_questions()
    left = next(o["code"] for o in q["options"] if o.get("value") == "left")
    case.answer_card(q["id"], left, "dott. Verdi")
    assert case.confirmations() == []                                     # a proposal
    with pytest.raises(ValueError):
        case.confirm_card(q["id"], "Dott. VERDI")                         # not the same person
    case.confirm_card(q["id"], "dott.ssa Neri")
    [record] = case.confirmations()
    assert record.card["side"] == "left" and record.confirmed and not record.needs_adjudication
    assert "Rossi" not in record.sentence and "[NOME]" in record.sentence and record.card["quote"] == record.sentence
    assert "Verdi" not in json.dumps(record.__dict__) and record.case == case_key("caso-1")
    assert record.part == part_of(record.case)


def test_an_answer_no_interpreter_gave_waits_for_a_third_person(case):
    [q] = case.card_questions()
    case.answer_card(q["id"], "other", "dott. Verdi", "bilaterale")
    case.confirm_card(q["id"], "dott.ssa Neri")
    assert case.confirmations() == []
    with pytest.raises(ValueError):
        case.confirm_card(q["id"], "dott. Verdi", "adjudicated")         # the third is neither of the two
    case.confirm_card(q["id"], "prof. Bruni", "adjudicated")
    [record] = case.confirmations()
    assert record.needs_adjudication and record.confirmed and record.card["written"] == "bilaterale"


def _record(part: str, side: str = "left") -> ReadingConfirmation:
    card = {**READING["texts"][0]["cards"][0], "side": side, "quote": "[NOME] presenta ectasia del rene sinistro."}
    for k in ("doubts", "alternatives"):
        card.pop(k)
    return ReadingConfirmation(case="c-x", document="note", sentence=card["quote"], card=card, doubt="side",
                               answer="value:1", system={}, decided_by="p-1", confirmed_by="p-2",
                               deidentified=True, part=part)


def test_a_de_identified_sentence_trains_only_when_the_interpreters_read_it_to_the_confirmed_card(tmp_path):
    good, wrong = _record("train"), _record("train")
    assert check_deidentified(good, lambda s: [{**good.card, "doubts": []}]) is True
    assert check_deidentified(wrong, lambda s: [{**wrong.card, "side": "right", "doubts": []}]) is False
    measured = _record("measure")
    store = ConfirmationStore(tmp_path / "archive.jsonl")
    for r in (good, wrong, measured):
        store.add(r)
    [row] = store.rows("train")
    assert json.loads(row["completion"][0]["content"])[0]["side"] == "left"
    assert "[NOME] presenta ectasia" in row["prompt"][0]["content"]
    assert len(store.rows("measure")) == 1                               # measured, never trained
    with pytest.raises(ValueError):
        store.add(ReadingConfirmation(case="c", document="n", sentence="x", card={"quote": "x"}, doubt="side",
                                      answer="a", system={}, decided_by="p-1"))   # not de-identified


def test_the_archive_is_collected_from_the_cases_once(case, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import reading_confirmations as rc

    [q] = case.card_questions()
    case.answer_card(q["id"], q["options"][0]["code"], "dott. Verdi")
    case.confirm_card(q["id"], "dott.ssa Neri")
    archive = tmp_path / "archive.jsonl"
    counts = rc.collect(case.path.parent, archive, check=True, read=lambda s: [])
    assert counts == {"added": 1, "known": 0, "checked": 1, "passed": 0}
    assert rc.collect(case.path.parent, archive, check=False)["known"] == 1


def test_the_reading_keeps_what_each_interpreter_read_of_a_doubted_field():
    from test_report_reading_sides import base, interpreter

    from melampo.data.report_reading import ReportReaders, build_linker

    text = "Ectasia del rene sinistro."
    a = base(words="Ectasia del rene", name="renal pelvis dilatation", site="rene", side="left", quote=text)
    b = {**a, "side": "right"}
    reading = ReportReaders(build_linker()[0], interpreters={"a": interpreter([a]), "b": interpreter([b])}) \
        .read("c1", text)
    [card] = reading["texts"][0]["cards"]
    assert "side" in card["doubts"] and card["alternatives"] == {"side": {"a": "left", "b": "right"}}
    assert "readings" not in card


def test_the_mask_asks_the_doubt_and_shows_the_readings_only_after_the_answer(case):
    import importlib.util

    spec = importlib.util.spec_from_file_location("review_mask_script", ROOT / "scripts/review_mask.py")
    mask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mask)
    before = mask.case_page(case)
    assert "Dubbi delle schede (1)" in before and "side: &quot;left&quot;" in before
    assert "Che cosa aveva letto ogni interprete" not in before
    [q] = case.card_questions()
    case.answer_card(q["id"], "value:1", "dott. Verdi")
    after = mask.case_page(case)
    assert "Che cosa aveva letto ogni interprete" in after and "Conferma di una seconda persona" in after
