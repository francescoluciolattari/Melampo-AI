"""The review mask (27 November 2026): the documents of a case and the owner's rules -- another patient's document
discarded for good, a document without identifiers used unless a person says no -- and the questions of the
reading, whose system reading is shown only after the answer."""

import importlib.util
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from melampo.data.case_attachments import DISCARDED, NOT_USED, USE
from melampo.data.case_review import CaseFolder
from melampo.data.patient_check import DOUBT, NOT_STATED, OTHER, SAME

ROOT = Path(__file__).resolve().parents[1]
ROSSI = {"patient_fiscal_code": "RSSMRA80A01H501U", "patient_name": "Mario", "patient_surname": "Rossi"}


def _pdf(text):
    from test_case_attachments import _digital_pdf

    return _digital_pdf(text)


@pytest.fixture()
def case(tmp_path):
    folder = CaseFolder.create("caso-1", ROSSI, tmp_path)
    folder.add("mio.pdf", _pdf("Ecografia. Codice fiscale RSSMRA80A01H501U. Reperto: milza di 13 cm."))
    folder.add("anonimo.pdf", _pdf("Ecografia dell'addome. Reperto: fegato di 12.9 cm."))
    folder.add("altro.pdf", _pdf("Ecografia. Codice fiscale BNCLCU75C02F205X. Reperto: milza di 9.1 cm."))
    folder.add("dubbio.pdf", _pdf("Eco. Paziente: Bianchi Luca. CF RSSMRA80A01H501U. Reperto: 11 cm."))
    folder.add("ricevuta.pdf", _pdf("Ricevuta di pagamento numero 4521, importo 40 euro."))
    return folder


def test_each_document_has_the_owners_rule(case):
    rows = {r.name: r for r in case.rows()}
    assert (rows["mio.pdf"].check, rows["mio.pdf"].question, rows["mio.pdf"].decision) == (SAME, "use", USE)
    assert (rows["anonimo.pdf"].check, rows["anonimo.pdf"].given, rows["anonimo.pdf"].decision) == \
        (NOT_STATED, USE, USE)                                  # no identifier: the patient's file, yes given
    other = rows["altro.pdf"]
    assert (other.check, other.decision, other.locked, other.question, other.options) == \
        (OTHER, DISCARDED, True, "", [])
    assert rows["dubbio.pdf"].check == DOUBT and rows["dubbio.pdf"].question == "whose"
    # of no recognised kind for the cues alone (no local model asked): what it is is asked, yes given; the local
    # model too recognising none gives no
    receipt = rows["ricevuta.pdf"]
    assert (receipt.question, receipt.given) == ("what", USE)
    assert "BNCLCU75C02F205X" not in (case.path / "review.json").read_text()   # never an identifier kept


def test_a_discarded_document_cannot_be_changed_and_a_person_decides_the_others(case):
    rows = {r.name: r for r in case.rows()}
    with pytest.raises(PermissionError):
        case.decide(rows["altro.pdf"].id, "yes", "dott. Verdi")
    assert case.decide(rows["anonimo.pdf"].id, "no", "dott. Verdi").decision == NOT_USED
    said = case.decide(rows["dubbio.pdf"].id, "another_patient", "dott. Verdi")
    assert said.decision == DISCARDED and said.locked              # a person's "another patient" holds too
    with pytest.raises(PermissionError):
        case.decide(rows["dubbio.pdf"].id, "this_patient", "dott. Verdi")
    with pytest.raises(ValueError):
        case.decide(rows["mio.pdf"].id, "yes", "")                 # who answers must be said
    assert case.decide(rows["ricevuta.pdf"].id, "not_medical", "dott. Verdi").decision == NOT_USED
    assert case.decisions() == {rows["mio.pdf"].id: USE, rows["anonimo.pdf"].id: NOT_USED,
                                rows["altro.pdf"].id: DISCARDED, rows["dubbio.pdf"].id: DISCARDED,
                                rows["ricevuta.pdf"].id: NOT_USED}


def test_the_decisions_hold_when_the_case_is_read(case):
    from melampo.data.ingestion import ClinicalIngestionPipeline

    rows = {r.name: r for r in case.rows()}
    case.decide(rows["anonimo.pdf"].id, "no", "dott. Verdi")
    prepared = ClinicalIngestionPipeline(read_reports=False).prepare_payload(case.payload("Controllo."))
    text = prepared["report_text"]
    assert "13 cm" in text and "12.9" not in text and "9.1" not in text and "11 cm" in text and "4521" in text
    by = {a["routing"]["document"]: a["set_aside"] for a in prepared["attachment_summary"]}
    assert by == {rows["mio.pdf"].id: "", rows["anonimo.pdf"].id: "not_used_by_a_person",
                  rows["altro.pdf"].id: "another_patient", rows["dubbio.pdf"].id: "",
                  rows["ricevuta.pdf"].id: ""}


def _mask():
    spec = importlib.util.spec_from_file_location("review_mask_script", ROOT / "scripts/review_mask.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_mask_shows_the_documents_and_keeps_a_discarded_one_locked(case):
    mask = _mask()
    mask.Handler.root = case.path.parent
    server = ThreadingHTTPServer(("127.0.0.1", 0), mask.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(f"{url}/case?id=caso-1").read().decode()
        assert "scartato — non modificabile" in page and "nessun identificativo scritto" in page
        assert 'value="yes" required checked' in page                 # yes given for the documents of the file
        other = next(r for r in case.rows() if r.name == "altro.pdf")
        data = urllib.parse.urlencode({"id": "caso-1", "doc": other.id, "answer": "yes", "by": "x"}).encode()
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(f"{url}/decide", data)
        assert refused.value.code == 403
        assert "caso-1" in urllib.request.urlopen(url + "/").read().decode()
    finally:
        server.shutdown()


def test_a_question_of_the_reading_shows_what_the_system_read_only_after_the_answer(case):
    from melampo.memory.review_question import Question

    q = Question(id="q1", kind="owner", report="attachment-1", language="it", sentence="Milza di 15 cm.",
                 mention="Milza", start=0, structure="spleen", structure_name="milza", question_it="Che cosa dice?",
                 options=[{"code": "measure", "label_it": "La frase dice solo una sua misura"}],
                 system={"reading": "SEGRETO-DEL-SISTEMA"}).as_dict()
    case.save_reading({"questions": [q], "to_a_person": [{"report": "note", "mention": "milza",
                                                          "reason": "doubt:part", "clause": "Milza di 15 cm."}]})
    mask = _mask()
    before = mask.case_page(case)
    assert "Che cosa dice?" in before and "SEGRETO-DEL-SISTEMA" not in before and "doubt:part" in before
    case.answer("q1", "measure", "dott. Verdi")
    after = mask.case_page(case)
    assert "SEGRETO-DEL-SISTEMA" in after and "dott. Verdi" in after
    assert case.review()["answers"]["q1"]["record"]["decided_by"] == "dott. Verdi"
