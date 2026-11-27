"""Is this document of this patient? (27 November 2026): the identifiers a document writes compared with the case's,
on this machine; another patient's document is kept out of the case and goes to a person, with its question."""

import json

import pytest

from melampo.data.case_attachments import CaseAttachment, process_case_attachments
from melampo.data.document_processing import ClinicalDocumentProcessor
from melampo.data.patient_check import (
    DOUBT,
    NOT_GIVEN,
    NOT_STATED,
    OTHER,
    SAME,
    Patient,
    birth_of_code,
    check_dicom,
    check_text,
)

ROSSI = Patient("RSSMRA80A01H501U", "Mario", "Rossi")


def test_the_fiscal_code_writes_the_date_of_birth_and_the_sex():
    assert birth_of_code("RSSMRA80A01H501U") == (1, 1, 80, "M")
    assert birth_of_code("BNCLRA85M52F205Z") == (12, 8, 85, "F")          # a woman: day + 40
    assert birth_of_code("RSSMRA80A01H50MU") == (1, 1, 80, "M")           # omocodia does not touch the date
    assert birth_of_code("not a code") is None


@pytest.mark.parametrize("text, outcome, compared", [
    ("Paziente: Rossi Mario, nato il 01/01/1980. TC addome.", SAME, ["birth_date_matches", "name_matches"]),
    ("Codice fiscale RSSMRA80A01H501U. Ecografia.", SAME, ["fiscal_code_matches"]),
    ("Codice fiscale BNCLCU75C02F205X. Ecografia.", OTHER, ["fiscal_code_differs"]),
    ("Data di nascita: 02/03/1975. Emocromo.", OTHER, ["birth_date_differs"]),
    ("Paziente: Bianchi Luca. TC addome.", DOUBT, ["name_differs"]),       # a name read alone: only a doubt
    ("Paziente: Bianchi Luca, CF RSSMRA80A01H501U.", DOUBT, ["fiscal_code_matches", "name_differs"]),
    ("Paziente con dolore in fossa iliaca destra.", NOT_STATED, []),       # "con dolore" is no name
    ("TC addome: nulla di rilevante.", NOT_STATED, []),
])
def test_the_identifiers_a_text_writes_are_compared_with_the_patients(text, outcome, compared):
    assert check_text(text, ROSSI) == {"check": outcome, "compared": compared}


def test_without_the_patients_identifiers_nothing_is_compared():
    assert check_text("Codice fiscale BNCLCU75C02F205X.", Patient())["check"] == NOT_GIVEN


def test_a_dicom_files_own_patient_fields_are_compared_before_they_are_removed():
    from test_dicom_volume import _ct_series

    one = _ct_series(2)[0]
    assert check_dicom(one, ROSSI) == {"check": SAME, "compared": ["name_matches"]}
    assert check_dicom(one, Patient("BNCLCU75C02F205X", "Luca", "Bianchi"))["check"] == OTHER


def _pdf(text):
    from test_case_attachments import _digital_pdf

    return _digital_pdf(text)


def test_another_patients_document_is_kept_out_of_the_case_and_its_question_goes_to_a_person():
    from melampo.data.report_reading import ReportReaders, build_linker

    bundle = process_case_attachments(
        [CaseAttachment("a.pdf", _pdf("Codice fiscale BNCLCU75C02F205X Emoglobina 9.1 g/dL (13.5-17.5)")),
         CaseAttachment("b.pdf", _pdf("Codice fiscale RSSMRA80A01H501U Emoglobina 13.2 g/dL (13.5-17.5)"))],
        ClinicalDocumentProcessor(), patient=ROSSI)
    first, second = bundle.attachments
    assert first.set_aside == "another_patient" and second.set_aside == ""
    assert first.summary()["routing"]["patient"] == {"check": OTHER, "compared": ["fiscal_code_differs"]}
    text = bundle.combined_text("Controllo.")
    assert "9.1" not in text and "messo da parte: another_patient" in text and "13.2" in text
    assert {o.source for o in bundle.observations()} == {"attachment-2:line-1"} or all(
        o.source.startswith("attachment-2") for o in bundle.observations())
    json.dumps(first.summary())
    assert "BNCLCU75C02F205X" not in json.dumps(first.summary())        # only the outcome travels

    def chat(prompt):
        return "[]"
    chat.local_only = True
    reading = ReportReaders(build_linker()[0], interpreters={"a": chat}).read("c1", "", bundle)
    by = {t["report"]: t for t in reading["texts"]}
    assert by["attachment-1"] == {"report": "attachment-1", "kinds": by["attachment-1"]["kinds"], "read": False,
                                  "reason": "another_patient"}
    [question] = [q for q in reading["questions"] if q["report"] == "attachment-1"]
    assert question["kind"] == "document:other" and question["system"]["identifiers_compared"] == \
        ["fiscal_code_differs"]
    assert "altro paziente" in question["question_it"].lower() or "non sono quelli del paziente" in \
        question["question_it"]


def test_another_patients_series_is_never_this_patients_study():
    from test_dicom_volume import _ct_series

    items = [CaseAttachment(f"IM{i}", data) for i, data in enumerate(_ct_series(3))]
    mine = process_case_attachments(items, ClinicalDocumentProcessor(), patient=ROSSI)
    theirs = process_case_attachments(items, ClinicalDocumentProcessor(),
                                      patient=Patient("BNCLCU75C02F205X", "Luca", "Bianchi"))
    assert len(mine.imaging_studies()) == 1 and theirs.imaging_studies() == []
