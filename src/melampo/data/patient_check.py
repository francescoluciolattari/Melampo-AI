"""Is this document of this patient? (27 November 2026, the owner's rule)

The router recognises what kind of document an attachment is; before anything of it enters the case, the
identifiers it writes are compared with the patient's, as a nurse checks the name on a report before filing it:

* the **fiscal code** (codice fiscale): written in the text, compared with the patient's; from it, the date of
  birth and the sex (positions 7-11 of the code, a public rule of the Agenzia delle Entrate);
* the **date of birth**, where the text labels it ("nato il", "data di nascita", "date of birth");
* the **name**, where the text labels it ("Paziente:", "Cognome e nome:", "Sig.ra"): the patient's surname or name
  must be among the words that follow;
* a **DICOM** file's own PatientName, PatientBirthDate and PatientSex, read before they are removed.

Each document gets one of:

* ``same``: an identifier matches and none contradicts;
* ``other``: an identifier contradicts and none matches -- another patient's document, put in by mistake: it is
  not read into the case and goes to a person;
* ``doubt``: one identifier matches and another contradicts: read, kept with its doubt, and shown to a person;
* ``not_stated``: the document writes no identifier (most reports reach Melampo without one): nothing to compare;
* ``not_given``: the case has no identifier of the patient to compare with.

Only the outcome and the kinds of identifier compared are kept ("fiscal_code_differs"); never an identifier.
Everything runs on this machine.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

SAME, OTHER, DOUBT, NOT_STATED, NOT_GIVEN = "same", "other", "doubt", "not_stated", "not_given"

# omocodia: a digit of the code may be replaced by one of these letters
_OMOCODE = "LMNPQRSTUV"
_CF = re.compile(r"\b[A-Z]{6}[0-9LMNPQRSTUV]{2}[ABCDEHLMPRST][0-9LMNPQRSTUV]{2}[A-Z][0-9LMNPQRSTUV]{3}[A-Z]\b")
_MONTHS = "ABCDEHLMPRST"
_DATE = r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{2,4})"
_BIRTH = re.compile(r"(?:nat[oa]\s+(?:a\s+[^\n,;]{1,40}?\s+)?il|data\s+di\s+nascita|nascita|date\s+of\s+birth|"
                    r"born(?:\s+on)?|d\.?o\.?b\.?)\s*[:.]?\s*" + _DATE, re.IGNORECASE)
# a labelled name: "Paziente: Rossi Mario", "Sig.ra Bianchi"; the words after it written with capitals
_NAMED = re.compile(r"(?i:paziente|patient|cognome\s+e\s+nome|nome\s+e\s+cognome|nominativo)\s*:\s*"
                    r"([A-ZÀ-Ý][A-Za-zÀ-ÿ']+(?:[ \t]+[A-ZÀ-Ý][A-Za-zÀ-ÿ']+){0,3})"
                    r"|(?i:sig\.(?:ra|na)?|signor[ae]?)[ \t]+([A-ZÀ-Ý][A-Za-zÀ-ÿ']+(?:[ \t]+[A-ZÀ-Ý][A-Za-zÀ-ÿ']+){0,3})")


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    return " ".join(re.sub(r"[^A-Za-z']+", " ", text).upper().split())


def _digits(code: str) -> str:
    """The code with the omocodia letters of its numeric positions back to digits."""
    chars = list(code)
    for i in (6, 7, 9, 10, 12, 13, 14):
        if chars[i] in _OMOCODE:
            chars[i] = str(_OMOCODE.index(chars[i]))
    return "".join(chars)


def birth_of_code(code: str) -> tuple[int, int, int, str] | None:
    """(day, month, two-digit year, sex) a fiscal code writes, or None."""
    code = (code or "").upper().replace(" ", "")
    if not _CF.fullmatch(code):
        return None
    code = _digits(code)
    day = int(code[9:11])
    sex = "F" if day > 40 else "M"
    return (day - 40 if day > 40 else day, _MONTHS.index(code[8]) + 1, int(code[6:8]), sex)


def _same_code(a: str, b: str) -> bool:
    return _digits(a) == _digits(b)


def _date(day: str, month: str, year: str) -> tuple[int, int, int] | None:
    try:
        d, m, y = int(day), int(month), int(year)
    except ValueError:
        return None
    if not (1 <= d <= 31 and 1 <= m <= 12):
        return None
    return d, m, y % 100


class Patient:
    """The patient's identifiers as the case gives them (the payload): fiscal code, name, surname, birth date."""

    def __init__(self, fiscal_code: str = "", name: str = "", surname: str = "", birth_date: str = ""):
        self.code = (fiscal_code or "").upper().replace(" ", "")
        self.code = self.code if _CF.fullmatch(self.code) else ""
        self.names = {w for w in _plain(f"{name} {surname}").split() if len(w) > 1}
        self.surname = {w for w in _plain(surname).split() if len(w) > 1}
        born = None
        m = re.search(_DATE, birth_date or "") or re.search(r"(\d{4})-(\d{2})-(\d{2})", birth_date or "")
        if m and len(m.group(1)) == 4:
            born = _date(m.group(3), m.group(2), m.group(1))
        elif m:
            born = _date(*m.groups())
        coded = birth_of_code(self.code) if self.code else None
        self.birth = born or (coded[:3] if coded else None)
        self.sex = coded[3] if coded else ""

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> Patient:
        p = payload or {}
        return cls(str(p.get("patient_fiscal_code") or ""), str(p.get("patient_name") or ""),
                   str(p.get("patient_surname") or ""), str(p.get("patient_birth_date") or ""))

    @property
    def given(self) -> bool:
        return bool(self.code or self.names or self.birth)


def _outcome(match: list[str], differ: list[str], name_is_certain: bool = False) -> str:
    """A name read in the text is less certain than a code or a date: alone it never makes a document
    another patient's, only a doubt for a person."""
    if differ and match:
        return DOUBT
    if differ:
        return DOUBT if differ == ["name_differs"] and not name_is_certain else OTHER
    return SAME if match else NOT_STATED


def check_text(text: str, patient: Patient) -> dict[str, Any]:
    """The identifiers the text writes, compared with the patient's."""
    if not patient.given:
        return {"check": NOT_GIVEN, "compared": []}
    match, differ = [], []
    codes = set(_CF.findall((text or "").upper()))
    if codes and patient.code:
        (match if any(_same_code(c, patient.code) for c in codes) else differ).append(
            "fiscal_code_matches" if any(_same_code(c, patient.code) for c in codes) else "fiscal_code_differs")
    if patient.birth:
        for m in _BIRTH.finditer(text or ""):
            born = _date(*m.groups()[-3:])
            if born is not None:
                (match if born == patient.birth else differ).append(
                    "birth_date_matches" if born == patient.birth else "birth_date_differs")
                break
    if patient.names:
        for m in _NAMED.finditer(text or ""):
            words = set(_plain(m.group(1) or m.group(2)).split())
            if not words:
                continue
            (match if words & patient.names else differ).append(
                "name_matches" if words & patient.names else "name_differs")
            break
    return {"check": _outcome(match, differ), "compared": sorted(set(match + differ))}


def check_dicom(data: bytes, patient: Patient) -> dict[str, Any]:
    """A DICOM file's own PatientName, PatientBirthDate and PatientSex, compared with the patient's before they are
    removed (``dicom_volume.deidentify_instance``). An anonymised field is not an identifier."""
    if not patient.given:
        return {"check": NOT_GIVEN, "compared": []}
    try:
        import io

        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True, force=True)
    except Exception:  # noqa: BLE001 -- unreadable: nothing to compare
        return {"check": NOT_STATED, "compared": []}
    match, differ = [], []
    name = {w for w in _plain(str(getattr(ds, "PatientName", "") or "").replace("^", " ")).split() if len(w) > 1}
    name -= {"ANONYMOUS", "ANONYMIZED", "ANON", "NONE", "UNKNOWN"}
    if name and patient.names:
        (match if name & patient.names else differ).append("name_matches" if name & patient.names else "name_differs")
    born = str(getattr(ds, "PatientBirthDate", "") or "")
    if len(born) == 8 and born.isdigit() and patient.birth:
        b = _date(born[6:8], born[4:6], born[:4])
        if b is not None:
            (match if b == patient.birth else differ).append(
                "birth_date_matches" if b == patient.birth else "birth_date_differs")
    sex = str(getattr(ds, "PatientSex", "") or "").upper()
    if sex in ("M", "F") and patient.sex and sex != patient.sex:
        differ.append("sex_differs")
    return {"check": _outcome(match, differ, name_is_certain=True), "compared": sorted(set(match + differ))}


# the question a person is asked of a document (``review_question``'s rule: closed answers, the system's reading
# shown only after the answer)
DOCUMENT_ANSWERS = (
    ("this_patient", "È un documento medico di questo paziente", "It is a medical document of this patient"),
    ("another_patient", "È di un altro paziente", "It is another patient's"),
    ("not_medical", "Non è un documento medico", "It is not a medical document"),
    ("not_pertinent", "È di questo paziente ma non riguarda il caso", "It is this patient's but not of this case"),
    ("cannot_say", "Non si può dire", "One cannot say"),
)
_WHY = {
    "kind_not_recognised": (("Il documento «{d}» non è stato riconosciuto come nessun tipo di documento medico. "
                             "Di che cosa si tratta?"),
                            ("The document \"{d}\" was recognised as no kind of medical document. What is it?")),
    OTHER: ("Gli identificativi scritti nel documento «{d}» non sono quelli del paziente. Di chi è?",
            "The identifiers written in the document \"{d}\" are not the patient's. Whose is it?"),
    DOUBT: ("Nel documento «{d}» un identificativo è del paziente e un altro no. Di chi è?",
            "In the document \"{d}\" one identifier is the patient's and another is not. Whose is it?"),
}


def document_question(report: str, why: str, system: dict[str, Any] | None = None) -> dict[str, Any]:
    """The question of one document for a person, complete in itself."""
    import hashlib
    import time

    it, en = _WHY[why]
    return {"id": "doc-" + hashlib.sha256(f"{report}|{why}".encode()).hexdigest()[:12], "kind": f"document:{why}",
            "report": report, "question_it": it.format(d=report), "question_en": en.format(d=report),
            "options": [{"code": c, "label_it": i, "label_en": e} for c, i, e in DOCUMENT_ANSWERS],
            "system": dict(system or {}), "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}


__all__ = ["DOUBT", "NOT_GIVEN", "NOT_STATED", "OTHER", "SAME", "Patient", "birth_of_code", "check_dicom",
           "check_text", "document_question"]
