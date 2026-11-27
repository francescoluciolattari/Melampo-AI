"""The review of a case's documents, as the review mask shows it (27 November 2026, the owner's rules).

A case is a folder on this computer, under ``data/private/cases/<case>/`` (never committed):

* ``patient.json``: the patient's identifiers (fiscal code, name, surname, date of birth), read only here;
* ``documents/``: the files of the case as they were added;
* ``review.json``: what a person decided of each document, and the answers to the reading's questions;
* ``reading.json``: the reading of the case, when it has been read (``report_reading``).

The rules of a document (``patient_check`` gives its outcome):

* **another patient's** (an identifier contradicts, none matches): **discarded**, and no one can change it;
  a document a person says is another patient's is discarded the same way;
* **without identifiers**, or of the same patient: part of the patient's file; the mask asks whether to use it,
  with **yes** given;
* **in doubt** (one identifier matches, another does not): the mask asks whose it is; until answered it is used,
  with its doubt;
* **of no recognised kind** for the router and its local model: the mask asks what it is, with **no** given
  (put in by mistake until a person says otherwise).

When the case is read, the decisions go with it (``payload()["document_decisions"]``) and hold
(``case_attachments.set_aside_of``).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .case_attachments import DISCARDED, NOT_USED, USE, CaseAttachment, document_id
from .patient_check import DOUBT, NOT_GIVEN, NOT_STATED, OTHER, SAME, Patient

ROOT = Path(__file__).resolve().parents[3]
CASES = ROOT / "data" / "private" / "cases"
CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")

# the answers a person gives of a document, and the decision each makes
USE_ANSWERS = (("yes", "Sì, usalo", "Yes, use it"), ("no", "No, non usarlo", "No, do not use it"))
DECISION_OF = {"yes": USE, "no": NOT_USED, "this_patient": USE, "another_patient": DISCARDED,
               "not_medical": NOT_USED, "not_pertinent": NOT_USED}
CHECK_LABEL = {SAME: "identificativi del paziente", NOT_STATED: "nessun identificativo scritto",
               NOT_GIVEN: "identificativi del paziente non dati", DOUBT: "identificativi in contrasto",
               OTHER: "identificativi di un altro paziente"}


@dataclass
class DocumentRow:
    """One document as the mask shows it."""

    id: str
    name: str
    kinds: list[str]
    check: str
    compared: list[str]
    question: str            # "use", "whose", "what", or "" when nothing can be asked (discarded)
    options: list[tuple[str, str, str]]
    given: str               # the decision a person finds given
    decision: str            # the decision that holds: a person's, or the given one
    answer: str = ""         # a person's answer, when given
    locked: bool = False     # discarded: no one can change it
    by: str = ""
    at: str = ""
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_name(name: str) -> str:
    name = Path(name or "").name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")[:120] or "documento"


class CaseFolder:
    """A case on this computer."""

    def __init__(self, path: Path):
        self.path = Path(path)

    # -- the case --------------------------------------------------------------------------------------------

    @classmethod
    def create(cls, case_id: str, patient: dict[str, str], root: Path = CASES) -> CaseFolder:
        if not CASE_ID.fullmatch(case_id or ""):
            raise ValueError("the case id is letters, digits, '-' and '_' (at most 64)")
        folder = cls(Path(root) / case_id)
        if folder.path.exists():
            raise ValueError(f"the case {case_id!r} exists already")
        (folder.path / "documents").mkdir(parents=True)
        keys = ("patient_fiscal_code", "patient_name", "patient_surname", "patient_birth_date")
        folder._write("patient.json", {k: str(patient.get(k, "")).strip() for k in keys})
        folder._write("review.json", {"documents": {}, "answers": {}})
        return folder

    @classmethod
    def all(cls, root: Path = CASES) -> list[CaseFolder]:
        root = Path(root)
        return sorted((cls(p) for p in root.iterdir() if (p / "patient.json").exists()), key=lambda c: c.id) \
            if root.exists() else []

    @property
    def id(self) -> str:
        return self.path.name

    def _read(self, name: str, default: Any) -> Any:
        p = self.path / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def _write(self, name: str, value: Any) -> None:
        p = self.path / name
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)

    def patient_fields(self) -> dict[str, str]:
        return dict(self._read("patient.json", {}))

    def patient(self) -> Patient:
        return Patient.from_payload(self.patient_fields())

    def review(self) -> dict[str, Any]:
        review = self._read("review.json", {})
        review.setdefault("documents", {})
        review.setdefault("answers", {})
        return review

    def reading(self) -> dict[str, Any] | None:
        return self._read("reading.json", None)

    # -- the documents ---------------------------------------------------------------------------------------

    def files(self) -> list[tuple[str, str, bytes]]:
        """(id, name, bytes) of each document, in the order they were added."""
        folder = self.path / "documents"
        out = []
        for p in sorted(folder.iterdir() if folder.exists() else (), key=lambda p: (p.stat().st_mtime_ns, p.name)):
            if p.is_file():
                data = p.read_bytes()
                out.append((document_id(data), p.name, data))
        return out

    def add(self, filename: str, data: bytes) -> str:
        """A document added to the case; the same file twice is one document."""
        if not data:
            raise ValueError("the file is empty")
        sid = document_id(data)
        if any(i == sid for i, _, _ in self.files()):
            raise ValueError("this file is in the case already")
        name, n = safe_name(filename), 1
        target = self.path / "documents" / name
        while target.exists():
            n += 1
            target = target.with_name(f"{Path(name).stem}-{n}{Path(name).suffix}")
        target.write_bytes(data)
        return sid

    def checks(self, processor: Any = None, router_chat: Any = None) -> dict[str, dict[str, Any]]:
        """Each document's kind and outcome of the identity check (no identifier kept), by document id; computed
        once per file and kept in the review."""
        review = self.review()
        known = review.setdefault("checks", {})
        files = self.files()
        missing = [f for f in files if f[0] not in known]
        if missing:
            from .case_attachments import process_case_attachments
            from .document_processing import ClinicalDocumentProcessor

            bundle = process_case_attachments([CaseAttachment(name, data) for _, name, data in missing],
                                              processor or ClinicalDocumentProcessor(), router_chat=router_chat,
                                              patient=self.patient())
            for (sid, _, _), a in zip(missing, bundle.attachments, strict=True):
                routing = a.routing or {}
                known[sid] = {"kinds": list(routing.get("kinds") or ([a.modality] if a.modality else [])),
                              "check": (routing.get("patient") or {}).get("check", NOT_STATED),
                              "compared": list((routing.get("patient") or {}).get("compared", [])),
                              "kind_not_recognised": "no_kind_recognised_read_as_narrative" in
                                                     (routing.get("notes") or ()),
                              "model_asked": routing.get("model_kinds") is not None,
                              "format": a.document_format}
            self._write("review.json", review)
        return known

    def rows(self, processor: Any = None, router_chat: Any = None) -> list[DocumentRow]:
        checks = self.checks(processor, router_chat)
        decided = self.review()["documents"]
        out = []
        for sid, name, _ in self.files():
            c = checks[sid]
            d = decided.get(sid, {})
            row = DocumentRow(id=sid, name=name, kinds=c["kinds"], check=c["check"], compared=c["compared"],
                              question="use", options=list(USE_ANSWERS), given=USE, decision=USE)
            if c["check"] == OTHER or d.get("decision") == DISCARDED:
                row.question, row.options, row.given, row.decision, row.locked = "", [], DISCARDED, DISCARDED, True
                row.notes.append("identificativi di un altro paziente: scartato, non modificabile"
                                 if c["check"] == OTHER else "di un altro paziente per chi l'ha rivisto: scartato,"
                                 " non modificabile")
            else:
                from .patient_check import DOCUMENT_ANSWERS

                if c["check"] == DOUBT:
                    row.question, row.options = "whose", [o for o in DOCUMENT_ANSWERS if o[0] != "cannot_say"]
                elif c["kind_not_recognised"]:
                    row.question, row.options = "what", [o for o in DOCUMENT_ANSWERS if o[0] != "cannot_say"]
                    if c["model_asked"]:
                        row.given = row.decision = NOT_USED
                    row.notes.append("tipo di documento non riconosciuto")
                if d.get("decision") in (USE, NOT_USED):
                    row.decision = d["decision"]
            row.answer, row.by, row.at = d.get("answer", ""), d.get("by", ""), d.get("at", "")
            out.append(row)
        return out

    def decide(self, sid: str, answer: str, by: str) -> DocumentRow:
        """A person's answer of a document. A discarded document cannot be changed."""
        rows = {r.id: r for r in self.rows()}
        row = rows.get(sid)
        if row is None:
            raise ValueError("unknown document")
        if row.locked:
            raise PermissionError("the document is another patient's: it is discarded and cannot be changed")
        if answer not in {o[0] for o in row.options}:
            raise ValueError(f"the answer must be one of {', '.join(o[0] for o in row.options)}")
        if not (by or "").strip():
            raise ValueError("who answers must be said")
        review = self.review()
        review["documents"][sid] = {"answer": answer, "decision": DECISION_OF[answer], "by": by.strip(),
                                    "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._write("review.json", review)
        return {r.id: r for r in self.rows()}[sid]

    def decisions(self) -> dict[str, str]:
        """The decision that holds for each document, by document id."""
        return {r.id: r.decision for r in self.rows()}

    def payload(self, report_text: str = "") -> dict[str, Any]:
        """The case as the ingestion reads it (``ClinicalIngestionPipeline.from_payload``), with the decisions."""
        return {"case_id": self.id, **self.patient_fields(), "report_text": report_text,
                "attachments": [{"filename": name, "data": data} for _, name, data in self.files()],
                "document_decisions": self.decisions()}

    def save_reading(self, reading: dict[str, Any]) -> None:
        self._write("reading.json", reading)

    # -- the questions of the reading --------------------------------------------------------------------------

    def questions(self) -> list[dict[str, Any]]:
        """The reading's questions with closed answers (``review_question``), the documents' apart."""
        return [q for q in (self.reading() or {}).get("questions", []) if not str(q.get("kind", "")).startswith(
            "document:")]

    def doubts(self) -> list[dict[str, Any]]:
        """What the reading sends to a person without a closed question (the doubts of the cards)."""
        return [d for d in (self.reading() or {}).get("to_a_person", [])
                if not str(d.get("reason", "")).startswith(("document_", "kind_not_recognised"))]

    def answer(self, qid: str, code: str, by: str, text: str = "") -> dict[str, Any]:
        """A person's answer to a question of the reading, kept as a review record (a proposal until a second
        person confirms it, ``review_feedback``)."""
        from ..memory.review_question import review_record

        question = next((q for q in self.questions() if q.get("id") == qid), None)
        if question is None:
            raise ValueError("unknown question")
        if not (by or "").strip():
            raise ValueError("who answers must be said")
        record = review_record(question, code, by.strip(), text)
        review = self.review()
        review["answers"][qid] = {"answer": code, "text": text.strip(), "by": by.strip(),
                                  "at": time.strftime("%Y-%m-%dT%H:%M:%S"), "record": asdict(record)}
        self._write("review.json", review)
        return review["answers"][qid]


__all__ = ["CASES", "DECISION_OF", "USE_ANSWERS", "CaseFolder", "DocumentRow"]
