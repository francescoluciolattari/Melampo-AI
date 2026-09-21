"""Where a confirmed diagnosis actually goes -- the piece confirm_and_train recognised but never executed.

Two entry points, both feeding this one function, per the design worked
through directly: a case at needs_review can be closed two ways -- a
document arrives with the case_id and confirmed diagnosis already
extracted (an EHR or lab system's own submission), or a physician opens
a list of pending cases and enters the diagnosis by hand. Both must
reach the same place, not two divergent code paths that could disagree
about what "confirmed" means or what happens to the data afterward.

**What "closing the case" does, decided directly, not assumed**:
- Builds outcome feedback (OutcomeFeedbackIngestor, already built,
  previously never called) comparing the confirmed diagnosis against
  what the system's nexus branch had proposed -- correct or not, both
  outcomes are real signal, not just the positive one.
- Persists the full record -- confirmed diagnosis, the case's own
  data, the outcome comparison -- to ConfirmedCaseStore: encrypted,
  with name/surname/national-ID-shaped fields anonymised before the
  write (confirmed_case_store.py). This is retention, not deletion --
  corrected directly after an earlier design step assumed the opposite.
- Removes the record from NexusCandidateStore, the *pending* list --
  the case is no longer awaiting review, so it no longer belongs among
  cases that are.

**What this does NOT yet do, stated rather than silently skipped**:
extracting DPO training pairs from the confirmed record
(preference_pairs.py, dpo_config.py) is separate, already-built
infrastructure this does not invoke here -- it operates on
ConfirmationRegistry's own Confirmation records, a further integration
step, not attempted in this change.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..governance.confirmation_registry import SOURCE_UNSPECIFIED, Confirmation
from .confirmed_case_store import ConfirmedCaseStore
from .nexus_candidate_store import NexusCandidateRecord, NexusCandidateStore
from .outcome_feedback import OutcomeFeedbackIngestor
from .patient_matching import find_matching_pending_records

PENDING_STATUSES = ("candidate", "needs_review")


@dataclass(frozen=True)
class ConfirmationResult:
    """What closing a case produced -- data the caller reports back, not a decision this module enforces itself."""

    case_id: str
    found_pending_record: bool
    outcome_feedback: dict[str, Any] | None
    confirmation: Confirmation | None
    closed_case_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "found_pending_record": self.found_pending_record,
            "outcome_feedback": self.outcome_feedback,
            "confirmation": self.confirmation.as_dict() if self.confirmation else None,
            "closed_case_ids": self.closed_case_ids,
        }


def _proposed_label(record: NexusCandidateRecord) -> str:
    """The system's own top proposal for this case, from whatever the nexus branch stored -- best-effort, not authoritative."""
    hypotheses = record.payload.get("case_context", {}).get("nexus", {}).get("alternative_hypotheses", [])
    if not hypotheses:
        hypotheses = record.payload.get("nexus", {}).get("alternative_hypotheses", [])
    if hypotheses and isinstance(hypotheses[0], dict):
        return str(hypotheses[0].get("label", ""))
    return ""


def _raised_labels(record: NexusCandidateRecord) -> list[str]:
    """Every alternative hypothesis label the nexus branch raised for this record, not just the top one.

    Kept alongside proposed_label (which stays the single best-effort top
    guess used for the live correct/incorrect outcome check) because
    preference_pairs.py's DPO extraction needs the *whole* raised set --
    a case where the confirmed diagnosis was raised alongside four wrong
    ones carries four contrasts, and only ever storing the top guess would
    silently throw three of them away before extraction ever got a chance
    to use them.
    """
    hypotheses = record.payload.get("case_context", {}).get("nexus", {}).get("alternative_hypotheses", [])
    if not hypotheses:
        hypotheses = record.payload.get("nexus", {}).get("alternative_hypotheses", [])
    return [str(item.get("label", "")) for item in hypotheses if isinstance(item, dict) and item.get("label")]


def _close_one_record(
    record: NexusCandidateRecord,
    diagnosis: str,
    *,
    source: str,
    candidate_store: NexusCandidateStore,
    confirmed_case_store: ConfirmedCaseStore,
    registry: Any = None,
    confirmation_source: str = SOURCE_UNSPECIFIED,
    reviewer_blinded_to_suggestion: bool | None = None,
    note: str | None = None,
) -> tuple[dict[str, Any], Confirmation]:
    """Close exactly one pending record: outcome feedback, persist (retained, anonymised), remove from pending, register the confirmation.

    Shared by every record a single submit_confirmed_diagnosis() call
    closes -- the primary case_id match and any other pending record the
    same patient-identifier match found (see that function's own
    docstring for why there can be more than one).
    """
    proposed_label = _proposed_label(record)
    raised_labels = _raised_labels(record)
    ingestor = OutcomeFeedbackIngestor(default_source=source)
    feedback_record = candidate_store.attach_outcome_feedback(
        candidate_id=record.candidate_id,
        feedback=ingestor.build_feedback(
            diagnostic_result={"case_id": record.case_id, "result_label": proposed_label},
            outcome={"accepted_labels": [diagnosis], "notes": note or ""},
        ).as_dict(),
    )

    confirmed_case_store.persist(
        {
            "case_id": record.case_id,
            "confirmed_diagnosis": diagnosis,
            "proposed_label": proposed_label,
            "raised_labels": raised_labels,
            "source": source,
            "case_context": record.payload.get("case_context", {}),
            "outcome_feedback": feedback_record.get("outcome_feedback", []),
        }
    )

    candidate_store.delete(record.candidate_id)

    confirmation = Confirmation(
        case_id=record.case_id, diagnosis=diagnosis, source=confirmation_source,
        reviewer_blinded_to_suggestion=reviewer_blinded_to_suggestion, note=note,
    )
    if registry is not None:
        registry.register(confirmation)
    return feedback_record, confirmation


def submit_confirmed_diagnosis(
    case_id: str,
    diagnosis: str,
    *,
    source: str,
    candidate_store: NexusCandidateStore,
    confirmed_case_store: ConfirmedCaseStore,
    registry: Any = None,
    confirmation_source: str = SOURCE_UNSPECIFIED,
    graph: Any = None,
    password: str | None = None,
    reviewer_blinded_to_suggestion: bool | None = None,
    note: str | None = None,
    patient_payload: dict[str, Any] | None = None,
) -> ConfirmationResult:
    """Close every pending record for this case -- the shared endpoint both confirmation paths call.

    `source` identifies which path called this (e.g. "document_recognition"
    or "physician_form") -- carried into the persisted record and the
    outcome feedback's own provenance, not used to change behaviour here;
    both paths are equally valid ways of supplying the same fact.

    `confirmation_source` is a *different* thing, deliberately kept
    separate rather than reusing `source` for both: ConfirmationRegistry's
    own `source` field is not a submission channel, it is the confirmation's
    clinical evidentiary basis (SOURCE_HISTOPATHOLOGY,
    SOURCE_CLINICAL_OUTCOME, SOURCE_INDEPENDENT_REVIEW,
    SOURCE_REFERENCE_STANDARD -- see governance/confirmation_registry.py),
    a genuinely different classification this project conflated once
    already before a test caught it: passing "physician_form" as if it
    were an evidentiary source got every confirmation correctly rejected
    by the registry (SOURCE_UNSPECIFIED is explicitly not admitted), which
    is the registry doing exactly its job, not a bug in it. Left at its
    default (SOURCE_UNSPECIFIED) here, a caller who wants registry
    admission must say plainly what the confirmation's evidentiary basis
    actually is -- never guessed from the submission channel.

    The primary record is found by case_id, unchanged. When `graph`,
    `password` and `patient_payload` are all supplied, this ALSO looks for
    other pending records belonging to the same patient via
    patient_matching.py (fiscal code, or name+surname+date+diagnostic
    question together) -- per the design agreed on directly: a patient can
    have more than one pending entry (recorded separately before a stable
    case_id linked them), and a confirmed diagnosis should close and train
    on all of them, not just the one the caller happened to reference by
    id. Every closed record is registered as its own Confirmation, since
    preference_pairs.py's extraction operates per case_id and each
    record's own raised alternatives are the contrast that record's pair
    needs.

    When `registry` (a ConfirmationRegistry) is supplied, every closure
    is also registered there -- the bridge preference_pairs.py's DPO
    extraction was missing: nothing called registry.register() from this
    path before this existed.
    """
    primary = candidate_store.find_by_case_id(case_id, statuses=PENDING_STATUSES)
    records = [primary] if primary is not None else []

    if graph is not None and password and patient_payload:
        for match in find_matching_pending_records(patient_payload, candidate_store, graph, password):
            if match.candidate_id not in {record.candidate_id for record in records}:
                records.append(match)

    if not records:
        return ConfirmationResult(case_id=case_id, found_pending_record=False, outcome_feedback=None, confirmation=None)

    closures = [
        _close_one_record(
            record, diagnosis, source=source, candidate_store=candidate_store,
            confirmed_case_store=confirmed_case_store, registry=registry, confirmation_source=confirmation_source,
            reviewer_blinded_to_suggestion=reviewer_blinded_to_suggestion, note=note,
        )
        for record in records
    ]
    primary_feedback, primary_confirmation = closures[0]
    return ConfirmationResult(
        case_id=case_id, found_pending_record=True, outcome_feedback=primary_feedback, confirmation=primary_confirmation,
        closed_case_ids=[record.case_id for record in records],
    )


def list_pending_cases(candidate_store: NexusCandidateStore) -> list[dict[str, Any]]:
    """Every case awaiting confirmation -- what a physician's selection form (the "maschera") lists from.

    Report text is included deliberately: a reviewer needs to recognise
    which case they are looking at. This is an internal, authorised
    clinical view, not the anonymised long-term record
    confirmed_case_store.py writes -- the two serve different audiences
    with different needs, not one relaxed for convenience.
    """
    return [
        {
            "case_id": record.case_id,
            "candidate_id": record.candidate_id,
            "created_at": record.created_at,
            "learning_status": record.learning_status,
            "report_text": record.payload.get("case_context", {}).get("report_text", ""),
        }
        for record in candidate_store.list_by_status(statuses=PENDING_STATUSES)
    ]


# --------------------------------------------------------------------------
# Path 1: an uploaded document, recognised automatically -- not general
# diagnosis extraction (a hard NLP problem this project has not solved),
# but an explicit, structured marker convention a submitting EHR or lab
# system would follow. Both Italian and English labels are accepted,
# matching this project's Italian-facing deployment.
# --------------------------------------------------------------------------

_CASE_ID_PATTERN = re.compile(r"(?:case[\s_-]?id|id[\s_-]?caso)\s*[:\-]\s*(\S+)", re.IGNORECASE)
_DIAGNOSIS_PATTERN = re.compile(
    r"(?:confirmed[\s_-]?diagnosis|diagnosi[\s_-]?confermata)\s*[:\-]\s*(.+?)(?:\n|$)", re.IGNORECASE
)


def extract_confirmation_from_document(text: str) -> tuple[str, str] | None:
    """(case_id, diagnosis) from an explicit marker pair in the document's text, or None if either is missing.

    Deliberately narrow: looks for the literal labels "Case ID"/"ID caso"
    and "Confirmed diagnosis"/"Diagnosi confermata" followed by a value --
    a convention a submitting system's own report template would need to
    follow, not an attempt to infer a diagnosis from unstructured
    narrative. Free-text diagnosis recognition is a real, hard NLP
    problem this project has not solved (see document_processing.py's own
    small, explicit clinical-term lexicon for the same posture) -- getting
    this wrong here would mean training on a diagnosis nobody actually
    confirmed, which is worse than requiring an explicit marker a real
    deployment's report template can be made to include.
    """
    case_id_match = _CASE_ID_PATTERN.search(text)
    diagnosis_match = _DIAGNOSIS_PATTERN.search(text)
    if not case_id_match or not diagnosis_match:
        return None
    return case_id_match.group(1).strip(), diagnosis_match.group(1).strip()


def submit_confirmation_document(
    text: str,
    *,
    candidate_store: NexusCandidateStore,
    confirmed_case_store: ConfirmedCaseStore,
) -> ConfirmationResult | None:
    """Path 1 end to end: extract, then submit -- None if the document does not carry both required markers.

    Takes already-parsed text (e.g. document_processing.py's
    ClinicalDocumentProcessor output), not a raw file -- parsing a PDF or
    image into text is that module's job, already built; this only
    recognises the confirmation markers within text once it exists.
    """
    extracted = extract_confirmation_from_document(text)
    if extracted is None:
        return None
    case_id, diagnosis = extracted
    return submit_confirmed_diagnosis(
        case_id, diagnosis, source="document_recognition",
        candidate_store=candidate_store, confirmed_case_store=confirmed_case_store,
    )
