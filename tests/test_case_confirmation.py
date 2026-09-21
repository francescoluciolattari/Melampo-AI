"""Tests for case_confirmation.py: the two confirmation paths agreed on --
a document with explicit markers, recognised automatically, and a
physician's form, both feeding the same submit_confirmed_diagnosis()
endpoint. Corrects an earlier design step: data is retained (encrypted,
anonymised), not deleted, once a case is closed.
"""

from melampo.training.case_confirmation import (
    extract_confirmation_from_document,
    list_pending_cases,
    submit_confirmation_document,
    submit_confirmed_diagnosis,
)
from melampo.training.confirmed_case_store import ConfirmedCaseStore
from melampo.training.nexus_candidate_store import NexusCandidateStore


def _pending_case(store, case_id="case-1", report_text="Initial findings.", proposed_label=""):
    payload = {"case_context": {"case_id": case_id, "report_text": report_text}}
    if proposed_label:
        payload["case_context"]["nexus"] = {"alternative_hypotheses": [{"label": proposed_label}]}
    return store.create_candidate(payload=payload, case_id=case_id, learning_status="needs_review")


def _stores(tmp_path):
    return NexusCandidateStore(), ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")


# --------------------------------------------------------------------------
# submit_confirmed_diagnosis: the shared endpoint both paths call
# --------------------------------------------------------------------------


def test_confirming_a_nonexistent_case_reports_not_found(tmp_path):
    candidate_store, confirmed_store = _stores(tmp_path)
    result = submit_confirmed_diagnosis(
        "no-such-case", "Sarcoidosis", source="test",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )
    assert result.found_pending_record is False


def test_confirming_a_pending_case_removes_it_from_the_pending_list(tmp_path):
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store)

    submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="test",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )

    assert candidate_store.find_by_case_id("case-1") is None


def test_confirming_a_pending_case_persists_it_to_the_confirmed_store(tmp_path):
    """The correction: retained, not deleted."""
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store, report_text="Persistent cough.")

    submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="test",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )

    assert len(confirmed_store) == 1
    stored = next(iter(confirmed_store.load()))
    assert stored["confirmed_diagnosis"] == "Sarcoidosis"
    assert stored["case_context"]["report_text"] == "Persistent cough."


def test_a_correct_proposal_is_recorded_as_such():
    """Both outcomes are real signal -- a system that guessed right is
    tracked, not just the negative case."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        candidate_store, confirmed_store = _stores(Path(d))
        _pending_case(candidate_store, proposed_label="Sarcoidosis")

        result = submit_confirmed_diagnosis(
            "case-1", "Sarcoidosis", source="test",
            candidate_store=candidate_store, confirmed_case_store=confirmed_store,
        )

    feedback = result.outcome_feedback["outcome_feedback"][-1]
    assert feedback["correct"] is True


def test_an_incorrect_proposal_is_recorded_as_such(tmp_path):
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store, proposed_label="Pneumonia")

    result = submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="test",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )

    feedback = result.outcome_feedback["outcome_feedback"][-1]
    assert feedback["correct"] is False


def test_the_submission_channel_is_carried_through_to_the_persisted_record(tmp_path):
    """confirmation.source is now the clinical evidentiary basis, a
    separate concern (see submit_confirmed_diagnosis's own docstring) --
    the submission channel ("physician_form"/"document_recognition") is
    what the persisted ConfirmedCaseStore record's own `source` field
    carries."""
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store)

    submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="physician_form",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )

    stored = next(iter(confirmed_store.load()))
    assert stored["source"] == "physician_form"


# --------------------------------------------------------------------------
# list_pending_cases: what the physician's form ("maschera") lists from
# --------------------------------------------------------------------------


def test_list_pending_cases_is_empty_with_nothing_pending(tmp_path):
    candidate_store, _ = _stores(tmp_path)
    assert list_pending_cases(candidate_store) == []


def test_list_pending_cases_includes_report_text_for_recognition(tmp_path):
    candidate_store, _ = _stores(tmp_path)
    _pending_case(candidate_store, report_text="Distinctive symptom description.")

    listed = list_pending_cases(candidate_store)

    assert listed[0]["case_id"] == "case-1"
    assert listed[0]["report_text"] == "Distinctive symptom description."


def test_list_pending_cases_excludes_already_closed_cases(tmp_path):
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store, case_id="case-1")
    _pending_case(candidate_store, case_id="case-2")
    submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="test",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )

    listed = list_pending_cases(candidate_store)

    assert len(listed) == 1
    assert listed[0]["case_id"] == "case-2"


# --------------------------------------------------------------------------
# extract_confirmation_from_document / submit_confirmation_document: Path 1
# --------------------------------------------------------------------------


def test_extracts_case_id_and_diagnosis_from_explicit_italian_markers():
    text = "Referto di conferma\nID caso: case-1\nDiagnosi confermata: Sindrome di Marfan\n"
    assert extract_confirmation_from_document(text) == ("case-1", "Sindrome di Marfan")


def test_extracts_from_explicit_english_markers_too():
    text = "Case ID: case-1\nConfirmed diagnosis: Marfan syndrome\n"
    assert extract_confirmation_from_document(text) == ("case-1", "Marfan syndrome")


def test_a_document_missing_either_marker_extracts_nothing():
    assert extract_confirmation_from_document("Case ID: case-1\nNo diagnosis marker here.") is None
    assert extract_confirmation_from_document("Confirmed diagnosis: Sarcoidosis\nNo case marker here.") is None


def test_unstructured_narrative_extracts_nothing():
    """The deliberate limitation: this is not general diagnosis NLP."""
    text = "The patient was ultimately diagnosed with sarcoidosis after biopsy, referring to case 1."
    assert extract_confirmation_from_document(text) is None


def test_submit_confirmation_document_closes_the_case_when_markers_are_present(tmp_path):
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store)
    text = "Case ID: case-1\nConfirmed diagnosis: Sarcoidosis\n"

    result = submit_confirmation_document(text, candidate_store=candidate_store, confirmed_case_store=confirmed_store)

    assert result.found_pending_record is True
    assert candidate_store.find_by_case_id("case-1") is None


def test_submit_confirmation_document_returns_none_without_markers(tmp_path):
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store)

    result = submit_confirmation_document(
        "No structured markers here.", candidate_store=candidate_store, confirmed_case_store=confirmed_store
    )

    assert result is None
    assert candidate_store.find_by_case_id("case-1") is not None  # untouched


def test_document_path_records_document_recognition_as_the_persisted_channel(tmp_path):
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store)
    text = "Case ID: case-1\nConfirmed diagnosis: Sarcoidosis\n"

    submit_confirmation_document(text, candidate_store=candidate_store, confirmed_case_store=confirmed_store)

    stored = next(iter(confirmed_store.load()))
    assert stored["source"] == "document_recognition"


# --------------------------------------------------------------------------
# Multi-match closure: when graph/password/patient_payload are supplied,
# every pending record matching the same patient closes together, not
# just the one the caller referenced by case_id.
# --------------------------------------------------------------------------


def _identifiers_dict(password="secret"):
    from melampo.training.patient_matching import PatientIdentifiers

    return PatientIdentifiers.from_payload(
        {"patient_name": "Mario", "patient_surname": "Rossi", "case_date": "2026-09-20",
         "diagnostic_question": "Evaluate for aortic root aneurysm"},
        password,
    ).as_dict()


def _graph():
    from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph

    return InMemoryConceptGraph.from_edges(
        [ConceptEdge("Marfan syndrome", "has_phenotype", "Aortic root aneurysm", weight=0.9)]
    )


def _patient_payload():
    return {"patient_name": "Mario", "patient_surname": "Rossi", "case_date": "2026-09-20",
            "diagnostic_question": "Evaluate for aortic root aneurysm"}


def test_two_pending_records_for_the_same_patient_close_together(tmp_path):
    candidate_store = NexusCandidateStore()
    identifiers = _identifiers_dict()
    candidate_store.create_candidate(
        payload={"case_context": {"case_id": "visit-1", "report_text": "First visit"}, "patient_identifiers": identifiers},
        case_id="visit-1", learning_status="needs_review",
    )
    candidate_store.create_candidate(
        payload={"case_context": {"case_id": "visit-2", "report_text": "Follow-up"}, "patient_identifiers": identifiers},
        case_id="visit-2", learning_status="needs_review",
    )
    confirmed_store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")

    result = submit_confirmed_diagnosis(
        "visit-1", "Marfan syndrome", source="physician_form",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
        graph=_graph(), password="secret", patient_payload=_patient_payload(),
    )

    assert sorted(result.closed_case_ids) == ["visit-1", "visit-2"]
    assert candidate_store.find_by_case_id("visit-1") is None
    assert candidate_store.find_by_case_id("visit-2") is None
    assert len(confirmed_store) == 2


def test_without_graph_password_or_patient_payload_only_the_primary_case_closes(tmp_path):
    """Backward compatible: the multi-match fallback is opt-in."""
    candidate_store = NexusCandidateStore()
    identifiers = _identifiers_dict()
    candidate_store.create_candidate(
        payload={"case_context": {"case_id": "visit-1", "report_text": "First visit"}, "patient_identifiers": identifiers},
        case_id="visit-1", learning_status="needs_review",
    )
    candidate_store.create_candidate(
        payload={"case_context": {"case_id": "visit-2", "report_text": "Follow-up"}, "patient_identifiers": identifiers},
        case_id="visit-2", learning_status="needs_review",
    )
    confirmed_store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")

    result = submit_confirmed_diagnosis(
        "visit-1", "Marfan syndrome", source="physician_form",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )

    assert result.closed_case_ids == ["visit-1"]
    assert candidate_store.find_by_case_id("visit-2") is not None


def test_an_unrelated_patients_pending_record_is_never_swept_in(tmp_path):
    candidate_store = NexusCandidateStore()
    candidate_store.create_candidate(
        payload={"case_context": {"case_id": "visit-1", "report_text": "First visit"}, "patient_identifiers": _identifiers_dict()},
        case_id="visit-1", learning_status="needs_review",
    )
    from melampo.training.patient_matching import PatientIdentifiers

    other_identifiers = PatientIdentifiers.from_payload(
        {"patient_name": "Luigi", "patient_surname": "Verdi", "case_date": "2026-09-20", "diagnostic_question": "unrelated"},
        "secret",
    ).as_dict()
    candidate_store.create_candidate(
        payload={"case_context": {"case_id": "other-patient", "report_text": "Unrelated"}, "patient_identifiers": other_identifiers},
        case_id="other-patient", learning_status="needs_review",
    )
    confirmed_store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")

    result = submit_confirmed_diagnosis(
        "visit-1", "Marfan syndrome", source="physician_form",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
        graph=_graph(), password="secret", patient_payload=_patient_payload(),
    )

    assert result.closed_case_ids == ["visit-1"]
    assert candidate_store.find_by_case_id("other-patient") is not None


def test_registering_with_a_confirmation_registry_admits_the_confirmation(tmp_path):
    from melampo.governance.confirmation_registry import (
        SOURCE_INDEPENDENT_REVIEW,
        ConfirmationRegistry,
    )

    candidate_store = NexusCandidateStore()
    candidate_store.create_candidate(payload={"case_context": {"case_id": "case-1"}}, case_id="case-1", learning_status="needs_review")
    confirmed_store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")
    registry = ConfirmationRegistry()

    submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="physician_form",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
        registry=registry, confirmation_source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=True,
    )

    assert len(registry.admitted) == 1
    assert registry.admitted[0].case_id == "case-1"


def test_confirmation_source_left_at_default_is_correctly_rejected_by_the_registry(tmp_path):
    """Not a bug to work around: ConfirmationRegistry's own automation-bias
    guard correctly refuses an unspecified evidentiary source -- the exact
    failure this project's own earlier conflation of "channel" and
    "evidentiary source" produced, caught by this test."""
    from melampo.governance.confirmation_registry import ConfirmationRegistry

    candidate_store = NexusCandidateStore()
    candidate_store.create_candidate(payload={"case_context": {"case_id": "case-1"}}, case_id="case-1", learning_status="needs_review")
    confirmed_store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")
    registry = ConfirmationRegistry()

    submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="physician_form",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store, registry=registry,
    )

    assert len(registry.admitted) == 0
    assert len(registry.rejected) == 1


def test_without_a_registry_nothing_is_registered_and_nothing_raises(tmp_path):
    candidate_store = NexusCandidateStore()
    candidate_store.create_candidate(payload={"case_context": {"case_id": "case-1"}}, case_id="case-1", learning_status="needs_review")
    confirmed_store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")

    result = submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="physician_form",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )

    assert result.found_pending_record is True


def test_raised_labels_are_persisted_alongside_the_top_proposed_label(tmp_path):
    candidate_store = NexusCandidateStore()
    candidate_store.create_candidate(
        payload={"case_context": {
            "case_id": "case-1",
            "nexus": {"alternative_hypotheses": [{"label": "Pneumonia"}, {"label": "Sarcoidosis"}, {"label": "Tuberculosis"}]},
        }},
        case_id="case-1", learning_status="needs_review",
    )
    confirmed_store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")

    submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="physician_form",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )

    stored = next(iter(confirmed_store.load()))
    assert stored["raised_labels"] == ["Pneumonia", "Sarcoidosis", "Tuberculosis"]
