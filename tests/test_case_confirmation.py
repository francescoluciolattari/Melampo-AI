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


def test_the_confirmation_source_is_carried_through(tmp_path):
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store)

    result = submit_confirmed_diagnosis(
        "case-1", "Sarcoidosis", source="physician_form",
        candidate_store=candidate_store, confirmed_case_store=confirmed_store,
    )

    assert result.confirmation.source == "physician_form"


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


def test_document_path_records_its_source_as_document_recognition(tmp_path):
    candidate_store, confirmed_store = _stores(tmp_path)
    _pending_case(candidate_store)
    text = "Case ID: case-1\nConfirmed diagnosis: Sarcoidosis\n"

    result = submit_confirmation_document(text, candidate_store=candidate_store, confirmed_case_store=confirmed_store)

    assert result.confirmation.source == "document_recognition"
