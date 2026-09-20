"""Tests for pending_case_router.py: routing an incoming payload against a
case already at needs_review, and the report-text merge that keeps the
newest update on top, referencing what came before without discarding it.
"""

import time

from melampo.training.nexus_candidate_store import NexusCandidateStore
from melampo.training.pending_case_router import (
    merge_report_text,
    route_payload,
    sweep_expired_pending_cases,
)


def _pending_store(case_id="case-1", report_text="Initial findings.", status="needs_review"):
    store = NexusCandidateStore()
    store.create_candidate(
        payload={"case_context": {"case_id": case_id, "report_text": report_text}},
        case_id=case_id, learning_status=status,
    )
    return store


# --------------------------------------------------------------------------
# route_payload: the three-way decision
# --------------------------------------------------------------------------


def test_a_case_id_never_seen_before_is_a_new_case():
    store = NexusCandidateStore()
    decision = route_payload({"case_id": "unseen"}, store)
    assert decision.action == "new_case"


def test_a_payload_with_no_case_id_is_always_a_new_case():
    store = _pending_store(case_id="case-1")
    decision = route_payload({"report_text": "no case id given"}, store)
    assert decision.action == "new_case"


def test_more_findings_for_a_pending_case_merges_and_reruns():
    store = _pending_store()
    decision = route_payload({"case_id": "case-1", "report_text": "Follow-up scan."}, store)
    assert decision.action == "merge_and_rerun"
    assert decision.existing_record is not None


def test_a_confirmed_diagnosis_for_a_pending_case_routes_to_confirmation():
    store = _pending_store()
    decision = route_payload({"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis"}, store)
    assert decision.action == "confirm_and_train"


def test_confirmation_takes_priority_even_if_report_text_is_also_present():
    """A payload could carry both a confirmed diagnosis and a closing note
    -- confirmation, not merge, is the right read of that combination."""
    store = _pending_store()
    decision = route_payload(
        {"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis", "report_text": "Biopsy confirmed."}, store
    )
    assert decision.action == "confirm_and_train"


def test_a_case_id_that_exists_but_is_already_promoted_is_treated_as_new():
    """PENDING_STATUSES excludes promoted/rejected -- a closed case's old
    case_id resurfacing is not the same case still open."""
    store = _pending_store(status="promoted")
    decision = route_payload({"case_id": "case-1", "report_text": "unrelated later data"}, store)
    assert decision.action == "new_case"


# --------------------------------------------------------------------------
# merge_report_text: newest on top, previous preserved, not discarded
# --------------------------------------------------------------------------


def test_the_new_report_appears_before_the_previous_one():
    merged = merge_report_text("Old findings.", "New findings.", new_date="2026-09-20")
    assert merged.index("New findings.") < merged.index("Old findings.")


def test_the_previous_report_text_is_fully_preserved():
    merged = merge_report_text("Old findings, verbatim.", "New findings.")
    assert "Old findings, verbatim." in merged


def test_an_empty_previous_report_returns_just_the_new_text_dated():
    merged = merge_report_text("", "First ever report.", new_date="2026-09-20")
    assert merged == "[Aggiornamento del 2026-09-20] First ever report."


def test_a_chain_of_three_merges_preserves_all_three_reports():
    first = merge_report_text("", "Report A.", new_date="2026-01-01")
    second = merge_report_text(first, "Report B.", new_date="2026-06-01")
    third = merge_report_text(second, "Report C.", new_date="2026-09-20")

    assert "Report A." in third
    assert "Report B." in third
    assert "Report C." in third
    assert third.index("Report C.") < third.index("Report B.") < third.index("Report A.")


def test_route_payload_uses_the_real_merge_function_not_a_reimplementation():
    store = _pending_store(report_text="Initial findings.")
    decision = route_payload({"case_id": "case-1", "report_text": "New scan."}, store)
    assert "New scan." in decision.merged_report_text
    assert "Initial findings." in decision.merged_report_text


# --------------------------------------------------------------------------
# sweep_expired_pending_cases: the one-year retention decision
# --------------------------------------------------------------------------


def test_a_pending_case_younger_than_the_retention_window_survives():
    store = _pending_store()
    deleted = sweep_expired_pending_cases(store, retention_seconds=365 * 24 * 60 * 60)
    assert deleted == []
    assert len(store.records) == 1


def test_a_pending_case_older_than_the_retention_window_is_deleted():
    store = _pending_store()
    record = next(iter(store.records.values()))
    record.created_at = time.time() - (400 * 24 * 60 * 60)

    deleted = sweep_expired_pending_cases(store, retention_seconds=365 * 24 * 60 * 60)

    assert deleted == [record.candidate_id]
    assert len(store.records) == 0


def test_a_promoted_case_is_never_swept_regardless_of_age():
    store = _pending_store(status="promoted")
    record = next(iter(store.records.values()))
    record.created_at = time.time() - (1000 * 24 * 60 * 60)

    deleted = sweep_expired_pending_cases(store, retention_seconds=365 * 24 * 60 * 60)

    assert deleted == []
    assert len(store.records) == 1


def test_the_default_retention_is_exactly_one_year():
    from melampo.training.pending_case_router import DEFAULT_RETENTION_SECONDS

    assert DEFAULT_RETENTION_SECONDS == 365 * 24 * 60 * 60


# --------------------------------------------------------------------------
# case_id generation for new_case: found necessary while verifying end to
# end through the real pipeline -- ingestion.from_payload() requires
# payload["case_id"] to already be present, which a payload identified
# only by patient fields (no case_id yet) does not have.
# --------------------------------------------------------------------------


def test_a_payload_with_no_case_id_and_no_match_gets_one_generated():
    store = NexusCandidateStore()
    decision = route_payload({"report_text": "no id given"}, store)
    assert decision.action == "new_case"
    assert decision.case_id != ""


def test_a_payload_with_an_explicit_case_id_keeps_it_unchanged():
    store = NexusCandidateStore()
    decision = route_payload({"case_id": "given-id", "report_text": "x"}, store)
    assert decision.case_id == "given-id"


def test_two_separate_new_case_payloads_get_different_generated_ids():
    store = NexusCandidateStore()
    first = route_payload({"report_text": "a"}, store)
    second = route_payload({"report_text": "b"}, store)
    assert first.case_id != second.case_id


# --------------------------------------------------------------------------
# route_payload's patient-identifier fallback: only attempted when graph
# and password are both supplied, and only when case_id itself found
# nothing.
# --------------------------------------------------------------------------


def _store_with_patient_identifiers(case_id="case-1", status="needs_review"):
    from melampo.training.patient_matching import PatientIdentifiers

    store = NexusCandidateStore()
    identifiers = PatientIdentifiers.from_payload(
        {"patient_name": "Mario", "patient_surname": "Rossi", "case_date": "2026-09-20",
         "diagnostic_question": "Evaluate for aortic root aneurysm"},
        password="secret",
    )
    store.create_candidate(
        payload={
            "case_context": {"case_id": case_id, "report_text": "Initial findings."},
            "patient_identifiers": identifiers.as_dict(),
        },
        case_id=case_id, learning_status=status,
    )
    return store


def _real_graph():
    from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph

    return InMemoryConceptGraph.from_edges(
        [ConceptEdge("Marfan syndrome", "has_phenotype", "Aortic root aneurysm", weight=0.9)]
    )


def test_patient_identifier_fallback_finds_a_case_with_no_case_id_given():
    store = _store_with_patient_identifiers()
    decision = route_payload(
        {"patient_name": "MARIO", "patient_surname": "Rossi", "case_date": "2026-09-20",
         "diagnostic_question": "Suspected aortic root aneurysm"},
        store, graph=_real_graph(), password="secret",
    )
    assert decision.action == "merge_and_rerun"
    assert decision.case_id == "case-1"


def test_patient_identifier_fallback_is_skipped_without_graph_or_password():
    store = _store_with_patient_identifiers()
    decision = route_payload(
        {"patient_name": "Mario", "patient_surname": "Rossi", "case_date": "2026-09-20",
         "diagnostic_question": "Suspected aortic root aneurysm"},
        store,
    )
    assert decision.action == "new_case"


def test_graph_may_be_a_lazy_callable_resolved_only_when_needed():
    store = _store_with_patient_identifiers()
    calls = {"n": 0}

    def lazy_graph():
        calls["n"] += 1
        return _real_graph()

    decision = route_payload(
        {"patient_name": "Mario", "patient_surname": "Rossi", "case_date": "2026-09-20",
         "diagnostic_question": "Suspected aortic root aneurysm"},
        store, graph=lazy_graph, password="secret",
    )
    assert decision.action == "merge_and_rerun"
    assert calls["n"] == 1


def test_the_lazy_graph_callable_is_never_invoked_when_case_id_matches_directly():
    store = _pending_store()  # keyed by case_id, no patient_identifiers stored

    def unexpected_call():
        raise AssertionError("graph should not be resolved when case_id matched directly")

    decision = route_payload({"case_id": "case-1", "report_text": "more data"}, store, graph=unexpected_call, password="secret")
    assert decision.action == "merge_and_rerun"
