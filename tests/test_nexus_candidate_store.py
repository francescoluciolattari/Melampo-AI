"""Tests for find_by_case_id() and delete(), added to NexusCandidateStore
to support pending_case_router.py's lookup-by-case_id and the confirm/expire
deletion paths -- the store previously only supported lookup by
candidate_id, a hash that changes on every create_candidate() call.
"""

from melampo.training.nexus_candidate_store import NexusCandidateStore


def test_find_by_case_id_returns_none_when_no_record_exists():
    store = NexusCandidateStore()
    assert store.find_by_case_id("nonexistent") is None


def test_find_by_case_id_finds_a_matching_record():
    store = NexusCandidateStore()
    store.create_candidate(payload={}, case_id="case-1", learning_status="needs_review")
    found = store.find_by_case_id("case-1")
    assert found is not None
    assert found.case_id == "case-1"


def test_find_by_case_id_respects_the_statuses_filter():
    store = NexusCandidateStore()
    store.create_candidate(payload={}, case_id="case-1", learning_status="promoted")
    assert store.find_by_case_id("case-1", statuses=["needs_review"]) is None
    assert store.find_by_case_id("case-1", statuses=["promoted"]) is not None


def test_find_by_case_id_returns_the_most_recent_when_several_exist():
    store = NexusCandidateStore()
    older = store.create_candidate(payload={}, case_id="case-1", learning_status="needs_review")
    older.created_at = 100.0
    newer = store.create_candidate(payload={}, case_id="case-1", learning_status="needs_review")
    newer.created_at = 200.0

    found = store.find_by_case_id("case-1")

    assert found.candidate_id == newer.candidate_id


def test_delete_removes_the_record():
    store = NexusCandidateStore()
    record = store.create_candidate(payload={}, case_id="case-1", learning_status="needs_review")
    store.delete(record.candidate_id)
    assert record.candidate_id not in store.records
    assert store.find_by_case_id("case-1") is None


def test_delete_of_a_nonexistent_id_does_not_raise():
    store = NexusCandidateStore()
    store.delete("nonexistent-id")  # must not raise


def test_delete_logs_the_event_in_the_audit_log():
    store = NexusCandidateStore()
    record = store.create_candidate(payload={}, case_id="case-1", learning_status="needs_review")
    store.delete(record.candidate_id)
    assert any(event["event"] == "candidate_deleted" for event in store.audit_log)
