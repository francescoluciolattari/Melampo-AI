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


# --------------------------------------------------------------------------
# Persistence: optional, event-sourced, reusing EncryptedJsonlStore -- the
# same mechanism already built for the UMLS cache and confirmed_case_store.py.
# Pure in-memory (password=None, path=None) is unchanged and remains the
# default -- every test above this point constructs NexusCandidateStore()
# with no arguments and must keep passing exactly as before.
# --------------------------------------------------------------------------


def test_a_store_with_no_password_or_path_is_pure_in_memory_as_before():
    store = NexusCandidateStore()
    record = store.create_candidate(payload={}, case_id="case-1", learning_status="candidate")
    assert record.candidate_id in store.records


def test_a_persisted_candidate_survives_a_new_store_instance(tmp_path):
    path = tmp_path / "candidates.jsonl"
    first = NexusCandidateStore(password="secret", path=path)
    record = first.create_candidate(
        payload={"case_context": {"case_id": "case-1", "report_text": "initial"}},
        case_id="case-1", learning_status="needs_review",
    )

    second = NexusCandidateStore(password="secret", path=path)

    assert second.find_by_case_id("case-1") is not None
    assert second.get(record.candidate_id).case_id == "case-1"


def test_attach_validation_is_persisted_and_visible_in_a_new_instance(tmp_path):
    path = tmp_path / "candidates.jsonl"
    first = NexusCandidateStore(password="secret", path=path)
    record = first.create_candidate(payload={}, case_id="case-1", learning_status="candidate")
    first.attach_validation(record.candidate_id, {"status": "reviewed", "allowed_for_promotion": True})

    second = NexusCandidateStore(password="secret", path=path)

    assert second.get(record.candidate_id).validation == {"status": "reviewed", "allowed_for_promotion": True}


def test_delete_is_persisted_across_instances(tmp_path):
    path = tmp_path / "candidates.jsonl"
    first = NexusCandidateStore(password="secret", path=path)
    record = first.create_candidate(payload={}, case_id="case-1", learning_status="candidate")
    first.delete(record.candidate_id)

    second = NexusCandidateStore(password="secret", path=path)

    assert second.find_by_case_id("case-1") is None
    assert record.candidate_id not in second.records


def test_only_the_latest_event_per_candidate_wins_on_reload(tmp_path):
    """Event-sourced replay, not a snapshot -- multiple mutations to the
    same candidate must fold to the last one, not accumulate duplicates."""
    path = tmp_path / "candidates.jsonl"
    first = NexusCandidateStore(password="secret", path=path)
    record = first.create_candidate(payload={}, case_id="case-1", learning_status="candidate")
    first.attach_validation(record.candidate_id, {"status": "first"})
    first.attach_validation(record.candidate_id, {"status": "second"})

    second = NexusCandidateStore(password="secret", path=path)

    assert len(second.records) == 1
    assert second.get(record.candidate_id).validation == {"status": "second"}


def test_the_wrong_password_cannot_read_a_persisted_store(tmp_path):
    from melampo.memory.encrypted_store import WrongPasswordError

    path = tmp_path / "candidates.jsonl"
    NexusCandidateStore(password="secret", path=path).create_candidate(payload={}, case_id="case-1")

    try:
        NexusCandidateStore(password="wrong-password", path=path)
        raised = False
    except WrongPasswordError:
        raised = True
    assert raised


def test_multiple_different_cases_all_survive_reload(tmp_path):
    path = tmp_path / "candidates.jsonl"
    first = NexusCandidateStore(password="secret", path=path)
    first.create_candidate(payload={}, case_id="case-1", learning_status="candidate")
    first.create_candidate(payload={}, case_id="case-2", learning_status="needs_review")

    second = NexusCandidateStore(password="secret", path=path)

    assert len(second.records) == 2
    assert second.find_by_case_id("case-1") is not None
    assert second.find_by_case_id("case-2") is not None
