"""Tests for confirmed_case_store.py: a confirmed case's record, kept
(not deleted) once training has used it, with identifying fields hashed
before the encrypted write -- correcting an earlier design step that
assumed deletion.
"""

from melampo.memory.encrypted_store import WrongPasswordError
from melampo.training.confirmed_case_store import (
    ConfirmedCaseStore,
    anonymize_identifying_fields,
)

_RECORD = {
    "case_id": "case-1",
    "case_context": {
        "report_text": "Persistent cough, fever.",
        "demographics": {"nome": "Mario", "cognome": "Rossi", "codice_fiscale": "RSSMRA80A01H501U"},
    },
    "confirmed_diagnosis": "Sarcoidosis",
}


# --------------------------------------------------------------------------
# anonymize_identifying_fields: HMAC, not a bare hash, nested fields included
# --------------------------------------------------------------------------


def test_an_identifying_field_is_replaced_with_a_hash():
    result = anonymize_identifying_fields({"nome": "Mario"}, "secret")
    assert result["nome"] != "Mario"
    assert len(result["nome"]) == 64  # a hex sha256 digest


def test_a_non_identifying_field_is_left_untouched():
    result = anonymize_identifying_fields({"case_id": "case-1", "nome": "Mario"}, "secret")
    assert result["case_id"] == "case-1"


def test_nested_identifying_fields_are_found_and_hashed():
    result = anonymize_identifying_fields({"case_context": {"demographics": {"cognome": "Rossi"}}}, "secret")
    assert result["case_context"]["demographics"]["cognome"] != "Rossi"


def test_the_same_value_and_password_always_hash_the_same():
    a = anonymize_identifying_fields({"nome": "Mario"}, "secret")
    b = anonymize_identifying_fields({"nome": "Mario"}, "secret")
    assert a["nome"] == b["nome"]


def test_the_same_value_hashes_differently_under_a_different_password():
    """The property that makes this an HMAC, not a bare hash: reversing it
    needs the same secret that protects the file it sits inside."""
    a = anonymize_identifying_fields({"nome": "Mario"}, "secret-one")
    b = anonymize_identifying_fields({"nome": "Mario"}, "secret-two")
    assert a["nome"] != b["nome"]


def test_an_empty_identifying_field_is_left_empty_not_hashed():
    result = anonymize_identifying_fields({"nome": ""}, "secret")
    assert result["nome"] == ""


def test_an_unlisted_identifying_looking_field_is_not_touched():
    """The allowlist is deliberate, not exhaustive by shape-guessing --
    only IDENTIFYING_FIELDS' exact names are caught."""
    result = anonymize_identifying_fields({"patient_notes": "Mario mentioned in passing"}, "secret")
    assert result["patient_notes"] == "Mario mentioned in passing"


# --------------------------------------------------------------------------
# ConfirmedCaseStore: persists, encrypted, with fields already anonymised
# --------------------------------------------------------------------------


def test_persist_then_load_round_trips_the_non_identifying_data(tmp_path):
    store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")
    store.persist(_RECORD)

    loaded = list(store.load())

    assert len(loaded) == 1
    assert loaded[0]["case_id"] == "case-1"
    assert loaded[0]["confirmed_diagnosis"] == "Sarcoidosis"
    assert loaded[0]["case_context"]["report_text"] == "Persistent cough, fever."


def test_identifying_fields_are_never_stored_in_the_clear(tmp_path):
    store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")
    store.persist(_RECORD)

    raw_ciphertext = (tmp_path / "confirmed.jsonl").read_bytes()

    assert b"Mario" not in raw_ciphertext
    assert b"Rossi" not in raw_ciphertext
    assert b"RSSMRA80A01H501U" not in raw_ciphertext


def test_wrong_password_cannot_read_the_store(tmp_path):
    store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")
    store.persist(_RECORD)

    wrong = ConfirmedCaseStore(password="wrong-password", path=tmp_path / "confirmed.jsonl")
    try:
        list(wrong.load())
        raised = False
    except WrongPasswordError:
        raised = True
    assert raised


def test_len_counts_persisted_records(tmp_path):
    store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")
    assert len(store) == 0
    store.persist(_RECORD)
    store.persist({**_RECORD, "case_id": "case-2"})
    assert len(store) == 2


# --------------------------------------------------------------------------
# delete(): a tombstone event, not a file rewrite -- the same principle
# NexusCandidateStore's own event log already uses.
# --------------------------------------------------------------------------


def test_delete_removes_a_record(tmp_path):
    store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")
    store.persist({"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis"})
    store.delete("case-1")
    assert len(store) == 0
    assert list(store.load()) == []


def test_delete_leaves_other_records_untouched(tmp_path):
    store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")
    store.persist({"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis"})
    store.persist({"case_id": "case-2", "confirmed_diagnosis": "Marfan syndrome"})
    store.delete("case-1")
    remaining = [record["case_id"] for record in store.load()]
    assert remaining == ["case-2"]


def test_a_deletion_persists_across_a_new_store_instance(tmp_path):
    path = tmp_path / "confirmed.jsonl"
    first = ConfirmedCaseStore(password="secret", path=path)
    first.persist({"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis"})
    first.delete("case-1")

    second = ConfirmedCaseStore(password="secret", path=path)

    assert len(second) == 0


def test_deleting_a_nonexistent_case_id_does_not_raise(tmp_path):
    store = ConfirmedCaseStore(password="secret", path=tmp_path / "confirmed.jsonl")
    store.delete("never-existed")  # must not raise
    assert len(store) == 0
