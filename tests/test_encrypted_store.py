"""Tests for the encrypted local store."""

import tempfile
from pathlib import Path

import pytest

from melampo.memory.encrypted_store import EncryptedJsonlStore, WrongPasswordError


def test_data_is_genuinely_encrypted_on_disk_not_just_obfuscated():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "store.jsonl"
        store = EncryptedJsonlStore(path=path, password="a-real-secret")
        store.append({"cui": "C0009044", "term": "Burns"})

        raw = path.read_bytes()

    assert b"Burns" not in raw
    assert b"C0009044" not in raw


def test_the_correct_password_decrypts_what_was_written():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "store.jsonl"
        EncryptedJsonlStore(path=path, password="a-real-secret").append({"term": "Burns"})

        loaded = list(EncryptedJsonlStore(path=path, password="a-real-secret").load())

    assert loaded == [{"term": "Burns"}]


def test_the_wrong_password_is_rejected_explicitly():
    """A decryption failure means either a wrong password or corruption --
    both are situations the caller must be told about, not silently
    skipped."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "store.jsonl"
        EncryptedJsonlStore(path=path, password="correct-secret").append({"term": "Burns"})

        wrong = EncryptedJsonlStore(path=path, password="wrong-secret")
        with pytest.raises(WrongPasswordError):
            list(wrong.load())


def test_two_stores_with_different_passwords_produce_different_ciphertext():
    """The same plaintext through two different passwords must not
    accidentally collide or leak structure."""
    with tempfile.TemporaryDirectory() as directory:
        path_a = Path(directory) / "a.jsonl"
        path_b = Path(directory) / "b.jsonl"
        EncryptedJsonlStore(path=path_a, password="secret-one").append({"term": "same"})
        EncryptedJsonlStore(path=path_b, password="secret-two").append({"term": "same"})

        assert path_a.read_bytes() != path_b.read_bytes()


def test_a_fresh_store_creates_its_own_random_salt():
    with tempfile.TemporaryDirectory() as directory:
        store = EncryptedJsonlStore(path=Path(directory) / "store.jsonl", password="x")
        assert store.salt_path.exists()
        assert len(store.salt_path.read_bytes()) == 16


def test_reopening_a_store_reuses_the_existing_salt_not_a_new_one():
    """A new random salt on every open would make the derived key different
    each time, and nothing previously written would ever decrypt again."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "store.jsonl"
        first = EncryptedJsonlStore(path=path, password="x")
        first.append({"term": "a"})
        salt_after_first = first.salt_path.read_bytes()

        second = EncryptedJsonlStore(path=path, password="x")
        second.append({"term": "b"})

        assert second.salt_path.read_bytes() == salt_after_first
        assert len(list(second.load())) == 2


def test_append_many_writes_every_record():
    with tempfile.TemporaryDirectory() as directory:
        store = EncryptedJsonlStore(path=Path(directory) / "store.jsonl", password="x")
        written = store.append_many([{"i": 0}, {"i": 1}, {"i": 2}])

    assert written == 3


def test_append_many_with_nothing_writes_nothing_and_creates_no_file():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "store.jsonl"
        store = EncryptedJsonlStore(path=path, password="x")
        assert store.append_many([]) == 0
        assert not path.exists()


def test_loading_a_missing_file_yields_nothing_not_an_error():
    with tempfile.TemporaryDirectory() as directory:
        store = EncryptedJsonlStore(path=Path(directory) / "never_written.jsonl", password="x")
        assert list(store.load()) == []
        assert len(store) == 0


def test_len_counts_decrypted_records():
    with tempfile.TemporaryDirectory() as directory:
        store = EncryptedJsonlStore(path=Path(directory) / "store.jsonl", password="x")
        store.append_many([{"i": 0}, {"i": 1}])
        assert len(store) == 2
