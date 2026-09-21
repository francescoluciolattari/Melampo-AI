"""Tests for training_extraction.py: extracting DPO pairs from
ConfirmedCaseStore via preference_pairs.py, then purging only the
confirmed cases extraction could genuinely use -- never a case with no
usable contrast, which stays retained rather than being deleted as if
training had consumed something it never touched.
"""

from melampo.governance.confirmation_registry import (
    SOURCE_INDEPENDENT_REVIEW,
    Confirmation,
    ConfirmationRegistry,
)
from melampo.training.confirmed_case_store import ConfirmedCaseStore
from melampo.training.training_extraction import extract_and_purge


def _confirmed_store(tmp_path, password="secret"):
    return ConfirmedCaseStore(password=password, path=tmp_path / "confirmed.jsonl")


def _admitted_confirmation(registry, case_id, diagnosis):
    registry.register(
        Confirmation(case_id=case_id, diagnosis=diagnosis, source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=True)
    )


def test_a_case_with_a_usable_alternative_produces_a_pair_and_is_purged(tmp_path):
    store = _confirmed_store(tmp_path)
    store.persist({"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis", "raised_labels": ["Pneumonia", "Sarcoidosis"]})
    registry = ConfirmationRegistry()
    _admitted_confirmation(registry, "case-1", "Sarcoidosis")

    report = extract_and_purge(store, registry)

    assert len(report.extraction.pairs) == 1
    assert report.purged_case_ids == ["case-1"]
    assert len(store) == 0


def test_a_case_with_no_alternative_produces_no_pair_and_is_not_purged(tmp_path):
    store = _confirmed_store(tmp_path)
    store.persist({"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis", "raised_labels": ["Sarcoidosis"]})
    registry = ConfirmationRegistry()
    _admitted_confirmation(registry, "case-1", "Sarcoidosis")

    report = extract_and_purge(store, registry)

    assert len(report.extraction.pairs) == 0
    assert report.purged_case_ids == []
    assert len(store) == 1


def test_a_case_never_registered_in_the_registry_is_not_purged(tmp_path):
    store = _confirmed_store(tmp_path)
    store.persist({"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis", "raised_labels": ["Pneumonia", "Sarcoidosis"]})
    registry = ConfirmationRegistry()  # nothing registered

    report = extract_and_purge(store, registry)

    assert report.purged_case_ids == []
    assert len(store) == 1


def test_a_case_where_the_confirmed_diagnosis_was_never_raised_is_not_purged(tmp_path):
    """A real miss -- worth a human seeing it, not silently deleted before
    anyone reviews it."""
    store = _confirmed_store(tmp_path)
    store.persist({"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis", "raised_labels": ["Pneumonia", "Tuberculosis"]})
    registry = ConfirmationRegistry()
    _admitted_confirmation(registry, "case-1", "Sarcoidosis")

    report = extract_and_purge(store, registry)

    assert report.extraction.cases_where_confirmed_was_not_raised == 1
    assert report.purged_case_ids == []
    assert len(store) == 1


def test_multiple_cases_are_independently_evaluated_for_purging(tmp_path):
    store = _confirmed_store(tmp_path)
    store.persist({"case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis", "raised_labels": ["Pneumonia", "Sarcoidosis"]})
    store.persist({"case_id": "case-2", "confirmed_diagnosis": "Marfan syndrome", "raised_labels": ["Marfan syndrome"]})
    registry = ConfirmationRegistry()
    _admitted_confirmation(registry, "case-1", "Sarcoidosis")
    _admitted_confirmation(registry, "case-2", "Marfan syndrome")

    report = extract_and_purge(store, registry)

    assert report.purged_case_ids == ["case-1"]
    remaining = [record["case_id"] for record in store.load()]
    assert remaining == ["case-2"]


def test_the_diagnostic_question_is_used_as_the_prompt_when_present(tmp_path):
    store = _confirmed_store(tmp_path)
    store.persist({
        "case_id": "case-1", "confirmed_diagnosis": "Sarcoidosis",
        "raised_labels": ["Pneumonia", "Sarcoidosis"],
        "case_context": {"diagnostic_question": "What explains the bilateral hilar lymphadenopathy?"},
    })
    registry = ConfirmationRegistry()
    _admitted_confirmation(registry, "case-1", "Sarcoidosis")

    report = extract_and_purge(store, registry)

    assert report.extraction.pairs[0].prompt == "What explains the bilateral hilar lymphadenopathy?"


def test_an_empty_confirmed_store_extracts_and_purges_nothing(tmp_path):
    store = _confirmed_store(tmp_path)
    registry = ConfirmationRegistry()

    report = extract_and_purge(store, registry)

    assert report.extraction.pairs == []
    assert report.purged_case_ids == []
