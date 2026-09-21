"""Tests for ConfirmationRegistry's persistence -- found necessary while
wiring training_extraction.py's periodic extraction step, which runs in a
genuinely separate process from the live service that registers
confirmations: the same class of gap already found and fixed for
NexusCandidateStore and NexusScheduler's queue.
"""

from datetime import date

from melampo.governance.confirmation_registry import (
    SOURCE_INDEPENDENT_REVIEW,
    SOURCE_UNSPECIFIED,
    Confirmation,
    ConfirmationRegistry,
)


def test_a_registry_with_no_password_or_path_is_pure_in_memory_as_before():
    registry = ConfirmationRegistry()
    registry.register(Confirmation(case_id="case-1", diagnosis="X", source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=True))
    assert len(registry.admitted) == 1


def test_an_admitted_confirmation_survives_a_new_registry_instance(tmp_path):
    path = tmp_path / "registry.jsonl"
    first = ConfirmationRegistry(password="secret", path=path)
    first.register(Confirmation(case_id="case-1", diagnosis="Sarcoidosis", source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=True))

    second = ConfirmationRegistry(password="secret", path=path)

    assert len(second.admitted) == 1
    assert second.admitted[0].case_id == "case-1"
    assert second.admitted[0].diagnosis == "Sarcoidosis"


def test_a_rejected_confirmation_survives_a_new_registry_instance(tmp_path):
    path = tmp_path / "registry.jsonl"
    first = ConfirmationRegistry(password="secret", path=path)
    first.register(Confirmation(case_id="case-1", diagnosis="X", source=SOURCE_UNSPECIFIED))

    second = ConfirmationRegistry(password="secret", path=path)

    assert len(second.rejected) == 1
    assert second.rejected[0].confirmation.case_id == "case-1"


def test_confirmed_on_date_round_trips_correctly(tmp_path):
    path = tmp_path / "registry.jsonl"
    first = ConfirmationRegistry(password="secret", path=path)
    first.register(
        Confirmation(
            case_id="case-1", diagnosis="Sarcoidosis", source=SOURCE_INDEPENDENT_REVIEW,
            reviewer_blinded_to_suggestion=True, confirmed_on=date(2026, 9, 20),
        )
    )

    second = ConfirmationRegistry(password="secret", path=path)

    assert second.admitted[0].confirmed_on == date(2026, 9, 20)


def test_duplicate_case_id_detection_works_across_separate_registry_instances(tmp_path):
    """The property that matters most: a real safety guard
    (_rejection_reason's duplicate check) must not silently stop working
    just because a second process happens to be involved."""
    path = tmp_path / "registry.jsonl"
    first = ConfirmationRegistry(password="secret", path=path)
    first.register(Confirmation(case_id="case-1", diagnosis="First", source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=True))

    second = ConfirmationRegistry(password="secret", path=path)
    admitted = second.register(
        Confirmation(case_id="case-1", diagnosis="Second, different", source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=True)
    )

    assert admitted is False
    assert len(second.rejected) == 1


def test_multiple_admitted_and_rejected_confirmations_all_survive_reload(tmp_path):
    path = tmp_path / "registry.jsonl"
    first = ConfirmationRegistry(password="secret", path=path)
    first.register(Confirmation(case_id="case-1", diagnosis="A", source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=True))
    first.register(Confirmation(case_id="case-2", diagnosis="B", source=SOURCE_UNSPECIFIED))
    first.register(Confirmation(case_id="case-3", diagnosis="C", source=SOURCE_INDEPENDENT_REVIEW, reviewer_blinded_to_suggestion=True))

    second = ConfirmationRegistry(password="secret", path=path)

    assert len(second.admitted) == 2
    assert len(second.rejected) == 1
