"""Tests for extracting DPO preference pairs from existing recorded data."""

from melampo.governance.confirmation_registry import (
    SOURCE_HISTOPATHOLOGY,
    SOURCE_SYSTEM_ACCEPTED,
    Confirmation,
    ConfirmationRegistry,
)
from melampo.training.preference_pairs import (
    PreferencePair,
    as_training_records,
    extract_preference_pairs,
)


def _registry(*entries: tuple[str, str, str]) -> ConfirmationRegistry:
    registry = ConfirmationRegistry()
    for case_id, diagnosis, source in entries:
        registry.register(Confirmation(case_id=case_id, diagnosis=diagnosis, source=source))
    return registry


# --------------------------------------------------------------------------
# The claim being verified: the pairs fall out of data already recorded
# --------------------------------------------------------------------------


def test_a_confirmed_case_with_alternatives_produces_pairs():
    registry = _registry(("c1", "sarcoidosis", SOURCE_HISTOPATHOLOGY))
    report = extract_preference_pairs({"c1": ["tuberculosis", "sarcoidosis", "lymphoma"]}, registry)

    assert len(report.pairs) == 2
    assert all(pair.preferred == "sarcoidosis" for pair in report.pairs)
    assert {pair.rejected for pair in report.pairs} == {"tuberculosis", "lymphoma"}


def test_one_pair_per_rejected_alternative_not_one_per_case():
    """A case where the right answer was raised alongside four wrong ones
    carries four contrasts; collapsing to one would discard three quarters
    of the signal."""
    registry = _registry(("c1", "right", SOURCE_HISTOPATHOLOGY))
    report = extract_preference_pairs({"c1": ["right", "w1", "w2", "w3", "w4"]}, registry)
    assert len(report.pairs) == 4


# --------------------------------------------------------------------------
# What must be excluded, and why each exclusion matters
# --------------------------------------------------------------------------


def test_a_non_independent_confirmation_produces_no_pairs():
    """The automation-bias guard: a case "confirmed" because nobody objected
    to the system's own suggestion would teach the model its guesses were
    right."""
    registry = _registry(("c1", "lymphoma", SOURCE_SYSTEM_ACCEPTED))
    report = extract_preference_pairs({"c1": ["lymphoma", "sarcoidosis"]}, registry)

    assert report.pairs == []
    assert report.cases_with_no_confirmation == 1


def test_a_case_with_no_alternative_produces_no_pair():
    """DPO learns from the contrast between two responses; a pair whose
    rejected half is empty carries no signal while still looking like
    training data."""
    registry = _registry(("c1", "coeliac disease", SOURCE_HISTOPATHOLOGY))
    report = extract_preference_pairs({"c1": ["coeliac disease"]}, registry)

    assert report.pairs == []
    assert report.cases_with_no_alternative == 1


def test_a_confirmed_diagnosis_never_raised_is_counted_as_a_miss_not_a_pair():
    """A real and interesting outcome -- the system did not think of what
    turned out to be true -- and not something to turn into a pair whose
    "preferred" answer the system never produced."""
    registry = _registry(("c1", "amyloidosis", SOURCE_HISTOPATHOLOGY))
    report = extract_preference_pairs({"c1": ["sarcoidosis", "tuberculosis"]}, registry)

    assert report.pairs == []
    assert report.cases_where_confirmed_was_not_raised == 1


def test_an_unconfirmed_case_produces_no_pairs():
    report = extract_preference_pairs({"c1": ["anything"]}, ConfirmationRegistry())
    assert report.pairs == []
    assert report.cases_with_no_confirmation == 1


def test_matching_is_case_and_whitespace_insensitive():
    registry = _registry(("c1", "  SARCOIDOSIS  ", SOURCE_HISTOPATHOLOGY))
    report = extract_preference_pairs({"c1": ["sarcoidosis", "lymphoma"]}, registry)
    assert len(report.pairs) == 1


# --------------------------------------------------------------------------
# Reporting: the exclusions are visible, not silent
# --------------------------------------------------------------------------


def test_the_report_accounts_for_every_case_it_was_given():
    """Whoever assembles a training set needs to see how much of the input
    was usable -- silently returning three pairs from fifty cases would hide
    that the data is thinner than it looks."""
    registry = _registry(
        ("c1", "sarcoidosis", SOURCE_HISTOPATHOLOGY),
        ("c2", "coeliac disease", SOURCE_HISTOPATHOLOGY),
        ("c3", "lymphoma", SOURCE_SYSTEM_ACCEPTED),
        ("c4", "amyloidosis", SOURCE_HISTOPATHOLOGY),
    )
    report = extract_preference_pairs(
        {
            "c1": ["tuberculosis", "sarcoidosis"],
            "c2": ["coeliac disease"],
            "c3": ["lymphoma", "sarcoidosis"],
            "c4": ["sarcoidosis", "tuberculosis"],
            "c5": ["something"],
        },
        registry,
    )

    assert report.usable_cases == 1
    assert report.cases_with_no_alternative == 1
    assert report.cases_where_confirmed_was_not_raised == 1
    assert report.cases_with_no_confirmation == 2, "c3 (rejected by registry) and c5 (never registered)"


def test_the_confirmation_source_is_carried_into_each_pair():
    """Provenance survives into the training data, so a training set can be
    audited back to what confirmed each example."""
    registry = _registry(("c1", "sarcoidosis", SOURCE_HISTOPATHOLOGY))
    report = extract_preference_pairs({"c1": ["sarcoidosis", "lymphoma"]}, registry)
    assert report.pairs[0].confirmation_source == SOURCE_HISTOPATHOLOGY


# --------------------------------------------------------------------------
# Output format
# --------------------------------------------------------------------------


def test_training_records_use_the_field_names_dpo_trainers_expect():
    pairs = [PreferencePair("c1", "q?", "right", "wrong", SOURCE_HISTOPATHOLOGY)]
    records = as_training_records(pairs)
    assert records == [{"prompt": "q?", "chosen": "right", "rejected": "wrong"}]


def test_a_supplied_prompt_is_used_when_given():
    registry = _registry(("c1", "sarcoidosis", SOURCE_HISTOPATHOLOGY))
    report = extract_preference_pairs(
        {"c1": ["sarcoidosis", "lymphoma"]}, registry, prompt_for_case={"c1": "What explains the hypercalcaemia?"}
    )
    assert report.pairs[0].prompt == "What explains the hypercalcaemia?"


def test_a_case_without_a_prompt_still_produces_a_pair():
    """The pair's value is in the contrast; the prompt can be filled in by
    whoever assembles the file, so a missing one must not drop the case."""
    registry = _registry(("c1", "sarcoidosis", SOURCE_HISTOPATHOLOGY))
    report = extract_preference_pairs({"c1": ["sarcoidosis", "lymphoma"]}, registry)
    assert len(report.pairs) == 1
    assert report.pairs[0].prompt
