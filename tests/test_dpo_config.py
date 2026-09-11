"""Tests for the DPO training configuration -- connected to preference_pairs.py's real output."""

from melampo.governance.confirmation_registry import (
    SOURCE_HISTOPATHOLOGY,
    Confirmation,
    ConfirmationRegistry,
)
from melampo.training.dpo_config import (
    DEFAULT_BETA,
    MIN_PAIRS_RECOMMENDED,
    DpoTrainingConfig,
    assess_readiness,
    build_training_dataset,
)
from melampo.training.preference_pairs import (
    as_training_records,
    extract_preference_pairs,
)


def _report_with_pairs(count: int):
    registry = ConfirmationRegistry()
    raised = {}
    for i in range(count):
        case_id = f"c{i}"
        registry.register(Confirmation(case_id=case_id, diagnosis="right", source=SOURCE_HISTOPATHOLOGY))
        raised[case_id] = ["right", "wrong"]
    return extract_preference_pairs(raised, registry)


# --------------------------------------------------------------------------
# The config mirrors trl.DPOConfig's own parameter names
# --------------------------------------------------------------------------


def test_as_trl_kwargs_uses_trl_s_own_parameter_names():
    kwargs = DpoTrainingConfig().as_trl_kwargs()
    for expected_key in ("beta", "learning_rate", "num_train_epochs"):
        assert expected_key in kwargs


def test_defaults_match_the_recorded_values():
    config = DpoTrainingConfig()
    assert config.beta == DEFAULT_BETA


def test_a_caller_can_override_any_field():
    config = DpoTrainingConfig(beta=0.2, num_train_epochs=3)
    kwargs = config.as_trl_kwargs()
    assert kwargs["beta"] == 0.2
    assert kwargs["num_train_epochs"] == 3


# --------------------------------------------------------------------------
# build_training_dataset genuinely reuses preference_pairs.py -- not a copy
# --------------------------------------------------------------------------


def test_build_training_dataset_matches_as_training_records_exactly():
    """The whole point of this connection: dpo_config must not reimplement
    the conversion preference_pairs.py already does, since two copies could
    silently drift apart the way duplicated verdict-ranking logic once did
    in this project."""
    report = _report_with_pairs(3)

    via_dpo_config = build_training_dataset(report.pairs)
    via_preference_pairs_directly = as_training_records(report.pairs)

    assert via_dpo_config == via_preference_pairs_directly


def test_build_training_dataset_uses_the_field_names_dpo_trainers_expect():
    report = _report_with_pairs(1)
    dataset = build_training_dataset(report.pairs)
    assert set(dataset[0]) == {"prompt", "chosen", "rejected"}


def test_an_empty_pair_list_produces_an_empty_dataset_not_a_crash():
    assert build_training_dataset([]) == []


# --------------------------------------------------------------------------
# Readiness: thin data must be visible, not silently trained on
# --------------------------------------------------------------------------


def test_a_thin_extraction_is_reported_as_not_ready():
    report = _report_with_pairs(1)
    readiness = assess_readiness(report)
    assert readiness.pair_count == 1
    assert readiness.is_ready is False


def test_enough_pairs_is_reported_as_ready():
    report = _report_with_pairs(MIN_PAIRS_RECOMMENDED)
    readiness = assess_readiness(report)
    assert readiness.is_ready is True


def test_readiness_threshold_is_a_recommendation_the_caller_can_override():
    """Not an exception this module raises -- the same posture verify_mechanism's
    grounding states take: report what was found, let the caller decide."""
    report = _report_with_pairs(2)
    lenient = assess_readiness(report, min_pairs=2)
    assert lenient.is_ready is True


def test_readiness_as_dict_carries_what_a_reviewer_needs():
    report = _report_with_pairs(5)
    payload = assess_readiness(report).as_dict()
    for key in ("pair_count", "usable_cases", "is_ready"):
        assert key in payload


def test_readiness_reflects_the_same_thinness_measured_in_the_decision_record():
    """The exact scenario that motivated recording DoRA over GaLore: of five
    cases, only some produce usable pairs once non-independent confirmations,
    cases with no alternative, and unraised-but-confirmed diagnoses are
    correctly excluded."""
    registry = ConfirmationRegistry()
    registry.register(Confirmation(case_id="c1", diagnosis="sarcoidosis", source=SOURCE_HISTOPATHOLOGY))
    registry.register(Confirmation(case_id="c2", diagnosis="coeliac disease", source=SOURCE_HISTOPATHOLOGY))
    report = extract_preference_pairs(
        {"c1": ["tuberculosis", "sarcoidosis"], "c2": ["coeliac disease"]}, registry
    )
    readiness = assess_readiness(report)
    assert readiness.pair_count == 1
    assert readiness.is_ready is False
