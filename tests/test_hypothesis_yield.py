import pytest

from melampo.governance.confirmation_registry import (
    SOURCE_HISTOPATHOLOGY,
    SOURCE_SYSTEM_ACCEPTED,
    Confirmation,
    ConfirmationRegistry,
)
from melampo.training.hypothesis_yield import (
    HypothesisFeatures,
    HypothesisOutcome,
    HypothesisYieldModel,
    outcomes_from_confirmations,
)


def _features(hops=2, support=0.8, corroboration=2, gap_count=0) -> HypothesisFeatures:
    return HypothesisFeatures(
        hops=hops, support=support, corroboration=corroboration, gap_count=gap_count
    )


def _outcomes(features: HypothesisFeatures, confirmed: int, total: int) -> list[HypothesisOutcome]:
    return [
        HypothesisOutcome(
            case_id=f"case_{index}",
            condition="x",
            features=features,
            confirmed=index < confirmed,
            confirmation_source=SOURCE_HISTOPATHOLOGY,
        )
        for index in range(total)
    ]


# --------------------------------------------------------------------------
# What is learned is a shape, not a diagnosis
# --------------------------------------------------------------------------


def test_the_bucket_describes_a_shape_a_clinician_would_recognise():
    assert _features(hops=2, support=0.8, corroboration=2).bucket() == (
        "mechanism|strong|corroborated|attested"
    )
    assert _features(hops=4, support=0.1, corroboration=1, gap_count=1).bucket() == (
        "chain|weak|single|with_gap"
    )


def test_shapes_are_read_off_a_hypothesis_without_coupling_to_its_class():
    class _Hypothesis:
        shortest_hops = 3
        support = 0.4
        corroboration = 1
        gap_count = 0

    features = HypothesisFeatures.from_hypothesis(_Hypothesis())
    assert features.hops == 3
    assert features.support == pytest.approx(0.4)


def test_a_malformed_hypothesis_falls_back_to_neutral_features():
    class _Bare:
        pass

    assert HypothesisFeatures.from_hypothesis(_Bare()).hops == 1


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------


def test_an_unobserved_shape_returns_the_full_interval_rather_than_a_guess():
    """A pattern nobody has measured is neither promoted nor suppressed."""
    estimate = HypothesisYieldModel().estimate(_features())
    assert (estimate.lower, estimate.upper) == (0.0, 1.0)
    assert estimate.observed == 0
    assert estimate.is_established is False


def test_a_productive_shape_is_learned_from_outcomes():
    model = HypothesisYieldModel()
    productive = _features(hops=2, support=0.8, corroboration=2)
    model.observe_many(_outcomes(productive, confirmed=8, total=10))

    estimate = model.estimate(productive)
    assert estimate.confirmed == 8
    assert estimate.observed == 10
    assert estimate.lower > 0.4
    assert estimate.is_established is True


def test_a_thin_bucket_stays_visibly_wide_instead_of_pretending_to_knowledge():
    model = HypothesisYieldModel()
    shape = _features()
    model.observe_many(_outcomes(shape, confirmed=2, total=2))

    estimate = model.estimate(shape)
    assert estimate.is_established is False
    assert estimate.upper - estimate.lower > 0.4


def test_the_same_rate_narrows_as_outcomes_accumulate():
    small, large = HypothesisYieldModel(), HypothesisYieldModel()
    shape = _features()
    small.observe_many(_outcomes(shape, confirmed=4, total=5))
    large.observe_many(_outcomes(shape, confirmed=40, total=50))

    narrow = large.estimate(shape)
    wide = small.estimate(shape)
    assert (narrow.upper - narrow.lower) < (wide.upper - wide.lower)


def test_shapes_are_estimated_independently():
    model = HypothesisYieldModel()
    productive = _features(hops=2, support=0.8)
    futile = _features(hops=4, support=0.1, corroboration=1)
    model.observe_many(_outcomes(productive, confirmed=9, total=10))
    model.observe_many(_outcomes(futile, confirmed=0, total=10))

    assert model.estimate(productive).lower > model.estimate(futile).upper


# --------------------------------------------------------------------------
# Ranking and suppression
# --------------------------------------------------------------------------


class _Hypothesis:
    def __init__(self, hops, support, corroboration=1, gap_count=0, label=""):
        self.shortest_hops = hops
        self.support = support
        self.corroboration = corroboration
        self.gap_count = gap_count
        self.label = label


def test_ranking_reads_the_upper_bound_so_unmeasured_shapes_are_not_suppressed():
    """A pattern nobody has measured is a reason to look, not to hide."""
    model = HypothesisYieldModel()
    futile = _features(hops=4, support=0.1, corroboration=1)
    model.observe_many(_outcomes(futile, confirmed=0, total=30))

    ranked = model.rank(
        [
            _Hypothesis(4, 0.1, 1, label="known futile"),
            _Hypothesis(2, 0.8, 2, label="never measured"),
        ]
    )
    assert ranked[0][0].label == "never measured"


def test_a_shape_is_suppressed_only_once_enough_outcomes_exist():
    model = HypothesisYieldModel()
    futile = _features(hops=4, support=0.1, corroboration=1)

    model.observe_many(_outcomes(futile, confirmed=0, total=3))
    assert model.suppressed(futile) is False, "three observations are not evidence of futility"

    model.observe_many(_outcomes(futile, confirmed=0, total=120))
    assert model.suppressed(futile) is True


def test_a_productive_shape_is_never_suppressed():
    model = HypothesisYieldModel()
    productive = _features()
    model.observe_many(_outcomes(productive, confirmed=15, total=20))
    assert model.suppressed(productive) is False


def test_the_report_shows_which_buckets_carry_weight():
    model = HypothesisYieldModel()
    model.observe_many(_outcomes(_features(hops=2, support=0.8), confirmed=6, total=10))
    model.observe_many(_outcomes(_features(hops=4, support=0.1, corroboration=1), confirmed=0, total=2))

    report = model.report()
    assert report["outcomes"] == 12
    assert report["buckets"] == 2
    assert report["established_buckets"] == 1


# --------------------------------------------------------------------------
# What trains it: only independent confirmations
# --------------------------------------------------------------------------


def _surfaced(case_id: str, condition: str) -> dict:
    return {"case_id": case_id, "condition": condition, "features": _features()}


def test_only_independently_confirmed_cases_produce_training_signal():
    """A model trained on accepted suggestions would learn from its own proposals."""
    registry = ConfirmationRegistry()
    registry.register(Confirmation("c1", "sarcoidosis", source=SOURCE_HISTOPATHOLOGY))
    registry.register(Confirmation("c2", "pneumonia", source=SOURCE_SYSTEM_ACCEPTED))

    outcomes = outcomes_from_confirmations(
        [_surfaced("c1", "sarcoidosis"), _surfaced("c2", "pneumonia")],
        registry.learning_set(),
    )
    assert [item.case_id for item in outcomes] == ["c1"]
    assert outcomes[0].confirmed is True


def test_a_hypothesis_that_did_not_match_the_confirmed_diagnosis_counts_as_unconfirmed():
    registry = ConfirmationRegistry()
    registry.register(Confirmation("c1", "sarcoidosis", source=SOURCE_HISTOPATHOLOGY))

    outcomes = outcomes_from_confirmations([_surfaced("c1", "amyloidosis")], registry.learning_set())
    assert outcomes[0].confirmed is False


def test_a_case_without_a_confirmation_teaches_nothing():
    """Absence of confirmation is not evidence the hypothesis was wrong."""
    outcomes = outcomes_from_confirmations([_surfaced("c9", "amyloidosis")], [])
    assert outcomes == []


def test_matching_ignores_case_and_spacing():
    registry = ConfirmationRegistry()
    registry.register(Confirmation("c1", "Congestive  Cardiac Failure", source=SOURCE_HISTOPATHOLOGY))
    outcomes = outcomes_from_confirmations(
        [_surfaced("c1", "congestive cardiac failure")], registry.learning_set()
    )
    assert outcomes[0].confirmed is True


def test_malformed_surfaced_entries_are_skipped():
    registry = ConfirmationRegistry()
    registry.register(Confirmation("c1", "x", source=SOURCE_HISTOPATHOLOGY))
    outcomes = outcomes_from_confirmations(
        [{"case_id": "c1", "condition": "", "features": _features()}, {"case_id": "c1"}],
        registry.learning_set(),
    )
    assert outcomes == []
