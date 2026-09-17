"""Stage A of the agreed validation plan: verify the quantum-cognition formalism
itself is implemented correctly, against the real mathematics (Busemeyer & Bruza,
2012), before any claim about clinical usefulness.

Every test here checks a property of the theory itself, provable
independently of any specific dataset -- not a fit to empirical data, which
is Stage B/C and not attempted here.
"""

import math

import pytest

from melampo.training.quantum_cognition_order_model import (
    Projector,
    QuantumBeliefState,
    order_effect,
    rotation_projector_pair,
    sequential_probability,
)

# --------------------------------------------------------------------------
# The state and projector constructors enforce the real mathematical
# constraints, not just accept any input
# --------------------------------------------------------------------------


def test_a_normalised_state_constructs_without_error():
    QuantumBeliefState.normalised((3.0 + 0j, 4.0 + 0j))  # norm 5, gets normalised to 1


def test_an_unnormalised_state_is_rejected_by_the_raw_constructor():
    """The raw constructor enforces the physical requirement directly --
    catching a caller who built a state by hand and forgot to normalise,
    rather than silently using an invalid quantum state."""
    with pytest.raises(ValueError, match="normalised"):
        QuantumBeliefState((1.0 + 0j, 1.0 + 0j))


def test_a_non_idempotent_matrix_is_rejected():
    with pytest.raises(ValueError, match="idempotent"):
        Projector(((0.5 + 0j, 0.3 + 0j), (0.3 + 0j, 0.5 + 0j)))


def test_a_non_hermitian_matrix_is_rejected():
    with pytest.raises(ValueError, match="Hermitian"):
        Projector(((1.0 + 0j, 1.0 + 1j), (0.0 + 0j, 0.0 + 0j)))


def test_a_valid_rotated_projector_pair_is_accepted():
    yes, no = rotation_projector_pair(math.pi / 4)
    assert yes.matrix is not None and no.matrix is not None


# --------------------------------------------------------------------------
# Total probability is conserved -- the Born rule applied to a complete set
# of outcomes must sum to exactly 1
# --------------------------------------------------------------------------


def test_complementary_projector_probabilities_sum_to_one():
    state = QuantumBeliefState.normalised((0.6 + 0j, 0.8 + 0j))
    yes, no = rotation_projector_pair(math.pi / 5)
    assert yes.probability(state) + no.probability(state) == pytest.approx(1.0, abs=1e-9)


def test_probability_is_always_between_zero_and_one():
    state = QuantumBeliefState.normalised((0.6 + 0j, 0.8 + 0j))
    for angle in (0.0, 0.3, 0.7, 1.5, 2.9):
        yes, _ = rotation_projector_pair(angle)
        p = yes.probability(state)
        assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------
# The central, falsifiable property: order effects vanish exactly when
# projectors commute, and appear when they do not. This is the property
# that distinguishes genuine quantum-cognition modelling from classical
# probability relabelled -- verified directly, not assumed.
# --------------------------------------------------------------------------


def test_compatible_questions_produce_exactly_zero_order_effect():
    """Same basis (angle 0 for both) means the two projectors commute --
    classical probability is exactly the special case where every question
    commutes with every other, and this must reduce to it exactly."""
    state = QuantumBeliefState.normalised((0.6 + 0j, 0.8 + 0j))
    a_yes, _ = rotation_projector_pair(0.0)
    b_yes, _ = rotation_projector_pair(0.0)

    assert a_yes.commutes_with(b_yes)
    assert order_effect(state, a_yes, b_yes) == pytest.approx(0.0, abs=1e-9)


def test_incompatible_questions_produce_a_genuine_nonzero_order_effect():
    state = QuantumBeliefState.normalised((0.6 + 0j, 0.8 + 0j))
    a_yes, _ = rotation_projector_pair(0.0)
    b_yes, _ = rotation_projector_pair(math.pi / 6)

    assert not a_yes.commutes_with(b_yes)
    assert abs(order_effect(state, a_yes, b_yes)) > 0.01


def test_the_order_effect_grows_with_how_incompatible_the_questions_are():
    """A qualitative sanity check on the model's own behaviour: a bigger
    rotation between the two questions' bases means more non-commutativity,
    which should mean a bigger order effect, up to a point."""
    state = QuantumBeliefState.normalised((0.6 + 0j, 0.8 + 0j))
    a_yes, _ = rotation_projector_pair(0.0)

    small_angle_effect = abs(order_effect(state, a_yes, rotation_projector_pair(math.pi / 12)[0]))
    large_angle_effect = abs(order_effect(state, a_yes, rotation_projector_pair(math.pi / 4)[0]))

    assert large_angle_effect > small_angle_effect


def test_swapping_the_order_of_the_same_two_questions_negates_the_effect():
    state = QuantumBeliefState.normalised((0.6 + 0j, 0.8 + 0j))
    a_yes, _ = rotation_projector_pair(0.0)
    b_yes, _ = rotation_projector_pair(math.pi / 6)

    assert order_effect(state, a_yes, b_yes) == pytest.approx(-order_effect(state, b_yes, a_yes), abs=1e-9)


# --------------------------------------------------------------------------
# Sequential probability itself: the Lüders collapse, verified against a
# manual step-by-step computation rather than trusted from the closed form
# --------------------------------------------------------------------------


def test_sequential_probability_matches_a_manual_step_by_step_collapse():
    state = QuantumBeliefState.normalised((0.6 + 0j, 0.8 + 0j))
    a_yes, _ = rotation_projector_pair(0.0)
    b_yes, _ = rotation_projector_pair(math.pi / 6)

    from_closed_form = sequential_probability(state, a_yes, b_yes)

    p_a = a_yes.probability(state)
    collapsed = a_yes.collapse(state)
    p_b_given_a = b_yes.probability(collapsed)
    from_manual_steps = p_a * p_b_given_a

    assert from_closed_form == pytest.approx(from_manual_steps, abs=1e-9)


def test_collapsing_a_zero_probability_outcome_raises_rather_than_returning_nonsense():
    state = QuantumBeliefState.normalised((1.0 + 0j, 0.0 + 0j))
    _, no = rotation_projector_pair(0.0)  # P(no) = 0 for this state at angle 0
    with pytest.raises(ValueError, match="zero-probability"):
        no.collapse(state)
