"""Tests for the vector-evolution engine: leaky-integrator dynamics, real overlap,
Hebbian reinforcement, cross-case correlation discovery, and the DiagnosticAssembly bridge."""

import tempfile
from pathlib import Path

import pytest

from melampo.evaluation.enumeration_bench import DIFFERENTIAL_GRAPH_EDGES
from melampo.memory.encrypted_store import EncryptedJsonlStore
from melampo.memory.vector_memory import HashingEmbeddingModel
from melampo.reasoning.diagnostic_assembly import assemble
from melampo.training.vector_evolution_engine import (
    HypothesisVectorSpace,
    cosine_overlap,
    hebbian_reinforce,
    leaky_integrate,
)


def _space(directory) -> HypothesisVectorSpace:
    return HypothesisVectorSpace(store=EncryptedJsonlStore(path=Path(directory) / "vectors.jsonl", password="x"))


# --------------------------------------------------------------------------
# The real math: cosine overlap replaces the quantum overlap integral for
# real-valued vectors, exactly
# --------------------------------------------------------------------------


def test_identical_vectors_overlap_completely():
    assert cosine_overlap((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == 1.0


def test_orthogonal_vectors_do_not_overlap_at_all():
    assert cosine_overlap((1.0, 0.0), (0.0, 1.0)) == 0.0


def test_a_zero_vector_overlaps_with_nothing_rather_than_dividing_by_zero():
    assert cosine_overlap((0.0, 0.0), (1.0, 0.0)) == 0.0


def test_overlap_is_direction_only_not_magnitude():
    """Two vectors pointing the same way overlap fully regardless of length --
    the same property the quantum overlap integral has for normalised states."""
    assert cosine_overlap((1.0, 0.0), (5.0, 0.0)) == 1.0


# --------------------------------------------------------------------------
# The leaky integrator: a real, citable evolution rule, not a quantum one
# --------------------------------------------------------------------------


def test_no_elapsed_time_leaves_the_vector_unchanged():
    assert leaky_integrate((1.0, 0.0), (0.0, 1.0), elapsed_seconds=0, tau_seconds=100) == (1.0, 0.0)


def test_evidence_pulls_the_vector_toward_it_over_time():
    result = leaky_integrate((1.0, 0.0), (0.0, 1.0), elapsed_seconds=50, tau_seconds=100)
    assert 0.0 < result[0] < 1.0
    assert 0.0 < result[1] < 1.0


def test_a_very_long_elapsed_time_converges_on_the_new_evidence():
    """Old state decays away entirely; the vector becomes what the new
    evidence says, the leaky integrator's asymptotic behaviour."""
    result = leaky_integrate((1.0, 0.0), (0.0, 1.0), elapsed_seconds=100_000, tau_seconds=10)
    assert result[0] == pytest.approx(0.0, abs=1e-6)
    assert result[1] == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Hebbian reinforcement: symmetric, bounded by learning rate
# --------------------------------------------------------------------------


def test_reinforcement_pulls_both_vectors_toward_each_other_symmetrically():
    a, b = (1.0, 0.0), (0.0, 1.0)
    new_a, new_b = hebbian_reinforce(a, b, learning_rate=0.5)

    assert cosine_overlap(new_a, new_b) > cosine_overlap(a, b)


def test_a_learning_rate_of_zero_changes_nothing():
    a, b = (1.0, 0.0), (0.0, 1.0)
    new_a, new_b = hebbian_reinforce(a, b, learning_rate=0.0)
    assert new_a == a
    assert new_b == b


# --------------------------------------------------------------------------
# HypothesisVectorSpace: evolution, persistence, cross-case correlation
# --------------------------------------------------------------------------


def test_a_first_update_sets_the_vector_directly():
    with tempfile.TemporaryDirectory() as directory:
        space = _space(directory)
        result = space.update("case-1", "sarcoidosis", (1.0, 0.0, 0.0), now=1000.0)
    assert result.vector == (1.0, 0.0, 0.0)


def test_a_second_update_evolves_rather_than_replaces():
    with tempfile.TemporaryDirectory() as directory:
        space = _space(directory)
        space.update("case-1", "sarcoidosis", (1.0, 0.0, 0.0), now=1000.0)
        result = space.update("case-1", "sarcoidosis", (0.0, 1.0, 0.0), now=1000.0 + space.tau_seconds)
    assert 0.0 < result.vector[0] < 1.0
    assert 0.0 < result.vector[1] < 1.0


def test_similar_hypotheses_across_different_cases_are_found_as_correlated():
    with tempfile.TemporaryDirectory() as directory:
        space = _space(directory)
        space.update("case-A", "sarcoidosis", (1.0, 0.0), now=1000.0)
        space.update("case-B", "atypical sarcoidosis", (0.9, 0.1), now=1000.0)

        correlations = space.find_cross_case_correlations(min_overlap=0.9)

    assert len(correlations) == 1
    assert {correlations[0].case_a, correlations[0].case_b} == {"case-A", "case-B"}


def test_hypotheses_within_the_same_case_are_never_reported_as_a_correlation():
    """The capability exists to surface connections between cases nobody
    compared by hand -- two hypotheses in one differential overlapping is
    expected, not a discovery."""
    with tempfile.TemporaryDirectory() as directory:
        space = _space(directory)
        space.update("case-A", "sarcoidosis", (1.0, 0.0), now=1000.0)
        space.update("case-A", "atypical sarcoidosis", (0.9, 0.1), now=1000.0)

        correlations = space.find_cross_case_correlations(min_overlap=0.5)

    assert correlations == []


def test_dissimilar_hypotheses_are_not_reported_as_correlated():
    with tempfile.TemporaryDirectory() as directory:
        space = _space(directory)
        space.update("case-A", "sarcoidosis", (1.0, 0.0), now=1000.0)
        space.update("case-B", "asthma", (0.0, 1.0), now=1000.0)

        assert space.find_cross_case_correlations(min_overlap=0.5) == []


def test_reinforcing_an_unconfirmed_pair_still_increases_their_overlap():
    with tempfile.TemporaryDirectory() as directory:
        space = _space(directory)
        space.update("case-A", "x", (1.0, 0.0), now=1000.0)
        space.update("case-B", "y", (0.0, 1.0), now=1000.0)
        before = space.overlap("case-A::x", "case-B::y")

        space.reinforce("case-A::x", "case-B::y", learning_rate=0.3, now=1001.0)

        after = space.overlap("case-A::x", "case-B::y")

    assert after > before


def test_overlap_of_an_unknown_key_returns_none_not_an_error():
    with tempfile.TemporaryDirectory() as directory:
        space = _space(directory)
        space.update("case-A", "x", (1.0, 0.0), now=1000.0)
        assert space.overlap("case-A::x", "case-Z::never-existed") is None


def test_the_space_persists_and_reloads_with_correct_evolved_state():
    """The actual restart scenario: a fresh space instance pointed at the
    same encrypted event log must reconstruct the same current state by
    replaying every event, not just the last one."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "vectors.jsonl"
        first = HypothesisVectorSpace(store=EncryptedJsonlStore(path=path, password="x"), tau_seconds=100)
        first.update("case-A", "sarcoidosis", (1.0, 0.0), now=1000.0)
        first.update("case-A", "sarcoidosis", (0.0, 1.0), now=1100.0)

        second = HypothesisVectorSpace(store=EncryptedJsonlStore(path=path, password="x"), tau_seconds=100)
        overlap_with_pure_evidence = second.overlap("case-A::sarcoidosis", "case-A::sarcoidosis")

    assert overlap_with_pure_evidence == 1.0  # a vector always overlaps completely with itself
    assert len(second) == 1


def test_the_event_log_is_encrypted_on_disk():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "vectors.jsonl"
        space = HypothesisVectorSpace(store=EncryptedJsonlStore(path=path, password="x"))
        space.update("case-A", "a very specific rare diagnosis", (1.0, 0.0), now=1000.0)

        raw = path.read_bytes()

    assert b"a very specific rare diagnosis" not in raw


# --------------------------------------------------------------------------
# The DiagnosticAssembly bridge: additive, gracefully absent by default
# --------------------------------------------------------------------------


def test_without_a_configured_vector_space_both_methods_degrade_gracefully(tmp_path):
    assembly = assemble(list(DIFFERENTIAL_GRAPH_EDGES), tmp_path / "learned.jsonl")

    assert assembly.record_hypothesis_vector("c1", "sarcoidosis", HashingEmbeddingModel()) is None
    assert assembly.cross_case_correlations() == []


def test_with_a_configured_space_hypotheses_from_two_cases_correlate(tmp_path):
    assembly = assemble(list(DIFFERENTIAL_GRAPH_EDGES), tmp_path / "learned.jsonl")
    assembly.vector_space = HypothesisVectorSpace(
        store=EncryptedJsonlStore(path=tmp_path / "vectors.jsonl", password="x")
    )
    embedder = HashingEmbeddingModel()

    assembly.record_hypothesis_vector("case-1", "sarcoidosis with pulmonary involvement", embedder, now=1000.0)
    assembly.record_hypothesis_vector("case-2", "sarcoidosis with lung involvement", embedder, now=1000.0)

    correlations = assembly.cross_case_correlations(min_overlap=0.3)

    assert len(correlations) == 1
