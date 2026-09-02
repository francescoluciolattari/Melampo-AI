import pytest

from melampo.memory.concept_paths import (
    STATE_DOCUMENTED,
    STATE_DOCUMENTED_EXCLUSION,
    STATE_GAP,
    STATE_UNCERTAIN_POSITIVE,
    STATE_WEAK_NEGATION,
    ConceptEdge,
    InMemoryConceptGraph,
    epistemic_state,
    find_paths,
    local_density,
)
from melampo.reasoning.discriminating_tests import (
    DiscriminatingTestSelector,
    WeightedHypothesis,
    guaranteed_information_gain,
)
from melampo.training.mechanism_enumeration import (
    MODE_HYPOTHESES,
    MODE_KNOWLEDGE_GAP,
    MechanismEnumerator,
)

# --------------------------------------------------------------------------
# The five states
# --------------------------------------------------------------------------


def test_the_five_states_fall_out_of_the_interval_without_being_named():
    assert epistemic_state(0.85, 0.92) == STATE_DOCUMENTED
    assert epistemic_state(0.40, 0.95) == STATE_UNCERTAIN_POSITIVE
    assert epistemic_state(0.00, 1.00) == STATE_GAP
    assert epistemic_state(0.00, 0.05) == STATE_DOCUMENTED_EXCLUSION
    assert epistemic_state(0.02, 0.40) == STATE_WEAK_NEGATION


def test_a_gap_and_a_documented_exclusion_are_no_longer_the_same_number():
    """The distinction a single float cannot carry."""
    gap = ConceptEdge.unknown("a", "causes", "b")
    exclusion = ConceptEdge("a", "causes", "b", weight=0.02, lower=0.0, upper=0.05)

    assert gap.is_gap and not exclusion.is_gap
    assert gap.bounds == (0.0, 1.0)
    assert exclusion.bounds == (0.0, 0.05)
    assert gap.width > exclusion.width
    assert gap.state != exclusion.state


def test_an_edge_without_bounds_stays_exact_so_existing_graphs_are_unchanged():
    edge = ConceptEdge("a", "causes", "b", weight=0.7)
    assert edge.bounds == (0.7, 0.7)
    assert edge.width == 0.0
    assert edge.is_gap is False


def test_an_unknown_edge_is_traversable_unlike_an_absent_one():
    known_only = InMemoryConceptGraph.from_edges([ConceptEdge("a", "causes", "b", 0.9)])
    with_gap = InMemoryConceptGraph.from_edges(
        [ConceptEdge("a", "causes", "b", 0.9), ConceptEdge.unknown("b", "causes", "c")]
    )
    assert find_paths(known_only, "a", "c", max_hops=3) == []
    assert find_paths(with_gap, "a", "c", max_hops=3)


# --------------------------------------------------------------------------
# Interval strength along a path
# --------------------------------------------------------------------------


def _mixed_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("finding", "caused_by", "mechanism", 0.9),
            ConceptEdge("mechanism", "caused_by", "known condition", 0.9),
            ConceptEdge.unknown("finding", "caused_by", "unmapped link"),
            ConceptEdge("unmapped link", "caused_by", "speculative condition", 0.8),
        ]
    )


def test_a_path_through_a_gap_has_a_wide_interval_and_a_zero_floor():
    graph = _mixed_graph()
    speculative = find_paths(graph, "finding", "speculative condition", max_hops=3)[0]
    documented = find_paths(graph, "finding", "known condition", max_hops=3)[0]

    assert speculative.gap_count == 1
    assert speculative.strength_lower == 0.0
    assert speculative.strength_upper == pytest.approx(0.8)
    assert speculative.epistemic_width > documented.epistemic_width

    assert documented.gap_count == 0
    assert documented.strength_lower == pytest.approx(documented.strength_upper)


def test_one_gap_per_path_is_enforceable():
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge.unknown("a", "causes", "b"),
            ConceptEdge.unknown("b", "causes", "c"),
        ]
    )
    assert find_paths(graph, "a", "c", max_hops=3, max_gap_edges=1) == []
    assert find_paths(graph, "a", "c", max_hops=3, max_gap_edges=2)


# --------------------------------------------------------------------------
# Local density
# --------------------------------------------------------------------------


def test_density_measures_the_neighbourhood_not_the_whole_graph():
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("dense finding", "causes", "x", 0.9),
            ConceptEdge("dense finding", "causes", "y", 0.8),
            ConceptEdge.unknown("sparse finding", "causes", "z"),
        ]
    )
    dense = local_density(graph, ["dense finding"])
    sparse = local_density(graph, ["sparse finding"])

    assert dense.density == 1.0
    assert sparse.density == 0.0
    assert sparse.gap_edges == 1


def test_a_concept_absent_from_the_graph_is_reported_rather_than_counted():
    graph = InMemoryConceptGraph.from_edges([ConceptEdge("a", "causes", "b", 0.9)])
    report = local_density(graph, ["a", "nowhere in the graph"])
    assert report.concepts_absent == ("nowhere in the graph",)
    assert report.concepts_present == ("a",)


# --------------------------------------------------------------------------
# Enumeration: the ceiling, not the point estimate
# --------------------------------------------------------------------------


def test_enumeration_no_longer_discards_the_uncertain_paths():
    """The correction: filtering on point support removed exactly what matters."""
    graph = _mixed_graph()
    enumerator = MechanismEnumerator(graph=graph, max_hops=3)
    labels = [
        item.condition
        for item in enumerator.enumerate(
            findings=["finding"],
            candidate_conditions=["known condition", "speculative condition"],
        )
    ]
    assert "speculative condition" in labels, "a path through a gap is a hypothesis, not noise"
    assert "known condition" in labels


def test_the_hypothesis_reports_ceiling_and_floor_separately():
    graph = _mixed_graph()
    enumerator = MechanismEnumerator(graph=graph, max_hops=3)
    hypotheses = {
        item.condition: item
        for item in enumerator.enumerate(
            findings=["finding"],
            candidate_conditions=["known condition", "speculative condition"],
        )
    }
    speculative = hypotheses["speculative condition"]
    known = hypotheses["known condition"]

    assert speculative.guaranteed == 0.0
    assert speculative.plausibility > 0.0
    assert speculative.epistemic_width > known.epistemic_width
    assert known.guaranteed == pytest.approx(known.plausibility)


# --------------------------------------------------------------------------
# Register switch on a sparse graph
# --------------------------------------------------------------------------


def _sparse_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge.unknown("finding a", "caused_by", "condition x"),
            ConceptEdge.unknown("finding a", "caused_by", "condition y"),
        ]
    )


def test_a_sparse_neighbourhood_switches_register_instead_of_falling_silent():
    enumerator = MechanismEnumerator(graph=_sparse_graph(), max_hops=2)
    outcome = enumerator.run(
        findings=["finding a"],
        candidate_conditions=["condition z", "condition w"],
    )
    assert outcome.mode == MODE_KNOWLEDGE_GAP
    assert outcome.hypotheses == []
    assert outcome.open_questions

    payload = outcome.as_dict()
    assert payload["open_questions"][0]["target"] == "graph_completion_queue"
    assert payload["open_questions"][0]["clinical_use"] is False
    assert "graph coverage" in payload["interpretation"]


def test_a_dense_neighbourhood_produces_ranked_hypotheses():
    enumerator = MechanismEnumerator(graph=_mixed_graph(), max_hops=3)
    outcome = enumerator.run(
        findings=["finding"],
        candidate_conditions=["known condition", "speculative condition"],
    )
    assert outcome.mode == MODE_HYPOTHESES
    assert outcome.open_questions == []
    assert outcome.hypotheses


def test_corroboration_by_two_findings_survives_a_sparse_neighbourhood():
    """A spurious path from one finding is easy on a sparse graph; two converging is not."""
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge.unknown("finding a", "caused_by", "noise"),
            ConceptEdge.unknown("finding b", "caused_by", "other noise"),
            ConceptEdge("finding a", "caused_by", "shared condition", 0.8),
            ConceptEdge("finding b", "caused_by", "shared condition", 0.8),
        ]
    )
    enumerator = MechanismEnumerator(graph=graph, max_hops=2, min_density=0.9)
    outcome = enumerator.run(
        findings=["finding a", "finding b"],
        candidate_conditions=["shared condition"],
    )
    assert outcome.mode == MODE_HYPOTHESES
    assert outcome.density.density < 0.9
    assert outcome.hypotheses[0].condition == "shared condition"
    assert outcome.hypotheses[0].corroboration == 2


def test_open_questions_skip_conditions_already_reached_by_a_path():
    enumerator = MechanismEnumerator(graph=_sparse_graph(), max_hops=2)
    outcome = enumerator.run(
        findings=["finding a"],
        candidate_conditions=["condition x", "condition unrelated"],
    )
    if outcome.mode == MODE_KNOWLEDGE_GAP:
        asked = {item.condition for item in outcome.open_questions}
        assert "condition x" not in asked


# --------------------------------------------------------------------------
# Discriminating tests: guaranteed gain
# --------------------------------------------------------------------------


def test_overlapping_intervals_guarantee_nothing():
    prior = {"a": 0.5, "b": 0.5}
    assert guaranteed_information_gain(prior, {"a": (0.0, 1.0), "b": (0.3, 0.4)}) == 0.0


def test_disjoint_intervals_guarantee_a_positive_gain():
    prior = {"a": 0.5, "b": 0.5}
    assert guaranteed_information_gain(prior, {"a": (0.8, 0.9), "b": (0.1, 0.2)}) > 0.0


def test_a_test_whose_advantage_is_a_missing_edge_no_longer_wins():
    """The inversion the point estimate produced, corrected.

    The point estimate reads a missing edge as a low value, so a test documented
    on one side and unmapped on the other scores higher than one documented on
    both — ignorance masquerading as diagnostic power. The guarantee reads the
    silence as the full interval, so it promises nothing.
    """
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("condition a", "indicates", "documented test", 0.9),
            ConceptEdge("condition b", "indicates", "documented test", 0.3),
            ConceptEdge("condition a", "indicates", "unmapped test", 0.9),
        ]
    )
    selector = DiscriminatingTestSelector(graph=graph)
    ranked = selector.rank([WeightedHypothesis("condition a", 0.5), WeightedHypothesis("condition b", 0.5)])
    by_name = {item.name: item for item in ranked}

    assert by_name["unmapped test"].information_gain > by_name["documented test"].information_gain
    assert by_name["unmapped test"].information_gain_lower == 0.0
    assert by_name["documented test"].information_gain_lower > 0.0
    assert ranked[0].name == "documented test"


def test_an_explicit_unknown_edge_also_guarantees_nothing():
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("condition a", "indicates", "gappy test", 0.9),
            ConceptEdge.unknown("condition b", "indicates", "gappy test"),
        ]
    )
    selector = DiscriminatingTestSelector(graph=graph)
    ranked = selector.rank([WeightedHypothesis("condition a", 0.5), WeightedHypothesis("condition b", 0.5)])
    assert ranked[0].information_gain_lower == 0.0


def test_the_point_estimate_stays_visible_alongside_the_guarantee():
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("condition a", "indicates", "t", 0.9),
            ConceptEdge("condition b", "indicates", "t", 0.2),
        ]
    )
    selector = DiscriminatingTestSelector(graph=graph)
    payload = selector.rank(
        [WeightedHypothesis("condition a", 0.5), WeightedHypothesis("condition b", 0.5)]
    )[0].as_dict()
    assert payload["information_gain_bits"] > 0
    assert payload["information_gain_lower_bits"] > 0
