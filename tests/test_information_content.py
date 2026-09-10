"""Tests for Information Content weighting and converging-path rewards."""

import itertools
import math

import pytest

from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph, find_paths
from melampo.memory.information_content import (
    BASIS_FREQUENCY,
    BASIS_GRAPH_STRUCTURE,
    BASIS_UNKNOWN,
    CONVERGENCE_REWARD_BASE,
    DEFAULT_IC,
    InformationContentTable,
    path_concepts,
    rank_paths,
    score_convergence,
    score_path,
)


def _marfan_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("marfan syndrome", "causes", "connective tissue weakness", 0.9),
            ConceptEdge("connective tissue weakness", "causes", "aortic root dilation", 0.85),
            ConceptEdge("marfan syndrome", "associated_with", "aortic root dilation", 0.7),
        ]
    )


# --------------------------------------------------------------------------
# Information Content: the formal version of an empirical observation
# --------------------------------------------------------------------------


def test_a_common_concept_scores_far_lower_than_a_rare_one():
    """The formal answer to what character comparison could not see: earlier
    in this project, "pulmonary embolism" vs "pulmonary oedema" outscored a
    correct paraphrase because they share "pulmonary". IC is the measure that
    says "pulmonary" carries almost no information."""
    table = InformationContentTable.from_frequencies(
        {"pulmonary": 5000, "aortic root dilation": 300, "marfan syndrome": 40}
    )
    assert table.value("pulmonary") < table.value("aortic root dilation")
    assert table.value("aortic root dilation") < table.value("marfan syndrome")
    assert table.value("pulmonary") < 0.1, "a ubiquitous term must carry almost no information"


def test_information_content_is_normalised_to_the_unit_range():
    """Raw -log(p) is unbounded; a path score multiplying unbounded factors
    would be neither comparable across graphs nor readable."""
    table = InformationContentTable.from_frequencies({"a": 1000, "b": 100, "c": 1})
    values = [table.value(concept) for concept in ("a", "b", "c")]
    assert all(0.0 <= value <= 1.0 for value in values)
    assert max(values) == pytest.approx(1.0)


def test_a_zero_frequency_concept_is_not_treated_as_maximally_specific():
    """Never observed is absence of evidence, not maximal specificity --
    -log(0) would be infinity, which is the wrong answer."""
    table = InformationContentTable.from_frequencies({"seen": 100, "never_seen": 0})
    assert table.get("never_seen").basis == BASIS_UNKNOWN
    assert table.value("never_seen") == DEFAULT_IC


def test_an_unknown_concept_gets_the_midpoint_not_zero_or_one():
    """Zero would make an unknown concept free to traverse (rewarding
    ignorance); one would make it maximally informative (rewarding it for the
    same reason). The midpoint is the honest position."""
    table = InformationContentTable.from_frequencies({"known": 10})
    entry = table.get("entirely absent")
    assert entry.value == DEFAULT_IC
    assert entry.basis == BASIS_UNKNOWN


def test_the_basis_of_every_score_is_recorded():
    """A reader must be able to tell a measurement from a default."""
    table = InformationContentTable.from_frequencies({"measured": 10})
    assert table.get("measured").basis == BASIS_FREQUENCY
    assert table.get("unmeasured").basis == BASIS_UNKNOWN


def test_an_empty_frequency_table_yields_an_empty_table_not_a_crash():
    assert len(InformationContentTable.from_frequencies({})) == 0
    assert len(InformationContentTable.from_frequencies({"a": 0, "b": -1})) == 0


# --------------------------------------------------------------------------
# Intrinsic IC from graph structure, for when no corpus frequency exists
# --------------------------------------------------------------------------


def test_graph_structure_makes_hub_concepts_general_and_leaf_concepts_specific():
    """The standard fallback when no corpus exists: a concept reachable from
    many others is general, one with few connections is specific."""
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("hub", "rel", "leaf_a", 0.5),
            ConceptEdge("hub", "rel", "leaf_b", 0.5),
            ConceptEdge("hub", "rel", "leaf_c", 0.5),
        ]
    )
    table = InformationContentTable.from_graph_structure(graph)
    assert table.value("hub") < table.value("leaf_a")
    assert table.get("hub").basis == BASIS_GRAPH_STRUCTURE


def test_graph_structure_on_an_empty_graph_yields_an_empty_table():
    assert len(InformationContentTable.from_graph_structure(InMemoryConceptGraph.from_edges([]))) == 0


# --------------------------------------------------------------------------
# Weighting paths
# --------------------------------------------------------------------------


def test_a_path_through_specific_concepts_outscores_one_through_general_ones():
    """The whole point of the weighting: same link strengths, different
    specificity, different conclusion about how much the path means."""
    specific_graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("marfan syndrome", "causes", "aortic root dilation", 0.8)]
    )
    general_graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("pulmonary", "part_of", "abnormality", 0.8)]
    )
    table = InformationContentTable.from_frequencies(
        {"pulmonary": 5000, "abnormality": 8000, "marfan syndrome": 40, "aortic root dilation": 300}
    )

    specific = score_path(find_paths(specific_graph, "marfan syndrome", "aortic root dilation")[0], table)
    general = score_path(find_paths(general_graph, "pulmonary", "abnormality")[0], table)

    assert specific.path_strength == pytest.approx(general.path_strength), "identical link strength"
    assert specific.weighted_strength > general.weighted_strength, "but not identical meaning"


def test_both_the_raw_and_weighted_strength_are_carried():
    """The supporting evidence for IC weighting is on UMLS/MeSH for word
    sense disambiguation, not on this graph for this task -- so the
    difference the weighting makes must be visible, not folded away."""
    table = InformationContentTable.from_frequencies({"marfan syndrome": 40, "aortic root dilation": 300})
    score = score_path(find_paths(_marfan_graph(), "marfan syndrome", "aortic root dilation")[0], table)
    assert score.path_strength > 0
    assert score.mean_information_content > 0
    assert score.weighted_strength == pytest.approx(score.path_strength * score.mean_information_content)


def test_a_score_resting_on_defaults_reports_that_it_is_not_measured():
    """A weighted score built mostly from DEFAULT_IC reports an assumption,
    not a measurement, and a reader must be able to tell which they have."""
    empty_table = InformationContentTable()
    score = score_path(find_paths(_marfan_graph(), "marfan syndrome", "aortic root dilation")[0], empty_table)
    assert score.all_concepts_measured is False


def test_path_concepts_lists_every_waypoint_in_order_without_repeats():
    path = find_paths(_marfan_graph(), "marfan syndrome", "aortic root dilation", max_hops=3)[0]
    concepts = path_concepts(path)
    assert concepts[0] == "marfan syndrome"
    assert concepts[-1] == "aortic root dilation"
    assert len(concepts) == len(set(concepts))


def test_rank_paths_orders_by_weighted_strength_strongest_first():
    table = InformationContentTable.from_frequencies(
        {"marfan syndrome": 40, "connective tissue weakness": 120, "aortic root dilation": 300}
    )
    ranked = rank_paths(find_paths(_marfan_graph(), "marfan syndrome", "aortic root dilation", max_hops=3), table)
    assert ranked == sorted(ranked, key=lambda item: -item.weighted_strength)


# --------------------------------------------------------------------------
# Converging paths: ONTOSPREAD's reward, and the failure mode it must avoid
# --------------------------------------------------------------------------


def test_two_independent_routes_score_above_the_best_single_one():
    """The graph-level analogue of this project's own cross-check principle:
    two independent routes to the same conclusion are worth more than one."""
    table = InformationContentTable.from_frequencies(
        {"marfan syndrome": 40, "connective tissue weakness": 120, "aortic root dilation": 300}
    )
    paths = find_paths(_marfan_graph(), "marfan syndrome", "aortic root dilation", max_hops=3)
    convergence = score_convergence(paths, table)

    assert convergence.independent_path_count == 2
    assert convergence.converged_strength > convergence.best_weighted_strength
    assert convergence.is_single_thread is False


def test_a_single_route_gets_no_convergence_bonus():
    table = InformationContentTable.from_frequencies({"a": 10, "b": 10})
    graph = InMemoryConceptGraph.from_edges([ConceptEdge("a", "rel", "b", 0.8)])
    convergence = score_convergence(find_paths(graph, "a", "b"), table)

    assert convergence.independent_path_count == 1
    assert convergence.convergence_multiplier == 1.0
    assert convergence.is_single_thread is True


def test_the_reward_is_sub_additive_so_connectivity_cannot_manufacture_certainty():
    """Each additional route adds less than the last. A formulation where the
    tenth path counted as much as the first would let a densely-connected
    region of the graph manufacture confidence by sheer connectivity."""
    increments = [CONVERGENCE_REWARD_BASE**index for index in range(1, 5)]
    assert increments == sorted(increments, reverse=True)
    assert all(later < earlier for earlier, later in itertools.pairwise(increments))


def test_converged_strength_never_exceeds_certainty():
    """A strength is probability-like; a convergence reward must not push it
    past 1.0 however many routes agree."""
    table = InformationContentTable.from_frequencies({name: 1 for name in "abcdef"})
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("a", "r1", "b", 1.0),
            ConceptEdge("a", "r2", "c", 1.0),
            ConceptEdge("c", "r3", "b", 1.0),
            ConceptEdge("a", "r4", "d", 1.0),
            ConceptEdge("d", "r5", "b", 1.0),
        ]
    )
    convergence = score_convergence(find_paths(graph, "a", "b", max_hops=3, max_paths=8), table)
    assert convergence.converged_strength <= 1.0


def test_routes_sharing_every_intermediate_are_not_counted_as_independent():
    """Two paths differing only in traversal direction, or sharing every
    waypoint, are one route reported twice -- counting them as corroboration
    would be self-confirmation."""
    table = InformationContentTable.from_frequencies({"a": 10, "b": 10, "via": 10})
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("a", "rel", "via", 0.9), ConceptEdge("via", "rel", "b", 0.9)]
    )
    convergence = score_convergence(find_paths(graph, "a", "b", max_hops=3), table)
    assert convergence.independent_path_count == 1


def test_an_intermediate_shared_by_every_route_is_reported():
    """A common waypoint is often the mechanism itself -- a real feature of
    the connection, not a defect -- but a reader should see when every route
    funnels through one node."""
    table = InformationContentTable.from_frequencies({"a": 10, "b": 10, "bottleneck": 10, "x": 10})
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("a", "r1", "bottleneck", 0.9),
            ConceptEdge("a", "r2", "x", 0.9),
            ConceptEdge("x", "r3", "bottleneck", 0.9),
            ConceptEdge("bottleneck", "r4", "b", 0.9),
        ]
    )
    convergence = score_convergence(find_paths(graph, "a", "b", max_hops=4), table)
    assert "bottleneck" in convergence.shared_intermediate_concepts


def test_no_paths_yields_a_zero_score_rather_than_a_crash():
    convergence = score_convergence([], InformationContentTable())
    assert convergence.converged_strength == 0.0
    assert convergence.independent_path_count == 0


def test_as_dict_carries_what_a_reviewer_needs():
    table = InformationContentTable.from_frequencies({"marfan syndrome": 40, "aortic root dilation": 300})
    paths = find_paths(_marfan_graph(), "marfan syndrome", "aortic root dilation", max_hops=3)
    payload = score_convergence(paths, table).as_dict()
    for key in ("independent_path_count", "converged_strength", "is_single_thread"):
        assert key in payload


def test_information_content_matches_the_classical_formula():
    """-log(p), normalised. Verified against a hand-computed case so the
    implementation cannot drift from the definition it claims to use."""
    table = InformationContentTable.from_frequencies({"common": 90, "rare": 10})
    raw_common, raw_rare = -math.log(0.9), -math.log(0.1)
    assert table.value("common") == pytest.approx(raw_common / raw_rare)
    assert table.value("rare") == pytest.approx(1.0)
