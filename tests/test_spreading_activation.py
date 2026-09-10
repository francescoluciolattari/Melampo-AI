"""Tests for constrained spreading activation."""


from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.information_content import InformationContentTable
from melampo.memory.spreading_activation import (
    DEFAULT_ALLOWED_RELATIONS,
    DEFAULT_DECAY,
    DEFAULT_MAX_HOPS,
    DEFAULT_THRESHOLD,
    RELATION_HAS_PHENOTYPE,
    mediating_concepts,
    spread,
)
from melampo.reasoning.illness_script import SCRIPT_RELATIONS


def _marfan_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("marfan syndrome", "causes", "connective tissue weakness", 0.9),
            ConceptEdge("connective tissue weakness", "causes", "aortic root dilation", 0.85),
            ConceptEdge("marfan syndrome", "associated_with", "aortic root dilation", 0.7),
            ConceptEdge("pulmonary", RELATION_HAS_PHENOTYPE, "aortic root dilation", 0.3),
        ]
    )


def _table() -> InformationContentTable:
    return InformationContentTable.from_frequencies(
        {"pulmonary": 5000, "aortic root dilation": 300, "connective tissue weakness": 120, "marfan syndrome": 40}
    )


# --------------------------------------------------------------------------
# The relation constraint: the documented reason unconstrained spreading fails
# --------------------------------------------------------------------------


def test_the_default_vocabulary_is_the_projects_own_not_a_new_one():
    """SCRIPT_RELATIONS already exists in illness_script.py; this must reuse
    it rather than define a parallel vocabulary that could drift."""
    assert SCRIPT_RELATIONS <= DEFAULT_ALLOWED_RELATIONS
    assert RELATION_HAS_PHENOTYPE in DEFAULT_ALLOWED_RELATIONS


def test_a_disallowed_relation_does_not_propagate_activation():
    """The specific failure mode constrained spreading exists to avoid:
    activation must not leak through every edge type indiscriminately."""
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("a", "unrelated_mention", "b", 0.99)]
    )
    result = spread(graph, ["a"], allowed_relations={"causes"})
    assert "b" not in {item.concept for item in result.activated}
    assert result.blocked_relation_count > 0


def test_a_disallowed_relation_is_reported_not_silently_dropped():
    """A spread that blocked nothing was effectively unconstrained -- a
    reader must be able to see whether the constraint did anything."""
    result = spread(_marfan_graph(), ["marfan syndrome"], allowed_relations={"causes"})
    assert result.blocked_relation_count > 0
    assert result.allowed_relations == ("causes",)


def test_narrowing_the_allowed_relations_narrows_what_is_reached():
    graph = _marfan_graph()
    wide = spread(graph, ["marfan syndrome"], allowed_relations={"causes", "associated_with"})
    narrow = spread(graph, ["marfan syndrome"], allowed_relations={"causes"})
    assert len(narrow.activated) <= len(wide.activated)


def test_an_inverse_relation_is_matched_by_its_base_type():
    """The constraint is about relation *type*, not direction -- a backward
    traversal of an allowed relation must not be blocked as unknown."""
    graph = InMemoryConceptGraph.from_edges([ConceptEdge("a", "causes", "b", 0.9)])
    forward = spread(graph, ["a"], allowed_relations={"causes"})
    backward = spread(graph, ["b"], allowed_relations={"causes"})
    assert "b" in {item.concept for item in forward.activated}
    assert "a" in {item.concept for item in backward.activated}


# --------------------------------------------------------------------------
# The decay and threshold constraints: what bounds the frontier
# --------------------------------------------------------------------------


def test_activation_decays_with_distance():
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("a", "causes", "b", 0.95), ConceptEdge("b", "causes", "c", 0.95)]
    )
    result = spread(graph, ["a"], allowed_relations={"causes"}, threshold=0.0)
    by_concept = {item.concept: item for item in result.activated}
    assert by_concept["b"].activation > by_concept["c"].activation


def test_activation_below_threshold_does_not_propagate():
    graph = InMemoryConceptGraph.from_edges([ConceptEdge("a", "causes", "b", 0.01)])
    result = spread(graph, ["a"], allowed_relations={"causes"}, threshold=0.5)
    assert "b" not in {item.concept for item in result.activated}
    assert result.frontier_stopped_at_threshold > 0


def test_a_high_threshold_reaches_fewer_concepts_than_a_low_one():
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("a", "causes", "b", 0.6), ConceptEdge("b", "causes", "c", 0.6)]
    )
    loose = spread(graph, ["a"], allowed_relations={"causes"}, threshold=0.01)
    strict = spread(graph, ["a"], allowed_relations={"causes"}, threshold=0.5)
    assert len(strict.activated) <= len(loose.activated)


def test_max_hops_bounds_how_far_activation_travels_regardless_of_decay():
    """Belt and braces: a pathological graph with weight-1.0 cycles could
    otherwise keep activation above threshold indefinitely."""
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge(f"n{i}", "causes", f"n{i + 1}", 1.0) for i in range(10)]
    )
    result = spread(graph, ["n0"], allowed_relations={"causes"}, threshold=0.0, decay=1.0, max_hops=2)
    assert all(item.hops <= 2 for item in result.activated)


# --------------------------------------------------------------------------
# IC is applied at the destination, not along the way
# --------------------------------------------------------------------------


def test_information_content_suppresses_a_reached_but_uninformative_concept():
    """The demonstration this module exists for: "pulmonary" is structurally
    reachable but its IC is near zero, so its weighted activation is
    negligible despite raw activation being present."""
    result = mediating_concepts(_marfan_graph(), "marfan syndrome", "aortic root dilation", table=_table())
    by_concept = {item.concept: item for item in result.activated}
    assert by_concept["pulmonary"].activation > 0, "reached structurally"
    assert by_concept["pulmonary"].weighted_activation < 0.05, "but suppressed by low IC"


def test_a_concept_with_no_ic_table_gets_the_default_not_a_crash():
    result = spread(_marfan_graph(), ["marfan syndrome"], allowed_relations={"causes"})
    assert all(0.0 <= item.information_content <= 1.0 for item in result.activated)


# --------------------------------------------------------------------------
# Multi-sourcing: what a relevance question actually needs
# --------------------------------------------------------------------------


def test_mediating_concepts_finds_what_is_reached_from_both_ends():
    """This is what "does X bear on Y" needs and find_paths alone does not
    give: not whether a route exists, but what mediates."""
    result = mediating_concepts(_marfan_graph(), "marfan syndrome", "aortic root dilation", table=_table())
    mediators = {item.concept for item in result.multiply_sourced()}
    assert "connective tissue weakness" in mediators


def test_the_true_mechanism_outranks_a_multiply_sourced_but_uninformative_concept():
    """Both "connective tissue weakness" and "pulmonary" are reached from both
    origins in this fixture, but only one is IC-weighted highly -- ranking by
    weighted_activation must put the real mechanism first."""
    result = mediating_concepts(_marfan_graph(), "marfan syndrome", "aortic root dilation", table=_table())
    ranked = result.ranked()
    assert ranked[0].concept == "connective tissue weakness"


def test_a_singly_sourced_concept_is_not_reported_as_multiply_sourced():
    graph = InMemoryConceptGraph.from_edges([ConceptEdge("a", "causes", "only_a_reaches_this", 0.9)])
    result = spread(graph, ["a", "b"], allowed_relations={"causes"})
    by_concept = {item.concept: item for item in result.activated}
    assert by_concept["only_a_reaches_this"].is_multiply_sourced is False
    assert by_concept["only_a_reaches_this"].source_count == 1


def test_origins_themselves_are_not_reported_as_activated():
    """A concept is not evidence of its own relevance; including origins
    would put the two concepts a relevance question already names at the top
    of its own answer."""
    result = mediating_concepts(_marfan_graph(), "marfan syndrome", "aortic root dilation", table=_table())
    activated_concepts = {item.concept for item in result.activated}
    assert "marfan syndrome" not in activated_concepts
    assert "aortic root dilation" not in activated_concepts


def test_no_overlap_between_two_spreads_is_a_distinct_finding_from_no_data():
    """A spread reaching nothing from one side differs from one reaching
    plenty on both sides with no overlap -- both must be visible, not
    collapsed into "no mediators found"."""
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("a", "causes", "only_from_a", 0.9), ConceptEdge("b", "causes", "only_from_b", 0.9)]
    )
    result = mediating_concepts(graph, "a", "b", allowed_relations={"causes"})
    assert result.multiply_sourced() == []
    assert len(result.activated) == 2, "both reached, neither shared -- visible, not hidden"


# --------------------------------------------------------------------------
# Defaults and reporting
# --------------------------------------------------------------------------


def test_defaults_are_used_when_not_overridden():
    result = spread(_marfan_graph(), ["marfan syndrome"])
    assert result.allowed_relations == tuple(sorted(DEFAULT_ALLOWED_RELATIONS))


def test_an_unreachable_origin_yields_an_empty_result_not_a_crash():
    result = spread(_marfan_graph(), ["nonexistent concept"])
    assert result.activated == []


def test_as_dict_carries_what_a_reviewer_needs():
    result = mediating_concepts(_marfan_graph(), "marfan syndrome", "aortic root dilation", table=_table())
    payload = result.as_dict()
    for key in ("multiply_sourced_count", "blocked_relation_count", "activated"):
        assert key in payload
    if payload["activated"]:
        for key in ("weighted_activation", "is_multiply_sourced", "source_count"):
            assert key in payload["activated"][0]


def test_default_constants_are_all_positive_and_sane():
    assert 0.0 < DEFAULT_DECAY <= 1.0
    assert 0.0 < DEFAULT_THRESHOLD < 1.0
    assert DEFAULT_MAX_HOPS >= 1
