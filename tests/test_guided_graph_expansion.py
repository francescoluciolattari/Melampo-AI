"""Tests for the model-guided graph expansion fallback."""

from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.guided_graph_expansion import (
    STOP_DEAD_END,
    STOP_FINAL,
    STOP_GAVE_UP,
    STOP_ITERATIONS,
    STOP_NO_ACTION,
    guided_expand,
)


def _chain_graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("marfan syndrome", "causes", "connective tissue weakness", 0.9),
            ConceptEdge("connective tissue weakness", "causes", "aortic root dilation", 0.85),
        ]
    )


def _scripted(*outputs: str):
    queue = iter(outputs)
    return lambda prompt: next(queue, "give_up()")


# --------------------------------------------------------------------------
# The four ways a walk can end
# --------------------------------------------------------------------------


def test_a_walk_that_reaches_a_final_concept_is_marked_found():
    result = guided_expand(
        _chain_graph(), "marfan syndrome", "aortic root dilation",
        _scripted("neighbor(connective tissue weakness)", "final(connective tissue weakness)"),
    )
    assert result.found_something is True
    assert result.final_concept == "connective tissue weakness"
    assert result.stop_reason == STOP_FINAL


def test_a_model_giving_up_is_a_distinct_named_outcome():
    result = guided_expand(_chain_graph(), "marfan syndrome", "aortic root dilation", _scripted("give_up()"))
    assert result.found_something is False
    assert result.stop_reason == STOP_GAVE_UP


def test_a_genuine_dead_end_stops_without_calling_the_model_again():
    """A concept whose only outgoing relation is not admitted has nothing to
    offer -- the walk must recognise this and stop, not hang."""
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("isolated", "unrelated_relation_not_admitted", "elsewhere", 0.5)]
    )
    result = guided_expand(graph, "isolated", "anything", _scripted("final(x)"))
    assert result.stop_reason == STOP_DEAD_END
    assert result.found_something is False


def test_iteration_budget_is_respected():
    long_chain = InMemoryConceptGraph.from_edges(
        [ConceptEdge(f"n{i}", "causes", f"n{i + 1}", 0.9) for i in range(10)]
    )
    always_move_on = _scripted(*[f"neighbor(n{i + 1})" for i in range(10)])
    result = guided_expand(long_chain, "n0", "n99", always_move_on, max_hops=3)
    assert result.stop_reason == STOP_ITERATIONS
    assert result.hops == 3


# --------------------------------------------------------------------------
# The property that must never break: no invented edge is ever followed
# --------------------------------------------------------------------------


def test_moving_to_a_concept_never_offered_is_refused_not_followed():
    """The whole safety property of this module: the model can only choose
    among the graph's actual neighbours, offered by name each turn. A move to
    anything else is not an invented edge silently taken -- it is treated as
    an ill-formed action, the same as any other output the walk cannot parse."""
    result = guided_expand(
        _chain_graph(), "marfan syndrome", "aortic root dilation",
        _scripted("neighbor(a concept that does not exist anywhere)"),
    )
    assert result.stop_reason == STOP_NO_ACTION
    assert result.found_something is False
    assert result.visited_path == ("marfan syndrome",), "no move away from the start happened"


def test_an_unparseable_output_stops_the_walk_rather_than_guessing():
    result = guided_expand(_chain_graph(), "marfan syndrome", "aortic root dilation", _scripted("I am not sure."))
    assert result.stop_reason == STOP_NO_ACTION


def test_every_concept_in_the_visited_path_is_a_real_graph_node():
    """Indirect confirmation of the same safety property: whatever path was
    actually walked, every step in it must be a concept the graph itself
    names via an edge -- never something the model introduced."""
    graph = _chain_graph()
    real_concepts = {"marfan syndrome", "connective tissue weakness", "aortic root dilation"}
    result = guided_expand(
        graph, "marfan syndrome", "aortic root dilation",
        _scripted("neighbor(connective tissue weakness)", "final(connective tissue weakness)"),
    )
    assert set(result.visited_path) <= real_concepts


# --------------------------------------------------------------------------
# Provenance: every result says where it came from
# --------------------------------------------------------------------------


def test_every_result_is_marked_as_coming_from_a_guided_walk():
    """Whatever the outcome, a caller must never mistake this for the
    deterministic pass's output."""
    for outputs in (("final(x)",), ("give_up()",), ("garbage",)):
        result = guided_expand(_chain_graph(), "marfan syndrome", "aortic root dilation", _scripted(*outputs))
        assert result.via_guided_expansion is True


def test_the_visited_path_starts_at_the_given_origin():
    result = guided_expand(_chain_graph(), "marfan syndrome", "aortic root dilation", _scripted("give_up()"))
    assert result.visited_path[0] == "marfan syndrome"


def test_as_dict_carries_what_a_reviewer_needs():
    result = guided_expand(
        _chain_graph(), "marfan syndrome", "aortic root dilation",
        _scripted("neighbor(connective tissue weakness)", "final(connective tissue weakness)"),
    )
    payload = result.as_dict()
    for key in ("final_concept", "visited_path", "stop_reason", "via_guided_expansion", "found_something"):
        assert key in payload


# --------------------------------------------------------------------------
# Relation constraint still applies -- a guided walk is not free of it
# --------------------------------------------------------------------------


def test_a_disallowed_relation_is_never_offered_as_a_neighbour():
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("a", "causes", "b", 0.9),
            ConceptEdge("a", "loosely_associated_with", "c", 0.9),
        ]
    )
    # A model told to move to "c" -- reachable only by a disallowed relation --
    # must find that move refused, since "c" is never in its offered neighbours.
    result = guided_expand(graph, "a", "b", _scripted("neighbor(c)"), allowed_relations={"causes"})
    assert result.stop_reason == STOP_NO_ACTION
