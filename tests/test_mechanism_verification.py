"""Tests for verifying a claimed mechanism against the concept graph."""

from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.information_content import InformationContentTable
from melampo.reasoning.mechanism_verification import (
    DISPOSITION_AGREED_AND_GROUNDED,
    DISPOSITION_AGREED_BUT_UNGROUNDED,
    DISPOSITION_DISAGREED_AND_UNGROUNDED,
    GROUNDING_CONNECTION_WITHOUT_THIS_MECHANISM,
    GROUNDING_NO_CONNECTION,
    GROUNDING_NOT_CHECKABLE,
    GROUNDING_SUPPORTED,
    cross_check_mechanisms,
    summarise,
    verify_mechanism,
)


def _graph() -> InMemoryConceptGraph:
    return InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("marfan syndrome", "causes", "connective tissue weakness", 0.9),
            ConceptEdge("connective tissue weakness", "causes", "aortic root dilation", 0.85),
            ConceptEdge("pulmonary", "has_phenotype", "aortic root dilation", 0.3),
        ]
    )


def _table() -> InformationContentTable:
    return InformationContentTable.from_frequencies(
        {"pulmonary": 5000, "aortic root dilation": 300, "connective tissue weakness": 120, "marfan syndrome": 40}
    )


def _answer(mechanism: str, factor: str = "marfan syndrome", target: str = "aortic root dilation") -> str:
    return f"{factor} | {target} | yes | {mechanism}"


# --------------------------------------------------------------------------
# Verifying one claim against the graph
# --------------------------------------------------------------------------


def test_a_mechanism_the_graph_supports_is_grounded():
    verification = verify_mechanism(
        _graph(), "marfan syndrome", "aortic root dilation", "connective tissue weakness", table=_table()
    )
    assert verification.grounding == GROUNDING_SUPPORTED
    assert verification.is_grounded is True
    assert verification.matched_concept == "connective tissue weakness"


def test_an_invented_mechanism_is_not_grounded_even_though_the_concepts_are_connected():
    """The graph links these two concepts, but not by what was claimed --
    distinct from the graph knowing of no connection at all."""
    verification = verify_mechanism(
        _graph(), "marfan syndrome", "aortic root dilation", "cosmic ray exposure", table=_table()
    )
    assert verification.grounding == GROUNDING_CONNECTION_WITHOUT_THIS_MECHANISM
    assert verification.is_grounded is False
    assert verification.graph_supports_any_connection is True
    assert "connective tissue weakness" in verification.candidate_mechanisms


def test_unconnected_concepts_yield_no_connection_not_merely_no_match():
    verification = verify_mechanism(_graph(), "marfan syndrome", "unrelated thing", "anything", table=_table())
    assert verification.grounding == GROUNDING_NO_CONNECTION
    assert verification.graph_supports_any_connection is False


def test_a_low_information_content_concept_does_not_count_as_support():
    """"pulmonary" is reached from both origins in this fixture but its
    weighted activation is ~0.004 -- admitting it would undo the entire
    Information Content result this builds on."""
    verification = verify_mechanism(
        _graph(), "marfan syndrome", "aortic root dilation", "pulmonary", table=_table()
    )
    assert verification.is_grounded is False


def test_an_unstated_factor_or_target_is_not_checkable():
    verification = verify_mechanism(_graph(), "", "aortic root dilation", "something", table=_table())
    assert verification.grounding == GROUNDING_NOT_CHECKABLE
    assert verification.notes


def test_no_mechanism_claimed_still_reports_what_the_graph_offers():
    verification = verify_mechanism(_graph(), "marfan syndrome", "aortic root dilation", "", table=_table())
    assert verification.is_grounded is False
    assert "connective tissue weakness" in verification.candidate_mechanisms


# --------------------------------------------------------------------------
# Matching a claim to a graph node
# --------------------------------------------------------------------------


def test_a_reordered_phrasing_matches_the_same_concept():
    """"weakness of connective tissue" and "connective tissue weakness" are
    one concept permuted -- containment misses it because the words are
    reordered rather than nested."""
    verification = verify_mechanism(
        _graph(), "marfan syndrome", "aortic root dilation", "weakness of connective tissue", table=_table()
    )
    assert verification.is_grounded is True


def test_sharing_a_word_is_not_enough_to_match():
    """The permissiveness must not extend to genuinely different claims:
    "aortic dilation" and "pulmonary dilation" share a word, not a word set."""
    graph = InMemoryConceptGraph.from_edges(
        [
            ConceptEdge("a", "causes", "aortic dilation", 0.9),
            ConceptEdge("aortic dilation", "causes", "b", 0.9),
        ]
    )
    table = InformationContentTable.from_frequencies({"a": 10, "b": 10, "aortic dilation": 10})
    verification = verify_mechanism(graph, "a", "b", "pulmonary dilation", table=table)
    assert verification.is_grounded is False


def test_synonym_resolution_is_a_recorded_miss_not_a_silent_guess():
    """"inherited aortopathy" and "connective tissue weakness" may name the
    same real thing; this module does not pretend to know that, and says so
    rather than guessing."""
    verification = verify_mechanism(
        _graph(), "marfan syndrome", "aortic root dilation", "inherited aortopathy", table=_table()
    )
    assert verification.is_grounded is False
    assert verification.candidate_mechanisms, "what the graph does offer is still reported"


# --------------------------------------------------------------------------
# The four combinations, and the one that was previously invisible
# --------------------------------------------------------------------------


def test_two_models_agreeing_on_a_grounded_mechanism_needs_no_review():
    check = cross_check_mechanisms(
        _graph(), _answer("connective tissue weakness"), _answer("connective tissue weakness"), table=_table()
    )
    assert check.disposition == DISPOSITION_AGREED_AND_GROUNDED
    assert check.needs_review is False


def test_two_models_agreeing_on_an_invented_mechanism_is_caught():
    """The case this module exists for. Without the graph, two models saying
    the same thing looked identical to clean agreement -- agreement measures
    whether two answers match each other, not whether either is grounded."""
    check = cross_check_mechanisms(
        _graph(), _answer("cosmic ray exposure"), _answer("cosmic ray exposure"), table=_table()
    )
    assert check.disposition == DISPOSITION_AGREED_BUT_UNGROUNDED
    assert check.models_agree is True, "they do agree -- that is exactly the danger"
    assert check.needs_review is True


def test_different_wording_of_one_grounded_concept_is_agreement():
    """The false-alarm case string comparison could never resolve: the graph
    resolves it without needing a synonym table, because both claims land on
    the same node."""
    check = cross_check_mechanisms(
        _graph(), _answer("connective tissue weakness"), _answer("weakness of connective tissue"), table=_table()
    )
    assert check.disposition == DISPOSITION_AGREED_AND_GROUNDED
    assert check.needs_review is False


def test_one_grounded_one_not_is_disagreement_needing_review():
    check = cross_check_mechanisms(
        _graph(), _answer("connective tissue weakness"), _answer("pulmonary"), table=_table()
    )
    assert check.disposition == DISPOSITION_DISAGREED_AND_UNGROUNDED
    assert check.needs_review is True


def test_each_model_is_checked_against_its_own_stated_factor_and_target():
    """If two models disagree about what the question is even relating, that
    is itself a finding -- normalising it away would hide it."""
    check = cross_check_mechanisms(
        _graph(),
        _answer("connective tissue weakness"),
        _answer("connective tissue weakness", factor="pulmonary", target="something else"),
        table=_table(),
    )
    assert check.primary.factor == "marfan syndrome"
    assert check.secondary.factor == "pulmonary"


def test_agreement_and_grounding_are_reported_as_separate_axes():
    """Collapsing them into one verdict would discard exactly what the graph
    was consulted for."""
    check = cross_check_mechanisms(
        _graph(), _answer("cosmic ray exposure"), _answer("cosmic ray exposure"), table=_table()
    )
    payload = check.as_dict()
    assert payload["models_agree"] is True
    assert payload["both_grounded"] is False


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_summarise_counts_by_disposition():
    grounded = cross_check_mechanisms(
        _graph(), _answer("connective tissue weakness"), _answer("connective tissue weakness"), table=_table()
    )
    ungrounded = cross_check_mechanisms(
        _graph(), _answer("cosmic ray exposure"), _answer("cosmic ray exposure"), table=_table()
    )
    summary = summarise([grounded, ungrounded])
    assert summary["cases"] == 2
    assert summary["needing_review"] == 1


def test_summarise_of_nothing_does_not_divide_by_zero():
    assert summarise([])["cases"] == 0


def test_as_dict_carries_what_a_reviewer_needs():
    check = cross_check_mechanisms(
        _graph(), _answer("cosmic ray exposure"), _answer("connective tissue weakness"), table=_table()
    )
    payload = check.as_dict()
    for key in ("disposition", "needs_review", "primary", "secondary"):
        assert key in payload
    assert "candidate_mechanisms" in payload["primary"]
