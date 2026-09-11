"""Tests for concept_paths.py's shared concept-matching and resolution utilities."""

from melampo.memory.concept_paths import (
    ConceptEdge,
    InMemoryConceptGraph,
    concept_names_match,
    mentioned_concepts,
    resolve_concept,
)

# --------------------------------------------------------------------------
# concept_names_match: one shared comparison rule, not duplicated per caller
# --------------------------------------------------------------------------


def test_concept_names_match_recognises_reordered_words():
    """The exact case that motivated moving this comparison here: two callers
    (mechanism matching and factor/target resolution) must use one rule, not
    two that happen to agree only on the cases already tested."""

    assert concept_names_match("kidney chronic disease", "chronic kidney disease") is True


def test_concept_names_match_still_refuses_genuinely_different_specific_terms():
    """The red line this whole line of work exists to hold: "pulmonary
    embolism" must never match "pulmonary oedema" merely for sharing a word."""

    assert concept_names_match("pulmonary embolism", "pulmonary oedema") is False


def test_concept_names_match_does_not_resolve_synonyms():

    assert concept_names_match("inherited aortopathy", "connective tissue weakness") is False


# --------------------------------------------------------------------------
# resolve_concept: two tiers, safe order, and mentioned_concepts untouched
# --------------------------------------------------------------------------


def test_resolve_concept_finds_an_exact_contiguous_mention_via_tier_one():

    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("chronic kidney disease", "causes", "x", 0.8)]
    )
    assert resolve_concept("chronic kidney disease (ckd)", graph) == "chronic kidney disease"


def test_resolve_concept_falls_back_to_word_set_matching_for_reordered_text():
    """The gap mentioned_concepts alone could not close: word order preserved
    is tier one's requirement; tier two tolerates reordering."""

    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge("chronic kidney disease", "causes", "x", 0.8)]
    )
    assert resolve_concept("kidney chronic disease", graph) == "chronic kidney disease"


def test_resolve_concept_returns_none_rather_than_guessing():

    graph = InMemoryConceptGraph.from_edges([ConceptEdge("chronic kidney disease", "causes", "x", 0.8)])
    assert resolve_concept("something entirely unrelated", graph) is None


def test_resolve_concept_prefers_tier_one_when_both_would_match():
    """Tier one (contiguous, the stricter and already-relied-upon test) runs
    first; tier two is a fallback, not an equal alternative."""

    graph = InMemoryConceptGraph.from_edges([ConceptEdge("chronic kidney disease", "causes", "x", 0.8)])
    # Both tiers would resolve this correctly; the point is tier one is tried
    # first and succeeds without needing tier two at all.
    assert resolve_concept("the patient's chronic kidney disease", graph) == "chronic kidney disease"


def test_mentioned_concepts_itself_is_unchanged_by_this_addition():
    """resolve_concept must not have altered mentioned_concepts's own
    behaviour -- grounding_judge.py relies on it as it was, on text that can
    be much longer than a single factor or target, and was not re-verified
    against a more permissive matching rule as part of this change."""
    graph = InMemoryConceptGraph.from_edges([ConceptEdge("chronic kidney disease", "causes", "x", 0.8)])
    # The reordered case that tier two of resolve_concept now handles must
    # still be invisible to mentioned_concepts on its own -- proving the
    # extra tolerance was added as a new, separate tier, not baked into the
    # existing, relied-upon function.
    assert mentioned_concepts("kidney chronic disease", graph) == []
