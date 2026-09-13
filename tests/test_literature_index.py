"""Tests for literature as a retrieval source with citable provenance."""

from melampo.evaluation.enumeration_bench import differential_graph
from melampo.memory.literature_index import (
    LiteratureIndex,
    LiteraturePassage,
    as_vetting_context,
)
from melampo.reasoning.rlm_graph_bridge import (
    ORIGIN_RLM_CONJECTURE,
    bridge,
    vet_rlm_claims,
)


def _checkable(passage_id: str, text: str, source_id: str = "pmid:12345678") -> LiteraturePassage:
    return LiteraturePassage(passage_id, text, "A title", source_id, 2024, "A journal")


class _Trajectory:
    final_answer = "Bilateral hilar lymphadenopathy."

    def evidence(self):
        return [
            {"record_id": "r1", "text": "bilateral hilar lymphadenopathy on CT"},
            {"record_id": "r2", "text": "hypercalcaemia confirmed"},
        ]


# --------------------------------------------------------------------------
# Citability: what makes a passage evidence rather than decoration
# --------------------------------------------------------------------------


def test_a_passage_with_a_resolvable_identifier_is_independently_checkable():
    assert _checkable("p1", "text", "pmid:12345678").is_independently_checkable is True
    assert _checkable("p2", "text", "doi:10.1000/xyz").is_independently_checkable is True


def test_a_passage_without_a_resolvable_identifier_is_not_checkable():
    """A source a reviewer cannot open is an assertion with a
    bibliography-shaped decoration -- worse than no citation, because it
    looks like one."""
    assert _checkable("p3", "text", "internal-note-4").is_independently_checkable is False


def test_search_excludes_uncheckable_passages_by_default():
    """Material that cannot be verified must not reach a vetting engine as
    though it could; a caller wanting everything asks explicitly."""
    index = LiteratureIndex()
    index.add(_checkable("p1", "Sarcoidosis causes hypercalcaemia.", "internal-note-4"))

    assert index.search(["sarcoidosis"], differential_graph()) == []
    assert index.search(["sarcoidosis"], differential_graph(), checkable_only=False)


def test_a_citation_string_carries_enough_to_find_the_source():
    citation = _checkable("p1", "text").citation()
    assert "pmid:12345678" in citation
    assert "2024" in citation


# --------------------------------------------------------------------------
# Retrieval: concept-matched, breadth-ranked
# --------------------------------------------------------------------------


def test_passages_are_found_by_the_concepts_they_mention():
    index = LiteratureIndex()
    index.add(_checkable("p1", "Hypercalcaemia in sarcoidosis arises from extrarenal calcitriol."))

    hits = index.search(["sarcoidosis", "hypercalcaemia"], differential_graph())

    assert len(hits) == 1
    assert set(hits[0].matched_concepts) == {"sarcoidosis", "hypercalcaemia"}


def test_a_passage_matching_more_concepts_ranks_first():
    """Breadth before strength, the same rule candidate_retrieval uses."""
    index = LiteratureIndex()
    index.add(_checkable("narrow", "Sarcoidosis is a granulomatous disease.", "pmid:111"))
    index.add(_checkable("broad", "Sarcoidosis causes hypercalcaemia and erythema nodosum.", "pmid:222"))

    hits = index.search(["sarcoidosis", "hypercalcaemia", "erythema nodosum"], differential_graph())

    assert hits[0].passage.passage_id == "broad"


def test_searching_for_nothing_returns_nothing():
    index = LiteratureIndex()
    index.add(_checkable("p1", "Sarcoidosis."))
    assert index.search([], differential_graph()) == []


def test_a_passage_mentioning_none_of_the_concepts_is_not_returned():
    index = LiteratureIndex()
    index.add(_checkable("p1", "An unrelated discussion of renal physiology."))
    assert index.search(["sarcoidosis"], differential_graph()) == []


# --------------------------------------------------------------------------
# Formatting for a vetting engine
# --------------------------------------------------------------------------


def test_each_passage_is_prefixed_with_its_citation_not_followed_by_it():
    """A model attributes what it reads to whatever is nearest; a reference
    trailing a long passage is easy to lose."""
    index = LiteratureIndex()
    index.add(_checkable("p1", "Sarcoidosis causes hypercalcaemia."))
    context = as_vetting_context(index.search(["sarcoidosis"], differential_graph()))

    assert context.index("pmid:12345678") < context.index("Sarcoidosis causes")


def test_truncation_drops_whole_passages_not_partial_text():
    """Half a passage under a full citation would attribute to a source
    something it did not finish saying."""
    index = LiteratureIndex()
    for i in range(5):
        index.add(_checkable(f"p{i}", "Sarcoidosis. " + ("x" * 500), f"pmid:{i}"))

    context = as_vetting_context(index.search(["sarcoidosis"], differential_graph(), limit=5), max_chars=1200)

    assert context.count("pmid:") <= 3
    assert not context.endswith("x" * 10) or context.count("[") == context.count("pmid:")


def test_an_empty_retrieval_formats_to_an_empty_string():
    assert as_vetting_context([]) == ""


# --------------------------------------------------------------------------
# The citation qualifier: same branch, different checkability
# --------------------------------------------------------------------------


def test_a_cited_conjecture_stays_in_the_rlm_branch():
    """The correction this implements: a claim citing literature is still
    produced by the RLM, not by a source standing peer to the graph or the
    patient's chart."""
    claims = [("hypercalcaemia", "erythema nodosum", "sarcoidosis")]
    vetted = vet_rlm_claims(claims, differential_graph(), citations_by_claim={0: ["pmid:123"]})

    assert vetted[0].as_dict()["origin"] == ORIGIN_RLM_CONJECTURE


def test_a_cited_conjecture_is_marked_citation_supported():
    claims = [("erythema nodosum", "night sweats", "some proposed link")]
    vetted = vet_rlm_claims(claims, differential_graph(), citations_by_claim={0: ["pmid:123"]})

    assert vetted[0].is_citation_supported is True


def test_an_uncited_conjecture_is_not_citation_supported():
    """The distinction that matters between two conjectures the graph cannot
    confirm: one can be checked today, the other waits for confirmations."""
    claims = [("erythema nodosum", "night sweats", "some proposed link")]
    vetted = vet_rlm_claims(claims, differential_graph())

    assert vetted[0].is_citation_supported is False
    assert vetted[0].is_candidate_conjecture is True, "still a conjecture either way"


def test_citations_do_not_make_an_ungrounded_claim_grounded():
    """A citation changes how quickly a claim can be checked, never whether
    the graph supports it -- promoting on a citation alone would be trusting
    the model's own reading of a paper it selected."""
    claims = [("erythema nodosum", "night sweats", "some proposed link")]
    vetted = vet_rlm_claims(claims, differential_graph(), citations_by_claim={0: ["pmid:123"]})

    assert vetted[0].is_grounded is False


def test_claims_without_citations_are_unaffected_by_the_new_parameter():
    """Backward compatibility: existing callers passing no citations get
    exactly what they got before."""
    claims = [("hypercalcaemia", "erythema nodosum", "sarcoidosis")]
    assert vet_rlm_claims(claims, differential_graph())[0].citations == ()


# --------------------------------------------------------------------------
# Wired into the bridge
# --------------------------------------------------------------------------


def test_the_bridge_retrieves_literature_for_the_case_concepts():
    index = LiteratureIndex()
    index.add(_checkable("p1", "Hypercalcaemia in sarcoidosis arises from extrarenal calcitriol."))

    result = bridge(_Trajectory(), differential_graph(), literature=index)

    assert len(result.retrieved_literature) == 1


def test_literature_is_kept_separate_from_the_graphs_own_output():
    """A passage from a case report is not an ontology edge; a system letting
    the two become interchangeable has given up its provenance design."""
    index = LiteratureIndex()
    index.add(_checkable("p1", "Sarcoidosis causes hypercalcaemia."))

    payload = bridge(_Trajectory(), differential_graph(), literature=index).as_dict()

    assert "retrieved_literature" in payload
    assert "hypotheses" in payload
    assert payload["retrieved_literature"] is not payload["hypotheses"]


def test_the_bridge_works_without_a_literature_index():
    result = bridge(_Trajectory(), differential_graph())
    assert result.retrieved_literature == []
