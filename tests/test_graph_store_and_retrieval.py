"""Tests for graph persistence and for retrieving candidate conditions from findings."""

import json

from melampo.evaluation.enumeration_bench import differential_graph
from melampo.memory.candidate_retrieval import retrieve_candidates
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.graph_store import (
    LearnedEdgeStore,
    build_persistent_graph,
    is_learned,
    learned_provenance,
)
from melampo.training.mechanism_enumeration import MechanismEnumerator

# --------------------------------------------------------------------------
# Persistence: what the system learns must survive a restart
# --------------------------------------------------------------------------


def test_a_learned_edge_survives_a_round_trip_with_its_interval_intact(tmp_path):
    """An interval-valued edge must not collapse back to a point estimate --
    the bounds are the epistemic state, and losing them would make a
    documented-rare edge indistinguishable from an unknown one."""
    store = LearnedEdgeStore(tmp_path / "learned.jsonl")
    store.append(
        ConceptEdge("rare syndrome", "causes", "unusual finding", 0.62, provenance="learned:c1", lower=0.41, upper=0.83)
    )

    loaded = store.load().edges[0]

    assert loaded.weight == 0.62
    assert loaded.lower == 0.41
    assert loaded.upper == 0.83


def test_a_store_that_has_never_been_written_is_empty_not_an_error(tmp_path):
    """The normal state of a new installation, not a fault -- raising would
    make every caller wrap a first run in try/except."""
    store = LearnedEdgeStore(tmp_path / "never_written.jsonl")
    assert store.load().edges == []
    assert store.count() == 0


def test_a_corrupt_line_is_reported_rather_than_stopping_the_load(tmp_path):
    """One bad line should not make the system refuse to start, and should
    not vanish silently either."""
    path = tmp_path / "learned.jsonl"
    path.write_text(
        json.dumps({"source": "a", "relation": "causes", "target": "b", "weight": 0.5})
        + "\nthis is not json\n"
        + json.dumps({"source": "c", "relation": "causes", "target": "d", "weight": 0.7})
        + "\n"
    )

    report = LearnedEdgeStore(path).load()

    assert len(report.edges) == 2, "the good lines still load"
    assert len(report.skipped_lines) == 1


def test_appending_does_not_rewrite_existing_content(tmp_path):
    """Append-only by design: an edge promoted after three confirmations is a
    claim about accumulated evidence, and rewriting would erase the trail."""
    store = LearnedEdgeStore(tmp_path / "learned.jsonl")
    store.append(ConceptEdge("a", "causes", "b", 0.5))
    store.append(ConceptEdge("c", "causes", "d", 0.7))

    edges = store.load().edges

    assert len(edges) == 2
    assert edges[0].source == "a", "the first edge is still first, not overwritten"


def test_append_many_writes_every_edge_in_one_open(tmp_path):
    store = LearnedEdgeStore(tmp_path / "learned.jsonl")
    written = store.append_many([ConceptEdge(f"s{i}", "causes", "t", 0.5) for i in range(5)])
    assert written == 5
    assert store.count() == 5


def test_append_many_with_nothing_writes_nothing_and_creates_no_file(tmp_path):
    path = tmp_path / "learned.jsonl"
    assert LearnedEdgeStore(path).append_many([]) == 0
    assert not path.exists()


# --------------------------------------------------------------------------
# The separation between imported and learned, which the design rests on
# --------------------------------------------------------------------------


def test_a_learned_edge_is_distinguishable_from_an_imported_one(tmp_path):
    """The distinction matters more, not less, as the learned layer grows:
    a clinician reading a path should see which links came from a published
    ontology and which the system inferred."""
    learned = ConceptEdge("a", "causes", "b", 0.5, provenance=learned_provenance("case-9", 3))
    imported = ConceptEdge("c", "has_phenotype", "d", 0.8, provenance="hpoa:OMIM:154700")

    assert is_learned(learned) is True
    assert is_learned(imported) is False


def test_an_edge_with_no_provenance_is_not_mistaken_for_learned():
    assert is_learned(ConceptEdge("a", "causes", "b", 0.5)) is False


def test_the_learned_layer_is_traversable_alongside_the_imported_one(tmp_path):
    store = LearnedEdgeStore(tmp_path / "learned.jsonl")
    store.append(ConceptEdge("rare syndrome", "causes", "unusual finding", 0.6, provenance=learned_provenance("c1", 3)))
    imported = [ConceptEdge("marfan syndrome", "has_phenotype", "aortic root dilation", 0.8)]

    graph, report = build_persistent_graph(imported, store)

    assert report.as_dict()["edges"] == 1
    assert [edge.target for edge in graph.edges_from("rare syndrome")] == ["unusual finding"]
    assert graph.edges_from("marfan syndrome"), "the imported layer is still there"


def test_build_persistent_graph_with_an_empty_store_returns_the_imported_layer(tmp_path):
    imported = [ConceptEdge("a", "causes", "b", 0.8)]
    graph, report = build_persistent_graph(imported, LearnedEdgeStore(tmp_path / "none.jsonl"))
    assert report.as_dict()["edges"] == 0
    assert graph.edges_from("a")


# --------------------------------------------------------------------------
# Candidate retrieval: the enumerator's missing half
# --------------------------------------------------------------------------


def test_conditions_are_found_from_findings_alone():
    """The gap this closes: MechanismEnumerator needs candidate conditions
    supplied, and nothing produced them from the case's findings."""
    report = retrieve_candidates(
        ["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"], differential_graph()
    )
    assert "sarcoidosis" in report.condition_names


def test_the_condition_touching_most_findings_ranks_first():
    """Breadth before strength: a condition linked to three of the presented
    findings is a better candidate than one linked to a single finding by a
    strong edge."""
    report = retrieve_candidates(
        ["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"], differential_graph()
    )
    assert report.condition_names[0] == "sarcoidosis"
    assert report.candidates[0].breadth == 3


def test_a_finding_is_never_proposed_as_a_diagnosis():
    """A first run of this retrieval ranked "night sweats" among the
    hypotheses -- a symptom reached two hops out via a shared disease.
    Direction, not relation-name matching, is what separates the disease
    side of an edge from the finding side."""
    report = retrieve_candidates(
        ["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"], differential_graph()
    )
    assert "night sweats" not in report.condition_names


def test_a_finding_the_graph_does_not_know_is_reported_not_dropped():
    """"The graph has never heard of this finding" is a fact worth
    surfacing -- it is exactly the coverage gap the density check downstream
    reasons about."""
    report = retrieve_candidates(["a finding nobody has recorded"], differential_graph())
    assert report.unresolved_findings == ["a finding nobody has recorded"]
    assert report.candidates == []


def test_a_finding_phrased_naturally_still_resolves():
    """Findings arrive phrased as a clinician writes them, not as graph node
    text -- the same resolution verify_mechanism needed."""
    report = retrieve_candidates(["the patient's hypercalcaemia"], differential_graph())
    assert "sarcoidosis" in report.condition_names


def test_excluded_conditions_are_not_returned():
    report = retrieve_candidates(
        ["bilateral hilar lymphadenopathy"], differential_graph(), exclude=["sarcoidosis"]
    )
    assert "sarcoidosis" not in report.condition_names


def test_truncation_is_reported_when_it_bites():
    graph = InMemoryConceptGraph.from_edges(
        [ConceptEdge(f"disease {i}", "manifests_as", "common finding", 0.5) for i in range(10)]
    )
    report = retrieve_candidates(["common finding"], graph, max_candidates=3)
    assert len(report.candidates) == 3
    assert report.truncated is True


def test_ordering_is_stable_across_runs():
    """A bench comparing runs to each other needs the order not to depend on
    dict iteration."""
    findings = ["bilateral hilar lymphadenopathy", "hypercalcaemia"]
    first = retrieve_candidates(findings, differential_graph()).condition_names
    second = retrieve_candidates(findings, differential_graph()).condition_names
    assert first == second


# --------------------------------------------------------------------------
# The two halves together: findings to ranked hypotheses, no manual input
# --------------------------------------------------------------------------


def test_the_full_chain_runs_from_findings_to_ranked_hypotheses():
    """What neither half could do alone: a case's findings produce a ranked
    differential with nobody supplying the candidate list by hand."""
    graph = differential_graph()
    findings = ["bilateral hilar lymphadenopathy", "hypercalcaemia", "erythema nodosum"]

    candidates = retrieve_candidates(findings, graph).condition_names
    outcome = MechanismEnumerator(graph=graph).run(findings, candidates)

    assert outcome.hypotheses
    assert outcome.hypotheses[0].condition == "sarcoidosis"


def test_the_full_chain_still_abstains_where_the_graph_cannot_support_a_conclusion():
    """Retrieval must not paper over a coverage gap by handing the enumerator
    candidates it has no real basis for."""
    graph = differential_graph()
    findings = ["periorbital purpura", "macroglossia"]

    candidates = retrieve_candidates(findings, graph).condition_names
    outcome = MechanismEnumerator(graph=graph).run(findings, candidates)

    assert outcome.mode == "knowledge_gap_questions"
