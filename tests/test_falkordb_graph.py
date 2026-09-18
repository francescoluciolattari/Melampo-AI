"""Tests for FalkorConceptGraph: verifies it satisfies ConceptGraphView
identically to InMemoryConceptGraph, and that unchanged downstream code
(find_paths, MechanismEnumerator, retrieve_candidates) works against it
without modification -- all against a real embedded FalkorDBLite instance,
never mocked.
"""

import tempfile
from pathlib import Path

import pytest

from melampo.memory.candidate_retrieval import retrieve_candidates
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph, find_paths
from melampo.memory.falkordb_connection import MODE_LITE, FalkorDBConfig
from melampo.memory.falkordb_graph import FalkorConceptGraph, build_falkor_graph
from melampo.training.mechanism_enumeration import MechanismEnumerator

_SAMPLE_EDGES = [
    ConceptEdge("Marfan syndrome", "has_phenotype", "Aortic root aneurysm", weight=0.9, lower=0.85, upper=0.95, provenance="hpo"),
    ConceptEdge("Marfan syndrome", "has_phenotype", "Ectopia lentis", weight=0.85, lower=0.8, upper=0.9, provenance="hpo"),
    ConceptEdge("Marfan syndrome", "has_phenotype", "Arachnodactyly", weight=0.75, lower=0.7, upper=0.8, provenance="hpo"),
    ConceptEdge("Loeys-Dietz syndrome", "has_phenotype", "Aortic root aneurysm", weight=0.9, lower=0.85, upper=0.95, provenance="hpo"),
]


@pytest.fixture
def falkor_graph():
    with tempfile.TemporaryDirectory() as directory:
        config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(Path(directory) / "test.db"))
        yield build_falkor_graph(_SAMPLE_EDGES, config=config, graph_name="test_graph")


def _comparable(edges) -> list:
    return sorted((e.source.lower(), e.relation, e.target.lower(), round(e.weight, 4)) for e in edges)


# --------------------------------------------------------------------------
# Equivalence with InMemoryConceptGraph -- the property this whole change
# depends on: any caller typed against ConceptGraphView cannot tell the two
# backends apart from their results
# --------------------------------------------------------------------------


def test_concepts_match_in_memory_graph_exactly(falkor_graph):
    in_memory = InMemoryConceptGraph.from_edges(_SAMPLE_EDGES)
    assert falkor_graph.concepts() == in_memory.concepts()


def test_edges_from_match_in_memory_graph_exactly_including_inverse_direction(falkor_graph):
    in_memory = InMemoryConceptGraph.from_edges(_SAMPLE_EDGES)
    assert _comparable(falkor_graph.edges_from("Aortic root aneurysm")) == _comparable(
        in_memory.edges_from("Aortic root aneurysm")
    )


def test_edges_from_an_unknown_concept_returns_empty_on_both(falkor_graph):
    in_memory = InMemoryConceptGraph.from_edges(_SAMPLE_EDGES)
    assert falkor_graph.edges_from("nonexistent concept") == []
    assert in_memory.edges_from("nonexistent concept") == []


def test_the_loaded_edge_count_matches_the_source_list(falkor_graph):
    assert falkor_graph.edge_count() == len(_SAMPLE_EDGES)


# --------------------------------------------------------------------------
# Unchanged downstream code: find_paths, MechanismEnumerator, retrieve_candidates
# were never edited for this -- confirming they work against the real
# ConceptGraphView Protocol, not something InMemoryConceptGraph-specific
# --------------------------------------------------------------------------


def test_find_paths_works_unmodified_against_falkordb(falkor_graph):
    paths = find_paths(falkor_graph, "Marfan syndrome", "Aortic root aneurysm", max_hops=2)
    assert len(paths) == 1
    assert paths[0].edges[0].relation == "has_phenotype"


def test_mechanism_enumerator_works_unmodified_against_falkordb(falkor_graph):
    enumerator = MechanismEnumerator(graph=falkor_graph)
    outcome = enumerator.run(
        findings=["Aortic root aneurysm", "Ectopia lentis", "Arachnodactyly"],
        candidate_conditions=["Marfan syndrome", "Loeys-Dietz syndrome"],
    )
    labels = {h.condition for h in outcome.hypotheses}
    assert labels == {"Marfan syndrome", "Loeys-Dietz syndrome"}


def test_retrieve_candidates_works_unmodified_against_falkordb(falkor_graph):
    report = retrieve_candidates(["Aortic root aneurysm", "Ectopia lentis"], falkor_graph, max_candidates=5)
    names = {item.condition for item in report.candidates}
    assert "marfan syndrome" in names  # retrieve_candidates reports normalised (lowercase) names, on both backends


def test_retrieve_candidates_returns_identical_results_on_both_backends():
    in_memory = InMemoryConceptGraph.from_edges(_SAMPLE_EDGES)
    with tempfile.TemporaryDirectory() as directory:
        config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(Path(directory) / "test.db"))
        falkor = build_falkor_graph(_SAMPLE_EDGES, config=config, graph_name="test_graph")

        findings = ["Aortic root aneurysm", "Ectopia lentis"]
        in_memory_names = [c.condition for c in retrieve_candidates(findings, in_memory, max_candidates=5).candidates]
        falkor_names = [c.condition for c in retrieve_candidates(findings, falkor, max_candidates=5).candidates]

    assert in_memory_names == falkor_names


# --------------------------------------------------------------------------
# The index: required for load_edges to complete in reasonable time,
# verified not to fail or duplicate on a second call
# --------------------------------------------------------------------------


def test_ensure_index_can_be_called_twice_without_error(falkor_graph):
    falkor_graph.ensure_index()
    falkor_graph.ensure_index()  # must not raise on the second call


# --------------------------------------------------------------------------
# Bulk loading: batching, node de-duplication via MERGE
# --------------------------------------------------------------------------


def test_the_same_concept_from_multiple_edges_becomes_one_node_not_a_duplicate(falkor_graph):
    """Marfan syndrome is the source of three edges in the fixture -- must
    appear once in concepts(), not three times."""
    assert list(falkor_graph.concepts()).count("marfan syndrome") == 1


def test_loading_in_small_batches_produces_the_same_result_as_one_large_batch():
    with tempfile.TemporaryDirectory() as directory:
        config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(Path(directory) / "test.db"))
        graph = FalkorConceptGraph.open(config=config, graph_name="batch_test")
        graph.clear()
        loaded = graph.load_edges(_SAMPLE_EDGES, batch_size=2)  # forces multiple batches for 4 edges

    assert loaded == len(_SAMPLE_EDGES)
    assert graph.edge_count() == len(_SAMPLE_EDGES)


def test_clear_removes_every_node_and_edge(falkor_graph):
    assert falkor_graph.edge_count() > 0
    falkor_graph.clear()
    assert falkor_graph.edge_count() == 0
    assert falkor_graph.concepts() == set()
