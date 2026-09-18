"""Tests for shortest_path_last_edges: native Cypher variable-length path
matching wired into retrieve_candidates, after two rejected attempts
(ROADMAP.md, H3) and a third option (FalkorDB's algo.BFS procedure)
rejected on direct evidence of unreliability, not just theory.
"""

import tempfile
from pathlib import Path

from melampo.memory.candidate_retrieval import retrieve_candidates
from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.falkordb_connection import MODE_LITE, FalkorDBConfig
from melampo.memory.falkordb_graph import build_falkor_graph

_SAMPLE_EDGES = [
    ConceptEdge("Marfan syndrome", "has_phenotype", "Aortic root aneurysm", weight=0.9, lower=0.85, upper=0.95),
    ConceptEdge("Marfan syndrome", "has_phenotype", "Ectopia lentis", weight=0.85, lower=0.8, upper=0.9),
    ConceptEdge("Marfan syndrome", "has_phenotype", "Arachnodactyly", weight=0.75, lower=0.7, upper=0.8),
    ConceptEdge("Loeys-Dietz syndrome", "has_phenotype", "Aortic root aneurysm", weight=0.9, lower=0.85, upper=0.95),
    ConceptEdge("SMAD3", "associated_gene", "Aneurysm-osteoarthritis syndrome", weight=1.0),
    ConceptEdge("SMAD3", "associated_gene", "Aortic root aneurysm", weight=0.6),
]


def _falkor_graph(directory):
    config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(Path(directory) / "test.db"))
    return build_falkor_graph(_SAMPLE_EDGES, config=config, graph_name="native_test")


# --------------------------------------------------------------------------
# shortest_path_last_edges: correctness of the raw primitive
# --------------------------------------------------------------------------


def test_finds_the_direct_disease_reached_by_reverse():
    with tempfile.TemporaryDirectory() as directory:
        graph = _falkor_graph(directory)
        results = graph.shortest_path_last_edges("aortic root aneurysm", max_hops=2)

    by_candidate = {r[0]: r for r in results}
    assert by_candidate["marfan syndrome"][1] == "has_phenotype"
    assert by_candidate["marfan syndrome"][2] is True  # reached_by_reverse
    assert by_candidate["marfan syndrome"][3] == 1  # hop_count


def test_a_gene_reached_by_reverse_is_included_in_raw_results():
    """The primitive returns every reachable concept, admissible or not --
    filtering is retrieve_candidates' job, verified separately below."""
    with tempfile.TemporaryDirectory() as directory:
        graph = _falkor_graph(directory)
        results = graph.shortest_path_last_edges("aortic root aneurysm", max_hops=2)

    by_candidate = {r[0]: r for r in results}
    assert "smad3" in by_candidate
    assert by_candidate["smad3"][1] == "associated_gene"
    assert by_candidate["smad3"][2] is True


def test_an_unknown_starting_concept_returns_an_empty_list():
    with tempfile.TemporaryDirectory() as directory:
        graph = _falkor_graph(directory)
        assert graph.shortest_path_last_edges("nonexistent concept", max_hops=2) == []


# --------------------------------------------------------------------------
# retrieve_candidates via the native path: exact equivalence with
# InMemoryConceptGraph on the same data, the property everything else here
# depends on
# --------------------------------------------------------------------------


def test_retrieve_candidates_matches_in_memory_graph_exactly():
    in_memory = InMemoryConceptGraph.from_edges(_SAMPLE_EDGES)
    with tempfile.TemporaryDirectory() as directory:
        falkor = _falkor_graph(directory)
        findings = ["Aortic root aneurysm", "Ectopia lentis", "Arachnodactyly"]
        in_memory_result = retrieve_candidates(findings, in_memory, max_candidates=10)
        falkor_result = retrieve_candidates(findings, falkor, max_candidates=10)

    in_memory_names = [(c.condition, c.nearest_hops, c.findings_linked) for c in in_memory_result.candidates]
    falkor_names = [(c.condition, c.nearest_hops, c.findings_linked) for c in falkor_result.candidates]
    assert in_memory_names == falkor_names


def test_a_gene_never_appears_as_a_candidate_via_the_native_path():
    """The specific admissibility rule this whole design protects: a gene
    reached by reverse traversal must never rank as a diagnosis."""
    with tempfile.TemporaryDirectory() as directory:
        falkor = _falkor_graph(directory)
        result = retrieve_candidates(["Aortic root aneurysm"], falkor, max_candidates=10)

    names = {c.condition for c in result.candidates}
    assert "smad3" not in names


def test_a_disease_reached_forward_from_a_gene_is_a_candidate_via_the_native_path():
    """The other side of the same rule: causes_disease traversed forward
    from a gene IS admissible, even though not reached by reverse."""
    edges = [
        ConceptEdge("BRCA1", "associated_gene", "hereditary breast cancer", weight=0.9),
        ConceptEdge("BRCA1", "causes_disease", "ovarian cancer syndrome", weight=0.8),
    ]
    with tempfile.TemporaryDirectory() as directory:
        config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(Path(directory) / "gene_test.db"))
        graph = build_falkor_graph(edges, config=config, graph_name="gene_forward_test")
        result = retrieve_candidates(["hereditary breast cancer"], graph, max_candidates=10)

    names = {c.condition for c in result.candidates}
    assert "ovarian cancer syndrome" in names
    assert "brca1" not in names


def test_max_hops_is_respected_via_the_native_path():
    with tempfile.TemporaryDirectory() as directory:
        falkor = _falkor_graph(directory)
        report = falkor.shortest_path_last_edges("aortic root aneurysm", max_hops=1)

    # At 1 hop, only the direct diseases -- not anything two hops away.
    assert all(hop_count == 1 for _, _, _, hop_count in report)


def test_exclude_list_is_respected_via_the_native_path():
    with tempfile.TemporaryDirectory() as directory:
        falkor = _falkor_graph(directory)
        result = retrieve_candidates(
            ["Aortic root aneurysm"], falkor, max_candidates=10, exclude=["Marfan syndrome"]
        )

    names = {c.condition for c in result.candidates}
    assert "marfan syndrome" not in names


# --------------------------------------------------------------------------
# concepts() caching: found necessary while verifying this against the real
# graph -- resolve_concept() calls it once per finding via
# mentioned_concepts(), free in memory but a real round trip against FalkorDB
# --------------------------------------------------------------------------


def test_concepts_is_cached_after_the_first_call(monkeypatch):
    with tempfile.TemporaryDirectory() as directory:
        graph = _falkor_graph(directory)
        calls = {"n": 0}
        real_graph_method = graph._graph

        def counting_graph():
            calls["n"] += 1
            return real_graph_method()

        graph.concepts()  # warm the cache first
        monkeypatch.setattr(graph, "_graph", counting_graph)
        graph.concepts()
        graph.concepts()

    assert calls["n"] == 0  # both calls served from cache, no query issued


def test_concepts_cache_is_invalidated_by_load_edges():
    with tempfile.TemporaryDirectory() as directory:
        graph = _falkor_graph(directory)
        before = graph.concepts()
        graph.load_edges([ConceptEdge("New Concept", "has_phenotype", "Another New Concept", weight=0.5)])
        after = graph.concepts()

    assert "new concept" in after
    assert "new concept" not in before


def test_concepts_cache_is_invalidated_by_clear():
    with tempfile.TemporaryDirectory() as directory:
        graph = _falkor_graph(directory)
        graph.concepts()  # warm the cache
        graph.clear()
        assert graph.concepts() == set()
