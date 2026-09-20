"""Tests for FalkorLiteratureIndex: real persistence and graph-linked search
for LiteraturePassage, replacing the never-instantiated
PersistentJsonlVectorStore. Verified identical to LiteratureIndex's own
in-memory implementation on the same input -- same retrieval contract,
different store.
"""

import tempfile
from pathlib import Path

from melampo.memory.concept_paths import ConceptEdge, InMemoryConceptGraph
from melampo.memory.falkordb_connection import MODE_LITE, FalkorDBConfig
from melampo.memory.falkordb_literature import FalkorLiteratureIndex
from melampo.memory.literature_index import LiteratureIndex, LiteraturePassage

_GRAPH_EDGES = [
    ConceptEdge("Marfan syndrome", "has_phenotype", "Aortic root aneurysm", weight=0.9),
    ConceptEdge("Marfan syndrome", "has_phenotype", "Ectopia lentis", weight=0.85),
]

_PASSAGES = [
    LiteraturePassage(
        passage_id="p1", text="Aortic root aneurysm is a key feature of Marfan syndrome.",
        title="Marfan review", source_id="pmid:12345", year=2020, publication="NEJM",
    ),
    LiteraturePassage(
        passage_id="p2", text="Ectopia lentis and aortic root aneurysm co-occur in Marfan patients.",
        title="Ocular findings", source_id="pmid:67890", year=2021, publication="JAMA",
    ),
    LiteraturePassage(
        passage_id="p3", text="Unrelated passage about diabetes management.",
        title="Diabetes", source_id="pmid:11111", year=2019, publication="Lancet",
    ),
]


def _graph():
    return InMemoryConceptGraph.from_edges(_GRAPH_EDGES)


def _index(directory):
    config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(Path(directory) / "test.db"))
    index = FalkorLiteratureIndex.open(config=config, graph_name="test_literature")
    index.clear()
    return index


# --------------------------------------------------------------------------
# Equivalence with LiteratureIndex -- the property this whole change
# depends on: search() returns the same passages, in the same order
# --------------------------------------------------------------------------


def test_search_matches_the_in_memory_index_exactly():
    with tempfile.TemporaryDirectory() as directory:
        falkor = _index(directory)
        falkor.add_many(_PASSAGES, source_graph=_graph())
        falkor_hits = [r.passage.passage_id for r in falkor.search(["aortic root aneurysm"], limit=5)]

    in_memory = LiteratureIndex()
    in_memory.add_many(_PASSAGES)
    in_memory_hits = [r.passage.passage_id for r in in_memory.search(["aortic root aneurysm"], graph=_graph(), limit=5)]

    assert falkor_hits == in_memory_hits


def test_a_passage_mentioning_more_queried_concepts_ranks_first():
    with tempfile.TemporaryDirectory() as directory:
        falkor = _index(directory)
        falkor.add_many(_PASSAGES, source_graph=_graph())
        hits = falkor.search(["aortic root aneurysm", "ectopia lentis"], limit=5)

    assert hits[0].passage.passage_id == "p2"  # mentions both
    assert hits[0].breadth == 2


def test_an_unrelated_passage_is_never_returned():
    with tempfile.TemporaryDirectory() as directory:
        falkor = _index(directory)
        falkor.add_many(_PASSAGES, source_graph=_graph())
        hits = falkor.search(["aortic root aneurysm"], limit=5)

    assert all(hit.passage.passage_id != "p3" for hit in hits)


def test_search_with_no_concepts_returns_nothing():
    with tempfile.TemporaryDirectory() as directory:
        falkor = _index(directory)
        falkor.add_many(_PASSAGES, source_graph=_graph())
        assert falkor.search([], limit=5) == []


def test_limit_is_respected():
    with tempfile.TemporaryDirectory() as directory:
        falkor = _index(directory)
        falkor.add_many(_PASSAGES, source_graph=_graph())
        hits = falkor.search(["aortic root aneurysm"], limit=1)

    assert len(hits) == 1


# --------------------------------------------------------------------------
# checkable_only: the property that distinguishes a citable conjecture
# from an uncitable one
# --------------------------------------------------------------------------


def test_checkable_only_excludes_a_passage_with_no_verifiable_source_id():
    unverifiable = LiteraturePassage(
        passage_id="p4", text="Aortic root aneurysm mentioned here too.",
        title="No source", source_id="internal-note", year=None, publication=None,
    )
    with tempfile.TemporaryDirectory() as directory:
        falkor = _index(directory)
        falkor.add_many([unverifiable], source_graph=_graph())

        checkable_hits = falkor.search(["aortic root aneurysm"], checkable_only=True)
        all_hits = falkor.search(["aortic root aneurysm"], checkable_only=False)

    assert checkable_hits == []
    assert len(all_hits) == 1


# --------------------------------------------------------------------------
# Persistence: the actual problem this module exists to solve --
# PersistentJsonlVectorStore was built and tested but never instantiated
# anywhere in production, so literature never survived a restart
# --------------------------------------------------------------------------


def test_passages_survive_a_restart():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "test.db"
        config = FalkorDBConfig(mode=MODE_LITE, lite_db_path=str(path))
        first = FalkorLiteratureIndex.open(config=config, graph_name="restart_test")
        first.clear()
        first.add_many(_PASSAGES, source_graph=_graph())

        second = FalkorLiteratureIndex.open(config=config, graph_name="restart_test")
        hits = second.search(["aortic root aneurysm"], limit=5)

    assert len(second) == 3
    assert len(hits) == 2


def test_the_same_concept_from_multiple_passages_becomes_one_node_not_a_duplicate():
    with tempfile.TemporaryDirectory() as directory:
        falkor = _index(directory)
        falkor.add_many(_PASSAGES, source_graph=_graph())
        result = falkor._graph().query(
            "MATCH (c:Concept {norm: 'aortic root aneurysm'}) RETURN count(c)"
        )

    assert result.result_set[0][0] == 1


def test_adding_the_same_passage_twice_does_not_duplicate_it():
    """The uniqueness constraint's whole purpose -- verified directly, not
    just declared."""
    with tempfile.TemporaryDirectory() as directory:
        falkor = _index(directory)
        falkor.add_many([_PASSAGES[0]], source_graph=_graph())
        falkor.add_many([_PASSAGES[0]], source_graph=_graph())

    assert len(falkor) == 1


def test_clear_removes_every_passage_and_concept_link():
    with tempfile.TemporaryDirectory() as directory:
        falkor = _index(directory)
        falkor.add_many(_PASSAGES, source_graph=_graph())
        assert len(falkor) > 0
        falkor.clear()
        assert len(falkor) == 0
        assert falkor.search(["aortic root aneurysm"]) == []
