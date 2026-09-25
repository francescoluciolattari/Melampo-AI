"""find_paths() prunes from the destination's side -- same paths, same order.

The pruning exists because of a measured defect: on the real graph a
disease known only through gene associations (two edges, both to genes)
made find_paths enumerate the whole three-hop tree through genes'
thousands of phenotype edges -- up to 134 seconds for a single
finding/candidate pair, over two and a half minutes for one pipeline run.
The fix must not change a single result: `_unpruned` below is the previous
implementation, kept verbatim as the reference, and every test compares
against it. (On the real graph the same comparison was run pair by pair
while building the fix: 119/119 identical on 60 sampled pairs, plus the
three slowest pairs -- 25 s, 16 s and 134.6 s before, under 0.01 s after --
also identical.)
"""

import random

import pytest

from melampo.memory.concept_paths import (
    ConceptEdge,
    ConceptPath,
    InMemoryConceptGraph,
    _ranked,
    find_paths,
    normalise_concept,
)


def _unpruned(graph, start, end, *, max_hops=3, min_edge_weight=0.0, max_paths=8, max_gap_edges=None):
    """The implementation before pruning, verbatim -- the reference every test compares with."""
    start_key, end_key = normalise_concept(start), normalise_concept(end)
    if not start_key or not end_key or start_key == end_key:
        return []
    found = []
    frontier = [(start_key, (), frozenset({start_key}))]
    for _ in range(max(1, max_hops)):
        next_frontier = []
        for concept, path, visited in frontier:
            for edge in graph.edges_from(concept):
                if edge.weight < min_edge_weight:
                    continue
                target_key = normalise_concept(edge.target)
                if target_key in visited:
                    continue
                extended = path + (edge,)
                if max_gap_edges is not None and sum(1 for item in extended if item.is_gap) > max_gap_edges:
                    continue
                if target_key == end_key:
                    found.append(ConceptPath(edges=extended))
                    if len(found) >= max_paths:
                        return _ranked(found)
                    continue
                next_frontier.append((target_key, extended, visited | {target_key}))
        frontier = next_frontier
        if not frontier:
            break
    return _ranked(found)


def _describe(paths):
    return [path.describe() for path in paths]


def _random_graph(seed: int, nodes: int, edges: int) -> InMemoryConceptGraph:
    rng = random.Random(seed)
    names = [f"c{i}" for i in range(nodes)]
    built = []
    for _ in range(edges):
        source, target = rng.sample(names, 2)
        weight = rng.choice([1.0, 0.9, 0.6, 0.3, 0.05])
        lower, upper = (weight, weight) if rng.random() < 0.8 else (0.0, 1.0)  # some gap edges
        built.append(ConceptEdge(source=source, relation=rng.choice(["has_phenotype", "associated_gene"]),
                                 target=target, weight=weight, lower=lower, upper=upper))
    return InMemoryConceptGraph.from_edges(built)


@pytest.mark.parametrize("seed", range(40))
def test_identical_to_the_unpruned_search_on_random_graphs(seed):
    rng = random.Random(1000 + seed)
    graph = _random_graph(seed, nodes=rng.randint(8, 40), edges=rng.randint(10, 160))
    names = sorted(graph.concepts())
    for _ in range(25):
        start, end = rng.choice(names), rng.choice(names)
        options = {
            "max_hops": rng.choice([1, 2, 3, 4]),
            "min_edge_weight": rng.choice([0.0, 0.1, 0.5]),
            "max_paths": rng.choice([1, 3, 8, 32]),
            "max_gap_edges": rng.choice([None, 0, 1]),
        }
        assert _describe(find_paths(graph, start, end, **options)) == _describe(_unpruned(graph, start, end, **options)), (start, end, options)


class _CountingGraph:
    """Counts edges_from() calls -- the unit of work find_paths spends."""

    def __init__(self, graph):
        self.graph, self.calls = graph, 0

    def edges_from(self, concept):
        self.calls += 1
        return self.graph.edges_from(concept)

    def concepts(self):
        return self.graph.concepts()


def _hub_graph():
    """The real defect's shape: a finding linked to many diseases, each to many
    phenotypes and genes, and a destination reachable only through one gene."""
    edges = []
    for d in range(40):
        edges.append(ConceptEdge(source=f"disease{d}", relation="has_phenotype", target="finding"))
        for p in range(25):
            edges.append(ConceptEdge(source=f"disease{d}", relation="has_phenotype", target=f"phen{d}_{p}"))
        edges.append(ConceptEdge(source=f"gene{d}", relation="causes", target=f"disease{d}"))
    edges.append(ConceptEdge(source="gene7", relation="causes", target="gene_only_disease"))
    return InMemoryConceptGraph.from_edges(edges)


def test_a_poorly_connected_destination_no_longer_costs_the_whole_neighbourhood():
    graph = _hub_graph()
    naive, pruned = _CountingGraph(graph), _CountingGraph(graph)
    before = _unpruned(naive, "finding", "gene_only_disease")
    after = find_paths(pruned, "finding", "gene_only_disease")
    assert _describe(after) == _describe(before)
    assert _describe(after) == ["finding -[inverse_has_phenotype]-> disease7 -[inverse_causes]-> gene7 -[causes]-> gene_only_disease"]
    # The unpruned search expands every one of the 40 diseases' ~26 neighbours;
    # the pruned one only what can still reach the destination.
    assert naive.calls > 1000
    assert pruned.calls < 20


def test_an_unreachable_destination_returns_nothing_at_once():
    graph = _CountingGraph(_hub_graph())
    edges = list(graph.graph.edges) + [ConceptEdge(source="island_a", relation="r", target="island_b")]
    graph = _CountingGraph(InMemoryConceptGraph.from_edges(edges))
    assert find_paths(graph, "finding", "island_b") == []
    assert graph.calls < 10
