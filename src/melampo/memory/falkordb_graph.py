"""ConceptGraphView backed by FalkorDB instead of an in-Python edge list.

Nothing downstream changes for this: `concept_paths.py`'s own docstring
already called for exactly this -- "Adapter-neutral. InMemoryConceptGraph
keeps the logic testable offline; production wiring goes through the
semantic memory adapter's graph traversal." `ConceptGraphView` is a
`Protocol` with two methods, `edges_from` and `concepts`; every consumer
(`find_paths`, `retrieve_candidates`, `MechanismEnumerator`,
`rank_differential`) is already typed against that Protocol, not against
`InMemoryConceptGraph` directly -- confirmed by reading each before writing
this. This class satisfies the same Protocol from a real FalkorDB
connection; none of those callers needed a single line changed.

**Schema.** One Cypher relationship type, `CONCEPT_EDGE`, for every edge
regardless of its clinical relation (has_phenotype, causes_disease,
conjectured_association, ...): Cypher relationship types must be literal
in a query, not parameterised, which would force one UNWIND batch per
distinct relation string during a bulk load -- impractical against a
vocabulary this data-driven and this large. The real relation name is
instead a property on the edge (`relation`), read back exactly as
`InMemoryConceptGraph` would report it. Nodes are `(:Concept {norm: ...})`,
keyed by `normalise_concept()` -- the same normalisation
`InMemoryConceptGraph` indexes by -- so the two backends agree on identity
for the same input.

**Direction.** `edges_from()` matches undirected (`-[rel]-`), then decides
outgoing vs incoming by comparing the matched relationship's start node to
the queried concept -- outgoing edges are returned as stored; incoming
edges are returned with `relation` prefixed `inverse_`, matching
`InMemoryConceptGraph._rebuild_index`'s exact inverse-edge construction, so
a caller cannot tell the two backends apart from result shape alone.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .concept_paths import ConceptEdge, normalise_concept
from .falkordb_connection import FalkorDBConfig, connect

DEFAULT_GRAPH_NAME = "melampo_concepts"
_BATCH_SIZE = 5000


@dataclass
class FalkorConceptGraph:
    """A ConceptGraphView over a FalkorDB graph -- embedded (Lite) or remote (Service), same code either way."""

    connection: Any
    graph_name: str = DEFAULT_GRAPH_NAME
    _concepts_cache: set[str] | None = field(default=None, repr=False)

    @classmethod
    def open(cls, config: FalkorDBConfig | None = None, graph_name: str = DEFAULT_GRAPH_NAME) -> "FalkorConceptGraph":
        """Connect using the project's configured backend (data/falkordb_config.toml by default)."""
        return cls(connection=connect(config), graph_name=graph_name)

    def _graph(self) -> Any:
        return self.connection.select_graph(self.graph_name)

    def edges_from_many(self, concepts: Sequence[str], *, batch_size: int = 150) -> dict[str, list[ConceptEdge]]:
        """edges_from() for many concepts, chunked into bounded round trips.

        Built for exactly one purpose: retrieve_candidates' breadth-first
        traversal visits an entire frontier level before moving to the
        next, so its per-node edges_from() calls can be replaced with a
        handful of calls per level instead of one per node.

        **Chunked, not one unbounded call, for a reason found the hard way,
        not assumed.** A single UNWIND covering an entire frontier level
        was tried first and measured directly against the real graph: for
        a common, high-fan-out finding like "hepatomegaly", the level-1
        frontier alone is 1,396 concepts, and fetching all of their edges
        in one query took 3.4 seconds and returned over 100,000 edges --
        slower than the 533-round-trip version it was meant to replace, on
        every one of three real test cases (0.5x, 0.7x, and 0.2x the
        original speed). The bottleneck moved from round-trip count to
        result-set volume and serialisation, which does not disappear by
        merging queries -- it is bounded here to a few hundred concepts per
        call, trading some round trips back for a result set small enough
        to stay fast, without reintroducing the original one-call-per-node
        cost.
        """
        keys = [normalise_concept(concept) for concept in concepts if normalise_concept(concept)]
        if not keys:
            return {}
        grouped: dict[str, list[ConceptEdge]] = {key: [] for key in keys}
        graph = self._graph()
        for start in range(0, len(keys), batch_size):
            chunk = keys[start : start + batch_size]
            result = graph.query(
                "UNWIND $norms AS concept_norm "
                "MATCH (c:Concept {norm: concept_norm})-[rel:CONCEPT_EDGE]-(other) "
                "RETURN concept_norm, rel.relation, rel.weight, rel.lower, rel.upper, rel.provenance, "
                "rel.source_name, rel.target_name, startNode(rel).norm = c.norm AS is_outgoing",
                params={"norms": chunk},
            )
            for row in result.result_set:
                concept_norm, relation, weight, lower, upper, provenance, source_name, target_name, is_outgoing = row
                if is_outgoing:
                    edge = ConceptEdge(
                        source=source_name, relation=relation, target=target_name,
                        weight=weight, provenance=provenance, lower=lower, upper=upper,
                    )
                else:
                    edge = ConceptEdge(
                        source=target_name, relation=f"inverse_{relation}", target=source_name,
                        weight=weight, provenance=provenance, lower=lower, upper=upper,
                    )
                grouped[concept_norm].append(edge)
        return grouped

    def shortest_path_last_edges(self, start: str, *, max_hops: int) -> list[tuple[str, str, bool, int]]:
        """For every concept reachable from `start` within max_hops, the (candidate, relation,
        reached_by_reverse, hop_count) needed to apply retrieve_candidates' admissibility and
        ranking logic -- in one native, server-side traversal.

        The third real attempt at this problem, after two rejected on real
        evidence (ROADMAP.md, H3): per-node edges_from() calls (533 round
        trips, slow); bulk edges_from_many() per BFS level (worse on all
        three real test cases -- the bottleneck was data volume, not round
        trips, for high-fan-out concepts). algo.BFS was tried next and
        rejected on stronger grounds: verified directly against this real
        graph, it silently returned zero results for "aortic root
        aneurysm" -- a real, common concept, not an edge case -- while
        working correctly for others. Traced to two open FalkorDB issues
        (#2725, #2727, filed 2026-09-04) describing exactly this failure
        mode in the Rust engine when algo.BFS is composed with other Cypher
        clauses, and corroborated independently by a real, comparable
        project (getzep/graphiti) cataloguing 23 FalkorDB bugs, several
        describing the same silent-empty-result pattern under different
        conditions. Confirmed the bundled FalkorDBLite binary is built with
        the Rust engine (rustc strings throughout), the same engine those
        issues name -- not assumed, verified directly against the binary.

        This uses native Cypher variable-length path matching instead --
        `-[:CONCEPT_EDGE*1..N]-`, a core OpenCypher feature predating and
        unrelated to the algo.* procedure layer where the bugs above live
        -- verified directly on the same real graph, same real cases: 0.098s
        for the "hepatomegaly" hub case that made every batching attempt
        slower (H3), correct and non-empty for "aortic root aneurysm"
        where algo.BFS silently failed.

        Returns only the shortest path's last edge per candidate, not every
        edge of every node visited -- the reason this stays fast where bulk
        edges_from_many() did not: a candidate's admissibility depends only
        on the final edge of its shortest path (see retrieve_candidates'
        own comments), so nothing else needs to leave the database.
        `reached_by_reverse` says whether the candidate was arrived at via
        the edge's original source (True) or target (False), matching
        edges_from()'s own inverse-direction convention exactly -- the
        caller applies the same admissibility rule retrieve_candidates
        already has and already tests, unchanged. `hop_count` is the
        shortest path's length, for the same breadth-then-proximity
        ranking retrieve_candidates already applies.
        """
        key = normalise_concept(start)
        if not key:
            return []
        result = self._graph().query(
            "MATCH path = (s:Concept {norm: $norm})-[:CONCEPT_EDGE*1.." + str(max_hops) + "]-(candidate:Concept) "
            "WITH candidate, path, length(path) AS hop_count "
            "ORDER BY hop_count ASC "
            "WITH candidate, collect(path)[0] AS shortest, collect(hop_count)[0] AS shortest_hop_count "
            "WITH candidate, last(relationships(shortest)) AS edge, shortest_hop_count "
            "RETURN candidate.norm, edge.relation, edge.source_name, edge.target_name, shortest_hop_count",
            params={"norm": key},
        )
        out: list[tuple[str, str, bool, int]] = []
        for candidate_norm, relation, source_name, target_name, hop_count in result.result_set:
            reached_by_reverse = normalise_concept(source_name) == candidate_norm
            out.append((candidate_norm, relation, reached_by_reverse, int(hop_count)))
        return out

    def edges_from(self, concept: str) -> Sequence[ConceptEdge]:
        key = normalise_concept(concept)
        if not key:
            return []
        result = self._graph().query(
            "MATCH (c:Concept {norm: $norm})-[rel:CONCEPT_EDGE]-(other) "
            "RETURN rel.relation, rel.weight, rel.lower, rel.upper, rel.provenance, "
            "rel.source_name, rel.target_name, startNode(rel).norm = c.norm AS is_outgoing",
            params={"norm": key},
        )
        edges: list[ConceptEdge] = []
        for relation, weight, lower, upper, provenance, source_name, target_name, is_outgoing in result.result_set:
            if is_outgoing:
                edges.append(
                    ConceptEdge(
                        source=source_name, relation=relation, target=target_name,
                        weight=weight, provenance=provenance, lower=lower, upper=upper,
                    )
                )
            else:
                # The inverse direction, matching InMemoryConceptGraph._rebuild_index
                # exactly: source and target swapped, relation prefixed.
                edges.append(
                    ConceptEdge(
                        source=target_name, relation=f"inverse_{relation}", target=source_name,
                        weight=weight, provenance=provenance, lower=lower, upper=upper,
                    )
                )
        return edges

    def concepts(self) -> set[str]:
        """Every concept name in the graph, cached after the first call.

        Found necessary, not assumed: `resolve_concept()` (concept_paths.py)
        calls this once per finding via `mentioned_concepts()`, which is
        free against InMemoryConceptGraph (an in-memory set) but was
        measured costing ~0.12s per call here -- transferring 31,317
        concept names over the connection each time. With three findings,
        that alone was ~0.36s of hidden overhead masking the native
        traversal's real speed. Invalidated by clear() and load_edges(),
        the only two operations that change which concepts exist.
        """
        if self._concepts_cache is None:
            result = self._graph().query("MATCH (c:Concept) RETURN c.norm")
            self._concepts_cache = {row[0] for row in result.result_set}
        return self._concepts_cache

    def edge_count(self) -> int:
        result = self._graph().query("MATCH ()-[rel:CONCEPT_EDGE]->() RETURN count(rel)")
        return int(result.result_set[0][0])

    def ensure_index(self) -> None:
        """Create the index on Concept.norm, if it does not already exist.

        Not optional for any real load: without it, every MERGE in
        load_edges scans every existing node to check for a match, turning
        an O(n) bulk load into something far worse as the graph grows.
        Found the hard way -- a first version omitted this, and a 5,000-edge
        batch timed out on a graph that had already accumulated tens of
        thousands of nodes from earlier batches.
        """
        try:
            self._graph().query("CREATE INDEX FOR (c:Concept) ON (c.norm)")
        except Exception as error:
            if "already indexed" not in str(error).lower() and "already exists" not in str(error).lower():
                raise

    def load_edges(self, edges: Iterable[ConceptEdge], *, batch_size: int = _BATCH_SIZE) -> int:
        """Bulk-load edges in batches, MERGE-ing nodes so the same concept from
        multiple edges becomes one node, not a duplicate per edge.

        Returns the number of edges loaded. Batched via UNWIND rather than one
        query per edge -- verified necessary, not assumed: the real HPO import
        is 1.27 million edges, and a query per edge at any real per-query
        overhead would make loading it impractical. The index (ensure_index)
        is created first for the same reason, verified the hard way: without
        it this times out well before finishing.
        """
        self.ensure_index()
        batch: list[dict[str, Any]] = []
        loaded = 0
        graph = self._graph()
        for edge in edges:
            low, high = edge.bounds
            batch.append(
                {
                    "source_norm": normalise_concept(edge.source),
                    "target_norm": normalise_concept(edge.target),
                    "source_name": edge.source,
                    "target_name": edge.target,
                    "relation": edge.relation,
                    "weight": edge.weight,
                    "lower": low,
                    "upper": high,
                    "provenance": edge.provenance,
                }
            )
            if len(batch) >= batch_size:
                loaded += self._load_batch(graph, batch)
                batch = []
        if batch:
            loaded += self._load_batch(graph, batch)
        self._concepts_cache = None
        return loaded

    @staticmethod
    def _load_batch(graph: Any, batch: list[dict[str, Any]]) -> int:
        graph.query(
            "UNWIND $rows AS row "
            "MERGE (s:Concept {norm: row.source_norm}) "
            "MERGE (t:Concept {norm: row.target_norm}) "
            "CREATE (s)-[:CONCEPT_EDGE {"
            "relation: row.relation, weight: row.weight, lower: row.lower, upper: row.upper, "
            "provenance: row.provenance, source_name: row.source_name, target_name: row.target_name"
            "}]->(t)",
            params={"rows": batch},
        )
        return len(batch)

    def clear(self) -> None:
        """Delete every node and edge in this graph -- for tests and reloads, not production use."""
        self._graph().query("MATCH (n) DETACH DELETE n")
        self._concepts_cache = None


def build_falkor_graph(
    edges: Iterable[ConceptEdge],
    *,
    config: FalkorDBConfig | None = None,
    graph_name: str = DEFAULT_GRAPH_NAME,
    clear_first: bool = True,
) -> FalkorConceptGraph:
    """FalkorConceptGraph.open() plus a bulk load in one call -- the common case."""
    graph = FalkorConceptGraph.open(config=config, graph_name=graph_name)
    if clear_first:
        graph.clear()
    graph.load_edges(edges)
    return graph
