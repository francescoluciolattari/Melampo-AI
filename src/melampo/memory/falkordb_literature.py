"""LiteratureIndex, persisted and graph-linked in FalkorDB instead of an in-memory list.

Two problems solved together, both verified real before building this.
First: `PersistentJsonlVectorStore` (literature_persistence.py) was built
and tested this session but never once instantiated anywhere in
production -- literature fetched from Europe PMC, ClinicalTrials.gov,
DailyMed or the EMA PMS never survived a restart. Second: `LiteratureIndex.search()`
re-scans every stored passage's raw text on every call
(`mentioned_concepts(passage.text, graph, ...)` inside the search loop),
an O(n) cost repeated per query that grows without bound as the corpus
grows.

**What is deliberately unchanged.** `LiteratureIndex`'s own decision to
match on concepts the graph already names, not embeddings, is not
revisited here -- it was reached by measuring what latent similarity costs
in a clinical setting (a heart-failure query retrieving "acute coronary
syndrome" because it is nearby in embedding space, not because it answers
the question), and FalkorDB making a vector index convenient is not a
reason to re-open a decision that was reasoned through, not defaulted
into. `checkable_only`'s default, `is_independently_checkable`'s
definition, and the breadth-first ranking are all identical to
`literature_index.py` -- this is the same retrieval contract, backed by a
different store.

**What changes, and why it is still the same answer, not a different
one.** `mentioned_concepts()` now runs once per passage, at ingestion
time, instead of once per passage on every search call -- the matches
themselves are identical (same function, same graph, same text), stored
as real `MENTIONED_IN` edges from `(:Concept)` to `(:LiteraturePassage)`
rather than recomputed. `search()` becomes a graph traversal from the
queried concepts outward, not a scan of the whole passage list.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .concept_paths import ConceptGraphView, mentioned_concepts, normalise_concept
from .falkordb_connection import FalkorDBConfig, connect
from .literature_index import LiteraturePassage, RetrievedPassage

DEFAULT_LITERATURE_GRAPH_NAME = "melampo_literature"


@dataclass
class FalkorLiteratureIndex:
    """LiteratureIndex's own retrieval contract, backed by FalkorDB: persistent, graph-linked, not embedding-matched."""

    connection: Any
    graph_name: str = DEFAULT_LITERATURE_GRAPH_NAME

    @classmethod
    def open(cls, config: FalkorDBConfig | None = None, graph_name: str = DEFAULT_LITERATURE_GRAPH_NAME) -> "FalkorLiteratureIndex":
        return cls(connection=connect(config), graph_name=graph_name)

    def _graph(self) -> Any:
        return self.connection.select_graph(self.graph_name)

    def ensure_indexes(self) -> None:
        """A uniqueness constraint on passage_id and an index on Concept.norm.

        Found necessary in a real published biomedical GraphRAG project's
        own account, not assumed: "Constraints ensure uniqueness on
        critical properties like Paper.pmid... Without them, duplicate
        entries would quietly corrupt the graph." The same MERGE-heavy
        ingestion pattern used for the concept graph (falkordb_graph.py)
        applies here.

        FalkorDB does not support Cypher's `CREATE CONSTRAINT` syntax --
        verified directly, not assumed from Cypher familiarity: it raises
        "Invalid constraint command use the GRAPH.CONSTRAINT command
        instead". The Python client exposes that as
        `create_node_unique_constraint(label, *properties)`.
        """
        graph = self._graph()
        try:
            graph.create_node_unique_constraint("LiteraturePassage", "passage_id")
        except Exception as error:
            if "already" not in str(error).lower():
                raise
        try:
            graph.query("CREATE INDEX FOR (c:Concept) ON (c.norm)")
        except Exception as error:
            if "already" not in str(error).lower():
                raise

    def add(self, passage: LiteraturePassage, source_graph: ConceptGraphView) -> None:
        self.add_many([passage], source_graph=source_graph)

    def add_many(self, passages: Iterable[LiteraturePassage], source_graph: ConceptGraphView) -> int:
        """Store passages and their concept links in one pass.

        `source_graph` is the same ConceptGraphView the rest of this
        project's reasoning already uses (InMemoryConceptGraph or
        FalkorConceptGraph) -- mentioned_concepts() needs it to find which
        of the graph's own concepts a passage's text names, the identical
        computation literature_index.py's search() already performs, just
        moved earlier.
        """
        self.ensure_indexes()
        graph = self._graph()
        passage_rows = []
        concept_links: list[dict[str, Any]] = []
        count = 0
        for passage in passages:
            count += 1
            passage_rows.append(
                {
                    "passage_id": passage.passage_id,
                    "text": passage.text,
                    "title": passage.title,
                    "source_id": passage.source_id,
                    "year": passage.year,
                    "publication": passage.publication,
                    "is_independently_checkable": passage.is_independently_checkable,
                }
            )
            for concept in mentioned_concepts(passage.text, source_graph, max_results=20):
                concept_links.append({"passage_id": passage.passage_id, "concept_norm": normalise_concept(concept)})
        if passage_rows:
            graph.query(
                "UNWIND $rows AS row "
                "MERGE (p:LiteraturePassage {passage_id: row.passage_id}) "
                "SET p.text = row.text, p.title = row.title, p.source_id = row.source_id, "
                "p.year = row.year, p.publication = row.publication, "
                "p.is_independently_checkable = row.is_independently_checkable",
                params={"rows": passage_rows},
            )
        if concept_links:
            graph.query(
                "UNWIND $rows AS row "
                "MATCH (p:LiteraturePassage {passage_id: row.passage_id}) "
                "MERGE (c:Concept {norm: row.concept_norm}) "
                "MERGE (c)-[:MENTIONED_IN]->(p)",
                params={"rows": concept_links},
            )
        return count

    def __len__(self) -> int:
        result = self._graph().query("MATCH (p:LiteraturePassage) RETURN count(p)")
        return int(result.result_set[0][0])

    def search(
        self, concepts: Sequence[str], *, limit: int = 5, checkable_only: bool = True
    ) -> list[RetrievedPassage]:
        """Passages mentioning these concepts, most relevant first -- a graph traversal, not a text scan.

        Same ranking rule as literature_index.py: breadth (how many of the
        queried concepts a passage matches) before anything else, ties
        broken by passage_id for stable ordering across runs.
        """
        wanted = [normalise_concept(item) for item in concepts if normalise_concept(item)]
        if not wanted:
            return []
        result = self._graph().query(
            "UNWIND $norms AS concept_norm "
            "MATCH (c:Concept {norm: concept_norm})-[:MENTIONED_IN]->(p:LiteraturePassage) "
            "RETURN p.passage_id, p.text, p.title, p.source_id, p.year, p.publication, "
            "collect(DISTINCT concept_norm) AS matched",
            params={"norms": wanted},
        )
        hits: list[RetrievedPassage] = []
        for passage_id, text, title, source_id, year, publication, matched in result.result_set:
            passage = LiteraturePassage(
                passage_id=passage_id, text=text, title=title, source_id=source_id, year=year, publication=publication
            )
            if checkable_only and not passage.is_independently_checkable:
                continue
            hits.append(RetrievedPassage(passage=passage, matched_concepts=tuple(sorted(matched))))
        hits.sort(key=lambda item: (-len(item.matched_concepts), item.passage.passage_id))
        return hits[:limit]

    def clear(self) -> None:
        """Delete every passage and concept link -- for tests and reloads, not production use."""
        self._graph().query("MATCH (n) DETACH DELETE n")
