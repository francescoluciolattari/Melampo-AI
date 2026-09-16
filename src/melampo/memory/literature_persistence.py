"""Give LiteratureIndex the persistence it never had, using the vector store this project already built.

An audit found `LiteratureIndex` had no persistence at all -- it lived only
in process memory and vanished on restart. This should not have been a
surprise: `vector_memory.py` is a complete, provider-neutral vector store
(durable JSONL persistence, Weaviate named as the recommended production
backend, a documented object-property schema) that was never instantiated
anywhere in production -- the same "built, never connected" pattern this
project has now found and fixed seven times over. It was exactly the
persistence layer literature needed, sitting unused the whole time.

**What this module deliberately does NOT do: change how relevance is
judged.** `PersistentJsonlVectorStore.search` ranks by embedding cosine
similarity through a `HashingEmbeddingModel` -- and `literature_index.py`'s
own design decision, stated directly in its docstring, was to search by
concept match instead, for a reason this project measured rather than
assumed: a RAG system asked about heart failure retrieves "acute coronary
syndrome" because it sits close in latent space, not because it answers the
question. Routing literature retrieval through vector search now would
undo that decision by accident, through a persistence change nobody meant
as a relevance change.

So the vector store here is storage, not a search engine: passages persist
in it and survive a restart, and `LiteratureIndex.search` still does its own
concept matching over whatever was loaded. The embedding computed for each
upserted record exists only because the store's API requires one -- nothing
here reads it back for ranking.

**One store, both PubMed and ClinicalTrials.gov content.** Both connectors
already produce `LiteraturePassage` objects in one shape; there was never a
reason for them to persist through two different mechanisms, and this
module treats them identically.
"""

from typing import Any

from .literature_index import LiteratureIndex, LiteraturePassage
from .vector_memory import PersistentJsonlVectorStore

LEARNING_STATUS_LITERATURE = "promoted"  # literature is retained, never a "candidate" awaiting review


def passage_to_metadata(passage: LiteraturePassage) -> dict[str, Any]:
    """Every LiteraturePassage field the vector store's metadata dict must round-trip losslessly."""
    return {
        "passage_id": passage.passage_id,
        "title": passage.title,
        "source_id": passage.source_id,
        "year": passage.year,
        "publication": passage.publication,
        "record_id": passage.passage_id,
    }


def metadata_to_passage(text: str, metadata: dict[str, Any]) -> LiteraturePassage:
    return LiteraturePassage(
        passage_id=str(metadata.get("passage_id", "")),
        text=text,
        title=str(metadata.get("title", "")),
        source_id=str(metadata.get("source_id", "")),
        year=metadata.get("year"),
        publication=metadata.get("publication"),
    )


def persist_passage(store: PersistentJsonlVectorStore, passage: LiteraturePassage) -> None:
    """Write one passage into the vector store.

    `upsert` deduplicates by `record_id` (set here to the passage's own id),
    so calling this twice for the same passage updates it in place rather
    than duplicating it -- the same passage discovered again by a later
    refresh does not grow the store.
    """
    store.upsert(
        text=passage.text,
        metadata=passage_to_metadata(passage),
        source=_origin_for(passage.source_id),
        learning_status=LEARNING_STATUS_LITERATURE,
    )


def persist_all(store: PersistentJsonlVectorStore, index: LiteratureIndex) -> int:
    """Write every passage currently in an index. Returns how many were written."""
    for passage in index.passages:
        persist_passage(store, passage)
    return len(index.passages)


def load_literature_index(store: PersistentJsonlVectorStore) -> LiteratureIndex:
    """Rebuild a LiteratureIndex from everything persisted in the vector store.

    The read path for the restart problem this module exists to solve: a
    process that starts fresh calls this once and has every passage a
    previous process ever persisted, searchable through the same concept
    matching as before -- nothing about `LiteratureIndex.search`'s behaviour
    changes because its passages now come from disk instead of from a
    connector call made moments ago.
    """
    index = LiteratureIndex()
    for record in store.records.values():
        index.add(metadata_to_passage(record.text, record.metadata))
    return index


def _origin_for(source_id: str) -> str:
    """A short label for the vector store's own `source` field, from the passage's own id prefix."""
    lowered = source_id.lower()
    if lowered.startswith(("pmid:", "pmcid:", "doi:")):
        return "europe_pmc"
    if lowered.startswith("nct:"):
        return "clinical_trials"
    return "literature"
