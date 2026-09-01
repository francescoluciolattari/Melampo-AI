"""Navigable context environment for RLM-on-Memory retrieval.

The environment exposes typed, provenance-carrying primitives over a case corpus
so a recursive retrieval strategy can navigate memory without ever receiving raw
text through an untracked path.

Two invariants are load-bearing:

1. Every returned fragment carries ``(document_id, char_start, char_end)``.
   Provenance is not optional metadata; a fragment without offsets cannot be
   emitted by this module.
2. Exploration coverage is measured, never assumed. An RLM removes the a priori
   lossy compression of fixed chunking and ``top_k`` truncation, but it does not
   remove information loss: what the root model never queries is never seen.
   The ``CoverageLedger`` makes that residual loss observable at runtime so the
   abstention layer can react to a confident conclusion drawn from a narrow
   exploration.

The module is dependency-free and performs no network access. Semantic search is
supplied by an injected callable so the environment stays provider-neutral, in
line with the repository contract/provider separation.
"""

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

SearchFn = Callable[[str, int], Sequence[dict[str, Any]]]
GraphExpandFn = Callable[[str, int], Sequence[dict[str, Any]]]


@dataclass(frozen=True)
class EnvironmentDocument:
    """A single addressable document in the case corpus."""

    document_id: str
    text: str
    source: str = "unknown"
    section: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.text)


@dataclass(frozen=True)
class Fragment:
    """A provenance-carrying slice of a document.

    ``char_start`` and ``char_end`` are absolute offsets into the source
    document text, which is what makes a claim verifiable down to the character.
    """

    document_id: str
    text: str
    char_start: int
    char_end: int
    source: str = "unknown"
    section: str | None = None
    page: int | None = None
    score: float | None = None

    def as_evidence(self, rank: int, focus: str) -> dict[str, Any]:
        """Render the fragment in the evidence shape used by the retrieval contract."""
        return {
            "record_id": f"{self.document_id}:{self.char_start}-{self.char_end}",
            "text": self.text,
            "source": self.source,
            "kind": "environment_fragment",
            "route": "rlm_environment",
            "focus": focus,
            "rank": rank,
            "grounding_score": self.score if self.score is not None else 0.0,
            "learning_status": "grounded",
            "provenance": {
                "document_id": self.document_id,
                "char_start": self.char_start,
                "char_end": self.char_end,
                "section": self.section,
                "page": self.page,
            },
        }


@dataclass
class CoverageLedger:
    """Track which portion of the corpus the retrieval strategy actually inspected.

    Coverage here is character-based rather than ``len(evidence) / top_k``. The
    RAG-era ratio penalises a strategy that explores widely and reports
    selectively, which is precisely the behaviour an RLM is expected to exhibit.
    """

    total_characters: int = 0
    _seen: dict[str, set[tuple[int, int]]] = field(default_factory=dict)
    queries_issued: list[str] = field(default_factory=list)
    llm_calls: int = 0

    def record_query(self, description: str) -> None:
        self.queries_issued.append(description)

    def record_llm_call(self) -> None:
        self.llm_calls += 1

    def record_span(self, document_id: str, char_start: int, char_end: int) -> None:
        if char_end <= char_start:
            return
        self._seen.setdefault(document_id, set()).add((char_start, char_end))

    def inspected_characters(self) -> int:
        total = 0
        for spans in self._seen.values():
            for start, end in _merge_spans(spans):
                total += end - start
        return total

    def coverage_ratio(self) -> float:
        if self.total_characters <= 0:
            return 0.0
        return min(1.0, self.inspected_characters() / self.total_characters)

    def snapshot(self) -> dict[str, Any]:
        return {
            "coverage_ratio": round(self.coverage_ratio(), 4),
            "inspected_characters": self.inspected_characters(),
            "total_characters": self.total_characters,
            "documents_touched": sorted(self._seen.keys()),
            "queries_issued": len(self.queries_issued),
            "llm_calls": self.llm_calls,
        }


def _merge_spans(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted(spans)
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


@dataclass
class ContextEnvironment:
    """Typed navigation surface over a case corpus.

    The environment deliberately does not expose a raw Python REPL over the case
    text. Generated code operates on these primitives instead, which keeps
    provenance mandatory, keeps the sandbox surface small, and keeps every
    inspection observable by the coverage ledger.
    """

    documents: dict[str, EnvironmentDocument] = field(default_factory=dict)
    search_fn: SearchFn | None = None
    graph_expand_fn: GraphExpandFn | None = None
    max_fragment_characters: int = 4000
    ledger: CoverageLedger = field(default_factory=CoverageLedger)

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[EnvironmentDocument],
        *,
        search_fn: SearchFn | None = None,
        graph_expand_fn: GraphExpandFn | None = None,
    ) -> "ContextEnvironment":
        indexed = {document.document_id: document for document in documents}
        environment = cls(documents=indexed, search_fn=search_fn, graph_expand_fn=graph_expand_fn)
        environment.ledger.total_characters = sum(document.length for document in indexed.values())
        return environment

    def describe(self) -> dict[str, Any]:
        """Inventory of the environment. Cheap, and does not count as inspection."""
        return {
            "document_count": len(self.documents),
            "total_characters": self.ledger.total_characters,
            "documents": [
                {
                    "document_id": document.document_id,
                    "source": document.source,
                    "section": document.section,
                    "page": document.page,
                    "characters": document.length,
                }
                for document in sorted(self.documents.values(), key=lambda item: item.document_id)
            ],
        }

    def list_documents(self) -> list[str]:
        return sorted(self.documents.keys())

    def grep(self, pattern: str, *, window: int = 240, limit: int = 20) -> list[Fragment]:
        """Case-insensitive literal search returning windowed, offset-anchored fragments."""
        if not pattern:
            return []
        self.ledger.record_query(f"grep:{pattern}")
        needle = pattern.lower()
        fragments: list[Fragment] = []
        for document_id in sorted(self.documents.keys()):
            document = self.documents[document_id]
            haystack = document.text.lower()
            cursor = 0
            while len(fragments) < limit:
                position = haystack.find(needle, cursor)
                if position < 0:
                    break
                start = max(0, position - window // 2)
                end = min(document.length, position + len(needle) + window // 2)
                fragments.append(self._fragment(document, start, end))
                cursor = position + len(needle)
            if len(fragments) >= limit:
                break
        return fragments

    def slice(self, document_id: str, char_start: int, char_end: int) -> Fragment:
        """Extract an explicit character range. Raises on unknown documents."""
        document = self.documents.get(document_id)
        if document is None:
            raise KeyError(f"unknown document_id: {document_id}")
        start = max(0, min(char_start, document.length))
        end = max(start, min(char_end, document.length))
        if end - start > self.max_fragment_characters:
            end = start + self.max_fragment_characters
        self.ledger.record_query(f"slice:{document_id}:{start}-{end}")
        return self._fragment(document, start, end)

    def search(self, query: str, limit: int = 5) -> list[Fragment]:
        """Semantic search delegated to the injected backend adapter.

        Returns an empty list when no backend is wired, so the environment stays
        executable in offline test runs without silently inventing evidence.
        """
        if self.search_fn is None or not query:
            return []
        self.ledger.record_query(f"search:{query}")
        fragments: list[Fragment] = []
        for hit in self.search_fn(query, limit):
            fragment = self._fragment_from_hit(hit)
            if fragment is not None:
                fragments.append(fragment)
        return fragments

    def graph_expand(self, object_key: str, depth: int = 1) -> list[dict[str, Any]]:
        """Follow typed relations in the semantic memory graph.

        Structure, not context width, is what gives a recursive strategy its
        navigation affordances, so graph traversal is a first-class primitive
        alongside text search.
        """
        if self.graph_expand_fn is None or not object_key:
            return []
        self.ledger.record_query(f"graph_expand:{object_key}:{depth}")
        return list(self.graph_expand_fn(object_key, max(1, depth)))

    def coverage(self) -> dict[str, Any]:
        return self.ledger.snapshot()

    def _fragment(self, document: EnvironmentDocument, start: int, end: int) -> Fragment:
        self.ledger.record_span(document.document_id, start, end)
        return Fragment(
            document_id=document.document_id,
            text=document.text[start:end],
            char_start=start,
            char_end=end,
            source=document.source,
            section=document.section,
            page=document.page,
        )

    def _fragment_from_hit(self, hit: dict[str, Any]) -> Fragment | None:
        document_id = hit.get("document_id") or hit.get("record_id")
        if not document_id:
            return None
        document = self.documents.get(str(document_id))
        if document is None:
            return None
        start = int(hit.get("char_start", 0) or 0)
        end = int(hit.get("char_end", 0) or 0)
        if end <= start:
            end = min(document.length, start + self.max_fragment_characters)
        fragment = self._fragment(document, start, end)
        score = hit.get("score")
        if score is None:
            return fragment
        return Fragment(
            document_id=fragment.document_id,
            text=fragment.text,
            char_start=fragment.char_start,
            char_end=fragment.char_end,
            source=fragment.source,
            section=fragment.section,
            page=fragment.page,
            score=float(score),
        )
