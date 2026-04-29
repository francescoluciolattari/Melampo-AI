from dataclasses import dataclass, field

from .vector_memory import InMemoryVectorStore


@dataclass
class SemanticMemoryStore:
    documents: list = field(default_factory=list)
    vector_store: InMemoryVectorStore = field(default_factory=InMemoryVectorStore)

    def add_document(self, item: dict) -> None:
        self.documents.append(item)
        text = item.get("text") or item.get("report_text") or item.get("content") or str(item)
        record_id = item.get("id") or item.get("record_id") or f"doc-{len(self.documents)}"
        self.vector_store.upsert_text(
            record_id=record_id,
            text=text,
            metadata=item.get("metadata", {}),
            source=item.get("source", "semantic_memory"),
            learning_status=item.get("learning_status", "candidate"),
        )

    def query(self, limit: int = 5) -> list:
        return self.documents[:limit]

    def semantic_search(self, query: str, limit: int = 5, promoted_only: bool = False) -> list:
        required_status = ["promoted"] if promoted_only else None
        return self.vector_store.search(query=query, limit=limit, required_status=required_status)

    def describe(self) -> dict:
        return {
            "document_count": len(self.documents),
            "vector_store": self.vector_store.describe(),
        }
