from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class VectorMemoryRecord:
    record_id: str
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    source: str = "unknown"
    learning_status: str = "candidate"

    def describe(self) -> dict:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "metadata": self.metadata,
            "learning_status": self.learning_status,
            "embedding_dim": len(self.embedding),
        }


@dataclass(slots=True)
class HashingEmbeddingModel:
    """Deterministic local embedding fallback for tests and offline research.

    This is not a replacement for a clinical embedding model. It provides a
    stable local vectorization contract so the memory layer can be exercised
    without network calls.
    """

    dimensions: int = 128

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            index = sum(ord(char) for char in token) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 6) for value in vector]


@dataclass(slots=True)
class InMemoryVectorStore:
    """Provider-neutral vector memory interface with an in-memory fallback.

    The interface is intentionally compatible with future Qdrant/Milvus/pgvector
    adapters while keeping the current prototype fully executable offline.
    """

    embedding_model: HashingEmbeddingModel = field(default_factory=HashingEmbeddingModel)
    backend: str = "in_memory_hashing_fallback"
    recommended_enterprise_backend: str = "qdrant_or_milvus_with_hybrid_search"
    records: dict[str, VectorMemoryRecord] = field(default_factory=dict)

    def upsert_text(self, record_id: str, text: str, metadata: dict | None = None, source: str = "unknown", learning_status: str = "candidate") -> dict:
        record = VectorMemoryRecord(
            record_id=record_id,
            text=text,
            embedding=self.embedding_model.embed(text),
            metadata=metadata or {},
            source=source,
            learning_status=learning_status,
        )
        self.records[record_id] = record
        return record.describe()

    def search(self, query: str, limit: int = 5, required_status: Iterable[str] | None = None) -> list[dict]:
        query_embedding = self.embedding_model.embed(query)
        statuses = set(required_status or [])
        scored = []
        for record in self.records.values():
            if statuses and record.learning_status not in statuses:
                continue
            score = self._cosine(query_embedding, record.embedding)
            scored.append({
                "record_id": record.record_id,
                "score": round(score, 6),
                "text": record.text,
                "metadata": record.metadata,
                "source": record.source,
                "learning_status": record.learning_status,
            })
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:limit]

    def promote(self, record_id: str, reason: str) -> dict:
        record = self.records[record_id]
        record.learning_status = "promoted"
        record.metadata = {**record.metadata, "promotion_reason": reason}
        return record.describe()

    def _cosine(self, first: list[float], second: list[float]) -> float:
        return sum(a * b for a, b in zip(first, second))

    def describe(self) -> dict:
        statuses: dict[str, int] = {}
        for record in self.records.values():
            statuses[record.learning_status] = statuses.get(record.learning_status, 0) + 1
        return {
            "backend": self.backend,
            "recommended_enterprise_backend": self.recommended_enterprise_backend,
            "record_count": len(self.records),
            "status_counts": statuses,
            "embedding_dimensions": self.embedding_model.dimensions,
        }
