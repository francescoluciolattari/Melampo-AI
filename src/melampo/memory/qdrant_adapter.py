from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .vector_memory import HashingEmbeddingModel, InMemoryVectorStore


@dataclass(slots=True)
class QdrantCollectionSpec:
    """Qdrant-compatible collection contract for future hybrid clinical memory."""

    collection_name: str = "melampo_clinical_memory"
    dense_vector_name: str = "text_dense"
    sparse_vector_name: str = "text_sparse"
    image_vector_name: str = "image_dense"
    distance: str = "Cosine"
    dense_dimensions: int = 128

    def describe(self) -> dict:
        return {
            "collection_name": self.collection_name,
            "dense_vector_name": self.dense_vector_name,
            "sparse_vector_name": self.sparse_vector_name,
            "image_vector_name": self.image_vector_name,
            "distance": self.distance,
            "dense_dimensions": self.dense_dimensions,
            "supports_named_vectors": True,
            "supports_hybrid_search": True,
            "supports_payload_filtering": True,
        }


@dataclass(slots=True)
class QdrantVectorMemoryAdapter:
    """Qdrant-compatible adapter with offline fallback.

    This class prepares Qdrant-style collection/upsert/query payloads while
    delegating execution to an in-memory fallback unless a real Qdrant client is
    attached later. It keeps tests deterministic and avoids hidden network
    dependencies.
    """

    spec: QdrantCollectionSpec = field(default_factory=QdrantCollectionSpec)
    embedding_model: HashingEmbeddingModel = field(default_factory=HashingEmbeddingModel)
    fallback_store: InMemoryVectorStore = field(default_factory=InMemoryVectorStore)
    mode: str = "offline_contract"

    def collection_schema(self) -> dict:
        return {
            "collection_name": self.spec.collection_name,
            "vectors": {
                self.spec.dense_vector_name: {
                    "size": self.spec.dense_dimensions,
                    "distance": self.spec.distance,
                },
                self.spec.image_vector_name: {
                    "size": self.spec.dense_dimensions,
                    "distance": self.spec.distance,
                },
            },
            "sparse_vectors": {
                self.spec.sparse_vector_name: {},
            },
        }

    def build_point(self, record_id: str, text: str, metadata: dict | None = None, source: str = "unknown") -> dict:
        dense = self.embedding_model.embed(text)
        sparse = self._sparse_from_text(text)
        return {
            "id": record_id,
            "vector": {
                self.spec.dense_vector_name: dense,
                self.spec.sparse_vector_name: sparse,
            },
            "payload": {
                "text": text,
                "metadata": metadata or {},
                "source": source,
                "memory_type": "clinical_semantic_memory",
            },
        }

    def upsert_text(self, record_id: str, text: str, metadata: dict | None = None, source: str = "unknown", learning_status: str = "candidate") -> dict:
        point = self.build_point(record_id=record_id, text=text, metadata=metadata, source=source)
        fallback = self.fallback_store.upsert_text(
            record_id=record_id,
            text=text,
            metadata={**(metadata or {}), "qdrant_point": point},
            source=source,
            learning_status=learning_status,
        )
        return {
            "mode": self.mode,
            "collection": self.spec.collection_name,
            "point": point,
            "fallback_record": fallback,
            "status": "stored_in_fallback_qdrant_contract_prepared",
        }

    def search(self, query: str, limit: int = 5, required_status: Iterable[str] | None = None) -> list[dict]:
        return self.fallback_store.search(query=query, limit=limit, required_status=required_status)

    def build_hybrid_query(self, query: str, limit: int = 5, filters: dict | None = None) -> dict:
        return {
            "collection_name": self.spec.collection_name,
            "query": {
                "dense": {
                    "name": self.spec.dense_vector_name,
                    "vector": self.embedding_model.embed(query),
                },
                "sparse": {
                    "name": self.spec.sparse_vector_name,
                    "vector": self._sparse_from_text(query),
                },
                "fusion": "reciprocal_rank_fusion_candidate",
            },
            "limit": limit,
            "filters": filters or {},
            "execution": "not_sent_offline_contract",
        }

    def describe(self) -> dict:
        return {
            "adapter": "qdrant_compatible_vector_memory_adapter",
            "mode": self.mode,
            "spec": self.spec.describe(),
            "fallback": self.fallback_store.describe(),
            "network_dependency": False,
        }

    def _sparse_from_text(self, text: str) -> dict:
        counts: dict[int, float] = {}
        for token in text.lower().split():
            index = sum(ord(char) for char in token) % 100000
            counts[index] = counts.get(index, 0.0) + 1.0
        indexes = sorted(counts.keys())
        values = [counts[index] for index in indexes]
        return {"indexes": indexes, "values": values}
