from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable


def _stable_id(text: str, namespace: str = "melampo") -> str:
    digest = hashlib.sha256(f"{namespace}:{text}".encode("utf-8")).hexdigest()
    return digest[:24]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [round(value / norm, 6) for value in vector]


def _text_embedding(text: str, dimensions: int = 128) -> list[float]:
    """Deterministic local fallback embedding.

    This is intentionally simple and dependency-free. Production deployments
    should replace it with a governed multimodal embedding provider while
    preserving the same document contract.
    """

    buckets = [0.0 for _ in range(dimensions)]
    for index, byte in enumerate(text.encode("utf-8", errors="ignore")):
        buckets[(byte + index) % dimensions] += ((byte % 31) + 1) / 31.0
    return _normalize(buckets)


@dataclass(slots=True)
class VectorMemoryRecord:
    record_id: str
    text: str
    dense_vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    modality: str = "text"
    source: str = "local"
    learning_status: str = "candidate"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def embedding(self) -> list[float]:
        return self.dense_vector

    def score_against(self, query_vector: list[float]) -> float:
        return round(sum(left * right for left, right in zip(self.dense_vector, query_vector)), 6)

    def describe(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "metadata": self.metadata,
            "learning_status": self.learning_status,
            "embedding_dim": len(self.dense_vector),
            "modality": self.modality,
            "updated_at": self.updated_at,
        }

    def as_evidence(self, score: float, rank: int) -> dict[str, Any]:
        return {
            "source": "vector_memory",
            "kind": "multimodal_rag_hit",
            "value": self.text[:500],
            "route": "post_training_memory_recall",
            "focus": self.metadata.get("focus", "multimodal_context"),
            "grounding_score": score,
            "rank": rank,
            "record_id": self.record_id,
            "modality": self.modality,
            "learning_status": self.learning_status,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class HashingEmbeddingModel:
    """Deterministic local embedding fallback for tests and offline research."""

    dimensions: int = 128

    def embed(self, text: str) -> list[float]:
        return _text_embedding(text=text, dimensions=self.dimensions)


@dataclass(slots=True)
class InMemoryVectorStore:
    """Provider-neutral vector memory substrate for RAG and continuous learning.

    Recommended production backend: Milvus/Zilliz because Melampo needs
    multimodal, multi-vector and hybrid search, metadata filtering, reranking,
    and scalable real-time upserts. The in-memory implementation is a safe
    dependency-free fallback used for tests, local research and air-gapped
    prototyping.
    """

    embedding_model: HashingEmbeddingModel = field(default_factory=HashingEmbeddingModel)
    backend: str = "local_in_memory"
    recommended_enterprise_backend: str = "milvus_or_zilliz_multi_vector_hybrid_search"
    collection_name: str = "melampo_multimodal_clinical_memory"
    records: dict[str, VectorMemoryRecord] = field(default_factory=dict)
    update_log: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def enterprise_default(cls) -> "InMemoryVectorStore":
        return cls(
            backend="milvus_or_zilliz_recommended_with_local_fallback",
            recommended_enterprise_backend="milvus_or_zilliz_multi_vector_hybrid_search",
            collection_name="melampo_multimodal_clinical_memory",
        )

    def upsert_text(self, record_id: str, text: str, metadata: dict | None = None, source: str = "unknown", learning_status: str = "candidate") -> dict:
        record = self.upsert(
            text=text,
            metadata={**(metadata or {}), "record_id": record_id},
            modality=str((metadata or {}).get("modality", "text")),
            source=source,
            learning_status=learning_status,
        )
        return record.describe()

    def upsert(self, text: str, metadata: dict[str, Any] | None = None, modality: str = "text", source: str = "local", learning_status: str = "candidate") -> VectorMemoryRecord:
        metadata = metadata or {}
        record_id = metadata.get("record_id") or _stable_id(text=text, namespace=self.collection_name)
        now = time.time()
        previous = self.records.get(record_id)
        record = VectorMemoryRecord(
            record_id=record_id,
            text=text,
            dense_vector=self.embedding_model.embed(text),
            metadata=metadata,
            modality=modality,
            source=source,
            learning_status=previous.learning_status if previous and previous.learning_status == "promoted" else learning_status,
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        self.records[record_id] = record
        self.update_log.append({
            "event": "upsert",
            "record_id": record_id,
            "modality": modality,
            "source": source,
            "learning_status": record.learning_status,
            "metadata_keys": sorted(metadata.keys()),
            "timestamp": now,
        })
        return record

    def upsert_many(self, documents: Iterable[dict[str, Any]]) -> list[VectorMemoryRecord]:
        return [
            self.upsert(
                text=str(document.get("text", "")),
                metadata=dict(document.get("metadata", {})),
                modality=str(document.get("modality", "text")),
                source=str(document.get("source", "local")),
                learning_status=str(document.get("learning_status", "candidate")),
            )
            for document in documents
        ]

    def search(self, query: str, limit: int = 5, required_status: Iterable[str] | None = None, filters: dict[str, Any] | None = None) -> list[dict]:
        filters = filters or {}
        query_vector = self.embedding_model.embed(query)
        statuses = set(required_status or [])
        scored = []
        for record in self.records.values():
            if statuses and record.learning_status not in statuses:
                continue
            if any(record.metadata.get(key) != value for key, value in filters.items()):
                continue
            score = record.score_against(query_vector)
            evidence = record.as_evidence(score=score, rank=0)
            evidence["text"] = record.text
            scored.append(evidence)
        scored.sort(key=lambda item: item["grounding_score"], reverse=True)
        for index, item in enumerate(scored[:limit]):
            item["rank"] = index + 1
        return scored[:limit]

    def search_with_metadata(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        hits = self.search(query=query, limit=top_k, filters=filters)
        return {
            "query": query,
            "top_k": top_k,
            "filters": filters or {},
            "hit_count": len(hits),
            "hits": hits,
            "backend": self.backend,
            "embedding_model": "local_deterministic_fallback",
        }

    def promote(self, record_id: str, reason: str) -> dict:
        record = self.records[record_id]
        record.learning_status = "promoted"
        record.metadata = {**record.metadata, "promotion_reason": reason}
        record.updated_at = time.time()
        self.update_log.append({"event": "promote", "record_id": record_id, "reason": reason, "timestamp": record.updated_at})
        return record.describe()

    def consolidate_case(self, case_payload: dict[str, Any], result: dict[str, Any] | None = None) -> dict[str, Any]:
        result = result or {}
        case_id = str(case_payload.get("case_id", "unknown_case"))
        text_parts = [case_id]
        for key in ["report_text", "ehr_text", "patient_complaints"]:
            value = case_payload.get(key)
            if value:
                text_parts.append(str(value))
        if result:
            top = result.get("coordinated", {}).get("differential", {}).get("hypotheses", [{}])[0]
            text_parts.append(f"top_hypothesis={top.get('label', 'none')}")
            text_parts.append(f"policy={result.get('coordinated', {}).get('policy', {})}")
        record = self.upsert(
            text="\n".join(text_parts),
            metadata={
                "record_id": f"case:{case_id}",
                "case_id": case_id,
                "focus": "multimodal_context",
                "memory_role": "post_training_case_trace",
            },
            modality="multimodal_case_trace",
            source="clinical_pipeline",
            learning_status="candidate",
        )
        return {"status": "consolidated", "record_id": record.record_id, "memory": self.describe()}

    def describe(self) -> dict:
        statuses: dict[str, int] = {}
        for record in self.records.values():
            statuses[record.learning_status] = statuses.get(record.learning_status, 0) + 1
        return {
            "backend": self.backend,
            "recommended_enterprise_backend": self.recommended_enterprise_backend,
            "collection_name": self.collection_name,
            "record_count": len(self.records),
            "status_counts": statuses,
            "embedding_dimensions": self.embedding_model.dimensions,
            "supports_real_time_updates": True,
            "supports_multimodal_metadata": True,
            "fallback_mode": "dependency_free_in_memory",
        }


VectorMemoryStore = InMemoryVectorStore
