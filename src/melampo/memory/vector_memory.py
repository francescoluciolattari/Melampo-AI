from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from .learning_status import normalize_learning_status, validate_learning_transition


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
    preserving the same document/object contract.
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
            "kind": "object_property_rag_hit",
            "value": self.text[:500],
            "route": "post_training_semantic_memory_recall",
            "focus": self.metadata.get("focus", "multimodal_context"),
            "grounding_score": score,
            "rank": rank,
            "record_id": self.record_id,
            "modality": self.modality,
            "learning_status": self.learning_status,
            "ontology_refs": self.metadata.get("ontology_refs", []),
            "relations": self.metadata.get("relations", []),
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
    """Provider-neutral semantic vector memory for RAG and continuous learning.

    Recommended production backend: Weaviate. Melampo prioritizes semantic
    knowledge, ontology-aware object-property relations, clinical context, and
    multimodal objects where text, image vectors, patient properties and
    SNOMED-like hierarchy references must remain connected in one searchable
    object graph. The in-memory implementation is a safe dependency-free
    fallback used for tests, local research and air-gapped prototyping.
    """

    embedding_model: HashingEmbeddingModel = field(default_factory=HashingEmbeddingModel)
    backend: str = "local_in_memory"
    recommended_enterprise_backend: str = "weaviate_object_property_semantic_graph_rag"
    collection_name: str = "melampo_semantic_clinical_memory"
    records: dict[str, VectorMemoryRecord] = field(default_factory=dict)
    update_log: list[dict[str, Any]] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    @classmethod
    def enterprise_default(cls) -> "InMemoryVectorStore":
        return cls(
            backend="weaviate_recommended_with_local_fallback",
            recommended_enterprise_backend="weaviate_object_property_semantic_graph_rag",
            collection_name="melampo_semantic_clinical_memory",
        )

    def ontology_schema_hint(self) -> dict[str, Any]:
        """Return the target Weaviate-style object-property schema concept."""
        return {
            "backend": "Weaviate",
            "classes": {
                "Symptom": ["name", "description", "snomed_code", "hasFinding", "suggestsPathology"],
                "Pathology": ["name", "description", "snomed_code", "hasSymptom", "hasImagingPattern", "hasRiskFactor"],
                "ClinicalCase": ["case_id", "demographics", "hasSymptom", "hasReport", "hasImage", "hasDifferential"],
                "ImagingStudy": ["study_id", "modality", "image_vector", "hasFinding", "belongsToCase"],
                "VisualConcept": ["name", "ontology_refs", "hasImprint", "supportsFinding", "supportsPathology"],
                "VisualRecognitionImprint": ["semantic_concept", "matrix_signature_hash", "recognition_matrix_vector", "variantOf", "derivedFromStudy"],
                "ClinicalDocument": ["source", "section", "text_vector", "mentionsSymptom", "mentionsPathology"],
            },
            "rationale": "preserve semantic relations and clinical context together with vectors",
        }

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
        dense_vector = self.embedding_model.embed(text)
        with self._lock:
            previous = self.records.get(record_id)
            requested_status = normalize_learning_status(learning_status)
            if previous and previous.learning_status == "promoted" and requested_status != "retired":
                requested_status = "promoted"
            record = VectorMemoryRecord(
                record_id=record_id,
                text=text,
                dense_vector=dense_vector,
                metadata=metadata,
                modality=modality,
                source=source,
                learning_status=requested_status,
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
                "ontology_refs": metadata.get("ontology_refs", []),
                "relation_count": len(metadata.get("relations", [])) if isinstance(metadata.get("relations", []), list) else 0,
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
        with self._lock:
            records = list(self.records.values())
        for record in records:
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
            "target_backend": "Weaviate object-property semantic graph RAG",
        }

    def transition_status(self, record_id: str, target_status: str, reason: str, evidence: dict[str, Any] | None = None) -> dict:
        with self._lock:
            record = self.records[record_id]
            transition = validate_learning_transition(record.learning_status, target_status, evidence=evidence or {})
            now = time.time()
            if not transition.allowed:
                self.update_log.append({
                    "event": "transition_rejected",
                    "record_id": record_id,
                    "target_status": target_status,
                    "reason": reason,
                    "transition": transition.as_dict(),
                    "timestamp": now,
                })
                return {"status": "rejected", "record": record.describe(), "transition": transition.as_dict()}
            record.learning_status = transition.target
            record.metadata = {**record.metadata, "status_transition_reason": reason}
            record.updated_at = now
            self.update_log.append({
                "event": "transition_status",
                "record_id": record_id,
                "target_status": transition.target,
                "reason": reason,
                "timestamp": now,
            })
            return {"status": "completed", "record": record.describe(), "transition": transition.as_dict()}

    def promote(self, record_id: str, reason: str) -> dict:
        result = self.transition_status(
            record_id=record_id,
            target_status="promoted",
            reason=reason,
            evidence={
                "rational_control_validation": True,
                "provenance_available": True,
                "clinical_deployment": False,
            },
        )
        return result["record"]

    def reject(self, record_id: str, reason: str) -> dict:
        return self.transition_status(record_id=record_id, target_status="rejected", reason=reason)["record"]

    def mark_needs_review(self, record_id: str, reason: str) -> dict:
        return self.transition_status(record_id=record_id, target_status="needs_review", reason=reason)["record"]

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
                "relations": [
                    {"from": f"case:{case_id}", "predicate": "hasClinicalTrace", "to": "clinical_pipeline_result"},
                ],
            },
            modality="multimodal_case_trace",
            source="clinical_pipeline",
            learning_status="candidate",
        )
        return {"status": "consolidated", "record_id": record.record_id, "memory": self.describe()}

    def describe(self) -> dict:
        statuses: dict[str, int] = {}
        with self._lock:
            records = list(self.records.values())
        for record in records:
            statuses[record.learning_status] = statuses.get(record.learning_status, 0) + 1
        return {
            "backend": self.backend,
            "recommended_enterprise_backend": self.recommended_enterprise_backend,
            "collection_name": self.collection_name,
            "record_count": len(records),
            "status_counts": statuses,
            "embedding_dimensions": self.embedding_model.dimensions,
            "supports_real_time_updates": True,
            "supports_multimodal_metadata": True,
            "supports_object_property_semantics": True,
            "supports_ontology_relation_metadata": True,
            "supports_governed_learning_status_transitions": True,
            "fallback_mode": "dependency_free_in_memory",
        }


@dataclass(slots=True)
class PersistentJsonlVectorStore(InMemoryVectorStore):
    """Durable local JSONL vector-memory backend for audit/research deployments.

    This backend is intentionally dependency-free and local. It is useful for
    repeatable tests, air-gapped research and small audit fixtures. Enterprise
    clinical deployments should still prefer a governed database/vector backend
    with tenant isolation, access control and operational monitoring.
    """

    path: str | Path = "melampo_vector_memory.jsonl"
    autosave: bool = True

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self._load_existing()

    def _record_payload(self, record: VectorMemoryRecord) -> dict[str, Any]:
        return {
            "record_id": record.record_id,
            "text": record.text,
            "dense_vector": record.dense_vector,
            "metadata": record.metadata,
            "modality": record.modality,
            "source": record.source,
            "learning_status": record.learning_status,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def _load_existing(self) -> None:
        path = Path(self.path)
        if not path.exists():
            return
        with self._lock:
            self.records.clear()
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                record = VectorMemoryRecord(
                    record_id=str(payload["record_id"]),
                    text=str(payload.get("text", "")),
                    dense_vector=[float(value) for value in payload.get("dense_vector", [])],
                    metadata=dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), dict) else {},
                    modality=str(payload.get("modality", "text")),
                    source=str(payload.get("source", "local")),
                    learning_status=normalize_learning_status(payload.get("learning_status", "candidate")),
                    created_at=float(payload.get("created_at", time.time())),
                    updated_at=float(payload.get("updated_at", time.time())),
                )
                self.records[record.record_id] = record

    def persist(self) -> dict[str, Any]:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = [json.dumps(self._record_payload(record), sort_keys=True, default=str) for record in self.records.values()]
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        tmp_path.replace(path)
        return {"status": "persisted", "path": str(path), "record_count": len(rows)}

    def upsert(self, text: str, metadata: dict[str, Any] | None = None, modality: str = "text", source: str = "local", learning_status: str = "candidate") -> VectorMemoryRecord:
        record = InMemoryVectorStore.upsert(self, text=text, metadata=metadata, modality=modality, source=source, learning_status=learning_status)
        if self.autosave:
            self.persist()
        return record

    def transition_status(self, record_id: str, target_status: str, reason: str, evidence: dict[str, Any] | None = None) -> dict:
        result = InMemoryVectorStore.transition_status(self, record_id=record_id, target_status=target_status, reason=reason, evidence=evidence)
        if self.autosave:
            self.persist()
        return result


VectorMemoryStore = InMemoryVectorStore
