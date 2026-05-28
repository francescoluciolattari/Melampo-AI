from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable

from .vector_memory import InMemoryVectorStore
from .visual_imprint import VisualRecognitionImprint
from .weaviate_schema import MelampoWeaviateSchema, WeaviateClassSchema


@dataclass(slots=True)
class WeaviateAdapterConfig:
    """Configuration for a Weaviate semantic-memory adapter."""

    endpoint: str | None = None
    api_key_env: str | None = None
    collection_prefix: str = "Melampo"
    enabled: bool = False
    timeout_seconds: int = 30
    dry_run: bool = True

    def describe(self) -> dict[str, Any]:
        return {
            "endpoint_configured": self.endpoint is not None,
            "api_key_env": self.api_key_env,
            "api_key_available": bool(os.getenv(self.api_key_env)) if self.api_key_env else False,
            "collection_prefix": self.collection_prefix,
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
            "dry_run": self.dry_run,
        }


@dataclass(slots=True)
class WeaviateSemanticMemoryAdapter:
    """Provider contract for Weaviate semantic object-property memory.

    The adapter is safe by default. It prepares schema, upsert and search
    payloads without network calls unless `enabled=True`, `dry_run=False`, an
    endpoint is configured and `weaviate-client` is installed. Live methods are
    conservative and return structured errors instead of raising dependency or
    configuration exceptions.
    """

    config: WeaviateAdapterConfig = field(default_factory=WeaviateAdapterConfig)
    schema: MelampoWeaviateSchema = field(default_factory=MelampoWeaviateSchema)

    def describe(self) -> dict[str, Any]:
        live_ready = self._live_ready()["ready"]
        return {
            "provider": "Weaviate",
            "role": "semantic_object_property_memory",
            "status": "live_ready" if live_ready else "contract_or_dry_run",
            "config": self.config.describe(),
            "schema_classes": self.schema.class_names(),
            "supports": [
                "object_property_semantics",
                "ontology_relations",
                "named_vectors",
                "multimodal_case_memory",
                "governed_learning_status",
                "optional_live_schema_materialization",
            ],
            "limitations": [
                "research_schema_not_medical_device",
                "live_calls_require_explicit_enabled_and_dry_run_false",
                "schema_mapping_may_need_backend_specific_refinement",
            ],
        }

    def _import_weaviate(self) -> dict[str, Any]:
        try:
            import weaviate  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency optional
            return {"available": False, "module": None, "error": str(exc)}
        return {"available": True, "module": weaviate, "error": None}

    def _live_ready(self) -> dict[str, Any]:
        imported = self._import_weaviate()
        ready = bool(self.config.enabled and not self.config.dry_run and self.config.endpoint and imported["available"])
        reasons = []
        if not self.config.enabled:
            reasons.append("adapter_disabled")
        if self.config.dry_run:
            reasons.append("dry_run_enabled")
        if not self.config.endpoint:
            reasons.append("endpoint_missing")
        if not imported["available"]:
            reasons.append("weaviate_client_unavailable")
        return {"ready": ready, "reasons": reasons, "import": imported}

    def schema_payload(self) -> dict[str, Any]:
        return self.schema.as_dict()

    def class_payload(self, class_schema: WeaviateClassSchema) -> dict[str, Any]:
        """Return a backend-neutral class payload suitable for adapter translation."""
        return class_schema.as_dict()

    def prepare_schema_materialization(self) -> dict[str, Any]:
        return {
            "status": "prepared",
            "operation": "schema_materialization",
            "backend": "Weaviate",
            "dry_run": self.config.dry_run,
            "live_ready": self._live_ready(),
            "schema": self.schema_payload(),
            "governance": {
                "hidden_network_call": False,
                "requires_explicit_live_enablement": True,
                "research_schema_not_medical_device": True,
            },
        }

    def materialize_schema(self) -> dict[str, Any]:
        readiness = self._live_ready()
        prepared = self.prepare_schema_materialization()
        if not readiness["ready"]:
            return {**prepared, "status": "not_executed", "reason": readiness["reasons"]}
        return {
            **prepared,
            "status": "requires_infrastructure_subclass",
            "reason": "live_client_call_deliberately_not_hardcoded_in_core",
            "recommended_next_step": "implement subclass that maps MelampoWeaviateSchema to the installed weaviate-client version",
        }

    def prepare_upsert(
        self,
        class_name: str,
        object_id: str,
        properties: dict[str, Any],
        references: list[dict[str, Any]] | None = None,
        vectors: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        if class_name not in self.schema.class_names():
            return {
                "status": "rejected",
                "reason": "unknown_schema_class",
                "class_name": class_name,
                "known_classes": self.schema.class_names(),
            }
        return {
            "status": "prepared",
            "operation": "upsert_object",
            "backend": "Weaviate",
            "class_name": class_name,
            "object_id": object_id,
            "properties": properties,
            "references": references or [],
            "vectors": vectors or {},
            "governance": {
                "requires_provenance": True,
                "learning_status": properties.get("learning_status", "candidate"),
                "hidden_network_call": False,
            },
        }

    def upsert_object(
        self,
        class_name: str,
        object_id: str,
        properties: dict[str, Any],
        references: list[dict[str, Any]] | None = None,
        vectors: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        prepared = self.prepare_upsert(class_name=class_name, object_id=object_id, properties=properties, references=references, vectors=vectors)
        if prepared["status"] == "rejected":
            return prepared
        readiness = self._live_ready()
        if not readiness["ready"]:
            return {**prepared, "status": "not_executed", "reason": readiness["reasons"]}
        return {
            **prepared,
            "status": "requires_infrastructure_subclass",
            "reason": "live_upsert_deliberately_not_hardcoded_in_core",
        }

    def prepare_semantic_search(
        self,
        class_name: str,
        query: str,
        target_vector: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        if class_name not in self.schema.class_names():
            return {
                "status": "rejected",
                "reason": "unknown_schema_class",
                "class_name": class_name,
                "known_classes": self.schema.class_names(),
            }
        return {
            "status": "prepared",
            "operation": "semantic_search",
            "backend": "Weaviate",
            "class_name": class_name,
            "query": query,
            "target_vector": target_vector,
            "filters": filters or {},
            "limit": limit,
            "governance": {
                "return_provenance_required": True,
                "return_relations_required": True,
                "hidden_network_call": False,
            },
        }

    def semantic_search(
        self,
        class_name: str,
        query: str,
        target_vector: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        prepared = self.prepare_semantic_search(class_name=class_name, query=query, target_vector=target_vector, filters=filters, limit=limit)
        if prepared["status"] == "rejected":
            return prepared
        readiness = self._live_ready()
        if not readiness["ready"]:
            return {**prepared, "status": "not_executed", "reason": readiness["reasons"], "hits": []}
        return {
            **prepared,
            "status": "requires_infrastructure_subclass",
            "reason": "live_search_deliberately_not_hardcoded_in_core",
            "hits": [],
        }


@dataclass(slots=True)
class WeaviateEnterpriseMemoryAdapter(WeaviateSemanticMemoryAdapter):
    """Phase-2 enterprise RAG adapter with safe local execution.

    The class models the production Weaviate flow Melampo needs without making
    hidden network calls. In dry-run or disabled mode it materializes the same
    object-property payloads into an in-process vector/object graph so the core
    pipeline, tests and air-gapped development can exercise real upsert/search
    behavior. Infrastructure-specific subclasses can replace the local fallback
    with actual Weaviate client calls while preserving this public contract.
    """

    fallback_store: InMemoryVectorStore = field(default_factory=InMemoryVectorStore.enterprise_default)
    object_graph: dict[str, dict[str, Any]] = field(default_factory=dict)
    relation_index: list[dict[str, Any]] = field(default_factory=list)

    def describe(self) -> dict[str, Any]:
        base = super().describe()
        return {
            **base,
            "status": "phase2_enterprise_contract_live_ready" if self._live_ready()["ready"] else "phase2_safe_local_object_graph",
            "fallback_store": self.fallback_store.describe(),
            "object_count": len(self.object_graph),
            "relation_count": len(self.relation_index),
            "supports": sorted(set(base["supports"] + [
                "hybrid_search_contract",
                "multi_target_vector_search_contract",
                "graph_expansion",
                "document_chunk_upsert",
                "case_trace_upsert",
                "local_object_graph_fallback",
            ])),
        }

    def materialize_schema(self) -> dict[str, Any]:
        prepared = self.prepare_schema_materialization()
        readiness = self._live_ready()
        status = "materialized_in_local_contract" if not readiness["ready"] else "requires_infrastructure_subclass"
        reason = readiness["reasons"] if not readiness["ready"] else ["live_client_call_deliberately_not_hardcoded_in_core"]
        return {
            **prepared,
            "status": status,
            "reason": reason,
            "materialized_classes": self.schema.class_names(),
            "local_object_graph_ready": True,
        }

    def _store_prepared_object(self, prepared: dict[str, Any], text: str | None = None) -> dict[str, Any]:
        if prepared.get("status") == "rejected":
            return prepared
        object_id = str(prepared["object_id"])
        class_name = str(prepared["class_name"])
        properties = dict(prepared.get("properties", {}))
        references = list(prepared.get("references", []))
        vectors = dict(prepared.get("vectors", {}))
        object_key = f"{class_name}:{object_id}"
        text_value = text or str(properties.get("text") or properties.get("description") or properties.get("name") or properties)
        metadata = {
            **properties,
            "record_id": object_key,
            "class_name": class_name,
            "object_id": object_id,
            "references": references,
            "relations": references,
            "vectors": sorted(vectors.keys()),
            "backend": "weaviate_phase2_local_contract",
        }
        record = self.fallback_store.upsert(
            text=text_value,
            metadata=metadata,
            modality=str(properties.get("modality", "clinical_object")),
            source="weaviate_enterprise_adapter",
            learning_status=str(properties.get("learning_status", "candidate")),
        )
        self.object_graph[object_key] = {
            "class_name": class_name,
            "object_id": object_id,
            "properties": properties,
            "references": references,
            "vectors": vectors,
            "record_id": record.record_id,
            "text": text_value,
        }
        for reference in references:
            relation = {
                "from": object_key,
                "predicate": reference.get("name") or reference.get("predicate") or reference.get("property") or "relatedTo",
                "to": reference.get("target_id") or reference.get("target") or reference.get("to") or "unknown",
                "target_class": reference.get("target_class") or reference.get("target"),
            }
            self.relation_index.append(relation)
        return {
            **prepared,
            "status": "stored_in_local_object_graph" if not self._live_ready()["ready"] else "requires_infrastructure_subclass",
            "record_id": record.record_id,
            "object_key": object_key,
            "hidden_network_call": False,
        }

    def upsert_object(
        self,
        class_name: str,
        object_id: str,
        properties: dict[str, Any],
        references: list[dict[str, Any]] | None = None,
        vectors: dict[str, list[float]] | None = None,
    ) -> dict[str, Any]:
        prepared = self.prepare_upsert(class_name=class_name, object_id=object_id, properties=properties, references=references, vectors=vectors)
        return self._store_prepared_object(prepared)

    def upsert_clinical_document_chunk(self, document: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(document.get("metadata", {}))
        text = str(document.get("text", ""))
        object_id = str(metadata.get("record_id") or document.get("record_id") or metadata.get("chunk_id") or "clinical_document_chunk")
        properties = {
            "source": str(metadata.get("source_path") or metadata.get("source_uri") or document.get("source", "unknown")),
            "section": metadata.get("section") or "unknown",
            "page": metadata.get("page"),
            "text": text,
            "publication_date": metadata.get("publication_date"),
            "license": metadata.get("license") or metadata.get("license_class") or "unknown",
            "learning_status": document.get("learning_status", "candidate"),
            "source_type": metadata.get("source_type", "clinical_document"),
            "provenance_quality": metadata.get("provenance_quality", 0.0),
            "modality": document.get("modality", "clinical_document_text"),
            "ontology_refs": metadata.get("ontology_refs", []),
        }
        references = [
            {"name": relation.get("predicate", "relatedTo"), "target": relation.get("to", "unknown"), **relation}
            for relation in metadata.get("relations", [])
            if isinstance(relation, dict)
        ]
        prepared = self.prepare_upsert(
            class_name="ClinicalDocument",
            object_id=object_id,
            properties=properties,
            references=references,
            vectors={"document_text_vector": self.fallback_store.embedding_model.embed(text)},
        )
        return self._store_prepared_object(prepared, text=text)

    def upsert_case_trace(self, case_payload: dict[str, Any], diagnostic_result: dict[str, Any] | None = None) -> dict[str, Any]:
        diagnostic_result = diagnostic_result or {}
        case_id = str(case_payload.get("case_id") or diagnostic_result.get("case_id") or "unknown_case")
        text_parts = [case_id]
        for key in ["report_text", "ehr_text", "patient_complaints"]:
            if case_payload.get(key):
                text_parts.append(str(case_payload[key]))
        if diagnostic_result:
            text_parts.append(f"result_label={diagnostic_result.get('result_label', 'unknown')}")
            text_parts.append(f"policy={diagnostic_result.get('policy', {})}")
        text = "\n".join(text_parts)
        properties = {
            "case_id": case_id,
            "demographics": case_payload.get("demographics", {}),
            "provenance": case_payload.get("provenance", {}),
            "learning_status": "candidate",
            "text": text,
            "modality": "multimodal_case_trace",
        }
        references = [
            {"name": "hasDifferential", "target": item.get("label", "unknown"), "target_class": "Pathology"}
            for item in diagnostic_result.get("differential", [])[:5]
            if isinstance(item, dict)
        ]
        prepared = self.prepare_upsert(
            class_name="ClinicalCase",
            object_id=f"case:{case_id}",
            properties=properties,
            references=references,
            vectors={"case_trace_vector": self.fallback_store.embedding_model.embed(text)},
        )
        return self._store_prepared_object(prepared, text=text)

    def hybrid_search(
        self,
        query: str,
        class_name: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 5,
        alpha: float = 0.65,
    ) -> dict[str, Any]:
        filters = filters or {}
        if class_name:
            filters = {**filters, "class_name": class_name}
        hits = self.fallback_store.search(query=query, limit=limit, filters=filters or None)
        enriched = []
        for rank, hit in enumerate(hits, start=1):
            text = str(hit.get("text") or hit.get("value") or "")
            lexical_overlap = _lexical_overlap(query, text)
            vector_score = float(hit.get("grounding_score", 0.0))
            final_score = round(alpha * vector_score + (1.0 - alpha) * lexical_overlap, 6)
            enriched.append({
                **hit,
                "rank": rank,
                "source": "weaviate",
                "kind": "object_property_rag_hit",
                "retrieval_backend": "weaviate_phase2_local_contract",
                "score_vector": vector_score,
                "score_bm25": lexical_overlap,
                "score_graph": self._graph_score(hit),
                "score_final": final_score,
                "grounding_score": final_score,
            })
        enriched.sort(key=lambda item: item["score_final"], reverse=True)
        for rank, hit in enumerate(enriched, start=1):
            hit["rank"] = rank
        return {
            "status": "completed",
            "operation": "hybrid_search",
            "backend": "Weaviate",
            "query": query,
            "class_name": class_name,
            "filters": filters,
            "alpha": alpha,
            "hit_count": len(enriched),
            "hits": enriched[:limit],
            "governance": {
                "return_provenance_required": True,
                "local_contract_execution": not self._live_ready()["ready"],
                "hidden_network_call": False,
            },
        }

    def multi_vector_search(
        self,
        query: str,
        target_vectors: Iterable[str] | None = None,
        class_name: str | None = None,
        limit: int = 5,
    ) -> dict[str, Any]:
        target_vectors = list(target_vectors or ["document_text_vector", "ontology_context_vector"])
        result = self.hybrid_search(query=query, class_name=class_name, limit=limit)
        hits = []
        for hit in result["hits"]:
            vectors = set(hit.get("metadata", {}).get("vectors", []))
            target_coverage = len(vectors.intersection(target_vectors)) / max(len(target_vectors), 1) if vectors else 0.0
            hits.append({
                **hit,
                "target_vectors": target_vectors,
                "target_vector_coverage": round(target_coverage, 3),
                "score_final": round(hit.get("score_final", hit.get("grounding_score", 0.0)) + target_coverage * 0.05, 6),
            })
        hits.sort(key=lambda item: item["score_final"], reverse=True)
        return {**result, "operation": "multi_vector_search", "target_vectors": target_vectors, "hits": hits[:limit]}

    def upsert_visual_imprint(self, imprint_payload: dict[str, Any]) -> dict[str, Any]:
        imprint = VisualRecognitionImprint.from_payload(imprint_payload)
        imprint_dict = imprint.as_dict()
        concept_id = str(imprint_dict["semantic_concept"]).replace(" ", "_")
        concept_properties = {
            "name": imprint_dict["semantic_concept"],
            "description": f"Visual concept associated with recognition imprints for {imprint_dict['semantic_concept']}",
            "ontology_refs": imprint_payload.get("ontology_refs", []),
            "learning_status": imprint_dict["learning_status"],
        }
        concept_result = self.upsert_object(
            class_name="VisualConcept",
            object_id=f"visual_concept:{concept_id}",
            properties=concept_properties,
            references=[],
            vectors={"visual_concept_text_vector": self.fallback_store.embedding_model.embed(imprint_dict["semantic_concept"])},
        )
        references = [
            {
                "name": "representsConcept",
                "target": f"visual_concept:{concept_id}",
                "target_class": "VisualConcept",
            }
        ]
        if imprint.source_object_id and imprint.source_object_id != "unknown":
            references.append({"name": "derivedFromStudy", "target": imprint.source_object_id, "target_class": "ImagingStudy"})
        imprint_result = self.upsert_object(
            class_name="VisualRecognitionImprint",
            object_id=imprint.imprint_id,
            properties={
                **imprint_dict,
                "provenance": {
                    **imprint_dict.get("provenance", {}),
                    "hidden_network_call": False,
                    "stored_as": "semantic_visual_imprint",
                },
            },
            references=references,
            vectors={
                "recognition_matrix_vector": imprint.vector,
                "semantic_concept_vector": self.fallback_store.embedding_model.embed(imprint.semantic_concept),
            },
        )
        return {
            "status": "completed" if imprint_result.get("status") == "stored_in_local_object_graph" else imprint_result.get("status"),
            "operation": "upsert_visual_imprint",
            "concept_result": concept_result,
            "imprint_result": imprint_result,
            "imprint": imprint_dict,
            "governance": {
                "hidden_network_call": False,
                "semantic_object_imprint_association": True,
                "clinical_warning": "Visual imprint memory is research-only and not a validated diagnosis.",
            },
        }

    def graph_expand(self, object_key: str, depth: int = 1) -> dict[str, Any]:
        frontier = {object_key}
        visited = {object_key}
        relations = []
        for _ in range(max(depth, 0)):
            next_frontier = set()
            for relation in self.relation_index:
                if relation["from"] in frontier or relation["to"] in frontier:
                    relations.append(relation)
                    next_frontier.add(relation["from"])
                    next_frontier.add(relation["to"])
            frontier = next_frontier - visited
            visited.update(next_frontier)
        return {
            "status": "completed",
            "object_key": object_key,
            "depth": depth,
            "visited": sorted(visited),
            "relations": relations,
            "object_count": len(visited),
        }

    def _graph_score(self, hit: dict[str, Any]) -> float:
        metadata = hit.get("metadata", {}) if isinstance(hit.get("metadata", {}), dict) else {}
        relations = metadata.get("relations") or metadata.get("references") or []
        ontology_refs = metadata.get("ontology_refs") or []
        return round(min(1.0, 0.08 * len(relations) + 0.05 * len(ontology_refs)), 6)

    # Compatibility with MemoryRetriever.
    def semantic_search(self, query: str, limit: int = 5, promoted_only: bool = False, **kwargs: Any) -> list[dict[str, Any]]:  # type: ignore[override]
        required_status = ["promoted"] if promoted_only else None
        hits = self.fallback_store.search(query=query, limit=limit, required_status=required_status)
        return [
            {
                **hit,
                "source": "weaviate",
                "kind": "object_property_rag_hit",
                "retrieval_backend": "weaviate_phase2_local_contract",
            }
            for hit in hits
        ]

    def search(self, query: str, limit: int = 5, required_status: Iterable[str] | None = None, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        return self.fallback_store.search(query=query, limit=limit, required_status=required_status, filters=filters)


def _lexical_overlap(query: str, text: str) -> float:
    query_terms = {term.strip(".,;:()[]{}!?\"'`)._").lower() for term in query.split() if term.strip()}
    text_terms = {term.strip(".,;:()[]{}!?\"'`)._").lower() for term in text.split() if term.strip()}
    query_terms.discard("")
    text_terms.discard("")
    if not query_terms:
        return 0.0
    return round(len(query_terms.intersection(text_terms)) / len(query_terms), 6)


# Backward-compatible alias for integrations that expect a concrete enterprise adapter.
WeaviateLiveAdapter = WeaviateEnterpriseMemoryAdapter
