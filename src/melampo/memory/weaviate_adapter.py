from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

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

        # The v4 Weaviate Python client has evolved quickly. To avoid baking an
        # unstable API into the core, live materialization is intentionally
        # represented as a boundary. Production deployments should implement the
        # exact client calls in an infrastructure-specific subclass.
        return {
            **prepared,
            "status": "requires_infrastructure_subclass",
            "reason": "live_client_call_deliberately_not_hardcoded_in_core",
            "recommended_next_step": "implement subclass that maps MelampoWeaviateSchema to the installed weaviate-client version",
        }

    def prepare_upsert(self, class_name: str, object_id: str, properties: dict[str, Any], references: list[dict[str, Any]] | None = None, vectors: dict[str, list[float]] | None = None) -> dict[str, Any]:
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

    def upsert_object(self, class_name: str, object_id: str, properties: dict[str, Any], references: list[dict[str, Any]] | None = None, vectors: dict[str, list[float]] | None = None) -> dict[str, Any]:
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

    def prepare_semantic_search(self, class_name: str, query: str, target_vector: str | None = None, filters: dict[str, Any] | None = None, limit: int = 5) -> dict[str, Any]:
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

    def semantic_search(self, class_name: str, query: str, target_vector: str | None = None, filters: dict[str, Any] | None = None, limit: int = 5) -> dict[str, Any]:
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
