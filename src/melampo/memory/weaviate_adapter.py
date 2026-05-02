from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .weaviate_schema import MelampoWeaviateSchema


@dataclass(slots=True)
class WeaviateAdapterConfig:
    """Configuration for a future Weaviate client adapter."""

    endpoint: str | None = None
    api_key_env: str | None = None
    collection_prefix: str = "Melampo"
    enabled: bool = False
    timeout_seconds: int = 30

    def describe(self) -> dict[str, Any]:
        return {
            "endpoint_configured": self.endpoint is not None,
            "api_key_env": self.api_key_env,
            "collection_prefix": self.collection_prefix,
            "enabled": self.enabled,
            "timeout_seconds": self.timeout_seconds,
        }


@dataclass(slots=True)
class WeaviateSemanticMemoryAdapter:
    """Provider contract for Weaviate semantic object-property memory.

    This class is deliberately network-safe: it does not import the Weaviate
    client or perform hidden calls. A production adapter can subclass or replace
    it while preserving the same request/response contract.
    """

    config: WeaviateAdapterConfig = field(default_factory=WeaviateAdapterConfig)
    schema: MelampoWeaviateSchema = field(default_factory=MelampoWeaviateSchema)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "Weaviate",
            "role": "semantic_object_property_memory",
            "status": "configured" if self.config.enabled and self.config.endpoint else "contract_only",
            "config": self.config.describe(),
            "schema_classes": self.schema.class_names(),
            "supports": [
                "object_property_semantics",
                "ontology_relations",
                "named_vectors",
                "multimodal_case_memory",
                "governed_learning_status",
            ],
            "limitations": [
                "no_network_call_in_contract_adapter",
                "production_client_required_for_live_weaviate",
                "research_schema_not_medical_device",
            ],
        }

    def schema_payload(self) -> dict[str, Any]:
        return self.schema.as_dict()

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
