from dataclasses import dataclass

from .contracts import ServiceContract
from .service_registry import ServiceRegistry


@dataclass
class RegistryBootstrap:
    """Populate a registry with the baseline Melampo service contracts."""

    def contracts(self) -> list[ServiceContract]:
        return [
            ServiceContract("volume_encoder", "api_for_service_volume_encoder", "service", "encoder"),
            ServiceContract("pathology_encoder", "api_for_service_pathology_encoder", "service", "encoder"),
            ServiceContract("text_encoder", "api_for_service_clinical_text_encoder", "service", "encoder"),
            ServiceContract("rag", "api_for_service_document_rag", "service", "retrieval"),
            ServiceContract("knowledge_graph", "api_for_service_knowledge_graph", "service", "knowledge"),
            ServiceContract("mcp_server", "api_for_service_mcp_server", "mcp", "orchestration"),
            ServiceContract("a2a_router", "api_for_service_a2a_router", "a2a", "orchestration"),
        ]

    def build(self) -> ServiceRegistry:
        registry = ServiceRegistry()
        for contract in self.contracts():
            registry.register(contract.name, contract.provider, contract.protocol, contract.role)
        return registry
