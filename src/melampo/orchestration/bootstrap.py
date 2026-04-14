from dataclasses import dataclass

from .contracts import ServiceContract
from .service_registry import ServiceRegistry


@dataclass
class RegistryBootstrap:
    """Populate a registry with the baseline Melampo service contracts."""

    def build(self) -> ServiceRegistry:
        registry = ServiceRegistry()
        for contract in [
            ServiceContract("volume_encoder", "api_for_service_volume_encoder", "service"),
            ServiceContract("pathology_encoder", "api_for_service_pathology_encoder", "service"),
            ServiceContract("text_encoder", "api_for_service_clinical_text_encoder", "service"),
            ServiceContract("rag", "api_for_service_document_rag", "service"),
            ServiceContract("knowledge_graph", "api_for_service_knowledge_graph", "service"),
            ServiceContract("mcp_server", "api_for_service_mcp_server", "mcp"),
            ServiceContract("a2a_router", "api_for_service_a2a_router", "a2a"),
        ]:
            registry.register(contract.name, contract.provider, contract.protocol)
        return registry
