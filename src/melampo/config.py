from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


@dataclass(slots=True)
class ServiceConfig:
    provider: str
    endpoint: Optional[str] = None
    api_key_env: Optional[str] = None
    timeout_seconds: int = 30
    enabled: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    project_name: str = "melampo"
    environment: str = "research"
    root_dir: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = Path("./data")
    allow_remote_models: bool = False
    abstention_threshold: float = 0.65
    risk_threshold: float = 0.35
    calibration_bin_count: int = 15
    service_registry: Dict[str, ServiceConfig] = field(default_factory=dict)


def build_default_config() -> RuntimeConfig:
    """Build a default runtime config with explicit placeholder services."""
    placeholders = {
        "volume_encoder": ServiceConfig(provider="api_for_service_volume_encoder"),
        "pathology_encoder": ServiceConfig(provider="api_for_service_pathology_encoder"),
        "clinical_text_encoder": ServiceConfig(provider="api_for_service_clinical_text_encoder"),
        "multimodal_reasoner": ServiceConfig(provider="api_for_service_multimodal_reasoner"),
        "critique_service": ServiceConfig(provider="api_for_service_critique_model"),
        "synthetic_case_generator": ServiceConfig(provider="api_for_service_synthetic_case_generator"),
        "vector_store": ServiceConfig(provider="api_for_service_vector_store"),
        "knowledge_graph": ServiceConfig(provider="api_for_service_knowledge_graph"),
        "document_rag": ServiceConfig(provider="api_for_service_document_rag"),
        "mcp_server": ServiceConfig(provider="api_for_service_mcp_server"),
        "a2a_router": ServiceConfig(provider="api_for_service_a2a_router"),
        "theoretical_quantum": ServiceConfig(provider="api_for_service_theoretical_quantum_module", enabled=False),
    }
    return RuntimeConfig(service_registry=placeholders)
