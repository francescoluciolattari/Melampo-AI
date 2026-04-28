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

    def describe(self) -> dict:
        return {
            "provider": self.provider,
            "endpoint_configured": self.endpoint is not None,
            "api_key_env": self.api_key_env,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
        }


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
    runtime_profile: str = "local_research"
    imaging_provider_strategy: str = "local_metadata"
    service_registry: Dict[str, ServiceConfig] = field(default_factory=dict)

    def describe(self) -> dict:
        enabled = {name: service.describe() for name, service in self.service_registry.items() if service.enabled}
        disabled = {name: service.describe() for name, service in self.service_registry.items() if not service.enabled}
        return {
            "project_name": self.project_name,
            "environment": self.environment,
            "runtime_profile": self.runtime_profile,
            "allow_remote_models": self.allow_remote_models,
            "imaging_provider_strategy": self.imaging_provider_strategy,
            "abstention_threshold": self.abstention_threshold,
            "risk_threshold": self.risk_threshold,
            "calibration_bin_count": self.calibration_bin_count,
            "enabled_services": enabled,
            "disabled_services": disabled,
            "service_count": len(self.service_registry),
        }


def build_default_config(runtime_profile: str = "local_research", allow_remote_models: bool = False, imaging_provider_strategy: str | None = None) -> RuntimeConfig:
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
    if imaging_provider_strategy is None:
        imaging_provider_strategy = "local_metadata"
    if runtime_profile == "remote_research":
        allow_remote_models = True
        imaging_provider_strategy = "hybrid_multimodal" if imaging_provider_strategy == "local_metadata" else imaging_provider_strategy
        placeholders["theoretical_quantum"].enabled = True
    return RuntimeConfig(
        runtime_profile=runtime_profile,
        allow_remote_models=allow_remote_models,
        imaging_provider_strategy=imaging_provider_strategy,
        service_registry=placeholders,
    )
