from dataclasses import dataclass
from typing import Any


@dataclass
class PathologyEncoder:
    config: Any | None = None
    provider: str = "api_for_service_pathology_encoder"

    def __post_init__(self) -> None:
        if self.config is not None:
            service_registry = getattr(self.config, "service_registry", {})
            pathology_service = service_registry.get("pathology_encoder") if isinstance(service_registry, dict) else None
            if pathology_service is not None:
                self.provider = getattr(pathology_service, "provider", self.provider)

    def encode(self, slide_id: str) -> dict:
        return {"provider": self.provider, "slide_id": slide_id}
