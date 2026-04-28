from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RemoteImagingProviderClient:
    """Remote imaging provider contract for future radiology/DICOM models.

    This client intentionally does not perform network calls yet. It defines the
    request/response contract and a safe fallback shape so the rest of Melampo
    can be wired for remote providers without depending on a live service.
    """

    provider_name: str
    endpoint: str | None = None
    timeout_seconds: int = 30
    enabled: bool = False

    def build_request(self, study_id: str, series_paths: list[str], metadata: dict, provider_selection: dict) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "endpoint": self.endpoint,
            "study_id": study_id,
            "series_paths": series_paths,
            "metadata": metadata,
            "provider_selection": provider_selection,
            "timeout_seconds": self.timeout_seconds,
        }

    def infer(self, study_id: str, series_paths: list[str], metadata: dict, provider_selection: dict) -> dict[str, Any]:
        request = self.build_request(
            study_id=study_id,
            series_paths=series_paths,
            metadata=metadata,
            provider_selection=provider_selection,
        )
        if not self.enabled or not self.endpoint:
            return {
                "provider": self.provider_name,
                "status": "not_called",
                "reason": "remote_provider_not_configured",
                "request": request,
                "features": {},
                "fallback_required": True,
            }
        return {
            "provider": self.provider_name,
            "status": "request_prepared",
            "reason": "network_call_not_implemented_in_research_stub",
            "request": request,
            "features": {},
            "fallback_required": True,
        }
