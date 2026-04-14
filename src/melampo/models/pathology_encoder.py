from dataclasses import dataclass


@dataclass
class PathologyEncoder:
    provider: str = "api_for_service_pathology_encoder"

    def encode(self, slide_id: str) -> dict:
        return {"provider": self.provider, "slide_id": slide_id}
