from dataclasses import dataclass


@dataclass
class VolumeEncoder:
    provider: str = "api_for_service_volume_encoder"

    def encode(self, study_id: str) -> dict:
        return {"provider": self.provider, "study_id": study_id}
