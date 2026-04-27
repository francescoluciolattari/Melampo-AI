from dataclasses import dataclass


@dataclass
class VolumeEncoder:
    provider: str = "api_for_service_volume_encoder"

    def encode(self, study_id: str, series_paths: list[str] | None = None, metadata: dict | None = None) -> dict:
        series_paths = series_paths or []
        metadata = metadata or {}
        return {
            "provider": self.provider,
            "study_id": study_id,
            "series_paths": series_paths,
            "image_count": len(series_paths),
            "has_local_images": bool(series_paths),
            "metadata": metadata,
        }
