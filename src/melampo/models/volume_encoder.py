from dataclasses import dataclass, field
from pathlib import Path

from .local_imaging_provider import LocalImagingFeatureProvider


@dataclass
class VolumeEncoder:
    """Future-facing imaging encoder adapter.

    The current implementation carries local image/DICOM paths, extracts
    technical local-imaging features, and exposes readiness/capability metadata.
    A real radiology/DICOM provider can later be attached behind the same
    interface without changing the clinical pipeline, CLI, or open-data loaders.
    """

    provider: str = "api_for_service_volume_encoder"
    provider_strategy: str = "future_multimodal_imaging_adapter"
    local_provider: LocalImagingFeatureProvider = field(default_factory=LocalImagingFeatureProvider)
    preferred_future_models: list[str] = field(
        default_factory=lambda: [
            "specialized_radiology_vision_language_model",
            "dicom_native_3d_foundation_model",
            "multimodal_clinical_vlm",
            "self_supervised_medical_image_encoder",
        ]
    )
    supported_modalities: tuple[str, ...] = ("CR", "DX", "CT", "MR", "US", "PT", "DICOM")

    def _infer_input_kind(self, series_paths: list[str], metadata: dict) -> str:
        modality = str(metadata.get("modality", metadata.get("Modality", ""))).upper()
        suffixes = {Path(path).suffix.lower() for path in series_paths}
        if modality in {"CT", "MR", "PT"}:
            return "volumetric_dicom_or_series"
        if ".dcm" in suffixes or not suffixes and series_paths:
            return "dicom_like_series"
        if suffixes.intersection({".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}):
            return "projection_or_image_file"
        return "metadata_only"

    def encode(self, study_id: str, series_paths: list[str] | None = None, metadata: dict | None = None) -> dict:
        series_paths = series_paths or []
        metadata = metadata or {}
        input_kind = self._infer_input_kind(series_paths, metadata)
        has_local_images = bool(series_paths)
        local_features = self.local_provider.extract(
            study_id=study_id,
            series_paths=series_paths,
            metadata=metadata,
            input_kind=input_kind,
        )
        encoder_ready = has_local_images and self.provider_strategy != "metadata_only"
        return {
            "provider": self.provider,
            "provider_strategy": self.provider_strategy,
            "study_id": study_id,
            "series_paths": series_paths,
            "image_count": len(series_paths),
            "has_local_images": has_local_images,
            "input_kind": input_kind,
            "supported_modalities": list(self.supported_modalities),
            "preferred_future_models": list(self.preferred_future_models),
            "encoder_ready": encoder_ready,
            "real_pixel_inference": False,
            "local_features": local_features,
            "routing_hint": local_features.get("routing_hint", "metadata_only_review"),
            "metadata": metadata,
            "notes": [
                "Current adapter carries image paths and extracts technical local-imaging features.",
                "Attach a real radiology/DICOM provider behind this interface for pixel-level inference.",
            ],
        }
