from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ImagingProviderSelection:
    strategy: str
    provider_kind: str
    provider_name: str
    real_pixel_inference: bool
    requires_remote: bool
    readiness_requirement: str

    def describe(self) -> dict:
        return {
            "strategy": self.strategy,
            "provider_kind": self.provider_kind,
            "provider_name": self.provider_name,
            "real_pixel_inference": self.real_pixel_inference,
            "requires_remote": self.requires_remote,
            "readiness_requirement": self.readiness_requirement,
        }


@dataclass(slots=True)
class ImagingProviderSelector:
    """Select the imaging provider family implied by the runtime strategy."""

    def select(self, strategy: str, input_kind: str) -> ImagingProviderSelection:
        if strategy == "local_pixels":
            return ImagingProviderSelection(
                strategy=strategy,
                provider_kind="local_pixel_adapter",
                provider_name="local_pixel_feature_provider",
                real_pixel_inference=True,
                requires_remote=False,
                readiness_requirement="local_images_required",
            )
        if strategy == "remote_radiology_vlm":
            return ImagingProviderSelection(
                strategy=strategy,
                provider_kind="remote_projection_radiology_vlm",
                provider_name="remote_radiology_vlm_provider",
                real_pixel_inference=True,
                requires_remote=True,
                readiness_requirement="local_or_remote_projection_images_required",
            )
        if strategy == "remote_dicom_3d":
            return ImagingProviderSelection(
                strategy=strategy,
                provider_kind="remote_3d_dicom_foundation_model",
                provider_name="remote_dicom_3d_provider",
                real_pixel_inference=True,
                requires_remote=True,
                readiness_requirement="dicom_or_volumetric_series_required",
            )
        if strategy == "hybrid_multimodal":
            provider_kind = "remote_3d_dicom_foundation_model" if input_kind == "volumetric_dicom_or_series" else "remote_projection_radiology_vlm"
            provider_name = "remote_dicom_3d_provider" if input_kind == "volumetric_dicom_or_series" else "remote_radiology_vlm_provider"
            return ImagingProviderSelection(
                strategy=strategy,
                provider_kind=provider_kind,
                provider_name=provider_name,
                real_pixel_inference=True,
                requires_remote=True,
                readiness_requirement="image_or_dicom_series_required",
            )
        return ImagingProviderSelection(
            strategy="local_metadata",
            provider_kind="local_metadata_adapter",
            provider_name="local_imaging_feature_provider",
            real_pixel_inference=False,
            requires_remote=False,
            readiness_requirement="metadata_or_paths_optional",
        )
