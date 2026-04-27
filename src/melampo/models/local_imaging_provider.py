from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LocalImagingFeatureProvider:
    """Local metadata/path imaging feature provider.

    This provider intentionally avoids clinical pixel interpretation. It gives
    Melampo a concrete local backend for technical image availability,
    modality, extension, and routing metadata while preserving the same future
    adapter surface for real radiology/DICOM providers.
    """

    provider_name: str = "local_imaging_feature_provider"

    def extract(self, study_id: str, series_paths: list[str], metadata: dict, input_kind: str) -> dict:
        suffixes = sorted({Path(path).suffix.lower() or "<no_suffix>" for path in series_paths})
        existing_paths = [path for path in series_paths if Path(path).exists()]
        missing_paths = [path for path in series_paths if not Path(path).exists()]
        modality = str(metadata.get("modality", metadata.get("Modality", metadata.get("source_modality", "unknown")))).upper()
        return {
            "provider": self.provider_name,
            "study_id": study_id,
            "input_kind": input_kind,
            "modality": modality,
            "path_count": len(series_paths),
            "existing_path_count": len(existing_paths),
            "missing_path_count": len(missing_paths),
            "file_suffixes": suffixes,
            "local_readiness": "ready" if series_paths and not missing_paths else "partial" if existing_paths else "metadata_only",
            "pixel_interpretation": "not_performed",
            "routing_hint": self._routing_hint(input_kind=input_kind, modality=modality),
        }

    def _routing_hint(self, input_kind: str, modality: str) -> str:
        if input_kind == "volumetric_dicom_or_series" or modality in {"CT", "MR", "PT"}:
            return "route_to_3d_dicom_provider"
        if input_kind in {"projection_or_image_file", "dicom_like_series"}:
            return "route_to_projection_radiology_provider"
        return "metadata_only_review"
