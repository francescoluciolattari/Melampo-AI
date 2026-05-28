from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..types import CaseContext, ClinicalObservation, ImagingStudy, Modality


@dataclass(slots=True)
class ClinicalIngestionPipeline:
    """Collect raw case assets into a single canonical case object."""

    def from_payload(self, payload: Mapping[str, Any]) -> CaseContext:
        raw_observations = payload.get("observations", [])
        observation_items = raw_observations if isinstance(raw_observations, list) else []
        observations = [
            ClinicalObservation(code=str(item["code"]), value=item.get("value"), unit=item.get("unit"), source=item.get("source"))
            for item in observation_items
            if isinstance(item, dict) and "code" in item
        ]
        raw_imaging = payload.get("imaging", [])
        imaging_items = raw_imaging if isinstance(raw_imaging, list) else []
        imaging = [
            ImagingStudy(
                study_id=str(item["study_id"]),
                modality=Modality(str(item["modality"])),
                series_paths=list(item.get("series_paths", [])) if isinstance(item.get("series_paths", []), list) else [],
                metadata=dict(item.get("metadata", {})) if isinstance(item.get("metadata", {}), dict) else {},
            )
            for item in imaging_items
            if isinstance(item, dict) and "study_id" in item and "modality" in item
        ]
        raw_demographics = payload.get("demographics", {})
        raw_provenance = payload.get("provenance", {})
        return CaseContext(
            case_id=str(payload["case_id"]),
            patient_id=str(payload["patient_id"]) if payload.get("patient_id") is not None else None,
            demographics=dict(raw_demographics) if isinstance(raw_demographics, dict) else {},
            observations=observations,
            imaging=imaging,
            report_text=str(payload.get("report_text", "")),
            ehr_text=str(payload.get("ehr_text", "")),
            provenance=dict(raw_provenance) if isinstance(raw_provenance, dict) else {},
        )
