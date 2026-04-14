from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..types import CaseContext, ClinicalObservation, ImagingStudy, Modality


@dataclass(slots=True)
class ClinicalIngestionPipeline:
    """Collect raw case assets into a single canonical case object."""

    def from_payload(self, payload: Mapping[str, object]) -> CaseContext:
        observations = [
            ClinicalObservation(code=item["code"], value=item.get("value"), unit=item.get("unit"), source=item.get("source"))
            for item in payload.get("observations", [])
        ]
        imaging = [
            ImagingStudy(
                study_id=item["study_id"],
                modality=Modality(item["modality"]),
                series_paths=list(item.get("series_paths", [])),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload.get("imaging", [])
        ]
        return CaseContext(
            case_id=str(payload["case_id"]),
            patient_id=payload.get("patient_id"),
            demographics=dict(payload.get("demographics", {})),
            observations=observations,
            imaging=imaging,
            report_text=str(payload.get("report_text", "")),
            ehr_text=str(payload.get("ehr_text", "")),
            provenance=dict(payload.get("provenance", {})),
        )
