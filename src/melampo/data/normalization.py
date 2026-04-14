from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from ..clinical.schemas import FHIRResourceEnvelope
from ..types import CaseContext


@dataclass(slots=True)
class ClinicalNormalizer:
    """Normalize raw cases into clinically interoperable envelopes."""

    def to_fhir_bundle(self, case: CaseContext) -> Dict[str, FHIRResourceEnvelope]:
        return {
            "patient": FHIRResourceEnvelope("Patient", {"id": case.patient_id or case.case_id}),
            "diagnostic_report": FHIRResourceEnvelope(
                "DiagnosticReport",
                {
                    "id": f"report-{case.case_id}",
                    "conclusion": case.report_text,
                    "subject": {"reference": case.patient_id or case.case_id},
                },
            ),
        }
