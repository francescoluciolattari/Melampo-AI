from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..clinical.schemas import FHIRResourceEnvelope
from ..types import CaseContext


@dataclass(slots=True)
class FHIRMapper:
    """Convert a canonical case into a minimal FHIR-like resource bundle."""

    def map_case(self, case: CaseContext) -> List[FHIRResourceEnvelope]:
        resources = [FHIRResourceEnvelope("Patient", {"id": case.patient_id or case.case_id})]
        if case.report_text:
            resources.append(
                FHIRResourceEnvelope(
                    "DiagnosticReport",
                    {"id": f"dr-{case.case_id}", "conclusion": case.report_text},
                )
            )
        for index, observation in enumerate(case.observations):
            resources.append(
                FHIRResourceEnvelope(
                    "Observation",
                    {
                        "id": f"obs-{case.case_id}-{index}",
                        "code": observation.code,
                        "value": observation.value,
                        "unit": observation.unit,
                    },
                )
            )
        return resources
