from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Modality(str, Enum):
    CT_3D = "ct_3d"
    MRI_3D = "mri_3d"
    WSI = "wsi"
    EHR_TEXT = "ehr_text"
    REPORT_TEXT = "report_text"
    LABS = "labs"
    DEMOGRAPHICS = "demographics"
    CR = "CR"
    DX = "DX"
    XR = "XR"
    CT = "CT"
    MR = "MR"
    US = "US"
    PT = "PT"
    DICOM = "DICOM"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str):
            normalized = value.strip().upper()
            aliases = {
                "CHEST_XRAY": "CR",
                "CXR": "CR",
                "XRAY": "XR",
                "X_RAY": "XR",
                "MRI": "MR",
                "PET": "PT",
            }
            normalized = aliases.get(normalized, normalized)
            for member in cls:
                if member.value == normalized or member.name == normalized:
                    return member
        return None


class SyntheticCaseType(str, Enum):
    REPLAY_CASE = "replay_case"
    RARE_CASE = "rare_case"
    BOUNDARY_CASE = "boundary_case"
    COUNTERFACTUAL = "counterfactual"


@dataclass(slots=True)
class ClinicalObservation:
    code: str
    value: Any
    unit: str | None = None
    source: str | None = None


@dataclass(slots=True)
class ImagingStudy:
    study_id: str
    modality: Modality
    series_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CaseContext:
    case_id: str
    patient_id: str | None = None
    demographics: dict[str, Any] = field(default_factory=dict)
    observations: list[ClinicalObservation] = field(default_factory=list)
    imaging: list[ImagingStudy] = field(default_factory=list)
    report_text: str = ""
    ehr_text: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EncodedCase:
    case_id: str
    feature_map: dict[str, Any] = field(default_factory=dict)
    multimodal_embedding: Any | None = None


@dataclass(slots=True)
class DifferentialHypothesis:
    label: str
    score: float
    rationale: str
    supporting_evidence: list[str] = field(default_factory=list)
    contradicting_evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class UncertaintyProfile:
    aleatoric: float = 0.0
    epistemic: float = 0.0
    grounding: float = 0.0
    shift: float = 0.0

    @property
    def total(self) -> float:
        return self.aleatoric + self.epistemic + self.grounding + self.shift


@dataclass(slots=True)
class DifferentialState:
    case_id: str
    hypotheses: list[DifferentialHypothesis] = field(default_factory=list)
    uncertainty: UncertaintyProfile = field(default_factory=UncertaintyProfile)
    abstain: bool = False
    escalation_reasons: list[str] = field(default_factory=list)
    evidence_trace: list[str] = field(default_factory=list)
    latent_state: Any | None = None


@dataclass(slots=True)
class SyntheticCase:
    synthetic_id: str
    case_type: SyntheticCaseType
    generated_context: CaseContext
    coherence_score: float
    accepted: bool
    provenance: dict[str, Any] = field(default_factory=dict)
