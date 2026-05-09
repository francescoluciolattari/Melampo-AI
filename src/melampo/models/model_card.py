from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelCard:
    name: str
    provider: str
    role: str
    intended_use: str
    modalities: list[str]
    deployment_status: str = "contract_or_research"
    limitations: list[str] = field(default_factory=list)
    safety_boundary: str = "Research scaffold only; not validated for autonomous diagnosis."
    validation_requirements: list[str] = field(default_factory=list)
    governance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "role": self.role,
            "intended_use": self.intended_use,
            "modalities": self.modalities,
            "deployment_status": self.deployment_status,
            "limitations": self.limitations,
            "safety_boundary": self.safety_boundary,
            "validation_requirements": self.validation_requirements,
            "governance": self.governance,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Model Card: {self.name}",
            "",
            f"- Provider: {self.provider}",
            f"- Role: {self.role}",
            f"- Deployment status: {self.deployment_status}",
            f"- Intended use: {self.intended_use}",
            f"- Safety boundary: {self.safety_boundary}",
            "",
            "## Modalities",
        ]
        lines.extend(f"- {item}" for item in self.modalities)
        lines.append("")
        lines.append("## Limitations")
        lines.extend(f"- {item}" for item in (self.limitations or ["Not specified"]))
        lines.append("")
        lines.append("## Validation requirements")
        lines.extend(f"- {item}" for item in (self.validation_requirements or ["Not specified"]))
        return "\n".join(lines) + "\n"


def default_phase4a_model_cards() -> list[ModelCard]:
    return [
        ModelCard(
            name="Pillar-0",
            provider="research_radiology_foundation_model",
            role="primary_radiology_foundation_model",
            intended_use="Generate governed radiology/volumetric imaging signals for visual_diagnostic_area.",
            modalities=["ct_3d", "mri_3d", "radiology_volume"],
            limitations=[
                "Not final diagnostic authority",
                "Requires local dataset validation and calibration",
                "Real backend must be explicitly configured",
            ],
            validation_requirements=[
                "DICOM/volume preprocessing validation",
                "Radiology benchmark by modality and anatomy",
                "Calibration and abstention analysis",
            ],
        ),
        ModelCard(
            name="Gemma 4",
            provider="local_or_private_open_weight_reasoning_backend",
            role="clinical_text_and_agentic_reasoning",
            intended_use="Reason over RAG-grounded clinical text and emit structured claims for language/context areas.",
            modalities=["report_text", "ehr_text", "clinical_text", "tool_trace"],
            limitations=[
                "Must be grounded by retrieved evidence",
                "Must return uncertainty and missing evidence",
                "Not final diagnostic authority",
            ],
            validation_requirements=[
                "Groundedness and faithfulness evaluation",
                "Safety rail evaluation",
                "Human review before clinical use",
            ],
        ),
        ModelCard(
            name="Claude Healthcare/Life Sciences",
            provider="external_optional_critic_backend",
            role="external_critic_and_scientific_research",
            intended_use="Provide optional critique, unsupported-claim detection, and regulatory/safety review.",
            modalities=["clinical_text", "literature", "tool_trace", "policy_trace"],
            limitations=[
                "External critic only",
                "Cannot override MelampoDiagnosticOrchestrator",
                "Requires audit trace and human review",
            ],
            validation_requirements=[
                "Critique quality evaluation",
                "Unsupported-claim detection benchmark",
                "Privacy and governance review for external calls",
            ],
        ),
    ]
