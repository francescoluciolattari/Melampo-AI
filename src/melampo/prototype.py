from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .app import build_default_runtime
from .config import RuntimeConfig, build_default_config


@dataclass(slots=True)
class PrototypeInputValidator:
    """Validate minimal clinical prototype payloads before running the core pipeline."""

    required_fields: tuple[str, ...] = ("case_id",)
    narrative_fields: tuple[str, ...] = ("report_text", "ehr_text", "patient_complaints")

    def validate(self, payload: Mapping[str, Any]) -> dict:
        errors = []
        warnings = []
        for field_name in self.required_fields:
            if not payload.get(field_name):
                errors.append(f"missing_required_field:{field_name}")
        if not any(payload.get(field_name) for field_name in self.narrative_fields) and not payload.get("imaging"):
            warnings.append("minimal_payload_without_narrative_or_imaging")
        return {
            "valid": not errors,
            "errors": errors,
            "warnings": warnings,
            "field_count": len(payload),
        }


@dataclass(slots=True)
class ClinicalPrototypeRunner:
    """Application-level runner for an executable Melampo clinical research prototype."""

    config: RuntimeConfig = field(default_factory=build_default_config)
    validator: PrototypeInputValidator = field(default_factory=PrototypeInputValidator)

    @classmethod
    def from_profile(cls, runtime_profile: str = "local_research") -> "ClinicalPrototypeRunner":
        return cls(config=build_default_config(runtime_profile=runtime_profile))

    def run_case(self, payload: Mapping[str, Any]) -> dict:
        validation = self.validator.validate(payload)
        if not validation["valid"]:
            return {
                "status": "invalid_payload",
                "validation": validation,
                "case_id": payload.get("case_id"),
            }

        runtime = build_default_runtime(config=self.config)
        result = runtime.pipeline.run(dict(payload))
        coordinated = result.get("coordinated", {})
        differential = coordinated.get("differential", {})
        hypotheses = differential.get("hypotheses", [])
        critique = result.get("critique", {})
        top_hypothesis = hypotheses[0] if hypotheses else {}
        return {
            "status": "completed",
            "validation": validation,
            "runtime": runtime.describe(),
            "case_id": result.get("case_id"),
            "top_hypothesis": top_hypothesis,
            "recommended_actions": differential.get("recommended_actions", []),
            "prioritized_actions": critique.get("prioritized_actions", []),
            "policy": coordinated.get("policy", {}),
            "state_summary": coordinated.get("state_summary", {}),
            "raw_result": result,
        }


def run_prototype_case(payload: Mapping[str, Any], runtime_profile: str = "local_research") -> dict:
    """Run one clinical research prototype case using the requested runtime profile."""

    return ClinicalPrototypeRunner.from_profile(runtime_profile=runtime_profile).run_case(payload)
