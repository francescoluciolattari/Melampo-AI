from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SpecialistModelResponse:
    """Structured response contract for any external specialist model."""

    provider: str
    model_name: str
    role: str
    status: str
    signals: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    uncertainty: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def as_area_signal(self, area: str) -> dict[str, Any]:
        return {
            "area": area,
            "provider": self.provider,
            "model_name": self.model_name,
            "role": self.role,
            "status": self.status,
            "signals": self.signals,
            "signal_count": len(self.signals),
            "salience_score": round(max(0.0, min(1.0, self.confidence)), 3),
            "uncertainty_score": round(max(0.0, min(1.0, self.uncertainty)), 3),
            "provenance": self.provenance,
            "limitations": self.limitations,
        }


@dataclass(slots=True)
class Pillar0RadiologyAdapter:
    """Pillar-0 contract for radiology/volumetric imaging signals.

    This adapter intentionally does not execute model inference. It defines the
    enterprise request/response surface expected from a real Pillar-0 backend.
    """

    provider: str = "pillar_0_research_backend"
    model_name: str = "Pillar-0"
    enabled: bool = False
    endpoint: str | None = None

    def infer_volume(self, study_id: str, series_paths: list[str], metadata: dict[str, Any] | None = None) -> SpecialistModelResponse:
        metadata = metadata or {}
        request = {
            "study_id": study_id,
            "series_paths": series_paths,
            "metadata": metadata,
            "endpoint_configured": self.endpoint is not None,
            "enabled": self.enabled,
        }
        if not self.enabled or not self.endpoint:
            return SpecialistModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                role="primary_radiology_foundation_model",
                status="not_called",
                signals={
                    "study_id": study_id,
                    "input_kind": "radiology_volume_or_series",
                    "image_path_count": len(series_paths),
                    "routing_hint": "configure_pillar_0_backend_for_ct_mri",
                },
                confidence=0.0,
                uncertainty=1.0,
                provenance={"request": request, "mode": "contract_only"},
                limitations=["network_call_not_implemented", "research_use_only", "not_final_diagnostic_arbiter"],
            )
        return SpecialistModelResponse(
            provider=self.provider,
            model_name=self.model_name,
            role="primary_radiology_foundation_model",
            status="request_prepared",
            signals={"study_id": study_id, "routing_hint": "pillar_0_remote_call_ready"},
            confidence=0.0,
            uncertainty=1.0,
            provenance={"request": request, "mode": "network_call_stub"},
            limitations=["actual_inference_adapter_required", "research_use_only"],
        )


@dataclass(slots=True)
class Gemma4ClinicalReasoningAdapter:
    """Gemma 4 contract for grounded clinical text and agentic reasoning."""

    provider: str = "gemma_4_local_or_private_backend"
    model_name: str = "Gemma 4"
    enabled: bool = False
    endpoint: str | None = None

    def reason_over_text(self, case_id: str, text: str, grounding: dict[str, Any] | None = None) -> SpecialistModelResponse:
        grounding = grounding or {}
        if not self.enabled or not self.endpoint:
            return SpecialistModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                role="clinical_text_and_agentic_reasoning",
                status="not_called",
                signals={
                    "case_id": case_id,
                    "text_preview": text[:160],
                    "grounding_available": bool(grounding),
                    "routing_hint": "configure_gemma_4_backend_for_grounded_text_reasoning",
                },
                confidence=0.0,
                uncertainty=1.0,
                provenance={"mode": "contract_only", "grounding_keys": sorted(grounding.keys())},
                limitations=["must_be_grounded_by_rag", "not_final_diagnostic_arbiter"],
            )
        return SpecialistModelResponse(
            provider=self.provider,
            model_name=self.model_name,
            role="clinical_text_and_agentic_reasoning",
            status="request_prepared",
            signals={"case_id": case_id, "routing_hint": "gemma_4_remote_or_local_call_ready"},
            confidence=0.0,
            uncertainty=1.0,
            provenance={"mode": "network_call_stub"},
            limitations=["actual_inference_adapter_required"],
        )


@dataclass(slots=True)
class ClaudeCritiqueAdapter:
    """Claude Healthcare/Life Sciences style contract for critique and research review."""

    provider: str = "claude_healthcare_life_sciences_optional_backend"
    model_name: str = "Claude Healthcare/Life Sciences"
    enabled: bool = False
    endpoint: str | None = None

    def critique(self, diagnostic_result: dict[str, Any], literature_context: dict[str, Any] | None = None) -> SpecialistModelResponse:
        literature_context = literature_context or {}
        if not self.enabled or not self.endpoint:
            return SpecialistModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                role="external_critic_and_scientific_research",
                status="not_called",
                signals={
                    "result_label": diagnostic_result.get("result_label", "unknown"),
                    "literature_context_available": bool(literature_context),
                    "routing_hint": "configure_optional_claude_critic_for_second_opinion",
                },
                confidence=0.0,
                uncertainty=1.0,
                provenance={"mode": "contract_only"},
                limitations=["optional_external_critic", "not_final_diagnostic_arbiter"],
            )
        return SpecialistModelResponse(
            provider=self.provider,
            model_name=self.model_name,
            role="external_critic_and_scientific_research",
            status="request_prepared",
            signals={"routing_hint": "claude_critic_call_ready"},
            confidence=0.0,
            uncertainty=1.0,
            provenance={"mode": "network_call_stub"},
            limitations=["actual_inference_adapter_required"],
        )
