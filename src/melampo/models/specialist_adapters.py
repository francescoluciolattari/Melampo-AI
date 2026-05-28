from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model_card import ModelCard
from .model_client import ModelClientConfig, SafeModelClient
from .model_response_schema import ClinicalClaim, SpecialistModelResponse


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _safe_claims(payload: dict[str, Any], default_claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    claims = payload.get("claims") if isinstance(payload, dict) else None
    return list(claims) if isinstance(claims, list) and claims else default_claims


@dataclass(slots=True)
class Pillar0RadiologyAdapter:
    """Safe adapter for Pillar-0 style radiology/volumetric imaging signals.

    The adapter supports disabled, dry-run, mock, local subprocess and explicit
    HTTP JSON execution through SafeModelClient. No external model is called in
    the default configuration.
    """

    provider: str = "pillar_0_research_backend"
    model_name: str = "Pillar-0"
    enabled: bool = False
    endpoint: str | None = None
    execution_mode: str = "disabled"
    client_config: ModelClientConfig | None = None
    client: SafeModelClient | None = None

    def _client(self) -> SafeModelClient:
        if self.client is not None:
            return self.client
        config = self.client_config or ModelClientConfig(
            mode=self.execution_mode,
            enabled=self.enabled,
            endpoint=self.endpoint,
            allow_remote=False,
        )
        return SafeModelClient(provider=self.provider, model_name=self.model_name, role="primary_radiology_foundation_model", config=config)

    def prepare_volume_request(self, study_id: str, series_paths: list[str], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        metadata = metadata or {}
        return {
            "study_id": study_id,
            "series_paths": list(series_paths),
            "metadata": metadata,
            "modality": metadata.get("modality", "unknown"),
            "series_count": len(series_paths),
            "input_kind": "radiology_volume_or_series",
            "endpoint_configured": self.endpoint is not None,
            "enabled": self.enabled,
            "governance": {
                "research_use_only": True,
                "not_final_diagnostic_arbiter": True,
                "requires_local_validation": True,
            },
        }

    def infer_volume(self, study_id: str, series_paths: list[str], metadata: dict[str, Any] | None = None) -> SpecialistModelResponse:
        request = self.prepare_volume_request(study_id=study_id, series_paths=series_paths, metadata=metadata)
        result = self._client().execute(request)
        mode = result.get("mode", self.execution_mode)
        trace = result.get("trace", {})
        if result["status"] == "not_called":
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
                provenance={"request": request, "mode": mode},
                limitations=["network_call_not_implemented", "research_use_only", "not_final_diagnostic_arbiter"],
                audit_trace={"model_execution": trace},
            )
        if result["status"] in {"request_prepared", "blocked", "failed"}:
            return SpecialistModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                role="primary_radiology_foundation_model",
                status=str(result["status"]),
                signals={"study_id": study_id, "routing_hint": "pillar_0_call_prepared", "mode": mode},
                confidence=0.0,
                uncertainty=1.0,
                provenance={"request": request, "mode": mode, "reason": result.get("reason"), "error": result.get("error")},
                limitations=["actual_inference_adapter_required", "research_use_only"],
                audit_trace={"model_execution": trace},
            )
        response = dict(result.get("response", {}))
        signals = dict(response.get("signals", {}))
        signals.setdefault("study_id", study_id)
        signals.setdefault("modality", request.get("modality", "unknown"))
        confidence = _clamp(float(response.get("confidence", 0.0) or 0.0))
        uncertainty = _clamp(float(response.get("uncertainty", 1.0 - confidence) or 0.0))
        default_claims = [
            ClinicalClaim(
                claim_id=f"pillar0:{study_id}:volume_signal",
                type="imaging_signal",
                normalized_entity=str(signals.get("primary_finding", "radiology_volume_signal")),
                polarity="present",
                confidence=confidence,
                uncertainty=uncertainty,
                ontology_refs=list(response.get("ontology_refs", [])),
                evidence_refs=[study_id],
                source_area="visual_diagnostic",
            ).as_dict()
        ]
        return SpecialistModelResponse.from_payload(
            {**response, "signals": signals, "claims": _safe_claims(response, default_claims), "status": result.get("status", "completed")},
            provider=self.provider,
            model_name=self.model_name,
            role="primary_radiology_foundation_model",
            provenance={"request": request, "mode": mode, "trace": trace},
            limitations=["research_use_only", "not_final_diagnostic_arbiter"],
        )

    def model_card(self) -> ModelCard:
        return ModelCard(
            name=self.model_name,
            provider=self.provider,
            role="primary_radiology_foundation_model",
            intended_use="Generate governed CT/MRI/radiology-volume signals for visual_diagnostic_area.",
            modalities=["ct_3d", "mri_3d", "radiology_volume"],
            limitations=["Not final diagnostic authority", "Requires local radiology validation", "Disabled unless explicitly configured"],
            validation_requirements=["Radiology benchmark", "Calibration", "Slice analysis by modality and anatomy"],
        )


@dataclass(slots=True)
class Gemma4ClinicalReasoningAdapter:
    """Safe adapter for Gemma 4 style grounded clinical reasoning."""

    provider: str = "gemma_4_local_or_private_backend"
    model_name: str = "Gemma 4"
    enabled: bool = False
    endpoint: str | None = None
    execution_mode: str = "disabled"
    client_config: ModelClientConfig | None = None
    client: SafeModelClient | None = None

    def _client(self) -> SafeModelClient:
        if self.client is not None:
            return self.client
        config = self.client_config or ModelClientConfig(mode=self.execution_mode, enabled=self.enabled, endpoint=self.endpoint, allow_remote=False)
        return SafeModelClient(provider=self.provider, model_name=self.model_name, role="clinical_text_and_agentic_reasoning", config=config)

    def prepare_reasoning_request(self, case_id: str, text: str, grounding: dict[str, Any] | None = None) -> dict[str, Any]:
        grounding = grounding or {}
        return {
            "case_id": case_id,
            "text": text,
            "text_preview": text[:300],
            "grounding": grounding,
            "grounding_available": bool(grounding),
            "required_output_schema": [
                "grounded_summary",
                "clinical_claims",
                "differential_suggestions",
                "missing_evidence",
                "contradictions",
                "uncertainty",
                "source_refs",
            ],
            "governance": {
                "must_be_grounded_by_rag": True,
                "not_final_diagnostic_arbiter": True,
                "return_uncertainty_and_provenance": True,
            },
        }

    def reason_over_text(self, case_id: str, text: str, grounding: dict[str, Any] | None = None) -> SpecialistModelResponse:
        request = self.prepare_reasoning_request(case_id=case_id, text=text, grounding=grounding)
        result = self._client().execute(request)
        mode = result.get("mode", self.execution_mode)
        trace = result.get("trace", {})
        if result["status"] == "not_called":
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
                provenance={"mode": mode, "grounding_keys": sorted((grounding or {}).keys())},
                limitations=["must_be_grounded_by_rag", "not_final_diagnostic_arbiter"],
                audit_trace={"model_execution": trace},
            )
        if result["status"] in {"request_prepared", "blocked", "failed"}:
            return SpecialistModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                role="clinical_text_and_agentic_reasoning",
                status=str(result["status"]),
                signals={"case_id": case_id, "routing_hint": "gemma_4_call_prepared", "mode": mode},
                confidence=0.0,
                uncertainty=1.0,
                provenance={"request": request, "mode": mode, "reason": result.get("reason"), "error": result.get("error")},
                limitations=["actual_inference_adapter_required", "must_be_grounded_by_rag"],
                audit_trace={"model_execution": trace},
            )
        response = dict(result.get("response", {}))
        confidence = _clamp(float(response.get("confidence", 0.0) or 0.0))
        uncertainty = _clamp(float(response.get("uncertainty", 1.0 - confidence) or 0.0))
        default_claims = [
            ClinicalClaim(
                claim_id=f"gemma4:{case_id}:grounded_text",
                type="clinical_text_reasoning",
                normalized_entity=str(response.get("grounded_summary", case_id))[:120],
                polarity="present",
                confidence=confidence,
                uncertainty=uncertainty,
                evidence_refs=list(response.get("source_refs", [])),
                source_area="language_listening",
            ).as_dict()
        ]
        signals = dict(response.get("signals", {}))
        signals.setdefault("case_id", case_id)
        signals.setdefault("grounded_summary", response.get("grounded_summary", ""))
        return SpecialistModelResponse.from_payload(
            {**response, "signals": signals, "claims": _safe_claims(response, default_claims), "status": result.get("status", "completed")},
            provider=self.provider,
            model_name=self.model_name,
            role="clinical_text_and_agentic_reasoning",
            provenance={"request": request, "mode": mode, "trace": trace},
            limitations=["must_be_grounded_by_rag", "not_final_diagnostic_arbiter"],
        )

    def model_card(self) -> ModelCard:
        return ModelCard(
            name=self.model_name,
            provider=self.provider,
            role="clinical_text_and_agentic_reasoning",
            intended_use="Reason over RAG-grounded clinical text and emit structured claims.",
            modalities=["report_text", "ehr_text", "clinical_text", "tool_trace"],
            limitations=["Not a standalone medical specialist", "Requires RAG grounding", "Disabled unless explicitly configured"],
            validation_requirements=["Groundedness evaluation", "Faithfulness evaluation", "Human review before clinical use"],
        )


@dataclass(slots=True)
class ClaudeCritiqueAdapter:
    """Safe adapter for Claude Healthcare/Life Sciences style critique."""

    provider: str = "claude_healthcare_life_sciences_optional_backend"
    model_name: str = "Claude Healthcare/Life Sciences"
    enabled: bool = False
    endpoint: str | None = None
    execution_mode: str = "disabled"
    client_config: ModelClientConfig | None = None
    client: SafeModelClient | None = None

    def _client(self) -> SafeModelClient:
        if self.client is not None:
            return self.client
        config = self.client_config or ModelClientConfig(mode=self.execution_mode, enabled=self.enabled, endpoint=self.endpoint, allow_remote=False)
        return SafeModelClient(provider=self.provider, model_name=self.model_name, role="external_critic_and_scientific_research", config=config)

    def prepare_critique_request(self, diagnostic_result: dict[str, Any], literature_context: dict[str, Any] | None = None) -> dict[str, Any]:
        literature_context = literature_context or {}
        return {
            "diagnostic_result": diagnostic_result,
            "literature_context": literature_context,
            "required_output_schema": [
                "critique_status",
                "unsupported_claims",
                "safety_flags",
                "regulatory_flags",
                "recommended_action",
                "confidence_in_critique",
            ],
            "governance": {
                "external_critic_only": True,
                "not_final_diagnostic_arbiter": True,
                "human_review_before_clinical_use": True,
            },
        }

    def critique(self, diagnostic_result: dict[str, Any], literature_context: dict[str, Any] | None = None) -> SpecialistModelResponse:
        request = self.prepare_critique_request(diagnostic_result=diagnostic_result, literature_context=literature_context)
        result = self._client().execute(request)
        mode = result.get("mode", self.execution_mode)
        trace = result.get("trace", {})
        if result["status"] == "not_called":
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
                provenance={"mode": mode},
                limitations=["optional_external_critic", "not_final_diagnostic_arbiter"],
                audit_trace={"model_execution": trace},
            )
        if result["status"] in {"request_prepared", "blocked", "failed"}:
            return SpecialistModelResponse(
                provider=self.provider,
                model_name=self.model_name,
                role="external_critic_and_scientific_research",
                status=str(result["status"]),
                signals={"routing_hint": "claude_critic_call_prepared", "mode": mode},
                confidence=0.0,
                uncertainty=1.0,
                provenance={"request": request, "mode": mode, "reason": result.get("reason"), "error": result.get("error")},
                limitations=["actual_inference_adapter_required", "external_critic_only"],
                audit_trace={"model_execution": trace},
            )
        response = dict(result.get("response", {}))
        confidence = _clamp(float(response.get("confidence", response.get("confidence_in_critique", 0.0)) or 0.0))
        uncertainty = _clamp(float(response.get("uncertainty", 1.0 - confidence) or 0.0))
        signals = dict(response.get("signals", {}))
        signals.setdefault("critique_status", response.get("critique_status", "needs_review"))
        signals.setdefault("recommended_action", response.get("recommended_action", "human_review"))
        return SpecialistModelResponse.from_payload(
            {
                **response,
                "signals": signals,
                "confidence": confidence,
                "uncertainty": uncertainty,
                "status": result.get("status", "completed"),
            },
            provider=self.provider,
            model_name=self.model_name,
            role="external_critic_and_scientific_research",
            provenance={"request": request, "mode": mode, "trace": trace},
            limitations=["optional_external_critic", "not_final_diagnostic_arbiter"],
        )

    def model_card(self) -> ModelCard:
        return ModelCard(
            name=self.model_name,
            provider=self.provider,
            role="external_critic_and_scientific_research",
            intended_use="Optional external critique and unsupported-claim/safety review.",
            modalities=["clinical_text", "literature", "tool_trace", "policy_trace"],
            limitations=["External critic only", "Cannot override final orchestrator", "Requires audit trace and human review"],
            validation_requirements=["Critique benchmark", "Unsupported-claim detection", "Privacy and governance review"],
        )
