from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..models.specialist_adapters import ClaudeCritiqueAdapter, Gemma4ClinicalReasoningAdapter, Pillar0RadiologyAdapter
from .model_capability_registry import ModelCapabilityRegistry


def _capability(registry: ModelCapabilityRegistry, name: str) -> dict[str, Any]:
    try:
        return registry.get(name).describe()
    except KeyError:
        return {"name": name, "status": "capability_not_registered"}


@dataclass(slots=True)
class SpecialistRuntime:
    """Governed bridge between capability registry and safe specialist adapters.

    The runtime centralizes all optional specialist-model wiring. Default
    adapters are disabled, so constructing and using this runtime never performs
    hidden network calls. External outputs are converted to area signals and are
    explicitly marked as non-final.
    """

    registry: ModelCapabilityRegistry = field(default_factory=ModelCapabilityRegistry.build_default)
    radiology_adapter: Pillar0RadiologyAdapter = field(default_factory=Pillar0RadiologyAdapter)
    text_adapter: Gemma4ClinicalReasoningAdapter = field(default_factory=Gemma4ClinicalReasoningAdapter)
    critic_adapter: ClaudeCritiqueAdapter = field(default_factory=ClaudeCritiqueAdapter)

    def radiology_signal(
        self,
        study_id: str,
        series_paths: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.radiology_adapter.infer_volume(study_id=study_id, series_paths=series_paths, metadata=metadata)
        return {
            "capability": _capability(self.registry, "Pillar-0"),
            "response": response.as_dict(),
            "area_signal": response.as_area_signal("visual_diagnostic"),
            "external_model_is_final_arbiter": False,
            "hidden_network_call": False,
            "governance": {
                "specialist_models_are_signal_providers_only": True,
                "final_authority": "MelampoDiagnosticOrchestrator",
                "research_use_only": True,
            },
        }

    def grounded_text_signal(
        self,
        case_id: str,
        text: str,
        grounding: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.text_adapter.reason_over_text(case_id=case_id, text=text, grounding=grounding)
        return {
            "capability": _capability(self.registry, "Gemma 4"),
            "response": response.as_dict(),
            "area_signal": response.as_area_signal("language_listening"),
            "external_model_is_final_arbiter": False,
            "hidden_network_call": False,
            "governance": {
                "language_model_must_be_grounded": True,
                "specialist_models_are_signal_providers_only": True,
                "final_authority": "MelampoDiagnosticOrchestrator",
                "research_use_only": True,
            },
        }

    def external_critique(
        self,
        diagnostic_result: dict[str, Any],
        literature_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.critic_adapter.critique(
            diagnostic_result=diagnostic_result,
            literature_context=literature_context,
        )
        return {
            "capability": _capability(self.registry, "Claude Healthcare/Life Sciences"),
            "response": response.as_dict(),
            "external_critic_is_final_arbiter": False,
            "hidden_network_call": False,
            "governance": {
                "external_critic_only": True,
                "final_authority": "MelampoDiagnosticOrchestrator",
                "human_review_before_clinical_use": True,
            },
        }
