from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ModelCapability:
    """Provider-neutral description of a model or service capability.

    The registry records intended roles and guardrails. It does not hardcode
    credentials, endpoints, network calls, or clinical authority.
    """

    name: str
    provider: str
    role: str
    modalities: tuple[str, ...]
    areas: tuple[str, ...]
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    deployment: str = "external_or_optional"
    priority: int = 100
    clinical_authority: str = "signal_provider_only"
    metadata: dict[str, Any] = field(default_factory=dict)

    def supports(self, area: str | None = None, modality: str | None = None, role: str | None = None) -> bool:
        if area is not None and area not in self.areas:
            return False
        if modality is not None and modality not in self.modalities:
            return False
        if role is not None and role != self.role:
            return False
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "role": self.role,
            "modalities": list(self.modalities),
            "areas": list(self.areas),
            "strengths": list(self.strengths),
            "limitations": list(self.limitations),
            "deployment": self.deployment,
            "priority": self.priority,
            "clinical_authority": self.clinical_authority,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ModelCapabilityRegistry:
    """Enterprise model registry for Melampo's multimodal/multimodel strategy."""

    capabilities: dict[str, ModelCapability] = field(default_factory=dict)

    @classmethod
    def build_default(cls) -> "ModelCapabilityRegistry":
        registry = cls()
        registry.register_many(
            [
                ModelCapability(
                    name="Pillar-0",
                    provider="research_radiology_foundation_model",
                    role="primary_radiology_foundation_model",
                    modalities=("ct_3d", "mri_3d", "radiology_volume"),
                    areas=("visual_diagnostic",),
                    strengths=(
                        "3d_ct_mri_signal_extraction",
                        "radiology_finding_prioritization",
                        "visual_area_salience_and_uncertainty",
                    ),
                    limitations=(
                        "not_final_diagnostic_arbiter",
                        "requires_local_validation_and_dataset_governance",
                        "research_use_until_clinically_validated",
                    ),
                    priority=10,
                    metadata={"decision": "replaces_medgemma_as_primary_radiology_model"},
                ),
                ModelCapability(
                    name="Gemma 4",
                    provider="open_weight_general_reasoning_model",
                    role="clinical_text_and_agentic_reasoning",
                    modalities=("report_text", "ehr_text", "clinical_text", "tool_trace"),
                    areas=("language_listening", "case_context", "critique"),
                    strengths=(
                        "open_weight_reasoning",
                        "agentic_workflow_support",
                        "clinical_text_summarization_when_grounded_by_rag",
                    ),
                    limitations=(
                        "not_medical_specialist_without_rag_grounding",
                        "not_final_diagnostic_arbiter",
                        "must_return_uncertainty_and_provenance",
                    ),
                    priority=20,
                    metadata={"decision": "replaces_medgemma_for_non_radiology_language_functions"},
                ),
                ModelCapability(
                    name="Claude Healthcare/Life Sciences",
                    provider="anthropic_or_compatible_external_critic",
                    role="external_critic_and_scientific_research",
                    modalities=("clinical_text", "literature", "tool_trace", "policy_trace"),
                    areas=("critique", "metacognition", "regulatory_review"),
                    strengths=(
                        "second_opinion_critique",
                        "scientific_literature_reasoning",
                        "mcp_tool_workflow_review",
                        "regulatory_and_safety_review_support",
                    ),
                    limitations=(
                        "not_primary_imaging_model",
                        "not_final_diagnostic_arbiter",
                        "requires_audit_trace_and_human_review_for_clinical_use",
                    ),
                    deployment="external_optional",
                    priority=30,
                ),
                ModelCapability(
                    name="Weaviate",
                    provider="semantic_object_property_vector_database",
                    role="semantic_memory_and_ontology_rag",
                    modalities=("text_vector", "image_vector", "object_reference", "ontology_reference"),
                    areas=("semantic_memory", "epidemiology", "case_context", "retrieval"),
                    strengths=(
                        "object_property_clinical_memory",
                        "ontology_aware_relations",
                        "multimodal_named_vectors",
                        "rag_with_context_and_provenance",
                    ),
                    limitations=(
                        "requires_schema_governance",
                        "requires_source_license_tracking",
                        "not_a_reasoning_model_by_itself",
                    ),
                    priority=5,
                    clinical_authority="memory_substrate_only",
                ),
                ModelCapability(
                    name="Docling",
                    provider="document_intelligence_parser",
                    role="clinical_document_processing",
                    modalities=("pdf", "docx", "pptx", "image_document", "table", "formula"),
                    areas=("document_rag", "semantic_memory"),
                    strengths=(
                        "layout_aware_document_conversion",
                        "table_and_formula_preservation",
                        "rag_ready_structured_chunks",
                    ),
                    limitations=(
                        "parser_not_reasoner",
                        "requires_downstream_clinical_chunk_validation",
                    ),
                    priority=15,
                    clinical_authority="ingestion_substrate_only",
                ),
            ]
        )
        return registry

    def register(self, capability: ModelCapability) -> None:
        self.capabilities[capability.name] = capability

    def register_many(self, capabilities: list[ModelCapability]) -> None:
        for capability in capabilities:
            self.register(capability)

    def get(self, name: str) -> ModelCapability:
        return self.capabilities[name]

    def select(self, area: str | None = None, modality: str | None = None, role: str | None = None) -> list[dict[str, Any]]:
        matches = [
            capability
            for capability in self.capabilities.values()
            if capability.supports(area=area, modality=modality, role=role)
        ]
        matches.sort(key=lambda item: item.priority)
        return [capability.describe() for capability in matches]

    def decision_record(self) -> dict[str, Any]:
        return {
            "strategy": "multimodal_multimodel_signal_orchestration",
            "final_diagnostic_authority": "MelampoDiagnosticOrchestrator",
            "external_models_are": "specialist_signal_providers_and_critics",
            "external_models_are_not": "sole_final_diagnostic_arbiters",
            "capabilities": [capability.describe() for capability in sorted(self.capabilities.values(), key=lambda item: item.priority)],
        }
