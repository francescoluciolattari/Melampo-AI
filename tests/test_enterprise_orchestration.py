from melampo.memory.vector_memory import VectorMemoryStore
from melampo.memory.weaviate_schema import MelampoWeaviateSchema
from melampo.orchestration.model_capability_registry import ModelCapabilityRegistry
from melampo.reasoning.diagnostic_orchestrator import MelampoDiagnosticOrchestrator


def test_model_capability_registry_records_core_decisions():
    registry = ModelCapabilityRegistry.build_default()
    visual = registry.select(area="visual_diagnostic")
    language = registry.select(area="language_listening")
    record = registry.decision_record()

    assert visual[0]["name"] == "Pillar-0"
    assert language[0]["name"] == "Gemma 4"
    assert record["final_diagnostic_authority"] == "MelampoDiagnosticOrchestrator"
    assert record["external_models_are_not"] == "sole_final_diagnostic_arbiters"


def test_weaviate_schema_contains_clinical_object_graph():
    schema = MelampoWeaviateSchema().as_dict()
    class_names = {class_schema["class"] for class_schema in schema["classes"]}

    assert "Symptom" in class_names
    assert "Pathology" in class_names
    assert "ClinicalCase" in class_names
    assert "ImagingStudy" in class_names
    assert "ClinicalDocument" in class_names
    assert schema["backend"] == "Weaviate"


def test_vector_memory_prefers_weaviate_and_preserves_relations():
    memory = VectorMemoryStore.enterprise_default()
    record = memory.upsert(
        text="Fever and cough may suggest infectious pathology.",
        metadata={
            "record_id": "symptom:febrile_cough",
            "focus": "language_listening",
            "ontology_refs": ["SNOMED:placeholder"],
            "relations": [{"from": "Symptom:Fever", "predicate": "suggestsPathology", "to": "Pathology:Infection"}],
        },
        source="unit_test",
    )
    hits = memory.search("cough infection", limit=1)

    assert memory.describe()["recommended_enterprise_backend"] == "weaviate_object_property_semantic_graph_rag"
    assert record.record_id == "symptom:febrile_cough"
    assert hits[0]["ontology_refs"] == ["SNOMED:placeholder"]
    assert hits[0]["relations"][0]["predicate"] == "suggestsPathology"


def test_diagnostic_orchestrator_abstains_when_policy_requires_it():
    orchestrator = MelampoDiagnosticOrchestrator()
    result = orchestrator.orchestrate(
        {
            "case_id": "case-1",
            "area_dynamics": {
                "pi_score": 0.1,
                "neuro_dynamic_metrics": {"mismatch_index": 0.9, "candidate_temperature": 1.2},
            },
            "coordinated": {
                "state_summary": {"uncertainty": 0.8},
                "differential": {"hypotheses": [{"label": "candidate_a", "score": 0.4}]},
                "policy": {"abstain": True, "escalate": True},
                "trace": ["trace-entry"],
            },
            "intuition": {"intuition": "candidate_a", "candidate_scores": []},
            "dream": {},
            "critique": {},
            "area_signals": {},
            "retrieval": {},
        }
    )

    assert result["result_label"] == "abstain_or_escalate"
    assert result["policy"]["abstain"] is True
    assert result["policy"]["escalate"] is True
    assert result["audit_trace"]["final_authority"] == "MelampoDiagnosticOrchestrator"
