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
            "relations": [
                {
                    "from": "Symptom:Fever",
                    "predicate": "suggestsPathology",
                    "to": "Pathology:Infection",
                }
            ],
        },
        source="unit_test",
    )
    hits = memory.search("cough infection", limit=1)

    assert (
        memory.describe()["recommended_enterprise_backend"]
        == "weaviate_object_property_semantic_graph_rag"
    )
    assert record.record_id == "symptom:febrile_cough"
    assert hits[0]["ontology_refs"] == ["SNOMED:placeholder"]
    assert hits[0]["relations"][0]["predicate"] == "suggestsPathology"


def test_diagnostic_orchestrator_abstains_when_policy_requires_it():
    orchestrator = MelampoDiagnosticOrchestrator()
    result = orchestrator.orchestrate(
        {
            "case_id": "case-1",
            "discordances": {"count": 1, "kinds": {"card": 1}},
            "coordinated": {
                "state_summary": {},
                "differential": {
                    "hypotheses": [{"label": "candidate_a", "score": 0.4}]
                },
                "policy": {"abstain": True, "escalate": True},
                "trace": ["trace-entry"],
            },
            "intuition": {"intuition": "candidate_a", "candidate_scores": []},
            "nexus": {},
            "critique": {},
            "area_signals": {},
            "retrieval": {},
        }
    )

    assert result["result_label"] == "abstain_or_escalate"
    assert result["policy"]["abstain"] is True
    assert result["policy"]["escalate"] is True
    assert result["discordances"]["count"] == 1
    assert result["audit_trace"]["final_authority"] == "MelampoDiagnosticOrchestrator"


def test_the_discordances_escalate_without_withholding_the_result():
    """Principle of doubt (28 November 2026): a discordance goes to a person; the result is not withheld for it,
    and nothing abstains on the measures of the coherence between areas any more."""
    result = MelampoDiagnosticOrchestrator().orchestrate(
        {
            "case_id": "case-2",
            "discordances": {"count": 2, "kinds": {"card": 2}},
            "coordinated": {
                "state_summary": {},
                "differential": {"hypotheses": [{"label": "pneumonia", "score": 0.8}]},
                "policy": {"abstain": False, "escalate": False},
            },
            "case_diagnosis": {"status": "ok", "ranking": [{"disease": "MONDO:0005249", "words": "pneumonia",
                                                           "posterior": 0.7, "branch": "candidate"}]},
            "intuition": {"intuition": "pneumonia", "candidate_scores": []},
        }
    )
    assert result["result_label"] == "pneumonia"
    assert result["policy"]["escalate"] is True
    assert result["policy"]["reasons"] == []


def test_the_result_comes_from_case_diagnosis_and_the_october_path_is_only_compared():
    """28 November 2026: the diagnosis is case_diagnosis's ranking; the October path (DifferentialEngine) is kept in
    the result for comparison only, until the evaluation of December 2026."""
    result = MelampoDiagnosticOrchestrator().orchestrate(
        {
            "case_id": "case-4",
            "coordinated": {
                "differential": {"hypotheses": [{"label": "candidate_1", "score": 2.0, "hypothesis_type": "x"}]},
                "policy": {"abstain": False, "escalate": False, "reasons": [], "flags": []},
            },
            "case_diagnosis": {
                "status": "ok",
                "ranking": [
                    {"disease": "MONDO:1", "words": "marfan syndrome", "posterior": 0.8, "log10_odds": 1.2,
                     "branch": "candidate", "origin": "proposed", "contributions": [{"manifestation": "HP:1"}]},
                    {"disease": "MONDO:2", "words": "loeys-dietz syndrome", "posterior": 0.2, "branch": "candidate"},
                ],
                "doubts": [{"disease": "marfan syndrome", "manifestations": ["HP:2"]}],
                "exams_to_propose": [{"disease": "marfan syndrome", "manifestation": "HP:3"}],
                "unexplained_findings": [],
            },
        }
    )
    assert result["result_label"] == "marfan syndrome"
    assert [h["label"] for h in result["differential"]] == ["marfan syndrome", "loeys-dietz syndrome"]
    assert result["differential"][0]["source"] == "case_diagnosis"
    assert result["diagnosis"]["exams_to_propose"] == [{"disease": "marfan syndrome", "manifestation": "HP:3"}]
    assert result["diagnosis"]["doubts"][0]["manifestations"] == ["HP:2"]
    assert result["october_path"]["role"] == "comparison_only_until_december_2026_evaluation"
    assert result["october_path"]["top"] == "candidate_1"
    assert result["october_path"]["top_is_placeholder"] is True
    assert result["policy"]["flags"] == []


def test_without_case_diagnosis_there_is_no_candidate_and_no_fallback():
    result = MelampoDiagnosticOrchestrator().orchestrate(
        {
            "case_id": "case-3",
            "coordinated": {
                "differential": {"hypotheses": [{"label": "candidate_1", "score": 2.0}]},
                "policy": {"abstain": False, "escalate": False, "reasons": [], "flags": ["no_evidence"]},
            },
        }
    )
    assert result["result_label"] == "no_candidate_found"
    assert result["differential"] == []
    assert result["policy"]["flags"] == ["no_evidence", "case_diagnosis:not_run"]
