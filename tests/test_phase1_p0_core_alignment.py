from melampo.memory.retriever import MemoryRetriever
from melampo.memory.semantic_memory import SemanticMemoryStore
from melampo.reasoning.clinical_pipeline import _derive_governance_scores


def test_memory_retriever_uses_semantic_vector_store_before_fallback():
    memory = SemanticMemoryStore()
    memory.add_document(
        {
            "record_id": "doc:pneumonia-guideline",
            "text": "Fever and cough can support pneumonia differential reasoning with imaging correlation.",
            "source": "unit_test_guideline",
            "learning_status": "promoted",
            "metadata": {
                "focus": "language_listening",
                "ontology_refs": ["SNOMED:233604007"],
                "relations": [
                    {
                        "from": "Symptom:Cough",
                        "predicate": "suggestsPathology",
                        "to": "Pathology:Pneumonia",
                    }
                ],
            },
        }
    )
    result = MemoryRetriever(memory_store=memory).retrieve(
        "fever cough pneumonia", top_k=3
    )

    assert result["status"] == "grounded_retrieval_ready"
    assert result["retrieval_mode"] == "semantic_vector_memory"
    assert result["retrieval_quality"]["memory_backed"] is True
    assert result["retrieval_quality"]["fallback_used"] is False
    assert result["evidence"][0]["source"] == "vector_memory"
    assert result["evidence"][0]["ontology_refs"] == ["SNOMED:233604007"]


def test_governance_scores_are_runtime_derived_not_hardcoded():
    low_risk = _derive_governance_scores(
        payload={"provenance": {"source": "unit_test"}, "clinical_severity": 0.1},
        discordances={"count": 0},
        retrieval={
            "evidence_count": 3,
            "retrieval_quality": {
                "coverage": 1.0,
                "memory_backed": True,
                "fallback_used": False,
                "mean_grounding_score": 0.8,
            },
        },
        ranked_evidence=[{"weight": 2.5}, {"weight": 2.2}],
        area_signals={
            "language_listening": {
                "salience_score": 0.8,
                "uncertainty_score": 0.1,
                "signal_count": 2,
            }
        },
    )
    high_risk = _derive_governance_scores(
        payload={"clinical_severity": 0.9},
        discordances={"count": 3},
        retrieval={
            "evidence_count": 0,
            "retrieval_quality": {
                "coverage": 0.0,
                "memory_backed": False,
                "fallback_used": True,
                "mean_grounding_score": 0.0,
            },
        },
        ranked_evidence=[],
        area_signals={
            "language_listening": {
                "salience_score": 0.1,
                "uncertainty_score": 0.9,
                "signal_count": 1,
            }
        },
    )

    assert low_risk["derivation"] == "runtime_governance_scores_not_hardcoded_constants"
    assert high_risk["risk"] > low_risk["risk"]
    assert high_risk["uncertainty"] > low_risk["uncertainty"]
    assert high_risk["nexus_coherence"] < low_risk["nexus_coherence"]
    assert (low_risk["discordances"], high_risk["discordances"]) == (0, 3)
