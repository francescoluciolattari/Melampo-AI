from melampo.memory.retriever import MemoryRetriever
from melampo.memory.semantic_memory import SemanticMemoryStore
from melampo.reasoning.case_signals import case_signals


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


def test_case_signals_are_explicit_rules_without_weights():
    quiet = case_signals(
        payload={"provenance": {"source": "unit_test"}, "clinical_severity": 0.1},
        discordances={"count": 0},
        retrieval={"evidence_count": 3, "retrieval_quality": {"memory_backed": True, "fallback_used": False}},
    )
    severe = case_signals(
        payload={"clinical_severity": "high"},
        discordances={"count": 3, "kinds": {"card": 3}},
        retrieval={"evidence_count": 0, "retrieval_quality": {"memory_backed": False, "fallback_used": True}},
    )
    assert quiet["derivation"] == "explicit_rules_without_weights"
    assert (quiet["escalate"], quiet["reasons"], quiet["flags"]) == (False, [], [])
    assert [r["rule"] for r in severe["reasons"]] == ["discordances_in_the_reading", "declared_severity_high"]
    assert [f["rule"] for f in severe["flags"]] == ["no_evidence", "no_provenance"]
    assert not {"risk", "uncertainty"} & set(severe)
