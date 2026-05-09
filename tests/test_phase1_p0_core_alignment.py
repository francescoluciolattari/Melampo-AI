from melampo.memory.retriever import MemoryRetriever
from melampo.memory.semantic_memory import SemanticMemoryStore
from melampo.reasoning.area_coherence import AreaCoherenceAnalyzer
from melampo.reasoning.clinical_pipeline import _derive_governance_scores


def _profile(result, first, second):
    pair = tuple(sorted((first, second)))
    return next(item for item in result["pair_profiles"] if item["pair"] == pair)


def test_language_epidemiology_shared_claim_is_coherent_after_canonical_pair_fix():
    analyzer = AreaCoherenceAnalyzer()
    area_signals = {
        "language_listening": {
            "area": "language_listening",
            "focus": "language_led",
            "merged_text": "patient reports cough and smoking exposure",
            "signal_count": 2,
            "salience_score": 0.7,
            "uncertainty_score": 0.2,
            "claims": [
                {"normalized_entity": "smoking exposure", "polarity": "present", "ontology_refs": ["SNOMED:77176002"]},
            ],
        },
        "epidemiology": {
            "area": "epidemiology",
            "focus": "epidemiology_led",
            "exposures": {"smoking": True},
            "signal_count": 2,
            "salience_score": 0.6,
            "uncertainty_score": 0.25,
            "claims": [
                {"normalized_entity": "smoking exposure", "polarity": "present", "ontology_refs": ["SNOMED:77176002"]},
            ],
        },
    }

    result = analyzer.analyze(area_signals)
    profile = _profile(result, "language_listening", "epidemiology")

    assert profile["status"] == "coherent"
    assert profile["prior_expected_interaction"] is True
    assert profile["agreement_score"] > profile["dynamic_mismatch_score"]
    assert ("epidemiology", "language_listening") in result["coherence_pairs"]


def test_explicit_claim_contradiction_raises_dynamic_mismatch():
    analyzer = AreaCoherenceAnalyzer()
    area_signals = {
        "language_listening": {
            "area": "language_listening",
            "focus": "language_led",
            "merged_text": "report describes pneumonia",
            "signal_count": 2,
            "salience_score": 0.8,
            "uncertainty_score": 0.15,
            "claims": [
                {"normalized_entity": "pneumonia", "polarity": "present", "ontology_refs": ["SNOMED:233604007"]},
            ],
        },
        "visual_diagnostic": {
            "area": "visual_diagnostic",
            "focus": "imaging_led",
            "signal_count": 2,
            "salience_score": 0.75,
            "uncertainty_score": 0.15,
            "claims": [
                {"normalized_entity": "pneumonia", "polarity": "absent", "ontology_refs": ["SNOMED:233604007"]},
            ],
        },
    }

    result = analyzer.analyze(area_signals)
    profile = _profile(result, "language_listening", "visual_diagnostic")

    assert profile["status"] == "mismatch"
    assert profile["contradiction_score"] >= 0.9
    assert result["mismatch_score"] > 0.4
    assert result["revision_pressure"] > 0.0


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
                "relations": [{"from": "Symptom:Cough", "predicate": "suggestsPathology", "to": "Pathology:Pneumonia"}],
            },
        }
    )
    result = MemoryRetriever(memory_store=memory).retrieve("fever cough pneumonia", top_k=3)

    assert result["status"] == "grounded_retrieval_ready"
    assert result["retrieval_mode"] == "semantic_vector_memory"
    assert result["retrieval_quality"]["memory_backed"] is True
    assert result["retrieval_quality"]["fallback_used"] is False
    assert result["evidence"][0]["source"] == "vector_memory"
    assert result["evidence"][0]["ontology_refs"] == ["SNOMED:233604007"]


def test_governance_scores_are_runtime_derived_not_hardcoded():
    low_risk = _derive_governance_scores(
        payload={"provenance": {"source": "unit_test"}, "clinical_severity": 0.1},
        area_dynamics={
            "coherence_score": 0.8,
            "mismatch_score": 0.1,
            "neuro_dynamic_metrics": {"mismatch_index": 0.1, "prediction_error": 0.1, "convergence_index": 0.8},
        },
        retrieval={
            "evidence_count": 3,
            "retrieval_quality": {"coverage": 1.0, "memory_backed": True, "fallback_used": False, "mean_grounding_score": 0.8},
        },
        ranked_evidence=[{"weight": 2.5}, {"weight": 2.2}],
        area_signals={"language_listening": {"salience_score": 0.8, "uncertainty_score": 0.1, "signal_count": 2}},
    )
    high_risk = _derive_governance_scores(
        payload={"clinical_severity": 0.9},
        area_dynamics={
            "coherence_score": 0.1,
            "mismatch_score": 0.9,
            "neuro_dynamic_metrics": {"mismatch_index": 0.9, "prediction_error": 0.8, "convergence_index": 0.1},
        },
        retrieval={
            "evidence_count": 0,
            "retrieval_quality": {"coverage": 0.0, "memory_backed": False, "fallback_used": True, "mean_grounding_score": 0.0},
        },
        ranked_evidence=[],
        area_signals={"language_listening": {"salience_score": 0.1, "uncertainty_score": 0.9, "signal_count": 1}},
    )

    assert low_risk["derivation"] == "runtime_governance_scores_not_hardcoded_constants"
    assert high_risk["risk"] > low_risk["risk"]
    assert high_risk["uncertainty"] > low_risk["uncertainty"]
    assert high_risk["dream_coherence"] < low_risk["dream_coherence"]
