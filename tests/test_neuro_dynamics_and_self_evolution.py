from melampo.memory.semantic_memory import SemanticMemoryStore
from melampo.reasoning.area_coherence import AreaCoherenceAnalyzer
from melampo.reasoning.neuro_dynamics import NeuroDynamicMetrics
from melampo.training.self_evolution import DreamSelfEvolutionLoop


def test_neuro_dynamic_metrics_are_emitted_by_area_coherence():
    area_signals = {
        "visual_diagnostic": {"salience_score": 0.8, "signal_count": 3},
        "language_listening": {"salience_score": 0.6, "signal_count": 2},
        "case_context": {"salience_score": 0.3, "signal_count": 2},
        "epidemiology": {"salience_score": 0.4, "signal_count": 2},
    }
    dynamics = AreaCoherenceAnalyzer().analyze(area_signals)
    neuro = dynamics["neuro_dynamic_metrics"]
    assert 0.0 <= neuro["pi_score"] <= 1.0
    assert 0.0 <= neuro["prediction_error"] <= 1.0
    assert 0.0 <= neuro["precision_weighted_coherence"] <= 1.0
    assert dynamics["pi_score"] == neuro["pi_score"]


def test_neuro_dynamic_metrics_direct_compute_contract():
    metrics = NeuroDynamicMetrics().compute(
        pair_profiles=[
            {"status": "coherent", "pair_salience": 1.0, "pair_signal_count": 4},
            {"status": "mismatch", "pair_salience": 0.2, "pair_signal_count": 1},
        ],
        coherence_score=0.8,
        mismatch_score=0.2,
        total_salience=1.2,
    )
    assert metrics["pi_score"] > 0.0
    assert metrics["deductive_gate"] >= 0.0
    assert metrics["interpretation"] == "computational_abstraction_not_literal_neurobiology"


def test_semantic_memory_indexes_documents_in_vector_memory():
    memory = SemanticMemoryStore()
    memory.add_document({
        "id": "doc-1",
        "text": "pulmonary opacity cough fever differential pneumonia",
        "source": "test_fixture",
        "learning_status": "promoted",
    })
    hits = memory.semantic_search("cough pneumonia", limit=1, promoted_only=True)
    assert hits
    assert hits[0]["record_id"] == "doc-1"
    assert memory.describe()["vector_store"]["record_count"] == 1


def test_dream_self_evolution_promotes_only_favorable_candidates():
    loop = DreamSelfEvolutionLoop()
    favorable = loop.rehearse(
        case_context={"case_id": "case-good", "report_text": "cough opacity", "patient_complaints": "fever"},
        area_dynamics={
            "coherence_pairs": [("language_listening", "visual_diagnostic")],
            "mismatch_pairs": [],
            "neuro_dynamic_metrics": {
                "pi_score": 0.8,
                "prediction_error": 0.1,
                "bias_suppression_score": 0.9,
            },
        },
    )
    assert favorable["evaluation"]["accepted"] is True
    assert favorable["memory_record"]["learning_status"] == "promoted"

    unfavorable = loop.rehearse(
        case_context={"case_id": "case-bad", "report_text": "uncertain", "patient_complaints": ""},
        area_dynamics={
            "coherence_pairs": [],
            "mismatch_pairs": [("epidemiology", "language_listening")],
            "neuro_dynamic_metrics": {
                "pi_score": 0.2,
                "prediction_error": 0.8,
                "bias_suppression_score": 0.2,
            },
        },
    )
    assert unfavorable["evaluation"]["accepted"] is False
    assert unfavorable["memory_record"]["learning_status"] == "candidate"
