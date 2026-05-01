from melampo.models.quantum_belief_layer import QuantumBeliefLayer
from melampo.reasoning.intuition_engine import IntuitionEngine


def _ranked_evidence():
    return [
        {"weight": 1.0, "item": {"source": "retrieval"}},
        {"weight": 0.75, "item": {"source": "guideline"}},
        {"weight": 0.4, "item": {"source": "epidemiology"}},
    ]


def test_high_pi_score_and_low_prediction_error_support_rapid_intuition():
    engine = IntuitionEngine(belief_layer=QuantumBeliefLayer())
    result = engine.infer(
        case_id="case-calibration-high-pi",
        ranked_evidence=_ranked_evidence(),
        dream={"rehearsal_profile": {"revision_bias": "exploratory"}, "alternative_hypotheses": []},
        quantum_allowed=True,
        area_signals={
            "visual_diagnostic": {"salience_score": 0.8, "signal_count": 3},
            "language_listening": {"salience_score": 0.7, "signal_count": 3},
        },
        area_dynamics={
            "coherence_score": 0.8,
            "mismatch_score": 0.1,
            "pi_score": 0.82,
            "prediction_error": 0.1,
            "precision_weighted_coherence": 0.86,
            "neuro_dynamic_metrics": {
                "pi_score": 0.82,
                "prediction_error": 0.1,
                "precision_weighted_coherence": 0.86,
                "deductive_gate": 0.9,
                "revision_pressure": 0.1,
                "intuition_gain": 1.18,
                "bias_suppression_score": 0.9,
                "conflict_load": 0.1,
            },
        },
    )
    assert result["deductive_filter"]["reasoning_mode"] == "rapid_intuition"
    assert result["candidate_scores"][0]["mode"] == "rapid_intuition"
    assert result["belief_update"]["precision_modulation"] > result["belief_update"]["conflict_modulation"]


def test_high_prediction_error_increases_revision_or_contradiction_pressure():
    engine = IntuitionEngine(belief_layer=QuantumBeliefLayer())
    result = engine.infer(
        case_id="case-calibration-high-error",
        ranked_evidence=_ranked_evidence(),
        dream={
            "rehearsal_profile": {
                "contradiction_rehearsal": True,
                "revision_bias": "conservative",
                "post_error_adjustment": "re-rank_alternatives",
            },
            "alternative_hypotheses": [{"label": "rare_alt", "kind": "contradiction_revision"}],
        },
        quantum_allowed=True,
        area_signals={
            "visual_diagnostic": {"salience_score": 0.4, "signal_count": 1},
            "epidemiology": {"salience_score": 0.2, "signal_count": 1},
        },
        area_dynamics={
            "coherence_score": 0.2,
            "mismatch_score": 0.9,
            "pi_score": 0.2,
            "prediction_error": 0.88,
            "precision_weighted_coherence": 0.2,
            "neuro_dynamic_metrics": {
                "pi_score": 0.2,
                "prediction_error": 0.88,
                "precision_weighted_coherence": 0.2,
                "deductive_gate": 0.1,
                "revision_pressure": 0.85,
                "intuition_gain": 0.72,
                "bias_suppression_score": 0.2,
                "conflict_load": 0.9,
            },
        },
    )
    assert result["candidate_scores"][0]["mode"] in {"rational_revision", "contradiction_revision"}
    assert result["deductive_filter"]["prediction_error"] >= 0.8
    assert result["belief_update"]["conflict_modulation"] > 0.0
