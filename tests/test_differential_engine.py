from melampo.reasoning.differential_engine import DifferentialEngine


def test_differential_engine_counts_evidence():
    engine = DifferentialEngine()
    result = engine.rank(
        [{"source": "retrieval", "kind": "grounded"}, {"source": "fusion", "kind": "engine"}, {"source": "intuition", "kind": "candidate"}],
        intuition={
            "candidate_scores": [{"label": "candidate_1", "score": 1.2}],
            "deductive_filter": {"reasoning_mode": "rapid_intuition"},
        },
        dream={
            "alternative_hypotheses": [{"label": "alt_1", "kind": "rare_case", "focus": "epidemiology"}],
            "rehearsal_profile": {"coherence_guidance": "multimodal_support", "boundary_case_hint": True},
        },
        area_dynamics={"mismatch_score": 0.3, "coherence_score": 0.5, "coherence_pairs": [("language_listening", "visual_diagnostic")], "mismatch_pairs": []},
    )
    assert result["evidence_count"] == 3
    assert result["hypotheses"][0]["label"] == "candidate_1"
    assert result["hypotheses"][0]["hypothesis_type"] == "primary_hypothesis"
    assert result["hypotheses"][0]["hypothesis_domain"] in ["multimodal_led", "imaging_led", "language_led", "epidemiology_led", "mismatch_resolution_led"]
    assert result["mismatch_score"] == 0.3
    assert result["hypotheses"][0]["support_signals"]
    assert result["support_profiles"]
    assert result["contradiction_profiles"]
    assert result["support_strength"] >= 0.0
    assert result["contradiction_strength"] >= 0.0
    assert result["recommended_actions"]
    assert result["recommended_actions"][0]["category"] in ["confirmation_test", "disambiguation_test", "multimodal_reconciliation"]
    assert result["recommended_tests"]
