from melampo.reasoning.differential_engine import DifferentialEngine


def test_differential_engine_counts_evidence():
    engine = DifferentialEngine()
    result = engine.rank(
        [{"source": "retrieval", "kind": "grounded"}, {"source": "intuition", "kind": "candidate"}],
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
    assert result["evidence_count"] == 2
    assert result["hypotheses"][0]["label"] == "candidate_1"
    assert result["mismatch_score"] == 0.3
    assert result["hypotheses"][0]["support_signals"]
    assert result["support_profiles"]
    assert result["contradiction_profiles"]
    assert result["support_strength"] >= 0.0
    assert result["contradiction_strength"] >= 0.0
    assert result["recommended_tests"]
