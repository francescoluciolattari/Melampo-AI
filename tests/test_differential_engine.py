from melampo.reasoning.differential_engine import DifferentialEngine


def test_differential_engine_counts_evidence():
    engine = DifferentialEngine()
    result = engine.rank(
        ["finding_a", "finding_b"],
        intuition={"candidate_scores": [{"label": "candidate_1", "score": 1.2}]},
        dream={"alternative_hypotheses": [{"label": "alt_1", "kind": "rare_case"}]},
        area_dynamics={"mismatch_score": 0.3, "coherence_score": 0.5},
    )
    assert result["evidence_count"] == 2
    assert result["hypotheses"][0]["label"] == "candidate_1"
    assert result["mismatch_score"] == 0.3
