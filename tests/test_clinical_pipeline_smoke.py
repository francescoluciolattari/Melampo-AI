from melampo.app import build_default_runtime


def test_clinical_pipeline_runs_minimal_payload():
    runtime = build_default_runtime()
    result = runtime.pipeline.run({"case_id": "case-001", "report_text": "possible pulmonary lesion"})
    assert result["case_id"] == "case-001"
    assert "retrieval" in result
    assert "area_signals" in result
    assert sorted(result["area_signals"].keys()) == ["case_context", "epidemiology", "language_listening", "visual_diagnostic"]
    assert "area_dynamics" in result
    assert result["area_dynamics"]["coherence_score"] >= 0.0
    assert result["area_dynamics"]["mismatch_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["top_areas"]
    assert result["intuition"]["deductive_filter"]["convergence_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["conflict_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["coherence_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["mismatch_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["area_pair_bonus"] >= 0.0
    assert result["intuition"]["deductive_filter"]["disagreement_penalty"] >= 0.0
    assert result["intuition"]["deductive_filter"]["revision_bias"] in ["exploratory", "conservative"]
    assert isinstance(result["intuition"]["deductive_filter"]["contradiction_rehearsal"], bool)
    assert result["intuition"]["deductive_filter"]["reasoning_mode"] in ["rapid_intuition", "rational_revision", "contradiction_revision"]
    assert result["intuition"]["rapid_intuition"] == "candidate_1"
    assert len(result["intuition"]["candidate_scores"]) == 3
    assert "coordinated" in result
    assert result["coordinated"]["differential"]["hypotheses"]
    assert result["coordinated"]["differential"]["hypotheses"][0]["hypothesis_type"] in ["primary_hypothesis", "revision_hypothesis", "contradiction_revision_hypothesis"]
    assert result["coordinated"]["differential"]["hypotheses"][0]["hypothesis_domain"] in ["multimodal_led", "imaging_led", "language_led", "epidemiology_led", "mismatch_resolution_led"]
    assert result["coordinated"]["differential"]["recommended_tests"]
    assert result["coordinated"]["differential"]["hypotheses"][0]["support_signals"]
    assert result["coordinated"]["differential"]["support_profiles"]
    assert result["coordinated"]["differential"]["contradiction_profiles"]
    assert "critique" in result
    assert result["critique"]["status"] == "reviewed"
    assert result["critique"]["suggestions"]
    assert "dream" in result
    assert "rehearsal_profile" in result["dream"]
    assert len(result["dream"]["alternative_hypotheses"]) >= 2
