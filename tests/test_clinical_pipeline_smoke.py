from melampo.app import build_default_runtime


def test_clinical_pipeline_runs_minimal_payload():
    runtime = build_default_runtime()
    result = runtime.pipeline.run({"case_id": "case-001", "report_text": "possible pulmonary lesion"})
    assert result["case_id"] == "case-001"
    assert "retrieval" in result
    assert "area_signals" in result
    assert sorted(result["area_signals"].keys()) == ["case_context", "epidemiology", "language_listening", "visual_diagnostic"]
    assert result["intuition"]["deductive_filter"]["top_areas"]
    assert result["intuition"]["deductive_filter"]["convergence_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["conflict_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["area_pair_bonus"] >= 0.0
    assert result["intuition"]["deductive_filter"]["disagreement_penalty"] >= 0.0
    assert result["intuition"]["deductive_filter"]["revision_bias"] in ["exploratory", "conservative"]
    assert isinstance(result["intuition"]["deductive_filter"]["contradiction_rehearsal"], bool)
    assert result["intuition"]["deductive_filter"]["reasoning_mode"] in ["rapid_intuition", "rational_revision", "contradiction_revision"]
    assert result["intuition"]["rapid_intuition"] == "candidate_1"
    assert len(result["intuition"]["candidate_scores"]) == 3
    assert "intuition" in result
    assert "coordinated" in result
    assert "dream" in result
    assert "rehearsal_profile" in result["dream"]
    assert len(result["dream"]["alternative_hypotheses"]) >= 2
