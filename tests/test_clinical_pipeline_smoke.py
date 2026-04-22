from melampo.app import build_default_runtime


def test_clinical_pipeline_runs_minimal_payload():
    runtime = build_default_runtime()
    result = runtime.pipeline.run({"case_id": "case-001", "report_text": "possible pulmonary lesion"})
    assert result["case_id"] == "case-001"
    assert "retrieval" in result
    assert "intuition" in result
    assert "coordinated" in result
    assert "dream" in result
