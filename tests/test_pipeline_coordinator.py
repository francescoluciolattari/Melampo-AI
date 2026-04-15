from melampo.reasoning.default_pipeline import DefaultPipelineFactory


def test_default_pipeline_factory_runs():
    pipeline = DefaultPipelineFactory().build()
    result = pipeline.run(case_id="case-1", evidence=["finding_a", "finding_b"], risk=0.2, uncertainty=0.1)
    assert result["policy"]["allow"] is True
    assert result["state"].case_id == "case-1"
    assert len(result["trace"]) >= 2
