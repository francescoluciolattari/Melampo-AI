from melampo.prototype import ClinicalPrototypeRunner, PrototypeInputValidator, run_prototype_case


def test_prototype_input_validator_rejects_missing_case_id():
    validation = PrototypeInputValidator().validate({"report_text": "possible lesion"})
    assert validation["valid"] is False
    assert "missing_required_field:case_id" in validation["errors"]


def test_prototype_runner_executes_valid_minimal_case():
    result = run_prototype_case(
        {
            "case_id": "case-proto-001",
            "report_text": "possible pulmonary lesion",
            "patient_complaints": "persistent cough",
        },
        runtime_profile="local_research",
    )
    assert result["status"] == "completed"
    assert result["case_id"] == "case-proto-001"
    assert result["runtime"]["config"]["runtime_profile"] == "local_research"
    assert result["top_hypothesis"]
    assert result["recommended_actions"]
    assert result["prioritized_actions"]
    assert result["state_summary"]["case_id"] == "case-proto-001"


def test_prototype_runner_preserves_invalid_payload_without_running_pipeline():
    runner = ClinicalPrototypeRunner.from_profile("remote_research")
    result = runner.run_case({"report_text": "missing id"})
    assert result["status"] == "invalid_payload"
    assert result["validation"]["valid"] is False
