from melampo.evaluation.calibration import ConfidenceCalibrationEvaluator
from melampo.evaluation.prospective_validation import ProspectiveValidationRegistry


def test_calibration_evaluator_reports_ece_and_brier():
    records = [
        {"confidence": 0.9, "correct": True},
        {"confidence": 0.8, "correct": True},
        {"confidence": 0.7, "correct": False},
    ]
    report = ConfidenceCalibrationEvaluator(bin_count=5).evaluate(records)
    thresholds = ConfidenceCalibrationEvaluator(bin_count=5).suggest_thresholds(
        report, target_min_accuracy=0.5
    )

    assert report.sample_count == 3
    assert report.expected_calibration_error >= 0.0
    assert report.brier_score >= 0.0
    assert thresholds["status"] in {"threshold_suggested", "no_threshold_suggested"}


def test_prospective_validation_registry_locks_prediction_and_attaches_outcome(
    tmp_path,
):
    registry = ProspectiveValidationRegistry()
    prediction = registry.create_prediction(
        {"case_id": "case-1", "report_text": "example"},
        {"case_id": "case-1", "result_label": "pneumonia"},
        protocol_id="unit-test-protocol",
    )
    outcome_result = registry.attach_outcome(
        prediction.prediction_id, {"accepted_labels": ["pneumonia"]}
    )
    evaluation = registry.evaluate()
    path = tmp_path / "prospective.jsonl"
    registry.save_jsonl(path)
    loaded = ProspectiveValidationRegistry.load_jsonl(path)

    assert prediction.locked is True
    assert outcome_result["status"] == "completed"
    assert evaluation["completed_outcomes"] == 1
    assert evaluation["prospective_accuracy"] == 1.0
    assert prediction.prediction_id in loaded.predictions
