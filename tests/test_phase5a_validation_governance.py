from melampo.evaluation.calibration import ConfidenceCalibrationEvaluator
from melampo.evaluation.clinical_benchmark import ClinicalBenchmarkRecord, ClinicalBenchmarkRunner
from melampo.evaluation.dataset_manifest import DatasetManifest, DatasetManifestRegistry
from melampo.evaluation.model_release_gate import ModelReleaseGate
from melampo.evaluation.rag_evaluation import RAGEvaluator, RAGEvaluationRecord
from melampo.evaluation.slice_analysis import SliceAnalysisRunner
from melampo.evaluation.validation_protocol import ValidationProtocol, ValidationProtocolRegistry
from melampo.governance.change_control import ChangeControlRegistry, ChangeRecord
from melampo.safety.rails import ClinicalSafetyRails


def _passing_benchmark_report():
    records = [
        ClinicalBenchmarkRecord(
            case_id="case-1",
            payload={"case_id": "case-1"},
            gold_labels=["pneumonia"],
            slices={"modality": "XR", "site": "synthetic_lab", "pathology_family": "infectious", "learning_status": "promoted"},
        ),
        ClinicalBenchmarkRecord(
            case_id="case-2",
            payload={"case_id": "case-2"},
            gold_labels=["normal"],
            slices={"modality": "XR", "site": "synthetic_lab", "pathology_family": "normal", "learning_status": "promoted"},
        ),
    ]

    def predict(payload):
        label = "pneumonia" if payload["case_id"] == "case-1" else "normal"
        return {
            "diagnostic_result": {
                "result_label": label,
                "top_hypothesis": {"label": label, "score": 0.82},
                "policy": {"abstain": False},
            }
        }

    return ClinicalBenchmarkRunner().run(records, predict)


def test_dataset_manifest_blocks_missing_governance_and_registers_valid_manifest():
    invalid = DatasetManifest.from_dict({"dataset_id": "bad", "name": "Bad", "source": ""})
    assert invalid.validate()["status"] == "blocked"

    valid = DatasetManifest(
        dataset_id="melampo-synthetic-xr-v1",
        name="Melampo synthetic XR validation set",
        source="internal_synthetic_fixture",
        license="internal_research",
        modalities=["XR"],
        label_schema={"labels": ["pneumonia", "normal"]},
        gold_standard="curated_research_fixture",
        deidentified=True,
        required_slices=["modality", "site", "pathology_family", "learning_status"],
    )
    validation = valid.validate()
    assert validation["status"] == "pass"
    assert validation["fingerprint"] == valid.fingerprint()

    registry = DatasetManifestRegistry()
    registered = registry.register(valid)
    assert registered["status"] == "registered"
    assert registry.summarize()["dataset_count"] == 1


def test_validation_protocol_locks_and_evaluates_observed_metrics():
    manifest = DatasetManifest(
        dataset_id="melampo-synthetic-xr-v1",
        name="Melampo synthetic XR validation set",
        source="internal_synthetic_fixture",
        license="internal_research",
        modalities=["XR"],
        label_schema={"labels": ["pneumonia", "normal"]},
        deidentified=True,
        required_slices=["modality", "site", "pathology_family", "learning_status"],
    )
    protocol = ValidationProtocol.default_research_protocol(dataset_id=manifest.dataset_id)
    assert protocol.readiness(manifest)["status"] == "blocked"
    locked = protocol.lock(model_version="phase4a-mock-models", memory_snapshot="phase3-memory-snapshot")
    assert locked["status"] == "locked"
    assert protocol.readiness(manifest)["status"] == "ready"
    observed = protocol.evaluate_observed_metrics(
        {
            "coverage": 1.0,
            "selective_accuracy": 0.9,
            "expected_calibration_error": 0.1,
            "provenance_completeness": 1.0,
        }
    )
    assert observed["status"] == "pass"

    registry = ValidationProtocolRegistry()
    registry.register(protocol)
    assert registry.summarize()["statuses"]["locked"] == 1


def test_slice_analysis_detects_underperforming_and_missing_required_slices():
    report = _passing_benchmark_report()
    slice_report = SliceAnalysisRunner(min_slice_size=1, min_selective_accuracy=0.8, min_coverage=0.8).run(
        report,
        required_slices=["modality", "site", "pathology_family", "learning_status", "age_band"],
    )
    assert slice_report.slice_count >= 4
    assert "age_band" in slice_report.missing_required_slices
    assert not slice_report.underperforming_slices


def test_model_release_gate_blocks_without_protocol_and_passes_research_when_complete():
    benchmark = _passing_benchmark_report()
    calibration = ConfidenceCalibrationEvaluator(bin_count=5).evaluate(benchmark.records)
    rag_report = RAGEvaluator().evaluate(
        [
            RAGEvaluationRecord(
                query="pneumonia opacity",
                answer="pneumonia opacity",
                evidence=[
                    {
                        "source": "clinical_document_processor",
                        "value": "pneumonia opacity",
                        "grounding_score": 0.9,
                        "record_id": "doc:1",
                        "metadata": {"source_path": "guideline.txt", "section": "diagnosis", "page": 1},
                    }
                ],
                expected_terms=["pneumonia", "opacity"],
            )
        ]
    )
    manifest = DatasetManifest(
        dataset_id="melampo-synthetic-xr-v1",
        name="Melampo synthetic XR validation set",
        source="internal_synthetic_fixture",
        license="internal_research",
        modalities=["XR"],
        label_schema={"labels": ["pneumonia", "normal"]},
        gold_standard="curated_research_fixture",
        deidentified=True,
        required_slices=["modality", "site", "pathology_family", "learning_status"],
    )
    protocol = ValidationProtocol.default_research_protocol(dataset_id=manifest.dataset_id)
    gate = ModelReleaseGate(min_sample_count=2, min_selective_accuracy=0.6, min_coverage=0.5)
    blocked = gate.evaluate(benchmark_report=benchmark, calibration_report=calibration, rag_report=rag_report, dataset_manifest=manifest)
    assert blocked.status == "blocked"
    assert "validation_protocol_missing" in blocked.failures

    protocol.lock(model_version="phase4a-mock-models", memory_snapshot="phase3-memory-snapshot")
    slice_report = SliceAnalysisRunner(min_selective_accuracy=0.6, min_coverage=0.5).run(benchmark, required_slices=protocol.required_slices)
    change = ChangeRecord(component="model", change_type="adapter", description="Phase 4A adapter release", risk_level="medium")
    change.approve("research_lead", {"release_gate_review": True})
    passed = gate.evaluate(
        benchmark_report=benchmark,
        calibration_report=calibration,
        rag_report=rag_report,
        protocol=protocol,
        dataset_manifest=manifest,
        slice_report=slice_report,
        change_control=change.as_dict(),
    )
    assert passed.status == "research_pass"
    assert passed.clinical_use_allowed is False
    assert passed.promotion_allowed is True


def test_change_control_requires_human_review_for_high_risk_changes():
    registry = ChangeControlRegistry()
    proposed = registry.propose(
        ChangeRecord(
            component="policy",
            change_type="threshold",
            description="Change diagnostic abstention threshold",
            risk_level="high",
        )
    )
    change_id = proposed["change"]["change_id"]
    review = registry.approve(change_id, reviewer="qa", evidence={"unit_tests": True})
    assert review["status"] == "needs_review"
    approved = registry.approve(change_id, reviewer="medical_governance", evidence={"human_governance_review": True})
    assert approved["status"] == "approved"
    assert registry.summarize()["statuses"]["approved"] == 1


def test_clinical_safety_rails_block_unsafe_retrieval_and_output():
    rails = ClinicalSafetyRails(min_provenance_fraction=0.8)
    input_decision = rails.evaluate_input({"case_id": "case-1", "provenance": {"contains_phi": False}})
    assert input_decision.status == "pass"

    retrieval_decision = rails.evaluate_retrieval(
        [{"source": "synthetic", "learning_status": "candidate", "metadata": {"source_type": "synthetic_dream_trace"}}]
    )
    assert retrieval_decision.status == "block"
    assert "retrieval_provenance_below_threshold" in retrieval_decision.reasons

    output_decision = rails.evaluate_output(
        {
            "result_label": "candidate_a",
            "melampo_metrics": {"mismatch_index": 0.95},
            "policy": {"abstain": False},
            "audit_trace": {"clinical_warning": "Research output; not a validated medical device."},
        }
    )
    assert output_decision.status == "block"
    assert "mismatch_index_above_safety_threshold" in output_decision.reasons
