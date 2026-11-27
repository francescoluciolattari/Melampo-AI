from melampo.evaluation.dataset_manifest import DatasetManifest, DatasetManifestRegistry
from melampo.evaluation.validation_protocol import (
    ValidationProtocol,
    ValidationProtocolRegistry,
)
from melampo.governance.change_control import ChangeControlRegistry, ChangeRecord
from melampo.safety.rails import ClinicalSafetyRails


def test_dataset_manifest_blocks_missing_governance_and_registers_valid_manifest():
    invalid = DatasetManifest.from_dict(
        {"dataset_id": "bad", "name": "Bad", "source": ""}
    )
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
    protocol = ValidationProtocol.default_research_protocol(
        dataset_id=manifest.dataset_id
    )
    assert protocol.readiness(manifest)["status"] == "blocked"
    locked = protocol.lock(
        model_version="phase4a-mock-models", memory_snapshot="phase3-memory-snapshot"
    )
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
    approved = registry.approve(
        change_id,
        reviewer="medical_governance",
        evidence={"human_governance_review": True},
    )
    assert approved["status"] == "approved"
    assert registry.summarize()["statuses"]["approved"] == 1


def test_clinical_safety_rails_block_unsafe_retrieval_and_output():
    rails = ClinicalSafetyRails(min_provenance_fraction=0.8)
    input_decision = rails.evaluate_input(
        {"case_id": "case-1", "provenance": {"contains_phi": False}}
    )
    assert input_decision.status == "pass"

    retrieval_decision = rails.evaluate_retrieval(
        [
            {
                "source": "synthetic",
                "learning_status": "candidate",
                "metadata": {"source_type": "synthetic_nexus_trace"},
            }
        ]
    )
    assert retrieval_decision.status == "block"
    assert "retrieval_provenance_below_threshold" in retrieval_decision.reasons

    output_decision = rails.evaluate_output(
        {
            "result_label": "candidate_a",
            "melampo_metrics": {"mismatch_index": 0.95},
            "policy": {"abstain": False},
            "audit_trace": {
                "clinical_warning": "Research output; not a validated medical device."
            },
        }
    )
    assert output_decision.status == "block"
    assert "mismatch_index_above_safety_threshold" in output_decision.reasons
