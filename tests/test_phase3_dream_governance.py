from melampo.memory.learning_status import validate_learning_transition
from melampo.memory.vector_memory import InMemoryVectorStore
from melampo.training.dream_candidate_store import DreamCandidateStore
from melampo.training.dream_scheduler import DreamScheduler, LowActivityPolicy
from melampo.training.outcome_feedback import OutcomeFeedbackIngestor
from melampo.training.promotion_policy import PromotionPolicy
from melampo.training.rational_control_validator import RationalControlValidator


def _favorable_area_dynamics():
    return {
        "coherence_pairs": [("language_listening", "visual_diagnostic")],
        "mismatch_pairs": [],
        "pi_score": 0.82,
        "convergence_index": 0.72,
        "mismatch_index": 0.12,
        "neuro_dynamic_metrics": {
            "pi_score": 0.82,
            "convergence_index": 0.72,
            "mismatch_index": 0.12,
            "prediction_error": 0.08,
            "bias_suppression_score": 0.88,
        },
    }


def test_learning_status_blocks_unvalidated_promotion():
    blocked = validate_learning_transition("candidate", "promoted", evidence={})
    assert blocked.allowed is False
    assert "promotion_requires_rational_control_validation" in blocked.reasons

    allowed = validate_learning_transition(
        "candidate",
        "promoted",
        evidence={
            "rational_control_validation": True,
            "provenance_available": True,
            "clinical_deployment": False,
        },
    )
    assert allowed.allowed is True


def test_rational_control_validator_validates_favorable_candidate():
    candidate = {
        "text": "Dream candidate for multimodal pneumonia-like alignment.",
        "source": "dream_scheduler",
        "metadata": {
            "case_id": "case-phase3",
            "pi_score": 0.82,
            "convergence_index": 0.72,
            "mismatch_index": 0.12,
            "provenance_quality": 0.9,
            "retrieval_coverage": 0.6,
        },
        "auto_evolution_plan": {
            "candidate_score": 0.78,
            "promotion_guardrails": [
                "requires rational-control validation",
                "requires provenance and source labeling",
                "requires no clinical deployment without prospective validation",
            ],
        },
    }
    validation = RationalControlValidator().evaluate(
        candidate=candidate,
        area_dynamics=_favorable_area_dynamics(),
        retrieval_context={"retrieval_coverage": 0.6},
        governance_scores={"risk": 0.1, "provenance_quality": 0.9, "retrieval_coverage": 0.6},
    )
    assert validation["allowed_for_promotion"] is True
    assert validation["status"] == "validated_for_promotion_review"


def test_promotion_policy_queues_for_review_by_default():
    store = DreamCandidateStore()
    record = store.create_candidate(
        payload={
            "text": "validated dream candidate",
            "metadata": {"case_id": "case-review", "provenance_quality": 0.9},
            "auto_evolution_plan": {"candidate_score": 0.8},
        },
        case_id="case-review",
    )
    validation = {
        "allowed_for_promotion": True,
        "status": "validated_for_promotion_review",
        "observed": {"provenance_available": True, "candidate_score": 0.8},
    }
    store.attach_validation(record.candidate_id, validation)
    decision = PromotionPolicy().decide(store.get(record.candidate_id).as_dict(), validation)
    updated = store.attach_promotion_decision(record.candidate_id, decision)
    assert decision["target_learning_status"] == "needs_review"
    assert updated["learning_status"] == "needs_review"


def test_dream_scheduler_runs_only_in_low_activity_window():
    scheduler = DreamScheduler(low_activity_policy=LowActivityPolicy(min_idle_seconds=10, max_active_requests=0))
    scheduler.enqueue(
        case_context={"case_id": "case-idle", "report_text": "opacity cough", "patient_complaints": "fever"},
        area_dynamics=_favorable_area_dynamics(),
        dream={
            "auto_evolution_plan": {
                "candidate_score": 0.78,
                "promotion_guardrails": [
                    "requires rational-control validation",
                    "requires provenance and source labeling",
                    "requires no clinical deployment without prospective validation",
                ],
            }
        },
        retrieval_context={"retrieval_coverage": 0.7},
        governance_scores={"risk": 0.1, "provenance_quality": 0.9, "retrieval_coverage": 0.7},
    )
    skipped = scheduler.run_once(activity={"active_requests": 1, "idle_seconds": 0})
    assert skipped["status"] == "skipped"
    assert skipped["queued_jobs"] == 1

    completed = scheduler.run_once(activity={"active_requests": 0, "idle_seconds": 20})
    assert completed["status"] == "completed"
    assert completed["processed_jobs"] == 1
    result = completed["results"][0]
    assert result["validation"]["rational_control_validation"] is True
    assert result["promotion_decision"]["target_learning_status"] == "needs_review"
    assert completed["vector_memory"]["status_counts"]["needs_review"] == 1


def test_outcome_feedback_attaches_to_candidate_and_memory():
    store = DreamCandidateStore()
    record = store.create_candidate({"text": "candidate", "metadata": {"case_id": "case-outcome"}}, case_id="case-outcome")
    ingestor = OutcomeFeedbackIngestor()
    updated = ingestor.attach_to_candidate(
        store,
        record.candidate_id,
        diagnostic_result={"case_id": "case-outcome", "result_label": "pneumonia", "top_hypothesis": {"score": 0.8}},
        outcome={"accepted_labels": ["pneumonia"], "notes": "reviewed by specialist"},
    )
    assert updated["outcome_feedback"][0]["correct"] is True

    vector_store = InMemoryVectorStore.enterprise_default()
    feedback = ingestor.build_feedback(
        diagnostic_result={"case_id": "case-outcome", "result_label": "pneumonia", "top_hypothesis": {"score": 0.8}},
        outcome={"accepted_labels": ["pneumonia"]},
    )
    memory = ingestor.consolidate_to_memory(vector_store, feedback, learning_status="candidate")
    assert memory["metadata"]["memory_role"] == "reviewed_outcome_feedback"
    assert vector_store.describe()["record_count"] == 1
