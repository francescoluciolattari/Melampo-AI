"""Tests for ModelRouter (D1): the two-step router that replaces a 12-line
stub which routed on a task name string, never called by anything,
verified before this rewrite. pick_pending_case() runs the same
pending_case_router.route_payload() check, callable early in
clinical_pipeline.run() before real signals exist; pick_mode() implements
the September decision record's table, callable once they do.
"""

from melampo.orchestration.model_router import (
    FEW_FINDINGS_THRESHOLD,
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    MODE_DUAL_PATH,
    MODE_DUAL_PATH_EXTENDED_BUDGET,
    MODE_ONE_SHOT,
    UNRESOLVED_MISMATCH_THRESHOLD,
    ModelRouter,
)
from melampo.training.nexus_candidate_store import NexusCandidateStore


def _router():
    return ModelRouter(candidate_store=NexusCandidateStore())


# --------------------------------------------------------------------------
# pick_pending_case: the same route_payload() check, just callable
# standalone from D1 rather than imported directly
# --------------------------------------------------------------------------


def test_pick_pending_case_recognises_a_brand_new_case():
    decision = _router().pick_pending_case({"case_id": "new-case"})
    assert decision.action == "new_case"


def test_pick_pending_case_finds_a_case_already_pending():
    store = NexusCandidateStore()
    store.create_candidate(
        payload={"case_context": {"case_id": "case-1", "report_text": "initial"}},
        case_id="case-1", learning_status="needs_review",
    )
    router = ModelRouter(candidate_store=store)

    decision = router.pick_pending_case({"case_id": "case-1", "report_text": "follow-up"})

    assert decision.action == "merge_and_rerun"


# --------------------------------------------------------------------------
# pick_mode: the four-row table from docs/rlm_on_memory_decision_record.md
# --------------------------------------------------------------------------


def test_low_risk_and_few_findings_is_one_shot():
    mode, reason = _router().pick_mode(findings=[], governance_scores={"risk": 0.1})
    assert mode == MODE_ONE_SHOT
    assert "low risk" in reason


def test_high_risk_is_dual_path():
    mode, reason = _router().pick_mode(findings=[], governance_scores={"risk": 0.9})
    assert mode == MODE_DUAL_PATH
    assert "risk" in reason


def test_many_findings_is_dual_path_even_with_low_risk():
    mode, reason = _router().pick_mode(
        findings=["a", "b", "c", "d"], governance_scores={"risk": 0.1}
    )
    assert mode == MODE_DUAL_PATH
    assert "findings" in reason


def test_unresolved_area_mismatch_is_dual_path_extended_budget():
    mode, reason = _router().pick_mode(
        findings=[], area_dynamics={"mismatch_score": 0.9}, governance_scores={"risk": 0.1}
    )
    assert mode == MODE_DUAL_PATH_EXTENDED_BUDGET
    assert "mismatch" in reason


def test_mismatch_takes_priority_even_with_low_risk_and_few_findings():
    """The table lists unresolved mismatch as its own row, not a special
    case of the other two -- it must win regardless of what the other
    signals say."""
    mode, _ = _router().pick_mode(
        findings=[], area_dynamics={"mismatch_score": 0.9}, governance_scores={"risk": 0.05}
    )
    assert mode == MODE_DUAL_PATH_EXTENDED_BUDGET


def test_the_middle_ground_between_thresholds_defaults_to_dual_path():
    """Neither clearly low-risk/few-findings nor high-risk/many-findings --
    the table names only the extremes, so this is a deliberate default,
    not an accident of the threshold ordering."""
    mode, reason = _router().pick_mode(
        findings=["a", "b"], governance_scores={"risk": 0.4}
    )
    assert mode == MODE_DUAL_PATH
    assert "neither clearly" in reason


def test_no_signals_at_all_defaults_to_one_shot():
    mode, _ = _router().pick_mode()
    assert mode == MODE_ONE_SHOT


def test_thresholds_are_the_documented_values():
    assert LOW_RISK_THRESHOLD == 0.3
    assert HIGH_RISK_THRESHOLD == 0.5
    assert FEW_FINDINGS_THRESHOLD == 2
    assert UNRESOLVED_MISMATCH_THRESHOLD == 0.5
