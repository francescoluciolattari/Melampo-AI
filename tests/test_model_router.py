"""Tests for ModelRouter (D1): the two-step router that replaces a 12-line
stub which routed on a task name string, never called by anything,
verified before this rewrite. pick_pending_case() runs the same
pending_case_router.route_payload() check, callable early in
clinical_pipeline.run() before real signals exist; pick_mode() implements
the September decision record's table, callable once they do.
"""

from melampo.orchestration.model_router import (
    FEW_FINDINGS_THRESHOLD,
    MODE_DUAL_PATH,
    MODE_DUAL_PATH_EXTENDED_BUDGET,
    MODE_ONE_SHOT,
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
        case_id="case-1",
        learning_status="needs_review",
    )
    router = ModelRouter(candidate_store=store)

    decision = router.pick_pending_case(
        {"case_id": "case-1", "report_text": "follow-up"}
    )

    assert decision.action == "merge_and_rerun"


# --------------------------------------------------------------------------
# pick_mode: the four-row table from docs-archive-2026-11:docs/rlm_on_memory_decision_record.md
# --------------------------------------------------------------------------


def test_few_findings_not_declared_severe_is_one_shot():
    mode, reason = _router().pick_mode(findings=["a"], signals={"declared_severity": "not_declared"})
    assert mode == MODE_ONE_SHOT
    assert reason == "no discordances, not declared severe, 1 findings"


def test_a_case_declared_severe_is_dual_path():
    mode, reason = _router().pick_mode(findings=[], signals={"declared_severity": "high"})
    assert (mode, reason) == (MODE_DUAL_PATH, "the case was declared severe")


def test_many_findings_is_dual_path():
    mode, reason = _router().pick_mode(findings=["a", "b", "c", "d"])
    assert (mode, reason) == (MODE_DUAL_PATH, "4 findings > 2")


def test_discordances_in_the_reading_are_dual_path_extended_budget():
    mode, reason = _router().pick_mode(
        findings=[],
        discordances={"count": 2, "kinds": {"card": 1, "reference": 0, "image": 0, "document": 1}},
        signals={"declared_severity": "high"},
    )
    assert mode == MODE_DUAL_PATH_EXTENDED_BUDGET
    assert reason == "2 discordances in the reading (card 1, document 1)"


def test_no_discordances_leave_the_other_rows():
    mode, _ = _router().pick_mode(findings=[], discordances={"count": 0})
    assert mode == MODE_ONE_SHOT


def test_no_signals_at_all_defaults_to_one_shot():
    mode, _ = _router().pick_mode()
    assert mode == MODE_ONE_SHOT


def test_thresholds_are_the_documented_values():
    assert FEW_FINDINGS_THRESHOLD == 2
