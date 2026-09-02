import pytest

from melampo.memory.context_environment import ContextEnvironment, EnvironmentDocument
from melampo.memory.retrieval_contract import (
    RETRIEVAL_MODE_RLM,
    assert_retrieval_contract,
    validate_retrieval_payload,
)
from melampo.reasoning.retrieval_reconciliation import (
    DISPOSITION_CONFIRMED,
    DISPOSITION_RLM_ONLY_VERIFIED,
    build_dual_path_payload,
    conflict_inputs_for_neuro_dynamics,
    reconcile,
)
from melampo.training.hypothesis_channel import (
    HYPOTHESIS_ROLE,
    HypothesisChannel,
    HypothesisEnvelope,
    IndeterminacyGate,
    assert_not_evidence,
)


def _environment() -> ContextEnvironment:
    return ContextEnvironment.from_documents(
        [
            EnvironmentDocument(
                document_id="report_1",
                text="Chest radiograph shows bibasilar opacities. Prednisone 40 mg daily was started.",
                source="radiology_report",
                page=1,
            ),
            EnvironmentDocument(
                document_id="note_1",
                text="Patient reports progressive dyspnoea over three weeks with no fever.",
                source="clinical_note",
                section="history",
            ),
        ]
    )


def test_environment_fragments_always_carry_character_offsets():
    environment = _environment()
    fragments = environment.grep("prednisone")

    assert fragments
    fragment = fragments[0]
    assert fragment.document_id == "report_1"
    assert fragment.char_end > fragment.char_start
    assert "Prednisone" in fragment.text

    evidence = fragment.as_evidence(rank=1, focus="therapy")
    assert evidence["provenance"]["char_start"] == fragment.char_start
    assert evidence["provenance"]["char_end"] == fragment.char_end
    assert evidence["record_id"].startswith("report_1:")


def test_coverage_is_measured_not_assumed():
    environment = _environment()
    assert environment.coverage()["coverage_ratio"] == 0.0

    environment.slice("report_1", 0, 20)
    partial = environment.coverage()
    assert 0.0 < partial["coverage_ratio"] < 1.0
    assert partial["documents_touched"] == ["report_1"]
    assert partial["queries_issued"] == 1


def test_overlapping_spans_are_not_double_counted():
    environment = _environment()
    environment.slice("report_1", 0, 30)
    first = environment.coverage()["inspected_characters"]
    environment.slice("report_1", 10, 30)
    second = environment.coverage()["inspected_characters"]
    assert first == second


def test_unknown_document_raises_instead_of_returning_empty():
    environment = _environment()
    with pytest.raises(KeyError):
        environment.slice("missing_document", 0, 10)


def test_search_without_backend_returns_nothing_rather_than_inventing_evidence():
    environment = _environment()
    assert environment.search("dyspnoea") == []


def test_contract_rejects_undeclared_memory_backing():
    payload = {
        "query": "therapy",
        "focus": "therapy",
        "target_areas": ["therapy"],
        "status": "grounded_retrieval_ready",
        "retrieval_mode": RETRIEVAL_MODE_RLM,
        "evidence": [{"record_id": "report_1:0-20", "text": "…"}],
        "evidence_count": 1,
        "retrieval_quality": {
            "memory_backed": False,
            "coverage": 0.4,
            "mean_grounding_score": 0.7,
            "fallback_used": False,
        },
    }
    codes = {violation.code for violation in validate_retrieval_payload(payload)}
    assert "undeclared_memory_backing" in codes


def test_contract_rejects_untraceable_evidence():
    payload = {
        "query": "therapy",
        "focus": "therapy",
        "target_areas": ["therapy"],
        "status": "grounded_retrieval_ready",
        "retrieval_mode": RETRIEVAL_MODE_RLM,
        "evidence": [{"text": "no provenance at all"}],
        "evidence_count": 1,
        "retrieval_quality": {
            "memory_backed": True,
            "coverage": 0.4,
            "mean_grounding_score": 0.7,
            "fallback_used": False,
        },
    }
    with pytest.raises(ValueError):
        assert_retrieval_contract(payload)


def _evidence(record_id: str, *, offsets: bool = True, score: float = 0.6) -> dict:
    item = {"record_id": record_id, "text": record_id, "grounding_score": score}
    if offsets:
        document_id, span = record_id.split(":")
        start, end = span.split("-")
        item["provenance"] = {"document_id": document_id, "char_start": int(start), "char_end": int(end)}
    return item


def test_reconciliation_confirms_overlap_and_admits_verified_recall_gain():
    one_shot = {"query": "q", "evidence": [_evidence("report_1:0-20"), _evidence("note_1:0-15")]}
    recursive = {"query": "q", "evidence": [_evidence("report_1:0-20"), _evidence("note_1:40-60")]}

    verdict = reconcile(one_shot, recursive)

    dispositions = {item["record_id"]: item["reconciliation"] for item in verdict.evidence}
    assert dispositions["report_1:0-20"] == DISPOSITION_CONFIRMED
    assert dispositions["note_1:40-60"] == DISPOSITION_RLM_ONLY_VERIFIED
    assert verdict.recall_gain == 1
    assert verdict.overreach_blocked == 0
    assert 0.0 < verdict.agreement_ratio < 1.0


def test_recursive_only_findings_without_offsets_are_discarded_as_overreach():
    one_shot = {"query": "q", "evidence": [_evidence("report_1:0-20")]}
    recursive = {
        "query": "q",
        "evidence": [_evidence("report_1:0-20"), {"text": "a synthesised connection", "grounding_score": 0.9}],
    }

    verdict = reconcile(one_shot, recursive)

    assert verdict.overreach_blocked == 1
    assert verdict.recall_gain == 0
    assert all(item.get("text") != "a synthesised connection" for item in verdict.evidence)


def test_full_disagreement_produces_maximum_conflict_signal():
    one_shot = {"query": "q", "evidence": [_evidence("report_1:0-20")]}
    recursive = {"query": "q", "evidence": [_evidence("note_1:0-15")]}

    verdict = reconcile(one_shot, recursive)

    assert verdict.agreement_ratio == 0.0
    assert verdict.conflict_signal > 0.5
    assert any("low confidence" in note for note in verdict.notes)

    inputs = conflict_inputs_for_neuro_dynamics(verdict)
    assert inputs["retrieval_conflict_signal"] == verdict.conflict_signal


def test_dual_path_payload_satisfies_the_shared_contract():
    one_shot = {
        "query": "therapy",
        "focus": "therapy",
        "target_areas": ["therapy"],
        "evidence": [_evidence("report_1:0-20")],
        "case_context_keys": ["age"],
    }
    recursive = {"query": "therapy", "evidence": [_evidence("report_1:0-20"), _evidence("note_1:0-15")]}

    verdict = reconcile(one_shot, recursive)
    payload = build_dual_path_payload(one_shot, recursive, verdict, coverage={"coverage_ratio": 0.62})

    assert_retrieval_contract(payload)
    assert payload["retrieval_quality"]["memory_backed"] is True
    assert payload["retrieval_quality"]["coverage"] == 0.62
    assert payload["reconciliation"]["recall_gain"] == 1


def test_hypothesis_channel_stays_closed_when_the_differential_has_converged():
    channel = HypothesisChannel()
    result = channel.open_for(
        [HypothesisEnvelope(label="atypical_presentation", novelty_score=0.8)],
        dynamics={"convergence_index": 0.9, "conflict_load": 0.6},
        risk=0.8,
    )
    assert result["channel_open"] is False
    assert result["hypotheses"] == []
    assert "differential already converged" in result["gate"]["blocking_reasons"]


def test_hypothesis_channel_opens_under_high_indeterminacy():
    channel = HypothesisChannel(max_hypotheses=2)
    result = channel.open_for(
        [
            HypothesisEnvelope(label="alt_a", novelty_score=0.3),
            HypothesisEnvelope(label="alt_b", novelty_score=0.9),
            HypothesisEnvelope(label="alt_c", novelty_score=0.6),
        ],
        dynamics={"convergence_index": 0.3, "conflict_load": 0.7},
        risk=0.6,
    )

    assert result["channel_open"] is True
    assert [item["label"] for item in result["hypotheses"]] == ["alt_b", "alt_c"]
    for hypothesis in result["hypotheses"]:
        assert hypothesis["role"] == HYPOTHESIS_ROLE
        assert hypothesis["usable_as_evidence"] is False
        assert hypothesis["learning_status"] == "candidate"


def test_hypotheses_cannot_cross_into_the_evidence_path():
    hypothesis = HypothesisEnvelope(label="alt_a", novelty_score=0.5).as_exclusion_hypothesis()
    with pytest.raises(ValueError):
        assert_not_evidence([hypothesis])

    assert_not_evidence([_evidence("report_1:0-20")])


def test_gate_requires_all_three_conditions():
    gate = IndeterminacyGate()
    assert gate.is_open({"convergence_index": 0.3, "conflict_load": 0.7}, risk=0.6) is True
    assert gate.is_open({"convergence_index": 0.3, "conflict_load": 0.1}, risk=0.6) is False
    assert gate.is_open({"convergence_index": 0.3, "conflict_load": 0.7}, risk=0.1) is False
