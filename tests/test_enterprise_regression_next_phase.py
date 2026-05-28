from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from melampo.app import build_default_runtime
from melampo.memory.vector_memory import PersistentJsonlVectorStore
from melampo.governance.audit_store import AppendOnlyAuditStore
from melampo.training.dream_candidate_store import DreamCandidateStore


def test_end_to_end_pipeline_golden_snapshot_remains_stable():
    payload = {
        "case_id": "golden-001",
        "report_text": "possible pulmonary lesion with cough",
        "patient_complaints": "cough",
        "exposures": {"smoking": "former"},
        "provenance": {"source": "synthetic_golden"},
    }
    result = build_default_runtime().pipeline.run(payload)

    assert result["case_id"] == "golden-001"
    assert result["bundle_keys"] == ["patient", "diagnostic_report"]
    assert result["governance_scores"] == {
        "risk": 0.094,
        "uncertainty": 0.403,
        "dream_coherence": 0.525,
        "missing_evidence": 1.0,
        "retrieval_coverage": 0.0,
        "mean_grounding_score": 0.513,
        "mean_area_uncertainty": 0.66,
        "mismatch_index": 0.028,
        "prediction_error": 0.044,
        "convergence_index": 0.703,
        "memory_backed_retrieval": False,
        "fallback_penalty": 0.2,
        "weak_provenance": 0.0,
        "clinical_severity": 0.0,
        "derivation": "runtime_governance_scores_not_hardcoded_constants",
    }
    assert {key: result["area_dynamics"][key] for key in ["coherence_score", "mismatch_score", "pi_score", "prediction_error", "precision_weighted_coherence", "deep_inference_score"]} == {
        "coherence_score": 0.554,
        "mismatch_score": 0.092,
        "pi_score": 0.669,
        "prediction_error": 0.044,
        "precision_weighted_coherence": 0.699,
        "deep_inference_score": 0.754,
    }
    assert result["intuition"]["intuition"] == "candidate_1"
    assert result["intuition"]["deductive_filter"]["reasoning_mode"] == "rapid_intuition"
    assert result["intuition"]["deductive_filter"]["top_areas"] == ["visual_diagnostic", "language_listening"]
    assert result["intuition"]["candidate_scores"] == [
        {"mode": "rapid_intuition", "label": "candidate_1", "score": 8.461},
        {"mode": "rational_revision", "label": "candidate_2", "score": 2.62},
        {"mode": "contradiction_revision", "label": "golden-001_alt_1", "score": 1.629},
    ]
    assert result["dream"]["accepted"] is False
    assert result["dream"]["filter_assessment"]["replay_mode"] == "corrective_replay"
    assert [item["kind"] for item in result["dream"]["alternative_hypotheses"]] == ["adjacent_case", "boundary_case", "contradiction_revision"]
    assert result["diagnostic_result"]["result_label"] == "candidate_1"
    assert result["diagnostic_result"]["policy"]["allow_candidate_result"] is True


def test_numeric_edge_cases_remain_bounded_without_property_dependency():
    runtime = build_default_runtime()
    severities = [-10, -1, 0, 0.5, 1, 10, "bad", None]
    for index, severity in enumerate(severities):
        result = runtime.pipeline.run({"case_id": f"edge-{index}", "report_text": "", "clinical_severity": severity})
        governance = result["governance_scores"]
        neuro = result["area_dynamics"]["neuro_dynamic_metrics"]
        for key in ["risk", "uncertainty", "dream_coherence", "mismatch_index", "prediction_error", "convergence_index"]:
            assert 0.0 <= governance[key] <= 1.0
        for key in ["pi_score", "prediction_error", "mismatch_index", "action_potential_gate", "deep_inference_score"]:
            assert 0.0 <= neuro[key] <= 1.0


def test_persistent_vector_store_survives_reload(tmp_path: Path):
    path = tmp_path / "memory.jsonl"
    store = PersistentJsonlVectorStore(path=path)
    store.upsert_text("record-1", "pulmonary opacity", metadata={"case_id": "case-1"})
    store.promote("record-1", "validated research fixture")

    reloaded = PersistentJsonlVectorStore(path=path)
    assert reloaded.describe()["record_count"] == 1
    assert reloaded.search("opacity", limit=1)[0]["record_id"] == "record-1"
    assert reloaded.records["record-1"].learning_status == "promoted"


def test_append_only_audit_store_persists_events(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    audit = AppendOnlyAuditStore(path)
    audit.append("candidate_promoted", {"candidate_id": "dream-1", "status": "candidate"})

    reloaded = AppendOnlyAuditStore(path)
    events = reloaded.read_all()
    assert len(events) == 1
    assert events[0]["event_type"] == "candidate_promoted"
    assert events[0]["payload"]["candidate_id"] == "dream-1"


def test_promotion_and_retrieval_can_run_concurrently():
    store = DreamCandidateStore()
    for index in range(20):
        record = store.create_candidate({"text": f"candidate {index}", "metadata": {"provenance": "synthetic"}}, case_id=f"case-{index}")
        store.attach_validation(
            record.candidate_id,
            {"allowed_for_promotion": True, "observed": {"provenance_available": True}},
        )

    ids = [record.candidate_id for record in store.list_by_status(["candidate"])]

    def promote(candidate_id: str) -> str:
        return store.attach_promotion_decision(candidate_id, {"target_learning_status": "promoted"})["learning_status"]

    def retrieve(_: int) -> int:
        return len(store.export_memory_documents(["candidate", "promoted"]))

    with ThreadPoolExecutor(max_workers=8) as executor:
        promoted = list(executor.map(promote, ids[:10]))
        counts = list(executor.map(retrieve, range(10)))

    assert promoted == ["promoted"] * 10
    assert min(counts) >= 20
    assert len(store.list_by_status(["promoted"])) == 10
