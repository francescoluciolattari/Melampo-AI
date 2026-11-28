from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from melampo.app import build_default_runtime
from melampo.governance.audit_store import AppendOnlyAuditStore
from melampo.memory.vector_memory import PersistentJsonlVectorStore
from melampo.training.nexus_candidate_store import NexusCandidateStore


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
    # explicit rules, no weights (28 November 2026)
    assert "governance_scores" not in result
    assert result["signals"]["escalate"] is True
    assert result["signals"]["reasons"] == [
        {"rule": "document_not_read", "sentence": "1 documents of the case were read by no interpreter"}
    ]
    assert [f["rule"] for f in result["signals"]["flags"]] == ["no_evidence"]
    # no interpreter on this machine: the note is not read, and that goes to a person (28 November 2026)
    assert result["discordances"]["kinds"]["unread"] == 1
    assert result["routing_reason"] == "1 discordances in the reading (unread 1)"
    assert "area_dynamics" not in result
    assert result["intuition"]["intuition"] == "candidate_1"
    assert result["intuition"]["candidate_scores"] == [
        {"label": "candidate_1", "score": 2.28, "source": "semantic_memory"},
        {"label": "candidate_2", "score": 2.05, "source": "knowledge_graph"},
        {"label": "candidate_3", "score": 1.81, "source": "episodic_memory"},
    ]
    assert result["nexus"]["accepted"] is False
    assert result["nexus"]["filter_assessment"]["replay_mode"] == "corrective_replay"
    assert [item["kind"] for item in result["nexus"]["alternative_hypotheses"]] == [
        "adjacent_case",
        "counterfactual_case",
        "contradiction_revision",
        "discordance_resolution",
    ]
    # the unread note escalates to a person and does not withhold the result; the result comes from
    # case_diagnosis (28 November 2026), the October path only compared
    policy = result["diagnostic_result"]["policy"]
    assert result["diagnostic_result"]["result_label"] == "no_candidate_found"
    assert (policy["abstain"], policy["escalate"], policy["reasons"]) == (False, True, [])
    assert policy["escalation_reasons"] == ["document_not_read"]
    # no disease profiles configured on this machine: case_diagnosis cannot run, and the result says why
    assert policy["flags"] == ["no_evidence", "case_diagnosis:no_profiles_configured"]
    assert result["diagnostic_result"]["differential"] == []
    assert result["diagnostic_result"]["october_path"]["top"] == "candidate_1"


def test_any_declared_severity_is_read_without_error():
    runtime = build_default_runtime()
    severities = [-10, -1, 0, 0.5, 1, 10, "bad", None]
    for index, severity in enumerate(severities):
        result = runtime.pipeline.run(
            {
                "case_id": f"edge-{index}",
                "report_text": "",
                "clinical_severity": severity,
            }
        )
        # a number is not a declared severity: only a word is (reasoning.case_signals)
        assert result["signals"]["declared_severity"] == "not_declared"


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
    audit.append(
        "candidate_promoted", {"candidate_id": "nexus-1", "status": "candidate"}
    )

    reloaded = AppendOnlyAuditStore(path)
    events = reloaded.read_all()
    assert len(events) == 1
    assert events[0]["event_type"] == "candidate_promoted"
    assert events[0]["payload"]["candidate_id"] == "nexus-1"


def test_promotion_and_retrieval_can_run_concurrently():
    store = NexusCandidateStore()
    for index in range(20):
        record = store.create_candidate(
            {"text": f"candidate {index}", "metadata": {"provenance": "synthetic"}},
            case_id=f"case-{index}",
        )
        store.attach_validation(
            record.candidate_id,
            {"allowed_for_promotion": True, "observed": {"provenance_available": True}},
        )

    ids = [record.candidate_id for record in store.list_by_status(["candidate"])]

    def promote(candidate_id: str) -> str:
        return store.attach_promotion_decision(
            candidate_id, {"target_learning_status": "promoted"}
        )["learning_status"]

    def retrieve(_: int) -> int:
        return len(store.export_memory_documents(["candidate", "promoted"]))

    with ThreadPoolExecutor(max_workers=8) as executor:
        promoted = list(executor.map(promote, ids[:10]))
        counts = list(executor.map(retrieve, range(10)))

    assert promoted == ["promoted"] * 10
    assert min(counts) >= 20
    assert len(store.list_by_status(["promoted"])) == 10
