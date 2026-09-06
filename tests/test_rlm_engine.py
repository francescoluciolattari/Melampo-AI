import pytest

from melampo.evaluation.depth_comparison import DepthCase, compare_depths
from melampo.governance.audit_store import AppendOnlyAuditStore
from melampo.memory.context_environment import EnvironmentDocument
from melampo.memory.weaviate_adapter import WeaviateEnterpriseMemoryAdapter
from melampo.memory.weaviate_schema import QUARANTINED_HYPOTHESIS_CLASS
from melampo.reasoning.rlm_engine import (
    DATA_CLASS_DEIDENTIFIED,
    DATA_CLASS_REAL,
    DATA_CLASS_SYNTHETIC,
    STOP_FINAL,
    STOP_ITERATIONS,
    STOP_NO_ACTION,
    STOP_ROOT_ERROR,
    Budget,
    DataClassViolation,
    RlmEngine,
    parse_actions,
)
from melampo.reasoning.rlm_wiring import (
    AUDIT_EVENT_TRAJECTORY,
    TrajectoryAuditWriter,
    documents_from_adapter_store,
    search_via_adapter,
)


def _doc(document_id: str, text: str, data_class: str = DATA_CLASS_SYNTHETIC) -> EnvironmentDocument:
    return EnvironmentDocument(document_id=document_id, text=text, source="report", metadata={"data_class": data_class})


DOCS = (
    _doc("report_1", "Chest radiograph shows bibasilar opacities. Prednisone 40 mg daily was started."),
    _doc("note_1", "Patient reports progressive dyspnoea over three weeks with no fever."),
)


def _scripted(*outputs: str):
    """A root model that returns the given outputs in order, then finals."""
    queue = list(outputs)

    def _model(prompt: str) -> str:
        return queue.pop(0) if queue else "final(done)"

    return _model


# --------------------------------------------------------------------------
# Parsing: the model's output is actions, never code
# --------------------------------------------------------------------------


def test_actions_are_parsed_one_per_line_with_quoted_arguments():
    actions, ignored = parse_actions('grep("prednisone")\nslice(report_1, 0, 20)\nfinal("answer, with comma")')
    assert [item.verb for item in actions] == ["grep", "slice", "final"]
    assert actions[0].args == ("prednisone",)
    assert actions[1].args == ("report_1", "0", "20")
    assert actions[2].args == ("answer, with comma",)
    assert ignored == []


def test_prose_is_recorded_as_ignored_not_executed():
    actions, ignored = parse_actions("Let me think.\ngrep(fever)\nimport os; os.system('rm -rf /')")
    assert [item.verb for item in actions] == ["grep"]
    assert len(ignored) == 2


def test_no_verb_outside_the_six_is_recognised():
    """There is nothing to escape from: unknown verbs are not actions."""
    actions, ignored = parse_actions("exec(print(1))\neval('1+1')\nopen('/etc/passwd')\n__import__('os')")
    assert actions == []
    assert len(ignored) == 4


# --------------------------------------------------------------------------
# Data class is enforced in code
# --------------------------------------------------------------------------


def test_unmarked_documents_are_refused():
    engine = RlmEngine(root_model=_scripted())
    with pytest.raises(DataClassViolation, match="declare data_class"):
        engine.run("c1", [EnvironmentDocument("d", "text", metadata={})], "q")


def test_real_data_is_refused_in_phase_one():
    engine = RlmEngine(root_model=_scripted())
    with pytest.raises(DataClassViolation, match="not permitted"):
        engine.run("c1", [_doc("d", "text", DATA_CLASS_REAL)], "q")


def test_synthetic_and_deidentified_are_admitted():
    engine = RlmEngine(root_model=_scripted())
    assert engine.run("c1", [_doc("d", "text", DATA_CLASS_SYNTHETIC)], "q").completed
    assert engine.run("c1", [_doc("d", "text", DATA_CLASS_DEIDENTIFIED)], "q").completed


def test_an_engine_can_be_widened_to_real_data_only_by_explicit_construction():
    engine = RlmEngine(root_model=_scripted(), allowed_data_classes=frozenset({DATA_CLASS_REAL}))
    trajectory = engine.run("c1", [_doc("d", "text", DATA_CLASS_REAL)], "q")
    assert trajectory.as_dict()["health_data"] is True


# --------------------------------------------------------------------------
# Depth is capped
# --------------------------------------------------------------------------


def test_depth_above_one_is_not_permitted():
    with pytest.raises(ValueError, match="not permitted"):
        RlmEngine(root_model=_scripted(), depth=2)


def test_depth_zero_discards_the_sub_model():
    engine = RlmEngine(root_model=_scripted(), sub_model=lambda q, f: "x", depth=0)
    assert engine.sub_model is None


def test_query_is_refused_at_depth_zero():
    engine = RlmEngine(root_model=_scripted('query("what dose", report_1, 0, 80)', "final(x)"), depth=0)
    trajectory = engine.run("c1", DOCS, "q")
    assert trajectory.steps[0].error == "query is not available at depth 0"


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


def test_a_run_navigates_and_completes_with_final():
    engine = RlmEngine(root_model=_scripted("describe()", "grep(prednisone)", "final(prednisone 40 mg daily)"))
    trajectory = engine.run("c1", DOCS, "what steroid dose?")

    assert trajectory.completed
    assert trajectory.stop_reason == STOP_FINAL
    assert trajectory.final_answer == "prednisone 40 mg daily"
    assert [step.action.verb for step in trajectory.steps] == ["describe", "grep", "final"]


def test_fragments_carry_offsets_and_coverage_is_measured():
    engine = RlmEngine(root_model=_scripted("grep(prednisone)", "final(x)"))
    trajectory = engine.run("c1", DOCS, "q")

    evidence = trajectory.evidence()
    assert evidence
    assert evidence[0]["provenance"]["char_end"] > evidence[0]["provenance"]["char_start"]
    assert 0.0 < trajectory.coverage["coverage_ratio"] < 1.0


def test_the_sub_model_is_called_at_depth_one_and_counted():
    calls = []

    def sub(question, fragment):
        calls.append((question, fragment))
        return "40 mg"

    engine = RlmEngine(root_model=_scripted('query("dose?", report_1, 0, 80)', "final(40 mg)"), sub_model=sub, depth=1)
    trajectory = engine.run("c1", DOCS, "q")

    assert len(calls) == 1
    assert "Prednisone" in calls[0][1]
    assert trajectory.budget["sub_model_calls"] == 1
    assert trajectory.coverage["llm_calls"] == 1


def test_a_run_without_final_is_not_reported_as_completed():
    """'The model stopped' and 'the model finished' look identical downstream."""
    engine = RlmEngine(root_model=_scripted("grep(fever)", "I have nothing more to add."))
    trajectory = engine.run("c1", DOCS, "q")
    assert trajectory.stop_reason == STOP_NO_ACTION
    assert trajectory.completed is False


def test_exhausting_the_iteration_budget_fails_explicitly():
    engine = RlmEngine(root_model=lambda prompt: "grep(fever)")
    trajectory = engine.run("c1", DOCS, "q", budget=Budget(max_iterations=3))
    assert trajectory.stop_reason == STOP_ITERATIONS
    assert trajectory.completed is False
    assert trajectory.budget["iterations"] == 3


def test_a_root_model_error_ends_the_run_with_a_named_reason():
    def failing(prompt):
        raise RuntimeError("provider down")

    trajectory = RlmEngine(root_model=failing).run("c1", DOCS, "q")
    assert trajectory.stop_reason == STOP_ROOT_ERROR


def test_a_bad_action_is_recorded_not_fatal():
    engine = RlmEngine(root_model=_scripted("slice(missing_document, 0, 10)", "final(x)"))
    trajectory = engine.run("c1", DOCS, "q")
    assert trajectory.steps[0].error
    assert trajectory.completed


def test_the_sub_model_budget_is_enforced():
    engine = RlmEngine(
        root_model=_scripted('query("a", report_1, 0, 10)', 'query("b", report_1, 0, 10)', "final(x)"),
        sub_model=lambda q, f: "y",
        depth=1,
    )
    trajectory = engine.run("c1", DOCS, "q", budget=Budget(max_sub_model_calls=1))
    assert trajectory.steps[1].error == "sub-model call budget exhausted"


# --------------------------------------------------------------------------
# Retrieval contract
# --------------------------------------------------------------------------


def test_a_completed_run_renders_in_the_shared_contract():
    from melampo.memory.retrieval_contract import assert_retrieval_contract

    engine = RlmEngine(root_model=_scripted("grep(prednisone)", "final(x)"))
    trajectory = engine.run("c1", DOCS, "q")
    payload = engine.to_retrieval_payload(trajectory, "q")

    assert_retrieval_contract(payload)
    assert payload["retrieval_mode"] == "rlm_environment"
    assert payload["retrieval_quality"]["coverage_basis"] == "corpus_characters"


def test_an_incomplete_run_yields_no_evidence_rather_than_partial_evidence():
    """A truncated dossier that looks whole is worse than none."""
    engine = RlmEngine(root_model=lambda prompt: "grep(prednisone)")
    trajectory = engine.run("c1", DOCS, "q", budget=Budget(max_iterations=2))
    payload = engine.to_retrieval_payload(trajectory, "q")

    assert trajectory.steps, "fragments were retrieved"
    assert payload["evidence"] == [], "but none is presented as the answer"
    assert payload["retrieval_quality"]["completed"] is False


# --------------------------------------------------------------------------
# Wiring: the environment inherits the quarantine
# --------------------------------------------------------------------------


def _adapter_with_candidate() -> WeaviateEnterpriseMemoryAdapter:
    adapter = WeaviateEnterpriseMemoryAdapter()
    adapter.fallback_store.upsert(
        text="amyloidosis documented in the admission report",
        metadata={"class_name": "Pathology", "document_id": "report_9", "source_type": "clinical_document"},
        learning_status="grounded",
    )
    adapter.fallback_store.upsert(
        text="amyloidosis considered as a synthetic alternative",
        metadata={"class_name": QUARANTINED_HYPOTHESIS_CLASS, "document_id": "cand_1", "source_type": "synthetic_dream_candidate"},
        learning_status="candidate",
    )
    return adapter


def test_the_engine_cannot_reach_a_quarantined_candidate_through_search():
    """Verified by attempt: a candidate is stored, and the engine tries to find it."""
    adapter = _adapter_with_candidate()
    engine = RlmEngine(root_model=_scripted("search(amyloidosis)", "final(x)"))
    documents = documents_from_adapter_store(adapter, data_class=DATA_CLASS_SYNTHETIC)
    trajectory = engine.run("c1", documents, "q", search_fn=search_via_adapter(adapter))

    surfaced = " ".join(fragment["text"] for step in trajectory.steps for fragment in step.fragments)
    assert "admission report" in surfaced
    assert "synthetic alternative" not in surfaced


def test_quarantined_records_are_absent_from_the_environment_itself():
    """Even describe() must not reveal a candidate's existence."""
    documents = documents_from_adapter_store(_adapter_with_candidate(), data_class=DATA_CLASS_SYNTHETIC)
    assert [item.document_id for item in documents] == ["report_9"]


# --------------------------------------------------------------------------
# Trajectories are health records
# --------------------------------------------------------------------------


def test_a_trajectory_is_written_to_the_audit_store_with_its_data_class(tmp_path):
    store = AppendOnlyAuditStore(path=tmp_path / "audit.jsonl")
    trajectory = RlmEngine(root_model=_scripted("final(x)")).run("c1", DOCS, "q")
    TrajectoryAuditWriter(store).write(trajectory, operator="tester")

    event = store.read_all()[-1]
    assert event["event_type"] == AUDIT_EVENT_TRAJECTORY
    assert event["metadata"]["health_data"] is False
    assert event["metadata"]["retention_class"] == "research"


def test_a_real_case_trajectory_is_marked_health_data(tmp_path):
    store = AppendOnlyAuditStore(path=tmp_path / "audit.jsonl")
    engine = RlmEngine(root_model=_scripted("final(x)"), allowed_data_classes=frozenset({DATA_CLASS_REAL}))
    trajectory = engine.run("c1", [_doc("d", "text", DATA_CLASS_REAL)], "q")
    TrajectoryAuditWriter(store).write(trajectory)

    metadata = store.read_all()[-1]["metadata"]
    assert metadata["health_data"] is True
    assert metadata["retention_class"] == "clinical_record"


# --------------------------------------------------------------------------
# Depth comparison
# --------------------------------------------------------------------------


def test_depth_comparison_is_paired_and_reports_a_verdict():
    def root(prompt):
        return "grep(prednisone)\nfinal(x)" if "query(" not in prompt else 'query("dose", report_1, 0, 80)\nfinal(x)'

    report = compare_depths([DepthCase("c1", DOCS, "q")], root, lambda q, f: "40 mg")
    payload = report.as_dict()
    assert payload["cases"] == 1
    assert payload["outcomes"][0]["depth0_completed"] and payload["outcomes"][0]["depth1_completed"]
    assert isinstance(payload["verdict"], str) and payload["verdict"]


def test_when_depth_one_adds_nothing_the_verdict_says_so():
    report = compare_depths([DepthCase("c1", DOCS, "q")], _scripted_loop("grep(prednisone)\nfinal(x)"), lambda q, f: "y")
    assert "not justified" in report.verdict()


def _scripted_loop(output: str):
    return lambda prompt: output
