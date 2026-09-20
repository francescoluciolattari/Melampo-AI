from melampo.app import build_default_runtime


def test_clinical_pipeline_runs_minimal_payload():
    runtime = build_default_runtime()
    result = runtime.pipeline.run({"case_id": "case-001", "report_text": "possible pulmonary lesion"})
    assert result["case_id"] == "case-001"
    assert "retrieval" in result
    assert "area_signals" in result
    assert sorted(result["area_signals"].keys()) == ["case_context", "epidemiology", "language_listening", "visual_diagnostic"]
    assert "area_dynamics" in result
    assert result["area_dynamics"]["coherence_score"] >= 0.0
    assert result["area_dynamics"]["mismatch_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["top_areas"]
    assert result["intuition"]["deductive_filter"]["convergence_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["conflict_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["coherence_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["mismatch_score"] >= 0.0
    assert result["intuition"]["deductive_filter"]["area_pair_bonus"] >= 0.0
    assert result["intuition"]["deductive_filter"]["disagreement_penalty"] >= 0.0
    assert result["intuition"]["deductive_filter"]["revision_bias"] in ["exploratory", "conservative"]
    assert isinstance(result["intuition"]["deductive_filter"]["contradiction_rehearsal"], bool)
    assert result["intuition"]["deductive_filter"]["reasoning_mode"] in ["rapid_intuition", "rational_revision", "contradiction_revision"]
    assert result["intuition"]["rapid_intuition"] == "candidate_1"
    assert len(result["intuition"]["candidate_scores"]) == 3
    assert "coordinated" in result
    assert result["coordinated"]["differential"]["hypotheses"]
    assert result["coordinated"]["differential"]["hypotheses"][0]["hypothesis_type"] in ["primary_hypothesis", "revision_hypothesis", "contradiction_revision_hypothesis"]
    assert result["coordinated"]["differential"]["hypotheses"][0]["hypothesis_domain"] in ["multimodal_led", "imaging_led", "language_led", "epidemiology_led", "mismatch_resolution_led"]
    assert result["coordinated"]["differential"]["recommended_actions"]
    assert result["coordinated"]["differential"]["recommended_actions"][0]["category"] in ["confirmation_test", "disambiguation_test", "multimodal_reconciliation"]
    assert result["coordinated"]["differential"]["recommended_tests"]
    assert result["coordinated"]["differential"]["hypotheses"][0]["support_signals"]
    assert result["coordinated"]["differential"]["support_profiles"]
    assert result["coordinated"]["differential"]["contradiction_profiles"]
    assert "critique" in result
    assert result["critique"]["status"] == "reviewed"
    assert result["critique"]["suggestions"]
    assert result["critique"]["prioritized_actions"]
    assert result["critique"]["prioritized_actions"][0]["priority"] in ["high", "medium"]
    assert "nexus" in result
    assert "filter_assessment" in result["nexus"]
    assert result["nexus"]["filter_assessment"]["replay_mode"] in ["stabilizing_replay", "boundary_replay", "corrective_replay"]
    assert "rehearsal_profile" in result["nexus"]
    assert len(result["nexus"]["alternative_hypotheses"]) >= 2


# --------------------------------------------------------------------------
# pending_case_routing: the check added to run() itself, reusing the same
# NexusCandidateStore the promotion chain writes to
# --------------------------------------------------------------------------


def test_a_first_time_case_is_routed_as_new():
    runtime = build_default_runtime()
    result = runtime.pipeline.run({"case_id": "case-pending-1", "report_text": "first report"})
    assert result["pending_case_routing"]["action"] == "new_case"


def test_a_case_pending_after_offline_processing_merges_new_findings():
    """The real lifecycle: a case enqueues on its first run, only becomes
    genuinely "pending" once a low-activity pass (run_once) has processed
    the queue -- exactly what a real scheduled trigger would do."""
    runtime = build_default_runtime()
    pipeline = runtime.pipeline

    pipeline.run({"case_id": "case-pending-2", "report_text": "Initial findings: persistent cough."})
    pipeline._nexus_scheduler_instance().run_once(activity={"active_requests": 0, "idle_seconds": 100})

    result = pipeline.run({"case_id": "case-pending-2", "report_text": "Follow-up CT: bilateral opacity confirmed."})

    routing = result["pending_case_routing"]
    assert routing["action"] == "merge_and_rerun"
    assert "Follow-up CT: bilateral opacity confirmed." in routing["merged_report_text"]
    assert "Initial findings: persistent cough." in routing["merged_report_text"]


def test_a_confirmed_diagnosis_for_a_pending_case_is_recognised_but_not_yet_acted_on():
    """confirm_and_train is attached to the result for a future caller to
    act on -- executing it (training extraction, then deleting the raw
    record) is separate, not-yet-built work, deliberately not done here."""
    runtime = build_default_runtime()
    pipeline = runtime.pipeline

    pipeline.run({"case_id": "case-pending-3", "report_text": "Initial findings."})
    pipeline._nexus_scheduler_instance().run_once(activity={"active_requests": 0, "idle_seconds": 100})

    result = pipeline.run({"case_id": "case-pending-3", "confirmed_diagnosis": "Sarcoidosis"})

    assert result["pending_case_routing"]["action"] == "confirm_and_train"


# --------------------------------------------------------------------------
# patient_matching.py's fallback, exercised through the real pipeline:
# a second submission for the same patient, with no case_id known, still
# finds and merges into the first case.
# --------------------------------------------------------------------------


def test_a_second_submission_with_no_case_id_finds_the_same_patient(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PASSWORD", "test-secret")
    monkeypatch.chdir(tmp_path)
    runtime = build_default_runtime()
    pipeline = runtime.pipeline

    first = pipeline.run({
        "report_text": "Initial findings.",
        "patient_name": "Mario", "patient_surname": "Rossi",
        "case_date": "2026-09-20", "diagnostic_question": "Evaluate for aortic root aneurysm",
    })
    generated_case_id = first["pending_case_routing"]["case_id"]
    assert generated_case_id != ""
    pipeline._nexus_scheduler_instance().run_once(activity={"active_requests": 0, "idle_seconds": 100})

    second = pipeline.run({
        "report_text": "Follow-up CT.",
        "patient_name": "MARIO", "patient_surname": "  Rossi  ",
        "case_date": "2026-09-20", "diagnostic_question": "Suspected aortic root aneurysm",
    })

    assert second["pending_case_routing"]["action"] == "merge_and_rerun"
    assert second["pending_case_routing"]["case_id"] == generated_case_id


def test_a_different_patient_same_date_is_treated_as_a_new_case(monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PASSWORD", "test-secret")
    monkeypatch.chdir(tmp_path)
    runtime = build_default_runtime()
    pipeline = runtime.pipeline

    pipeline.run({
        "report_text": "Initial findings.",
        "patient_name": "Mario", "patient_surname": "Rossi",
        "case_date": "2026-09-20", "diagnostic_question": "Evaluate for aortic root aneurysm",
    })
    pipeline._nexus_scheduler_instance().run_once(activity={"active_requests": 0, "idle_seconds": 100})

    other = pipeline.run({
        "report_text": "Unrelated report.",
        "patient_name": "Luigi", "patient_surname": "Verdi",
        "case_date": "2026-09-20", "diagnostic_question": "Evaluate for aortic root aneurysm",
    })

    assert other["pending_case_routing"]["action"] == "new_case"
