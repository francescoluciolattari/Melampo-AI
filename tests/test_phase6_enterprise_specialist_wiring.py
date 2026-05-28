from melampo.areas.language_listening_area import LanguageListeningArea
from melampo.areas.visual_diagnostic_area import VisualDiagnosticArea
from melampo.orchestration.specialist_runtime import SpecialistRuntime
from melampo.reasoning.diagnostic_result import DiagnosticResult, DreamSummary, IntuitionSummary, MelampoMetrics
from melampo.reasoning.area_coherence import AreaCoherenceAnalyzer


def test_specialist_runtime_is_safe_by_default():
    runtime = SpecialistRuntime()

    radiology = runtime.radiology_signal(
        study_id="study-1",
        series_paths=["/tmp/nonexistent"],
        metadata={"modality": "CT"},
    )

    assert radiology["external_model_is_final_arbiter"] is False
    assert radiology["response"]["status"] == "not_called"
    assert radiology["area_signal"]["uncertainty_score"] == 1.0
    assert radiology["hidden_network_call"] is False

    text = runtime.grounded_text_signal(
        case_id="case-1",
        text="Patient reports cough and fever.",
        grounding={"retrieval": {"evidence_count": 0}},
    )

    assert text["external_model_is_final_arbiter"] is False
    assert text["response"]["status"] == "not_called"
    assert text["area_signal"]["uncertainty_score"] == 1.0
    assert text["hidden_network_call"] is False


def test_visual_area_accepts_specialist_signal_without_granting_authority():
    area = VisualDiagnosticArea()
    signal = area.integrate(
        volume_features={"study_id": "study-1"},
        pathology_features={},
        specialist_signal={
            "area_signal": {
                "claims": [{"claim_id": "c1", "normalized_entity": "opacity"}],
                "salience_score": 0.7,
                "uncertainty_score": 0.3,
            }
        },
    )

    assert signal["area"] == "visual_diagnostic"
    assert signal["claims"][0]["normalized_entity"] == "opacity"
    assert signal["governance"]["specialist_models_are_signal_providers_only"] is True
    assert signal["governance"]["final_authority"] == "MelampoDiagnosticOrchestrator"


def test_language_area_accepts_grounded_specialist_signal():
    area = LanguageListeningArea()
    signal = area.integrate(
        report_text="No pleural effusion.",
        patient_complaints="Chest pain.",
        specialist_signal={
            "area_signal": {
                "claims": [{"claim_id": "g1", "normalized_entity": "chest pain"}],
                "missing_evidence": ["troponin"],
                "salience_score": 0.6,
                "uncertainty_score": 0.4,
            }
        },
    )

    assert signal["area"] == "language_listening"
    assert signal["claims"][0]["normalized_entity"] == "chest pain"
    assert "troponin" in signal["missing_evidence"]
    assert signal["governance"]["language_model_must_be_grounded"] is True


def test_diagnostic_result_serializes_enterprise_contract():
    result = DiagnosticResult(
        case_id="case-1",
        result_label="abstain_or_escalate",
        top_hypothesis={"label": "none", "score": 0.0},
        differential=[],
        intuition=IntuitionSummary(),
        melampo_metrics=MelampoMetrics(pi_score=0.2, mismatch_index=0.9, action_potential_gate=0.1),
        support={},
        policy={"abstain": True, "reasons": ["test"]},
        critique={},
        dream=DreamSummary(),
        model_capability_decision_record={"strategy": "test"},
        audit_trace={},
    ).as_dict()

    assert result["schema_version"] == "diagnostic_result.v1"
    assert result["audit_trace"]["final_authority"] == "MelampoDiagnosticOrchestrator"
    assert result["audit_trace"]["external_models_are_not_final_arbiters"] is True
    assert "not a validated medical device" in result["audit_trace"]["clinical_warning"]
    assert result["dream"]["promotion_policy"]["automatic_clinical_promotion_allowed"] is False


def test_neuro_metrics_include_deep_inference_and_action_potential_gate():
    dynamics = AreaCoherenceAnalyzer().analyze(
        {
            "visual_diagnostic": {"salience_score": 0.8, "signal_count": 3, "claims": [{"normalized_entity": "opacity"}]},
            "language_listening": {"salience_score": 0.6, "signal_count": 2, "claims": [{"normalized_entity": "opacity"}]},
            "case_context": {"salience_score": 0.3, "signal_count": 1},
        }
    )
    neuro = dynamics["neuro_dynamic_metrics"]

    assert 0.0 <= neuro["action_potential_gate"] <= 1.0
    assert 0.0 <= neuro["deep_inference_score"] <= 1.0
    assert 0.0 <= neuro["noise_suppression_score"] <= 1.0
    assert dynamics["deep_inference_score"] == neuro["deep_inference_score"]
