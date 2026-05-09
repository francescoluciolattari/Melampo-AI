from melampo.models.model_card import default_phase4a_model_cards
from melampo.models.model_client import ModelClientConfig, SafeModelClient
from melampo.models.model_response_schema import ClinicalClaim, SpecialistModelResponse
from melampo.models.specialist_adapters import ClaudeCritiqueAdapter, Gemma4ClinicalReasoningAdapter, Pillar0RadiologyAdapter
from melampo.orchestration.model_execution_trace import ModelExecutionTrace


def test_safe_model_client_disabled_and_dry_run_do_not_call_network():
    disabled = SafeModelClient(
        provider="provider",
        model_name="model",
        role="role",
        config=ModelClientConfig(mode="disabled", enabled=False),
    )
    disabled_result = disabled.execute({"case_id": "case-1"})
    assert disabled_result["status"] == "not_called"
    assert disabled.trace.summary()["record_count"] == 1

    dry_run = SafeModelClient(
        provider="provider",
        model_name="model",
        role="role",
        config=ModelClientConfig(mode="dry_run", enabled=True, endpoint="https://example.invalid", allow_remote=False),
    )
    dry_result = dry_run.execute({"case_id": "case-1"})
    assert dry_result["status"] == "request_prepared"
    assert dry_result["hidden_network_call"] is False
    assert dry_run.trace.dump()[0]["hidden_network_call"] is False


def test_safe_model_client_blocks_http_without_remote_allowance():
    client = SafeModelClient(
        provider="provider",
        model_name="model",
        role="role",
        config=ModelClientConfig(mode="http_json", enabled=True, endpoint="https://example.invalid", allow_remote=False),
    )
    result = client.execute({"case_id": "case-1"})
    assert result["status"] == "blocked"
    assert result["reason"] == "remote_execution_not_allowed_or_endpoint_missing"


def test_specialist_response_schema_preserves_claims_and_area_signal():
    claim = ClinicalClaim(
        claim_id="claim-1",
        type="finding",
        normalized_entity="pulmonary opacity",
        polarity="present",
        confidence=0.8,
        uncertainty=0.2,
        ontology_refs=["SNOMED:placeholder"],
        evidence_refs=["doc:1"],
        source_area="visual_diagnostic",
    )
    response = SpecialistModelResponse(
        provider="provider",
        model_name="model",
        role="role",
        status="completed",
        confidence=0.8,
        uncertainty=0.2,
        claims=[claim.as_dict()],
    )
    signal = response.as_area_signal("visual_diagnostic")
    assert signal["claims"][0]["normalized_entity"] == "pulmonary opacity"
    assert signal["signal_count"] == 1
    assert signal["salience_score"] == 0.8


def test_pillar0_adapter_mock_returns_visual_claim_and_trace():
    adapter = Pillar0RadiologyAdapter(
        enabled=True,
        execution_mode="mock",
        client_config=ModelClientConfig(
            mode="mock",
            enabled=True,
            mock_payload={
                "status": "completed",
                "signals": {"primary_finding": "pulmonary opacity", "anatomical_region": "lung"},
                "confidence": 0.74,
                "uncertainty": 0.26,
                "ontology_refs": ["SNOMED:placeholder"],
            },
        ),
    )
    response = adapter.infer_volume("study-1", ["series-1.dcm"], {"modality": "CT"})
    assert response.status == "completed"
    assert response.confidence == 0.74
    assert response.claims[0]["source_area"] == "visual_diagnostic"
    assert response.as_area_signal("visual_diagnostic")["model_name"] == "Pillar-0"


def test_gemma4_adapter_mock_requires_grounding_and_returns_claim():
    adapter = Gemma4ClinicalReasoningAdapter(
        enabled=True,
        execution_mode="mock",
        client_config=ModelClientConfig(
            mode="mock",
            enabled=True,
            mock_payload={
                "status": "completed",
                "grounded_summary": "Cough and fever are grounded by retrieved literature.",
                "source_refs": ["doc:guideline:1"],
                "confidence": 0.67,
                "uncertainty": 0.33,
                "missing_evidence": ["oxygen saturation"],
            },
        ),
    )
    response = adapter.reason_over_text("case-1", "cough fever", {"hits": [{"record_id": "doc:guideline:1"}]})
    assert response.status == "completed"
    assert response.claims[0]["source_area"] == "language_listening"
    assert response.missing_evidence == ["oxygen saturation"]
    assert response.provenance["request"]["governance"]["must_be_grounded_by_rag"] is True


def test_claude_critic_adapter_mock_is_critic_only():
    adapter = ClaudeCritiqueAdapter(
        enabled=True,
        execution_mode="mock",
        client_config=ModelClientConfig(
            mode="mock",
            enabled=True,
            mock_payload={
                "status": "completed",
                "critique_status": "needs_revision",
                "unsupported_claims": ["unsupported finality"],
                "safety_flags": ["research_only"],
                "recommended_action": "escalate_for_human_review",
                "confidence_in_critique": 0.71,
            },
        ),
    )
    response = adapter.critique({"result_label": "candidate_a"}, {"hits": []})
    assert response.status == "completed"
    assert response.signals["critique_status"] == "needs_revision"
    assert "not_final_diagnostic_arbiter" in response.limitations


def test_model_cards_are_available_for_phase4a():
    cards = default_phase4a_model_cards()
    names = {card.name for card in cards}
    assert {"Pillar-0", "Gemma 4", "Claude Healthcare/Life Sciences"}.issubset(names)
    assert "not validated" in cards[0].safety_boundary.lower()
    assert "Validation requirements" in cards[0].to_markdown()


def test_model_execution_trace_records_lifecycle():
    trace = ModelExecutionTrace()
    record = trace.start(
        provider="provider",
        model_name="model",
        role="role",
        mode="mock",
        request_id="req-1",
        hidden_network_call=False,
    )
    record.finish("completed")
    summary = trace.summary()
    assert summary["record_count"] == 1
    assert summary["statuses"]["completed"] == 1
