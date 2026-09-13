from pathlib import Path

from melampo.cli import main_decision_record, main_weaviate_schema
from melampo.data.document_processing import ClinicalDocumentProcessor
from melampo.memory.weaviate_adapter import WeaviateSemanticMemoryAdapter
from melampo.models.specialist_adapters import (
    ClaudeCritiqueAdapter,
    Gemma4ClinicalReasoningAdapter,
    Pillar0RadiologyAdapter,
)


def test_specialist_adapters_are_safe_by_default():
    pillar = Pillar0RadiologyAdapter()
    gemma = Gemma4ClinicalReasoningAdapter()
    claude = ClaudeCritiqueAdapter()

    pillar_response = pillar.infer_volume("study-1", ["/tmp/nonexistent.dcm"], {"modality": "CT"})
    gemma_response = gemma.reason_over_text("case-1", "Patient reports cough and fever.", {"hits": []})
    claude_response = claude.critique({"result_label": "abstain_or_escalate"})

    assert pillar_response.status == "not_called"
    assert gemma_response.status == "not_called"
    assert claude_response.status == "not_called"
    assert pillar_response.as_area_signal("visual_diagnostic")["area"] == "visual_diagnostic"
    assert gemma_response.as_area_signal("language_listening")["limitations"]


def test_weaviate_adapter_prepares_schema_without_network_calls():
    adapter = WeaviateSemanticMemoryAdapter()
    prepared = adapter.prepare_schema_materialization()
    search = adapter.semantic_search("ClinicalCase", "cough fever", limit=3)
    rejected = adapter.prepare_upsert("UnknownClass", "id-1", {})

    assert prepared["status"] == "prepared"
    assert prepared["governance"]["hidden_network_call"] is False
    assert search["status"] == "not_executed"
    assert search["hits"] == []
    assert rejected["status"] == "rejected"


def test_document_processor_plain_text_fallback(tmp_path: Path):
    path = tmp_path / "guideline.txt"
    path.write_text("Fever and cough can support infectious differential reasoning." * 5, encoding="utf-8")
    processor = ClinicalDocumentProcessor(chunk_size=80, chunk_overlap=10)
    result = processor.process_document(path, metadata={"source_type": "unit_test"}, prefer_structured_parser=False)

    assert result["status"] == "completed"
    assert result["parser"] == "plain_text_fallback"
    assert result["chunk_count"] >= 1
    assert result["documents"][0]["learning_status"] == "candidate"


def test_enterprise_cli_helpers_write_json(tmp_path: Path):
    decision_path = tmp_path / "decision.json"
    schema_path = tmp_path / "schema.json"

    assert main_decision_record(["--output", str(decision_path)]) == 0
    assert main_weaviate_schema(["--output", str(schema_path)]) == 0
    assert "MelampoDiagnosticOrchestrator" in decision_path.read_text(encoding="utf-8")
    assert "Weaviate" in schema_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Docling removed: Nemotron-Parse default, LlamaParse as an optional
# cloud-only cross-check, graceful degradation preserved throughout
# --------------------------------------------------------------------------


def test_no_docling_reference_remains_in_the_public_contract(tmp_path):
    processor = ClinicalDocumentProcessor()
    description = processor.describe()
    assert "docling" not in str(description).lower()
    assert description["recommended_parser"] == "Nemotron-Parse"


def test_nemotron_parse_is_reported_unavailable_without_configuration():
    processor = ClinicalDocumentProcessor()
    assert processor._nemotron_parse_available()["available"] is False


def test_nemotron_parse_is_reported_available_once_configured():
    processor = ClinicalDocumentProcessor(nemotron_parse_endpoint="https://example", nemotron_parse_api_key="k")
    assert processor._nemotron_parse_available()["available"] is True


def test_llamaparse_is_off_by_default_and_cloud_only(tmp_path):
    """LlamaParse's own documentation offers VPC as its closest on-premise
    equivalent, not true on-premise -- it must never be the sole parser."""
    processor = ClinicalDocumentProcessor()
    assert processor._llamaparse_available()["available"] is False
    description = processor.describe()
    assert "no true on-premise" in description["cross_check_parser"].lower() or "vpc" in description["cross_check_parser"].lower()


def test_processing_falls_back_to_plain_text_when_nemotron_parse_is_unconfigured(tmp_path):
    """The graceful-degradation contract Docling held is preserved under the
    new name -- callers already handling this status need no change."""
    path = tmp_path / "referto.txt"
    path.write_text("The patient presents with fever and cough.")
    processor = ClinicalDocumentProcessor()

    result = processor.process_document(path)

    assert result["status"] == "completed"
    assert result["parser"] == "plain_text_fallback"
    assert result["parser_result"]["reason"] == "nemotron_parse_unavailable"


def test_cross_check_is_off_by_default(tmp_path):
    """also_cross_check_with_llamaparse defaults to False so an on-premise-only
    deployment (Nemotron-Parse configured, LlamaParse not) does not report a
    spurious unavailable cross-check on every call."""
    path = tmp_path / "referto.txt"
    path.write_text("text")
    result = ClinicalDocumentProcessor().process_document(path)
    assert "cross_check" not in result


def test_cross_check_result_is_attached_when_requested(tmp_path):
    path = tmp_path / "referto.txt"
    path.write_text("text")
    result = ClinicalDocumentProcessor().process_document(path, also_cross_check_with_llamaparse=True)
    assert "cross_check" in result
    assert result["cross_check"]["status"] == "not_executed"


def test_the_capability_registry_lists_nemotron_parse_and_llamaparse_not_docling():
    from melampo.orchestration.model_capability_registry import ModelCapabilityRegistry

    registry = ModelCapabilityRegistry.build_default()
    names = set(registry.capabilities)
    assert "Docling" not in names
    assert "Nemotron-Parse" in names
    assert "LlamaParse" in names


def test_llamaparse_capability_entry_documents_the_on_premise_limitation():
    from melampo.orchestration.model_capability_registry import ModelCapabilityRegistry

    registry = ModelCapabilityRegistry.build_default()
    limitations = registry.capabilities["LlamaParse"].limitations
    assert any("on_premise" in item for item in limitations)
