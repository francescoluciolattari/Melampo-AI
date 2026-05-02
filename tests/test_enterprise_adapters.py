from pathlib import Path

from melampo.cli import main_decision_record, main_weaviate_schema
from melampo.data.document_processing import ClinicalDocumentProcessor
from melampo.memory.weaviate_adapter import WeaviateSemanticMemoryAdapter
from melampo.models.specialist_adapters import ClaudeCritiqueAdapter, Gemma4ClinicalReasoningAdapter, Pillar0RadiologyAdapter


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
    result = processor.process_document(path, metadata={"source_type": "unit_test"}, prefer_docling=False)

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
