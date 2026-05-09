from pathlib import Path

from melampo.data.document_processing import ClinicalDocumentProcessor
from melampo.evaluation.rag_evaluation import RAGEvaluationRecord, RAGEvaluator
from melampo.memory.retriever import MemoryRetriever
from melampo.memory.weaviate_adapter import WeaviateEnterpriseMemoryAdapter


def test_phase2_document_processor_preserves_governed_clinical_metadata(tmp_path: Path):
    path = tmp_path / "guideline.txt"
    path.write_text(
        "# Differential diagnosis\n"
        "Pneumonia with cough, fever, and focal opacity requires imaging correlation.\n"
        "Contact doctor@example.com on 01/02/2026.\n"
        "# Epidemiology\n"
        "Smoking and exposure history change pre-test probability.",
        encoding="utf-8",
    )
    processor = ClinicalDocumentProcessor(chunk_size=140, chunk_overlap=20)
    result = processor.process_document(
        path,
        metadata={
            "source_type": "guideline",
            "license": "research_use",
            "publication_date": "2026-01-01",
            "source_uri": "local://guideline",
        },
        prefer_docling=False,
    )

    assert result["status"] == "completed"
    assert result["document_id"].startswith("doc:")
    assert result["chunk_count"] >= 2
    first = result["documents"][0]
    assert first["metadata"]["section"] in {"Differential diagnosis", "Epidemiology"}
    assert "Pathology:Pneumonia" in {ref for doc in result["documents"] for ref in doc["metadata"]["ontology_refs"]}
    assert any(doc["metadata"]["redacted"] for doc in result["documents"])
    assert result["enterprise_metadata"]["governance_status"] == "complete"


def test_phase2_weaviate_enterprise_adapter_upserts_searches_and_expands_graph(tmp_path: Path):
    path = tmp_path / "guideline.txt"
    path.write_text("Pneumonia cough fever opacity differential diagnosis with source provenance.", encoding="utf-8")
    processor = ClinicalDocumentProcessor(chunk_size=90, chunk_overlap=10)
    processed = processor.process_document(
        path,
        metadata={"source_type": "guideline", "license": "research_use", "publication_date": "2026-01-01"},
        prefer_docling=False,
    )
    adapter = WeaviateEnterpriseMemoryAdapter()
    materialized = adapter.materialize_schema()
    upsert = processor.upsert_processed_document(processed, adapter)
    search = adapter.hybrid_search("pneumonia opacity", class_name="ClinicalDocument", limit=3)

    assert materialized["status"] == "materialized_in_local_contract"
    assert upsert["stored"] == processed["chunk_count"]
    assert search["status"] == "completed"
    assert search["hits"]
    assert search["hits"][0]["source"] == "weaviate"
    assert search["hits"][0]["score_final"] >= 0.0

    object_key = search["hits"][0]["metadata"]["record_id"]
    expanded = adapter.graph_expand(object_key, depth=1)
    assert expanded["status"] == "completed"
    assert expanded["object_count"] >= 1


def test_phase2_memory_retriever_uses_enterprise_weaviate_adapter(tmp_path: Path):
    path = tmp_path / "guideline.txt"
    path.write_text("Pneumonia cough fever opacity should be grounded by clinical document provenance.", encoding="utf-8")
    processor = ClinicalDocumentProcessor(chunk_size=120, chunk_overlap=10)
    processed = processor.process_document(path, metadata={"source_type": "guideline"}, prefer_docling=False)
    adapter = WeaviateEnterpriseMemoryAdapter()
    processor.upsert_processed_document(processed, adapter)

    result = MemoryRetriever(memory_store=adapter, fallback_enabled=False).retrieve("cough opacity pneumonia", top_k=2)

    assert result["status"] == "grounded_retrieval_ready"
    assert result["retrieval_mode"] == "semantic_vector_memory"
    assert result["retrieval_quality"]["memory_backed"] is True
    assert result["evidence"][0]["source"] in {"weaviate", "vector_memory"}
    assert result["evidence"][0]["provenance"]


def test_phase2_rag_evaluator_reports_enterprise_metrics():
    evidence = [
        {
            "source": "weaviate",
            "text": "Pneumonia cough fever opacity differential diagnosis.",
            "grounding_score": 0.9,
            "record_id": "ClinicalDocument:doc1",
            "metadata": {"section": "Differential diagnosis", "source_path": "guideline.txt", "ontology_refs": ["Pathology:Pneumonia"]},
        },
        {
            "source": "weaviate",
            "text": "Unrelated administrative note.",
            "grounding_score": 0.2,
            "record_id": "ClinicalDocument:doc2",
            "metadata": {"section": "Admin", "source_path": "note.txt"},
        },
    ]
    report = RAGEvaluator().evaluate([
        RAGEvaluationRecord(
            query="pneumonia cough opacity",
            evidence=evidence,
            answer="Pneumonia is supported by cough fever opacity.",
            expected_terms=["pneumonia", "opacity"],
            required_sources=["weaviate"],
        )
    ])
    thresholds = RAGEvaluator.enterprise_thresholds(report, min_context_precision=0.4, min_provenance=0.8)

    assert report.sample_count == 1
    assert report.context_precision >= 0.5
    assert report.provenance_completeness == 1.0
    assert thresholds["status"] in {"pass", "needs_review"}
