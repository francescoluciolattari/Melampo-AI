from pathlib import Path

from melampo.ingestion.text_document_ingestion import TextDocumentIngestor
from melampo.memory.qdrant_adapter import QdrantVectorMemoryAdapter


def test_qdrant_adapter_prepares_hybrid_contract_and_uses_fallback():
    adapter = QdrantVectorMemoryAdapter()
    schema = adapter.collection_schema()
    assert "vectors" in schema
    assert "sparse_vectors" in schema

    stored = adapter.upsert_text(
        record_id="guideline-1",
        text="pneumonia cough fever opacity differential diagnosis",
        metadata={"document_type": "guideline"},
        source="test_guideline",
        learning_status="candidate",
    )
    assert stored["status"] == "stored_in_fallback_qdrant_contract_prepared"
    query = adapter.build_hybrid_query("cough opacity", limit=3)
    assert query["query"]["fusion"] == "reciprocal_rank_fusion_candidate"
    hits = adapter.search("pneumonia opacity", limit=1)
    assert hits
    assert hits[0]["record_id"] == "guideline-1"


def test_text_document_ingestor_chunks_and_indexes_text(tmp_path):
    document_path = tmp_path / "guideline.txt"
    document_path.write_text(
        "Pneumonia differential diagnosis includes cough, fever, focal opacity, exposure context, and clinical risk stratification. " * 8,
        encoding="utf-8",
    )
    ingestor = TextDocumentIngestor(chunk_size=180, chunk_overlap=30)
    result = ingestor.ingest_path(document_path, source="synthetic_guideline", document_type="guideline_text")
    assert result["status"] == "ingested"
    assert result["chunk_count"] >= 2
    assert result["vector_memory"]["record_count"] == result["chunk_count"]
    assert result["chunks"][0]["metadata"]["source"] == "synthetic_guideline"
