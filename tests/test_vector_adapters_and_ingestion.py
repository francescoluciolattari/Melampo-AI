
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
