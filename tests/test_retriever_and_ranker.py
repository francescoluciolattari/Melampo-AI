from melampo.memory.retriever import MemoryRetriever
from melampo.models.evidence_ranker import EvidenceRanker


def test_retriever_infers_focus_and_ranker_orders_grounded_items():
    retriever = MemoryRetriever()
    result = retriever.retrieve("CT shows pulmonary lesion in a smoker")
    assert result["status"] == "grounded_retrieval_ready"
    assert result["focus"] == "visual_diagnostic"
    assert result["evidence_count"] == 3
    assert all("grounding_score" in item for item in result["evidence"])

    ranked = EvidenceRanker().rank(result["evidence"])
    assert ranked[0]["rank"] == 1
    assert ranked[0]["weight"] >= ranked[-1]["weight"]
    assert ranked[0]["item"]["source"] in ["semantic_memory", "knowledge_graph", "episodic_memory"]
