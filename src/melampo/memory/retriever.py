from dataclasses import dataclass


@dataclass
class MemoryRetriever:
    """Retriever facade over episodic, semantic, and graph-oriented grounding paths."""

    def _infer_focus(self, query: str) -> str:
        lowered = query.lower()
        if any(token in lowered for token in ["ct", "mri", "rmn", "tac", "lesion", "nodule", "imaging"]):
            return "visual_diagnostic"
        if any(token in lowered for token in ["cough", "pain", "fatigue", "history", "symptom", "complaint"]):
            return "language_listening"
        if any(token in lowered for token in ["travel", "smoking", "exposure", "occupation", "prevalence"]):
            return "epidemiology"
        return "multimodal_context"

    def retrieve(self, query: str) -> dict:
        query = query or ""
        focus = self._infer_focus(query)
        evidence = [
            {
                "source": "semantic_memory",
                "kind": "summary",
                "value": query[:120],
                "route": "semantic_grounding",
                "focus": focus,
                "grounding_score": 0.78,
            },
            {
                "source": "episodic_memory",
                "kind": "analogy",
                "value": f"analogy_for:{query[:40]}",
                "route": "case_based_recall",
                "focus": focus,
                "grounding_score": 0.64,
            },
            {
                "source": "knowledge_graph",
                "kind": "relation",
                "value": f"kg_link_for:{query[:40]}",
                "route": "graph_grounding",
                "focus": focus,
                "grounding_score": 0.7,
            },
        ]
        return {
            "query": query,
            "focus": focus,
            "status": "grounded_retrieval_ready",
            "evidence": evidence,
            "evidence_count": len(evidence),
        }
