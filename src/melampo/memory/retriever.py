from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass
class MemoryRetriever:
    """Retriever facade over episodic, semantic, vector and graph grounding paths.

    Phase-1/P0 upgrade: when a semantic/vector memory object is provided, this
    retriever uses real in-process memory hits before falling back to the old
    dependency-free contract evidence. This keeps tests deterministic while
    making the pipeline ready for Weaviate-backed adapters.
    """

    memory_store: Any | None = None
    fallback_enabled: bool = True

    def _infer_focus(self, query: str) -> str:
        lowered = query.lower()
        if any(token in lowered for token in ["ct", "mri", "rmn", "tac", "lesion", "nodule", "imaging"]):
            return "visual_diagnostic"
        if any(token in lowered for token in ["cough", "pain", "fatigue", "history", "symptom", "complaint", "fever"]):
            return "language_listening"
        if any(token in lowered for token in ["travel", "smoking", "exposure", "occupation", "prevalence"]):
            return "epidemiology"
        return "multimodal_context"

    def _search_memory(
        self,
        query: str,
        limit: int,
        required_status: Iterable[str] | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self.memory_store is None:
            return []
        filters = filters or {}
        statuses = list(required_status or [])
        store = self.memory_store

        if hasattr(store, "semantic_search"):
            promoted_only = statuses == ["promoted"]
            try:
                hits = store.semantic_search(query=query, limit=limit, promoted_only=promoted_only)
            except TypeError:
                hits = store.semantic_search(query, limit)
            return [hit for hit in hits if isinstance(hit, dict)]

        if hasattr(store, "search"):
            try:
                hits = store.search(query=query, limit=limit, required_status=statuses or None, filters=filters)
            except TypeError:
                hits = store.search(query, limit)
            return [hit for hit in hits if isinstance(hit, dict)]

        if hasattr(store, "search_with_metadata"):
            result = store.search_with_metadata(query=query, top_k=limit, filters=filters)
            hits = result.get("hits", []) if isinstance(result, dict) else []
            return [hit for hit in hits if isinstance(hit, dict)]

        vector_store = getattr(store, "vector_store", None)
        if vector_store is not None and hasattr(vector_store, "search"):
            hits = vector_store.search(query=query, limit=limit, required_status=statuses or None, filters=filters)
            return [hit for hit in hits if isinstance(hit, dict)]
        return []

    def _normalize_hit(self, hit: dict[str, Any], rank: int, focus: str) -> dict[str, Any]:
        evidence = dict(hit)
        evidence.setdefault("source", "vector_memory")
        evidence.setdefault("kind", "object_property_rag_hit")
        evidence.setdefault("route", "post_training_semantic_memory_recall")
        evidence.setdefault("focus", focus)
        evidence.setdefault("rank", rank)
        evidence.setdefault("grounding_score", _safe_float(hit.get("score", hit.get("grounding_score", 0.0))))
        evidence.setdefault("learning_status", hit.get("learning_status", "candidate"))
        evidence.setdefault("provenance", hit.get("provenance", hit.get("metadata", {})))
        evidence["retrieval_backend"] = "semantic_vector_memory"
        return evidence

    def _fallback_evidence(self, query: str, focus: str) -> list[dict[str, Any]]:
        return [
            {
                "source": "semantic_memory",
                "kind": "summary",
                "value": query[:120],
                "route": "semantic_grounding_fallback",
                "focus": focus,
                "grounding_score": 0.58,
                "learning_status": "fallback_contract",
            },
            {
                "source": "episodic_memory",
                "kind": "analogy",
                "value": f"analogy_for:{query[:40]}",
                "route": "case_based_recall_fallback",
                "focus": focus,
                "grounding_score": 0.46,
                "learning_status": "fallback_contract",
            },
            {
                "source": "knowledge_graph",
                "kind": "relation",
                "value": f"kg_link_for:{query[:40]}",
                "route": "graph_grounding_fallback",
                "focus": focus,
                "grounding_score": 0.5,
                "learning_status": "fallback_contract",
            },
        ]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        required_status: Iterable[str] | None = None,
        filters: dict[str, Any] | None = None,
        case_context: dict[str, Any] | None = None,
        target_areas: list[str] | None = None,
    ) -> dict[str, Any]:
        query = query or ""
        focus = self._infer_focus(query)
        target_areas = target_areas or [focus]
        filters = filters or {}
        hits = self._search_memory(query=query, limit=top_k, required_status=required_status, filters=filters)

        if hits:
            evidence = [self._normalize_hit(hit, rank=index + 1, focus=focus) for index, hit in enumerate(hits[:top_k])]
            mean_grounding = sum(_safe_float(item.get("grounding_score", 0.0)) for item in evidence) / max(len(evidence), 1)
            coverage = _clamp(len(evidence) / max(top_k, 1))
            return {
                "query": query,
                "focus": focus,
                "target_areas": target_areas,
                "status": "grounded_retrieval_ready",
                "retrieval_mode": "semantic_vector_memory",
                "evidence": evidence,
                "evidence_count": len(evidence),
                "retrieval_quality": {
                    "memory_backed": True,
                    "coverage": coverage,
                    "coverage_basis": "topk_ratio",
                    "mean_grounding_score": round(mean_grounding, 3),
                    "fallback_used": False,
                    "filters": filters,
                    "required_status": list(required_status or []),
                },
                "case_context_keys": sorted((case_context or {}).keys()),
            }

        if not self.fallback_enabled:
            return {
                "query": query,
                "focus": focus,
                "target_areas": target_areas,
                "status": "no_retrieval_evidence",
                "retrieval_mode": "empty_memory_no_fallback",
                "evidence": [],
                "evidence_count": 0,
                "retrieval_quality": {
                    "memory_backed": False,
                    "coverage": 0.0,
                    "coverage_basis": "not_applicable",
                    "mean_grounding_score": 0.0,
                    "fallback_used": False,
                    "filters": filters,
                    "required_status": list(required_status or []),
                },
                "case_context_keys": sorted((case_context or {}).keys()),
            }

        evidence = self._fallback_evidence(query=query, focus=focus)
        return {
            "query": query,
            "focus": focus,
            "target_areas": target_areas,
            "status": "grounded_retrieval_ready",
            "retrieval_mode": "fallback_contract_only",
            "evidence": evidence,
            "evidence_count": len(evidence),
            "retrieval_quality": {
                "memory_backed": False,
                "coverage": 0.0,
                "coverage_basis": "not_applicable",
                "mean_grounding_score": round(sum(item["grounding_score"] for item in evidence) / max(len(evidence), 1), 3),
                "fallback_used": True,
                "filters": filters,
                "required_status": list(required_status or []),
            },
            "case_context_keys": sorted((case_context or {}).keys()),
        }
