from dataclasses import dataclass


@dataclass
class EvidenceRanker:
    """Evidence ranker producing grounded support ordering from source and metadata."""

    SOURCE_PRIORITY = {
        "weaviate": 0.96,
        "vector_memory": 0.92,
        "semantic_memory": 0.9,
        "clinical_document_processor": 0.85,
        "knowledge_graph": 0.8,
        "retrieval": 0.75,
        "bundle": 0.72,
        "episodic_memory": 0.7,
        "fusion": 0.68,
        "service": 0.6,
    }

    KIND_PRIORITY = {
        "object_property_rag_hit": 0.86,
        "clinical_document_chunk": 0.84,
        "summary": 0.8,
        "grounded": 0.78,
        "relation": 0.75,
        "candidate": 0.7,
        "analogy": 0.65,
        "bundle_keys": 0.55,
        "engine": 0.5,
        "provider": 0.45,
    }

    def rank(self, items: list) -> list:
        scored = []
        for item in items:
            source = item.get("source", "unknown") if isinstance(item, dict) else "unknown"
            kind = item.get("kind", "signal") if isinstance(item, dict) else "signal"
            grounding_score = float(item.get("grounding_score", 0.5)) if isinstance(item, dict) else 0.5
            provenance_bonus = 0.05 if isinstance(item, dict) and item.get("provenance") else 0.0
            relation_bonus = 0.04 if isinstance(item, dict) and item.get("relations") else 0.0
            promoted_bonus = 0.05 if isinstance(item, dict) and item.get("learning_status") == "promoted" else 0.0
            source_weight = self.SOURCE_PRIORITY.get(source, 0.4)
            kind_weight = self.KIND_PRIORITY.get(kind, 0.4)
            weight = round(grounding_score + source_weight + kind_weight + provenance_bonus + relation_bonus + promoted_bonus, 3)
            scored.append({"item": item, "weight": weight})

        scored.sort(key=lambda entry: entry["weight"], reverse=True)
        ranked = []
        for index, entry in enumerate(scored):
            ranked.append({"rank": index + 1, "item": entry["item"], "weight": entry["weight"]})
        return ranked
