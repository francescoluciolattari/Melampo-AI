from dataclasses import dataclass


@dataclass
class EvidenceRanker:
    """Evidence ranker producing grounded support ordering from source and metadata."""

    SOURCE_PRIORITY = {
        "semantic_memory": 0.9,
        "knowledge_graph": 0.8,
        "episodic_memory": 0.7,
        "retrieval": 0.75,
        "bundle": 0.72,
        "fusion": 0.68,
        "service": 0.6,
    }

    KIND_PRIORITY = {
        "summary": 0.8,
        "relation": 0.75,
        "analogy": 0.65,
        "grounded": 0.78,
        "bundle_keys": 0.55,
        "engine": 0.5,
        "provider": 0.45,
        "candidate": 0.7,
    }

    def rank(self, items: list) -> list:
        scored = []
        for item in items:
            source = item.get("source", "unknown") if isinstance(item, dict) else "unknown"
            kind = item.get("kind", "signal") if isinstance(item, dict) else "signal"
            grounding_score = float(item.get("grounding_score", 0.5)) if isinstance(item, dict) else 0.5
            source_weight = self.SOURCE_PRIORITY.get(source, 0.4)
            kind_weight = self.KIND_PRIORITY.get(kind, 0.4)
            weight = round(grounding_score + source_weight + kind_weight, 3)
            scored.append({"item": item, "weight": weight})

        scored.sort(key=lambda entry: entry["weight"], reverse=True)
        ranked = []
        for index, entry in enumerate(scored):
            ranked.append({"rank": index + 1, "item": entry["item"], "weight": entry["weight"]})
        return ranked
