from dataclasses import dataclass


@dataclass
class MemoryRetriever:
    """Minimal retriever facade over episodic and semantic memory."""

    def retrieve(self, query: str) -> dict:
        return {"query": query, "status": "retriever_placeholder"}
