from dataclasses import dataclass


@dataclass
class KnowledgeGraphClient:
    config: object

    def lookup(self, query: str) -> dict:
        return {"provider": "api_for_service_knowledge_graph", "query": query}
