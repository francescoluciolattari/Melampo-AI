from dataclasses import dataclass


@dataclass
class RetrievalAdapter:
    provider: str = "api_for_service_document_rag"

    def retrieve(self, query: str) -> dict:
        return {"provider": self.provider, "query": query}
