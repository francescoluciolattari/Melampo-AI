from dataclasses import dataclass, field


@dataclass
class SemanticMemoryStore:
    documents: list = field(default_factory=list)

    def add_document(self, item: dict) -> None:
        self.documents.append(item)

    def query(self, limit: int = 5) -> list:
        return self.documents[:limit]
