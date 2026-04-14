from dataclasses import dataclass, field


@dataclass
class EpisodicMemoryStore:
    cases: list = field(default_factory=list)

    def add_case(self, item: dict) -> None:
        self.cases.append(item)

    def retrieve(self, limit: int = 5) -> list:
        return self.cases[:limit]
