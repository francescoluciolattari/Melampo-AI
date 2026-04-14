from dataclasses import dataclass, field


@dataclass
class DecisionTrace:
    """Minimal auditable trace for reasoning outcomes."""

    steps: list = field(default_factory=list)

    def add(self, item: str) -> None:
        self.steps.append(item)

    def dump(self) -> list:
        return list(self.steps)
