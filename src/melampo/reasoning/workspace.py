from dataclasses import dataclass, field


@dataclass
class DifferentialWorkspace:
    items: list = field(default_factory=list)

    def push(self, item: dict) -> None:
        self.items.append(item)

    def snapshot(self) -> list:
        return list(self.items)
