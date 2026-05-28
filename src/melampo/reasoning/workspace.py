from dataclasses import dataclass, field
from threading import RLock


@dataclass
class DifferentialWorkspace:
    items: list = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def push(self, item: dict) -> None:
        with self._lock:
            self.items.append(item)

    def snapshot(self) -> list:
        with self._lock:
            return list(self.items)
