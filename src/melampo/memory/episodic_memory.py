from dataclasses import dataclass, field
from threading import RLock


@dataclass
class EpisodicMemoryStore:
    cases: list = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def add_case(self, item: dict) -> None:
        with self._lock:
            self.cases.append(item)

    def retrieve(self, limit: int = 5) -> list:
        with self._lock:
            return list(self.cases[:limit])
