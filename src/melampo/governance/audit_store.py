from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any


@dataclass(slots=True)
class AppendOnlyAuditStore:
    """Small append-only JSONL audit store for durable research traces.

    The store provides local durability without introducing external services.
    It is intentionally minimal and should be replaced by an immutable, access
    controlled audit ledger in regulated deployments.
    """

    path: str | Path
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def append(self, event_type: str, payload: dict[str, Any], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        event = {
            "event_type": event_type,
            "payload": payload,
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        return event

    def read_all(self) -> list[dict[str, Any]]:
        path = Path(self.path)
        if not path.exists():
            return []
        with self._lock:
            lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def describe(self) -> dict[str, Any]:
        events = self.read_all()
        event_counts: dict[str, int] = {}
        for event in events:
            event_type = str(event.get("event_type", "unknown"))
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        return {"path": str(self.path), "event_count": len(events), "event_counts": event_counts}
