from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelExecutionRecord:
    provider: str
    model_name: str
    role: str
    mode: str
    status: str
    hidden_network_call: bool
    request_id: str
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    latency_ms: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str, error: str | None = None) -> "ModelExecutionRecord":
        self.finished_at = time.time()
        self.latency_ms = round((self.finished_at - self.started_at) * 1000.0, 3)
        self.status = status
        self.error = error
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "role": self.role,
            "mode": self.mode,
            "status": self.status,
            "hidden_network_call": self.hidden_network_call,
            "request_id": self.request_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ModelExecutionTrace:
    records: list[ModelExecutionRecord] = field(default_factory=list)

    def start(
        self,
        *,
        provider: str,
        model_name: str,
        role: str,
        mode: str,
        request_id: str,
        hidden_network_call: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ModelExecutionRecord:
        record = ModelExecutionRecord(
            provider=provider,
            model_name=model_name,
            role=role,
            mode=mode,
            status="started",
            hidden_network_call=hidden_network_call,
            request_id=request_id,
            metadata=metadata or {},
        )
        self.records.append(record)
        return record

    def dump(self) -> list[dict[str, Any]]:
        return [record.as_dict() for record in self.records]

    def summary(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for record in self.records:
            statuses[record.status] = statuses.get(record.status, 0) + 1
        return {"record_count": len(self.records), "statuses": statuses, "records": self.dump()}
