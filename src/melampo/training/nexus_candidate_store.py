from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from ..memory.learning_status import (
    normalize_learning_status,
    validate_learning_transition,
)


def _stable_candidate_id(case_id: str, payload: dict[str, Any], created_at: float) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(f"nexus:{case_id}:{canonical}:{created_at}".encode()).hexdigest()
    return f"nexus:{case_id}:{digest[:16]}"


@dataclass(slots=True)
class NexusCandidateRecord:
    candidate_id: str
    case_id: str
    created_at: float
    payload: dict[str, Any]
    source: str = "nexus_branch"
    learning_status: str = "candidate"
    validation: dict[str, Any] | None = None
    promotion_decision: dict[str, Any] | None = None
    outcome_feedback: list[dict[str, Any]] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "case_id": self.case_id,
            "created_at": self.created_at,
            "payload": self.payload,
            "source": self.source,
            "learning_status": self.learning_status,
            "validation": self.validation,
            "promotion_decision": self.promotion_decision,
            "outcome_feedback": self.outcome_feedback,
            "audit": self.audit,
        }

    def to_memory_document(self) -> dict[str, Any]:
        text = str(self.payload.get("text") or self.payload.get("summary") or json.dumps(self.payload, sort_keys=True, default=str))
        metadata = dict(self.payload.get("metadata", {})) if isinstance(self.payload.get("metadata", {}), dict) else {}
        return {
            "text": text,
            "source": self.source,
            "learning_status": self.learning_status,
            "modality": "nexus_replay_candidate",
            "metadata": {
                **metadata,
                "record_id": self.candidate_id,
                "candidate_id": self.candidate_id,
                "case_id": self.case_id,
                "memory_role": "governed_nexus_replay_candidate",
                "synthetic": True,
                "validation_status": (self.validation or {}).get("status", "not_validated"),
            },
        }


@dataclass(slots=True)
class NexusCandidateStore:
    """Append-style in-memory store for governed nexus/self-evolution candidates."""

    records: dict[str, NexusCandidateRecord] = field(default_factory=dict)
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def create_candidate(
        self,
        payload: dict[str, Any],
        case_id: str | None = None,
        source: str = "nexus_branch",
        learning_status: str = "candidate",
    ) -> NexusCandidateRecord:
        payload = payload or {}
        case_id = str(case_id or payload.get("case_id") or payload.get("metadata", {}).get("case_id") or "unknown_case")
        created_at = time.time()
        candidate_id = str(payload.get("candidate_id") or _stable_candidate_id(case_id=case_id, payload=payload, created_at=created_at))
        record = NexusCandidateRecord(
            candidate_id=candidate_id,
            case_id=case_id,
            created_at=created_at,
            payload=payload,
            source=source,
            learning_status=normalize_learning_status(learning_status),
            audit=[{"event": "candidate_created", "timestamp": created_at, "source": source}],
        )
        with self._lock:
            self.records[candidate_id] = record
            self.audit_log.append({"event": "candidate_created", "candidate_id": candidate_id, "timestamp": created_at})
        return record

    def get(self, candidate_id: str) -> NexusCandidateRecord:
        with self._lock:
            return self.records[candidate_id]

    def list_by_status(self, statuses: Iterable[str] | None = None) -> list[NexusCandidateRecord]:
        normalized = {normalize_learning_status(status) for status in statuses} if statuses else set()
        with self._lock:
            records = list(self.records.values())
        if normalized:
            records = [record for record in records if record.learning_status in normalized]
        records.sort(key=lambda item: item.created_at)
        return records

    def attach_validation(self, candidate_id: str, validation: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = self.records[candidate_id]
            record.validation = validation
            event = {"event": "validation_attached", "candidate_id": candidate_id, "timestamp": time.time(), "status": validation.get("status")}
            record.audit.append(event)
            self.audit_log.append(event)
            return record.as_dict()

    def attach_promotion_decision(self, candidate_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = self.records[candidate_id]
            record.promotion_decision = decision
            target_status = normalize_learning_status(decision.get("target_learning_status", record.learning_status))
            transition = validate_learning_transition(
                current=record.learning_status,
                target=target_status,
                evidence={
                    "rational_control_validation": bool((record.validation or {}).get("allowed_for_promotion", False)),
                    "provenance_available": bool((record.validation or {}).get("observed", {}).get("provenance_available", False)),
                    "clinical_deployment": False,
                },
            )
            if transition.allowed:
                record.learning_status = target_status
            else:
                decision = {**decision, "transition_rejected": transition.as_dict()}
                record.promotion_decision = decision
            event = {
                "event": "promotion_decision_attached",
                "candidate_id": candidate_id,
                "timestamp": time.time(),
                "target_learning_status": target_status,
                "transition_allowed": transition.allowed,
            }
            record.audit.append(event)
            self.audit_log.append(event)
            return record.as_dict()

    def attach_outcome_feedback(self, candidate_id: str, feedback: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            record = self.records[candidate_id]
            feedback = {**feedback, "attached_at": time.time()}
            record.outcome_feedback.append(feedback)
            event = {"event": "outcome_feedback_attached", "candidate_id": candidate_id, "timestamp": feedback["attached_at"]}
            record.audit.append(event)
            self.audit_log.append(event)
            return record.as_dict()

    def export_memory_documents(self, statuses: Iterable[str] | None = None) -> list[dict[str, Any]]:
        return [record.to_memory_document() for record in self.list_by_status(statuses=statuses)]

    def save_jsonl(self, path: str | Path) -> None:
        with self._lock:
            rows = [json.dumps(record.as_dict(), sort_keys=True, default=str) for record in self.records.values()]
        Path(path).write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")

    @classmethod
    def load_jsonl(cls, path: str | Path) -> NexusCandidateStore:
        store = cls()
        path = Path(path)
        if not path.exists():
            return store
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            record = NexusCandidateRecord(
                candidate_id=item["candidate_id"],
                case_id=item["case_id"],
                created_at=float(item["created_at"]),
                payload=dict(item.get("payload", {})),
                source=str(item.get("source", "nexus_branch")),
                learning_status=normalize_learning_status(item.get("learning_status")),
                validation=item.get("validation"),
                promotion_decision=item.get("promotion_decision"),
                outcome_feedback=list(item.get("outcome_feedback", [])),
                audit=list(item.get("audit", [])),
            )
            store.records[record.candidate_id] = record
        return store
