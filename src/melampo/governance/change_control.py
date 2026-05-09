from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


_HIGH_RISK_COMPONENTS = {"model", "memory", "retriever", "policy", "dream_branch", "orchestrator"}


def _hash_change(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


@dataclass(slots=True)
class ChangeRecord:
    component: str
    change_type: str
    description: str
    planned: bool = True
    risk_level: str = "medium"
    validation_required: list[str] = field(default_factory=list)
    rollback_plan: str = "restore_previous_locked_version"
    approval_status: str = "draft"
    created_at: float = field(default_factory=time.time)
    change_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.validation_required:
            requirements = ["unit_tests", "audit_trace_review"]
            if self.component in _HIGH_RISK_COMPONENTS or self.risk_level in {"medium", "high"}:
                requirements.extend(["benchmark_regression", "release_gate_review"])
            if self.risk_level == "high":
                requirements.extend(["human_governance_review", "rollback_drill"])
            self.validation_required = list(dict.fromkeys(requirements))
        if self.change_id is None:
            self.change_id = _hash_change({
                "component": self.component,
                "change_type": self.change_type,
                "description": self.description,
                "created_at": self.created_at,
            })

    def as_dict(self) -> dict[str, Any]:
        return {
            "change_id": self.change_id,
            "component": self.component,
            "change_type": self.change_type,
            "description": self.description,
            "planned": self.planned,
            "risk_level": self.risk_level,
            "validation_required": list(self.validation_required),
            "rollback_plan": self.rollback_plan,
            "approval_status": self.approval_status,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "governance": {
                "clinical_warning": "Change-control approval is research governance, not regulatory clearance.",
                "pccp_like": True,
            },
        }

    def approve(self, reviewer: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        evidence = evidence or {}
        if self.risk_level == "high" and not evidence.get("human_governance_review"):
            self.approval_status = "needs_review"
            return {"status": "needs_review", "reason": "high_risk_change_requires_human_governance_review", "change": self.as_dict()}
        self.approval_status = "approved"
        self.metadata = {**self.metadata, "approved_by": reviewer, "approval_evidence": evidence, "approved_at": time.time()}
        return {"status": "approved", "change": self.as_dict()}

    def reject(self, reviewer: str, reason: str) -> dict[str, Any]:
        self.approval_status = "rejected"
        self.metadata = {**self.metadata, "rejected_by": reviewer, "rejection_reason": reason, "rejected_at": time.time()}
        return {"status": "rejected", "change": self.as_dict()}


@dataclass(slots=True)
class ChangeControlRegistry:
    records: dict[str, ChangeRecord] = field(default_factory=dict)

    def propose(self, record: ChangeRecord | dict[str, Any]) -> dict[str, Any]:
        record_obj = record if isinstance(record, ChangeRecord) else ChangeRecord(**record)
        self.records[str(record_obj.change_id)] = record_obj
        return {"status": "proposed", "change": record_obj.as_dict()}

    def get(self, change_id: str) -> ChangeRecord:
        return self.records[change_id]

    def approve(self, change_id: str, reviewer: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.records[change_id].approve(reviewer=reviewer, evidence=evidence)

    def reject(self, change_id: str, reviewer: str, reason: str) -> dict[str, Any]:
        return self.records[change_id].reject(reviewer=reviewer, reason=reason)

    def summarize(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        risk_levels: dict[str, int] = {}
        for record in self.records.values():
            statuses[record.approval_status] = statuses.get(record.approval_status, 0) + 1
            risk_levels[record.risk_level] = risk_levels.get(record.risk_level, 0) + 1
        return {"change_count": len(self.records), "statuses": statuses, "risk_levels": risk_levels}
