from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from .dataset_manifest import DatasetManifest


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


@dataclass(slots=True)
class ValidationEndpoint:
    name: str
    metric: str
    threshold: float
    direction: str = "gte"
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "threshold": self.threshold,
            "direction": self.direction,
            "required": self.required,
        }

    def evaluate(self, observed: dict[str, Any]) -> dict[str, Any]:
        value = observed.get(self.metric)
        try:
            if value is None:
                raise TypeError("missing metric")
            numeric_value = float(value)
        except (TypeError, ValueError):
            return {"name": self.name, "status": "missing", "metric": self.metric, "observed": value, "required": self.required}
        if self.direction == "lte":
            passed = numeric_value <= self.threshold
        else:
            passed = numeric_value >= self.threshold
        return {
            "name": self.name,
            "status": "pass" if passed else "fail",
            "metric": self.metric,
            "observed": numeric_value,
            "threshold": self.threshold,
            "direction": self.direction,
            "required": self.required,
        }


@dataclass(slots=True)
class ValidationProtocol:
    """Locked research validation protocol for Melampo Phase 5A.

    Protocols become auditable only after `lock()` is called. After locking,
    callers should create a new protocol revision instead of mutating study
    thresholds, model versions or memory snapshots.
    """

    protocol_id: str
    title: str
    intended_use: str = "research_only"
    dataset_id: str = "unknown_dataset"
    locked_model_version: str = "unlocked_model"
    locked_memory_snapshot: str = "unlocked_memory"
    endpoints: list[ValidationEndpoint] = field(default_factory=list)
    required_slices: list[str] = field(default_factory=list)
    status: str = "draft"
    created_at: float = field(default_factory=time.time)
    locked_at: float | None = None
    governance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default_research_protocol(
        cls,
        protocol_id: str = "melampo_phase5a_research_protocol",
        dataset_id: str = "unknown_dataset",
    ) -> "ValidationProtocol":
        return cls(
            protocol_id=protocol_id,
            title="Melampo Phase 5A research validation protocol",
            intended_use="research_only",
            dataset_id=dataset_id,
            endpoints=[
                ValidationEndpoint("minimum_coverage", "coverage", 0.5, "gte", True),
                ValidationEndpoint("minimum_selective_accuracy", "selective_accuracy", 0.6, "gte", True),
                ValidationEndpoint("maximum_expected_calibration_error", "expected_calibration_error", 0.25, "lte", True),
                ValidationEndpoint("minimum_rag_provenance", "provenance_completeness", 0.8, "gte", True),
            ],
            required_slices=["modality", "pathology_family", "site", "learning_status"],
            governance={
                "clinical_warning": "Research protocol only; not clinical validation or regulatory clearance.",
                "prediction_lock_required": True,
                "dataset_manifest_required": True,
            },
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": self.protocol_id,
            "title": self.title,
            "intended_use": self.intended_use,
            "dataset_id": self.dataset_id,
            "locked_model_version": self.locked_model_version,
            "locked_memory_snapshot": self.locked_memory_snapshot,
            "endpoints": [endpoint.as_dict() for endpoint in self.endpoints],
            "required_slices": list(self.required_slices),
            "status": self.status,
            "created_at": self.created_at,
            "locked_at": self.locked_at,
            "governance": self.governance,
        }

    def fingerprint(self) -> str:
        return _canonical_hash(self.as_dict())

    def lock(self, model_version: str, memory_snapshot: str) -> dict[str, Any]:
        if self.status != "draft":
            return {"status": "not_locked", "reason": "protocol_not_in_draft_state", "protocol_status": self.status}
        self.locked_model_version = model_version
        self.locked_memory_snapshot = memory_snapshot
        self.locked_at = time.time()
        self.status = "locked"
        return {"status": "locked", "protocol_id": self.protocol_id, "fingerprint": self.fingerprint()}

    def readiness(self, dataset_manifest: DatasetManifest | None = None) -> dict[str, Any]:
        failures: list[str] = []
        warnings: list[str] = []
        if self.status != "locked":
            failures.append("protocol_not_locked")
        if self.intended_use != "research_only":
            failures.append("intended_use_not_research_only")
        if self.locked_model_version == "unlocked_model":
            failures.append("model_version_not_locked")
        if self.locked_memory_snapshot == "unlocked_memory":
            failures.append("memory_snapshot_not_locked")
        if not self.endpoints:
            failures.append("endpoints_missing")
        if dataset_manifest is None:
            warnings.append("dataset_manifest_not_attached")
        else:
            manifest_validation = dataset_manifest.validate()
            if manifest_validation["status"] != "pass":
                failures.extend(f"dataset:{failure}" for failure in manifest_validation["failures"])
            missing_slices = sorted(set(self.required_slices) - set(dataset_manifest.required_slices))
            if missing_slices:
                warnings.append(f"dataset_missing_protocol_slices:{','.join(missing_slices)}")
        return {
            "status": "ready" if not failures else "blocked",
            "failures": failures,
            "warnings": warnings,
            "protocol_id": self.protocol_id,
            "fingerprint": self.fingerprint(),
        }

    def evaluate_observed_metrics(self, observed: dict[str, Any]) -> dict[str, Any]:
        endpoint_results = [endpoint.evaluate(observed) for endpoint in self.endpoints]
        failures = [result["name"] for result in endpoint_results if result["status"] != "pass" and result.get("required", True)]
        return {
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "endpoint_results": endpoint_results,
            "observed": observed,
        }


@dataclass(slots=True)
class ValidationProtocolRegistry:
    protocols: dict[str, ValidationProtocol] = field(default_factory=dict)

    def register(self, protocol: ValidationProtocol) -> dict[str, Any]:
        self.protocols[protocol.protocol_id] = protocol
        return {"status": "registered", "protocol_id": protocol.protocol_id, "protocol_status": protocol.status}

    def get(self, protocol_id: str) -> ValidationProtocol:
        return self.protocols[protocol_id]

    def summarize(self) -> dict[str, Any]:
        statuses: dict[str, int] = {}
        for protocol in self.protocols.values():
            statuses[protocol.status] = statuses.get(protocol.status, 0) + 1
        return {"protocol_count": len(self.protocols), "statuses": statuses}
