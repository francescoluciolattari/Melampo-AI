from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


@dataclass(slots=True)
class ClinicalClaim:
    """Structured claim emitted by a specialist model or Melampo area.

    Claims are deliberately small and auditable. They are used by the area
    coherence, RAG, critic, and diagnostic orchestration layers to reason about
    support, contradiction, missing evidence, and provenance.
    """

    claim_id: str
    type: str
    normalized_entity: str
    polarity: str = "present"
    confidence: float = 0.0
    uncertainty: float = 1.0
    ontology_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    source_area: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "type": self.type,
            "normalized_entity": self.normalized_entity,
            "polarity": self.polarity,
            "confidence": round(_clamp(float(self.confidence)), 3),
            "uncertainty": round(_clamp(float(self.uncertainty)), 3),
            "ontology_refs": list(self.ontology_refs),
            "evidence_refs": list(self.evidence_refs),
            "source_area": self.source_area,
            "provenance": dict(self.provenance),
            "limitations": list(self.limitations),
        }


@dataclass(slots=True)
class SpecialistModelResponse:
    """Unified structured response contract for every external specialist model."""

    provider: str
    model_name: str
    role: str
    status: str
    signals: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    uncertainty: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    contradictions: list[dict[str, Any]] = field(default_factory=list)
    audit_trace: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "role": self.role,
            "status": self.status,
            "signals": self.signals,
            "confidence": round(_clamp(float(self.confidence)), 3),
            "uncertainty": round(_clamp(float(self.uncertainty)), 3),
            "provenance": self.provenance,
            "limitations": self.limitations,
            "claims": self.claims,
            "missing_evidence": self.missing_evidence,
            "contradictions": self.contradictions,
            "audit_trace": self.audit_trace,
            "created_at": self.created_at,
        }

    def as_area_signal(self, area: str) -> dict[str, Any]:
        signal_count = len(self.signals)
        if self.claims:
            signal_count = max(signal_count, len(self.claims))
        return {
            "area": area,
            "provider": self.provider,
            "model_name": self.model_name,
            "role": self.role,
            "status": self.status,
            "signals": self.signals,
            "claims": self.claims,
            "missing_evidence": self.missing_evidence,
            "contradictions": self.contradictions,
            "signal_count": signal_count,
            "salience_score": round(_clamp(float(self.confidence)), 3),
            "uncertainty_score": round(_clamp(float(self.uncertainty)), 3),
            "provenance": self.provenance,
            "limitations": self.limitations,
            "audit_trace": self.audit_trace,
        }

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        provider: str,
        model_name: str,
        role: str,
        default_status: str = "completed",
        provenance: dict[str, Any] | None = None,
        limitations: list[str] | None = None,
    ) -> "SpecialistModelResponse":
        payload = payload if isinstance(payload, dict) else {}
        confidence = payload.get("confidence", payload.get("score", 0.0))
        uncertainty = payload.get("uncertainty", 1.0 - _clamp(float(confidence or 0.0)))
        return cls(
            provider=str(payload.get("provider", provider)),
            model_name=str(payload.get("model_name", model_name)),
            role=str(payload.get("role", role)),
            status=str(payload.get("status", default_status)),
            signals=dict(payload.get("signals", {})),
            confidence=_clamp(float(confidence or 0.0)),
            uncertainty=_clamp(float(uncertainty or 0.0)),
            provenance={**(provenance or {}), **dict(payload.get("provenance", {}))},
            limitations=list(limitations or []) + list(payload.get("limitations", [])),
            claims=list(payload.get("claims", [])),
            missing_evidence=list(payload.get("missing_evidence", [])),
            contradictions=list(payload.get("contradictions", [])),
            audit_trace=dict(payload.get("audit_trace", {})),
        )
