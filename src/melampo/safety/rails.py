from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class RailDecision:
    stage: str
    status: str
    reasons: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "reasons": list(self.reasons),
            "actions": list(self.actions),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ClinicalSafetyRails:
    """Phase 5A clinical safety rails for research-only outputs.

    These rails enforce research boundaries around inputs, retrieval evidence and
    diagnostic outputs. They are deterministic and dependency-free so they can be
    run in CI and audit traces.
    """

    max_mismatch_index: float = 0.7
    min_provenance_fraction: float = 0.8
    block_definitive_language: bool = True

    def evaluate_input(self, payload: dict[str, Any]) -> RailDecision:
        reasons: list[str] = []
        if not isinstance(payload, dict):
            reasons.append("payload_not_dict")
        elif not payload.get("case_id"):
            reasons.append("case_id_missing")
        provenance = payload.get("provenance", {}) if isinstance(payload, dict) else {}
        if isinstance(provenance, dict) and provenance.get("contains_phi", False):
            reasons.append("phi_not_marked_deidentified")
        return RailDecision(
            stage="input",
            status="block" if reasons else "pass",
            reasons=reasons,
            actions=["deidentify_or_attach_case_id"] if reasons else [],
            metadata={"clinical_warning": "Input rails do not replace data governance review."},
        )

    def evaluate_retrieval(self, evidence: list[dict[str, Any]]) -> RailDecision:
        reasons: list[str] = []
        if not evidence:
            reasons.append("retrieval_evidence_missing")
            provenance_fraction = 0.0
        else:
            provenance_ready = 0
            synthetic_as_fact = 0
            for item in evidence:
                metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
                has_source = bool(item.get("source") or metadata.get("source_path") or metadata.get("source_uri")) if isinstance(item, dict) else False
                has_trace = bool(item.get("record_id") or metadata.get("page") is not None or metadata.get("section")) if isinstance(item, dict) else False
                if has_source and has_trace:
                    provenance_ready += 1
                learning_status = str(item.get("learning_status", metadata.get("learning_status", ""))) if isinstance(item, dict) else ""
                source_type = str(metadata.get("source_type", item.get("source", ""))) if isinstance(item, dict) else ""
                if learning_status == "candidate" and "synthetic" in source_type:
                    synthetic_as_fact += 1
            provenance_fraction = provenance_ready / max(len(evidence), 1)
            if provenance_fraction < self.min_provenance_fraction:
                reasons.append("retrieval_provenance_below_threshold")
            if synthetic_as_fact:
                reasons.append("synthetic_candidate_used_as_fact")
        return RailDecision(
            stage="retrieval",
            status="block" if reasons else "pass",
            reasons=reasons,
            actions=["abstain_or_request_grounded_sources"] if reasons else [],
            metadata={"provenance_fraction": round(provenance_fraction, 3), "min_provenance_fraction": self.min_provenance_fraction},
        )

    def evaluate_output(self, diagnostic_result: dict[str, Any]) -> RailDecision:
        reasons: list[str] = []
        actions: list[str] = []
        result_label = str(diagnostic_result.get("result_label", "")) if isinstance(diagnostic_result, dict) else ""
        metrics = diagnostic_result.get("melampo_metrics", {}) if isinstance(diagnostic_result, dict) else {}
        policy = diagnostic_result.get("policy", {}) if isinstance(diagnostic_result, dict) else {}
        warning = diagnostic_result.get("audit_trace", {}).get("clinical_warning", "") if isinstance(diagnostic_result, dict) else ""
        if self.block_definitive_language and result_label.lower() in {"diagnosis", "definitive_diagnosis", "final_diagnosis"}:
            reasons.append("definitive_diagnostic_language_blocked")
        mismatch_index = _safe_float(metrics.get("mismatch_index"), 0.0) if isinstance(metrics, dict) else 0.0
        if mismatch_index > self.max_mismatch_index:
            reasons.append("mismatch_index_above_safety_threshold")
            actions.append("escalate_for_human_review")
        if isinstance(policy, dict) and policy.get("abstain", False) and result_label != "abstain_or_escalate":
            reasons.append("policy_abstain_not_reflected_in_result_label")
        if "not a validated medical device" not in warning.lower() and "research output" not in warning.lower():
            reasons.append("clinical_research_warning_missing")
        dream = diagnostic_result.get("dream", {}) if isinstance(diagnostic_result, dict) else {}
        if isinstance(dream, dict) and dream.get("accepted") and dream.get("auto_evolution_plan", {}).get("status") == "candidate" and result_label not in {"abstain_or_escalate", ""}:
            actions.append("verify_dream_candidate_not_used_as_clinical_fact")
        status = "block" if reasons else "pass"
        if actions and status == "pass":
            status = "warn"
        return RailDecision(stage="output", status=status, reasons=reasons, actions=actions, metadata={"mismatch_index": mismatch_index})

    def apply_all(self, payload: dict[str, Any], evidence: list[dict[str, Any]], diagnostic_result: dict[str, Any]) -> dict[str, Any]:
        decisions = [self.evaluate_input(payload), self.evaluate_retrieval(evidence), self.evaluate_output(diagnostic_result)]
        blocked = [decision for decision in decisions if decision.status == "block"]
        warned = [decision for decision in decisions if decision.status == "warn"]
        return {
            "status": "block" if blocked else "warn" if warned else "pass",
            "decisions": [decision.as_dict() for decision in decisions],
            "clinical_use_allowed": False,
            "clinical_warning": "Research-only safety rails; not clinical validation or medical-device clearance.",
        }
