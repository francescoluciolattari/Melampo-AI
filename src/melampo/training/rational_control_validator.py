from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _nested(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cursor: Any = payload
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


@dataclass(slots=True)
class RationalControlRubric:
    min_pi_score: float = 0.55
    min_convergence_index: float = 0.45
    max_mismatch_index: float = 0.45
    max_risk: float = 0.25
    min_candidate_score: float = 0.45
    min_provenance_quality: float = 0.5
    min_retrieval_coverage: float = 0.2
    hard_reject_risk: float = 0.75
    required_guardrails: tuple[str, ...] = (
        "requires rational-control validation",
        "requires provenance and source labeling",
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_pi_score": self.min_pi_score,
            "min_convergence_index": self.min_convergence_index,
            "max_mismatch_index": self.max_mismatch_index,
            "max_risk": self.max_risk,
            "min_candidate_score": self.min_candidate_score,
            "min_provenance_quality": self.min_provenance_quality,
            "min_retrieval_coverage": self.min_retrieval_coverage,
            "hard_reject_risk": self.hard_reject_risk,
            "required_guardrails": list(self.required_guardrails),
        }


@dataclass(slots=True)
class RationalControlValidator:
    """Validate dream/self-evolution candidates before memory promotion."""

    rubric: RationalControlRubric = field(default_factory=RationalControlRubric)

    def evaluate(
        self,
        candidate: dict[str, Any],
        area_dynamics: dict[str, Any] | None = None,
        retrieval_context: dict[str, Any] | None = None,
        governance_scores: dict[str, Any] | None = None,
        outcome_feedback: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        candidate = candidate or {}
        area_dynamics = area_dynamics or candidate.get("area_dynamics", {}) or {}
        retrieval_context = retrieval_context or candidate.get("retrieval_context", {}) or {}
        governance_scores = governance_scores or candidate.get("governance_scores", {}) or {}
        outcome_feedback = outcome_feedback or {}
        metadata = candidate.get("metadata", {}) if isinstance(candidate.get("metadata", {}), dict) else {}
        auto_plan = candidate.get("auto_evolution_plan", {}) if isinstance(candidate.get("auto_evolution_plan", {}), dict) else {}
        if not auto_plan:
            auto_plan = metadata.get("auto_evolution_plan", {}) if isinstance(metadata.get("auto_evolution_plan", {}), dict) else {}
        neuro = area_dynamics.get("neuro_dynamic_metrics", {}) if isinstance(area_dynamics, dict) else {}

        pi_score = _safe_float(metadata.get("pi_score", neuro.get("pi_score", area_dynamics.get("pi_score", 0.0))))
        convergence_index = _safe_float(metadata.get("convergence_index", neuro.get("convergence_index", area_dynamics.get("convergence_index", 0.0))))
        mismatch_index = _safe_float(metadata.get("mismatch_index", neuro.get("mismatch_index", area_dynamics.get("mismatch_index", 0.0))))
        risk = _safe_float(governance_scores.get("risk", metadata.get("risk", auto_plan.get("risk", 0.0))))
        candidate_score = _safe_float(auto_plan.get("candidate_score", metadata.get("candidate_score", 0.0)))
        provenance_quality = _safe_float(governance_scores.get("provenance_quality", metadata.get("provenance_quality", 0.0)))
        retrieval_coverage = _safe_float(governance_scores.get("retrieval_coverage", retrieval_context.get("retrieval_coverage", metadata.get("retrieval_coverage", 0.0))))
        guardrails = list(auto_plan.get("promotion_guardrails", metadata.get("promotion_guardrails", [])))
        provenance_available = bool(metadata.get("source") or metadata.get("case_id") or candidate.get("source") or provenance_quality > 0.0)
        favorable_outcome = bool(outcome_feedback.get("correct", False) or outcome_feedback.get("review_status") == "accepted")

        failures: list[str] = []
        if pi_score < self.rubric.min_pi_score:
            failures.append("pi_score_below_threshold")
        if convergence_index < self.rubric.min_convergence_index:
            failures.append("convergence_index_below_threshold")
        if mismatch_index > self.rubric.max_mismatch_index:
            failures.append("mismatch_index_above_threshold")
        if risk > self.rubric.max_risk:
            failures.append("risk_above_threshold")
        if candidate_score < self.rubric.min_candidate_score:
            failures.append("candidate_score_below_threshold")
        if provenance_quality < self.rubric.min_provenance_quality and not provenance_available:
            failures.append("provenance_quality_below_threshold")
        if retrieval_coverage < self.rubric.min_retrieval_coverage and not favorable_outcome:
            failures.append("retrieval_coverage_below_threshold")
        for guardrail in self.rubric.required_guardrails:
            if guardrails and guardrail not in guardrails:
                failures.append(f"missing_guardrail:{guardrail}")
        hard_reject = risk >= self.rubric.hard_reject_risk or "clinical_deployment" in str(candidate).lower()

        allowed_for_promotion = not failures and not hard_reject
        if hard_reject:
            status = "rejected"
        elif allowed_for_promotion:
            status = "validated_for_promotion_review"
        else:
            status = "needs_review"

        return {
            "status": status,
            "allowed_for_promotion": allowed_for_promotion,
            "rational_control_validation": True,
            "failures": failures,
            "hard_reject": hard_reject,
            "criteria": self.rubric.as_dict(),
            "observed": {
                "pi_score": round(_clamp(pi_score), 3),
                "convergence_index": round(_clamp(convergence_index), 3),
                "mismatch_index": round(_clamp(mismatch_index), 3),
                "risk": round(_clamp(risk), 3),
                "candidate_score": round(_clamp(candidate_score), 3),
                "provenance_quality": round(_clamp(provenance_quality), 3),
                "retrieval_coverage": round(_clamp(retrieval_coverage), 3),
                "provenance_available": provenance_available,
                "favorable_outcome": favorable_outcome,
            },
            "governance": {
                "human_review_before_clinical_use": True,
                "synthetic_candidate_not_clinical_truth": True,
                "promotion_target": "vector_memory_candidate_or_synthetic_curriculum_only",
            },
        }
