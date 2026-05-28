from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..memory.learning_status import validate_learning_transition


@dataclass(slots=True)
class PromotionPolicy:
    """Decide how validated dream candidates move through governed memory states."""

    allow_automatic_promotion: bool = False
    promote_to_review_by_default: bool = True
    min_candidate_score: float = 0.55
    require_human_review_for_promoted: bool = True
    promotion_scope: str = "synthetic_curriculum_or_vector_memory_only"

    def decide(self, candidate: dict[str, Any], validation: dict[str, Any]) -> dict[str, Any]:
        candidate = candidate or {}
        validation = validation or {}
        payload = candidate.get("payload", candidate)
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata", {}), dict) else {}
        auto_plan = payload.get("auto_evolution_plan", metadata.get("auto_evolution_plan", {}))
        candidate_score = float(auto_plan.get("candidate_score", metadata.get("candidate_score", validation.get("observed", {}).get("candidate_score", 0.0))))
        validation_allowed = bool(validation.get("allowed_for_promotion", False))
        current_status = candidate.get("learning_status", "candidate")

        reasons: list[str] = []
        if not validation_allowed:
            reasons.append("rational_control_validation_not_satisfied")
        if candidate_score < self.min_candidate_score:
            reasons.append("candidate_score_below_promotion_policy_threshold")

        if validation.get("status") == "rejected" or validation.get("hard_reject", False):
            target = "rejected"
            action = "reject_candidate"
        elif reasons:
            target = "needs_review" if self.promote_to_review_by_default else "candidate"
            action = "hold_for_review"
        elif self.allow_automatic_promotion and not self.require_human_review_for_promoted:
            target = "promoted"
            action = "promote_to_governed_memory"
        else:
            target = "needs_review"
            action = "queue_for_human_or_protocol_review"

        transition = validate_learning_transition(
            current=current_status,
            target=target,
            evidence={
                "rational_control_validation": validation_allowed,
                "provenance_available": bool(validation.get("observed", {}).get("provenance_available", False)),
                "clinical_deployment": False,
            },
        )
        if not transition.allowed and target == "promoted":
            target = "needs_review"
            action = "promotion_blocked_queue_for_review"
            reasons.extend(transition.reasons)

        return {
            "action": action,
            "target_learning_status": target,
            "candidate_score": round(max(0.0, min(1.0, candidate_score)), 3),
            "reasons": reasons,
            "transition": transition.as_dict(),
            "policy": {
                "allow_automatic_promotion": self.allow_automatic_promotion,
                "require_human_review_for_promoted": self.require_human_review_for_promoted,
                "promotion_scope": self.promotion_scope,
                "clinical_deployment_allowed": False,
            },
        }
