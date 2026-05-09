from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_LEARNING_STATUSES = ("candidate", "needs_review", "promoted", "rejected", "retired")

GOVERNED_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "candidate": ("candidate", "needs_review", "promoted", "rejected", "retired"),
    "needs_review": ("needs_review", "promoted", "rejected", "retired"),
    "promoted": ("promoted", "retired"),
    "rejected": ("rejected", "retired"),
    "retired": ("retired",),
}


def normalize_learning_status(status: str | None) -> str:
    value = str(status or "candidate").strip().lower()
    return value if value in ALLOWED_LEARNING_STATUSES else "candidate"


@dataclass(frozen=True, slots=True)
class LearningStatusTransition:
    current: str
    target: str
    allowed: bool
    reasons: list[str] = field(default_factory=list)
    governance: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "target": self.target,
            "allowed": self.allowed,
            "reasons": self.reasons,
            "governance": self.governance,
        }


def validate_learning_transition(
    current: str | None,
    target: str | None,
    evidence: dict[str, Any] | None = None,
) -> LearningStatusTransition:
    evidence = evidence or {}
    normalized_current = normalize_learning_status(current)
    normalized_target = normalize_learning_status(target)
    reasons: list[str] = []

    if normalized_target not in GOVERNED_TRANSITIONS.get(normalized_current, ()):  # defensive fallback
        reasons.append("transition_not_allowed_by_learning_status_policy")

    if normalized_target == "promoted":
        if not evidence.get("rational_control_validation", False):
            reasons.append("promotion_requires_rational_control_validation")
        if not evidence.get("provenance_available", False):
            reasons.append("promotion_requires_provenance")
        if evidence.get("clinical_deployment", False):
            reasons.append("promotion_cannot_enable_clinical_deployment")

    allowed = not reasons
    return LearningStatusTransition(
        current=normalized_current,
        target=normalized_target,
        allowed=allowed,
        reasons=reasons,
        governance={
            "allowed_statuses": list(ALLOWED_LEARNING_STATUSES),
            "dream_generated_default_status": "candidate",
            "clinical_warning": "Learning-status promotion is research memory governance only, not clinical deployment approval.",
        },
    )
