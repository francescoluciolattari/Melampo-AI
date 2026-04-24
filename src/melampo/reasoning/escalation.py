from dataclasses import dataclass


@dataclass
class EscalationPolicy:
    """Escalation policy with lightweight reason tracing."""

    def decide(self, risk: float, uncertainty: float) -> dict:
        reasons = []
        if risk > 0.35:
            reasons.append("risk_above_threshold")
        if uncertainty > 0.65:
            reasons.append("uncertainty_above_threshold")
        escalate = bool(reasons)
        level = "high" if (risk > 0.5 or uncertainty > 0.8) else "moderate" if escalate else "low"
        return {
            "escalate": escalate,
            "risk": risk,
            "uncertainty": uncertainty,
            "reasons": reasons,
            "level": level,
        }
