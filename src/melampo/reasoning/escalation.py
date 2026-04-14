from dataclasses import dataclass


@dataclass
class EscalationPolicy:
    """Minimal escalation policy placeholder."""

    def decide(self, risk: float, uncertainty: float) -> dict:
        escalate = risk > 0.35 or uncertainty > 0.65
        return {"escalate": escalate, "risk": risk, "uncertainty": uncertainty}
