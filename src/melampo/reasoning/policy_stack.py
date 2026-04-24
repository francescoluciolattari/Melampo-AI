from dataclasses import dataclass

from .escalation import EscalationPolicy
from ..models.abstention import AbstentionPolicy
from ..models.risk_gate import RiskGate


@dataclass
class PolicyStack:
    """Coordinate abstention, risk gating, and escalation decisions."""

    abstention: AbstentionPolicy
    risk_gate: RiskGate
    escalation: EscalationPolicy

    def evaluate(self, risk: float, uncertainty: float) -> dict:
        abstain = self.abstention.should_abstain(uncertainty)
        allow = self.risk_gate.allow(risk)
        escalation = self.escalation.decide(risk=risk, uncertainty=uncertainty)
        reasons = []
        if abstain:
            reasons.append("abstention_triggered")
        if not allow:
            reasons.append("risk_gate_blocked")
        reasons.extend(escalation.get("reasons", []))
        decision_band = "blocked" if (abstain or not allow) else "guarded" if escalation.get("escalate", False) else "clear"
        return {
            "abstain": abstain,
            "allow": allow,
            "escalate": escalation["escalate"],
            "risk": risk,
            "uncertainty": uncertainty,
            "reasons": reasons,
            "decision_band": decision_band,
            "escalation_level": escalation.get("level", "low"),
        }
