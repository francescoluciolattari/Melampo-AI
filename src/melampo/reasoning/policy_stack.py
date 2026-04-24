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
        abstention = self.abstention.assess(uncertainty)
        risk_gate = self.risk_gate.assess(risk)
        escalation = self.escalation.decide(risk=risk, uncertainty=uncertainty)
        reasons = []
        reasons.extend(abstention.get("reasons", []))
        reasons.extend(risk_gate.get("reasons", []))
        reasons.extend(escalation.get("reasons", []))
        decision_band = "blocked" if (abstention["abstain"] or not risk_gate["allow"]) else "guarded" if escalation.get("escalate", False) else "clear"
        return {
            "abstain": abstention["abstain"],
            "allow": risk_gate["allow"],
            "escalate": escalation["escalate"],
            "risk": risk,
            "uncertainty": uncertainty,
            "reasons": reasons,
            "decision_band": decision_band,
            "escalation_level": escalation.get("level", "low"),
            "abstention_assessment": abstention,
            "risk_assessment": risk_gate,
        }
