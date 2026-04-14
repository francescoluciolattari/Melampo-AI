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
        return {
            "abstain": abstain,
            "allow": allow,
            "escalate": escalation["escalate"],
            "risk": risk,
            "uncertainty": uncertainty,
        }
