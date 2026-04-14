from melampo.models.abstention import AbstentionPolicy
from melampo.models.risk_gate import RiskGate
from melampo.reasoning.escalation import EscalationPolicy
from melampo.reasoning.policy_stack import PolicyStack


def test_policy_stack_escalates_high_risk():
    stack = PolicyStack(
        abstention=AbstentionPolicy(threshold=0.65),
        risk_gate=RiskGate(threshold=0.35),
        escalation=EscalationPolicy(),
    )
    result = stack.evaluate(risk=0.8, uncertainty=0.2)
    assert result["allow"] is False
    assert result["escalate"] is True
