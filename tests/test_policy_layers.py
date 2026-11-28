from melampo.models.abstention import AbstentionPolicy
from melampo.models.risk_gate import RiskGate
from melampo.reasoning.escalation import EscalationPolicy
from melampo.reasoning.policy_stack import PolicyStack


def test_policy_layers_expose_structured_assessments():
    abstention = AbstentionPolicy(threshold=0.65).assess(0.7)
    assert abstention["abstain"] is True
    assert abstention["reasons"]
    assert abstention["level"] in ["high", "guarded", "low"]

    risk = RiskGate(threshold=0.35).assess(0.4)
    assert risk["allow"] is False
    assert risk["reasons"]
    assert risk["band"] in ["high", "guarded", "low"]

    policy = PolicyStack(
        abstention=AbstentionPolicy(threshold=0.65),
        risk_gate=RiskGate(threshold=0.35),
        escalation=EscalationPolicy(),
    ).evaluate(risk=0.4, uncertainty=0.7)
    assert policy["decision_band"] in ["blocked", "guarded", "clear"]
    assert policy["abstention_assessment"]
    assert policy["risk_assessment"]
