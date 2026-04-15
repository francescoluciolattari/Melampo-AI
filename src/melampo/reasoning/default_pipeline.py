from dataclasses import dataclass

from ..models.abstention import AbstentionPolicy
from ..models.risk_gate import RiskGate
from .differential_engine import DifferentialEngine
from .escalation import EscalationPolicy
from .pipeline_coordinator import PipelineCoordinator
from .policy_stack import PolicyStack


@dataclass
class DefaultPipelineFactory:
    """Factory for a minimally connected reasoning pipeline."""

    def build(self) -> PipelineCoordinator:
        return PipelineCoordinator(
            differential_engine=DifferentialEngine(),
            policy_stack=PolicyStack(
                abstention=AbstentionPolicy(threshold=0.65),
                risk_gate=RiskGate(threshold=0.35),
                escalation=EscalationPolicy(),
            ),
        )
