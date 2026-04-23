from dataclasses import dataclass

from .pipeline_state import PipelineState
from .policy_stack import PolicyStack
from .decision_trace import DecisionTrace
from .differential_engine import DifferentialEngine


@dataclass
class PipelineCoordinator:
    """Coordinate differential ranking, tracing, and policy evaluation."""

    differential_engine: DifferentialEngine
    policy_stack: PolicyStack

    def run(self, case_id: str, evidence: list, risk: float, uncertainty: float, intuition: dict | None = None, dream: dict | None = None, area_dynamics: dict | None = None) -> dict:
        state = PipelineState(case_id=case_id, evidence=list(evidence), risk=risk, uncertainty=uncertainty)
        differential = self.differential_engine.rank(evidence, intuition=intuition, dream=dream, area_dynamics=area_dynamics)
        policy = self.policy_stack.evaluate(risk=risk, uncertainty=uncertainty)
        trace = DecisionTrace()
        trace.add(f"case:{case_id}")
        trace.add(f"evidence_count:{len(evidence)}")
        trace.add(f"policy:{policy}")
        trace.add(f"differential_top:{differential['hypotheses'][0]['label']}")
        state.abstain = policy["abstain"]
        state.escalate = policy["escalate"]
        return {
            "state": state,
            "differential": differential,
            "policy": policy,
            "trace": trace.dump(),
        }
