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
        reasoning_mode = differential.get("reasoning_mode", "rapid_intuition")
        top_hypothesis = differential.get("hypotheses", [{"label": "none", "hypothesis_domain": "multimodal_led"}])[0]
        action_count = len(differential.get("recommended_actions", []))
        mismatch_score = float(differential.get("mismatch_score", 0.0))

        trace = DecisionTrace()
        trace.add(f"case:{case_id}")
        trace.add(f"evidence_count:{len(evidence)}")
        trace.add(f"policy:{policy}")
        trace.add(f"reasoning_mode:{reasoning_mode}")
        trace.add(f"differential_top:{top_hypothesis['label']}")
        trace.add(f"top_domain:{top_hypothesis.get('hypothesis_domain', 'multimodal_led')}")
        trace.add(f"recommended_actions:{action_count}")
        trace.add(f"mismatch_score:{mismatch_score}")

        state.abstain = policy["abstain"]
        state.escalate = policy["escalate"]
        return {
            "state": state,
            "differential": differential,
            "policy": policy,
            "trace": trace.dump(),
        }
