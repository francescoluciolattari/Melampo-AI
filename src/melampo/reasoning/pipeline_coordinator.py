from dataclasses import dataclass

from .decision_trace import DecisionTrace
from .differential_engine import DifferentialEngine
from .pipeline_state import PipelineState
from .policy_stack import PolicyStack


@dataclass
class PipelineCoordinator:
    """Coordinate differential ranking, tracing, and policy evaluation."""

    differential_engine: DifferentialEngine
    policy_stack: PolicyStack

    def run(
        self,
        case_id: str,
        evidence: list,
        risk: float,
        uncertainty: float,
        intuition: dict | None = None,
        nexus: dict | None = None,
        discordances: dict | None = None,
    ) -> dict:
        state = PipelineState(
            case_id=case_id, evidence=list(evidence), risk=risk, uncertainty=uncertainty
        )
        differential = self.differential_engine.rank(
            evidence, intuition=intuition, nexus=nexus, discordances=discordances
        )
        policy = self.policy_stack.evaluate(risk=risk, uncertainty=uncertainty)
        top_hypothesis = differential.get(
            "hypotheses", [{"label": "none", "hypothesis_domain": "multimodal_led"}]
        )[0]
        action_count = len(differential.get("recommended_actions", []))

        state.hypotheses = list(differential.get("hypotheses", []))
        state.abstain = policy["abstain"]
        state.escalate = policy["escalate"]

        trace = DecisionTrace()
        trace.add_kv("case", case_id)
        trace.add_kv("evidence_count", len(evidence))
        trace.add_kv("policy_band", policy.get("decision_band", "clear"))
        trace.add_kv("differential_top", top_hypothesis["label"])
        trace.add_kv(
            "top_domain", top_hypothesis.get("hypothesis_domain", "multimodal_led")
        )
        trace.add_kv("recommended_actions", action_count)
        trace.add_kv("discordances", differential.get("discordances", 0))

        return {
            "state": state,
            "state_summary": state.summary(),
            "differential": differential,
            "policy": policy,
            "trace": trace.dump(),
        }
