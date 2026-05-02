from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..orchestration.model_capability_registry import ModelCapabilityRegistry


def _safe_get(payload: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


@dataclass(slots=True)
class DiagnosticOrchestratorPolicy:
    """Final deterministic policy for Melampo diagnostic orchestration."""

    min_pi_score: float = 0.35
    max_mismatch_index: float = 0.7
    max_uncertainty: float = 0.75
    require_policy_allow: bool = True

    def evaluate(self, pipeline_result: dict[str, Any]) -> dict[str, Any]:
        area_dynamics = pipeline_result.get("area_dynamics", {})
        neuro = area_dynamics.get("neuro_dynamic_metrics", {}) if isinstance(area_dynamics, dict) else {}
        coordinated = pipeline_result.get("coordinated", {})
        policy = coordinated.get("policy", {}) if isinstance(coordinated, dict) else {}
        pi_score = float(area_dynamics.get("pi_score", neuro.get("pi_score", 0.0))) if isinstance(area_dynamics, dict) else 0.0
        mismatch_index = float(neuro.get("mismatch_index", area_dynamics.get("mismatch_score", 0.0))) if isinstance(area_dynamics, dict) else 1.0
        uncertainty = float(_safe_get(coordinated, ("state_summary", "uncertainty"), 0.0) or 0.0)
        policy_abstain = bool(policy.get("abstain", False))
        policy_escalate = bool(policy.get("escalate", False))
        reasons = []
        if pi_score < self.min_pi_score:
            reasons.append("pi_score_below_threshold")
        if mismatch_index > self.max_mismatch_index:
            reasons.append("mismatch_index_above_threshold")
        if uncertainty > self.max_uncertainty:
            reasons.append("uncertainty_above_threshold")
        if self.require_policy_allow and policy_abstain:
            reasons.append("policy_abstention_requested")
        if policy_escalate:
            reasons.append("policy_escalation_requested")
        return {
            "allow_candidate_result": not reasons,
            "abstain": bool(reasons or policy_abstain),
            "escalate": bool(policy_escalate or mismatch_index > self.max_mismatch_index or uncertainty > self.max_uncertainty),
            "reasons": reasons,
            "thresholds": {
                "min_pi_score": self.min_pi_score,
                "max_mismatch_index": self.max_mismatch_index,
                "max_uncertainty": self.max_uncertainty,
            },
            "observed": {
                "pi_score": pi_score,
                "mismatch_index": mismatch_index,
                "uncertainty": uncertainty,
                "policy_abstain": policy_abstain,
                "policy_escalate": policy_escalate,
            },
        }


@dataclass(slots=True)
class MelampoDiagnosticOrchestrator:
    """Audit-first final diagnostic orchestrator for Melampo.

    This class is the final controller. It is not an LLM wrapper. It consumes the
    full clinical pipeline result, model capability registry, policy state,
    intuition payload, differential hypotheses, area dynamics and critique. It
    returns a structured research diagnostic result with abstention and
    escalation. External models remain specialist signal providers and critics.
    """

    registry: ModelCapabilityRegistry = field(default_factory=ModelCapabilityRegistry.build_default)
    policy: DiagnosticOrchestratorPolicy = field(default_factory=DiagnosticOrchestratorPolicy)

    def orchestrate(self, pipeline_result: dict[str, Any]) -> dict[str, Any]:
        coordinated = pipeline_result.get("coordinated", {})
        differential = coordinated.get("differential", {}) if isinstance(coordinated, dict) else {}
        hypotheses = list(differential.get("hypotheses", [])) if isinstance(differential, dict) else []
        top_hypothesis = hypotheses[0] if hypotheses else {"label": "none", "score": 0.0, "rationale": "no_hypothesis"}
        intuition = pipeline_result.get("intuition", {})
        area_dynamics = pipeline_result.get("area_dynamics", {})
        neuro = area_dynamics.get("neuro_dynamic_metrics", {}) if isinstance(area_dynamics, dict) else {}
        dream = pipeline_result.get("dream", {})
        critique = pipeline_result.get("critique", {})
        orchestration_policy = self.policy.evaluate(pipeline_result)

        support = {
            "visual": pipeline_result.get("area_signals", {}).get("visual_diagnostic", {}),
            "textual": pipeline_result.get("area_signals", {}).get("language_listening", {}),
            "context": pipeline_result.get("area_signals", {}).get("case_context", {}),
            "epidemiology": pipeline_result.get("area_signals", {}).get("epidemiology", {}),
            "semantic_memory": pipeline_result.get("retrieval", {}),
        }
        result_label = top_hypothesis.get("label", "none") if orchestration_policy["allow_candidate_result"] else "abstain_or_escalate"
        diagnostic_result = {
            "case_id": pipeline_result.get("case_id", "unknown_case"),
            "result_label": result_label,
            "top_hypothesis": top_hypothesis,
            "differential": hypotheses,
            "intuition": {
                "selected": intuition.get("intuition", "none") if isinstance(intuition, dict) else "none",
                "reasoning_mode": _safe_get(intuition, ("deductive_filter", "reasoning_mode"), "unknown"),
                "candidate_scores": intuition.get("candidate_scores", []) if isinstance(intuition, dict) else [],
            },
            "melampo_metrics": {
                "pi_score": area_dynamics.get("pi_score", neuro.get("pi_score", 0.0)) if isinstance(area_dynamics, dict) else 0.0,
                "precision_weighted_coherence": area_dynamics.get("precision_weighted_coherence", neuro.get("precision_weighted_coherence", 0.0)) if isinstance(area_dynamics, dict) else 0.0,
                "prediction_error": area_dynamics.get("prediction_error", neuro.get("prediction_error", 0.0)) if isinstance(area_dynamics, dict) else 0.0,
                "convergence_index": neuro.get("convergence_index", 0.0),
                "mismatch_index": neuro.get("mismatch_index", 0.0),
                "deductive_gate": neuro.get("deductive_gate", 0.0),
                "candidate_temperature": neuro.get("candidate_temperature", 1.0),
                "belief_update_rate": neuro.get("belief_update_rate", 0.0),
            },
            "support": support,
            "policy": orchestration_policy,
            "critique": critique,
            "dream": {
                "accepted": dream.get("accepted", False) if isinstance(dream, dict) else False,
                "auto_evolution_plan": dream.get("auto_evolution_plan", {}) if isinstance(dream, dict) else {},
                "alternative_hypotheses": dream.get("alternative_hypotheses", []) if isinstance(dream, dict) else [],
            },
            "model_capability_decision_record": self.registry.decision_record(),
            "audit_trace": {
                "pipeline_trace": coordinated.get("trace", []) if isinstance(coordinated, dict) else [],
                "external_models_are_not_final_arbiters": True,
                "final_authority": "MelampoDiagnosticOrchestrator",
                "clinical_warning": "Research output; not a validated medical device or standalone diagnosis.",
            },
        }
        return diagnostic_result
