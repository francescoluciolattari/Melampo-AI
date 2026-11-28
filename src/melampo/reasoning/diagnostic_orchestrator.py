from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..orchestration.model_capability_registry import ModelCapabilityRegistry
from ..utils.values import safe_float as _safe_float
from .diagnostic_result import (
    DiagnosticResult,
    IntuitionSummary,
    NexusSummary,
)


def _safe_get(
    payload: dict[str, Any], path: tuple[str, ...], default: Any = None
) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


@dataclass(slots=True)
class DiagnosticOrchestratorPolicy:
    """Final deterministic policy for Melampo diagnostic orchestration."""

    max_uncertainty: float = 0.75
    require_policy_allow: bool = True

    def evaluate(self, pipeline_result: dict[str, Any]) -> dict[str, Any]:
        """The gate of the result: uncertainty and the policy's abstention or escalation.

        Until 28 November 2026 it also abstained on a pi_score below 0.35 and a mismatch between areas above 0.7,
        measures of the coherence between areas (removed, never measured). The case's discordances escalate:
        they go to a person, and they do not withhold the result (principle of doubt).
        """
        coordinated = pipeline_result.get("coordinated", {})
        policy = coordinated.get("policy", {}) if isinstance(coordinated, dict) else {}
        discordances = pipeline_result.get("discordances", {})
        discordance_count = (
            int(_safe_float(discordances.get("count", 0)))
            if isinstance(discordances, dict)
            else 0
        )
        uncertainty = _safe_float(
            _safe_get(coordinated, ("state_summary", "uncertainty"), 0.0)
        )
        policy_abstain = bool(policy.get("abstain", False))
        policy_escalate = bool(policy.get("escalate", False))
        reasons = []
        if uncertainty > self.max_uncertainty:
            reasons.append("uncertainty_above_threshold")
        if self.require_policy_allow and policy_abstain:
            reasons.append("policy_abstention_requested")
        if policy_escalate:
            reasons.append("policy_escalation_requested")
        return {
            "allow_candidate_result": not reasons,
            "abstain": bool(reasons or policy_abstain),
            "escalate": bool(
                policy_escalate
                or discordance_count > 0
                or uncertainty > self.max_uncertainty
            ),
            "reasons": reasons,
            "thresholds": {
                "max_uncertainty": self.max_uncertainty,
            },
            "observed": {
                "discordances": discordance_count,
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
    intuition payload, differential hypotheses, discordances and critique. It
    returns a structured research diagnostic result with abstention and
    escalation. External models remain specialist signal providers and critics.
    """

    registry: ModelCapabilityRegistry = field(
        default_factory=ModelCapabilityRegistry.build_default
    )
    policy: DiagnosticOrchestratorPolicy = field(
        default_factory=DiagnosticOrchestratorPolicy
    )

    def orchestrate(self, pipeline_result: dict[str, Any]) -> dict[str, Any]:
        coordinated = pipeline_result.get("coordinated", {})
        differential = (
            coordinated.get("differential", {}) if isinstance(coordinated, dict) else {}
        )
        hypotheses = (
            list(differential.get("hypotheses", []))
            if isinstance(differential, dict)
            else []
        )
        top_hypothesis = (
            hypotheses[0]
            if hypotheses
            else {"label": "none", "score": 0.0, "rationale": "no_hypothesis"}
        )
        intuition = pipeline_result.get("intuition", {})
        nexus = pipeline_result.get("nexus", {})
        critique = pipeline_result.get("critique", {})
        orchestration_policy = self.policy.evaluate(pipeline_result)

        support = {
            "visual": pipeline_result.get("area_signals", {}).get(
                "visual_diagnostic", {}
            ),
            "textual": pipeline_result.get("area_signals", {}).get(
                "language_listening", {}
            ),
            "context": pipeline_result.get("area_signals", {}).get("case_context", {}),
            "epidemiology": pipeline_result.get("area_signals", {}).get(
                "epidemiology", {}
            ),
            "semantic_memory": pipeline_result.get("retrieval", {}),
        }
        result_label = (
            top_hypothesis.get("label", "none")
            if orchestration_policy["allow_candidate_result"]
            else "abstain_or_escalate"
        )
        result = DiagnosticResult(
            case_id=str(pipeline_result.get("case_id", "unknown_case")),
            result_label=str(result_label),
            top_hypothesis=top_hypothesis,
            differential=hypotheses,
            intuition=IntuitionSummary(
                selected=intuition.get("intuition", "none")
                if isinstance(intuition, dict)
                else "none",
                candidate_scores=intuition.get("candidate_scores", [])
                if isinstance(intuition, dict)
                else [],
            ),
            discordances=dict(pipeline_result.get("discordances") or {}),
            support=support,
            policy=orchestration_policy,
            critique=critique,
            nexus=NexusSummary(
                accepted=nexus.get("accepted", False)
                if isinstance(nexus, dict)
                else False,
                auto_evolution_plan=nexus.get("auto_evolution_plan", {})
                if isinstance(nexus, dict)
                else {},
                alternative_hypotheses=nexus.get("alternative_hypotheses", [])
                if isinstance(nexus, dict)
                else [],
            ),
            model_capability_decision_record=self.registry.decision_record(),
            audit_trace={
                "pipeline_trace": coordinated.get("trace", [])
                if isinstance(coordinated, dict)
                else [],
                "external_models_are_not_final_arbiters": True,
                "final_authority": "MelampoDiagnosticOrchestrator",
                "clinical_warning": "Research output; not a validated medical device or standalone diagnosis.",
            },
        )
        return result.as_dict()
