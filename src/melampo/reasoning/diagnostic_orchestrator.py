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


def _is_placeholder(label: str) -> bool:
    head, _, tail = label.partition("_")
    return head == "candidate" and tail.isdigit()


@dataclass(slots=True)
class DiagnosticOrchestratorPolicy:
    """Final deterministic policy for Melampo diagnostic orchestration."""

    require_policy_allow: bool = True

    def evaluate(self, pipeline_result: dict[str, Any]) -> dict[str, Any]:
        """The gate of the result: the policy of the case's signals (``reasoning.case_signals``), by explicit rules.

        Nothing withholds the result but an explicit abstention of the policy: the reasons escalate to a person, each
        with its sentence, and the flags (no evidence, no provenance) are written on the result. Until 28 November
        2026 it also withheld on a hand-weighted uncertainty above 0.75, on the risk gate's 0.35 and, before that, on
        pi_score and the mismatch between areas: thresholds never measured, removed.
        """
        coordinated = pipeline_result.get("coordinated", {})
        policy = coordinated.get("policy", {}) if isinstance(coordinated, dict) else {}
        discordances = pipeline_result.get("discordances", {})
        discordance_count = (
            int(_safe_float(discordances.get("count", 0)))
            if isinstance(discordances, dict)
            else 0
        )
        policy_abstain = bool(policy.get("abstain", False))
        reasons = ["policy_abstention_requested"] if self.require_policy_allow and policy_abstain else []
        return {
            "allow_candidate_result": not reasons,
            "abstain": bool(reasons),
            "escalate": bool(policy.get("escalate", False) or discordance_count > 0),
            "reasons": reasons,
            "escalation_reasons": list(policy.get("reasons", [])),
            "flags": list(policy.get("flags", [])),
            "sentences": list(policy.get("sentences", [])),
            "observed": {"discordances": discordance_count},
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
        label = str(top_hypothesis.get("label", "none"))
        # a placeholder position (``candidate_N``, see intuition_engine) is not a diagnosis: it is never the result
        if _is_placeholder(label):
            label = "no_candidate_found"
        result_label = (
            label
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
