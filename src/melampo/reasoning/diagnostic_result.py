from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class IntuitionSummary:
    selected: str = "none"
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "candidate_scores": list(self.candidate_scores),
        }


@dataclass(frozen=True, slots=True)
class NexusSummary:
    accepted: bool = False
    auto_evolution_plan: dict[str, Any] = field(default_factory=dict)
    alternative_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    promotion_policy: dict[str, Any] = field(
        default_factory=lambda: {
            "nexus_outputs_are_candidate_only": True,
            "automatic_clinical_promotion_allowed": False,
            "promotion_requires": [
                "source_grounding",
                "contradiction_review",
                "human_review_before_clinical_use",
                "audit_trace",
            ],
        }
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "auto_evolution_plan": dict(self.auto_evolution_plan),
            "alternative_hypotheses": list(self.alternative_hypotheses),
            "promotion_policy": dict(self.promotion_policy),
        }


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    """Enterprise-grade typed final research diagnostic result.

    This is not a clinical diagnosis. It is an auditable research output owned
    by MelampoDiagnosticOrchestrator. External models are signal providers or
    critics only and cannot become final diagnostic arbiters.
    """

    case_id: str
    result_label: str
    top_hypothesis: dict[str, Any]
    differential: list[dict[str, Any]]
    intuition: IntuitionSummary
    discordances: dict[str, Any]
    support: dict[str, Any]
    policy: dict[str, Any]
    critique: dict[str, Any]
    nexus: NexusSummary
    model_capability_decision_record: dict[str, Any]
    audit_trace: dict[str, Any]
    schema_version: str = "diagnostic_result.v2"
    created_at: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "case_id": self.case_id,
            "result_label": self.result_label,
            "top_hypothesis": dict(self.top_hypothesis),
            "differential": list(self.differential),
            "intuition": self.intuition.as_dict(),
            "discordances": dict(self.discordances),
            "support": dict(self.support),
            "policy": dict(self.policy),
            "critique": dict(self.critique),
            "nexus": self.nexus.as_dict(),
            "model_capability_decision_record": dict(
                self.model_capability_decision_record
            ),
            "audit_trace": {
                **dict(self.audit_trace),
                "final_authority": "MelampoDiagnosticOrchestrator",
                "external_models_are_not_final_arbiters": True,
                "clinical_warning": "Research output; not a validated medical device or standalone diagnosis.",
            },
        }
